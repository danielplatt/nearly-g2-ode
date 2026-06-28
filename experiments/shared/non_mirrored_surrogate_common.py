"""Reusable XGBoost helpers for non-mirrored surrogate searches."""

from __future__ import annotations

import json
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Callable, TypeVar

import numpy as np
import xgboost as xgb
from mpmath import mp

from problem import SolverConfig
from solver.two_sided_refinement import (
    TwoSidedCandidateTrack,
    TwoSidedNewtonSettings,
    TwoSidedRefinementStageReport,
    two_sided_newton_refine,
)
from solver.two_sided_shooting import TwoSidedResidualResult, TwoSidedSearchPoint, params_from_two_sided_scaled, two_sided_residual

from .non_mirrored_common import (
    RANDOM_SEED,
    RegionSpec,
    SearchCandidate,
    SearchSeed,
    SelectedCandidate,
    _asymmetry_distance,
    _candidate_payload,
    _classify_track,
    _coordinates,
    _diverse_candidates,
    _event,
    _in_region,
    _mp_string,
    _physical_payload,
    _point_distance,
    _point_distance_between,
    _point_from_values,
    _point_payload,
    _region_summary,
    _replace_track,
    _result_payload,
    _successful,
    _track_final_result,
    _track_payload,
    _verify_point,
    _write_jsonl_event,
)


T = TypeVar("T")
RESIDUAL_FLOOR = 1e-300
FEATURE_NAMES = (
    "u_left",
    "v_left",
    "r_left",
    "u_right",
    "v_right",
    "r_right",
    "s",
    "distance_from_berger",
    "asymmetry_distance",
    "exp_s",
    "left_alpha_ratio",
    "right_omega_ratio",
)


@dataclass(frozen=True)
class SurrogateModels:
    """Trained branch, norm, and optional residual-vector models."""

    branch_model: object
    norm_models: tuple[object, ...]
    vector_models: tuple[tuple[object, ...], ...]
    metrics: dict[str, float | int | None]


@dataclass(frozen=True)
class SurrogateProposal:
    """One point proposed by a surrogate acquisition pass."""

    index: int
    region: str
    point: TwoSidedSearchPoint
    branch_probability: float
    mean_log_norm: float
    std_log_norm: float
    vector_log_norm: float | None
    score: float


class ConstantBranchModel:
    """Fallback branch model for one-class synthetic or degenerate data."""

    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, features):
        """Return constant two-column probabilities."""
        positive = np.full(features.shape[0], self.probability)
        return np.column_stack([1.0 - positive, positive])


class ConstantResidualModel:
    """Fallback scalar regressor for degenerate residual data."""

    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, features):
        """Return constant predictions."""
        return np.full(features.shape[0], self.value)


class XGBoostBranchModel:
    """Predict-probability adapter around a native XGBoost booster."""

    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    def predict_proba(self, features):
        """Return two-column branch probabilities."""
        positive = self.booster.predict(xgb.DMatrix(features))
        return np.column_stack([1.0 - positive, positive])


class XGBoostResidualModel:
    """Predict adapter around a native XGBoost regressor booster."""

    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    def predict(self, features):
        """Return residual predictions."""
        return self.booster.predict(xgb.DMatrix(features))


class _TimeoutExpired(Exception):
    """Raised by the signal-based timeout guard."""


@contextmanager
def _time_limit(seconds: float | None):
    """Raise `_TimeoutExpired` if a block exceeds the requested seconds."""
    if seconds is None:
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(_TimeoutExpired()))
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _run_with_timeout(callback: Callable[[], T], timeout_seconds: float | None) -> tuple[T | None, str | None]:
    """Run one callback and convert timeout into a nonfatal status."""
    try:
        with _time_limit(timeout_seconds):
            return callback(), None
    except _TimeoutExpired:
        return None, "timeout"


def _latin_hypercube_values(spec: RegionSpec, count: int, rng: Random) -> list[tuple[float, ...]]:
    """Return one Latin-hypercube batch inside a rectangular region."""
    columns = []
    for low, high in spec.ranges:
        lo = float(low)
        values = [lo + (float(high) - lo) * ((index + rng.random()) / count) for index in range(count)]
        rng.shuffle(values)
        columns.append(values)
    return list(zip(*columns))


def _lhs_region_seeds(spec: RegionSpec, rng: Random, start_index: int) -> list[SearchSeed]:
    """Return exactly the requested number of filtered Latin-hypercube seeds."""
    seeds: list[SearchSeed] = []
    attempts = 0
    while len(seeds) < spec.samples and attempts < 100:
        attempts += 1
        batch_size = max(spec.samples - len(seeds), spec.samples // 2, 64)
        for values in _latin_hypercube_values(spec, batch_size, rng):
            point = _point_from_values(values)
            if _in_region(point, spec):
                seeds.append(SearchSeed(start_index + len(seeds), spec.name, "lhs", point))
                if len(seeds) == spec.samples:
                    break
    if len(seeds) != spec.samples:
        raise RuntimeError(f"Could not sample enough Latin-hypercube points for {spec.name!r}.")
    return seeds


def _training_seeds(regions: tuple[RegionSpec, ...], seed: int = RANDOM_SEED) -> list[SearchSeed]:
    """Return deterministic training seeds for a region list."""
    rng = Random(seed)
    seeds: list[SearchSeed] = []
    for spec in regions:
        seeds.extend(_lhs_region_seeds(spec, rng, len(seeds)))
    return seeds


def _lhs_points(spec: RegionSpec, count: int, rng: Random) -> list[TwoSidedSearchPoint]:
    """Return filtered Latin-hypercube points for acquisition."""
    points: list[TwoSidedSearchPoint] = []
    attempts = 0
    while len(points) < count and attempts < 100:
        attempts += 1
        batch_size = max(count - len(points), count // 2, 256)
        for values in _latin_hypercube_values(spec, batch_size, rng):
            point = _point_from_values(values)
            if _in_region(point, spec):
                points.append(point)
                if len(points) == count:
                    break
    if len(points) != count:
        raise RuntimeError(f"Could not sample enough acquisition points for {spec.name!r}.")
    return points


def _feature_row(point: TwoSidedSearchPoint) -> list[float]:
    """Return the numeric feature row for one scaled point."""
    return [float(value) for value in _coordinates(point)] + [
        float(_point_distance(point)),
        float(_asymmetry_distance(point)),
        float(mp.exp(point.s)),
        float(1 + point.r_left),
        float(1 + point.r_right),
    ]


def _features_for_points(points: list[TwoSidedSearchPoint]) -> np.ndarray:
    """Return a finite feature matrix for scaled points."""
    return np.asarray([_feature_row(point) for point in points], dtype=np.float64)


def _labels_from_candidates(candidates: list[SearchCandidate]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return branch, log-norm, and signed residual-component labels."""
    branch = np.asarray([candidate.result.failure is None for candidate in candidates], dtype=np.int32)
    logs, residuals = [], []
    for candidate in candidates:
        if candidate.result.failure is None:
            values = [float(value) for value in candidate.result.residual]
            logs.append(np.log10(max(float(candidate.result.residual_norm), RESIDUAL_FLOOR)))
            residuals.append(values)
        else:
            logs.append(np.nan)
            residuals.append([np.nan] * 8)
    return branch, np.asarray(logs, dtype=np.float64), np.asarray(residuals, dtype=np.float64)


def _branch_probabilities(model, features: np.ndarray) -> np.ndarray:
    """Return branch-success probabilities."""
    probabilities = model.predict_proba(features)
    return probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]


def _ensemble_predictions(models: tuple[object, ...], features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ensemble mean and standard deviation."""
    predictions = np.vstack([model.predict(features) for model in models])
    return predictions.mean(axis=0), predictions.std(axis=0)


def _vector_predictions(vector_models: tuple[tuple[object, ...], ...], features: np.ndarray) -> np.ndarray | None:
    """Return mean signed residual component predictions."""
    if not vector_models:
        return None
    columns = [_ensemble_predictions(models, features)[0] for models in vector_models]
    return np.column_stack(columns)


def _split_indices(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic train/validation index arrays."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(count)
    split = max(1, int(0.8 * count))
    return indices[:split], indices[split:]


def _fit_branch_model(features: np.ndarray, labels: np.ndarray, seed: int):
    """Fit the branch classifier with a constant fallback for one-class data."""
    if len(np.unique(labels)) < 2:
        return ConstantBranchModel(float(labels[0]) if len(labels) else 0.0)
    params = _xgb_params("binary:logistic", seed) | {"eval_metric": "logloss", "eta": 0.06}
    return XGBoostBranchModel(xgb.train(params, xgb.DMatrix(features, label=labels), num_boost_round=160))


def _xgb_params(objective: str, seed: int) -> dict:
    """Return shared compact XGBoost parameters."""
    return {
        "objective": objective,
        "max_depth": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "nthread": 4,
        "seed": seed,
    }


def _fit_regressor_ensemble(features: np.ndarray, labels: np.ndarray, seed: int) -> tuple[object, ...]:
    """Fit a five-model XGBoost regressor ensemble."""
    finite = np.isfinite(labels)
    if finite.sum() < 2:
        fallback = 30.0 if finite.sum() == 0 else float(labels[finite][0])
        return tuple(ConstantResidualModel(fallback) for _ in range(5))
    models = []
    for offset in range(5):
        params = _xgb_params("reg:squarederror", seed + offset) | {"eta": 0.05}
        booster = xgb.train(params, xgb.DMatrix(features[finite], label=labels[finite]), num_boost_round=180)
        models.append(XGBoostResidualModel(booster))
    return tuple(models)


def _fit_vector_models(features: np.ndarray, residuals: np.ndarray, seed: int) -> tuple[tuple[object, ...], ...]:
    """Fit eight signed residual-component ensembles."""
    return tuple(_fit_regressor_ensemble(features, residuals[:, index], seed + 1000 * index) for index in range(8))


def _top_k_recall(predicted_logs: np.ndarray, true_logs: np.ndarray, successful: np.ndarray) -> float | None:
    """Return recall of the genuinely lowest validation residuals."""
    success_indices = np.flatnonzero(successful & np.isfinite(true_logs))
    if len(success_indices) == 0:
        return None
    k = max(1, min(50, len(success_indices) // 10 or 1))
    true_top = set(success_indices[np.argsort(true_logs[success_indices])[:k]])
    predicted_top = set(success_indices[np.argsort(predicted_logs[success_indices])[:k]])
    return len(true_top & predicted_top) / k


def _validation_metrics(
    models: SurrogateModels,
    features: np.ndarray,
    branch: np.ndarray,
    logs: np.ndarray,
    residuals: np.ndarray,
    val_idx: np.ndarray,
    branch_cutoff: float,
) -> dict[str, float | int | None]:
    """Compute deterministic validation metrics."""
    if len(val_idx) == 0:
        return {"validation_size": 0}
    val_features, successful = features[val_idx], branch[val_idx] == 1
    branch_probs = _branch_probabilities(models.branch_model, val_features)
    norm_mean, _std = _ensemble_predictions(models.norm_models, val_features)
    metrics = _basic_metrics(branch_probs, successful, norm_mean, logs[val_idx], branch_cutoff)
    vector = _vector_predictions(models.vector_models, val_features)
    if vector is not None and np.any(successful):
        metrics["vector_component_mae"] = float(np.mean(np.abs(vector[successful] - residuals[val_idx][successful])))
    return metrics | {"validation_size": int(len(val_idx))}


def _basic_metrics(branch_probs, successful, norm_mean, true_logs, branch_cutoff) -> dict:
    """Return branch and norm-model validation metrics."""
    top_valid = branch_probs >= branch_cutoff
    return {
        "branch_accuracy": float(np.mean(top_valid == successful)),
        "top_predicted_valid_precision": float(np.mean(successful[top_valid])) if np.any(top_valid) else None,
        "residual_mae_log10": float(np.mean(np.abs(norm_mean[successful] - true_logs[successful]))) if np.any(successful) else None,
        "top_k_low_residual_recall": _top_k_recall(norm_mean, true_logs, successful),
    }


def _train_models(
    candidates: list[SearchCandidate],
    *,
    seed: int,
    branch_cutoff: float,
    include_vector_models: bool,
) -> SurrogateModels:
    """Train branch, norm, and optional vector surrogates from true labels."""
    features = _features_for_points([candidate.seed.point for candidate in candidates])
    branch, logs, residuals = _labels_from_candidates(candidates)
    train_idx, val_idx = _split_indices(len(candidates), seed)
    branch_model = _fit_branch_model(features[train_idx], branch[train_idx], seed)
    norm_models = _fit_regressor_ensemble(features[train_idx], logs[train_idx], seed + 100)
    vector_models = _fit_vector_models(features[train_idx], residuals[train_idx], seed + 200) if include_vector_models else ()
    partial = SurrogateModels(branch_model, norm_models, vector_models, {})
    metrics = _validation_metrics(partial, features, branch, logs, residuals, val_idx, branch_cutoff)
    metrics |= {"train_size": int(len(train_idx)), "total_size": len(candidates), "successes": int(branch.sum())}
    return SurrogateModels(branch_model, norm_models, vector_models, metrics)


def _evaluate_seed_event(seed: SearchSeed, path: Path, event_type: str, config: SolverConfig) -> SearchCandidate:
    """Evaluate one seed with the true Taylor solver and write a named event."""
    with mp.workdps(config.working_dps):
        candidate = SearchCandidate(seed, two_sided_residual(seed.point, config))
    _write_jsonl_event(path, _event(event_type, _candidate_payload(candidate)))
    return candidate


def _evaluate_training_set(
    seeds: list[SearchSeed],
    path: Path,
    config: SolverConfig,
    progress_label: str,
    existing: list[SearchCandidate] | None = None,
) -> list[SearchCandidate]:
    """Evaluate true labels for a fixed seed list."""
    candidates = list(existing or ())
    completed = {candidate.seed.index for candidate in candidates}
    for index, seed in enumerate(seeds, start=1):
        if seed.index in completed:
            continue
        candidates.append(_evaluate_seed_event(seed, path, "training_sample", config))
        if index % 100 == 0:
            print(f"  {progress_label}: {index}/{len(seeds)}", flush=True)
    return candidates


def _proposal_payload(proposal: SurrogateProposal) -> dict:
    """Return JSON-ready surrogate proposal data."""
    return {
        "index": proposal.index,
        "region": proposal.region,
        "point": _point_payload(proposal.point),
        "distance": _mp_string(_point_distance(proposal.point)),
        "asymmetry": _mp_string(_asymmetry_distance(proposal.point)),
        "branch_probability": proposal.branch_probability,
        "mean_log_norm": proposal.mean_log_norm,
        "std_log_norm": proposal.std_log_norm,
        "vector_log_norm": proposal.vector_log_norm,
        "score": proposal.score,
    }


def _score_points(
    models: SurrogateModels,
    region: str,
    points: list[TwoSidedSearchPoint],
    start_index: int,
    *,
    branch_cutoff: float,
    optimism_weight: float,
) -> list[SurrogateProposal]:
    """Score candidate points with norm and vector surrogate predictions."""
    features = _features_for_points(points)
    branch_probs = _branch_probabilities(models.branch_model, features)
    norm_mean, norm_std = _ensemble_predictions(models.norm_models, features)
    vector_log = _vector_log_norms(models, features)
    return _build_proposals(region, points, start_index, branch_probs, norm_mean, norm_std, vector_log, branch_cutoff, optimism_weight)


def _vector_log_norms(models: SurrogateModels, features: np.ndarray) -> np.ndarray | None:
    """Return log10 max-norm from vector-head predictions if available."""
    vector = _vector_predictions(models.vector_models, features)
    if vector is None:
        return None
    return np.log10(np.maximum(np.max(np.abs(vector), axis=1), RESIDUAL_FLOOR))


def _build_proposals(region, points, start, branch_probs, mean, std, vector_log, cutoff, optimism) -> list[SurrogateProposal]:
    """Build proposal records that pass the branch-probability cutoff."""
    proposals = []
    for offset, point in enumerate(points):
        if branch_probs[offset] < cutoff:
            continue
        vector_value = None if vector_log is None else float(vector_log[offset])
        base_score = float(mean[offset] if vector_value is None else min(mean[offset], vector_value))
        score = base_score - optimism * float(std[offset])
        proposals.append(SurrogateProposal(start + offset, region, point, float(branch_probs[offset]), float(mean[offset]), float(std[offset]), vector_value, score))
    return proposals


def _select_diverse_proposals(proposals: list[SurrogateProposal], quota: int) -> list[SurrogateProposal]:
    """Select low-score proposals while preserving spread."""
    ordered = sorted(proposals, key=lambda item: item.score)
    selected = ordered[: min(quota // 2, len(ordered))]
    pool = ordered[: max(quota * 20, quota)]
    while len(selected) < quota:
        keys = {_point_payload(item.point).__repr__() for item in selected}
        remaining = [item for item in pool if _point_payload(item.point).__repr__() not in keys]
        if not remaining or not selected:
            break
        picked = max(remaining, key=lambda item: min(_point_distance_between(item.point, chosen.point) for chosen in selected))
        selected.append(picked)
    return selected


def _acquire_surrogate_proposals(
    models: SurrogateModels,
    path: Path,
    regions: tuple[RegionSpec, ...],
    quotas: dict[str, int],
    *,
    chunk_size: int,
    branch_cutoff: float,
    optimism_weight: float,
    existing: list[SurrogateProposal] | None = None,
) -> list[SurrogateProposal]:
    """Generate, score, and persist region-quota surrogate proposals."""
    rng = Random(RANDOM_SEED + 55)
    selected: list[SurrogateProposal] = list(existing or ())
    proposal_index = 0
    for spec in regions:
        existing_region = [proposal for proposal in selected if proposal.region == spec.name]
        if len(existing_region) >= quotas[spec.name]:
            proposal_index = _advance_region_rng(spec, rng, proposal_index, chunk_size)
            continue
        kept, proposal_index = _score_region(models, spec, quotas, rng, path, proposal_index, chunk_size, branch_cutoff, optimism_weight, existing_region)
        existing_indices = {proposal.index for proposal in existing_region}
        selected.extend(proposal for proposal in kept if proposal.index not in existing_indices)
    return selected


def _advance_region_rng(spec: RegionSpec, rng: Random, proposal_index: int, chunk_size: int) -> int:
    """Consume a completed region's random samples to preserve later regions."""
    remaining = spec.samples
    while remaining:
        count = min(chunk_size, remaining)
        remaining -= count
        batch_spec = RegionSpec(spec.name, spec.ranges, count, 0, 0, 0, spec.min_distance, spec.max_distance, spec.min_asymmetry)
        _lhs_points(batch_spec, count, rng)
        proposal_index += count
    return proposal_index


def _score_region(models, spec, quotas, rng, path, proposal_index, chunk_size, cutoff, optimism, existing=None) -> tuple[list[SurrogateProposal], int]:
    """Score and select proposals for one acquisition region."""
    kept: list[SurrogateProposal] = []
    remaining = spec.samples
    while remaining:
        count = min(chunk_size, remaining)
        remaining -= count
        batch_spec = RegionSpec(spec.name, spec.ranges, count, 0, 0, 0, spec.min_distance, spec.max_distance, spec.min_asymmetry)
        proposals = _score_points(models, spec.name, _lhs_points(batch_spec, count, rng), proposal_index, branch_cutoff=cutoff, optimism_weight=optimism)
        proposal_index += count
        kept.extend(proposals)
        kept = sorted(kept, key=lambda item: item.score)[: max(quotas[spec.name] * 50, 1000)]
    selected = _select_diverse_proposals(kept, quotas[spec.name])
    existing_indices = {proposal.index for proposal in existing or ()}
    for proposal in selected:
        if proposal.index not in existing_indices:
            _write_jsonl_event(path, _event("surrogate_proposal", _proposal_payload(proposal)))
    print(f"  surrogate proposals selected for {spec.name}: {len(set(existing_indices) | {proposal.index for proposal in selected})}", flush=True)
    return selected, proposal_index


def _evaluate_proposals(
    proposals: list[SurrogateProposal],
    path: Path,
    config: SolverConfig,
    existing: list[SearchCandidate] | None = None,
) -> list[SearchCandidate]:
    """Evaluate surrogate proposals with the true cheap Taylor solver."""
    candidates = list(existing or ())
    completed = {candidate.seed.index for candidate in candidates}
    for index, proposal in enumerate(proposals, start=1):
        seed = SearchSeed(100_000 + index, proposal.region, "surrogate", proposal.point)
        if seed.index in completed:
            continue
        candidates.append(_evaluate_seed_event(seed, path, "true_proposal_evaluation", config))
        if index % 50 == 0:
            print(f"  surrogate proposals evaluated: {index}/{len(proposals)}", flush=True)
    return candidates


def _select_refinement_candidates_by_region(
    candidates: list[SearchCandidate],
    quotas: dict[str, int],
) -> list[SelectedCandidate]:
    """Select best and diverse true proposals for order-6 refinement."""
    selected: list[SelectedCandidate] = []
    for region, quota in quotas.items():
        successful = _successful([candidate for candidate in candidates if candidate.seed.region == region])
        best = successful[: min(quota // 2, len(successful))]
        diverse = _diverse_candidates(successful, best, quota - len(best))
        selected.extend(SelectedCandidate(len(selected) + 1, "region-best", item) for item in best)
        selected.extend(SelectedCandidate(len(selected) + 1, "region-diverse", item) for item in diverse)
    return selected


def _timeout_result(point: TwoSidedSearchPoint, config: SolverConfig, message: str) -> TwoSidedResidualResult:
    """Return a failed residual result for timeout diagnostics."""
    params, local_config = params_from_two_sided_scaled(point, template_config=config)
    return TwoSidedResidualResult(point, params, local_config, (), mp.inf, None, None, None, None, (0, 0), {}, message)


def _timeout_stage(selection_or_track, settings: TwoSidedNewtonSettings, message: str) -> TwoSidedRefinementStageReport:
    """Build a timeout refinement stage report."""
    point = selection_or_track.candidate.seed.point if isinstance(selection_or_track, SelectedCandidate) else _track_final_result(selection_or_track).point
    result = _timeout_result(point, settings.config, message)
    return TwoSidedRefinementStageReport(settings, result, result, (), message)


def _initial_track_with_timeout(
    selection: SelectedCandidate,
    path: Path,
    settings: TwoSidedNewtonSettings,
    timeout_seconds: float | None,
) -> TwoSidedCandidateTrack:
    """Run order-6 refinement with a nonfatal timeout guard."""
    stage, status = _run_with_timeout(lambda: two_sided_newton_refine(selection.candidate.seed.point, settings), timeout_seconds)
    if stage is None:
        stage = _timeout_stage(selection, settings, status or "timeout")
    track = TwoSidedCandidateTrack(selection.candidate.seed.index, selection.candidate.seed.region, selection.candidate.seed.point, selection.candidate.result, (stage,), (), "inconclusive")
    _write_jsonl_event(path, _event("refinement_stage", {"stage": _stage_payload(stage), "track": _track_payload(track)}))
    return track


def _promote_track_with_timeout(
    track: TwoSidedCandidateTrack,
    refs: tuple[TwoSidedResidualResult, ...],
    path: Path,
    settings: TwoSidedNewtonSettings,
    verify_configs: tuple[SolverConfig, ...],
    timeout_seconds: float | None,
) -> TwoSidedCandidateTrack:
    """Run order-10 refinement plus verification with a timeout guard."""
    promoted, status = _run_with_timeout(lambda: _promote_track_core(track, refs, settings, verify_configs), timeout_seconds)
    if promoted is None:
        stage = _timeout_stage(track, settings, status or "timeout")
        promoted = _replace_track(track, track.stages + (stage,), (), "branch_failure")
    _write_jsonl_event(path, _event("candidate_classification", _track_payload(promoted)))
    return promoted


def _promote_track_core(track, refs, settings, verify_configs) -> TwoSidedCandidateTrack:
    """Run order-10 refinement and high-order verification for one track."""
    stage = two_sided_newton_refine(track.stages[-1].final.point, settings)
    verifications = _verify_point(stage.final.point, verify_configs)
    promoted = _replace_track(track, track.stages + (stage,), verifications, "inconclusive")
    return _replace_track(promoted, promoted.stages, promoted.verifications, _classify_track(promoted, refs))


def _stage_payload(stage) -> dict:
    """Return JSON-ready data for one refinement stage."""
    return {
        "settings": {"name": stage.settings.name, "order": stage.settings.config.series_order},
        "status": stage.status,
        "initial": _result_payload(stage.initial),
        "final": _result_payload(stage.final),
        "steps": [{"index": step.index, "status": step.status, "damping": _mp_string(step.damping)} for step in stage.steps],
    }


def _model_metrics_payload(models: SurrogateModels, phase: str) -> dict:
    """Return JSON-ready model metrics."""
    return {"phase": phase, "metrics": models.metrics, "feature_names": FEATURE_NAMES, "has_vector_models": bool(models.vector_models)}


def _best_payload(candidates: list[SearchCandidate], limit: int = 20) -> list[dict]:
    """Return the best true residual candidates as JSON-ready records."""
    return [_candidate_payload(candidate) for candidate in _successful(candidates)[:limit]]


def _classification_counts(tracks) -> dict[str, int]:
    """Return classification counts for promoted tracks."""
    counts: dict[str, int] = {}
    for track in tracks:
        counts[track.classification] = counts.get(track.classification, 0) + 1
    return counts


def _best_stable_by_region(tracks) -> dict[str, str | None]:
    """Return the best finite order-18 verification residual per region."""
    best: dict[str, float] = {}
    for track in tracks:
        if len(track.verifications) < 2 or track.verifications[-1].failure:
            continue
        value = float(track.verifications[-1].residual_norm)
        best[track.seed_region] = min(best.get(track.seed_region, value), value)
    return {region: _mp_string(mp.mpf(value)) for region, value in best.items()}


def _physical_payloads(tracks) -> list[dict[str, str | None]]:
    """Return physical parameters for promoted tracks."""
    return [_physical_payload(_track_final_result(track).point) for track in tracks]


def _write_run_start(path: Path, payload: dict) -> None:
    """Write a timestamped run-start event."""
    _write_jsonl_event(path, {"event": "run_start", "time_utc": datetime.now(timezone.utc).isoformat(), **payload})


def _load_candidates_from_jsonl(path: Path, event_type: str) -> list[SearchCandidate]:
    """Load previously completed candidate evaluations from a JSONL checkpoint."""
    if not path.exists():
        return []
    candidates = []
    for event in _iter_jsonl_events(path):
        if event.get("event") == event_type:
            candidates.append(_candidate_from_payload(event))
    return candidates


def _load_refinement_tracks_from_jsonl(
    path: Path,
    settings_by_order: dict[int, TwoSidedNewtonSettings],
) -> list[TwoSidedCandidateTrack]:
    """Load completed order-6 refinement tracks from a JSONL checkpoint."""
    tracks: dict[int, TwoSidedCandidateTrack] = {}
    if not path.exists():
        return []
    for event in _iter_jsonl_events(path):
        if event.get("event") == "refinement_stage":
            track = _track_from_payload(event["track"], settings_by_order)
            tracks[track.seed_rank] = track
    return list(tracks.values())


def _load_classified_tracks_from_jsonl(
    path: Path,
    settings_by_order: dict[int, TwoSidedNewtonSettings],
) -> list[TwoSidedCandidateTrack]:
    """Load promoted candidate tracks from classification events."""
    tracks: dict[int, TwoSidedCandidateTrack] = {}
    if not path.exists():
        return []
    for event in _iter_jsonl_events(path):
        if event.get("event") == "candidate_classification":
            track = _track_from_payload(event, settings_by_order)
            tracks[track.seed_rank] = track
    return list(tracks.values())


def _load_proposals_from_jsonl(path: Path) -> list[SurrogateProposal]:
    """Load previously selected surrogate proposals from a JSONL checkpoint."""
    if not path.exists():
        return []
    return [_proposal_from_payload(event) for event in _iter_jsonl_events(path) if event.get("event") == "surrogate_proposal"]


def _has_event(path: Path, event_type: str) -> bool:
    """Return whether a JSONL checkpoint contains an event type."""
    return any(event.get("event") == event_type for event in _iter_jsonl_events(path)) if path.exists() else False


def _iter_jsonl_events(path: Path):
    """Yield parseable JSON events from a checkpoint file."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _candidate_from_payload(payload: dict) -> SearchCandidate:
    """Rebuild a `SearchCandidate` from a persisted event payload."""
    point = _point_from_payload(payload["seed_point"])
    seed = SearchSeed(int(payload["seed_index"]), payload["region"], payload["source"], point)
    return SearchCandidate(seed, _result_from_payload(payload["result"]))


def _proposal_from_payload(payload: dict) -> SurrogateProposal:
    """Rebuild a `SurrogateProposal` from a persisted event payload."""
    return SurrogateProposal(
        int(payload["index"]),
        payload["region"],
        _point_from_payload(payload["point"]),
        float(payload["branch_probability"]),
        float(payload["mean_log_norm"]),
        float(payload["std_log_norm"]),
        None if payload.get("vector_log_norm") is None else float(payload["vector_log_norm"]),
        float(payload["score"]),
    )


def _track_from_payload(payload: dict, settings_by_order: dict[int, TwoSidedNewtonSettings]) -> TwoSidedCandidateTrack:
    """Rebuild a candidate track from a persisted track payload."""
    seed_point = _point_from_payload(payload["seed_point"])
    stages = tuple(_stage_from_payload(stage, settings_by_order) for stage in payload["stages"])
    verifications = tuple(_result_from_payload(result) for result in payload.get("verifications", ()))
    return TwoSidedCandidateTrack(
        int(payload["seed_index"]),
        payload["region"],
        seed_point,
        _result_from_payload(payload["scout"]),
        stages,
        verifications,
        payload["classification"],
    )


def _stage_from_payload(payload: dict, settings_by_order: dict[int, TwoSidedNewtonSettings]) -> TwoSidedRefinementStageReport:
    """Rebuild a refinement stage without reconstructing per-step internals."""
    order = int(payload["settings"]["order"])
    settings = settings_by_order[order]
    return TwoSidedRefinementStageReport(
        settings,
        _result_from_payload(payload["initial"]),
        _result_from_payload(payload["final"]),
        (),
        payload["status"],
    )


def _point_from_payload(payload: dict) -> TwoSidedSearchPoint:
    """Rebuild one scaled point from JSON payload strings."""
    return TwoSidedSearchPoint(
        mp.mpf(payload["u_left"]),
        mp.mpf(payload["v_left"]),
        mp.mpf(payload["r_left"]),
        mp.mpf(payload["u_right"]),
        mp.mpf(payload["v_right"]),
        mp.mpf(payload["r_right"]),
        mp.mpf(payload["s"]),
    )


def _result_from_payload(payload: dict) -> TwoSidedResidualResult:
    """Rebuild a residual result from JSON payload strings."""
    point = _point_from_payload(payload["point"])
    params, config = params_from_two_sided_scaled(point)
    residual = tuple(mp.mpf(value) for value in payload["residual"])
    branch = {key: mp.mpf(value) for key, value in payload["branch_diagnostics"].items()}
    return TwoSidedResidualResult(
        point,
        params,
        config,
        residual,
        mp.mpf(payload["residual_norm"]),
        None,
        None,
        None if payload["left_l"] is None else mp.mpf(payload["left_l"]),
        None if payload["right_l"] is None else mp.mpf(payload["right_l"]),
        tuple(payload["patch_counts"]),
        branch,
        payload["failure"],
    )
