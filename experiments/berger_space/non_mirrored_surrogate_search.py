"""XGBoost-assisted search for non-mirrored two-sided candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random

import numpy as np
import xgboost as xgb
from mpmath import mp

from problem import DEFAULT_CONFIG, SolverConfig
from solver.two_sided_refinement import TwoSidedNewtonSettings
from solver.two_sided_shooting import BASE_TWO_SIDED_POINT, TwoSidedResidualResult, TwoSidedSearchPoint, two_sided_residual

from ..shared.non_mirrored_common import (
    RANDOM_SEED,
    S_MIN,
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
    _initial_track,
    _mp_string,
    _physical_payload,
    _point_distance,
    _point_distance_between,
    _point_from_values,
    _point_payload,
    _region_summary,
    _result_payload,
    _successful,
    _track_final_result,
    _track_payload,
    _verify_point,
    _write_jsonl_event,
    _write_summary,
)


OUTPUT_DIR = Path("output/non_mirrored_surrogate")
SCOUT_CONFIG = SolverConfig(4, 30, 15, mp.mpf("0.95"), 0, DEFAULT_CONFIG.match_t)
ORDER6_CONFIG = SolverConfig(6, 40, 20, mp.mpf("0.9"), 0, DEFAULT_CONFIG.match_t)
ORDER10_CONFIG = SolverConfig(10, 70, 30, mp.mpf("0.7"), 1, DEFAULT_CONFIG.match_t)
VERIFY14_CONFIG = SolverConfig(14, 90, 35, mp.mpf("0.6"), 2, DEFAULT_CONFIG.match_t)
VERIFY18_CONFIG = SolverConfig(18, 110, 45, mp.mpf("0.55"), 2, DEFAULT_CONFIG.match_t)
VERIFY_CONFIGS = (VERIFY14_CONFIG, VERIFY18_CONFIG)

ORDER6_SETTINGS = TwoSidedNewtonSettings(
    "order-6",
    ORDER6_CONFIG,
    mp.mpf("1e-3"),
    mp.mpf("1e-8"),
    3,
    max_abs_coordinate=mp.mpf("12"),
    min_s_coordinate=S_MIN,
)
ORDER10_SETTINGS = TwoSidedNewtonSettings(
    "order-10",
    ORDER10_CONFIG,
    mp.mpf("3e-4"),
    mp.mpf("1e-10"),
    3,
    max_abs_coordinate=mp.mpf("12"),
    min_s_coordinate=S_MIN,
)

TRAINING_REGIONS = (
    RegionSpec("near", ((-0.6, 0.6), (-0.6, 0.6), (-1.5, 1.5), (-0.6, 0.6), (-0.6, 0.6), (-1.5, 1.5), (-0.8, 0.8)), 700, 0, 0, 0, mp.mpf("0.05"), mp.mpf("1.5")),
    RegionSpec("medium", ((-1.5, 1.5), (-1.5, 1.5), (-3, 3), (-1.5, 1.5), (-1.5, 1.5), (-3, 3), (-1.5, 1.5)), 1100, 0, 0, 0, mp.mpf("1"), mp.mpf("3")),
    RegionSpec("far", ((-3, 3), (-3, 3), (-6, 6), (-3, 3), (-3, 3), (-6, 6), (-2.5, 2)), 1200, 0, 0, 0, mp.mpf("3"), mp.mpf("6")),
    RegionSpec("asymmetric", ((-2, 2), (-2, 2), (-4, 4), (-2, 2), (-2, 2), (-4, 4), (-1.5, 1.5)), 1400, 0, 0, 0, mp.mpf("0.5"), mp.mpf("4"), mp.mpf("0.75")),
    RegionSpec("wide", ((-4, 4), (-4, 4), (-8, 8), (-4, 4), (-4, 4), (-8, 8), (-3.5, 2.5)), 600, 0, 0, 0, mp.mpf("4"), mp.mpf("8"), mp.mpf("0.5")),
)

ACQUISITION_REGIONS = (
    RegionSpec("near", TRAINING_REGIONS[0].ranges, 150_000, 0, 0, 0, TRAINING_REGIONS[0].min_distance, TRAINING_REGIONS[0].max_distance),
    RegionSpec("medium", TRAINING_REGIONS[1].ranges, 220_000, 0, 0, 0, TRAINING_REGIONS[1].min_distance, TRAINING_REGIONS[1].max_distance),
    RegionSpec("far", TRAINING_REGIONS[2].ranges, 240_000, 0, 0, 0, TRAINING_REGIONS[2].min_distance, TRAINING_REGIONS[2].max_distance),
    RegionSpec("asymmetric", TRAINING_REGIONS[3].ranges, 270_000, 0, 0, 0, TRAINING_REGIONS[3].min_distance, TRAINING_REGIONS[3].max_distance, TRAINING_REGIONS[3].min_asymmetry),
    RegionSpec("wide", TRAINING_REGIONS[4].ranges, 120_000, 0, 0, 0, TRAINING_REGIONS[4].min_distance, TRAINING_REGIONS[4].max_distance, TRAINING_REGIONS[4].min_asymmetry),
)
PROPOSAL_QUOTAS = {"near": 50, "medium": 80, "far": 90, "asymmetric": 120, "wide": 60}
ACQUISITION_CHUNK_SIZE = 50_000
BRANCH_PROBABILITY_CUTOFF = 0.6
OPTIMISM_WEIGHT = 0.75
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
    """Trained XGBoost models and validation metrics."""

    branch_model: object
    residual_models: tuple[object, ...]
    metrics: dict[str, float | int | None]


@dataclass(frozen=True)
class SurrogateProposal:
    """One candidate proposed by the surrogate acquisition step."""

    index: int
    region: str
    point: TwoSidedSearchPoint
    branch_probability: float
    mean_log_norm: float
    std_log_norm: float
    score: float


class ConstantBranchModel:
    """Fallback branch model for degenerate tiny datasets."""

    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, features):
        """Return a constant two-column probability array."""
        negative = np.full(features.shape[0], 1.0 - self.probability)
        positive = np.full(features.shape[0], self.probability)
        return np.column_stack([negative, positive])


class ConstantResidualModel:
    """Fallback residual model for degenerate tiny datasets."""

    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, features):
        """Return a constant residual prediction."""
        return np.full(features.shape[0], self.value)


class XGBoostBranchModel:
    """Small predict_proba adapter around a native XGBoost booster."""

    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    def predict_proba(self, features):
        """Return two-column branch probabilities."""
        positive = self.booster.predict(xgb.DMatrix(features))
        return np.column_stack([1.0 - positive, positive])


class XGBoostResidualModel:
    """Small predict adapter around a native XGBoost booster."""

    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    def predict(self, features):
        """Return log-residual predictions."""
        return self.booster.predict(xgb.DMatrix(features))


def _output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped JSONL and summary output paths."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    stem = f"{stamp}-seed{RANDOM_SEED}"
    return OUTPUT_DIR / f"{stem}-training.jsonl", OUTPUT_DIR / f"{stem}-summary.json"


def _latin_hypercube_values(spec: RegionSpec, count: int, rng: Random) -> list[tuple[float, ...]]:
    """Return one Latin-hypercube batch inside a rectangular region."""
    columns = []
    for low, high in spec.ranges:
        lo = float(low)
        span = float(high) - lo
        values = [lo + span * ((index + rng.random()) / count) for index in range(count)]
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
                index = start_index + len(seeds)
                seeds.append(SearchSeed(index, spec.name, "lhs", point))
                if len(seeds) == spec.samples:
                    break
    if len(seeds) != spec.samples:
        raise RuntimeError(f"Could not sample enough Latin-hypercube points for {spec.name!r}.")
    return seeds


def _training_seeds(seed: int = RANDOM_SEED) -> list[SearchSeed]:
    """Return the deterministic 5000-point surrogate training seed set."""
    rng = Random(seed)
    seeds: list[SearchSeed] = []
    for spec in TRAINING_REGIONS:
        seeds.extend(_lhs_region_seeds(spec, rng, len(seeds)))
    return seeds


def _feature_row(point: TwoSidedSearchPoint) -> list[float]:
    """Return the numeric feature row for one scaled point."""
    coords = [float(value) for value in _coordinates(point)]
    return coords + [
        float(_point_distance(point)),
        float(_asymmetry_distance(point)),
        float(mp.exp(point.s)),
        float(1 + point.r_left),
        float(1 + point.r_right),
    ]


def _features_for_points(points: list[TwoSidedSearchPoint]) -> np.ndarray:
    """Return a finite feature matrix for scaled search points."""
    return np.asarray([_feature_row(point) for point in points], dtype=np.float64)


def _labels_from_candidates(candidates: list[SearchCandidate]) -> tuple[np.ndarray, np.ndarray]:
    """Return branch and log-residual labels for evaluated candidates."""
    branch = np.asarray([candidate.result.failure is None for candidate in candidates], dtype=np.int32)
    logs = []
    for candidate in candidates:
        norm = float(candidate.result.residual_norm) if candidate.result.failure is None else np.nan
        logs.append(np.log10(max(norm, RESIDUAL_FLOOR)) if np.isfinite(norm) else np.nan)
    return branch, np.asarray(logs, dtype=np.float64)


def _branch_probabilities(model, features: np.ndarray) -> np.ndarray:
    """Return predicted branch-success probabilities."""
    probabilities = model.predict_proba(features)
    return probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]


def _residual_predictions(models: tuple[object, ...], features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ensemble mean and standard deviation of log-residual predictions."""
    predictions = np.vstack([model.predict(features) for model in models])
    return predictions.mean(axis=0), predictions.std(axis=0)


def _split_indices(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic train/validation index arrays."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(count)
    split = max(1, int(0.8 * count))
    return indices[:split], indices[split:]


def _fit_branch_model(features: np.ndarray, labels: np.ndarray, seed: int):
    """Fit the branch classifier, with a constant fallback for one-class data."""
    if len(np.unique(labels)) < 2:
        return ConstantBranchModel(float(labels[0]) if len(labels) else 0.0)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
        "eta": 0.06,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "nthread": 4,
        "seed": seed,
    }
    booster = xgb.train(params, xgb.DMatrix(features, label=labels), num_boost_round=160)
    return XGBoostBranchModel(booster)


def _fit_residual_models(features: np.ndarray, labels: np.ndarray, seed: int) -> tuple[object, ...]:
    """Fit a small residual-model ensemble on branch-valid samples."""
    finite = np.isfinite(labels)
    if finite.sum() < 2:
        fallback = 30.0 if finite.sum() == 0 else float(labels[finite][0])
        return tuple(ConstantResidualModel(fallback) for _ in range(5))
    models = []
    for offset in range(5):
        params = {
            "objective": "reg:squarederror",
            "max_depth": 3,
            "eta": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            "nthread": 4,
            "seed": seed + offset,
        }
        booster = xgb.train(params, xgb.DMatrix(features[finite], label=labels[finite]), num_boost_round=180)
        models.append(XGBoostResidualModel(booster))
    return tuple(models)


def _validation_metrics(
    models: SurrogateModels,
    features: np.ndarray,
    branch: np.ndarray,
    logs: np.ndarray,
    val_idx: np.ndarray,
) -> dict[str, float | int | None]:
    """Compute deterministic validation metrics for the surrogate models."""
    if len(val_idx) == 0:
        return {"validation_size": 0}
    val_features = features[val_idx]
    val_branch = branch[val_idx]
    branch_probs = _branch_probabilities(models.branch_model, val_features)
    predicted_branch = branch_probs >= BRANCH_PROBABILITY_CUTOFF
    mean_logs, _std = _residual_predictions(models.residual_models, val_features)
    successful = val_branch == 1
    top_valid = branch_probs >= BRANCH_PROBABILITY_CUTOFF
    metrics = {
        "validation_size": int(len(val_idx)),
        "branch_accuracy": float(np.mean(predicted_branch == successful)),
        "top_predicted_valid_precision": float(np.mean(successful[top_valid])) if np.any(top_valid) else None,
        "residual_mae_log10": float(np.mean(np.abs(mean_logs[successful] - logs[val_idx][successful]))) if np.any(successful) else None,
        "top_k_low_residual_recall": _top_k_recall(mean_logs, logs[val_idx], successful),
    }
    return metrics


def _top_k_recall(predicted_logs: np.ndarray, true_logs: np.ndarray, successful: np.ndarray) -> float | None:
    """Return recall of the truly lowest residual validation samples."""
    success_indices = np.flatnonzero(successful & np.isfinite(true_logs))
    if len(success_indices) == 0:
        return None
    k = max(1, min(50, len(success_indices) // 10 or 1))
    true_top = set(success_indices[np.argsort(true_logs[success_indices])[:k]])
    predicted_top = set(success_indices[np.argsort(predicted_logs[success_indices])[:k]])
    return len(true_top & predicted_top) / k


def _train_models(candidates: list[SearchCandidate], seed: int = RANDOM_SEED) -> SurrogateModels:
    """Train branch and residual surrogates from true ODE evaluations."""
    features = _features_for_points([candidate.seed.point for candidate in candidates])
    branch, logs = _labels_from_candidates(candidates)
    train_idx, val_idx = _split_indices(len(candidates), seed)
    branch_model = _fit_branch_model(features[train_idx], branch[train_idx], seed)
    residual_models = _fit_residual_models(features[train_idx], logs[train_idx], seed + 100)
    partial = SurrogateModels(branch_model, residual_models, {})
    metrics = _validation_metrics(partial, features, branch, logs, val_idx)
    metrics |= {"train_size": int(len(train_idx)), "total_size": int(len(candidates)), "successes": int(branch.sum())}
    return SurrogateModels(branch_model, residual_models, metrics)


def _evaluate_seed_event(seed: SearchSeed, path: Path, event_type: str) -> SearchCandidate:
    """Evaluate one seed with the true Taylor solver and write a named event."""
    with mp.workdps(SCOUT_CONFIG.working_dps):
        candidate = SearchCandidate(seed, two_sided_residual(seed.point, SCOUT_CONFIG))
    _write_jsonl_event(path, _event(event_type, _candidate_payload(candidate)))
    return candidate


def _evaluate_training_set(path: Path) -> list[SearchCandidate]:
    """Evaluate the full 5000-point true training dataset."""
    candidates = []
    for index, seed in enumerate(_training_seeds(), start=1):
        candidates.append(_evaluate_seed_event(seed, path, "training_sample"))
        if index % 100 == 0:
            print(f"  training samples evaluated: {index}/5000", flush=True)
    return candidates


def _proposal_payload(proposal: SurrogateProposal) -> dict:
    """Return JSON-ready data for one surrogate proposal."""
    return {
        "index": proposal.index,
        "region": proposal.region,
        "point": _point_payload(proposal.point),
        "distance": _mp_string(_point_distance(proposal.point)),
        "asymmetry": _mp_string(_asymmetry_distance(proposal.point)),
        "branch_probability": proposal.branch_probability,
        "mean_log_norm": proposal.mean_log_norm,
        "std_log_norm": proposal.std_log_norm,
        "score": proposal.score,
    }


def _score_points(
    models: SurrogateModels,
    region: str,
    points: list[TwoSidedSearchPoint],
    start_index: int,
) -> list[SurrogateProposal]:
    """Score one batch of candidate points with the surrogate ensemble."""
    features = _features_for_points(points)
    branch_probs = _branch_probabilities(models.branch_model, features)
    mean_logs, std_logs = _residual_predictions(models.residual_models, features)
    proposals = []
    for offset, point in enumerate(points):
        if branch_probs[offset] < BRANCH_PROBABILITY_CUTOFF:
            continue
        score = float(mean_logs[offset] - OPTIMISM_WEIGHT * std_logs[offset])
        proposals.append(SurrogateProposal(start_index + offset, region, point, float(branch_probs[offset]), float(mean_logs[offset]), float(std_logs[offset]), score))
    return proposals


def _lhs_points(spec: RegionSpec, count: int, rng: Random) -> list[TwoSidedSearchPoint]:
    """Return filtered Latin-hypercube points for acquisition."""
    points = []
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


def _select_diverse_proposals(proposals: list[SurrogateProposal], quota: int) -> list[SurrogateProposal]:
    """Select low-score proposals while preserving geometric spread."""
    ordered = sorted(proposals, key=lambda item: item.score)
    selected = ordered[: min(quota // 2, len(ordered))]
    candidate_pool = ordered[: max(quota * 20, quota)]
    while len(selected) < quota:
        keys = {_point_payload(item.point).__repr__() for item in selected}
        remaining = [item for item in candidate_pool if _point_payload(item.point).__repr__() not in keys]
        if not remaining or not selected:
            break
        picked = max(remaining, key=lambda item: min(_point_distance_between(item.point, chosen.point) for chosen in selected))
        selected.append(picked)
    return selected


def _acquire_surrogate_proposals(models: SurrogateModels, path: Path) -> list[SurrogateProposal]:
    """Generate and select surrogate proposals from a large acquisition pool."""
    rng = Random(RANDOM_SEED + 55)
    selected: list[SurrogateProposal] = []
    proposal_index = 0
    for spec in ACQUISITION_REGIONS:
        kept: list[SurrogateProposal] = []
        remaining = spec.samples
        while remaining:
            count = min(ACQUISITION_CHUNK_SIZE, remaining)
            remaining -= count
            points = _lhs_points(RegionSpec(spec.name, spec.ranges, count, 0, 0, 0, spec.min_distance, spec.max_distance, spec.min_asymmetry), count, rng)
            proposals = _score_points(models, spec.name, points, proposal_index)
            proposal_index += count
            kept.extend(proposals)
            kept = sorted(kept, key=lambda item: item.score)[: max(PROPOSAL_QUOTAS[spec.name] * 50, 1000)]
        region_selected = _select_diverse_proposals(kept, PROPOSAL_QUOTAS[spec.name])
        for proposal in region_selected:
            _write_jsonl_event(path, _event("surrogate_proposal", _proposal_payload(proposal)))
        print(f"  surrogate proposals selected for {spec.name}: {len(region_selected)}", flush=True)
        selected.extend(region_selected)
    return selected


def _evaluate_proposals(proposals: list[SurrogateProposal], path: Path) -> list[SearchCandidate]:
    """Evaluate surrogate proposals with the true cheap Taylor solver."""
    candidates = []
    for index, proposal in enumerate(proposals, start=1):
        seed = SearchSeed(100_000 + index, proposal.region, "surrogate", proposal.point)
        candidates.append(_evaluate_seed_event(seed, path, "true_proposal_evaluation"))
        if index % 50 == 0:
            print(f"  surrogate proposals evaluated: {index}/{len(proposals)}", flush=True)
    return candidates


def _select_refinement_candidates(candidates: list[SearchCandidate], count: int) -> list[SelectedCandidate]:
    """Select best and diverse verified proposals for order-6 refinement."""
    successful = _successful(candidates)
    best_count = min(count // 2, len(successful))
    best = successful[:best_count]
    diverse = _diverse_candidates(successful, best, count - best_count)
    selected = [SelectedCandidate(index + 1, "surrogate-best", candidate) for index, candidate in enumerate(best)]
    selected += [SelectedCandidate(len(selected) + index + 1, "surrogate-diverse", candidate) for index, candidate in enumerate(diverse)]
    return selected[:count]


def _model_metrics_payload(models: SurrogateModels, phase: str) -> dict:
    """Return JSON-ready model metrics."""
    return {"phase": phase, "metrics": models.metrics, "feature_names": FEATURE_NAMES}


def _best_payload(candidates: list[SearchCandidate], limit: int = 20) -> list[dict]:
    """Return the best true residual candidates as JSON-ready records."""
    return [_candidate_payload(candidate) for candidate in _successful(candidates)[:limit]]


def _classification_counts(tracks) -> dict[str, int]:
    """Return classification counts for promoted tracks."""
    counts: dict[str, int] = {}
    for track in tracks:
        counts[track.classification] = counts.get(track.classification, 0) + 1
    return counts


def main() -> None:
    """Run the XGBoost-assisted non-mirrored search."""
    jsonl_path, summary_path = _output_paths()
    print(f"writing training JSONL to {jsonl_path}", flush=True)
    _write_jsonl_event(jsonl_path, _event("run_start", {"random_seed": RANDOM_SEED, "regions": [spec.name for spec in TRAINING_REGIONS], "feature_names": FEATURE_NAMES}))

    training_candidates = _evaluate_training_set(jsonl_path)
    initial_models = _train_models(training_candidates)
    _write_jsonl_event(jsonl_path, _event("model_metrics", _model_metrics_payload(initial_models, "initial")))

    proposals = _acquire_surrogate_proposals(initial_models, jsonl_path)
    proposal_candidates = _evaluate_proposals(proposals, jsonl_path)
    combined_candidates = training_candidates + proposal_candidates
    final_models = _train_models(combined_candidates, RANDOM_SEED + 1)
    _write_jsonl_event(jsonl_path, _event("model_metrics", _model_metrics_payload(final_models, "after_proposals")))

    refs = _verify_point(BASE_TWO_SIDED_POINT, VERIFY_CONFIGS)
    selected = _select_refinement_candidates(proposal_candidates, 40)
    tracks = [_initial_track(selection, jsonl_path, ORDER6_SETTINGS) for selection in selected]
    tracks.sort(key=lambda track: _track_final_result(track).residual_norm)
    promoted = []
    for track in tracks[:16]:
        stage_track = _initial_promoted(track, refs, jsonl_path)
        promoted.append(stage_track)

    payload = {
        "training_summary": _region_summary(training_candidates),
        "proposal_summary": _region_summary(proposal_candidates),
        "model_metrics": {"initial": initial_models.metrics, "after_proposals": final_models.metrics},
        "best_training_candidates": _best_payload(training_candidates),
        "best_proposal_candidates": _best_payload(proposal_candidates),
        "symmetric_references": [_result_payload(result) for result in refs],
        "classification_counts": _classification_counts(promoted),
        "tracks": [_track_payload(track) for track in promoted],
        "physical_parameters": [_physical_payload(_track_final_result(track).point) for track in promoted],
    }
    _write_jsonl_event(jsonl_path, _event("run_summary", payload))
    _write_summary(summary_path, payload)
    print(f"summary written to {summary_path}", flush=True)


def _initial_promoted(track, refs: tuple[TwoSidedResidualResult, ...], path: Path):
    """Run order-10 refinement and classify one promoted surrogate track."""
    from ..shared.non_mirrored_common import _promote_track

    return _promote_track(track, refs, path, ORDER10_SETTINGS, VERIFY_CONFIGS)


if __name__ == "__main__":
    main()
