"""Shared helpers for non-mirrored two-sided search experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random

from mpmath import mp

from problem import DEFAULT_CONFIG, SolverConfig
from solver.two_sided_refinement import (
    TwoSidedCandidateTrack,
    TwoSidedNewtonSettings,
    two_sided_newton_refine,
)
from solver.two_sided_shooting import (
    TwoSidedResidualResult,
    TwoSidedSearchPoint,
    params_from_two_sided_scaled,
    two_sided_residual,
)


RANDOM_SEED = 1729
with mp.workdps(80):
    MIN_MATCH_T = mp.mpf("0.01")
    S_MIN = mp.log(MIN_MATCH_T / DEFAULT_CONFIG.match_t)


@dataclass(frozen=True)
class RegionSpec:
    """One non-mirrored search region with selection quotas."""

    name: str
    ranges: tuple[tuple[mp.mpf, mp.mpf], ...]
    samples: int
    best_quota: int
    diverse_quota: int
    promote_quota: int
    min_distance: mp.mpf = mp.zero
    max_distance: mp.mpf = mp.inf
    min_asymmetry: mp.mpf = mp.zero


@dataclass(frozen=True)
class SearchSeed:
    """One deterministic scout seed."""

    index: int
    region: str
    source: str
    point: TwoSidedSearchPoint


@dataclass(frozen=True)
class SearchCandidate:
    """One evaluated scout seed."""

    seed: SearchSeed
    result: TwoSidedResidualResult


@dataclass(frozen=True)
class SelectedCandidate:
    """One scout candidate selected for refinement."""

    rank: int
    reason: str
    candidate: SearchCandidate


def _coordinates(point: TwoSidedSearchPoint) -> tuple[mp.mpf, ...]:
    """Return scaled coordinates as a tuple."""
    return (
        point.u_left,
        point.v_left,
        point.r_left,
        point.u_right,
        point.v_right,
        point.r_right,
        point.s,
    )


def _point_from_values(values) -> TwoSidedSearchPoint:
    """Build one two-sided search point from numeric values."""
    return TwoSidedSearchPoint(*(mp.mpf(value) for value in values))


def _point_distance(point: TwoSidedSearchPoint) -> mp.mpf:
    """Return max-distance from the Berger base point."""
    return max(abs(value) for value in _coordinates(point))


def _asymmetry_distance(point: TwoSidedSearchPoint) -> mp.mpf:
    """Return max-distance from the mirrored subspace."""
    return max(
        abs(point.u_left - point.u_right),
        abs(point.v_left - point.v_right),
        abs(point.r_left - point.r_right),
    )


def _point_distance_between(left: TwoSidedSearchPoint, right: TwoSidedSearchPoint) -> mp.mpf:
    """Return max-distance between two scaled points."""
    return max(abs(lval - rval) for lval, rval in zip(_coordinates(left), _coordinates(right)))


def _point_key(point: TwoSidedSearchPoint) -> tuple[str, ...]:
    """Return a stable key for deduplicating points."""
    return tuple(_mp_string(value) for value in _coordinates(point))


def _random_point(spec: RegionSpec, rng: Random) -> TwoSidedSearchPoint:
    """Sample one point from a region's rectangular ranges."""
    return _point_from_values(rng.uniform(float(low), float(high)) for low, high in spec.ranges)


def _in_region(point: TwoSidedSearchPoint, spec: RegionSpec) -> bool:
    """Return whether one point satisfies a region's filters."""
    distance = _point_distance(point)
    return (
        spec.min_distance <= distance <= spec.max_distance
        and _asymmetry_distance(point) >= spec.min_asymmetry
        and point.s > S_MIN
    )


def _region_seeds(spec: RegionSpec, rng: Random, start_index: int) -> list[SearchSeed]:
    """Return reproducible random seeds for one scout region."""
    seeds = []
    attempts = 0
    while len(seeds) < spec.samples and attempts < spec.samples * 1000:
        attempts += 1
        point = _random_point(spec, rng)
        if _in_region(point, spec):
            seeds.append(SearchSeed(start_index + len(seeds), spec.name, "random", point))
    if len(seeds) != spec.samples:
        raise RuntimeError(f"Could not sample enough points for region {spec.name!r}.")
    return seeds


def _control_seeds() -> list[SearchSeed]:
    """Return deterministic base and explicitly asymmetric control seeds."""
    values = [
        (0, 0, 0, 0, 0, 0, 0),
        (0.2, -0.1, 0.3, -0.2, 0.1, -0.3, 0),
        (-0.8, 0.4, -1.0, 0.8, -0.4, 1.0, 0.2),
        (1.5, -1.0, 2.0, -1.5, 1.0, -2.0, -0.5),
    ]
    return [SearchSeed(index, "control", "control", _point_from_values(row)) for index, row in enumerate(values)]


def _search_seeds(regions: tuple[RegionSpec, ...], seed: int = RANDOM_SEED) -> list[SearchSeed]:
    """Return the full deterministic scout seed list for a set of regions."""
    rng = Random(seed)
    seeds = _control_seeds()
    for spec in regions:
        seeds.extend(_region_seeds(spec, rng, len(seeds)))
    return seeds


def _mp_string(value) -> str | None:
    """Serialize one mpmath value as a decimal string."""
    return None if value is None else mp.nstr(value, 80)


def _point_payload(point: TwoSidedSearchPoint) -> dict[str, str | None]:
    """Return JSON-ready scaled point coordinates."""
    names = ("u_left", "v_left", "r_left", "u_right", "v_right", "r_right", "s")
    return {name: _mp_string(value) for name, value in zip(names, _coordinates(point))}


def _result_payload(result: TwoSidedResidualResult) -> dict:
    """Return a compact JSON-ready residual result."""
    return {
        "point": _point_payload(result.point),
        "residual_norm": _mp_string(result.residual_norm),
        "residual": [_mp_string(value) for value in result.residual],
        "left_l": _mp_string(result.left_l),
        "right_l": _mp_string(result.right_l),
        "failure": result.failure,
        "patch_counts": list(result.patch_counts),
        "branch_diagnostics": {key: _mp_string(value) for key, value in result.branch_diagnostics.items()},
        "config_order": result.config.series_order,
        "config_dps": result.config.working_dps,
    }


def _event(event_type: str, payload: dict) -> dict:
    """Build one timestamped JSONL event."""
    return {"event": event_type, "time_utc": datetime.now(timezone.utc).isoformat(), **payload}


def _write_jsonl_event(path: Path, event: dict) -> None:
    """Append one JSON event and flush it immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()


def _output_paths(output_dir: Path, suffix: str, now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped JSONL and summary output paths."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    stem = f"{stamp}-seed{RANDOM_SEED}-{suffix}"
    return output_dir / f"{stem}.jsonl", output_dir / f"{stem}-summary.json"


def _candidate_payload(candidate: SearchCandidate) -> dict:
    """Return JSON-ready scout candidate data."""
    return {
        "seed_index": candidate.seed.index,
        "region": candidate.seed.region,
        "source": candidate.seed.source,
        "distance": _mp_string(_point_distance(candidate.seed.point)),
        "asymmetry": _mp_string(_asymmetry_distance(candidate.seed.point)),
        "seed_point": _point_payload(candidate.seed.point),
        "result": _result_payload(candidate.result),
    }


def _evaluate_seed(seed: SearchSeed, path: Path, config: SolverConfig) -> SearchCandidate:
    """Evaluate and persist one scout seed."""
    with mp.workdps(config.working_dps):
        candidate = SearchCandidate(seed, two_sided_residual(seed.point, config))
    _write_jsonl_event(path, _event("scout_result", _candidate_payload(candidate)))
    return candidate


def _sort_key(candidate: SearchCandidate) -> tuple[bool, mp.mpf]:
    """Sort successful candidates before failures by residual norm."""
    return candidate.result.failure is not None, candidate.result.residual_norm


def _successful(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    """Return branch-valid candidates sorted by residual norm."""
    return [candidate for candidate in sorted(candidates, key=_sort_key) if candidate.result.failure is None]


def _diverse_candidates(
    candidates: list[SearchCandidate],
    selected: list[SearchCandidate],
    quota: int,
) -> list[SearchCandidate]:
    """Greedily select max-min separated candidates."""
    chosen = list(selected)
    output = []
    while len(output) < quota:
        keys = {_point_key(item.seed.point) for item in chosen}
        remaining = [candidate for candidate in candidates if _point_key(candidate.seed.point) not in keys]
        if not remaining or not chosen:
            break
        picked = max(
            remaining,
            key=lambda candidate: min(_point_distance_between(candidate.seed.point, item.seed.point) for item in chosen),
        )
        chosen.append(picked)
        output.append(picked)
    return output


def _select_region_candidates(spec: RegionSpec, candidates: list[SearchCandidate]) -> list[SelectedCandidate]:
    """Select best and diverse candidates for one region."""
    region_candidates = _successful([candidate for candidate in candidates if candidate.seed.region == spec.name])
    best = region_candidates[: spec.best_quota]
    diverse = _diverse_candidates(region_candidates, best, spec.diverse_quota)
    selected = [SelectedCandidate(index + 1, "best", candidate) for index, candidate in enumerate(best)]
    offset = len(selected)
    selected += [SelectedCandidate(offset + index + 1, "diverse", candidate) for index, candidate in enumerate(diverse)]
    return selected


def _verify_point(
    point: TwoSidedSearchPoint,
    verify_configs: tuple[SolverConfig, ...],
) -> tuple[TwoSidedResidualResult, ...]:
    """Evaluate one point at high-order verification configs."""
    results = []
    for config in verify_configs:
        with mp.workdps(config.working_dps):
            results.append(two_sided_residual(point, config))
    return tuple(results)


def _track_final_result(track: TwoSidedCandidateTrack) -> TwoSidedResidualResult:
    """Return the latest residual result in one refinement track."""
    return track.stages[-1].final if track.stages else track.scout_result


def _verification_norms(track: TwoSidedCandidateTrack) -> tuple[mp.mpf, ...]:
    """Return finite verification norms for a completed track."""
    return tuple(result.residual_norm for result in track.verifications if result.failure is None)


def _has_failure(track: TwoSidedCandidateTrack) -> bool:
    """Return whether any required stage or verification failed."""
    return any(stage.final.failure for stage in track.stages) or any(result.failure for result in track.verifications)


def _stable_within_factor(norms: tuple[mp.mpf, ...], factor: mp.mpf) -> bool:
    """Return whether nonzero norms are stable within a multiplicative factor."""
    if len(norms) < 2:
        return False
    positive = [norm for norm in norms if norm != 0]
    return not positive or max(positive) <= factor * min(positive)


def _comparable_to_symmetric(
    track: TwoSidedCandidateTrack,
    symmetric_refs: tuple[TwoSidedResidualResult, ...],
) -> bool:
    """Return whether verification residuals are comparable to symmetric errors."""
    if len(track.verifications) != len(symmetric_refs):
        return False
    for result, reference in zip(track.verifications, symmetric_refs):
        threshold = max(mp.mpf("1e-8"), mp.mpf("100") * reference.residual_norm)
        if result.failure or result.residual_norm > threshold:
            return False
    return True


def _classify_track(
    track: TwoSidedCandidateTrack,
    symmetric_refs: tuple[TwoSidedResidualResult, ...],
) -> str:
    """Classify one non-mirrored refinement track."""
    if track.scout_result.failure or _has_failure(track):
        return "branch_failure"
    norms = _verification_norms(track)
    final = _track_final_result(track)
    asymmetry = _asymmetry_distance(final.point)
    if asymmetry >= mp.mpf("0.05") and max(norms or (mp.inf,)) < mp.mpf("1e-6") and _stable_within_factor(norms, mp.mpf("10")):
        return "possible_non_mirrored_candidate"
    if final.residual_norm < mp.mpf("1e-8") and max(norms or (mp.zero,)) > mp.mpf("1e-4"):
        return "finite_order_artifact"
    if asymmetry < mp.mpf("0.02") and _comparable_to_symmetric(track, symmetric_refs):
        return "flows_to_symmetric"
    return "inconclusive"


def _replace_track(
    track: TwoSidedCandidateTrack,
    stages,
    verifications,
    classification: str,
) -> TwoSidedCandidateTrack:
    """Return one track with updated refinement data."""
    return TwoSidedCandidateTrack(
        track.seed_rank,
        track.seed_region,
        track.seed_point,
        track.scout_result,
        tuple(stages),
        tuple(verifications),
        classification,
    )


def _stage_payload(stage) -> dict:
    """Return JSON-ready data for one refinement stage."""
    return {
        "settings": {"name": stage.settings.name, "order": stage.settings.config.series_order},
        "status": stage.status,
        "initial": _result_payload(stage.initial),
        "final": _result_payload(stage.final),
        "steps": [{"index": step.index, "status": step.status, "damping": _mp_string(step.damping)} for step in stage.steps],
    }


def _track_payload(track: TwoSidedCandidateTrack) -> dict:
    """Return JSON-ready data for one candidate track."""
    final = _track_final_result(track)
    return {
        "seed_index": track.seed_rank,
        "region": track.seed_region,
        "seed_point": _point_payload(track.seed_point),
        "final_point": _point_payload(final.point),
        "distance": _mp_string(_point_distance(final.point)),
        "asymmetry": _mp_string(_asymmetry_distance(final.point)),
        "classification": track.classification,
        "scout": _result_payload(track.scout_result),
        "stages": [_stage_payload(stage) for stage in track.stages],
        "verifications": [_result_payload(result) for result in track.verifications],
    }


def _initial_track(
    selection: SelectedCandidate,
    path: Path,
    settings: TwoSidedNewtonSettings,
) -> TwoSidedCandidateTrack:
    """Run the first refinement stage for one selected scout."""
    stage = two_sided_newton_refine(selection.candidate.seed.point, settings)
    track = TwoSidedCandidateTrack(
        selection.candidate.seed.index,
        selection.candidate.seed.region,
        selection.candidate.seed.point,
        selection.candidate.result,
        (stage,),
        (),
        "inconclusive",
    )
    _write_jsonl_event(path, _event("refinement_stage", {"stage": _stage_payload(stage), "track": _track_payload(track)}))
    return track


def _promote_track(
    track: TwoSidedCandidateTrack,
    refs: tuple[TwoSidedResidualResult, ...],
    path: Path,
    settings: TwoSidedNewtonSettings,
    verify_configs: tuple[SolverConfig, ...],
) -> TwoSidedCandidateTrack:
    """Run the second refinement stage and high-order verification for one track."""
    stage = two_sided_newton_refine(track.stages[-1].final.point, settings)
    verifications = _verify_point(stage.final.point, verify_configs)
    promoted = _replace_track(track, track.stages + (stage,), verifications, "inconclusive")
    classified = _replace_track(promoted, promoted.stages, promoted.verifications, _classify_track(promoted, refs))
    _write_jsonl_event(path, _event("candidate_classification", _track_payload(classified)))
    return classified


def _region_summary(candidates: list[SearchCandidate]) -> dict[str, dict[str, int]]:
    """Return scout success/failure counts by region."""
    summary = {}
    for region in sorted({candidate.seed.region for candidate in candidates}):
        subset = [candidate for candidate in candidates if candidate.seed.region == region]
        successes = [candidate for candidate in subset if candidate.result.failure is None]
        summary[region] = {"total": len(subset), "successes": len(successes), "failures": len(subset) - len(successes)}
    return summary


def _write_summary(path: Path, payload: dict) -> None:
    """Write the final summary JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _physical_payload(point: TwoSidedSearchPoint) -> dict[str, str | None]:
    """Return physical parameter values for one scaled point."""
    params, config = params_from_two_sided_scaled(point)
    return {
        "a": _mp_string(params.left.a),
        "c": _mp_string(params.left.c),
        "alpha": _mp_string(params.left.alpha),
        "d": _mp_string(params.right.d),
        "f": _mp_string(params.right.f),
        "omega": _mp_string(params.right.omega),
        "m": _mp_string(config.match_t),
        "T": _mp_string(params.interval_end),
    }


def _print_best(candidates: list[SearchCandidate]) -> None:
    """Print a small best-scout table by region."""
    for region in sorted({candidate.seed.region for candidate in candidates}):
        print(f"\nbest {region} scouts:", flush=True)
        for candidate in _successful([item for item in candidates if item.seed.region == region])[:5]:
            norm = mp.nstr(candidate.result.residual_norm, 12)
            asymmetry = mp.nstr(_asymmetry_distance(candidate.seed.point), 8)
            print(f"  seed={candidate.seed.index}, norm={norm}, asym={asymmetry}", flush=True)
