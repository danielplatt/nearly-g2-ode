"""Refine selected candidates from the calibrated non-mirrored grid scout."""

from __future__ import annotations

import argparse
import json
import signal
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

from mpmath import mp

from problem import DEFAULT_CONFIG, DEFAULT_PARAMS, ProblemParameters, SolverConfig
from solver.two_sided_refinement import (
    TwoSidedCandidateTrack,
    TwoSidedNewtonSettings,
    TwoSidedRefinementStageReport,
    two_sided_newton_refine,
)
from solver.two_sided_shooting import BASE_TWO_SIDED_POINT, TwoSidedResidualResult, TwoSidedSearchPoint, params_from_two_sided_scaled, two_sided_residual

from ..shared.non_mirrored_common import (
    RANDOM_SEED,
    S_MIN,
    SearchCandidate,
    SearchSeed,
    _asymmetry_distance,
    _coordinates,
    _event,
    _mp_string,
    _output_paths as _common_output_paths,
    _point_distance,
    _point_distance_between,
    _result_payload,
    _stage_payload,
    _track_final_result,
    _track_payload,
    _verification_norms,
    _write_jsonl_event,
    _write_summary,
)

from . import non_mirrored_grid_search as grid_search


OUTPUT_DIR = Path("output/non_mirrored_grid_refinements")
REFINEMENT_VERSION = "grid-refine-v1"
OUTPUT_SUFFIX = "non-mirrored-grid-refine-v1"
SELECTION_VERSION = "balanced-local-v1"
BALANCED_SELECTION_MODE = "balanced-50"
LOCAL_MINIMA_SELECTION_MODE = "local-minima"
SELECTION_MODES = (BALANCED_SELECTION_MODE, LOCAL_MINIMA_SELECTION_MODE)
DEFAULT_SELECTION_QUOTA = 50
LOCAL_BEST_QUOTA = 36
LOCAL_BEST_THRESHOLD = mp.mpf("0.075")
LOCAL_DIVERSE_THRESHOLD = mp.mpf("0.15")
ASYM_FILL_THRESHOLD = mp.mpf("0.1")
MIN_DIVERSE_ASYMMETRY = mp.mpf("0.4")
NON_BERGER_DISTANCE = mp.mpf("0.05")
NON_MIRRORED_ASYMMETRY = mp.mpf("0.05")
ORDER6_TIMEOUT_SECONDS = 12 * 60
PROMOTION_TIMEOUT_SECONDS = 25 * 60
DEFAULT_MAX_NEWTON_COORDINATE = mp.mpf("2")
MAX_NEWTON_COORDINATE = DEFAULT_MAX_NEWTON_COORDINATE

SCOUT_CONFIG = grid_search.SCOUT_CONFIG
ORDER6_CONFIG = SolverConfig(6, 40, 20, mp.mpf("0.9"), 0, DEFAULT_CONFIG.match_t)
ORDER10_CONFIG = SolverConfig(10, 70, 30, mp.mpf("0.7"), 1, DEFAULT_CONFIG.match_t)
VERIFY14_CONFIG = SolverConfig(14, 90, 35, mp.mpf("0.6"), 2, DEFAULT_CONFIG.match_t)
VERIFY18_CONFIG = SolverConfig(18, 110, 45, mp.mpf("0.55"), 2, DEFAULT_CONFIG.match_t)
VERIFY_CONFIGS = (VERIFY14_CONFIG, VERIFY18_CONFIG)
REFERENCE_CONFIGS = (SCOUT_CONFIG, ORDER6_CONFIG, ORDER10_CONFIG, VERIFY14_CONFIG, VERIFY18_CONFIG)

def _settings_for_max_coordinate(max_coordinate: mp.mpf) -> tuple[TwoSidedNewtonSettings, TwoSidedNewtonSettings, TwoSidedNewtonSettings]:
    """Return refinement settings with the requested coordinate guard."""
    return (
        TwoSidedNewtonSettings(
            "order-6-grid-refine",
            ORDER6_CONFIG,
            mp.mpf("1e-3"),
            mp.mpf("1e-8"),
            3,
            max_abs_coordinate=max_coordinate,
            min_s_coordinate=S_MIN,
        ),
        TwoSidedNewtonSettings(
            "order-10-grid-refine",
            ORDER10_CONFIG,
            mp.mpf("3e-4"),
            mp.mpf("1e-10"),
            3,
            max_abs_coordinate=max_coordinate,
            min_s_coordinate=S_MIN,
        ),
        TwoSidedNewtonSettings(
            "order-14-grid-correction",
            VERIFY14_CONFIG,
            mp.mpf("1e-4"),
            mp.mpf("1e-12"),
            2,
            max_abs_coordinate=max_coordinate,
            min_s_coordinate=S_MIN,
        ),
    )


def _configure_newton_settings(max_coordinate: mp.mpf) -> None:
    """Set the process-local refinement coordinate guard."""
    global MAX_NEWTON_COORDINATE, ORDER6_SETTINGS, ORDER10_SETTINGS, ORDER14_SETTINGS, SETTINGS_BY_ORDER
    MAX_NEWTON_COORDINATE = max_coordinate
    ORDER6_SETTINGS, ORDER10_SETTINGS, ORDER14_SETTINGS = _settings_for_max_coordinate(max_coordinate)
    SETTINGS_BY_ORDER = {
        ORDER6_CONFIG.series_order: ORDER6_SETTINGS,
        ORDER10_CONFIG.series_order: ORDER10_SETTINGS,
        VERIFY14_CONFIG.series_order: ORDER14_SETTINGS,
    }


_configure_newton_settings(DEFAULT_MAX_NEWTON_COORDINATE)


@dataclass(frozen=True)
class SelectedGridCandidate:
    """One selected grid scout candidate and the reason it was selected."""

    rank: int
    reason: str
    candidate: SearchCandidate


class _TimeoutExpired(Exception):
    """Raised by the signal-based timeout guard."""


T = TypeVar("T")


@contextmanager
def _time_limit(seconds: float | None):
    """Raise `_TimeoutExpired` if a block exceeds the requested wall time."""
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


def _positive_int(value: str) -> int:
    """Parse one positive integer CLI argument."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    """Parse one positive float CLI argument."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_mpf(value: str) -> mp.mpf:
    """Parse one positive mpmath decimal CLI argument."""
    with mp.workdps(80):
        parsed = mp.mpf(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _iter_jsonl_events(path: Path):
    """Yield complete JSONL events, ignoring a possible partial final line."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _jsonl_has_event(path: Path, event_type: str) -> bool:
    """Return whether one JSONL checkpoint contains an event type."""
    return any(event.get("event") == event_type for event in _iter_jsonl_events(path))


def _latest_completed_scout_jsonl() -> Path:
    """Return the newest completed grid-search JSONL checkpoint."""
    pattern = f"*-seed{RANDOM_SEED}-{grid_search.OUTPUT_SUFFIX}.jsonl"
    candidates = sorted(grid_search.OUTPUT_DIR.glob(pattern), reverse=True)
    for path in candidates:
        if _jsonl_has_event(path, "run_summary"):
            return path
    raise FileNotFoundError(f"No completed grid-search checkpoint found under {grid_search.OUTPUT_DIR}")


def _scout_run_start(path: Path) -> dict:
    """Return the run_start event from one scout checkpoint."""
    for event in _iter_jsonl_events(path):
        if event.get("event") == "run_start":
            return event
    raise ValueError(f"{path} does not contain a run_start event")


def _scout_region(run_start: dict) -> str:
    """Return the scout grid region, defaulting to the original Berger chart."""
    return str(run_start.get("grid", {}).get("region", "near"))


def _base_params_for_region(region: str) -> ProblemParameters:
    """Return the physical base endpoint parameters for one persisted scout region."""
    try:
        return grid_search._base_params_for_region(region)
    except ValueError:
        return DEFAULT_PARAMS


def _base_params_for_run_start(run_start: dict) -> ProblemParameters:
    """Return the physical base endpoint parameters for one scout run_start event."""
    return _base_params_for_region(_scout_region(run_start))


def _uses_default_base(region: str) -> bool:
    """Return whether one scout region uses the original Berger endpoint chart."""
    return _base_params_for_region(region) == DEFAULT_PARAMS


def _physical_payload(point: TwoSidedSearchPoint, base_params: ProblemParameters) -> dict[str, str | None]:
    """Return physical parameter values for one scaled point in a chosen chart."""
    params, config = params_from_two_sided_scaled(point, base_params=base_params)
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


def _config_from_payload(payload: dict) -> SolverConfig:
    """Return a minimal solver config matching a serialized residual payload."""
    order = int(payload.get("config_order", SCOUT_CONFIG.series_order))
    dps = int(payload.get("config_dps", SCOUT_CONFIG.working_dps))
    if order == SCOUT_CONFIG.series_order:
        template = SCOUT_CONFIG
    elif order == ORDER6_CONFIG.series_order:
        template = ORDER6_CONFIG
    elif order == ORDER10_CONFIG.series_order:
        template = ORDER10_CONFIG
    elif order == VERIFY14_CONFIG.series_order:
        template = VERIFY14_CONFIG
    elif order == VERIFY18_CONFIG.series_order:
        template = VERIFY18_CONFIG
    else:
        template = DEFAULT_CONFIG
    return SolverConfig(order, dps, template.target_dps, template.step_safety, template.sample_points, template.match_t)


def _result_from_payload(payload: dict, base_params: ProblemParameters = DEFAULT_PARAMS) -> TwoSidedResidualResult:
    """Rebuild one residual result from JSON payload strings."""
    point = _point_from_payload(payload["point"])
    config = _config_from_payload(payload)
    params, local_config = params_from_two_sided_scaled(point, base_params=base_params, template_config=config)
    residual = tuple(mp.mpf(value) for value in payload["residual"])
    branch = {key: mp.mpf(value) for key, value in payload["branch_diagnostics"].items()}
    return TwoSidedResidualResult(
        point,
        params,
        local_config,
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


def _candidate_from_payload(payload: dict) -> SearchCandidate:
    """Rebuild a scout candidate from a persisted JSONL payload."""
    point = _point_from_payload(payload["seed_point"])
    seed = SearchSeed(int(payload["seed_index"]), payload["region"], payload["source"], point)
    return SearchCandidate(seed, _result_from_payload(payload["result"], _base_params_for_region(seed.region)))


def _stage_from_payload(payload: dict, base_params: ProblemParameters = DEFAULT_PARAMS) -> TwoSidedRefinementStageReport:
    """Rebuild a refinement stage without reconstructing per-step internals."""
    order = int(payload["settings"]["order"])
    return TwoSidedRefinementStageReport(
        SETTINGS_BY_ORDER[order],
        _result_from_payload(payload["initial"], base_params),
        _result_from_payload(payload["final"], base_params),
        (),
        payload["status"],
    )


def _track_from_payload(payload: dict) -> TwoSidedCandidateTrack:
    """Rebuild one candidate track from a persisted track payload."""
    region = payload["region"]
    base_params = _base_params_for_region(region)
    stages = tuple(_stage_from_payload(stage, base_params) for stage in payload["stages"])
    verifications = tuple(_result_from_payload(result, base_params) for result in payload.get("verifications", ()))
    return TwoSidedCandidateTrack(
        int(payload["seed_index"]),
        region,
        _point_from_payload(payload["seed_point"]),
        _result_from_payload(payload["scout"], base_params),
        stages,
        verifications,
        payload["classification"],
    )


def _load_scout_candidates(path: Path) -> list[SearchCandidate]:
    """Load all scout_result candidates from one grid-search JSONL checkpoint."""
    return [_candidate_from_payload(event) for event in _iter_jsonl_events(path) if event.get("event") == "scout_result"]


def _successful_candidates(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    """Return branch-valid candidates sorted by residual norm and seed index."""
    return sorted(
        (candidate for candidate in candidates if candidate.result.failure is None),
        key=lambda candidate: (candidate.result.residual_norm, candidate.seed.index),
    )


def _candidate_norm(candidate: SearchCandidate) -> mp.mpf:
    """Return the scout residual norm, or infinity for failures."""
    return candidate.result.residual_norm if candidate.result.failure is None else mp.inf


def _grid_shape(run_start: dict) -> tuple[int, ...]:
    """Return the axis counts from a scout run_start event."""
    return tuple(int(value) for value in run_start["grid"]["axis_counts"])


def _grid_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return row-major strides for the persisted grid seed order."""
    strides = []
    product = 1
    for size in reversed(shape):
        strides.append(product)
        product *= size
    return tuple(reversed(strides))


def _grid_coordinate(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return the grid coordinate tuple for one seed index."""
    strides = _grid_strides(shape)
    remaining = index
    coordinates = []
    for stride, size in zip(strides, shape):
        coordinates.append(remaining // stride)
        remaining %= stride
        if coordinates[-1] >= size:
            raise ValueError(f"Seed index {index} is outside grid shape {shape}")
    return tuple(coordinates)


def _grid_index(coordinates: tuple[int, ...], shape: tuple[int, ...]) -> int:
    """Return the seed index for one grid coordinate tuple."""
    return sum(value * stride for value, stride in zip(coordinates, _grid_strides(shape)))


def _neighbor_indices(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return nearest-neighbor grid indices for one seed index."""
    coordinates = _grid_coordinate(index, shape)
    output = []
    for dimension, size in enumerate(shape):
        for offset in (-1, 1):
            neighbor = list(coordinates)
            neighbor[dimension] += offset
            if 0 <= neighbor[dimension] < size:
                output.append(_grid_index(tuple(neighbor), shape))
    return tuple(output)


def _local_minima(candidates: list[SearchCandidate], shape: tuple[int, ...]) -> list[SearchCandidate]:
    """Return branch-valid nearest-neighbor local minima in residual norm."""
    by_index = {candidate.seed.index: candidate for candidate in candidates}
    minima = []
    for candidate in candidates:
        if candidate.result.failure is not None:
            continue
        norm = candidate.result.residual_norm
        if all(norm <= _candidate_norm(by_index[neighbor]) for neighbor in _neighbor_indices(candidate.seed.index, shape) if neighbor in by_index):
            minima.append(candidate)
    return sorted(minima, key=lambda candidate: (candidate.result.residual_norm, candidate.seed.index))


def _swapped_point_key(point: TwoSidedSearchPoint) -> tuple[str, ...]:
    """Return the exact serialized key after swapping left/right endpoint coordinates."""
    swapped = (point.u_right, point.v_right, point.r_right, point.u_left, point.v_left, point.r_left, point.s)
    return tuple(_mp_string(value) for value in swapped)


def _point_key(point: TwoSidedSearchPoint) -> tuple[str, ...]:
    """Return an exact serialized point key."""
    return tuple(_mp_string(value) for value in _coordinates(point))


def _canonical_swap_key(candidate: SearchCandidate) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return a left/right-swap-invariant key for one candidate."""
    direct = _point_key(candidate.seed.point)
    swapped = _swapped_point_key(candidate.seed.point)
    return min(direct, swapped), max(direct, swapped)


def _dedupe_left_right(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    """Deduplicate exact left/right-swapped candidates, keeping the lower seed index."""
    kept: dict[tuple[tuple[str, ...], tuple[str, ...]], SearchCandidate] = {}
    for candidate in sorted(candidates, key=lambda item: item.seed.index):
        key = _canonical_swap_key(candidate)
        if key not in kept:
            kept[key] = candidate
    return list(kept.values())


def _candidate_selected_payload(selection: SelectedGridCandidate) -> dict:
    """Return JSON-ready data for one selected scout candidate."""
    candidate = selection.candidate
    return {
        "rank": selection.rank,
        "reason": selection.reason,
        "seed_index": candidate.seed.index,
        "region": candidate.seed.region,
        "source": candidate.seed.source,
        "distance": _mp_string(_point_distance(candidate.seed.point)),
        "asymmetry": _mp_string(_asymmetry_distance(candidate.seed.point)),
        "residual_norm": _mp_string(candidate.result.residual_norm),
        "seed_point": {name: _mp_string(value) for name, value in zip(grid_search.COORDINATE_NAMES, _coordinates(candidate.seed.point))},
    }


def _append_unique_selection(
    selected: list[SelectedGridCandidate],
    candidate: SearchCandidate,
    reason: str,
    selected_indices: set[int],
    selected_keys: set[tuple[tuple[str, ...], tuple[str, ...]]],
) -> bool:
    """Append one candidate if it has not already been selected."""
    key = _canonical_swap_key(candidate)
    if candidate.seed.index in selected_indices or key in selected_keys:
        return False
    selected.append(SelectedGridCandidate(len(selected) + 1, reason, candidate))
    selected_indices.add(candidate.seed.index)
    selected_keys.add(key)
    return True


def _select_diverse(
    pool: list[SearchCandidate],
    selected: list[SelectedGridCandidate],
    quota: int,
    reason: str,
    selected_indices: set[int],
    selected_keys: set[tuple[tuple[str, ...], tuple[str, ...]]],
) -> None:
    """Greedily fill selections with max-min separated candidates."""
    while len(selected) < quota:
        remaining = [candidate for candidate in pool if candidate.seed.index not in selected_indices and _canonical_swap_key(candidate) not in selected_keys]
        if not remaining:
            return
        if not selected:
            picked = min(remaining, key=lambda candidate: (candidate.result.residual_norm, candidate.seed.index))
        else:
            picked = max(
                remaining,
                key=lambda candidate: (
                    min(_point_distance_between(candidate.seed.point, item.candidate.seed.point) for item in selected),
                    -float(candidate.result.residual_norm),
                    -candidate.seed.index,
                ),
            )
        _append_unique_selection(selected, picked, reason, selected_indices, selected_keys)


def _select_candidates(candidates: list[SearchCandidate], run_start: dict, quota: int = DEFAULT_SELECTION_QUOTA) -> list[SelectedGridCandidate]:
    """Return the deterministic balanced grid-refinement selection."""
    shape = _grid_shape(run_start)
    local_minima = _dedupe_left_right(_local_minima(candidates, shape))
    local_minima.sort(key=lambda candidate: (candidate.result.residual_norm, candidate.seed.index))
    successes = _dedupe_left_right(_successful_candidates(candidates))
    successes.sort(key=lambda candidate: (candidate.result.residual_norm, candidate.seed.index))

    selected: list[SelectedGridCandidate] = []
    selected_indices: set[int] = set()
    selected_keys: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

    for candidate in local_minima:
        if len([item for item in selected if item.reason == "local-best"]) >= min(LOCAL_BEST_QUOTA, quota):
            break
        if candidate.result.residual_norm < LOCAL_BEST_THRESHOLD:
            _append_unique_selection(selected, candidate, "local-best", selected_indices, selected_keys)

    diverse_pool = [
        candidate
        for candidate in local_minima
        if candidate.result.residual_norm < LOCAL_DIVERSE_THRESHOLD and _asymmetry_distance(candidate.seed.point) >= MIN_DIVERSE_ASYMMETRY
    ]
    _select_diverse(diverse_pool, selected, quota, "local-diverse", selected_indices, selected_keys)

    asym_pool = [
        candidate
        for candidate in successes
        if candidate.result.residual_norm < ASYM_FILL_THRESHOLD and _asymmetry_distance(candidate.seed.point) >= MIN_DIVERSE_ASYMMETRY
    ]
    for candidate in asym_pool:
        if len(selected) >= quota:
            break
        _append_unique_selection(selected, candidate, "asym-fill", selected_indices, selected_keys)

    for candidate in successes:
        if len(selected) >= quota:
            break
        _append_unique_selection(selected, candidate, "best-fill", selected_indices, selected_keys)

    return selected


def _select_local_minima(candidates: list[SearchCandidate], run_start: dict, max_residual: mp.mpf | None = None) -> list[SelectedGridCandidate]:
    """Return all canonical branch-valid local minima from one scout grid."""
    shape = _grid_shape(run_start)
    local_minima = _dedupe_left_right(_local_minima(candidates, shape))
    if max_residual is not None:
        local_minima = [candidate for candidate in local_minima if candidate.result.residual_norm < max_residual]
    local_minima.sort(key=lambda candidate: (candidate.result.residual_norm, candidate.seed.index))
    return [SelectedGridCandidate(index + 1, "local-minimum", candidate) for index, candidate in enumerate(local_minima)]


def _select_for_mode(
    candidates: list[SearchCandidate],
    run_start: dict,
    selection_mode: str,
    quota: int | None,
    local_minimum_max_residual: mp.mpf | None,
) -> list[SelectedGridCandidate]:
    """Select candidates according to the requested refinement policy."""
    if selection_mode == BALANCED_SELECTION_MODE:
        return _select_candidates(candidates, run_start, quota or DEFAULT_SELECTION_QUOTA)
    if selection_mode == LOCAL_MINIMA_SELECTION_MODE:
        return _select_local_minima(candidates, run_start, local_minimum_max_residual)
    raise ValueError(f"Unknown selection mode {selection_mode!r}")


def _reference_residuals(base_params: ProblemParameters = DEFAULT_PARAMS) -> tuple[TwoSidedResidualResult, ...]:
    """Return base-chart reference residuals at all refinement orders."""
    results = []
    for config in REFERENCE_CONFIGS:
        with mp.workdps(config.working_dps):
            results.append(two_sided_residual(BASE_TWO_SIDED_POINT, config, base_params=base_params))
    return tuple(results)


def _verify_point(point: TwoSidedSearchPoint, base_params: ProblemParameters = DEFAULT_PARAMS) -> tuple[TwoSidedResidualResult, ...]:
    """Evaluate one point at the high-order verification configs."""
    results = []
    for config in VERIFY_CONFIGS:
        with mp.workdps(config.working_dps):
            results.append(two_sided_residual(point, config, base_params=base_params))
    return tuple(results)


def _verification_thresholds(references: tuple[TwoSidedResidualResult, ...]) -> tuple[mp.mpf, ...]:
    """Return Berger-relative order-14/order-18 recovery thresholds."""
    return tuple(max(mp.mpf("1e-8"), mp.mpf("1000") * result.residual_norm) for result in references)


def _stable_within_factor(norms: tuple[mp.mpf, ...], factor: mp.mpf) -> bool:
    """Return whether verification norms are stable within a multiplicative factor."""
    positive = [norm for norm in norms if norm != 0]
    return len(norms) >= 2 and (not positive or max(positive) <= factor * min(positive))


def _track_final(track: TwoSidedCandidateTrack) -> TwoSidedResidualResult:
    """Return the final residual result for one track."""
    return track.stages[-1].final if track.stages else track.scout_result


def _has_failed_stage(track: TwoSidedCandidateTrack) -> bool:
    """Return whether any refinement stage failed fatally."""
    fatal = {"branch_failure", "jacobian_failure", "no_improvement", "timeout"}
    return any(stage.status in fatal or stage.final.failure for stage in track.stages)


def _classify_track(
    track: TwoSidedCandidateTrack,
    references: tuple[TwoSidedResidualResult, ...],
    *,
    allow_recovered_berger: bool = True,
) -> str:
    """Classify one grid-refinement track."""
    if track.scout_result.failure or any(result.failure for result in track.verifications):
        return "failed"
    final = _track_final(track)
    norms = _verification_norms(track)
    distance = _point_distance(final.point)
    asymmetry = _asymmetry_distance(final.point)
    if final.residual_norm < mp.mpf("1e-8") and norms and max(norms) > mp.mpf("1e-4"):
        return "finite_order_artifact"
    if allow_recovered_berger and len(norms) == 2 and distance < mp.mpf("1e-3"):
        if all(norm <= threshold for norm, threshold in zip(norms, _verification_thresholds(references))):
            return "recovered_berger"
    if len(norms) == 2 and (distance >= NON_BERGER_DISTANCE or not allow_recovered_berger):
        if max(norms) < mp.mpf("1e-8") and _stable_within_factor(norms, mp.mpf("10")):
            return "possible_non_mirrored_candidate" if asymmetry >= NON_MIRRORED_ASYMMETRY else "possible_symmetric_non_berger_candidate"
    if _has_failed_stage(track):
        return "failed"
    return "inconclusive"


def _timeout_result(
    point: TwoSidedSearchPoint,
    config: SolverConfig,
    message: str,
    base_params: ProblemParameters = DEFAULT_PARAMS,
) -> TwoSidedResidualResult:
    """Return a failed residual result for timeout diagnostics."""
    params, local_config = params_from_two_sided_scaled(point, base_params=base_params, template_config=config)
    return TwoSidedResidualResult(point, params, local_config, (), mp.inf, None, None, None, None, (0, 0), {}, message)


def _timeout_stage(
    point: TwoSidedSearchPoint,
    settings: TwoSidedNewtonSettings,
    message: str,
    base_params: ProblemParameters = DEFAULT_PARAMS,
) -> TwoSidedRefinementStageReport:
    """Return a refinement stage report for a timeout."""
    result = _timeout_result(point, settings.config, message, base_params)
    return TwoSidedRefinementStageReport(settings, result, result, (), message)


def _run_order6(
    selection: SelectedGridCandidate,
    path: Path,
    timeout_seconds: float | None,
    base_params: ProblemParameters = DEFAULT_PARAMS,
) -> TwoSidedCandidateTrack:
    """Run guarded order-6 refinement for one selection."""
    stage, status = _run_with_timeout(
        lambda: two_sided_newton_refine(selection.candidate.seed.point, ORDER6_SETTINGS, base_params=base_params),
        timeout_seconds,
    )
    if stage is None:
        stage = _timeout_stage(selection.candidate.seed.point, ORDER6_SETTINGS, status or "timeout", base_params)
    track = TwoSidedCandidateTrack(selection.candidate.seed.index, selection.candidate.seed.region, selection.candidate.seed.point, selection.candidate.result, (stage,), (), "inconclusive")
    _write_jsonl_event(path, _event("refinement_stage", {"selection_reason": selection.reason, "stage": _stage_payload(stage), "track": _track_payload(track)}))
    return track


def _deserves_order10(stage: TwoSidedRefinementStageReport) -> bool:
    """Return whether an order-6 stage deserves order-10 refinement."""
    return stage.final.failure is None and stage.status not in {"timeout", "jacobian_failure", "no_improvement", "branch_failure"} and stage.final.residual_norm < stage.initial.residual_norm


def _deserves_order14(stage: TwoSidedRefinementStageReport) -> bool:
    """Return whether an order-10 attractor deserves order-14 correction."""
    final = stage.final
    return final.failure is None and (final.residual_norm < mp.mpf("1e-6") or _point_distance(final.point) < mp.mpf("0.02"))


def _deserves_verification(stage: TwoSidedRefinementStageReport) -> bool:
    """Return whether the latest stage deserves high-order verification."""
    final = stage.final
    return final.failure is None and (final.residual_norm < mp.mpf("1e-4") or _point_distance(final.point) < mp.mpf("0.05"))


def _promote_core(
    track: TwoSidedCandidateTrack,
    base_params: ProblemParameters = DEFAULT_PARAMS,
) -> tuple[tuple[TwoSidedRefinementStageReport, ...], tuple[TwoSidedResidualResult, ...]]:
    """Run order-10, optional order-14 correction, and high-order verification."""
    order10 = two_sided_newton_refine(track.stages[-1].final.point, ORDER10_SETTINGS, base_params=base_params)
    stages = [order10]
    if _deserves_order14(order10):
        stages.append(two_sided_newton_refine(order10.final.point, ORDER14_SETTINGS, base_params=base_params))
    verifications = _verify_point(stages[-1].final.point, base_params) if _deserves_verification(stages[-1]) else ()
    return tuple(stages), verifications


def _promote_track(
    track: TwoSidedCandidateTrack,
    references: tuple[TwoSidedResidualResult, ...],
    path: Path,
    timeout_seconds: float | None,
    base_params: ProblemParameters = DEFAULT_PARAMS,
    *,
    allow_recovered_berger: bool = True,
) -> TwoSidedCandidateTrack:
    """Run guarded promotion, verification, classification, and persistence."""
    if not track.stages or not _deserves_order10(track.stages[-1]):
        classified = TwoSidedCandidateTrack(
            track.seed_rank,
            track.seed_region,
            track.seed_point,
            track.scout_result,
            track.stages,
            (),
            _classify_track(track, references, allow_recovered_berger=allow_recovered_berger),
        )
        _write_jsonl_event(path, _event("candidate_classification", _classification_payload(classified)))
        return classified
    promoted, status = _run_with_timeout(lambda: _promote_core(track, base_params), timeout_seconds)
    if promoted is None:
        point = track.stages[-1].final.point
        timeout = _timeout_stage(point, ORDER10_SETTINGS, status or "timeout", base_params)
        classified = TwoSidedCandidateTrack(track.seed_rank, track.seed_region, track.seed_point, track.scout_result, track.stages + (timeout,), (), "failed")
        _write_jsonl_event(path, _event("candidate_classification", _classification_payload(classified)))
        return classified
    promoted_stages, verifications = promoted
    stages = track.stages + promoted_stages
    candidate = TwoSidedCandidateTrack(track.seed_rank, track.seed_region, track.seed_point, track.scout_result, stages, verifications, "inconclusive")
    classified = TwoSidedCandidateTrack(
        track.seed_rank,
        track.seed_region,
        track.seed_point,
        track.scout_result,
        stages,
        verifications,
        _classify_track(candidate, references, allow_recovered_berger=allow_recovered_berger),
    )
    _write_jsonl_event(path, _event("candidate_classification", _classification_payload(classified)))
    return classified


def _classification_payload(track: TwoSidedCandidateTrack) -> dict:
    """Return JSON-ready classification data for one refined grid candidate."""
    payload = _track_payload(track)
    payload["source"] = "grid_refine"
    return payload


def _selection_payload(selection: SelectedGridCandidate) -> dict:
    """Return JSON-ready selection event payload."""
    payload = _candidate_selected_payload(selection)
    payload["event"] = "candidate_selected"
    return payload


def _selection_config_payload(
    quota: int | None,
    order6_timeout: float | None,
    promotion_timeout: float | None,
    selection_mode: str = BALANCED_SELECTION_MODE,
    local_minimum_max_residual: mp.mpf | None = None,
) -> dict:
    """Return JSON-ready selection/refinement config."""
    return {
        "selection_version": SELECTION_VERSION,
        "selection_mode": selection_mode,
        "selection_quota": quota,
        "local_minimum_max_residual": _mp_string(local_minimum_max_residual),
        "local_best_quota": LOCAL_BEST_QUOTA,
        "local_best_threshold": _mp_string(LOCAL_BEST_THRESHOLD),
        "local_diverse_threshold": _mp_string(LOCAL_DIVERSE_THRESHOLD),
        "asym_fill_threshold": _mp_string(ASYM_FILL_THRESHOLD),
        "min_diverse_asymmetry": _mp_string(MIN_DIVERSE_ASYMMETRY),
        "order6_timeout_seconds": order6_timeout,
        "promotion_timeout_seconds": promotion_timeout,
        "max_newton_coordinate": _mp_string(MAX_NEWTON_COORDINATE),
        "s_min": _mp_string(S_MIN),
    }


def _settings_payload() -> list[dict]:
    """Return JSON-ready refinement settings metadata."""
    return [
        {
            "name": settings.name,
            "order": settings.config.series_order,
            "dps": settings.config.working_dps,
            "fd_step": _mp_string(settings.fd_step),
            "max_steps": settings.max_steps,
            "max_abs_coordinate": _mp_string(settings.max_abs_coordinate),
            "min_s_coordinate": _mp_string(settings.min_s_coordinate),
        }
        for settings in (ORDER6_SETTINGS, ORDER10_SETTINGS, ORDER14_SETTINGS)
    ]


def _run_start_payload(
    jsonl_path: Path,
    summary_path: Path,
    scout_jsonl: Path,
    quota: int | None,
    order6_timeout: float | None,
    promotion_timeout: float | None,
    selection_mode: str = BALANCED_SELECTION_MODE,
    local_minimum_max_residual: mp.mpf | None = None,
    scout_region: str = "near",
) -> dict:
    """Return checkpoint metadata for one grid-refinement run."""
    base_params = _base_params_for_region(scout_region)
    return {
        "random_seed": RANDOM_SEED,
        "refinement_version": REFINEMENT_VERSION,
        "scout_region": scout_region,
        "base_params": grid_search._base_params_payload(base_params),
        "scout_jsonl": str(scout_jsonl),
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "selection_config": _selection_config_payload(quota, order6_timeout, promotion_timeout, selection_mode, local_minimum_max_residual),
        "settings": _settings_payload(),
        "verify_orders": [config.series_order for config in VERIFY_CONFIGS],
    }


def _output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped output paths for grid refinement."""
    return _common_output_paths(OUTPUT_DIR, OUTPUT_SUFFIX, now)


def _summary_path_for_jsonl(path: Path) -> Path:
    """Return the summary path paired with one refinement JSONL."""
    return path.with_name(f"{path.stem}-summary.json")


def _checkpoint_is_compatible(
    path: Path,
    scout_jsonl: Path,
    quota: int | None,
    order6_timeout: float | None,
    promotion_timeout: float | None,
    selection_mode: str = BALANCED_SELECTION_MODE,
    local_minimum_max_residual: mp.mpf | None = None,
    scout_region: str = "near",
) -> bool:
    """Return whether an incomplete checkpoint can be resumed."""
    if _jsonl_has_event(path, "run_summary"):
        return False
    starts = [event for event in _iter_jsonl_events(path) if event.get("event") == "run_start"]
    if not starts:
        return False
    expected = _run_start_payload(
        path,
        _summary_path_for_jsonl(path),
        scout_jsonl,
        quota,
        order6_timeout,
        promotion_timeout,
        selection_mode,
        local_minimum_max_residual,
        scout_region,
    )
    return all(starts[-1].get(key) == value for key, value in expected.items() if key not in {"jsonl_path", "summary_path"})


def _latest_incomplete_checkpoint(
    scout_jsonl: Path,
    quota: int | None,
    order6_timeout: float | None,
    promotion_timeout: float | None,
    selection_mode: str = BALANCED_SELECTION_MODE,
    local_minimum_max_residual: mp.mpf | None = None,
    scout_region: str = "near",
) -> Path | None:
    """Return the newest compatible incomplete refinement checkpoint."""
    pattern = f"*-seed{RANDOM_SEED}-{OUTPUT_SUFFIX}.jsonl"
    candidates = sorted(OUTPUT_DIR.glob(pattern), reverse=True)
    return next(
        (
            path
            for path in candidates
            if _checkpoint_is_compatible(
                path,
                scout_jsonl,
                quota,
                order6_timeout,
                promotion_timeout,
                selection_mode,
                local_minimum_max_residual,
                scout_region,
            )
        ),
        None,
    )


def _resume_or_new_paths(
    scout_jsonl: Path,
    quota: int | None,
    order6_timeout: float | None,
    promotion_timeout: float | None,
    selection_mode: str = BALANCED_SELECTION_MODE,
    local_minimum_max_residual: mp.mpf | None = None,
    scout_region: str = "near",
    *,
    resume: bool = True,
    now: datetime | None = None,
) -> tuple[Path, Path, bool]:
    """Return refinement paths, resuming if possible."""
    if resume and now is None:
        checkpoint = _latest_incomplete_checkpoint(
            scout_jsonl,
            quota,
            order6_timeout,
            promotion_timeout,
            selection_mode,
            local_minimum_max_residual,
            scout_region,
        )
        if checkpoint is not None:
            return checkpoint, _summary_path_for_jsonl(checkpoint), True
    jsonl_path, summary_path = _output_paths(now)
    return jsonl_path, summary_path, False


def _selected_seed_indices(path: Path) -> set[int]:
    """Return seed indices already written as candidate_selected events."""
    return {int(event["seed_index"]) for event in _iter_jsonl_events(path) if event.get("event") == "candidate_selected"}


def _classified_seed_indices(path: Path) -> set[int]:
    """Return seed indices already classified in a refinement checkpoint."""
    return {int(event["seed_index"]) for event in _iter_jsonl_events(path) if event.get("event") == "candidate_classification"}


def _load_classified_tracks(path: Path) -> list[TwoSidedCandidateTrack]:
    """Load classified tracks from a refinement checkpoint."""
    return [_track_from_payload(event) for event in _iter_jsonl_events(path) if event.get("event") == "candidate_classification"]


def _write_missing_selections(path: Path, selections: list[SelectedGridCandidate]) -> None:
    """Persist any selection events not already present."""
    existing = _selected_seed_indices(path)
    for selection in selections:
        if selection.candidate.seed.index not in existing:
            _write_jsonl_event(path, _event("candidate_selected", _candidate_selected_payload(selection)))


def _classify_counts(tracks: list[TwoSidedCandidateTrack]) -> dict[str, int]:
    """Return classification counts for refined tracks."""
    return dict(Counter(track.classification for track in tracks))


def _best_verified_tracks(
    tracks: list[TwoSidedCandidateTrack],
    base_params: ProblemParameters = DEFAULT_PARAMS,
    limit: int = 20,
) -> list[dict]:
    """Return compact best-track summaries by final verification norm."""
    def sort_key(track: TwoSidedCandidateTrack):
        norms = _verification_norms(track)
        return (max(norms) if norms else mp.inf, _track_final_result(track).residual_norm, track.seed_rank)

    output = []
    for track in sorted(tracks, key=sort_key)[:limit]:
        final = _track_final_result(track)
        output.append(
            {
                "seed_index": track.seed_rank,
                "region": track.seed_region,
                "classification": track.classification,
                "final_residual_norm": _mp_string(final.residual_norm),
                "verification_norms": [_mp_string(norm) for norm in _verification_norms(track)],
                "distance": _mp_string(_point_distance(final.point)),
                "asymmetry": _mp_string(_asymmetry_distance(final.point)),
                "final_point": {name: _mp_string(value) for name, value in zip(grid_search.COORDINATE_NAMES, _coordinates(final.point))},
                "physical_parameters": _physical_payload(final.point, base_params),
            }
        )
    return output


def _summary_payload(
    scout_jsonl: Path,
    scout_region: str,
    base_params: ProblemParameters,
    references: tuple[TwoSidedResidualResult, ...],
    selections: list[SelectedGridCandidate],
    tracks: list[TwoSidedCandidateTrack],
    selection_config: dict,
) -> dict:
    """Return JSON-ready final refinement summary."""
    return {
        "random_seed": RANDOM_SEED,
        "refinement_version": REFINEMENT_VERSION,
        "scout_region": scout_region,
        "base_params": grid_search._base_params_payload(base_params),
        "scout_jsonl": str(scout_jsonl),
        "selection_config": selection_config,
        "selection_count": len(selections),
        "classified_count": len(tracks),
        "classification_counts": _classify_counts(tracks),
        "selections": [_candidate_selected_payload(selection) for selection in selections],
        "reference_residuals": [_result_payload(result) for result in references],
        "best_verified_tracks": _best_verified_tracks(tracks, base_params),
        "tracks": [_classification_payload(track) for track in tracks],
    }


def _print_selections(selections: list[SelectedGridCandidate]) -> None:
    """Print a compact selected-candidate table."""
    print(f"selected candidates: {len(selections)}", flush=True)
    for selection in selections:
        candidate = selection.candidate
        print(
            "  "
            f"{selection.rank:02d} seed={candidate.seed.index} reason={selection.reason} "
            f"norm={mp.nstr(candidate.result.residual_norm, 8)} "
            f"dist={mp.nstr(_point_distance(candidate.seed.point), 6)} "
            f"asym={mp.nstr(_asymmetry_distance(candidate.seed.point), 6)}",
            flush=True,
        )


def _resolve_scout_jsonl(path_text: str | None) -> Path:
    """Return the requested or newest completed scout checkpoint."""
    return Path(path_text) if path_text else _latest_completed_scout_jsonl()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for grid refinement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scout-jsonl", default=None, help="completed grid scout JSONL to process")
    parser.add_argument("--selection-mode", choices=SELECTION_MODES, default=BALANCED_SELECTION_MODE, help="candidate selection policy")
    parser.add_argument("--quota", type=_positive_int, default=DEFAULT_SELECTION_QUOTA, help="number of candidates to select in balanced-50 mode")
    parser.add_argument("--local-minimum-max-residual", type=_positive_mpf, default=None, help="optional scout residual cutoff for local-minima mode")
    parser.add_argument("--dry-run", action="store_true", help="print selected candidates without refining")
    parser.add_argument("--no-resume", action="store_true", help="start a fresh refinement checkpoint")
    parser.add_argument("--order6-timeout", type=_positive_float, default=float(ORDER6_TIMEOUT_SECONDS), help="order-6 timeout in seconds")
    parser.add_argument("--promotion-timeout", type=_positive_float, default=float(PROMOTION_TIMEOUT_SECONDS), help="promotion timeout in seconds")
    parser.add_argument(
        "--max-newton-coordinate",
        type=_positive_mpf,
        default=DEFAULT_MAX_NEWTON_COORDINATE,
        help="coordinate guard for damped Newton trials",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the grid scout refinement stage."""
    args = _parse_args(argv)
    _configure_newton_settings(args.max_newton_coordinate)
    scout_jsonl = _resolve_scout_jsonl(args.scout_jsonl)
    run_start = _scout_run_start(scout_jsonl)
    scout_region = _scout_region(run_start)
    base_params = _base_params_for_run_start(run_start)
    allow_recovered_berger = _uses_default_base(scout_region)
    candidates = _load_scout_candidates(scout_jsonl)
    selection_quota = args.quota if args.selection_mode == BALANCED_SELECTION_MODE else None
    selections = _select_for_mode(candidates, run_start, args.selection_mode, selection_quota, args.local_minimum_max_residual)
    _print_selections(selections)
    if args.dry_run:
        return

    jsonl_path, summary_path, resumed = _resume_or_new_paths(
        scout_jsonl,
        selection_quota,
        args.order6_timeout,
        args.promotion_timeout,
        args.selection_mode,
        args.local_minimum_max_residual,
        scout_region,
        resume=not args.no_resume,
    )
    run_payload = _run_start_payload(
        jsonl_path,
        summary_path,
        scout_jsonl,
        selection_quota,
        args.order6_timeout,
        args.promotion_timeout,
        args.selection_mode,
        args.local_minimum_max_residual,
        scout_region,
    )
    if resumed:
        print(f"resuming refinement checkpoint {jsonl_path}", flush=True)
    else:
        print(f"writing refinement JSONL to {jsonl_path}", flush=True)
        _write_jsonl_event(jsonl_path, _event("run_start", run_payload))
    _write_missing_selections(jsonl_path, selections)

    references = _reference_residuals(base_params)
    completed = _classified_seed_indices(jsonl_path)
    for selection in selections:
        seed_index = selection.candidate.seed.index
        if seed_index in completed:
            print(f"seed {seed_index}: reused completed classification", flush=True)
            continue
        track = _run_order6(selection, jsonl_path, args.order6_timeout, base_params)
        classified = _promote_track(
            track,
            references[-2:],
            jsonl_path,
            args.promotion_timeout,
            base_params,
            allow_recovered_berger=allow_recovered_berger,
        )
        print(f"seed {seed_index}: {classified.classification}", flush=True)

    tracks = _load_classified_tracks(jsonl_path)
    payload = _summary_payload(scout_jsonl, scout_region, base_params, references, selections, tracks, run_payload["selection_config"])
    _write_jsonl_event(jsonl_path, _event("run_summary", payload))
    _write_summary(summary_path, payload)
    print(f"classifications: {payload['classification_counts']}", flush=True)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
