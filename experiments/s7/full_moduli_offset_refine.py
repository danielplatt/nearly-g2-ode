"""Refine local minima from the S7 terminal-offset full-moduli scout."""

from __future__ import annotations

import argparse
import json
import signal
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypeVar

from mpmath import mp

from problem import ProblemParameters, SolverConfig
from solver.march import solve_two_sided
from solver.two_sided_shooting import config_with_match_t

from ..shared.non_mirrored_common import RANDOM_SEED, _event, _mp_string, _output_paths, _write_jsonl_event, _write_summary
from . import full_moduli_offset_scout as scout
from .right_germ import S7RightGermPoint, offset_moduli_from_point, params_with_right_offset_moduli_germ
from .search_common import S_MIN


OUTPUT_DIR = Path("output/s7_full_moduli_offset_refinements")
REFINEMENT_VERSION = "s7-full-moduli-offset-refine-v1"
OUTPUT_SUFFIX = "s7-full-moduli-offset-refine-v1"
SELECTION_VERSION = "target-local-minima-v1"

with mp.workdps(80):
    ORDER6_CONFIG = SolverConfig(6, 50, 24, mp.mpf("0.7"), 0, scout.DEFAULT_MATCH_T)
    ORDER10_CONFIG = SolverConfig(10, 80, 35, mp.mpf("0.65"), 1, scout.DEFAULT_MATCH_T)
    VERIFY14_CONFIG = SolverConfig(14, 100, 42, mp.mpf("0.55"), 2, scout.DEFAULT_MATCH_T)
    VERIFY18_CONFIG = SolverConfig(18, 110, 45, mp.mpf("0.5"), 2, scout.DEFAULT_MATCH_T)
    DEFAULT_MAX_RESIDUAL = mp.mpf("0.075")
    DEFAULT_MAX_COORDINATE = mp.mpf("1.5")

EVALUATION_CONFIGS = {
    ORDER6_CONFIG.series_order: ORDER6_CONFIG,
    ORDER10_CONFIG.series_order: ORDER10_CONFIG,
    VERIFY14_CONFIG.series_order: VERIFY14_CONFIG,
    VERIFY18_CONFIG.series_order: VERIFY18_CONFIG,
}
VERIFY_CONFIGS = (VERIFY14_CONFIG, VERIFY18_CONFIG)


@dataclass(frozen=True)
class OffsetScoutCandidate:
    """One persisted terminal-offset scout candidate."""

    seed: scout.FullModuliOffsetSeed
    result: scout.FullModuliOffsetResult


@dataclass(frozen=True)
class SelectedOffsetCandidate:
    """One selected scout local minimum."""

    rank: int
    reason: str
    candidate: OffsetScoutCandidate


@dataclass(frozen=True)
class OffsetResidualResult:
    """One terminal-offset matching residual evaluation."""

    target: str
    point: scout.FullModuliOffsetPoint
    config: SolverConfig
    residual: tuple[mp.mpf, ...]
    residual_norm: mp.mpf
    raw_residual_norm: mp.mpf
    germ_residual_norm: mp.mpf
    germ_success: bool
    germ_evaluations: int
    left_l: mp.mpf | None
    right_l: mp.mpf | None
    patch_counts: tuple[int, int]
    failure: str | None = None


@dataclass(frozen=True)
class OffsetNewtonSettings:
    """Numerical settings for one S7 terminal-offset Newton stage."""

    name: str
    config: SolverConfig
    fd_step: mp.mpf
    tolerance: mp.mpf
    max_steps: int
    dampings: tuple[mp.mpf, ...] = field(default_factory=lambda: (mp.one, mp.mpf("0.5"), mp.mpf("0.25"), mp.mpf("0.125"), mp.mpf("0.0625")))
    max_abs_coordinate: mp.mpf | None = None
    min_s_coordinate: mp.mpf | None = None


@dataclass(frozen=True)
class OffsetNewtonStepReport:
    """Diagnostic data for one attempted 7D Gauss-Newton step."""

    index: int
    point_before: scout.FullModuliOffsetPoint
    residual_before: OffsetResidualResult
    delta: tuple[mp.mpf, ...] | None
    damping: mp.mpf | None
    residual_after: OffsetResidualResult
    condition_number: mp.mpf | None
    trial_norms: tuple[tuple[mp.mpf, mp.mpf, bool, str | None], ...]
    status: str


@dataclass(frozen=True)
class OffsetRefinementStageReport:
    """Complete report for one terminal-offset refinement stage."""

    settings: OffsetNewtonSettings
    initial: OffsetResidualResult
    final: OffsetResidualResult
    steps: tuple[OffsetNewtonStepReport, ...]
    status: str


@dataclass(frozen=True)
class OffsetCandidateTrack:
    """One selected S7 offset scout followed through refinement."""

    seed_index: int
    target: str
    seed_point: scout.FullModuliOffsetPoint
    scout_result: scout.FullModuliOffsetResult
    stages: tuple[OffsetRefinementStageReport, ...]
    verifications: tuple[OffsetResidualResult, ...]
    classification: str


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


def _settings_for_max_coordinate(max_coordinate: mp.mpf) -> tuple[OffsetNewtonSettings, OffsetNewtonSettings, OffsetNewtonSettings]:
    """Return the terminal-offset refinement ladder."""
    return (
        OffsetNewtonSettings(
            "order-6-offset-refine",
            ORDER6_CONFIG,
            mp.mpf("1e-3"),
            mp.mpf("1e-8"),
            3,
            max_abs_coordinate=max_coordinate,
            min_s_coordinate=S_MIN,
        ),
        OffsetNewtonSettings(
            "order-10-offset-refine",
            ORDER10_CONFIG,
            mp.mpf("3e-4"),
            mp.mpf("1e-10"),
            3,
            max_abs_coordinate=max_coordinate,
            min_s_coordinate=S_MIN,
        ),
        OffsetNewtonSettings(
            "order-14-offset-correction",
            VERIFY14_CONFIG,
            mp.mpf("1e-4"),
            mp.mpf("1e-12"),
            2,
            max_abs_coordinate=max_coordinate,
            min_s_coordinate=S_MIN,
        ),
    )


def _coordinates(point: scout.FullModuliOffsetPoint) -> tuple[mp.mpf, ...]:
    """Return the seven scaled terminal-offset coordinates."""
    return point.u_left, point.v_left, point.r_left, point.u_right, point.v_right, point.r_right, point.s


def _point_distance(point: scout.FullModuliOffsetPoint) -> mp.mpf:
    """Return max coordinate distance from the known target chart center."""
    return max(abs(value) for value in _coordinates(point))


def _point_with_delta(point: scout.FullModuliOffsetPoint, index: int, delta: mp.mpf) -> scout.FullModuliOffsetPoint:
    """Return one point with one coordinate shifted."""
    values = list(_coordinates(point))
    values[index] += delta
    return scout.FullModuliOffsetPoint(*values)


def _shift_point(point: scout.FullModuliOffsetPoint, delta: tuple[mp.mpf, ...], damping: mp.mpf) -> scout.FullModuliOffsetPoint:
    """Apply one damped 7D Gauss-Newton delta."""
    shifted = point
    for index, value in enumerate(delta):
        shifted = _point_with_delta(shifted, index, damping * value)
    return shifted


def _coordinate_rejection(point: scout.FullModuliOffsetPoint, settings: OffsetNewtonSettings) -> str | None:
    """Return a rejection reason when a trial point violates coordinate guards."""
    if settings.min_s_coordinate is not None and point.s <= settings.min_s_coordinate:
        return "s_bound"
    if settings.max_abs_coordinate is not None and max(abs(value) for value in _coordinates(point)) > settings.max_abs_coordinate:
        return "coordinate_bound"
    return None


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
    """Return whether one JSONL checkpoint contains a specific event."""
    return any(event.get("event") == event_type for event in _iter_jsonl_events(path))


def _latest_completed_scout_jsonl() -> Path:
    """Return the newest completed terminal-offset scout checkpoint."""
    pattern = f"*-seed{RANDOM_SEED}-{scout.SCOUT_VERSION}.jsonl"
    candidates = sorted(scout.OUTPUT_DIR.glob(pattern), reverse=True)
    for path in candidates:
        if _jsonl_has_event(path, "run_summary"):
            return path
    raise FileNotFoundError(f"No completed terminal-offset scout found under {scout.OUTPUT_DIR}")


def _scout_run_start(path: Path) -> dict:
    """Return the run_start event from one scout checkpoint."""
    for event in _iter_jsonl_events(path):
        if event.get("event") == "run_start":
            return event
    raise ValueError(f"{path} does not contain a run_start event")


def _point_from_payload(payload: dict) -> scout.FullModuliOffsetPoint:
    """Rebuild one terminal-offset scout point from persisted JSON strings."""
    return scout.FullModuliOffsetPoint(*(mp.mpf(payload[name]) for name in scout.COORDINATE_NAMES))


def _result_from_payload(payload: dict) -> scout.FullModuliOffsetResult:
    """Rebuild one scout result from a persisted scout_result event."""
    seed = scout.FullModuliOffsetSeed(int(payload["seed_index"]), payload["target"], _point_from_payload(payload["point"]))
    residual = tuple(mp.mpf(value) for value in payload.get("residual", ()))
    failure = payload.get("failure")
    return scout.FullModuliOffsetResult(
        seed=seed,
        residual=residual,
        residual_norm=mp.inf if failure else mp.mpf(payload["residual_norm"]),
        raw_residual_norm=mp.inf if failure else mp.mpf(payload["raw_residual_norm"]),
        germ_residual_norm=mp.inf if payload.get("germ_residual_norm") is None else mp.mpf(payload["germ_residual_norm"]),
        germ_success=bool(payload.get("germ_success")),
        germ_evaluations=int(payload.get("germ_evaluations", 0)),
        left_l=None if payload.get("left_l") is None else mp.mpf(payload["left_l"]),
        right_l=None if payload.get("right_l") is None else mp.mpf(payload["right_l"]),
        patch_counts=tuple(payload.get("patch_counts", (0, 0))),
        failure=failure,
    )


def _load_scout_candidates(path: Path, targets: tuple[str, ...] | None = None) -> list[OffsetScoutCandidate]:
    """Load all terminal-offset scout candidates from one completed JSONL."""
    target_set = None if targets is None else set(targets)
    output = []
    for event in _iter_jsonl_events(path):
        if event.get("event") != "scout_result":
            continue
        if target_set is not None and event.get("target") not in target_set:
            continue
        result = _result_from_payload(event)
        output.append(OffsetScoutCandidate(result.seed, result))
    return output


def _candidate_norm(candidate: OffsetScoutCandidate) -> mp.mpf:
    """Return one candidate residual norm, treating failures as infinity."""
    return candidate.result.residual_norm if candidate.result.failure is None else mp.inf


def _grid_shape(run_start: dict) -> tuple[int, ...]:
    """Return the 7D scout grid shape from run_start metadata."""
    axis_count = int(run_start["axis_count"])
    return tuple(axis_count for _ in scout.COORDINATE_NAMES)


def _grid_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return row-major strides for the persisted grid order."""
    strides = []
    product = 1
    for size in reversed(shape):
        strides.append(product)
        product *= size
    return tuple(reversed(strides))


def _grid_coordinate(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return the grid coordinate tuple for one local seed index."""
    remaining = index
    coordinates = []
    for stride, size in zip(_grid_strides(shape), shape):
        coordinates.append(remaining // stride)
        remaining %= stride
        if coordinates[-1] >= size:
            raise ValueError(f"Seed index {index} is outside grid shape {shape}")
    return tuple(coordinates)


def _grid_index(coordinates: tuple[int, ...], shape: tuple[int, ...]) -> int:
    """Return the local seed index for one grid coordinate tuple."""
    return sum(value * stride for value, stride in zip(coordinates, _grid_strides(shape)))


def _neighbor_indices(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return nearest-neighbor local grid indices for one seed index."""
    coordinates = _grid_coordinate(index, shape)
    output = []
    for dimension, size in enumerate(shape):
        for offset in (-1, 1):
            neighbor = list(coordinates)
            neighbor[dimension] += offset
            if 0 <= neighbor[dimension] < size:
                output.append(_grid_index(tuple(neighbor), shape))
    return tuple(output)


def _local_index(candidate: OffsetScoutCandidate, per_target: int) -> int:
    """Return the seed index local to its target block."""
    return candidate.seed.index % per_target


def _target_local_minima(candidates: list[OffsetScoutCandidate], shape: tuple[int, ...]) -> list[OffsetScoutCandidate]:
    """Return target-wise nearest-neighbor local minima in scout residual."""
    per_target = 1
    for size in shape:
        per_target *= size

    grouped: dict[str, list[OffsetScoutCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.seed.target].append(candidate)

    minima: list[OffsetScoutCandidate] = []
    for group in grouped.values():
        by_local = {_local_index(candidate, per_target): candidate for candidate in group}
        for candidate in group:
            if candidate.result.failure is not None:
                continue
            local = _local_index(candidate, per_target)
            norm = candidate.result.residual_norm
            if all(norm <= _candidate_norm(by_local[neighbor]) for neighbor in _neighbor_indices(local, shape) if neighbor in by_local):
                minima.append(candidate)
    return sorted(minima, key=lambda candidate: (candidate.result.residual_norm, candidate.seed.target, candidate.seed.index))


def _select_local_minima(
    candidates: list[OffsetScoutCandidate],
    run_start: dict,
    max_residual: mp.mpf | None = DEFAULT_MAX_RESIDUAL,
    limit: int | None = None,
) -> list[SelectedOffsetCandidate]:
    """Select target-wise terminal-offset scout local minima."""
    minima = _target_local_minima(candidates, _grid_shape(run_start))
    if max_residual is not None:
        minima = [candidate for candidate in minima if candidate.result.residual_norm < max_residual]
    if limit is not None:
        minima = minima[:limit]
    return [SelectedOffsetCandidate(index + 1, "local-minimum", candidate) for index, candidate in enumerate(minima)]


def _target_params(target: str) -> ProblemParameters:
    """Return known S7 target parameters."""
    return scout._target_params(target)


def _local_config(point: scout.FullModuliOffsetPoint, template_config: SolverConfig) -> SolverConfig:
    """Return the point-specific interval/match config."""
    match_t = template_config.match_t * mp.exp(point.s)
    return config_with_match_t(template_config, match_t)


def _evaluate_raw(target: str, point: scout.FullModuliOffsetPoint, config: SolverConfig) -> OffsetResidualResult:
    """Evaluate one point at a requested finite order without reference subtraction."""
    base = _target_params(target)
    local_config = _local_config(point, config)
    right_point = S7RightGermPoint(point.u_right, point.v_right, point.r_right)
    try:
        params, germ = params_with_right_offset_moduli_germ(
            target=target,
            point=right_point,
            left_params=scout._left_from_point(base, point),
            interval_end=2 * local_config.match_t,
            order=local_config.series_order,
        )
        result = solve_two_sided(params, local_config)
    except (TypeError, ValueError, ZeroDivisionError, RuntimeError) as exc:
        return OffsetResidualResult(target, point, local_config, (), mp.inf, mp.inf, mp.inf, False, 0, None, None, (0, 0), str(exc))

    residual = tuple(result.mismatch_q)
    raw_norm = max(abs(value) for value in residual)
    return OffsetResidualResult(
        target=target,
        point=point,
        config=local_config,
        residual=residual,
        residual_norm=raw_norm,
        raw_residual_norm=raw_norm,
        germ_residual_norm=germ.residual_norm,
        germ_success=germ.success,
        germ_evaluations=germ.evaluations,
        left_l=result.left_l,
        right_l=result.right_l,
        patch_counts=(len(result.left.patches), len(result.right.patches)),
    )


@lru_cache(maxsize=None)
def _reference_residual(target: str, order: int) -> tuple[mp.mpf, ...]:
    """Return the known target raw residual vector for one finite order."""
    zero = scout.FullModuliOffsetPoint(*(mp.zero for _ in scout.COORDINATE_NAMES))
    result = _evaluate_raw(target, zero, EVALUATION_CONFIGS[order])
    if result.failure is not None:
        raise RuntimeError(f"Could not evaluate {target} order-{order} reference: {result.failure}")
    return result.residual


def _calibrated_residual(target: str, point: scout.FullModuliOffsetPoint, config: SolverConfig) -> OffsetResidualResult:
    """Evaluate a point and subtract the known-target finite-order bias."""
    try:
        with mp.workdps(config.working_dps):
            raw = _evaluate_raw(target, point, config)
            if raw.failure is not None:
                return raw
            reference = _reference_residual(target, config.series_order)
            residual = tuple(value - ref for value, ref in zip(raw.residual, reference))
            norm = max(abs(value) for value in residual)
            return OffsetResidualResult(
                target=target,
                point=point,
                config=raw.config,
                residual=residual,
                residual_norm=norm,
                raw_residual_norm=raw.raw_residual_norm,
                germ_residual_norm=raw.germ_residual_norm,
                germ_success=raw.germ_success,
                germ_evaluations=raw.germ_evaluations,
                left_l=raw.left_l,
                right_l=raw.right_l,
                patch_counts=raw.patch_counts,
                failure=None,
            )
    except (TypeError, ValueError, ZeroDivisionError, RuntimeError) as exc:
        local_config = _local_config(point, config)
        return OffsetResidualResult(target, point, local_config, (), mp.inf, mp.inf, mp.inf, False, 0, None, None, (0, 0), str(exc))


def _finite_difference_jacobian(result: OffsetResidualResult, settings: OffsetNewtonSettings) -> tuple[mp.matrix, tuple[mp.mpf, ...], mp.mpf]:
    """Compute the centered finite-difference residual Jacobian in 7D."""
    rows = [[mp.zero for _ in range(len(scout.COORDINATE_NAMES))] for _ in range(len(result.residual))]
    for col in range(len(scout.COORDINATE_NAMES)):
        plus = _calibrated_residual(result.target, _point_with_delta(result.point, col, settings.fd_step), settings.config)
        minus = _calibrated_residual(result.target, _point_with_delta(result.point, col, -settings.fd_step), settings.config)
        if plus.failure or minus.failure:
            raise ValueError(f"Cannot difference failed residuals in column {col}.")
        for row, (left, right) in enumerate(zip(plus.residual, minus.residual)):
            rows[row][col] = (left - right) / (2 * settings.fd_step)
    matrix = mp.matrix(rows)
    _, singulars, _ = mp.svd(matrix)
    positive = [value for value in singulars if value != 0]
    condition = mp.inf if not positive else max(positive) / min(positive)
    return matrix, tuple(singulars), condition


def _newton_delta(result: OffsetResidualResult, settings: OffsetNewtonSettings) -> tuple[tuple[mp.mpf, ...], mp.mpf]:
    """Solve the overdetermined terminal-offset Gauss-Newton system."""
    matrix, _singulars, condition = _finite_difference_jacobian(result, settings)
    rhs = mp.matrix([[-value] for value in result.residual])
    solved, _residual = mp.qr_solve(matrix, rhs)
    return tuple(solved[row] for row in range(solved.rows)), condition


def _try_dampings(
    result: OffsetResidualResult,
    delta: tuple[mp.mpf, ...],
    settings: OffsetNewtonSettings,
) -> tuple[OffsetResidualResult | None, mp.mpf | None, tuple[tuple[mp.mpf, mp.mpf, bool, str | None], ...]]:
    """Return the first damped trial that strictly improves the residual."""
    trials = []
    for damping in settings.dampings:
        point = _shift_point(result.point, delta, damping)
        rejection = _coordinate_rejection(point, settings)
        if rejection is not None:
            trials.append((damping, mp.inf, True, rejection))
            continue
        trial = _calibrated_residual(result.target, point, settings.config)
        trials.append((damping, trial.residual_norm, trial.failure is not None, trial.failure))
        if trial.failure is None and trial.residual_norm < result.residual_norm:
            return trial, damping, tuple(trials)
    return None, None, tuple(trials)


def _failed_step(index: int, result: OffsetResidualResult, status: str) -> OffsetNewtonStepReport:
    """Build a failed step report."""
    return OffsetNewtonStepReport(index, result.point, result, None, None, result, None, (), status)


def _attempt_step(index: int, result: OffsetResidualResult, settings: OffsetNewtonSettings) -> OffsetNewtonStepReport:
    """Attempt one damped Gauss-Newton step."""
    try:
        delta, condition = _newton_delta(result, settings)
    except (TypeError, ValueError, ZeroDivisionError):
        return _failed_step(index, result, "jacobian_failure")
    trial, damping, trials = _try_dampings(result, delta, settings)
    if trial is None:
        return OffsetNewtonStepReport(index, result.point, result, delta, None, result, condition, trials, "no_improvement")
    return OffsetNewtonStepReport(index, result.point, result, delta, damping, trial, condition, trials, "improved")


def _stage_status(result: OffsetResidualResult, settings: OffsetNewtonSettings, steps: list[OffsetNewtonStepReport]) -> str | None:
    """Return a terminal stage status if one has already been reached."""
    if result.failure:
        return "branch_failure"
    if result.residual_norm <= settings.tolerance:
        return "tolerance_hit"
    if steps and steps[-1].status != "improved":
        return steps[-1].status
    return None


def offset_newton_refine(
    target: str,
    point: scout.FullModuliOffsetPoint,
    settings: OffsetNewtonSettings,
) -> OffsetRefinementStageReport:
    """Run one nonfatal damped Gauss-Newton stage."""
    with mp.workdps(settings.config.working_dps):
        current = _calibrated_residual(target, point, settings.config)
        initial = current
        steps: list[OffsetNewtonStepReport] = []
        for index in range(settings.max_steps):
            status = _stage_status(current, settings, steps)
            if status is not None:
                return OffsetRefinementStageReport(settings, initial, current, tuple(steps), status)
            step = _attempt_step(index, current, settings)
            steps.append(step)
            current = step.residual_after
        status = _stage_status(current, settings, steps) or "max_steps"
        return OffsetRefinementStageReport(settings, initial, current, tuple(steps), status)


def _verify_point(target: str, point: scout.FullModuliOffsetPoint) -> tuple[OffsetResidualResult, ...]:
    """Evaluate one point at the high-order verification configs."""
    return tuple(_calibrated_residual(target, point, config) for config in VERIFY_CONFIGS)


def _track_final(track: OffsetCandidateTrack) -> OffsetResidualResult:
    """Return the final residual carried by one track."""
    return track.stages[-1].final if track.stages else _scout_result_as_residual(track.target, track.scout_result)


def _verification_norms(track: OffsetCandidateTrack) -> tuple[mp.mpf, ...]:
    """Return high-order verification norms for one track."""
    return tuple(result.residual_norm for result in track.verifications if result.failure is None)


def _stable_within_factor(norms: tuple[mp.mpf, ...], factor: mp.mpf) -> bool:
    """Return whether verification norms are stable within a multiplicative factor."""
    positive = [norm for norm in norms if norm != 0]
    return len(norms) >= 2 and (not positive or max(positive) <= factor * min(positive))


def _has_failed_stage(track: OffsetCandidateTrack) -> bool:
    """Return whether a refinement stage ended fatally."""
    fatal = {"branch_failure", "jacobian_failure", "no_improvement", "timeout"}
    return any(stage.status in fatal or stage.final.failure for stage in track.stages)


def _classify_track(track: OffsetCandidateTrack) -> str:
    """Classify one S7 terminal-offset refinement track."""
    if track.scout_result.failure or any(result.failure for result in track.verifications):
        return "failed"
    final = _track_final(track)
    norms = _verification_norms(track)
    distance = _point_distance(final.point)
    recovered_label = f"recovered_{track.target}_s7"
    if final.residual_norm < mp.mpf("1e-8") and norms and max(norms) > mp.mpf("1e-4"):
        return "finite_order_artifact"
    if len(norms) == 2 and distance < mp.mpf("1e-3") and max(norms) < mp.mpf("1e-8"):
        return recovered_label
    if len(norms) == 2 and distance >= mp.mpf("0.05"):
        if max(norms) < mp.mpf("1e-8") and _stable_within_factor(norms, mp.mpf("10")):
            return "possible_other_s7_root"
    return "failed" if _has_failed_stage(track) else "inconclusive"


def _deserves_order10(stage: OffsetRefinementStageReport) -> bool:
    """Return whether an order-6 stage deserves order-10 refinement."""
    return stage.final.failure is None and stage.final.residual_norm < stage.initial.residual_norm


def _deserves_order14(stage: OffsetRefinementStageReport) -> bool:
    """Return whether an order-10 attractor deserves order-14 correction."""
    final = stage.final
    return final.failure is None and (final.residual_norm < mp.mpf("1e-6") or _point_distance(final.point) < mp.mpf("0.02"))


def _deserves_verification(stage: OffsetRefinementStageReport) -> bool:
    """Return whether the latest stage deserves high-order verification."""
    final = stage.final
    return final.failure is None and (final.residual_norm < mp.mpf("1e-4") or _point_distance(final.point) < mp.mpf("0.05"))


def _scout_result_as_residual(target: str, result: scout.FullModuliOffsetResult) -> OffsetResidualResult:
    """Convert a scout result into the local residual payload shape."""
    return OffsetResidualResult(
        target=target,
        point=result.seed.point,
        config=scout.SCOUT_CONFIG,
        residual=result.residual,
        residual_norm=result.residual_norm,
        raw_residual_norm=result.raw_residual_norm,
        germ_residual_norm=result.germ_residual_norm,
        germ_success=result.germ_success,
        germ_evaluations=result.germ_evaluations,
        left_l=result.left_l,
        right_l=result.right_l,
        patch_counts=result.patch_counts,
        failure=result.failure,
    )


def _timeout_result(target: str, point: scout.FullModuliOffsetPoint, config: SolverConfig, message: str) -> OffsetResidualResult:
    """Return a failed residual result for timeout diagnostics."""
    local_config = _local_config(point, config)
    return OffsetResidualResult(target, point, local_config, (), mp.inf, mp.inf, mp.inf, False, 0, None, None, (0, 0), message)


def _timeout_stage(target: str, point: scout.FullModuliOffsetPoint, settings: OffsetNewtonSettings, message: str) -> OffsetRefinementStageReport:
    """Return a refinement stage report for a timeout."""
    result = _timeout_result(target, point, settings.config, message)
    return OffsetRefinementStageReport(settings, result, result, (), message)


def _run_order6(selection: SelectedOffsetCandidate, settings: OffsetNewtonSettings, timeout_seconds: float | None) -> OffsetCandidateTrack:
    """Run guarded order-6 refinement for one selection."""
    target = selection.candidate.seed.target
    stage, status = _run_with_timeout(lambda: offset_newton_refine(target, selection.candidate.seed.point, settings), timeout_seconds)
    if stage is None:
        stage = _timeout_stage(target, selection.candidate.seed.point, settings, status or "timeout")
    return OffsetCandidateTrack(
        selection.candidate.seed.index,
        target,
        selection.candidate.seed.point,
        selection.candidate.result,
        (stage,),
        (),
        "inconclusive",
    )


def _promote_core(
    track: OffsetCandidateTrack,
    order10_settings: OffsetNewtonSettings,
    order14_settings: OffsetNewtonSettings,
) -> tuple[tuple[OffsetRefinementStageReport, ...], tuple[OffsetResidualResult, ...]]:
    """Run order-10, optional order-14 correction, and high-order verification."""
    order10 = offset_newton_refine(track.target, track.stages[-1].final.point, order10_settings)
    stages = [order10]
    if _deserves_order14(order10):
        stages.append(offset_newton_refine(track.target, order10.final.point, order14_settings))
    verifications = _verify_point(track.target, stages[-1].final.point) if _deserves_verification(stages[-1]) else ()
    return tuple(stages), verifications


def _promote_track(
    track: OffsetCandidateTrack,
    order10_settings: OffsetNewtonSettings,
    order14_settings: OffsetNewtonSettings,
    timeout_seconds: float | None,
) -> OffsetCandidateTrack:
    """Run guarded promotion, verification, and classification."""
    if not track.stages or not _deserves_order10(track.stages[-1]):
        return OffsetCandidateTrack(track.seed_index, track.target, track.seed_point, track.scout_result, track.stages, (), _classify_track(track))
    promoted, status = _run_with_timeout(lambda: _promote_core(track, order10_settings, order14_settings), timeout_seconds)
    if promoted is None:
        point = track.stages[-1].final.point
        timeout = _timeout_stage(track.target, point, order10_settings, status or "timeout")
        failed = OffsetCandidateTrack(track.seed_index, track.target, track.seed_point, track.scout_result, track.stages + (timeout,), (), "failed")
        return failed
    promoted_stages, verifications = promoted
    candidate = OffsetCandidateTrack(
        track.seed_index,
        track.target,
        track.seed_point,
        track.scout_result,
        track.stages + promoted_stages,
        verifications,
        "inconclusive",
    )
    return OffsetCandidateTrack(
        candidate.seed_index,
        candidate.target,
        candidate.seed_point,
        candidate.scout_result,
        candidate.stages,
        candidate.verifications,
        _classify_track(candidate),
    )


def _point_payload(point: scout.FullModuliOffsetPoint) -> dict[str, str | None]:
    """Return JSON-ready point coordinates."""
    return scout._point_payload(point)


def _physical_payload(target: str, point: scout.FullModuliOffsetPoint) -> dict:
    """Return physical left/right endpoint parameters for one scaled point."""
    base = _target_params(target)
    left = scout._left_from_point(base, point)
    right_offset = offset_moduli_from_point(target, S7RightGermPoint(point.u_right, point.v_right, point.r_right))
    local_config = _local_config(point, ORDER6_CONFIG)
    return {
        "target_chart": target,
        "interval_end": _mp_string(2 * local_config.match_t),
        "left": {
            "a": _mp_string(left.a),
            "c": _mp_string(left.c),
            "alpha": _mp_string(left.alpha),
        },
        "right_offset": {f"q{index}": _mp_string(value) for index, value in enumerate(right_offset, start=1)},
    }


def _residual_payload(result: OffsetResidualResult) -> dict:
    """Return JSON-ready data for one residual evaluation."""
    return {
        "target": result.target,
        "order": result.config.series_order,
        "config_dps": result.config.working_dps,
        "point": _point_payload(result.point),
        "residual": [_mp_string(value) for value in result.residual],
        "residual_norm": _mp_string(result.residual_norm),
        "raw_residual_norm": _mp_string(result.raw_residual_norm),
        "germ_residual_norm": _mp_string(result.germ_residual_norm),
        "germ_success": result.germ_success,
        "germ_evaluations": result.germ_evaluations,
        "left_l": _mp_string(result.left_l),
        "right_l": _mp_string(result.right_l),
        "l_gap": None if result.left_l is None or result.right_l is None else _mp_string(abs(result.left_l - result.right_l)),
        "patch_counts": list(result.patch_counts),
        "failure": result.failure,
    }


def _step_payload(step: OffsetNewtonStepReport) -> dict:
    """Return JSON-ready data for one Newton step."""
    return {
        "index": step.index,
        "point_before": _point_payload(step.point_before),
        "residual_before": _residual_payload(step.residual_before),
        "delta": None if step.delta is None else [_mp_string(value) for value in step.delta],
        "damping": _mp_string(step.damping),
        "condition_number": _mp_string(step.condition_number),
        "trial_norms": [
            {
                "damping": _mp_string(damping),
                "residual_norm": _mp_string(norm),
                "failed": failed,
                "failure": failure,
            }
            for damping, norm, failed, failure in step.trial_norms
        ],
        "residual_after": _residual_payload(step.residual_after),
        "status": step.status,
    }


def _stage_payload(stage: OffsetRefinementStageReport) -> dict:
    """Return JSON-ready data for one refinement stage."""
    return {
        "settings": {
            "name": stage.settings.name,
            "order": stage.settings.config.series_order,
            "dps": stage.settings.config.working_dps,
            "fd_step": _mp_string(stage.settings.fd_step),
            "tolerance": _mp_string(stage.settings.tolerance),
            "max_steps": stage.settings.max_steps,
            "max_abs_coordinate": _mp_string(stage.settings.max_abs_coordinate),
            "min_s_coordinate": _mp_string(stage.settings.min_s_coordinate),
        },
        "initial": _residual_payload(stage.initial),
        "final": _residual_payload(stage.final),
        "steps": [_step_payload(step) for step in stage.steps],
        "status": stage.status,
    }


def _selected_payload(selection: SelectedOffsetCandidate) -> dict:
    """Return JSON-ready data for one selected local minimum."""
    candidate = selection.candidate
    return {
        "rank": selection.rank,
        "reason": selection.reason,
        "seed_index": candidate.seed.index,
        "target": candidate.seed.target,
        "distance": _mp_string(_point_distance(candidate.seed.point)),
        "scout_residual_norm": _mp_string(candidate.result.residual_norm),
        "scout_raw_residual_norm": _mp_string(candidate.result.raw_residual_norm),
        "scout_germ_residual_norm": _mp_string(candidate.result.germ_residual_norm),
        "point": _point_payload(candidate.seed.point),
        "physical": _physical_payload(candidate.seed.target, candidate.seed.point),
    }


def _track_payload(track: OffsetCandidateTrack) -> dict:
    """Return JSON-ready data for one classified track."""
    final = _track_final(track)
    return {
        "seed_index": track.seed_index,
        "target": track.target,
        "classification": track.classification,
        "seed_point": _point_payload(track.seed_point),
        "scout": scout._result_payload(track.scout_result),
        "stages": [_stage_payload(stage) for stage in track.stages],
        "verifications": [_residual_payload(result) for result in track.verifications],
        "verification_norms": [_mp_string(norm) for norm in _verification_norms(track)],
        "final_residual_norm": _mp_string(final.residual_norm),
        "final_point": _point_payload(final.point),
        "distance": _mp_string(_point_distance(final.point)),
        "physical": _physical_payload(track.target, final.point),
    }


def _selection_config_payload(
    max_residual: mp.mpf | None,
    targets: tuple[str, ...] | None,
    limit: int | None,
    order6_timeout: float | None,
    promotion_timeout: float | None,
    max_coordinate: mp.mpf,
) -> dict:
    """Return JSON-ready selection/refinement config."""
    return {
        "selection_version": SELECTION_VERSION,
        "selection_mode": "target-local-minima",
        "local_minimum_max_residual": _mp_string(max_residual),
        "targets": None if targets is None else list(targets),
        "limit": limit,
        "order6_timeout_seconds": order6_timeout,
        "promotion_timeout_seconds": promotion_timeout,
        "max_coordinate": _mp_string(max_coordinate),
        "verify_orders": [config.series_order for config in VERIFY_CONFIGS],
    }


def _settings_payload(settings: tuple[OffsetNewtonSettings, ...]) -> list[dict]:
    """Return JSON-ready stage settings metadata."""
    return [
        {
            "name": item.name,
            "order": item.config.series_order,
            "dps": item.config.working_dps,
            "fd_step": _mp_string(item.fd_step),
            "tolerance": _mp_string(item.tolerance),
            "max_steps": item.max_steps,
            "max_abs_coordinate": _mp_string(item.max_abs_coordinate),
            "min_s_coordinate": _mp_string(item.min_s_coordinate),
        }
        for item in settings
    ]


def _run_start_payload(
    jsonl_path: Path,
    summary_path: Path,
    scout_jsonl: Path,
    selection_config: dict,
    settings: tuple[OffsetNewtonSettings, ...],
) -> dict:
    """Return checkpoint metadata for one refinement run."""
    return {
        "random_seed": RANDOM_SEED,
        "refinement_version": REFINEMENT_VERSION,
        "scout_jsonl": str(scout_jsonl),
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "selection_config": selection_config,
        "settings": _settings_payload(settings),
    }


def _output_refinement_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped output paths for terminal-offset refinement."""
    return _output_paths(OUTPUT_DIR, OUTPUT_SUFFIX, now)


def _summary_path_for_jsonl(path: Path) -> Path:
    """Return the summary path paired with one refinement JSONL."""
    return path.with_name(f"{path.stem}-summary.json")


def _checkpoint_is_compatible(path: Path, scout_jsonl: Path, selection_config: dict, settings: tuple[OffsetNewtonSettings, ...]) -> bool:
    """Return whether an incomplete checkpoint can be resumed."""
    if _jsonl_has_event(path, "run_summary"):
        return False
    starts = [event for event in _iter_jsonl_events(path) if event.get("event") == "run_start"]
    if not starts:
        return False
    expected = _run_start_payload(path, _summary_path_for_jsonl(path), scout_jsonl, selection_config, settings)
    return all(starts[-1].get(key) == value for key, value in expected.items() if key not in {"jsonl_path", "summary_path"})


def _latest_incomplete_checkpoint(
    scout_jsonl: Path,
    selection_config: dict,
    settings: tuple[OffsetNewtonSettings, ...],
) -> Path | None:
    """Return the newest compatible incomplete refinement checkpoint."""
    pattern = f"*-seed{RANDOM_SEED}-{OUTPUT_SUFFIX}.jsonl"
    candidates = sorted(OUTPUT_DIR.glob(pattern), reverse=True)
    return next((path for path in candidates if _checkpoint_is_compatible(path, scout_jsonl, selection_config, settings)), None)


def _resume_or_new_paths(
    scout_jsonl: Path,
    selection_config: dict,
    settings: tuple[OffsetNewtonSettings, ...],
    *,
    resume: bool = True,
    now: datetime | None = None,
) -> tuple[Path, Path, bool]:
    """Return refinement paths, resuming a compatible incomplete checkpoint if possible."""
    if resume and now is None:
        checkpoint = _latest_incomplete_checkpoint(scout_jsonl, selection_config, settings)
        if checkpoint is not None:
            return checkpoint, _summary_path_for_jsonl(checkpoint), True
    jsonl_path, summary_path = _output_refinement_paths(now)
    return jsonl_path, summary_path, False


def _selected_seed_indices(path: Path) -> set[int]:
    """Return seed indices already persisted as candidate_selected events."""
    return {int(event["seed_index"]) for event in _iter_jsonl_events(path) if event.get("event") == "candidate_selected"}


def _classified_seed_indices(path: Path) -> set[int]:
    """Return seed indices already classified in one checkpoint."""
    return {int(event["seed_index"]) for event in _iter_jsonl_events(path) if event.get("event") == "candidate_classification"}


def _classified_payloads(path: Path) -> list[dict]:
    """Return classification payloads from one checkpoint without event metadata."""
    payloads = []
    for event in _iter_jsonl_events(path):
        if event.get("event") == "candidate_classification":
            payloads.append({key: value for key, value in event.items() if key not in {"event", "time_utc"}})
    return payloads


def _write_missing_selections(path: Path, selections: list[SelectedOffsetCandidate]) -> None:
    """Persist selected candidates not already present in a resumed checkpoint."""
    existing = _selected_seed_indices(path)
    for selection in selections:
        if selection.candidate.seed.index not in existing:
            _write_jsonl_event(path, _event("candidate_selected", _selected_payload(selection)))


def _best_tracks(tracks: list[OffsetCandidateTrack], limit: int = 20) -> list[dict]:
    """Return compact best-track summaries."""
    def sort_key(track: OffsetCandidateTrack):
        norms = _verification_norms(track)
        final = _track_final(track)
        return (max(norms) if norms else mp.inf, final.residual_norm, track.seed_index)

    return [_track_payload(track) for track in sorted(tracks, key=sort_key)[:limit]]


def _best_track_payloads(track_payloads: list[dict], limit: int = 20) -> list[dict]:
    """Return compact best-track summaries from serialized payloads."""
    def sort_key(payload: dict):
        norms = tuple(mp.mpf(value) for value in payload.get("verification_norms", ()))
        final = mp.mpf(payload.get("final_residual_norm", "inf"))
        return (max(norms) if norms else mp.inf, final, int(payload["seed_index"]))

    return sorted(track_payloads, key=sort_key)[:limit]


def _summary_payload(
    scout_jsonl: Path,
    scout_start: dict,
    selections: list[SelectedOffsetCandidate],
    tracks: list[OffsetCandidateTrack],
    selection_config: dict,
) -> dict:
    """Return JSON-ready final refinement summary."""
    counts = Counter(track.classification for track in tracks)
    return {
        "random_seed": RANDOM_SEED,
        "refinement_version": REFINEMENT_VERSION,
        "scout_jsonl": str(scout_jsonl),
        "scout_axis_count": scout_start["axis_count"],
        "selection_config": selection_config,
        "selection_count": len(selections),
        "classified_count": len(tracks),
        "classification_counts": dict(counts),
        "selections": [_selected_payload(selection) for selection in selections],
        "best_tracks": _best_tracks(tracks),
        "tracks": [_track_payload(track) for track in tracks],
    }


def _summary_payload_from_track_payloads(
    scout_jsonl: Path,
    scout_start: dict,
    selections: list[SelectedOffsetCandidate],
    track_payloads: list[dict],
    selection_config: dict,
) -> dict:
    """Return final summary data from serialized classification payloads."""
    counts = Counter(payload["classification"] for payload in track_payloads)
    return {
        "random_seed": RANDOM_SEED,
        "refinement_version": REFINEMENT_VERSION,
        "scout_jsonl": str(scout_jsonl),
        "scout_axis_count": scout_start["axis_count"],
        "selection_config": selection_config,
        "selection_count": len(selections),
        "classified_count": len(track_payloads),
        "classification_counts": dict(counts),
        "selections": [_selected_payload(selection) for selection in selections],
        "best_tracks": _best_track_payloads(track_payloads),
        "tracks": track_payloads,
    }


def _parse_targets(value: str) -> tuple[str, ...]:
    """Parse a comma-separated target list."""
    targets = tuple(item.strip() for item in value.split(",") if item.strip())
    if not targets:
        raise argparse.ArgumentTypeError("must include at least one target")
    for target in targets:
        _target_params(target)
    return targets


def _optional_positive_mpf(value: str) -> mp.mpf | None:
    """Parse a positive mpmath value, or `none`/`all` for no cutoff."""
    if value.lower() in {"none", "all"}:
        return None
    parsed = mp.mpf(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive, or 'none'")
    return parsed


def _positive_mpf(value: str) -> mp.mpf:
    """Parse a positive mpmath decimal CLI argument."""
    parsed = mp.mpf(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    """Parse a positive float CLI argument."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI argument."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scout-jsonl", type=Path, default=None)
    parser.add_argument("--max-residual", type=_optional_positive_mpf, default=DEFAULT_MAX_RESIDUAL)
    parser.add_argument("--targets", type=_parse_targets, default=None)
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--order6-timeout", type=_positive_float, default=20 * 60)
    parser.add_argument("--promotion-timeout", type=_positive_float, default=40 * 60)
    parser.add_argument("--max-coordinate", type=_positive_mpf, default=DEFAULT_MAX_COORDINATE)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def _print_selection(selection: SelectedOffsetCandidate) -> None:
    """Print one selected candidate."""
    point = selection.candidate.seed.point
    values = ", ".join(f"{name}={mp.nstr(value, 8)}" for name, value in zip(scout.COORDINATE_NAMES, _coordinates(point)))
    print(
        f"  {selection.rank:02d}: seed={selection.candidate.seed.index} target={selection.candidate.seed.target} "
        f"scout={mp.nstr(selection.candidate.result.residual_norm, 12)} {values}",
        flush=True,
    )


def _resolve_scout_jsonl(path: Path | None) -> Path:
    """Return the requested or newest completed terminal-offset scout checkpoint."""
    return path if path is not None else _latest_completed_scout_jsonl()


def main(argv: list[str] | None = None) -> None:
    """Run terminal-offset local-minimum refinement."""
    args = _parse_args(argv)
    scout_jsonl = _resolve_scout_jsonl(args.scout_jsonl)
    scout_start = _scout_run_start(scout_jsonl)
    candidates = _load_scout_candidates(scout_jsonl, args.targets)
    selections = _select_local_minima(candidates, scout_start, args.max_residual, args.limit)

    print("S7 full-moduli terminal-offset local-minimum refinement", flush=True)
    print(f"version: {REFINEMENT_VERSION}", flush=True)
    print(f"scout: {scout_jsonl}", flush=True)
    print(f"selected local minima: {len(selections)}", flush=True)
    for selection in selections:
        _print_selection(selection)
    if args.dry_run:
        return

    settings = _settings_for_max_coordinate(args.max_coordinate)
    order6_settings, order10_settings, order14_settings = settings
    selection_config = _selection_config_payload(
        args.max_residual,
        args.targets,
        args.limit,
        args.order6_timeout,
        args.promotion_timeout,
        args.max_coordinate,
    )
    jsonl_path, summary_path, resumed = _resume_or_new_paths(
        scout_jsonl,
        selection_config,
        settings,
        resume=not args.no_resume,
    )
    if resumed:
        print(f"resuming refinement checkpoint {jsonl_path}", flush=True)
    else:
        print(f"writing refinement JSONL to {jsonl_path}", flush=True)
        _write_jsonl_event(jsonl_path, _event("run_start", _run_start_payload(jsonl_path, summary_path, scout_jsonl, selection_config, settings)))
    _write_missing_selections(jsonl_path, selections)

    completed = _classified_seed_indices(jsonl_path)
    for index, selection in enumerate(selections, start=1):
        if selection.candidate.seed.index in completed:
            print(f"seed {selection.candidate.seed.index}: reused completed classification", flush=True)
            continue
        track = _run_order6(selection, order6_settings, args.order6_timeout)
        for stage in track.stages:
            _write_jsonl_event(
                jsonl_path,
                _event(
                    "refinement_stage",
                    {
                        "selection_rank": selection.rank,
                        "selection_reason": selection.reason,
                        "seed_index": selection.candidate.seed.index,
                        "target": selection.candidate.seed.target,
                        "stage": _stage_payload(stage),
                    },
                ),
            )
        classified = _promote_track(track, order10_settings, order14_settings, args.promotion_timeout)
        for stage in classified.stages[len(track.stages) :]:
            _write_jsonl_event(
                jsonl_path,
                _event(
                    "refinement_stage",
                    {
                        "selection_rank": selection.rank,
                        "selection_reason": selection.reason,
                        "seed_index": selection.candidate.seed.index,
                        "target": selection.candidate.seed.target,
                        "stage": _stage_payload(stage),
                    },
                ),
        )
        _write_jsonl_event(jsonl_path, _event("candidate_classification", _track_payload(classified)))
        final = _track_final(classified)
        print(
            f"classified {index}/{len(selections)} seed={classified.seed_index} target={classified.target} "
            f"class={classified.classification} final={mp.nstr(final.residual_norm, 12)} "
            f"distance={mp.nstr(_point_distance(final.point), 8)}",
            flush=True,
        )

    track_payloads = _classified_payloads(jsonl_path)
    summary = _summary_payload_from_track_payloads(scout_jsonl, scout_start, selections, track_payloads, selection_config)
    _write_jsonl_event(jsonl_path, _event("run_summary", summary))
    _write_summary(summary_path, summary)
    print(f"wrote {jsonl_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    print(f"classifications: {summary['classification_counts']}", flush=True)


if __name__ == "__main__":
    main()
