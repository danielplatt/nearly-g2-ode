"""Shared recovery and scout helpers for fixed-chart S7 searches."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING, localcontext
from datetime import datetime
from itertools import product
from pathlib import Path
from random import Random
from typing import Callable, Iterable

from mpmath import mp

from problem import (
    DEFAULT_CONFIG,
    LeftEndpointParameters,
    ProblemParameters,
    SolverConfig,
    round_s7_candidate_parameters,
    squashed_s7_parameters,
)
from solver.march import solve_two_sided
from solver.two_sided_shooting import config_with_match_t

from ..shared.non_mirrored_common import RANDOM_SEED, _event, _mp_string, _output_paths, _write_jsonl_event, _write_summary


COORDINATE_NAMES = ("u", "v", "r")
POINT_PAYLOAD_NAMES = ("u", "v", "r", "s")
RECOVERY_OUTPUT_DIR = Path("output/s7_recovery_calibration")
SCOUT_OUTPUT_DIR = Path("output/s7_scout_searches")
RECOVERY_VERSION = "s7-recovery-v1"
SCOUT_VERSION = "s7-scout-v1"
RECOVERY_SUFFIX_TEMPLATE = "s7-{target}-recovery-v1"
SCOUT_SUFFIX = "s7-scout-v1"
TARGET_NAMES = ("round", "squashed")

with mp.workdps(80):
    MIN_MATCH_T = mp.mpf("0.01")
    S_MIN = mp.log(MIN_MATCH_T / DEFAULT_CONFIG.match_t)
    DEFAULT_SCOUT_SPACING = mp.mpf("0.075")
    SCOUT_STEP_SAFETY = mp.mpf("0.95")

SCOUT_CONFIG = SolverConfig(6, 40, 20, SCOUT_STEP_SAFETY, 0, DEFAULT_CONFIG.match_t)
ORDER8_CONFIG = SolverConfig(8, 50, 24, mp.mpf("0.9"), 0, DEFAULT_CONFIG.match_t)
ORDER10_CONFIG = SolverConfig(10, 70, 30, mp.mpf("0.7"), 1, DEFAULT_CONFIG.match_t)
VERIFY14_CONFIG = SolverConfig(14, 90, 35, mp.mpf("0.6"), 2, DEFAULT_CONFIG.match_t)
VERIFY18_CONFIG = SolverConfig(18, 110, 45, mp.mpf("0.55"), 2, DEFAULT_CONFIG.match_t)
VERIFY_CONFIGS = (VERIFY14_CONFIG, VERIFY18_CONFIG)
REFERENCE_CONFIGS = (SCOUT_CONFIG, ORDER8_CONFIG, ORDER10_CONFIG, VERIFY14_CONFIG, VERIFY18_CONFIG)

RECOVERY_SHELL_RADII = ("1e-3",)
RECOVERY_RANDOM_SHELL_SAMPLES = 0
RECOVERY_LOCAL_BOX_RADIUS = mp.mpf("0.1")
RECOVERY_LOCAL_BOX_SAMPLES = 0
MAX_RECOVERY_COORDINATE = mp.mpf("1.5")

ORDER8_SETTINGS = None
ORDER10_SETTINGS = None
ORDER14_SETTINGS = None


@dataclass(frozen=True)
class S7Target:
    """One fixed right-chart S7 target."""

    name: str
    recovered_label: str
    params_builder: Callable[[], ProblemParameters]


@dataclass(frozen=True)
class S7SearchPoint:
    """Scaled coordinates for fixed-chart S7 searches."""

    u: mp.mpf
    v: mp.mpf
    r: mp.mpf
    s: mp.mpf


@dataclass(frozen=True)
class S7SearchSeed:
    """One deterministic S7 scout/recovery seed."""

    index: int
    target: str
    region: str
    source: str
    point: S7SearchPoint


@dataclass(frozen=True)
class S7ResidualResult:
    """One fixed-chart S7 matching residual evaluation."""

    point: S7SearchPoint
    params: ProblemParameters
    config: SolverConfig
    residual: tuple[mp.mpf, ...]
    residual_norm: mp.mpf
    left_l: mp.mpf | None
    right_l: mp.mpf | None
    patch_counts: tuple[int, int]
    branch_diagnostics: dict[str, mp.mpf]
    failure: str | None = None


@dataclass(frozen=True)
class S7NewtonSettings:
    """Numerical settings for one fixed-chart S7 Newton stage."""

    name: str
    config: SolverConfig
    fd_step: mp.mpf
    tolerance: mp.mpf
    max_steps: int
    dampings: tuple[mp.mpf, ...] = field(default_factory=lambda: (mp.one, mp.mpf("0.5"), mp.mpf("0.25"), mp.mpf("0.125"), mp.mpf("0.0625")))
    max_abs_coordinate: mp.mpf | None = None
    min_s_coordinate: mp.mpf | None = None


@dataclass(frozen=True)
class S7NewtonStepReport:
    """Diagnostic data for one attempted S7 Gauss-Newton step."""

    index: int
    point_before: S7SearchPoint
    residual_before: S7ResidualResult
    delta: tuple[mp.mpf, ...] | None
    damping: mp.mpf | None
    residual_after: S7ResidualResult
    condition_number: mp.mpf | None
    trial_norms: tuple[tuple[mp.mpf, mp.mpf, bool, str | None], ...]
    status: str


@dataclass(frozen=True)
class S7RefinementStageReport:
    """Complete report for one fixed-chart S7 refinement stage."""

    settings: S7NewtonSettings
    initial: S7ResidualResult
    final: S7ResidualResult
    steps: tuple[S7NewtonStepReport, ...]
    status: str


@dataclass(frozen=True)
class S7CandidateTrack:
    """One S7 scout seed followed through refinement."""

    seed_index: int
    target: str
    region: str
    seed_point: S7SearchPoint
    scout_result: S7ResidualResult
    stages: tuple[S7RefinementStageReport, ...]
    verifications: tuple[S7ResidualResult, ...]
    classification: str


@dataclass(frozen=True)
class S7ScoutCandidate:
    """One evaluated S7 scout seed."""

    seed: S7SearchSeed
    result: S7ResidualResult


@dataclass(frozen=True)
class S7GridRegion:
    """One rectangular S7 scout grid in 3D scaled coordinates."""

    name: str
    bounds: tuple[tuple[mp.mpf, mp.mpf], ...]
    coordinate_names: tuple[str, ...] = COORDINATE_NAMES
    parameterization: str = "default"


TARGETS = {
    "round": S7Target("round", "recovered_round_s7", round_s7_candidate_parameters),
    "squashed": S7Target("squashed", "recovered_squashed_s7", squashed_s7_parameters),
}

with mp.workdps(80):
    POSITIVE_AC_ALPHA_SCALE = mp.sqrt(5) / 50
    DEFAULT_SCOUT_REGION = S7GridRegion(
        "default",
        (
            (mp.mpf("-1.2"), mp.mpf("1.2")),
            (mp.mpf("-1.2"), mp.mpf("1.2")),
            (mp.mpf("-2.5"), mp.mpf("2.5")),
        ),
        COORDINATE_NAMES,
        "a=a0*exp(u), c=c0*exp(v), alpha=alpha0*(1+r)",
    )
    POSITIVE_AC_SCOUT_REGION = S7GridRegion(
        "positive-ac",
        (
            (mp.mpf("-1.2"), mp.mpf("1.2")),
            (mp.mpf("0.05"), mp.mpf("0.4")),
            (mp.mpf("-3.5"), mp.mpf("3.5")),
        ),
        ("u", "rho", "r"),
        "a=a0*exp(u), c=3*a*rho with rho in [0.05,0.4], alpha=(sqrt(5)/50)*r",
    )

SCOUT_REGIONS = {
    DEFAULT_SCOUT_REGION.name: DEFAULT_SCOUT_REGION,
    POSITIVE_AC_SCOUT_REGION.name: POSITIVE_AC_SCOUT_REGION,
}


def _target(name: str) -> S7Target:
    """Return one known S7 target descriptor."""
    try:
        return TARGETS[name]
    except KeyError as exc:
        known = ", ".join(TARGET_NAMES)
        raise ValueError(f"Unknown S7 target {name!r}; choose one of: {known}") from exc


def _scout_region(name: str) -> S7GridRegion:
    """Return one named S7 scout region."""
    try:
        return SCOUT_REGIONS[name]
    except KeyError as exc:
        known = ", ".join(sorted(SCOUT_REGIONS))
        raise ValueError(f"Unknown S7 scout region {name!r}; choose one of: {known}") from exc


def _parameter_region_for_seed(seed: S7SearchSeed) -> str:
    """Return the parameterization region for one seed."""
    return seed.region if seed.region in SCOUT_REGIONS else DEFAULT_SCOUT_REGION.name


def _coordinates(point: S7SearchPoint) -> tuple[mp.mpf, ...]:
    """Return active scaled S7 search coordinates as a tuple."""
    return point.u, point.v, point.r


def _point_from_values(values) -> S7SearchPoint:
    """Build one S7 search point from numeric values."""
    parsed = tuple(mp.mpf(value) for value in values)
    if len(parsed) == 3:
        return S7SearchPoint(*parsed, mp.zero)
    if len(parsed) == 4:
        return S7SearchPoint(*parsed)
    raise ValueError("S7 search points need 3 active coordinates, optionally followed by fixed s.")


def _point_with_delta(point: S7SearchPoint, index: int, delta: mp.mpf) -> S7SearchPoint:
    """Return one scaled point with one coordinate shifted."""
    values = list(_coordinates(point))
    values[index] += delta
    return S7SearchPoint(*values, point.s)


def _point_distance(point: S7SearchPoint) -> mp.mpf:
    """Return max-distance from the target base point."""
    return max(abs(value) for value in _coordinates(point))


def _point_payload(point: S7SearchPoint) -> dict[str, str | None]:
    """Return JSON-ready scaled point coordinates."""
    values = (point.u, point.v, point.r, point.s)
    return {name: _mp_string(value) for name, value in zip(POINT_PAYLOAD_NAMES, values)}


def _params_payload(params: ProblemParameters) -> dict:
    """Return JSON-ready target parameters."""
    return {
        "lambda": _mp_string(params.lam),
        "interval_end": _mp_string(params.interval_end),
        "right_chart": params.right_chart,
        "fixed_right_label": None if params.fixed_right is None else params.fixed_right.label,
        "left": {
            "a": _mp_string(params.left.a),
            "c": _mp_string(params.left.c),
            "alpha": _mp_string(params.left.alpha),
        },
    }


def params_from_s7_scaled(
    point: S7SearchPoint,
    *,
    base_params: ProblemParameters,
    template_config: SolverConfig,
    region: str = DEFAULT_SCOUT_REGION.name,
) -> tuple[ProblemParameters, SolverConfig]:
    """Convert scaled S7 coordinates into left data and a matching config."""
    match_t = template_config.match_t
    if region == DEFAULT_SCOUT_REGION.name:
        left = LeftEndpointParameters(
            a=base_params.left.a * mp.exp(point.u),
            c=base_params.left.c * mp.exp(point.v),
            alpha=base_params.left.alpha * (1 + point.r),
        )
    elif region == POSITIVE_AC_SCOUT_REGION.name:
        a = abs(base_params.left.a) * mp.exp(point.u)
        rho = point.v
        left = LeftEndpointParameters(
            a=a,
            c=3 * a * rho,
            alpha=POSITIVE_AC_ALPHA_SCALE * point.r,
        )
    else:
        raise ValueError(f"Unknown S7 parameter region {region!r}.")
    params = ProblemParameters(
        lam=base_params.lam,
        interval_end=2 * match_t,
        left=left,
        right=base_params.right,
        right_chart=base_params.right_chart,
        fixed_right=base_params.fixed_right,
    )
    return params, config_with_match_t(template_config, match_t)


def _failure_result(point: S7SearchPoint, config: SolverConfig, params: ProblemParameters, message: str) -> S7ResidualResult:
    """Build a nonfatal failed S7 residual result."""
    return S7ResidualResult(point, params, config, (), mp.inf, None, None, (0, 0), {}, message)


def s7_residual(
    point: S7SearchPoint,
    config: SolverConfig,
    *,
    base_params: ProblemParameters,
    region: str = DEFAULT_SCOUT_REGION.name,
) -> S7ResidualResult:
    """Evaluate the fixed-chart S7 raw q mismatch."""
    params, local_config = params_from_s7_scaled(point, base_params=base_params, template_config=config, region=region)
    try:
        result = solve_two_sided(params, local_config)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return _failure_result(point, local_config, params, str(exc))
    residual = tuple(result.mismatch_q)
    norm = max(abs(value) for value in residual)
    diagnostics = {}
    for side_name, side in (("left", result.left), ("right", result.right)):
        for key in ("min_sum27", "min_sum36", "max_gap", "min_product"):
            diagnostics[f"{side_name}_{key}"] = side.diagnostics[key]
    return S7ResidualResult(
        point=point,
        params=params,
        config=local_config,
        residual=residual,
        residual_norm=norm,
        left_l=result.left_l,
        right_l=result.right_l,
        patch_counts=(len(result.left.patches), len(result.right.patches)),
        branch_diagnostics=diagnostics,
    )


def _calibrated_result(result: S7ResidualResult, reference: S7ResidualResult | None) -> S7ResidualResult:
    """Subtract one known-target finite-order residual vector, if supplied."""
    if reference is None or result.failure:
        return result
    residual = tuple(value - ref for value, ref in zip(result.residual, reference.residual))
    norm = max(abs(value) for value in residual)
    return S7ResidualResult(
        point=result.point,
        params=result.params,
        config=result.config,
        residual=residual,
        residual_norm=norm,
        left_l=result.left_l,
        right_l=result.right_l,
        patch_counts=result.patch_counts,
        branch_diagnostics=result.branch_diagnostics,
        failure=result.failure,
    )


def s7_calibrated_residual(
    point: S7SearchPoint,
    config: SolverConfig,
    *,
    base_params: ProblemParameters,
    reference: S7ResidualResult | None,
    region: str = DEFAULT_SCOUT_REGION.name,
) -> S7ResidualResult:
    """Evaluate the S7 residual after subtracting a known-target finite-order bias."""
    return _calibrated_result(s7_residual(point, config, base_params=base_params, region=region), reference)


def _result_payload(result: S7ResidualResult) -> dict:
    """Return a compact JSON-ready S7 residual result."""
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
        "interval_end": _mp_string(result.params.interval_end),
        "match_t": _mp_string(result.config.match_t),
        "left": {
            "a": _mp_string(result.params.left.a),
            "c": _mp_string(result.params.left.c),
            "alpha": _mp_string(result.params.left.alpha),
        },
    }


def _candidate_payload(candidate: S7ScoutCandidate) -> dict:
    """Return JSON-ready S7 scout candidate data."""
    return {
        "seed_index": candidate.seed.index,
        "target": candidate.seed.target,
        "region": candidate.seed.region,
        "source": candidate.seed.source,
        "distance": _mp_string(_point_distance(candidate.seed.point)),
        "seed_point": _point_payload(candidate.seed.point),
        "result": _result_payload(candidate.result),
    }


def _stage_payload(stage: S7RefinementStageReport) -> dict:
    """Return JSON-ready refinement stage diagnostics."""
    return {
        "name": stage.settings.name,
        "status": stage.status,
        "initial": _result_payload(stage.initial),
        "final": _result_payload(stage.final),
        "steps": [
            {
                "index": step.index,
                "status": step.status,
                "point_before": _point_payload(step.point_before),
                "delta": None if step.delta is None else [_mp_string(value) for value in step.delta],
                "damping": _mp_string(step.damping),
                "condition_number": _mp_string(step.condition_number),
                "residual_before": _mp_string(step.residual_before.residual_norm),
                "residual_after": _mp_string(step.residual_after.residual_norm),
                "trial_norms": [
                    [_mp_string(damping), _mp_string(norm), failed, failure]
                    for damping, norm, failed, failure in step.trial_norms
                ],
            }
            for step in stage.steps
        ],
    }


def _track_payload(track: S7CandidateTrack) -> dict:
    """Return JSON-ready S7 candidate-track data."""
    return {
        "seed_index": track.seed_index,
        "target": track.target,
        "region": track.region,
        "seed_point": _point_payload(track.seed_point),
        "scout": _result_payload(track.scout_result),
        "stages": [_stage_payload(stage) for stage in track.stages],
        "verifications": [_result_payload(result) for result in track.verifications],
        "verification_norms": [_mp_string(result.residual_norm) for result in track.verifications],
        "classification": track.classification,
    }


def _coordinate_rejection(point: S7SearchPoint, settings: S7NewtonSettings) -> str | None:
    """Return the reason one trial point violates a coordinate guard."""
    if settings.min_s_coordinate is not None and point.s <= settings.min_s_coordinate:
        return "m_floor_rejected"
    if settings.max_abs_coordinate is None:
        return None
    if max(abs(value) for value in _coordinates(point)) > settings.max_abs_coordinate:
        return "coordinate_bound"
    return None


def _finite_difference_jacobian(
    point: S7SearchPoint,
    settings: S7NewtonSettings,
    base_params: ProblemParameters,
    reference: S7ResidualResult | None = None,
    region: str = DEFAULT_SCOUT_REGION.name,
) -> tuple[mp.matrix, tuple[mp.mpf, ...], mp.mpf]:
    """Compute the centered finite-difference S7 residual Jacobian."""
    rows = [[mp.zero for _ in range(3)] for _ in range(8)]
    for col in range(3):
        plus = s7_calibrated_residual(
            _point_with_delta(point, col, settings.fd_step),
            settings.config,
            base_params=base_params,
            reference=reference,
            region=region,
        )
        minus = s7_calibrated_residual(
            _point_with_delta(point, col, -settings.fd_step),
            settings.config,
            base_params=base_params,
            reference=reference,
            region=region,
        )
        if plus.failure or minus.failure:
            raise ValueError(f"Cannot difference failed residuals in column {col}.")
        for row, (left, right) in enumerate(zip(plus.residual, minus.residual)):
            rows[row][col] = (left - right) / (2 * settings.fd_step)
    matrix = mp.matrix(rows)
    _, singulars, _ = mp.svd(matrix)
    positive = [value for value in singulars if value != 0]
    condition = mp.inf if not positive else max(positive) / min(positive)
    return matrix, tuple(singulars), condition


def _newton_delta(
    result: S7ResidualResult,
    settings: S7NewtonSettings,
    base_params: ProblemParameters,
    reference: S7ResidualResult | None = None,
    region: str = DEFAULT_SCOUT_REGION.name,
) -> tuple[tuple[mp.mpf, ...], mp.mpf]:
    """Solve the overdetermined fixed-chart S7 Gauss-Newton system."""
    matrix, _singulars, condition = _finite_difference_jacobian(result.point, settings, base_params, reference, region)
    rhs = mp.matrix([[-value] for value in result.residual])
    solved, _residual = mp.qr_solve(matrix, rhs)
    return tuple(solved[row] for row in range(solved.rows)), condition


def _shift_point(point: S7SearchPoint, delta: tuple[mp.mpf, ...], damping: mp.mpf) -> S7SearchPoint:
    """Apply one damped S7 Gauss-Newton delta."""
    shifted = point
    for index, value in enumerate(delta):
        shifted = _point_with_delta(shifted, index, damping * value)
    return shifted


def _try_dampings(
    result: S7ResidualResult,
    delta: tuple[mp.mpf, ...],
    settings: S7NewtonSettings,
    base_params: ProblemParameters,
    reference: S7ResidualResult | None = None,
    region: str = DEFAULT_SCOUT_REGION.name,
) -> tuple[S7ResidualResult | None, mp.mpf | None, tuple[tuple[mp.mpf, mp.mpf, bool, str | None], ...]]:
    """Return the first damped trial that strictly improves the residual."""
    trials = []
    for damping in settings.dampings:
        point = _shift_point(result.point, delta, damping)
        rejection = _coordinate_rejection(point, settings)
        if rejection is not None:
            trials.append((damping, mp.inf, True, rejection))
            continue
        trial = s7_calibrated_residual(point, settings.config, base_params=base_params, reference=reference, region=region)
        trials.append((damping, trial.residual_norm, trial.failure is not None, trial.failure))
        if trial.failure is None and trial.residual_norm < result.residual_norm:
            return trial, damping, tuple(trials)
    return None, None, tuple(trials)


def _failed_step(index: int, result: S7ResidualResult, status: str) -> S7NewtonStepReport:
    """Build a failed S7 Newton step report."""
    return S7NewtonStepReport(index, result.point, result, None, None, result, None, (), status)


def _attempt_step(
    index: int,
    result: S7ResidualResult,
    settings: S7NewtonSettings,
    base_params: ProblemParameters,
    reference: S7ResidualResult | None = None,
    region: str = DEFAULT_SCOUT_REGION.name,
) -> S7NewtonStepReport:
    """Attempt one damped S7 Gauss-Newton step."""
    try:
        delta, condition = _newton_delta(result, settings, base_params, reference, region)
    except (TypeError, ValueError, ZeroDivisionError):
        return _failed_step(index, result, "jacobian_failure")
    trial, damping, trials = _try_dampings(result, delta, settings, base_params, reference, region)
    if trial is None:
        return S7NewtonStepReport(index, result.point, result, delta, None, result, condition, trials, "no_improvement")
    return S7NewtonStepReport(index, result.point, result, delta, damping, trial, condition, trials, "improved")


def _stage_status(result: S7ResidualResult, settings: S7NewtonSettings, steps: list[S7NewtonStepReport]) -> str | None:
    """Return a terminal stage status if one has already been reached."""
    if result.failure:
        return "branch_failure"
    if result.residual_norm <= settings.tolerance:
        return "tolerance_hit"
    if steps and steps[-1].status != "improved":
        return steps[-1].status
    return None


def s7_newton_refine(
    point: S7SearchPoint,
    settings: S7NewtonSettings,
    *,
    base_params: ProblemParameters,
    reference: S7ResidualResult | None = None,
    region: str = DEFAULT_SCOUT_REGION.name,
) -> S7RefinementStageReport:
    """Run one nonfatal damped Gauss-Newton stage for S7."""
    with mp.workdps(settings.config.working_dps):
        current = s7_calibrated_residual(point, settings.config, base_params=base_params, reference=reference, region=region)
        initial = current
        steps: list[S7NewtonStepReport] = []
        for index in range(settings.max_steps):
            status = _stage_status(current, settings, steps)
            if status is not None:
                return S7RefinementStageReport(settings, initial, current, tuple(steps), status)
            step = _attempt_step(index, current, settings, base_params, reference, region)
            steps.append(step)
            current = step.residual_after
        status = _stage_status(current, settings, steps) or "max_steps"
        return S7RefinementStageReport(settings, initial, current, tuple(steps), status)


def _newton_settings() -> tuple[S7NewtonSettings, S7NewtonSettings, S7NewtonSettings]:
    """Return the calibrated S7 recovery ladder."""
    return (
        S7NewtonSettings(
            "order-8-s7-recovery",
            ORDER8_CONFIG,
            mp.mpf("1e-3"),
            mp.mpf("1e-8"),
            3,
            max_abs_coordinate=MAX_RECOVERY_COORDINATE,
            min_s_coordinate=S_MIN,
        ),
        S7NewtonSettings(
            "order-10-s7-recovery",
            ORDER10_CONFIG,
            mp.mpf("3e-4"),
            mp.mpf("1e-10"),
            3,
            max_abs_coordinate=MAX_RECOVERY_COORDINATE,
            min_s_coordinate=S_MIN,
        ),
        S7NewtonSettings(
            "order-14-s7-correction",
            VERIFY14_CONFIG,
            mp.mpf("1e-4"),
            mp.mpf("1e-12"),
            2,
            max_abs_coordinate=MAX_RECOVERY_COORDINATE,
            min_s_coordinate=S_MIN,
        ),
    )


def _evaluate_seed(seed: S7SearchSeed, config: SolverConfig = SCOUT_CONFIG) -> S7ScoutCandidate:
    """Evaluate one cheap S7 scout/recovery seed."""
    target = _target(seed.target)
    with mp.workdps(config.working_dps):
        return S7ScoutCandidate(
            seed,
            s7_residual(
                seed.point,
                config,
                base_params=target.params_builder(),
                region=_parameter_region_for_seed(seed),
            ),
        )


def _evaluate_recovery_seed(seed: S7SearchSeed, reference: S7ResidualResult) -> S7ScoutCandidate:
    """Evaluate one recovery seed using a reference-subtracted scout residual."""
    target = _target(seed.target)
    with mp.workdps(SCOUT_CONFIG.working_dps):
        return S7ScoutCandidate(
            seed,
            s7_calibrated_residual(
                seed.point,
                SCOUT_CONFIG,
                base_params=target.params_builder(),
                reference=reference,
                region=_parameter_region_for_seed(seed),
            ),
        )


def _verify_point(
    point: S7SearchPoint,
    base_params: ProblemParameters,
    region: str = DEFAULT_SCOUT_REGION.name,
) -> tuple[S7ResidualResult, ...]:
    """Evaluate one point at high-order S7 verification configs."""
    output = []
    for config in VERIFY_CONFIGS:
        with mp.workdps(config.working_dps):
            output.append(s7_residual(point, config, base_params=base_params, region=region))
    return tuple(output)


def _reference_residuals(target: S7Target) -> tuple[S7ResidualResult, ...]:
    """Return known-target residuals at all recovery orders."""
    base_params = target.params_builder()
    point = S7SearchPoint(mp.zero, mp.zero, mp.zero, mp.zero)
    output = []
    for config in REFERENCE_CONFIGS:
        with mp.workdps(config.working_dps):
            output.append(s7_residual(point, config, base_params=base_params))
    return tuple(output)


def _track_final(track: S7CandidateTrack) -> S7ResidualResult:
    """Return the final residual carried by one S7 track."""
    return track.stages[-1].final if track.stages else track.scout_result


def _verification_norms(track: S7CandidateTrack) -> tuple[mp.mpf, ...]:
    """Return high-order verification norms for one S7 track."""
    return tuple(result.residual_norm for result in track.verifications)


def _verification_thresholds(references: tuple[S7ResidualResult, ...]) -> tuple[mp.mpf, ...]:
    """Return target-relative order-14/order-18 recovery thresholds."""
    return tuple(max(mp.mpf("1e-8"), mp.mpf("1000") * result.residual_norm) for result in references)


def _stable_within_factor(norms: tuple[mp.mpf, ...], factor: mp.mpf) -> bool:
    """Return whether verification norms are stable within a multiplicative factor."""
    positive = [norm for norm in norms if norm != 0]
    return len(norms) >= 2 and (not positive or max(positive) <= factor * min(positive))


def _has_failed_stage(track: S7CandidateTrack) -> bool:
    """Return whether a refinement stage ended fatally."""
    fatal = {"branch_failure", "jacobian_failure", "no_improvement"}
    return any(stage.status in fatal or stage.final.failure for stage in track.stages)


def _deserves_order10(stage: S7RefinementStageReport) -> bool:
    """Return whether an order-8 stage deserves order-10 refinement."""
    return stage.final.failure is None and stage.final.residual_norm < stage.initial.residual_norm


def _deserves_order14(stage: S7RefinementStageReport) -> bool:
    """Return whether a low-order attractor deserves order-14 correction."""
    final = stage.final
    return final.failure is None and (final.residual_norm < mp.mpf("1e-6") or _point_distance(final.point) < mp.mpf("0.02"))


def _deserves_verification(stage: S7RefinementStageReport) -> bool:
    """Return whether a refined S7 point deserves high-order verification."""
    final = stage.final
    return final.failure is None and (final.residual_norm < mp.mpf("1e-4") or _point_distance(final.point) < mp.mpf("0.05"))


def _classify_track(track: S7CandidateTrack, target: S7Target, references: tuple[S7ResidualResult, ...]) -> str:
    """Classify one S7 recovery track."""
    if track.scout_result.failure or any(result.failure for result in track.verifications):
        return "failed"
    final = _track_final(track)
    norms = _verification_norms(track)
    if final.residual_norm < mp.mpf("1e-8") and norms and max(norms) > mp.mpf("1e-4"):
        return "finite_order_artifact"
    if len(norms) == 2 and _point_distance(final.point) < mp.mpf("1e-3"):
        if all(norm <= threshold for norm, threshold in zip(norms, _verification_thresholds(references))):
            return target.recovered_label
    if len(norms) == 2 and _point_distance(final.point) >= mp.mpf("0.05"):
        if max(norms) < mp.mpf("1e-8") and _stable_within_factor(norms, mp.mpf("10")):
            return "possible_other_s7_root"
    return "failed" if _has_failed_stage(track) else "inconclusive"


def _run_recovery_track(seed: S7SearchSeed, target: S7Target, references: tuple[S7ResidualResult, ...]) -> S7CandidateTrack:
    """Run scout, order-8, order-10, and verification for one S7 seed."""
    scout = _evaluate_recovery_seed(seed, references[0])
    if scout.result.failure:
        return S7CandidateTrack(seed.index, seed.target, seed.region, seed.point, scout.result, (), (), "failed")
    order8_settings, order10_settings, order14_settings = _newton_settings()
    base_params = target.params_builder()
    parameter_region = _parameter_region_for_seed(seed)
    order8 = s7_newton_refine(seed.point, order8_settings, base_params=base_params, reference=references[1], region=parameter_region)
    stages = [order8]
    if order8.final.residual_norm <= order8_settings.tolerance and _point_distance(order8.final.point) < mp.mpf("1e-3"):
        verifications = _verify_point(order8.final.point, base_params, parameter_region)
        track = S7CandidateTrack(seed.index, seed.target, seed.region, seed.point, scout.result, tuple(stages), verifications, "inconclusive")
        return S7CandidateTrack(seed.index, seed.target, seed.region, seed.point, scout.result, tuple(stages), verifications, _classify_track(track, target, references[-2:]))
    if order8.final.failure or order8.status in {"jacobian_failure", "no_improvement", "branch_failure"} or not _deserves_order10(order8):
        track = S7CandidateTrack(seed.index, seed.target, seed.region, seed.point, scout.result, tuple(stages), (), "inconclusive")
        return S7CandidateTrack(seed.index, seed.target, seed.region, seed.point, scout.result, tuple(stages), (), _classify_track(track, target, references[-2:]))
    order10 = s7_newton_refine(order8.final.point, order10_settings, base_params=base_params, reference=references[2], region=parameter_region)
    stages.append(order10)
    if order10.final.residual_norm <= order10_settings.tolerance and _point_distance(order10.final.point) < mp.mpf("1e-3"):
        verifications = _verify_point(order10.final.point, base_params, parameter_region)
        track = S7CandidateTrack(seed.index, seed.target, seed.region, seed.point, scout.result, tuple(stages), verifications, "inconclusive")
        return S7CandidateTrack(seed.index, seed.target, seed.region, seed.point, scout.result, tuple(stages), verifications, _classify_track(track, target, references[-2:]))
    if _deserves_order14(order10):
        stages.append(s7_newton_refine(order10.final.point, order14_settings, base_params=base_params, reference=references[3], region=parameter_region))
    verifications = _verify_point(stages[-1].final.point, base_params, parameter_region) if _deserves_verification(stages[-1]) else ()
    track = S7CandidateTrack(seed.index, seed.target, seed.region, seed.point, scout.result, tuple(stages), verifications, "inconclusive")
    return S7CandidateTrack(seed.index, seed.target, seed.region, seed.point, scout.result, tuple(stages), verifications, _classify_track(track, target, references[-2:]))


def _axis_shell_seeds(target_name: str, radius: mp.mpf, start_index: int) -> list[S7SearchSeed]:
    """Return coordinate-axis recovery seeds at one max-norm radius."""
    seeds = []
    for coordinate in range(3):
        for sign in (-1, 1):
            values = [mp.zero for _ in range(3)]
            values[coordinate] = sign * radius
            seeds.append(S7SearchSeed(start_index + len(seeds), target_name, f"shell_{mp.nstr(radius, 8)}", "axis", _point_from_values(values)))
    return seeds


def _random_shell_point(radius: mp.mpf, rng: Random) -> S7SearchPoint:
    """Return one reproducible random point on a 3D max-norm shell."""
    values = [mp.mpf(rng.uniform(-float(radius), float(radius))) for _ in range(3)]
    face = rng.randrange(3)
    values[face] = radius if rng.randrange(2) else -radius
    return _point_from_values(values)


def _shell_seeds(target_name: str, radius: mp.mpf, start_index: int, rng: Random) -> list[S7SearchSeed]:
    """Return axis and random shell recovery seeds for one radius."""
    seeds = _axis_shell_seeds(target_name, radius, start_index)
    for _ in range(RECOVERY_RANDOM_SHELL_SAMPLES):
        seeds.append(S7SearchSeed(start_index + len(seeds), target_name, f"shell_{mp.nstr(radius, 8)}", "random_shell", _random_shell_point(radius, rng)))
    return seeds


def _local_box_seeds(target_name: str, start_index: int, rng: Random) -> list[S7SearchSeed]:
    """Return blind local-box seeds around one known S7 solution."""
    seeds = []
    radius = float(RECOVERY_LOCAL_BOX_RADIUS)
    for index in range(RECOVERY_LOCAL_BOX_SAMPLES):
        values = [rng.uniform(-radius, radius) for _ in range(3)]
        seeds.append(S7SearchSeed(start_index + index, target_name, "local_box", "random_box", _point_from_values(values)))
    return seeds


def recovery_seeds(target_name: str, seed: int = RANDOM_SEED) -> list[S7SearchSeed]:
    """Return all deterministic shell/local-box recovery seeds for one S7 target."""
    _target(target_name)
    rng = Random(seed)
    seeds: list[S7SearchSeed] = []
    for radius_text in RECOVERY_SHELL_RADII:
        seeds.extend(_shell_seeds(target_name, mp.mpf(radius_text), len(seeds), rng))
    seeds.extend(_local_box_seeds(target_name, len(seeds), rng))
    return seeds


def _jsonl_events(path: Path):
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
    return any(event.get("event") == event_type for event in _jsonl_events(path))


def _completed_recovery_seed_indices(path: Path) -> set[int]:
    """Return recovery seed indices already classified in one checkpoint."""
    return {int(event["seed_index"]) for event in _jsonl_events(path) if event.get("event") == "candidate_classification"}


def _classified_payloads(path: Path) -> list[dict]:
    """Return all classified S7 recovery tracks from a checkpoint."""
    return [event for event in _jsonl_events(path) if event.get("event") == "candidate_classification"]


def _recovery_output_paths(target_name: str, now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped output paths for one S7 recovery calibration."""
    return _output_paths(RECOVERY_OUTPUT_DIR, RECOVERY_SUFFIX_TEMPLATE.format(target=target_name), now)


def _recovery_run_start_payload(jsonl_path: Path, summary_path: Path, target: S7Target) -> dict:
    """Return checkpoint metadata for one S7 recovery run."""
    return {
        "random_seed": RANDOM_SEED,
        "recovery_version": RECOVERY_VERSION,
        "target": target.name,
        "target_params": _params_payload(target.params_builder()),
        "coordinate_names": list(COORDINATE_NAMES),
        "shell_radii": list(RECOVERY_SHELL_RADII),
        "random_shell_samples": RECOVERY_RANDOM_SHELL_SAMPLES,
        "local_box_radius": _mp_string(RECOVERY_LOCAL_BOX_RADIUS),
        "local_box_samples": RECOVERY_LOCAL_BOX_SAMPLES,
        "max_recovery_coordinate": _mp_string(MAX_RECOVERY_COORDINATE),
        "s_min": _mp_string(S_MIN),
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
    }


def _shell_counts(payloads: list[dict]) -> dict[str, dict[str, int]]:
    """Return classification counts by shell/local-box region."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for payload in payloads:
        counts[payload["region"]][payload["classification"]] += 1
    return {region: dict(counter) for region, counter in counts.items()}


def _largest_recovery_radius(target: S7Target, counts: dict[str, dict[str, int]], fraction: mp.mpf) -> str | None:
    """Return the largest shell radius whose recovery fraction exceeds a threshold."""
    largest = None
    for radius_text in RECOVERY_SHELL_RADII:
        region = f"shell_{mp.nstr(mp.mpf(radius_text), 8)}"
        region_counts = counts.get(region, {})
        total = sum(region_counts.values())
        recovered = region_counts.get(target.recovered_label, 0)
        if total and mp.mpf(recovered) / total >= fraction:
            largest = radius_text
    return largest


def _recovery_summary_payload(path: Path, target: S7Target, references: tuple[S7ResidualResult, ...], seeds: list[S7SearchSeed]) -> dict:
    """Return JSON-ready final S7 recovery summary."""
    payloads = _classified_payloads(path)
    counts = Counter(payload["classification"] for payload in payloads)
    shell_counts = _shell_counts(payloads)
    return {
        "target": target.name,
        "recovered_label": target.recovered_label,
        "reference_residuals": [_result_payload(result) for result in references],
        "seed_count": len(seeds),
        "classified_count": len(payloads),
        "classification_counts": dict(counts),
        "shell_counts": shell_counts,
        "largest_any_recovery_radius": _largest_recovery_radius(target, shell_counts, mp.mpf("1e-30")),
        "largest_eighty_percent_recovery_radius": _largest_recovery_radius(target, shell_counts, mp.mpf("0.8")),
        "local_box_recovered": shell_counts.get("local_box", {}).get(target.recovered_label, 0) > 0,
        "tracks": payloads,
    }


def _print_references(references: tuple[S7ResidualResult, ...]) -> None:
    """Print known-target reference residuals for the recovery configs."""
    for result in references:
        print(f"reference order {result.config.series_order}: norm={mp.nstr(result.residual_norm, 12)} failure={result.failure}", flush=True)


def _print_recovery_summary(payload: dict) -> None:
    """Print a compact human-readable recovery summary."""
    print(f"classified: {payload['classified_count']}/{payload['seed_count']}", flush=True)
    print(f"classifications: {payload['classification_counts']}", flush=True)
    print(f"largest radius with any recovery: {payload['largest_any_recovery_radius']}", flush=True)
    print(f"largest radius with >=80% recovery: {payload['largest_eighty_percent_recovery_radius']}", flush=True)
    print(f"local box recovered target: {payload['local_box_recovered']}", flush=True)


def main_recovery(target_name: str) -> None:
    """Run one fixed-chart S7 recovery calibration."""
    target = _target(target_name)
    jsonl_path, summary_path = _recovery_output_paths(target.name)
    print(f"writing JSONL events to {jsonl_path}", flush=True)
    _write_jsonl_event(jsonl_path, _event("run_start", _recovery_run_start_payload(jsonl_path, summary_path, target)))
    references = _reference_residuals(target)
    _print_references(references)
    seeds = recovery_seeds(target.name)
    completed = _completed_recovery_seed_indices(jsonl_path)
    for index, seed in enumerate(seeds, start=1):
        if seed.index in completed:
            continue
        track = _run_recovery_track(seed, target, references)
        _write_jsonl_event(jsonl_path, _event("candidate_classification", _track_payload(track)))
        print(f"seed {index}/{len(seeds)} ({seed.region}, {seed.source}): {track.classification}", flush=True)
    payload = _recovery_summary_payload(jsonl_path, target, references, seeds)
    _print_recovery_summary(payload)
    _write_jsonl_event(jsonl_path, _event("run_summary", payload))
    _write_summary(summary_path, payload)
    print(f"summary written to {summary_path}", flush=True)


def _axis_values(low: mp.mpf, high: mp.mpf, max_spacing: mp.mpf) -> tuple[mp.mpf, ...]:
    """Return inclusive axis values with mesh width no larger than max_spacing."""
    with localcontext() as context:
        context.prec = 100
        low_decimal = Decimal(mp.nstr(low, 80))
        high_decimal = Decimal(mp.nstr(high, 80))
        spacing_decimal = Decimal(mp.nstr(max_spacing, 80))
        if spacing_decimal <= 0:
            raise ValueError("max_spacing must be positive")
        if high_decimal < low_decimal:
            raise ValueError("axis upper bound must be at least the lower bound")
        intervals = max(1, int(((high_decimal - low_decimal) / spacing_decimal).to_integral_value(rounding=ROUND_CEILING)))
        step = (high_decimal - low_decimal) / Decimal(intervals)
        values = [low_decimal + step * index for index in range(intervals + 1)]
        values[-1] = high_decimal
    with mp.workdps(max(mp.dps, 80)):
        return tuple(mp.mpf(str(value)) for value in values)


def scout_axes(spacing: mp.mpf = DEFAULT_SCOUT_SPACING, region: str = DEFAULT_SCOUT_REGION.name) -> tuple[tuple[mp.mpf, ...], ...]:
    """Return the S7 scout grid axes for one named region."""
    scout_region = _scout_region(region)
    return tuple(_axis_values(low, high, spacing) for low, high in scout_region.bounds)


def scout_seed_count(
    targets: tuple[str, ...] = TARGET_NAMES,
    spacing: mp.mpf = DEFAULT_SCOUT_SPACING,
    limit: int | None = None,
    region: str = DEFAULT_SCOUT_REGION.name,
) -> int:
    """Return the full S7 scout seed count before any debugging limit."""
    count = 1
    for axis in scout_axes(spacing, region):
        count *= len(axis)
    count *= len(targets)
    return min(count, limit) if limit is not None else count


def scout_grid_metadata(
    targets: tuple[str, ...] = TARGET_NAMES,
    spacing: mp.mpf = DEFAULT_SCOUT_SPACING,
    limit: int | None = None,
    region: str = DEFAULT_SCOUT_REGION.name,
) -> dict:
    """Return JSON-ready metadata for the S7 scout grid."""
    scout_region = _scout_region(region)
    axes = scout_axes(spacing, region)
    full_per_target = 1
    for axis in axes:
        full_per_target *= len(axis)
    full_seed_count = full_per_target * len(targets)
    seed_count = min(full_seed_count, limit) if limit is not None else full_seed_count
    return {
        "region": scout_region.name,
        "targets": list(targets),
        "coordinate_names": list(scout_region.coordinate_names),
        "bounds": [[_mp_string(low), _mp_string(high)] for low, high in scout_region.bounds],
        "parameterization": scout_region.parameterization,
        "max_grid_spacing": _mp_string(spacing),
        "axis_counts": [len(axis) for axis in axes],
        "full_per_target": full_per_target,
        "full_seed_count": full_seed_count,
        "seed_count": seed_count,
        "limit": limit,
    }


def scout_seeds(
    targets: tuple[str, ...] = TARGET_NAMES,
    spacing: mp.mpf = DEFAULT_SCOUT_SPACING,
    limit: int | None = None,
    region: str = DEFAULT_SCOUT_REGION.name,
) -> list[S7SearchSeed]:
    """Return deterministic S7 scout grid seeds."""
    scout_region = _scout_region(region)
    axes = scout_axes(spacing, scout_region.name)
    seeds = []
    for target_name in targets:
        _target(target_name)
        for values in product(*axes):
            if limit is not None and len(seeds) >= limit:
                return seeds
            seeds.append(S7SearchSeed(len(seeds), target_name, scout_region.name, "s7_grid", _point_from_values(values)))
    return seeds


def _completed_scout_seed_indices(path: Path) -> set[int]:
    """Return scout seed indices already persisted in one checkpoint."""
    return {int(event["seed_index"]) for event in _jsonl_events(path) if event.get("event") == "scout_result"}


def _scout_payloads(path: Path) -> list[dict]:
    """Return all completed S7 scout payloads from a checkpoint."""
    return [event for event in _jsonl_events(path) if event.get("event") == "scout_result"]


def _evaluate_scout_seed_payload(seed: S7SearchSeed) -> dict:
    """Evaluate one S7 scout seed and return a JSON-ready payload."""
    return _candidate_payload(_evaluate_seed(seed, SCOUT_CONFIG))


def _evaluate_scout_seed_payloads(seeds: list[S7SearchSeed], workers: int, chunksize: int | None = None) -> Iterable[dict]:
    """Yield JSON-ready S7 scout payloads, optionally in parallel."""
    if workers <= 1:
        for seed in seeds:
            yield _evaluate_scout_seed_payload(seed)
        return
    actual_chunksize = chunksize or 8
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(_evaluate_scout_seed_payload, seeds, chunksize=actual_chunksize)


def _run_scouts(
    seeds: list[S7SearchSeed],
    jsonl_path: Path,
    workers: int,
    *,
    progress_every: int = 1000,
    chunksize: int | None = None,
) -> None:
    """Evaluate missing S7 scout seeds and append results to the checkpoint."""
    completed = _completed_scout_seed_indices(jsonl_path)
    pending = [seed for seed in seeds if seed.index not in completed]
    total = len(seeds)
    print(f"loaded completed scouts: {len(completed)}/{total}", flush=True)
    print(f"pending scouts: {len(pending)} with workers={workers}", flush=True)
    done = len(completed)
    for payload in _evaluate_scout_seed_payloads(pending, workers, chunksize):
        _write_jsonl_event(jsonl_path, _event("scout_result", payload))
        done += 1
        if progress_every and (done == total or done % progress_every == 0):
            print(f"scouts complete: {done}/{total}", flush=True)


def _payload_success(payload: dict) -> bool:
    """Return whether one S7 scout payload is branch-valid."""
    return payload["result"]["failure"] is None


def _payload_norm(payload: dict) -> mp.mpf:
    """Return the residual norm for sorting one scout payload."""
    return mp.mpf(payload["result"]["residual_norm"])


def _compact_scout_payload(payload: dict) -> dict:
    """Return a compact scout summary payload."""
    return {
        "seed_index": payload["seed_index"],
        "target": payload["target"],
        "region": payload["region"],
        "source": payload["source"],
        "distance": payload["distance"],
        "residual_norm": payload["result"]["residual_norm"],
        "failure": payload["result"]["failure"],
        "seed_point": payload["seed_point"],
        "physical": {
            "interval_end": payload["result"]["interval_end"],
            "left": payload["result"]["left"],
        },
    }


def _scout_summary(payloads: list[dict]) -> dict[str, dict[str, int]]:
    """Return scout success/failure counts by target."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for payload in payloads:
        counts[payload["target"]]["total"] += 1
        counts[payload["target"]]["successes" if _payload_success(payload) else "failures"] += 1
    return {target: dict(counter) for target, counter in sorted(counts.items())}


def scout_summary_payload(jsonl_path: Path, metadata: dict, best_limit: int = 20) -> dict:
    """Return a compact JSON-ready S7 scout summary."""
    payloads = _scout_payloads(jsonl_path)
    successes = [payload for payload in payloads if _payload_success(payload)]
    failures = [payload for payload in payloads if not _payload_success(payload)]
    best = sorted(successes, key=_payload_norm)[:best_limit]
    return {
        "random_seed": RANDOM_SEED,
        "scout_version": SCOUT_VERSION,
        "grid": metadata["grid"],
        "scout_config": metadata["scout_config"],
        "scout_count": len(payloads),
        "scout_summary": _scout_summary(payloads),
        "classification_counts": {"scout_success": len(successes), "scout_failure": len(failures)},
        "failure_messages": dict(Counter(payload["result"]["failure"] for payload in failures)),
        "best_scouts": [_compact_scout_payload(payload) for payload in best],
    }


def _scout_output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped output paths for the S7 scout run."""
    return _output_paths(SCOUT_OUTPUT_DIR, SCOUT_SUFFIX, now)


def _summary_path_for_scout_jsonl(path: Path) -> Path:
    """Return the summary path paired with one S7 scout checkpoint."""
    return path.with_name(f"{path.stem}-summary.json")


def _scout_config_payload() -> dict:
    """Return JSON-ready S7 scout solver settings."""
    return {
        "series_order": SCOUT_CONFIG.series_order,
        "working_dps": SCOUT_CONFIG.working_dps,
        "target_dps": SCOUT_CONFIG.target_dps,
        "step_safety": _mp_string(SCOUT_CONFIG.step_safety),
        "sample_points": SCOUT_CONFIG.sample_points,
        "match_t": _mp_string(SCOUT_CONFIG.match_t),
    }


def _scout_run_start_payload(
    jsonl_path: Path,
    summary_path: Path,
    targets: tuple[str, ...],
    spacing: mp.mpf,
    limit: int | None,
    region: str = DEFAULT_SCOUT_REGION.name,
) -> dict:
    """Return checkpoint metadata identifying one S7 scout run."""
    return {
        "random_seed": RANDOM_SEED,
        "scout_version": SCOUT_VERSION,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "target_params": {name: _params_payload(_target(name).params_builder()) for name in targets},
        "grid": scout_grid_metadata(targets, spacing, limit, region),
        "scout_config": _scout_config_payload(),
    }


def _scout_checkpoint_is_compatible(
    path: Path,
    targets: tuple[str, ...],
    spacing: mp.mpf,
    limit: int | None,
    region: str = DEFAULT_SCOUT_REGION.name,
) -> bool:
    """Return whether one incomplete S7 scout checkpoint can be resumed."""
    if _jsonl_has_event(path, "run_summary"):
        return False
    starts = [event for event in _jsonl_events(path) if event.get("event") == "run_start"]
    if not starts:
        return False
    expected = _scout_run_start_payload(path, _summary_path_for_scout_jsonl(path), targets, spacing, limit, region)
    ignored = {"jsonl_path", "summary_path"}
    return all(starts[-1].get(key) == value for key, value in expected.items() if key not in ignored)


def _latest_incomplete_scout_checkpoint(
    targets: tuple[str, ...],
    spacing: mp.mpf,
    limit: int | None,
    region: str = DEFAULT_SCOUT_REGION.name,
) -> Path | None:
    """Return the newest compatible incomplete S7 scout checkpoint, if present."""
    candidates = sorted(SCOUT_OUTPUT_DIR.glob(f"*-seed{RANDOM_SEED}-{SCOUT_SUFFIX}.jsonl"), reverse=True)
    return next((path for path in candidates if _scout_checkpoint_is_compatible(path, targets, spacing, limit, region)), None)


def _resume_or_new_scout_paths(
    targets: tuple[str, ...],
    spacing: mp.mpf,
    limit: int | None,
    region: str = DEFAULT_SCOUT_REGION.name,
    *,
    resume: bool = True,
    now: datetime | None = None,
) -> tuple[Path, Path, bool]:
    """Return S7 scout output paths, resuming a compatible incomplete checkpoint when possible."""
    if resume and now is None:
        checkpoint = _latest_incomplete_scout_checkpoint(targets, spacing, limit, region)
        if checkpoint is not None:
            return checkpoint, _summary_path_for_scout_jsonl(checkpoint), True
    jsonl_path, summary_path = _scout_output_paths(now)
    return jsonl_path, summary_path, False


def _default_workers() -> int:
    """Return a conservative default process count for scout evaluation."""
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count - 1))


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI argument."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    """Parse a nonnegative integer CLI argument."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_mpf(value: str) -> mp.mpf:
    """Parse a positive mpmath CLI argument."""
    parsed = mp.mpf(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parse_targets(value: str) -> tuple[str, ...]:
    """Parse one comma-separated target list."""
    targets = tuple(item.strip() for item in value.split(",") if item.strip())
    if not targets:
        raise argparse.ArgumentTypeError("must include at least one target")
    for name in targets:
        _target(name)
    return targets


def _parse_region(value: str) -> str:
    """Parse one named S7 scout region."""
    _scout_region(value)
    return value


def _print_scout_dry_run(targets: tuple[str, ...], spacing: mp.mpf, limit: int | None, region: str) -> None:
    """Print S7 grid size and rough runtime estimates without creating files."""
    metadata = scout_grid_metadata(targets, spacing, limit, region)
    count = metadata["seed_count"]
    print(f"region: {metadata['region']}", flush=True)
    print(f"targets: {','.join(targets)}", flush=True)
    print(f"parameterization: {metadata['parameterization']}", flush=True)
    print(f"max spacing: {mp.nstr(spacing, 12)}", flush=True)
    print(f"axis counts: {metadata['axis_counts']}", flush=True)
    print(f"scout points: {count}", flush=True)
    print(f"serial scout estimate at 1.2-2.0 s/point: {count * 1.2 / 3600:.2f}-{count * 2.0 / 3600:.2f} h", flush=True)


def main_scout(argv: list[str] | None = None) -> None:
    """Run the fixed-chart S7 long scout grid."""
    parser = argparse.ArgumentParser(description="Long fixed-chart S7 scout grid around round and squashed targets.")
    parser.add_argument("--targets", type=_parse_targets, default=TARGET_NAMES, help="comma-separated targets: round,squashed")
    parser.add_argument("--region", type=_parse_region, default=DEFAULT_SCOUT_REGION.name, help="scout region: default or positive-ac")
    parser.add_argument("--workers", type=_positive_int, default=_default_workers(), help="parallel worker processes")
    parser.add_argument("--spacing", type=_positive_mpf, default=DEFAULT_SCOUT_SPACING, help="maximum grid spacing")
    parser.add_argument("--limit", type=_nonnegative_int, default=None, help="debugging cap on generated seeds")
    parser.add_argument("--dry-run", action="store_true", help="print grid size and runtime estimate without running")
    parser.add_argument("--no-resume", action="store_true", help="start a fresh checkpoint instead of resuming")
    parser.add_argument("--progress-every", type=_positive_int, default=1000, help="print progress every N completed scouts")
    parser.add_argument("--chunksize", type=_positive_int, default=None, help="multiprocessing map chunk size")
    args = parser.parse_args(argv)
    if args.dry_run:
        _print_scout_dry_run(args.targets, args.spacing, args.limit, args.region)
        return

    jsonl_path, summary_path, resumed = _resume_or_new_scout_paths(args.targets, args.spacing, args.limit, args.region, resume=not args.no_resume)
    metadata = _scout_run_start_payload(jsonl_path, summary_path, args.targets, args.spacing, args.limit, args.region)
    if resumed:
        print(f"resuming JSONL checkpoint {jsonl_path}", flush=True)
    else:
        print(f"writing JSONL events to {jsonl_path}", flush=True)
        _write_jsonl_event(jsonl_path, _event("run_start", metadata))
    seeds = scout_seeds(args.targets, args.spacing, args.limit, args.region)
    print(f"S7 scout seeds: {len(seeds)}", flush=True)
    _run_scouts(seeds, jsonl_path, args.workers, progress_every=args.progress_every, chunksize=args.chunksize)
    payload = scout_summary_payload(jsonl_path, metadata)
    print(f"scout summary: {payload['scout_summary']}", flush=True)
    if payload["best_scouts"]:
        best = payload["best_scouts"][0]
        print(f"best scout: target={best['target']} seed={best['seed_index']} norm={best['residual_norm']}", flush=True)
    _write_jsonl_event(jsonl_path, _event("run_summary", payload))
    _write_summary(summary_path, payload)
    print(f"summary written to {summary_path}", flush=True)
