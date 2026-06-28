"""Damped Gauss-Newton refinement for full two-sided matching."""

from __future__ import annotations

from dataclasses import dataclass, field

from mpmath import mp

from problem import DEFAULT_PARAMS, ProblemParameters, SolverConfig
from .two_sided_shooting import (
    TwoSidedResidualResult,
    TwoSidedSearchPoint,
    finite_difference_two_sided_jacobian,
    point_with_delta,
    two_sided_residual,
)


def _default_dampings() -> tuple[mp.mpf, ...]:
    """Return the fixed damped Gauss-Newton backtracking schedule."""
    return (mp.one, mp.mpf("0.5"), mp.mpf("0.25"), mp.mpf("0.125"), mp.mpf("0.0625"))


@dataclass(frozen=True)
class TwoSidedNewtonSettings:
    """Numerical settings for one non-mirrored refinement stage."""

    name: str
    config: SolverConfig
    fd_step: mp.mpf
    tolerance: mp.mpf
    max_steps: int
    dampings: tuple[mp.mpf, ...] = field(default_factory=_default_dampings)
    max_abs_coordinate: mp.mpf | None = None
    min_s_coordinate: mp.mpf | None = None


@dataclass(frozen=True)
class TwoSidedNewtonStepReport:
    """Diagnostic data for one attempted Gauss-Newton step."""

    index: int
    point_before: TwoSidedSearchPoint
    residual_before: TwoSidedResidualResult
    delta: tuple[mp.mpf, ...] | None
    damping: mp.mpf | None
    residual_after: TwoSidedResidualResult
    condition_number: mp.mpf | None
    trial_norms: tuple[tuple[mp.mpf, mp.mpf, bool, str | None], ...]
    status: str


@dataclass(frozen=True)
class TwoSidedRefinementStageReport:
    """Complete report for one non-mirrored refinement stage."""

    settings: TwoSidedNewtonSettings
    initial: TwoSidedResidualResult
    final: TwoSidedResidualResult
    steps: tuple[TwoSidedNewtonStepReport, ...]
    status: str


@dataclass(frozen=True)
class TwoSidedCandidateTrack:
    """One non-mirrored scout seed followed through refinement."""

    seed_rank: int
    seed_region: str
    seed_point: TwoSidedSearchPoint
    scout_result: TwoSidedResidualResult
    stages: tuple[TwoSidedRefinementStageReport, ...]
    verifications: tuple[TwoSidedResidualResult, ...]
    classification: str


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


def _shift_point(point: TwoSidedSearchPoint, delta: tuple[mp.mpf, ...], damping: mp.mpf) -> TwoSidedSearchPoint:
    """Apply one damped Gauss-Newton delta to a scaled search point."""
    shifted = point
    for index, value in enumerate(delta):
        shifted = point_with_delta(shifted, index, damping * value)
    return shifted


def _coordinate_rejection(point: TwoSidedSearchPoint, settings: TwoSidedNewtonSettings) -> str | None:
    """Return the reason one trial point violates a coordinate guard."""
    if settings.min_s_coordinate is not None and point.s <= settings.min_s_coordinate:
        return "m_floor_rejected"
    if settings.max_abs_coordinate is None:
        return None
    if max(abs(value) for value in _coordinates(point)) > settings.max_abs_coordinate:
        return "coordinate_bound"
    return None


def _newton_delta(
    result: TwoSidedResidualResult,
    settings: TwoSidedNewtonSettings,
    base_params: ProblemParameters,
) -> tuple[tuple[mp.mpf, ...], mp.mpf]:
    """Solve the overdetermined finite-difference Gauss-Newton system."""
    jacobian = finite_difference_two_sided_jacobian(
        result.point,
        settings.config,
        settings.fd_step,
        base_params=base_params,
    )
    rhs = mp.matrix([[-value] for value in result.residual])
    solved, _residual = mp.qr_solve(jacobian.matrix, rhs)
    return tuple(solved[row] for row in range(solved.rows)), jacobian.condition_number


def _try_dampings(
    result: TwoSidedResidualResult,
    delta: tuple[mp.mpf, ...],
    settings: TwoSidedNewtonSettings,
    base_params: ProblemParameters,
) -> tuple[TwoSidedResidualResult | None, mp.mpf | None, tuple[tuple[mp.mpf, mp.mpf, bool, str | None], ...]]:
    """Return the first damped trial that strictly improves the residual."""
    trials = []
    for damping in settings.dampings:
        point = _shift_point(result.point, delta, damping)
        rejection = _coordinate_rejection(point, settings)
        if rejection is not None:
            trials.append((damping, mp.inf, True, rejection))
            continue
        trial = two_sided_residual(point, settings.config, base_params=base_params)
        trials.append((damping, trial.residual_norm, trial.failure is not None, trial.failure))
        if trial.failure is None and trial.residual_norm < result.residual_norm:
            return trial, damping, tuple(trials)
    return None, None, tuple(trials)


def _failed_step(index: int, result: TwoSidedResidualResult, status: str) -> TwoSidedNewtonStepReport:
    """Build a step report for a failed Gauss-Newton setup."""
    return TwoSidedNewtonStepReport(index, result.point, result, None, None, result, None, (), status)


def _attempt_step(
    index: int,
    result: TwoSidedResidualResult,
    settings: TwoSidedNewtonSettings,
    base_params: ProblemParameters,
) -> TwoSidedNewtonStepReport:
    """Attempt one damped Gauss-Newton step from the current residual."""
    try:
        delta, condition = _newton_delta(result, settings, base_params)
    except (TypeError, ValueError, ZeroDivisionError):
        return _failed_step(index, result, "jacobian_failure")
    trial, damping, trials = _try_dampings(result, delta, settings, base_params)
    if trial is None:
        return TwoSidedNewtonStepReport(index, result.point, result, delta, None, result, condition, trials, "no_improvement")
    return TwoSidedNewtonStepReport(index, result.point, result, delta, damping, trial, condition, trials, "improved")


def _stage_status(
    result: TwoSidedResidualResult,
    settings: TwoSidedNewtonSettings,
    steps: list[TwoSidedNewtonStepReport],
) -> str | None:
    """Return a terminal stage status if one has already been reached."""
    if result.failure:
        return "branch_failure"
    if result.residual_norm <= settings.tolerance:
        return "tolerance_hit"
    if steps and steps[-1].status != "improved":
        return steps[-1].status
    return None


def two_sided_newton_refine(
    point: TwoSidedSearchPoint,
    settings: TwoSidedNewtonSettings,
    *,
    base_params: ProblemParameters = DEFAULT_PARAMS,
) -> TwoSidedRefinementStageReport:
    """Run one nonfatal damped Gauss-Newton refinement stage."""
    with mp.workdps(settings.config.working_dps):
        current = two_sided_residual(point, settings.config, base_params=base_params)
        initial = current
        steps: list[TwoSidedNewtonStepReport] = []
        for index in range(settings.max_steps):
            status = _stage_status(current, settings, steps)
            if status is not None:
                return TwoSidedRefinementStageReport(settings, initial, current, tuple(steps), status)
            step = _attempt_step(index, current, settings, base_params)
            steps.append(step)
            current = step.residual_after
        status = _stage_status(current, settings, steps) or "max_steps"
        return TwoSidedRefinementStageReport(settings, initial, current, tuple(steps), status)
