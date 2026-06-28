"""Damped Newton refinement for mirror-closing search points."""

from __future__ import annotations

from dataclasses import dataclass, field

from mpmath import mp

from problem import DEFAULT_PARAMS, ProblemParameters, SolverConfig
from .mirror_shooting import MirrorResidualResult, MirrorSearchPoint, finite_difference_jacobian, mirror_residual, point_with_delta


def _default_dampings() -> tuple[mp.mpf, ...]:
    """Return the fixed damped-Newton backtracking schedule."""
    return (mp.one, mp.mpf("0.5"), mp.mpf("0.25"), mp.mpf("0.125"), mp.mpf("0.0625"))


@dataclass(frozen=True)
class NewtonSettings:
    """Numerical settings for one Newton refinement stage."""

    name: str
    config: SolverConfig
    fd_step: mp.mpf
    tolerance: mp.mpf
    max_steps: int
    dampings: tuple[mp.mpf, ...] = field(default_factory=_default_dampings)
    max_abs_coordinate: mp.mpf | None = None
    min_s_coordinate: mp.mpf | None = None


@dataclass(frozen=True)
class NewtonStepReport:
    """Diagnostic data for one attempted Newton step."""

    index: int
    point_before: MirrorSearchPoint
    residual_before: MirrorResidualResult
    delta: tuple[mp.mpf, ...] | None
    damping: mp.mpf | None
    residual_after: MirrorResidualResult
    condition_number: mp.mpf | None
    trial_norms: tuple[tuple[mp.mpf, mp.mpf, bool, str | None], ...]
    status: str


@dataclass(frozen=True)
class RefinementStageReport:
    """Complete report for one refinement stage."""

    settings: NewtonSettings
    initial: MirrorResidualResult
    final: MirrorResidualResult
    steps: tuple[NewtonStepReport, ...]
    status: str


@dataclass(frozen=True)
class CandidateTrack:
    """One scout seed followed through refinement and verification."""

    seed_rank: int
    seed_region: str
    seed_point: MirrorSearchPoint
    scout_result: MirrorResidualResult
    stages: tuple[RefinementStageReport, ...]
    verifications: tuple[MirrorResidualResult, ...]
    classification: str


def _shift_point(point: MirrorSearchPoint, delta: tuple[mp.mpf, ...], damping: mp.mpf) -> MirrorSearchPoint:
    """Apply one damped Newton delta to a scaled search point."""
    shifted = point
    for index, value in enumerate(delta):
        shifted = point_with_delta(shifted, index, damping * value)
    return shifted


def _coordinate_rejection(point: MirrorSearchPoint, settings: NewtonSettings) -> str | None:
    """Return the reason one trial point violates a coordinate guard."""
    if settings.min_s_coordinate is not None and point.s <= settings.min_s_coordinate:
        return "m_floor_rejected"
    if settings.max_abs_coordinate is None:
        return None
    if max(abs(point.u), abs(point.v), abs(point.r), abs(point.s)) > settings.max_abs_coordinate:
        return "coordinate_bound"
    return None


def _within_coordinate_bound(point: MirrorSearchPoint, settings: NewtonSettings) -> bool:
    """Return whether one trial point respects all coordinate guards."""
    return _coordinate_rejection(point, settings) is None


def _newton_delta(
    result: MirrorResidualResult,
    settings: NewtonSettings,
    base_params: ProblemParameters,
) -> tuple[tuple[mp.mpf, ...], mp.mpf] | None:
    """Solve the finite-difference Newton system at one point."""
    jacobian = finite_difference_jacobian(result.point, settings.config, settings.fd_step, base_params=base_params)
    rhs = mp.matrix([[-value] for value in result.residual])
    solved = mp.lu_solve(jacobian.matrix, rhs)
    return tuple(solved[row] for row in range(solved.rows)), jacobian.condition_number


def _try_dampings(
    result: MirrorResidualResult,
    delta: tuple[mp.mpf, ...],
    settings: NewtonSettings,
    base_params: ProblemParameters,
) -> tuple[MirrorResidualResult | None, mp.mpf | None, tuple[tuple[mp.mpf, mp.mpf, bool, str | None], ...]]:
    """Return the first damped Newton trial that strictly improves."""
    trials = []
    for damping in settings.dampings:
        point = _shift_point(result.point, delta, damping)
        rejection = _coordinate_rejection(point, settings)
        if rejection is not None:
            trials.append((damping, mp.inf, True, rejection))
            continue
        trial = mirror_residual(point, settings.config, base_params=base_params)
        trials.append((damping, trial.residual_norm, trial.failure is not None, trial.failure))
        if trial.failure is None and trial.residual_norm < result.residual_norm:
            return trial, damping, tuple(trials)
    return None, None, tuple(trials)


def _failed_step(index: int, result: MirrorResidualResult, status: str) -> NewtonStepReport:
    """Build a step report for a failed Newton setup."""
    return NewtonStepReport(index, result.point, result, None, None, result, None, (), status)


def _attempt_step(
    index: int,
    result: MirrorResidualResult,
    settings: NewtonSettings,
    base_params: ProblemParameters,
) -> NewtonStepReport:
    """Attempt one damped Newton step from the current residual."""
    try:
        solved = _newton_delta(result, settings, base_params)
    except (TypeError, ValueError, ZeroDivisionError):
        return _failed_step(index, result, "jacobian_failure")
    delta, condition = solved
    trial, damping, trials = _try_dampings(result, delta, settings, base_params)
    if trial is None:
        return NewtonStepReport(index, result.point, result, delta, None, result, condition, trials, "no_improvement")
    return NewtonStepReport(index, result.point, result, delta, damping, trial, condition, trials, "improved")


def _stage_status(result: MirrorResidualResult, settings: NewtonSettings, steps: list[NewtonStepReport]) -> str | None:
    """Return a terminal stage status if one has already been reached."""
    if result.failure:
        return "branch_failure"
    if result.residual_norm <= settings.tolerance:
        return "tolerance_hit"
    if steps and steps[-1].status != "improved":
        return steps[-1].status
    return None


def newton_refine(
    point: MirrorSearchPoint,
    settings: NewtonSettings,
    *,
    base_params: ProblemParameters = DEFAULT_PARAMS,
) -> RefinementStageReport:
    """Run one nonfatal damped-Newton refinement stage."""
    with mp.workdps(settings.config.working_dps):
        current = mirror_residual(point, settings.config, base_params=base_params)
        initial = current
        steps: list[NewtonStepReport] = []
        for index in range(settings.max_steps):
            status = _stage_status(current, settings, steps)
            if status is not None:
                return RefinementStageReport(settings, initial, current, tuple(steps), status)
            step = _attempt_step(index, current, settings, base_params)
            steps.append(step)
            current = step.residual_after
        status = _stage_status(current, settings, steps) or "max_steps"
        return RefinementStageReport(settings, initial, current, tuple(steps), status)
