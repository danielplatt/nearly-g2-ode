"""Maximal-volume matching utilities for the G2 q-system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mpmath import mp

from problem import DEFAULT_CONFIG, DEFAULT_PARAMS, ProblemParameters, SolverConfig
from problem.charts import LEFT_CHART, WeightedChart, right_chart_for_params
from problem.q_system import mean_curvature, volume_density
from problem.types import State
from solver.march import (
    BranchSample,
    SeriesPatch,
    _branch_sample,
    _build_patch,
    _check_branch,
    _initial_patch,
    _match_data,
    _next_tau,
    _residual_max,
)


MAX_VOLUME_VERSION = "g2-max-volume-v1"


@dataclass(frozen=True)
class MaxVolumeSettings:
    """Numerical controls for one-sided maximal-volume event finding."""

    config: SolverConfig = DEFAULT_CONFIG
    max_tau: mp.mpf | None = None
    bisection_steps: int = 48
    event_tolerance: mp.mpf = mp.mpf("1e-30")


@dataclass(frozen=True)
class MaxVolumeSideResult:
    """One endpoint march stopped at its maximal-volume principal orbit."""

    chart_name: str
    status: str
    max_tau: mp.mpf | None
    max_y: State[mp.mpf] | None
    max_ydot: State[mp.mpf] | None
    max_q: State[mp.mpf] | None
    max_qdot: State[mp.mpf] | None
    volume: mp.mpf | None
    mean_curvature: mp.mpf | None
    patches: tuple[SeriesPatch, ...]
    invariant_log: tuple[BranchSample, ...]
    diagnostics: dict[str, Any]
    failure: str | None = None


@dataclass(frozen=True)
class MaxVolumeMatchResult:
    """A two-ended maximal-volume matching residual."""

    params: ProblemParameters
    settings: MaxVolumeSettings
    left: MaxVolumeSideResult
    right: MaxVolumeSideResult
    residual: tuple[mp.mpf, ...]
    residual_norm: mp.mpf
    reconstructed_interval: mp.mpf | None
    interval_error: mp.mpf | None
    failure: str | None = None


def _max_tau(settings: MaxVolumeSettings, params: ProblemParameters) -> mp.mpf:
    """Return the local-time cap for event finding."""
    if settings.max_tau is not None:
        return settings.max_tau
    return params.interval_end


def _is_finite(value: mp.mpf) -> bool:
    """Return whether one mpmath scalar is finite."""
    return bool(mp.isfinite(value))


def _curvature_at(
    chart: WeightedChart,
    patch: SeriesPatch,
    tau: mp.mpf,
    params: ProblemParameters,
) -> tuple[mp.mpf, State[mp.mpf], State[mp.mpf], BranchSample]:
    """Evaluate physical mean curvature and branch data at one local time."""
    _y, _ydot, q, qdot = _match_data(chart, patch, tau, params)
    sample = _branch_sample(chart, tau, q, params)
    _check_branch(sample)
    curvature = mean_curvature(q, qdot)
    if not _is_finite(curvature):
        raise ValueError("mean curvature is not finite")
    return curvature, q, qdot, sample


def _sample_taus(patch: SeriesPatch, next_tau: mp.mpf, config: SolverConfig) -> tuple[mp.mpf, ...]:
    """Return interior sample times plus the step endpoint for event checks."""
    step = next_tau - patch.centre
    samples = [
        patch.centre + step * mp.mpf(index) / (config.sample_points + 1)
        for index in range(1, config.sample_points + 1)
    ]
    samples.append(next_tau)
    return tuple(tau for tau in samples if tau > patch.centre)


def _bracket_crossed(left_value: mp.mpf, right_value: mp.mpf, tolerance: mp.mpf) -> bool:
    """Return whether a scalar event is bracketed."""
    return abs(left_value) <= tolerance or abs(right_value) <= tolerance or left_value * right_value <= 0


def _bisect_event(
    chart: WeightedChart,
    patch: SeriesPatch,
    low_tau: mp.mpf,
    high_tau: mp.mpf,
    low_value: mp.mpf,
    high_value: mp.mpf,
    params: ProblemParameters,
    settings: MaxVolumeSettings,
) -> mp.mpf:
    """Bisect one same-patch mean-curvature sign bracket."""
    if abs(low_value) <= settings.event_tolerance:
        return low_tau
    if abs(high_value) <= settings.event_tolerance:
        return high_tau
    left_tau = low_tau
    right_tau = high_tau
    left_value = low_value
    right_value = high_value
    for _ in range(settings.bisection_steps):
        mid_tau = (left_tau + right_tau) / 2
        mid_value, _q, _qdot, _sample = _curvature_at(chart, patch, mid_tau, params)
        if abs(mid_value) <= settings.event_tolerance:
            return mid_tau
        if left_value * mid_value <= 0:
            right_tau = mid_tau
            right_value = mid_value
        else:
            left_tau = mid_tau
            left_value = mid_value
    return (left_tau + right_tau) / 2


def _success_result(
    chart: WeightedChart,
    patches: list[SeriesPatch],
    invariant_log: list[BranchSample],
    residual_maxima: list[mp.mpf],
    event_tau: mp.mpf,
    params: ProblemParameters,
) -> MaxVolumeSideResult:
    """Build a successful maximal-volume side result."""
    y, ydot, q, qdot = _match_data(chart, patches[-1], event_tau, params)
    sample = _branch_sample(chart, event_tau, q, params)
    _check_branch(sample)
    invariant_log.append(sample)
    curvature = mean_curvature(q, qdot)
    diagnostics = _diagnostics(invariant_log, residual_maxima)
    diagnostics["physical_t"] = chart.physical_t(event_tau, params)
    return MaxVolumeSideResult(
        chart_name=chart.name,
        status="max_volume",
        max_tau=event_tau,
        max_y=y,
        max_ydot=ydot,
        max_q=q,
        max_qdot=qdot,
        volume=volume_density(chart.physical_t(event_tau, params), q, params),
        mean_curvature=curvature,
        patches=tuple(patches),
        invariant_log=tuple(invariant_log),
        diagnostics=diagnostics,
    )


def _diagnostics(invariant_log: list[BranchSample], residual_maxima: list[mp.mpf]) -> dict[str, Any]:
    """Return JSON-friendly-ish diagnostics for one march."""
    if not invariant_log:
        return {"residual_maxima": residual_maxima}
    return {
        "residual_maxima": residual_maxima,
        "min_sum27": min(sample.sum27 for sample in invariant_log),
        "min_sum36": min(sample.sum36 for sample in invariant_log),
        "max_gap": max(sample.gap for sample in invariant_log),
        "min_product": min(sample.product for sample in invariant_log),
    }


def _failure_result(
    chart: WeightedChart,
    status: str,
    message: str,
    patches: list[SeriesPatch],
    invariant_log: list[BranchSample],
    residual_maxima: list[mp.mpf],
) -> MaxVolumeSideResult:
    """Build a failed maximal-volume side result."""
    return MaxVolumeSideResult(
        chart_name=chart.name,
        status=status,
        max_tau=None,
        max_y=None,
        max_ydot=None,
        max_q=None,
        max_qdot=None,
        volume=None,
        mean_curvature=None,
        patches=tuple(patches),
        invariant_log=tuple(invariant_log),
        diagnostics=_diagnostics(invariant_log, residual_maxima),
        failure=message,
    )


def march_to_max_volume(
    chart: WeightedChart,
    params: ProblemParameters = DEFAULT_PARAMS,
    settings: MaxVolumeSettings | None = None,
) -> MaxVolumeSideResult:
    """March one endpoint chart until the principal-orbit volume is stationary."""
    settings = settings or MaxVolumeSettings()
    config = settings.config
    mp.dps = config.working_dps
    target_tau = _max_tau(settings, params)
    patches: list[SeriesPatch] = []
    invariant_log: list[BranchSample] = []
    residual_maxima: list[mp.mpf] = []
    try:
        patches.append(_initial_patch(chart, params, config))
        residual_maxima.append(_residual_max(chart, patches[-1].coefficients, mp.zero, params))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return _failure_result(chart, "event_failure", str(exc), patches, invariant_log, residual_maxima)

    previous_tau: mp.mpf | None = None
    previous_value: mp.mpf | None = None
    while patches[-1].centre < target_tau:
        patch = patches[-1]
        try:
            next_tau = _next_tau(patch, target_tau, config)
            for tau in _sample_taus(patch, next_tau, config):
                curvature, _q, _qdot, sample = _curvature_at(chart, patch, tau, params)
                invariant_log.append(sample)
                if previous_tau is not None and previous_value is not None:
                    if _bracket_crossed(previous_value, curvature, settings.event_tolerance):
                        event_tau = _bisect_event(
                            chart,
                            patch,
                            previous_tau,
                            tau,
                            previous_value,
                            curvature,
                            params,
                            settings,
                        )
                        return _success_result(chart, patches, invariant_log, residual_maxima, event_tau, params)
                previous_tau = tau
                previous_value = curvature
            y_next = patch.evaluate(next_tau)
            next_patch = _build_patch(chart, next_tau, y_next, config.series_order, params)
            residual_maxima.append(_residual_max(chart, next_patch.coefficients, next_tau, params))
            patches.append(next_patch)
        except ValueError as exc:
            return _failure_result(chart, "branch_exit", str(exc), patches, invariant_log, residual_maxima)
        except (TypeError, ZeroDivisionError) as exc:
            return _failure_result(chart, "event_failure", str(exc), patches, invariant_log, residual_maxima)

    return _failure_result(
        chart,
        "no_max_volume",
        f"no mean-curvature sign change before tau={mp.nstr(target_tau, 30)}",
        patches,
        invariant_log,
        residual_maxima,
    )


def max_volume_match(
    params: ProblemParameters = DEFAULT_PARAMS,
    settings: MaxVolumeSettings | None = None,
) -> MaxVolumeMatchResult:
    """Match the left and right endpoint marches at their maximal-volume orbits."""
    settings = settings or MaxVolumeSettings()
    with mp.workdps(settings.config.working_dps):
        left = march_to_max_volume(LEFT_CHART, params, settings)
        right = march_to_max_volume(right_chart_for_params(params), params, settings)
    if left.failure or right.failure or left.max_q is None or right.max_q is None:
        failure = left.failure or right.failure or "max-volume event failed"
        return MaxVolumeMatchResult(params, settings, left, right, (), mp.inf, None, None, failure)
    residual = tuple(lval - rval for lval, rval in zip(left.max_q, right.max_q))
    norm = max(abs(value) for value in residual)
    assert left.max_tau is not None and right.max_tau is not None
    reconstructed = left.max_tau + right.max_tau
    return MaxVolumeMatchResult(
        params=params,
        settings=settings,
        left=left,
        right=right,
        residual=residual,
        residual_norm=norm,
        reconstructed_interval=reconstructed,
        interval_error=reconstructed - params.interval_end,
    )
