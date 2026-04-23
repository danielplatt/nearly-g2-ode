"""Two-sided weighted Taylor marching with midpoint matching in raw q."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, log10
from typing import Any

from mpmath import mp

from problem.charts import LEFT_CHART, RIGHT_CHART, WeightedChart
from problem.initial_data import DEFAULT_CONFIG, DEFAULT_PARAMS, ProblemParameters, SolverConfig
from problem.q_system import branch_quantities, mean_curvature
from problem.taylor_seed import build_weighted_series, initial_weighted_series, weighted_series_residual
from problem.types import State
from solver.series import evaluate_coefficients, state_to_coefficients, state_to_series


@dataclass(frozen=True)
class SeriesPatch:
    """One local Taylor patch for one weighted endpoint chart."""

    centre: mp.mpf
    coefficients: State[list[mp.mpf]]
    radius_estimate: mp.mpf

    @property
    def order(self) -> int:
        """Return the Taylor truncation order."""
        return len(self.coefficients.y1) - 1

    def evaluate(self, tau: mp.mpf) -> State[mp.mpf]:
        """Evaluate the patch at one local chart time."""
        local = tau - self.centre
        return State.from_iterable(evaluate_coefficients(component, local) for component in self.coefficients)

    def derivative(self, tau: mp.mpf) -> State[mp.mpf]:
        """Evaluate the local derivative dy/dtau at one chart time."""
        local = tau - self.centre
        derived = state_to_coefficients(state_to_series(self.coefficients).map(lambda value: value.derivative()))
        return State.from_iterable(evaluate_coefficients(component, local) for component in derived)


@dataclass(frozen=True)
class BranchSample:
    """Recorded branch data at one sampled physical time."""

    physical_t: mp.mpf
    sum27: mp.mpf
    sum36: mp.mpf
    gap: mp.mpf
    product: mp.mpf


@dataclass(frozen=True)
class SideResult:
    """Complete output of one endpoint march to the common match point."""

    chart_name: str
    patches: list[SeriesPatch]
    match_tau: mp.mpf
    match_y: State[mp.mpf]
    match_ydot: State[mp.mpf]
    match_q: State[mp.mpf]
    match_qdot: State[mp.mpf]
    invariant_log: list[BranchSample]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class TwoSidedResult:
    """Complete output of the two-sided weighted Berger validation run."""

    left: SideResult
    right: SideResult
    mismatch_q: State[mp.mpf]
    mismatch_norm: mp.mpf
    left_l: mp.mpf
    right_l: mp.mpf
    diagnostics: dict[str, Any]


def _residual_max(chart: WeightedChart, coefficients: State[list[mp.mpf]], centre: mp.mpf, params: ProblemParameters) -> mp.mpf:
    """Measure the largest weighted coefficient residual on one patch."""
    residual = weighted_series_residual(chart, coefficients, centre, params)
    return max(abs(value) for component in residual for value in component[:-1])


def _tail_radius(coefficients: State[list[mp.mpf]]) -> mp.mpf:
    """Estimate the local convergence radius from the coefficient tail."""
    order = len(coefficients.y1) - 1
    growths = []
    for degree in range(max(1, order - 2), order + 1):
        bound = max(abs(component[degree]) for component in coefficients)
        if bound != 0:
            growths.append(mp.power(bound, mp.mpf(1) / degree))
    if not growths:
        raise ValueError("Could not estimate a patch radius from zero tail coefficients.")
    return mp.mpf("0.8") / max(growths)


def _build_patch(
    chart: WeightedChart,
    centre: mp.mpf,
    y0: State[mp.mpf],
    order: int,
    params: ProblemParameters,
) -> SeriesPatch:
    """Build one positive-centre weighted Taylor patch."""
    coeffs = build_weighted_series(chart, centre, y0, order, params)
    return SeriesPatch(centre=centre, coefficients=coeffs, radius_estimate=_tail_radius(coeffs))


def _initial_patch(chart: WeightedChart, params: ProblemParameters, config: SolverConfig) -> SeriesPatch:
    """Build the singular-end weighted patch for one side."""
    coeffs = initial_weighted_series(chart, params, config)
    return SeriesPatch(centre=mp.zero, coefficients=coeffs, radius_estimate=_tail_radius(coeffs))


def _branch_sample(chart: WeightedChart, tau: mp.mpf, q: State[mp.mpf], params: ProblemParameters) -> BranchSample:
    """Sample the raw q branch quantities at one local chart time."""
    branch = branch_quantities(chart.physical_t(tau, params), q, params)
    return BranchSample(
        physical_t=chart.physical_t(tau, params),
        sum27=branch.sum27,
        sum36=branch.sum36,
        gap=branch.gap,
        product=branch.product,
    )


def _check_branch(sample: BranchSample) -> None:
    """Require that the current sample stays on the intended real branch."""
    if sample.product <= 0:
        raise ValueError(f"Branch failure at t={sample.physical_t}: -(q2+q7)(q3+q6)(q4+q5) must stay positive.")
    if sample.sum27 <= 0:
        raise ValueError(f"Branch failure at t={sample.physical_t}: q2 + q7 must stay positive.")
    if sample.sum36 <= 0:
        raise ValueError(f"Branch failure at t={sample.physical_t}: q3 + q6 must stay positive.")
    if sample.gap >= 0:
        raise ValueError(f"Branch failure at t={sample.physical_t}: q4 + q5 must stay negative.")


def _record_samples(
    chart: WeightedChart,
    patch: SeriesPatch,
    next_tau: mp.mpf,
    params: ProblemParameters,
    config: SolverConfig,
    log: list[BranchSample],
) -> None:
    """Record interior branch samples for one local step."""
    step = next_tau - patch.centre
    for idx in range(1, config.sample_points + 1):
        tau = patch.centre + step * mp.mpf(idx) / (config.sample_points + 1)
        q = chart.y_to_q(tau, patch.evaluate(tau), params)
        sample = _branch_sample(chart, tau, q, params)
        _check_branch(sample)
        log.append(sample)


def _next_tau(patch: SeriesPatch, target_tau: mp.mpf, config: SolverConfig) -> mp.mpf:
    """Advance by the configured safety factor times the radius estimate."""
    step = config.step_safety * patch.radius_estimate
    if step <= 0:
        raise ValueError("Taylor marcher produced a non-positive step.")
    return min(patch.centre + step, target_tau)


def _match_data(
    chart: WeightedChart,
    patch: SeriesPatch,
    tau: mp.mpf,
    params: ProblemParameters,
) -> tuple[State[mp.mpf], State[mp.mpf], State[mp.mpf], State[mp.mpf]]:
    """Recover y, ydot, q, and physical qdot at one local time."""
    y = patch.evaluate(tau)
    q = chart.y_to_q(tau, y, params)
    local_qdot = chart.local_q_rhs(tau, q, params)
    return y, chart.local_qdot_to_ydot(tau, y, local_qdot), q, chart.physical_qdot(local_qdot)


def _march_side(
    chart: WeightedChart,
    target_tau: mp.mpf,
    params: ProblemParameters,
    config: SolverConfig,
) -> SideResult:
    """March one weighted endpoint chart to the common match point."""
    patches = [_initial_patch(chart, params, config)]
    invariant_log: list[BranchSample] = []
    residual_maxima = [_residual_max(chart, patches[0].coefficients, mp.zero, params)]
    while patches[-1].centre < target_tau:
        next_tau = _next_tau(patches[-1], target_tau, config)
        _record_samples(chart, patches[-1], next_tau, params, config, invariant_log)
        y_next = patches[-1].evaluate(next_tau)
        next_patch = _build_patch(chart, next_tau, y_next, config.series_order, params)
        residual_maxima.append(_residual_max(chart, next_patch.coefficients, next_tau, params))
        patches.append(next_patch)
    match_y, match_ydot, match_q, match_qdot = _match_data(chart, patches[-1], target_tau, params)
    final_sample = _branch_sample(chart, target_tau, match_q, params)
    _check_branch(final_sample)
    invariant_log.append(final_sample)
    diagnostics = {
        "patch_centres": [patch.centre for patch in patches],
        "residual_maxima": residual_maxima,
        "min_sum27": min(sample.sum27 for sample in invariant_log),
        "min_sum36": min(sample.sum36 for sample in invariant_log),
        "max_gap": max(sample.gap for sample in invariant_log),
        "min_product": min(sample.product for sample in invariant_log),
    }
    return SideResult(
        chart_name=chart.name,
        patches=patches,
        match_tau=target_tau,
        match_y=match_y,
        match_ydot=match_ydot,
        match_q=match_q,
        match_qdot=match_qdot,
        invariant_log=invariant_log,
        diagnostics=diagnostics,
    )


def agreement_digits(left: mp.mpf, right: mp.mpf) -> int:
    """Estimate decimal agreement digits between two scalar quantities."""
    diff = abs(left - right)
    if diff == 0:
        return 99
    return max(0, int(floor(-log10(float(diff)))))


def solve_two_sided(
    params: ProblemParameters = DEFAULT_PARAMS,
    config: SolverConfig = DEFAULT_CONFIG,
) -> TwoSidedResult:
    """March the left and right weighted systems and compare raw q in the middle."""
    mp.dps = config.working_dps
    left = _march_side(LEFT_CHART, config.match_t, params, config)
    right = _march_side(RIGHT_CHART, params.interval_end - config.match_t, params, config)
    mismatch_q = left.match_q - right.match_q
    mismatch_norm = max(abs(value) for value in mismatch_q)
    left_l = mean_curvature(left.match_q, left.match_qdot)
    right_l = mean_curvature(right.match_q, right.match_qdot)
    diagnostics = {
        "match_t": config.match_t,
        "left_tau": config.match_t,
        "right_tau": params.interval_end - config.match_t,
        "mismatch_norm": mismatch_norm,
        "l_gap": abs(left_l - right_l),
    }
    return TwoSidedResult(
        left=left,
        right=right,
        mismatch_q=mismatch_q,
        mismatch_norm=mismatch_norm,
        left_l=left_l,
        right_l=right_l,
        diagnostics=diagnostics,
    )
