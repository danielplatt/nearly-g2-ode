"""Generic q-space Taylor patch marching."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, log10
from typing import Any

from mpmath import mp

from problem.initial_data import DEFAULT_CONFIG, DEFAULT_PARAMS, ProblemParameters, SolverConfig
from problem.q_system import branch_quantities, q_rhs
from problem.taylor_seed import initial_q_series
from problem.types import State
from problem.weights import q_to_y, qdot_to_ydot

from .series import evaluate_coefficients, state_to_coefficients, state_to_series


@dataclass(frozen=True)
class SeriesPatch:
    """One local Taylor patch for the q-system."""

    centre: mp.mpf
    coefficients: State[list[mp.mpf]]
    radius_estimate: mp.mpf

    @property
    def order(self) -> int:
        """Return the Taylor truncation order for this patch."""
        return len(self.coefficients.y1) - 1

    def evaluate(self, t: mp.mpf) -> State[mp.mpf]:
        """Evaluate the patch at one physical time."""
        local = t - self.centre
        return State.from_iterable(evaluate_coefficients(component, local) for component in self.coefficients)

    def derivative(self, t: mp.mpf) -> State[mp.mpf]:
        """Evaluate the patch derivative at one physical time."""
        local = t - self.centre
        derived = state_to_coefficients(state_to_series(self.coefficients).map(lambda value: value.derivative()))
        return State.from_iterable(evaluate_coefficients(component, local) for component in derived)


@dataclass(frozen=True)
class BranchSample:
    """Recorded branch data at one sample point."""

    t: mp.mpf
    sum27: mp.mpf
    sum36: mp.mpf
    gap: mp.mpf
    product: mp.mpf


@dataclass(frozen=True)
class MarchResult:
    """Complete output of marching q to the midpoint."""

    patches: list[SeriesPatch]
    midpoint_q: State[mp.mpf]
    midpoint_qdot: State[mp.mpf]
    midpoint_y: State[mp.mpf]
    midpoint_ydot: State[mp.mpf]
    invariant_log: list[BranchSample]
    diagnostics: dict[str, Any]


def _residual_max(coefficients: State[list[mp.mpf]], params: ProblemParameters) -> mp.mpf:
    """Measure the largest coefficient residual of q' - q_rhs on one patch."""
    series = state_to_series(coefficients)
    residual = state_to_coefficients(series.map(lambda value: value.derivative()) - q_rhs(None, series, params))
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


def _build_patch(centre: mp.mpf, q0: State[mp.mpf], order: int, params: ProblemParameters) -> SeriesPatch:
    """Build one regular Taylor patch around a positive centre."""
    coeffs = State.from_iterable([[value] + [mp.zero] * order for value in q0])
    for degree in range(order):
        rhs_coeffs = state_to_coefficients(q_rhs(None, state_to_series(coeffs), params))
        for component, rhs in zip(coeffs, rhs_coeffs):
            component[degree + 1] = rhs[degree] / (degree + 1)
    return SeriesPatch(centre=centre, coefficients=coeffs, radius_estimate=_tail_radius(coeffs))


def _initial_patch(params: ProblemParameters, config: SolverConfig) -> SeriesPatch:
    """Build the singular-end patch from the stored zero jet and alpha."""
    coeffs = initial_q_series(params, config)
    return SeriesPatch(centre=mp.zero, coefficients=coeffs, radius_estimate=_tail_radius(coeffs))


def _branch_sample(t: mp.mpf, q: State[mp.mpf], params: ProblemParameters) -> BranchSample:
    """Sample the q-branch quantities at one point."""
    branch = branch_quantities(t, q, params)
    return BranchSample(t=t, sum27=branch.sum27, sum36=branch.sum36, gap=branch.gap, product=branch.product)


def _check_branch(sample: BranchSample) -> None:
    """Require that the current sample stays on the intended real branch."""
    if sample.product <= 0:
        raise ValueError(f"Branch failure at t={sample.t}: -(q2+q7)(q3+q6)(q4+q5) must stay positive.")
    if sample.sum27 <= 0:
        raise ValueError(f"Branch failure at t={sample.t}: q2 + q7 must stay positive.")
    if sample.sum36 <= 0:
        raise ValueError(f"Branch failure at t={sample.t}: q3 + q6 must stay positive.")
    if sample.gap >= 0:
        raise ValueError(f"Branch failure at t={sample.t}: q4 + q5 must stay negative.")


def _record_samples(
    patch: SeriesPatch,
    next_centre: mp.mpf,
    params: ProblemParameters,
    config: SolverConfig,
    log: list[BranchSample],
    include_centre: bool,
) -> None:
    """Record the centre and interior branch samples for one step."""
    if include_centre and patch.centre > 0:
        centre_sample = _branch_sample(patch.centre, patch.evaluate(patch.centre), params)
        _check_branch(centre_sample)
        log.append(centre_sample)
    step = next_centre - patch.centre
    for idx in range(1, config.sample_points + 1):
        t = patch.centre + step * mp.mpf(idx) / (config.sample_points + 1)
        sample = _branch_sample(t, patch.evaluate(t), params)
        _check_branch(sample)
        log.append(sample)


def _next_centre(patch: SeriesPatch, config: SolverConfig) -> mp.mpf:
    """Advance by the configured safety factor times the radius estimate."""
    step = config.step_safety * patch.radius_estimate
    if step <= 0:
        raise ValueError("Taylor marcher produced a non-positive step.")
    return min(patch.centre + step, config.target_t)


def agreement_digits(left: mp.mpf, right: mp.mpf) -> int:
    """Estimate decimal agreement digits between two midpoint quantities."""
    diff = abs(left - right)
    if diff == 0:
        return 99
    return max(0, int(floor(-log10(float(diff)))))


def solve_to_midpoint(
    params: ProblemParameters = DEFAULT_PARAMS,
    config: SolverConfig = DEFAULT_CONFIG,
) -> MarchResult:
    """March the q-system from t=0 to the symmetry point target_t."""
    mp.dps = config.working_dps
    patches = [_initial_patch(params, config)]
    invariant_log: list[BranchSample] = []
    residual_maxima = [_residual_max(patches[0].coefficients, params)]
    while patches[-1].centre < config.target_t:
        next_centre = _next_centre(patches[-1], config)
        _record_samples(patches[-1], next_centre, params, config, invariant_log, include_centre=len(patches) == 1)
        next_patch = _build_patch(next_centre, patches[-1].evaluate(next_centre), config.series_order, params)
        residual_maxima.append(_residual_max(next_patch.coefficients, params))
        patches.append(next_patch)
    midpoint_q = patches[-1].evaluate(config.target_t)
    midpoint_qdot = patches[-1].derivative(config.target_t)
    midpoint_y = q_to_y(config.target_t, midpoint_q, params)
    midpoint_ydot = qdot_to_ydot(config.target_t, midpoint_q, midpoint_qdot, params)
    final_sample = _branch_sample(config.target_t, midpoint_q, params)
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
    return MarchResult(
        patches=patches,
        midpoint_q=midpoint_q,
        midpoint_qdot=midpoint_qdot,
        midpoint_y=midpoint_y,
        midpoint_ydot=midpoint_ydot,
        invariant_log=invariant_log,
        diagnostics=diagnostics,
    )
