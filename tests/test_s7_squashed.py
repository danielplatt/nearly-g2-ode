"""Tests for the derived squashed-S7 endpoint chart."""

from __future__ import annotations

from mpmath import mp

from problem import (
    S7_P2_RIGHT_CHART,
    SolverConfig,
    State,
    branch_quantities,
    initial_right_series,
    q_rhs,
    squashed_s7_parameters,
    weighted_series_residual,
)
from solver.march import solve_two_sided


def _max_abs(values) -> mp.mpf:
    """Return the largest absolute value in an iterable."""
    return max(abs(value) for value in values)


def _squashed_q(t: mp.mpf) -> State[mp.mpf]:
    """Return the independently derived squashed-S7 q(t) in project conventions."""
    sqrt5 = mp.sqrt(5)
    phases = (mp.zero, 2 * mp.pi / 3, 4 * mp.pi / 3)
    signs = (1, -1, 1, -1, -1, 1, -1, 1)
    a_values = []
    b_values = []
    d_values = []
    for phase in phases:
        c = mp.cos(t + phase)
        s = mp.sin(t + phase)
        a_values.append(1 / sqrt5)
        b_values.append((2 * c + 1) / sqrt5)
        d_values.append(-2 * s)
    a1, a2, a3 = a_values
    b1, b2, b3 = b_values
    d1, d2, d3 = d_values
    components = (
        a1 * a2 * a3,
        a1 * a2 * b3,
        a1 * b2 * a3,
        a1 * (b2 * b3 - d2 * d3),
        b1 * a2 * a3,
        a2 * (b1 * b3 - d1 * d3),
        a3 * (b1 * b2 - d1 * d2),
        b1 * b2 * b3 - b1 * d2 * d3 - d1 * b2 * d3 - d1 * d2 * b3,
    )
    return State.from_iterable(sign * value for sign, value in zip(signs, components))


def test_squashed_s7_right_endpoint_data_match_explicit_formula() -> None:
    """The p2 right chart should encode the explicit squashed-S7 endpoint."""
    with mp.workdps(80):
        params = squashed_s7_parameters()
        endpoint = _squashed_q(params.interval_end)
        assert params.fixed_right is not None
        assert _max_abs(endpoint - params.fixed_right.offset) < mp.mpf("1e-40")
        branch = branch_quantities(params.interval_end - mp.mpf("0.1"), _squashed_q(params.interval_end - mp.mpf("0.1")), params)
        assert branch.sum27 > 0
        assert branch.sum36 > 0
        assert branch.gap < 0
        assert branch.product > 0


def test_squashed_s7_explicit_formula_solves_q_system() -> None:
    """The independently derived formula should satisfy the raw q-system."""
    with mp.workdps(80):
        params = squashed_s7_parameters()
        t = mp.mpf("0.4")
        h = mp.mpf("1e-7")
        derivative = (_squashed_q(t + h) - _squashed_q(t - h)) * (1 / (2 * h))
        residual = derivative - q_rhs(t, _squashed_q(t), params)
        assert _max_abs(residual) < mp.mpf("1e-12")


def test_squashed_s7_p2_right_seed_solves_weighted_equations() -> None:
    """The fixed p2 endpoint jet should generate a valid weighted Taylor seed."""
    with mp.workdps(80):
        params = squashed_s7_parameters()
        config = SolverConfig(24, 100, 30, mp.mpf("0.5"), 0, params.interval_end / 2)
        coeffs = initial_right_series(params, config)
        residual = weighted_series_residual(S7_P2_RIGHT_CHART, coeffs, mp.zero, params)
        assert max(abs(value) for component in residual for value in component[: config.series_order - 4]) < mp.mpf("1e-20")


def test_squashed_s7_two_sided_march_matches_at_midpoint() -> None:
    """The squashed-S7 endpoint conditions should close in the middle."""
    with mp.workdps(80):
        params = squashed_s7_parameters()
        config = SolverConfig(24, 120, 30, mp.mpf("0.6"), 0, params.interval_end / 2)
        result = solve_two_sided(params, config)
        assert result.mismatch_norm < mp.mpf("1e-12")
