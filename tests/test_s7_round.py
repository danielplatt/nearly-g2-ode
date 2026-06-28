"""Tests for the first round-S7 validation target."""

from __future__ import annotations

from mpmath import mp

from experiments.s7.round_validation import build_left_preset, build_params
from problem import (
    DEFAULT_CONFIG,
    DEFAULT_PARAMS,
    LEFT_CHART,
    ProblemParameters,
    S7_P3_RIGHT_CHART,
    State,
    initial_left_series,
    initial_right_series,
    left_first_jet_from_values,
    left_zero_jet_from_values,
    q_rhs,
    round_s7_candidate_parameters,
    round_s7_left_parameters,
    source_alpha,
    weighted_m_minus_one_residual,
    weighted_series_residual,
)
from solver.march import solve_two_sided


def _max_abs(state) -> mp.mpf:
    """Return the largest absolute component in one state."""
    return max(abs(value) for value in state)


def _round_s7_left_problem() -> ProblemParameters:
    """Build a left-only test package; the right endpoint is intentionally unused."""
    preset = round_s7_left_parameters()
    return ProblemParameters(
        lam=preset.lam,
        interval_end=DEFAULT_PARAMS.interval_end,
        left=preset.left,
        right=DEFAULT_PARAMS.right,
    )


def _round_s7_q(t: mp.mpf) -> State[mp.mpf]:
    """Return the independently derived round-S7 q(t) in project conventions."""
    sqrt5 = mp.sqrt(5)
    phases = (mp.zero, 2 * mp.pi / 3, 4 * mp.pi / 3)
    signs = (1, 1, -1, -1, -1, -1, 1, 1)
    a_values = []
    b_values = []
    d_values = []
    for phase in phases:
        cosine = mp.cos(t + phase)
        sine = mp.sin(t + phase)
        a_values.append(1 / sqrt5)
        b_values.append((2 * cosine + 1) / sqrt5)
        d_values.append(-2 * sine)
    a1, a2, a3 = a_values
    b1, b2, b3 = b_values
    d1, d2, d3 = d_values
    components = (
        a1 * a2 * a3,
        a1 * b2 * a3,
        a1 * a2 * b3,
        a1 * (b2 * b3 - d2 * d3),
        b1 * a2 * a3,
        a3 * (b1 * b2 - d1 * d2),
        a2 * (b1 * b3 - d1 * d3),
        b1 * b2 * b3 - b1 * d2 * d3 - d1 * b2 * d3 - d1 * d2 * b3,
    )
    return State.from_iterable(sign * value for sign, value in zip(signs, components))


def test_round_s7_left_preset_matches_derived_values() -> None:
    """The round-S7 left endpoint should use the sign-flipped S7 branch."""
    with mp.workdps(80):
        sqrt5 = mp.sqrt(5)
        preset = round_s7_left_parameters()
        assert abs(preset.lam - 6 / sqrt5) < mp.mpf("1e-40")
        assert abs(preset.left.a - sqrt5 / 25) < mp.mpf("1e-40")
        assert abs(preset.left.c + 3 * sqrt5 / 5) < mp.mpf("1e-40")
        assert abs(preset.left.alpha - sqrt5 / 50) < mp.mpf("1e-40")
        assert abs(source_alpha(preset.left.a, preset.left.c, preset.lam) + preset.left.alpha) < mp.mpf("1e-40")


def test_round_s7_left_zero_and_first_jets_match_formula() -> None:
    """The formula-derived left jet should match the known round-S7 values."""
    with mp.workdps(80):
        sqrt5 = mp.sqrt(5)
        sqrt15 = mp.sqrt(15)
        preset = build_left_preset()
        y0 = left_zero_jet_from_values(preset.left.a, preset.left.c, preset.lam)
        y1 = left_first_jet_from_values(preset.left.a, preset.left.c, preset.left.alpha, preset.lam)
        expected_y0 = State(
            mp.zero,
            -sqrt15 / 25,
            -sqrt15 / 25,
            23 * sqrt5 / 25,
            sqrt5 / 25,
            13 * sqrt15 / 25,
            13 * sqrt15 / 25,
            -144 * sqrt5 / 25,
        )
        assert _max_abs(y0 - expected_y0) < mp.mpf("1e-40")
        assert y1.y1 == y1.y4 == y1.y5 == y1.y8 == 0
        assert abs(y1.y2 - preset.left.alpha) < mp.mpf("1e-40")
        assert abs(y1.y3 + preset.left.alpha) < mp.mpf("1e-40")
        assert abs(y1.y6 + 23 * sqrt5 / 50) < mp.mpf("1e-40")
        assert abs(y1.y7 - 23 * sqrt5 / 50) < mp.mpf("1e-40")


def test_round_s7_left_weighted_seed_solves_the_left_equations() -> None:
    """The left singular Taylor seed should satisfy the weighted system."""
    with mp.workdps(80):
        params = _round_s7_left_problem()
        config = type(DEFAULT_CONFIG)(8, 80, 30, DEFAULT_CONFIG.step_safety, 2, DEFAULT_CONFIG.match_t)
        m_minus_one = weighted_m_minus_one_residual(LEFT_CHART, params)
        coeffs = initial_left_series(params, config)
        residual = weighted_series_residual(LEFT_CHART, coeffs, mp.zero, params)
        assert _max_abs(m_minus_one) < mp.mpf("1e-30")
        assert max(abs(value) for component in residual for value in component[:-1]) < mp.mpf("1e-20")


def test_round_s7_right_endpoint_data_match_explicit_formula() -> None:
    """The p3 right chart should encode the derived round-S7 endpoint."""
    with mp.workdps(80):
        params = round_s7_candidate_parameters()
        endpoint = _round_s7_q(params.interval_end)
        assert params.fixed_right is not None
        assert params.right_chart == "s7_p3"
        assert _max_abs(endpoint - params.fixed_right.offset) < mp.mpf("1e-40")
        assert _max_abs(build_params().fixed_right.offset - params.fixed_right.offset) < mp.mpf("1e-40")


def test_round_s7_explicit_formula_solves_q_system() -> None:
    """The derived round-S7 formula should satisfy the raw q-system."""
    with mp.workdps(80):
        params = round_s7_candidate_parameters()
        t = mp.mpf("0.4")
        h = mp.mpf("1e-7")
        derivative = (_round_s7_q(t + h) - _round_s7_q(t - h)) * (1 / (2 * h))
        residual = derivative - q_rhs(t, _round_s7_q(t), params)
        assert _max_abs(residual) < mp.mpf("1e-12")


def test_round_s7_p3_right_seed_solves_weighted_equations() -> None:
    """The fixed p3 endpoint jet should generate a valid weighted Taylor seed."""
    with mp.workdps(80):
        params = round_s7_candidate_parameters()
        config = type(DEFAULT_CONFIG)(24, 100, 30, DEFAULT_CONFIG.step_safety, 2, DEFAULT_CONFIG.match_t)
        coeffs = initial_right_series(params, config)
        residual = weighted_series_residual(S7_P3_RIGHT_CHART, coeffs, mp.zero, params)
        assert max(abs(value) for component in residual for value in component[: config.series_order - 4]) < mp.mpf(
            "1e-20"
        )


def test_round_s7_two_sided_march_matches_at_midpoint() -> None:
    """The round-S7 endpoint conditions should close in the middle."""
    with mp.workdps(80):
        params = round_s7_candidate_parameters()
        config = type(DEFAULT_CONFIG)(24, 120, 30, mp.mpf("0.6"), 0, params.interval_end / 2)
        result = solve_two_sided(params, config)
        assert result.mismatch_norm < mp.mpf("1e-12")
