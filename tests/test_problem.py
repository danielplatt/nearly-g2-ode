"""Tests for the two-ended geometry layer and mirrored endpoint data."""

from __future__ import annotations

from mpmath import mp

from problem import (
    DEFAULT_PARAMS,
    LEFT_CHART,
    LEFT_RHO,
    RIGHT_CHART,
    RIGHT_RHO,
    SolverConfig,
    State,
    initial_left_series,
    initial_right_series,
    left_first_jet,
    left_zero_jet,
    mirrored_problem_parameters,
    right_first_jet,
    right_zero_jet,
    source_alpha,
    weighted_m_minus_one_residual,
    weighted_series_residual,
)


def _max_abs(state) -> mp.mpf:
    """Return the largest absolute component in one state."""
    return max(abs(value) for value in state)


def _doubled_sphere_params():
    """Return the mirrored round-S7 candidate parameter package."""
    sqrt5 = mp.sqrt(5)
    lam = 6 / sqrt5
    a = sqrt5 / 25
    c = -3 * sqrt5 / 5
    return mirrored_problem_parameters(a, c, source_alpha(a, c, lam), lam, mp.pi / 3)


def _non_mirrored_params():
    """Return one admissible non-mirrored endpoint package."""
    with mp.workdps(80):
        left = DEFAULT_PARAMS.left
        right_type_a = -DEFAULT_PARAMS.right.d * mp.mpf("1.15")
        right_type_c = -DEFAULT_PARAMS.right.f * mp.mpf("0.8")
        right_alpha = source_alpha(right_type_a, right_type_c, DEFAULT_PARAMS.lam)
        return type(DEFAULT_PARAMS)(
            lam=DEFAULT_PARAMS.lam,
            interval_end=DEFAULT_PARAMS.interval_end,
            left=left,
            right=type(DEFAULT_PARAMS.right)(d=-right_type_a, f=-right_type_c, omega=-right_alpha),
        )


def _positive_ac_params():
    """Return one mirrored package on the real ac > 0 endpoint branch."""
    with mp.workdps(80):
        sqrt5 = mp.sqrt(5)
        lam = 6 / sqrt5
        return mirrored_problem_parameters(sqrt5 / 20, 3 * sqrt5 / 100, -sqrt5 / 50, lam, mp.pi / 3)


def _mirror_state(state):
    """Mirror one left weighted state into right weighted coordinates."""
    return State(state.y8, -state.y4, state.y6, -state.y2, -state.y7, state.y3, -state.y5, state.y1)


def test_left_and_right_jets_match_the_berger_branch() -> None:
    """The formula-derived endpoint jets should reproduce the checked Berger data."""
    with mp.workdps(80):
        sqrt5 = mp.sqrt(5)
        sqrt15 = mp.sqrt(15)
        left0 = left_zero_jet(DEFAULT_PARAMS)
        right0 = right_zero_jet(DEFAULT_PARAMS)
        assert abs(left0.y1 - 9 * sqrt5 / 100) < mp.mpf("1e-40")
        assert abs(left0.y5 - 23 * sqrt5 / 100) < mp.mpf("1e-40")
        assert abs(right0.y1 + 9 * sqrt5 / 100) < mp.mpf("1e-40")
        assert abs(right0.y3 - 2 * sqrt15 / 25) < mp.mpf("1e-40")
        assert _max_abs(left_first_jet(DEFAULT_PARAMS) - DEFAULT_PARAMS.left.alpha * LEFT_RHO) < mp.mpf("1e-40")
        assert _max_abs(right_first_jet(DEFAULT_PARAMS) - DEFAULT_PARAMS.right.omega * RIGHT_RHO) < mp.mpf(
            "1e-40"
        )


def test_doubled_sphere_jets_match_the_source_formulas_and_mirror() -> None:
    """The round-S7 candidate should use formula-derived left data and mirrored right data."""
    with mp.workdps(80):
        sqrt5 = mp.sqrt(5)
        sqrt15 = mp.sqrt(15)
        params = _doubled_sphere_params()
        left0 = left_zero_jet(params)
        left1 = left_first_jet(params)
        expected_left0 = State(
            mp.zero,
            -sqrt15 / 25,
            -sqrt15 / 25,
            23 * sqrt5 / 25,
            sqrt5 / 25,
            13 * sqrt15 / 25,
            13 * sqrt15 / 25,
            -144 * sqrt5 / 25,
        )
        assert abs(params.left.alpha + sqrt5 / 50) < mp.mpf("1e-40")
        assert abs(left1.y6 - 23 * sqrt5 / 50) < mp.mpf("1e-40")
        assert _max_abs(left0 - expected_left0) < mp.mpf("1e-40")
        assert _max_abs(right_zero_jet(params) - _mirror_state(left0)) < mp.mpf("1e-40")
        assert _max_abs(right_first_jet(params) - _mirror_state(left1)) < mp.mpf("1e-40")


def test_endpoint_charts_round_trip_pointwise() -> None:
    """The left and right weighted maps should be mutually inverse away from tau=0."""
    with mp.workdps(80):
        for chart, y0, y1, tau in (
            (LEFT_CHART, left_zero_jet(DEFAULT_PARAMS), left_first_jet(DEFAULT_PARAMS), mp.mpf("0.1")),
            (RIGHT_CHART, right_zero_jet(DEFAULT_PARAMS), right_first_jet(DEFAULT_PARAMS), mp.mpf("0.1")),
        ):
            y = y0 + mp.mpf("0.3") * y1
            q = chart.y_to_q(tau, y, DEFAULT_PARAMS)
            recovered = chart.q_to_y(tau, q, DEFAULT_PARAMS)
            assert _max_abs(y - recovered) < mp.mpf("1e-40")


def test_right_chart_local_derivative_flips_the_raw_q_sign() -> None:
    """The right chart should satisfy dq/dtau = -dq/dt."""
    with mp.workdps(80):
        tau = mp.mpf("0.2")
        y = right_zero_jet(DEFAULT_PARAMS) + tau * right_first_jet(DEFAULT_PARAMS)
        q = RIGHT_CHART.y_to_q(tau, y, DEFAULT_PARAMS)
        local_qdot = RIGHT_CHART.local_q_rhs(tau, q, DEFAULT_PARAMS)
        physical_qdot = RIGHT_CHART.physical_qdot(local_qdot)
        assert _max_abs(local_qdot + physical_qdot) < mp.mpf("1e-40")


def test_weighted_m_minus_one_residual_vanishes_at_both_berger_ends() -> None:
    """The formula-derived left and mirrored right zero jets should cancel the singular residuals."""
    with mp.workdps(80):
        for params in (DEFAULT_PARAMS, _doubled_sphere_params()):
            left_residual = weighted_m_minus_one_residual(LEFT_CHART, params)
            right_residual = weighted_m_minus_one_residual(RIGHT_CHART, params)
            assert _max_abs(left_residual) < mp.mpf("1e-30")
            assert _max_abs(right_residual) < mp.mpf("1e-30")


def test_non_mirrored_right_jets_are_formula_derived() -> None:
    """Independent right endpoint data should no longer require mirror symmetry."""
    with mp.workdps(80):
        params = _non_mirrored_params()
        assert abs(params.right.d + params.left.a) > mp.mpf("1e-4")
        right0 = right_zero_jet(params)
        right1 = right_first_jet(params)
        residual = weighted_m_minus_one_residual(RIGHT_CHART, params)
        assert _max_abs(residual) < mp.mpf("1e-30")
        assert all(mp.isfinite(value) for value in right0)
        assert all(mp.isfinite(value) for value in right1)


def test_positive_ac_branch_jets_satisfy_the_weighted_equations() -> None:
    """The exploratory ac > 0 branch should define real, consistent endpoint jets."""
    config = SolverConfig(4, 80, 50, mp.mpf("0.5"), 0, mp.pi / 6)
    with mp.workdps(config.working_dps):
        params = _positive_ac_params()
        left0 = left_zero_jet(params)
        left1 = left_first_jet(params)
        right0 = right_zero_jet(params)
        right1 = right_first_jet(params)
        assert params.left.a > 0
        assert params.left.c > 0
        assert 3 * params.left.a - params.left.c > 0
        assert all(mp.isfinite(value) for value in left0)
        assert all(mp.isfinite(value) for value in left1)
        assert _max_abs(weighted_m_minus_one_residual(LEFT_CHART, params)) < mp.mpf("1e-30")
        assert _max_abs(weighted_m_minus_one_residual(RIGHT_CHART, params)) < mp.mpf("1e-30")
        assert _max_abs(right0 - _mirror_state(left0)) < mp.mpf("1e-40")
        assert _max_abs(right1 - _mirror_state(left1)) < mp.mpf("1e-40")

        left_residual = weighted_series_residual(LEFT_CHART, initial_left_series(params, config), mp.zero, params)
        right_residual = weighted_series_residual(RIGHT_CHART, initial_right_series(params, config), mp.zero, params)
        assert max(abs(value) for component in left_residual for value in component[:-1]) < mp.mpf("1e-20")
        assert max(abs(value) for component in right_residual for value in component[:-1]) < mp.mpf("1e-20")


def test_opposite_mu_requires_a_different_square_root_branch() -> None:
    """The opposite left mu germ is singular only after flipping the p1 branch."""
    with mp.workdps(80):
        old_branch = mirrored_problem_parameters(
            DEFAULT_PARAMS.left.a,
            DEFAULT_PARAMS.left.c,
            DEFAULT_PARAMS.left.alpha,
            DEFAULT_PARAMS.lam,
            DEFAULT_PARAMS.interval_end,
            left_mu=1,
        )
        flipped_branch = mirrored_problem_parameters(
            DEFAULT_PARAMS.left.a,
            DEFAULT_PARAMS.left.c,
            DEFAULT_PARAMS.left.alpha,
            DEFAULT_PARAMS.lam,
            DEFAULT_PARAMS.interval_end,
            left_mu=1,
            p_signs=(1, 1, 1),
        )
        assert _max_abs(weighted_m_minus_one_residual(LEFT_CHART, old_branch)) > mp.mpf("1")
        assert _max_abs(weighted_m_minus_one_residual(LEFT_CHART, flipped_branch)) < mp.mpf("1e-30")
        assert _max_abs(weighted_m_minus_one_residual(RIGHT_CHART, flipped_branch)) > mp.mpf("0.1")


def test_mixed_endpoint_p_signs_make_opposite_mu_taylor_germs_consistent() -> None:
    """Opposite mu gives valid one-sided Taylor germs with endpoint-local p-signs."""
    config = SolverConfig(6, 80, 50, mp.mpf("0.5"), 0, mp.pi / 6)
    with mp.workdps(config.working_dps):
        params = mirrored_problem_parameters(
            DEFAULT_PARAMS.left.a,
            DEFAULT_PARAMS.left.c,
            DEFAULT_PARAMS.left.alpha,
            DEFAULT_PARAMS.lam,
            DEFAULT_PARAMS.interval_end,
            left_mu=1,
            right_mu=1,
            p_signs=(1, 1, 1),
            right_p_signs=(-1, 1, -1),
        )
        assert _max_abs(weighted_m_minus_one_residual(LEFT_CHART, params)) < mp.mpf("1e-30")
        assert _max_abs(weighted_m_minus_one_residual(RIGHT_CHART, params)) < mp.mpf("1e-30")

        left_residual = weighted_series_residual(LEFT_CHART, initial_left_series(params, config), mp.zero, params)
        right_residual = weighted_series_residual(RIGHT_CHART, initial_right_series(params, config), mp.zero, params)
        assert max(abs(value) for component in left_residual for value in component[:-1]) < mp.mpf("1e-20")
        assert max(abs(value) for component in right_residual for value in component[:-1]) < mp.mpf("1e-20")
