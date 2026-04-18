"""Tests for the q-series seed and midpoint marcher."""

from __future__ import annotations

from mpmath import mp

from problem import DEFAULT_CONFIG, DEFAULT_PARAMS, SolverConfig, initial_q_series, recovered_y_series, series_residual, y_first_jet, y_zero_jet
from solver import solve_to_midpoint


SMOKE_CONFIG = SolverConfig(
    series_order=12,
    working_dps=80,
    target_dps=30,
    step_safety=mp.mpf("0.5"),
    sample_points=5,
    target_t=DEFAULT_CONFIG.target_t,
)
SMOKE_REFINED = SolverConfig(
    series_order=16,
    working_dps=100,
    target_dps=40,
    step_safety=mp.mpf("0.5"),
    sample_points=5,
    target_t=DEFAULT_CONFIG.target_t,
)


def test_initial_q_series_recovers_the_prescribed_y_jets() -> None:
    """The singular q-series seed should reproduce y(0) and the forced first jet."""
    with mp.workdps(SMOKE_CONFIG.working_dps):
        q_coeffs = initial_q_series(DEFAULT_PARAMS, SMOKE_CONFIG)
        y_coeffs = recovered_y_series(q_coeffs, DEFAULT_PARAMS)
        y0 = y_zero_jet(DEFAULT_PARAMS)
        y1 = y_first_jet(DEFAULT_PARAMS)
        assert max(abs(left - right) for left, right in zip(y_coeffs.map(lambda coeffs: coeffs[0]), y0)) < mp.mpf("1e-25")
        assert max(abs(left - right) for left, right in zip(y_coeffs.map(lambda coeffs: coeffs[1]), y1)) < mp.mpf("1e-20")


def test_initial_q_series_satisfies_the_q_equations_to_truncation() -> None:
    """The generated singular seed should solve q' = q_rhs through the tested order."""
    with mp.workdps(SMOKE_CONFIG.working_dps):
        residual = series_residual(initial_q_series(DEFAULT_PARAMS, SMOKE_CONFIG), DEFAULT_PARAMS)
        assert max(abs(value) for component in residual for value in component[:-1]) < mp.mpf("1e-20")


def test_midpoint_defect_is_stable_under_refinement() -> None:
    """Baseline and refined midpoint defects should agree componentwise."""
    baseline = solve_to_midpoint(DEFAULT_PARAMS, SMOKE_CONFIG)
    refined = solve_to_midpoint(DEFAULT_PARAMS, SMOKE_REFINED)
    digits = []
    for left, right in zip(baseline.midpoint_ydot, refined.midpoint_ydot):
        digits.append(mp.inf if left == right else -mp.log10(abs(left - right)))
    assert min(digits) > mp.mpf("3.5")
