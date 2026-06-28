"""Tests for the two-sided weighted Taylor seeds and midpoint matcher."""

from __future__ import annotations

from mpmath import mp

from problem import (
    DEFAULT_CONFIG,
    DEFAULT_PARAMS,
    LEFT_CHART,
    RIGHT_CHART,
    SolverConfig,
    initial_left_series,
    initial_right_series,
    left_first_jet,
    left_zero_jet,
    right_first_jet,
    right_zero_jet,
    weighted_series_residual,
)
from experiments.doubled_sphere import build_params as doubled_sphere_params
from solver import solve_two_sided


SMOKE_CONFIG = SolverConfig(
    series_order=12,
    working_dps=80,
    target_dps=30,
    step_safety=mp.mpf("0.5"),
    sample_points=5,
    match_t=DEFAULT_CONFIG.match_t,
)
SMOKE_REFINED = SolverConfig(
    series_order=16,
    working_dps=100,
    target_dps=40,
    step_safety=mp.mpf("0.5"),
    sample_points=5,
    match_t=DEFAULT_CONFIG.match_t,
)


def _max_abs(state) -> mp.mpf:
    """Return the largest absolute component in one state."""
    return max(abs(value) for value in state)


def test_initial_weighted_series_recover_the_prescribed_endpoint_jets() -> None:
    """The singular weighted series seeds should reproduce both endpoint jets."""
    with mp.workdps(SMOKE_CONFIG.working_dps):
        for params in (DEFAULT_PARAMS, doubled_sphere_params()):
            left_coeffs = initial_left_series(params, SMOKE_CONFIG)
            right_coeffs = initial_right_series(params, SMOKE_CONFIG)
            assert _max_abs(left_coeffs.map(lambda coeffs: coeffs[0]) - left_zero_jet(params)) < mp.mpf("1e-25")
            assert _max_abs(right_coeffs.map(lambda coeffs: coeffs[0]) - right_zero_jet(params)) < mp.mpf("1e-25")
            assert _max_abs(left_coeffs.map(lambda coeffs: coeffs[1]) - left_first_jet(params)) < mp.mpf("1e-20")
            assert _max_abs(right_coeffs.map(lambda coeffs: coeffs[1]) - right_first_jet(params)) < mp.mpf("1e-20")


def test_initial_weighted_series_satisfy_the_weighted_equations_to_truncation() -> None:
    """The left and right singular seeds should solve their weighted systems."""
    with mp.workdps(SMOKE_CONFIG.working_dps):
        for params in (DEFAULT_PARAMS, doubled_sphere_params()):
            left_residual = weighted_series_residual(LEFT_CHART, initial_left_series(params, SMOKE_CONFIG), mp.zero, params)
            right_residual = weighted_series_residual(RIGHT_CHART, initial_right_series(params, SMOKE_CONFIG), mp.zero, params)
            assert max(abs(value) for component in left_residual for value in component[:-1]) < mp.mpf("1e-20")
            assert max(abs(value) for component in right_residual for value in component[:-1]) < mp.mpf("1e-20")


def test_two_sided_match_and_l_value_are_stable_under_refinement() -> None:
    """Midpoint q-matching and l(pi/6) should be stable under refinement."""
    baseline = solve_two_sided(DEFAULT_PARAMS, SMOKE_CONFIG)
    refined = solve_two_sided(DEFAULT_PARAMS, SMOKE_REFINED)
    mismatch_digits = []
    for left, right in zip(baseline.mismatch_q, refined.mismatch_q):
        mismatch_digits.append(mp.inf if left == right else -mp.log10(abs(left - right)))
    assert min(mismatch_digits) > mp.mpf("3.5")
    l_digits = []
    for left, right in ((baseline.left_l, refined.left_l), (baseline.right_l, refined.right_l)):
        l_digits.append(mp.inf if left == right else -mp.log10(abs(left - right)))
    assert min(l_digits) > mp.mpf("3.5")
