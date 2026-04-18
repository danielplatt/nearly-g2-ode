"""Tests for the q-system geometry and stored singular-end data."""

from __future__ import annotations

from mpmath import mp

from problem import DEFAULT_PARAMS, RHO, m_minus_one_residual, q_to_y, y_first_jet, y_to_q, y_zero_jet


def test_zero_and_first_jet_match_the_verified_branch() -> None:
    """The stored singular-end data should match the checked local branch."""
    with mp.workdps(80):
        sqrt5 = mp.sqrt(5)
        sqrt15 = mp.sqrt(15)
        y0 = y_zero_jet(DEFAULT_PARAMS)
        y1 = y_first_jet(DEFAULT_PARAMS)
        assert abs(y0.y1 - 9 * sqrt5 / 100) < mp.mpf("1e-40")
        assert abs(y0.y5 - 23 * sqrt5 / 100) < mp.mpf("1e-40")
        assert abs(y0.y6 - 2 * sqrt15 / 25) < mp.mpf("1e-40")
        assert y1 == DEFAULT_PARAMS.alpha * RHO


def test_weighted_maps_round_trip_pointwise() -> None:
    """The pointwise weighted maps should be mutually inverse away from t=0."""
    with mp.workdps(80):
        t = mp.mpf("0.1")
        y = y_zero_jet(DEFAULT_PARAMS) + mp.mpf("0.3") * y_first_jet(DEFAULT_PARAMS)
        q = y_to_q(t, y, DEFAULT_PARAMS)
        recovered = q_to_y(t, q, DEFAULT_PARAMS)
        assert max(abs(left - right) for left, right in zip(y, recovered)) < mp.mpf("1e-40")


def test_m_minus_one_residual_vanishes_for_the_stored_zero_jet() -> None:
    """The stored base branch should cancel the q-derived singular residual."""
    with mp.workdps(80):
        residual = m_minus_one_residual(DEFAULT_PARAMS)
        assert max(abs(value) for value in residual) < mp.mpf("1e-30")
