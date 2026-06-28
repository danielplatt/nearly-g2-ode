"""Tests for numerical S7 right-germ helpers."""

from __future__ import annotations

from mpmath import mp

from experiments.s7.full_moduli_firstjet_scout import scout_seed_count
from experiments.s7.right_germ import (
    S7RightGermPoint,
    offset_moduli_from_point,
    solve_right_firstjet_germ,
    solve_right_offset_moduli_germ,
)
from experiments.s7.right_moduli_chart import p2_offset_defect, p3_offset_defect


def test_zero_firstjet_germ_returns_exact_homogeneous_series() -> None:
    """The zero first-jet germ should reuse the exact known S7 right series."""
    with mp.workdps(80):
        solution = solve_right_firstjet_germ("round", S7RightGermPoint(mp.zero, mp.zero, mp.zero), order=6)
        assert solution.success
        assert solution.residual_norm == 0
        assert solution.evaluations == 0
        assert solution.fixed_right.series_coefficients is not None
        assert len(solution.fixed_right.series_coefficients.y1) == 7


def test_full_moduli_firstjet_scout_count() -> None:
    """The default 7D first-jet grid count should be deterministic."""
    assert scout_seed_count(("round", "squashed"), 4) == 32768


def test_zero_offset_moduli_germ_returns_exact_homogeneous_series() -> None:
    """The zero offset-moduli germ should reuse the exact known S7 right series."""
    with mp.workdps(80):
        solution = solve_right_offset_moduli_germ("round", S7RightGermPoint(mp.zero, mp.zero, mp.zero), order=6)
        assert solution.success
        assert solution.residual_norm == 0
        assert solution.evaluations == 0
        assert p3_offset_defect(solution.fixed_right.offset) == 0


def test_small_offset_moduli_germs_solve_sampled_equations() -> None:
    """Small p2/p3 terminal-offset perturbations should have solved Taylor germs."""
    with mp.workdps(80):
        point = S7RightGermPoint(mp.mpf("0.01"), mp.mpf("-0.02"), mp.mpf("0.015"))
        round_solution = solve_right_offset_moduli_germ("round", point, order=6)
        squashed_solution = solve_right_offset_moduli_germ("squashed", point, order=6)

        assert round_solution.success
        assert squashed_solution.success
        assert round_solution.residual_norm < mp.mpf("1e-8")
        assert squashed_solution.residual_norm < mp.mpf("1e-8")
        assert p3_offset_defect(round_solution.fixed_right.offset) == 0
        assert p2_offset_defect(squashed_solution.fixed_right.offset) == 0
        assert offset_moduli_from_point("round", point) == round_solution.fixed_right.offset
        assert offset_moduli_from_point("squashed", point) == squashed_solution.fixed_right.offset
