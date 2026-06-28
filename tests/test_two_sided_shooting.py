"""Tests for full non-mirrored two-sided shooting helpers."""

from __future__ import annotations

from mpmath import mp

from problem import DEFAULT_CONFIG, DEFAULT_PARAMS, SolverConfig
from solver.two_sided_refinement import TwoSidedNewtonSettings, two_sided_newton_refine
from solver.two_sided_shooting import (
    BASE_TWO_SIDED_POINT,
    TwoSidedSearchPoint,
    finite_difference_two_sided_jacobian,
    params_from_two_sided_scaled,
    two_sided_residual,
)


SMOKE_CONFIG = SolverConfig(4, 35, 15, mp.mpf("0.85"), 0, DEFAULT_CONFIG.match_t)
RESIDUAL_CONFIG = SolverConfig(14, 90, 35, mp.mpf("0.6"), 1, DEFAULT_CONFIG.match_t)
JACOBIAN_CONFIG = SolverConfig(4, 35, 15, mp.mpf("0.85"), 0, DEFAULT_CONFIG.match_t)


def test_two_sided_scaled_coordinates_recover_and_perturb_berger() -> None:
    """The zero point should reproduce Berger while preserving signs under perturbation."""
    with mp.workdps(80):
        params, config = params_from_two_sided_scaled(BASE_TWO_SIDED_POINT, template_config=RESIDUAL_CONFIG)
        assert abs(params.left.a - DEFAULT_PARAMS.left.a) < mp.mpf("1e-40")
        assert abs(params.right.d - DEFAULT_PARAMS.right.d) < mp.mpf("1e-40")
        assert abs(params.interval_end - 2 * config.match_t) < mp.mpf("1e-40")
        point = TwoSidedSearchPoint(mp.mpf("0.1"), mp.mpf("-0.2"), mp.mpf("0.3"), mp.mpf("-0.4"), mp.mpf("0.5"), mp.mpf("-0.6"), mp.mpf("0.2"))
        perturbed, perturbed_config = params_from_two_sided_scaled(point)
        assert perturbed.left.a > 0
        assert perturbed.left.c < 0
        assert perturbed.right.d < 0
        assert perturbed.right.f > 0
        assert perturbed_config.match_t > 0


def test_berger_full_two_sided_residual_is_small() -> None:
    """Berger should satisfy the full 8-component two-sided match."""
    with mp.workdps(RESIDUAL_CONFIG.working_dps):
        result = two_sided_residual(BASE_TWO_SIDED_POINT, RESIDUAL_CONFIG)
        assert result.failure is None
        assert result.residual_norm < mp.mpf("1e-4")
        assert len(result.residual) == 8


def test_two_sided_jacobian_shape_and_finiteness() -> None:
    """The non-mirrored Jacobian should be an 8 by 7 finite matrix."""
    with mp.workdps(JACOBIAN_CONFIG.working_dps):
        result = finite_difference_two_sided_jacobian(BASE_TWO_SIDED_POINT, JACOBIAN_CONFIG, mp.mpf("1e-4"))
        assert result.matrix.rows == 8
        assert result.matrix.cols == 7
        for row in range(result.matrix.rows):
            for col in range(result.matrix.cols):
                assert mp.isfinite(result.matrix[row, col])
        assert result.singular_values[-1] > mp.mpf("1e-12")


def test_two_sided_newton_stage_is_nonfatal_near_berger() -> None:
    """A smoke Gauss-Newton stage should improve or return a clean diagnostic."""
    settings = TwoSidedNewtonSettings("smoke", SMOKE_CONFIG, mp.mpf("1e-3"), mp.mpf("1e-6"), 1)
    point = TwoSidedSearchPoint(mp.mpf("0.02"), mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero)
    with mp.workdps(SMOKE_CONFIG.working_dps):
        report = two_sided_newton_refine(point, settings)
        assert report.status in {"max_steps", "tolerance_hit", "jacobian_failure", "no_improvement", "branch_failure"}
        assert report.final.residual_norm <= report.initial.residual_norm or report.status != "max_steps"
