"""Tests for one-sided mirror shooting and local Jacobians."""

from __future__ import annotations

from mpmath import mp

from problem import DEFAULT_PARAMS, REFINED_CONFIG, SolverConfig
from solver.mirror_shooting import (
    BASE_POINT,
    MirrorSearchPoint,
    finite_difference_jacobian,
    mirror_residual,
    params_from_scaled,
)


SMOKE_JACOBIAN_CONFIG = SolverConfig(
    series_order=10,
    working_dps=70,
    target_dps=30,
    step_safety=mp.mpf("0.5"),
    sample_points=3,
    match_t=REFINED_CONFIG.match_t,
)


def test_scaled_coordinates_recover_and_perturb_berger_data() -> None:
    """Scaled coordinates should preserve the intended signs and mirror convention."""
    with mp.workdps(80):
        params, config = params_from_scaled(BASE_POINT, template_config=REFINED_CONFIG)
        assert abs(params.left.a - DEFAULT_PARAMS.left.a) < mp.mpf("1e-40")
        assert abs(params.left.c - DEFAULT_PARAMS.left.c) < mp.mpf("1e-40")
        assert abs(params.left.alpha - DEFAULT_PARAMS.left.alpha) < mp.mpf("1e-40")
        assert abs(config.match_t - REFINED_CONFIG.match_t) < mp.mpf("1e-40")
        perturbed, perturbed_config = params_from_scaled(MirrorSearchPoint(mp.mpf("0.1"), mp.mpf("-0.2"), mp.mpf("0.3"), mp.mpf("0.4")))
        assert perturbed.left.a > 0
        assert perturbed.left.c < 0
        assert perturbed_config.match_t > 0
        assert abs(perturbed.right.d + perturbed.left.a) < mp.mpf("1e-40")
        assert abs(perturbed.right.f + perturbed.left.c) < mp.mpf("1e-40")
        assert abs(perturbed.right.omega + perturbed.left.alpha) < mp.mpf("1e-40")


def test_berger_mirror_residual_is_small_with_refined_config() -> None:
    """Berger should satisfy the one-sided mirror-closing equations."""
    with mp.workdps(REFINED_CONFIG.working_dps):
        result = mirror_residual(BASE_POINT, REFINED_CONFIG)
        assert result.failure is None
        assert result.residual_norm < mp.mpf("1e-18")


def test_finite_difference_jacobian_is_finite_and_nonsingular() -> None:
    """The Berger mirror residual should have a numerically nonzero Jacobian."""
    with mp.workdps(SMOKE_JACOBIAN_CONFIG.working_dps):
        result = finite_difference_jacobian(BASE_POINT, SMOKE_JACOBIAN_CONFIG, mp.mpf("1e-4"))
        assert result.matrix.rows == 4
        assert result.matrix.cols == 4
        for row in range(4):
            for col in range(4):
                assert mp.isfinite(result.matrix[row, col])
        assert result.singular_values[-1] > mp.mpf("1e-8")
