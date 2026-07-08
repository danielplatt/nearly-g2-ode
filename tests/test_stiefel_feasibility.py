"""Tests for Stiefel feasibility checks."""

from __future__ import annotations

from mpmath import mp

import experiments.stiefel_feasibility
from experiments.stiefel import feasibility


def test_homogeneous_nearly_parallel_stiefel_defects_vanish() -> None:
    """The known SO(5)-homogeneous Stiefel NP conditions should be exact."""
    with mp.workdps(80):
        params = feasibility.homogeneous_nearly_parallel_parameters()
        defects = feasibility.homogeneous_np_defects(params)
        assert max(abs(value) for value in defects.values()) < mp.mpf("1e-70")


def test_natural_so4_action_on_stiefel_is_cohomogeneity_two() -> None:
    """The natural SO4 action is not compatible with the current 1D q-system."""
    assert feasibility.generic_so4_stiefel_orbit_dimension() == 5
    assert feasibility.natural_so4_stiefel_cohomogeneity() == 2
    assert feasibility.current_q_system_principal_orbit_dimension() == 6
    assert not feasibility.current_q_system_is_stiefel_ready()


def test_stiefel_summary_records_not_ready_verdict() -> None:
    """The command summary should clearly distinguish algebraic and ODE readiness."""
    summary = feasibility.build_summary()
    assert summary["version"] == "stiefel-feasibility-v1"
    assert summary["homogeneous_calibration"]["max_abs_defect"] == "0.0"
    assert summary["natural_so4_action"]["cohomogeneity"] == 2
    assert summary["current_q_system"]["cohomogeneity"] == 1
    assert summary["current_q_system"]["stiefel_ready"] is False
    assert summary["has_known_cohomogeneity_one_calibration_action"] is False
    assert summary["verdict"].startswith("not-ready")
    assert experiments.stiefel_feasibility.main is feasibility.main


def test_standard_stiefel_calibration_action_candidates_do_not_fit() -> None:
    """The standard connected homogeneous-calibration candidates should all fail."""
    candidates = feasibility.action_candidates_preserving_homogeneous_stiefel_geometry()
    by_group = {candidate.group: candidate for candidate in candidates}
    assert by_group["SO(5)"].cohomogeneity == 0
    assert by_group["SO(4) fixing a line"].cohomogeneity == 2
    assert by_group["U(2)"].dimension < 6
    assert by_group["irreducible SO(3)"].dimension < 6
    assert not feasibility.has_known_cohomogeneity_one_calibration_action()
