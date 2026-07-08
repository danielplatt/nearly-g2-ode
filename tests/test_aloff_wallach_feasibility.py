"""Tests for Aloff-Wallach feasibility checks."""

from __future__ import annotations

import experiments.aloff_wallach_feasibility
from experiments.aloff_wallach import feasibility


def test_generic_aloff_wallach_has_no_endpoint_volume_candidate() -> None:
    """The generic homogeneous SU3 setup is transitive or too small."""
    candidates = feasibility.generic_aloff_wallach_action_candidates()
    by_group = {candidate.group: candidate for candidate in candidates}
    assert by_group["SU(3)"].cohomogeneity == 0
    assert by_group["S(U(2)U(1))"].dimension < 6
    assert by_group["SO(3)"].dimension < 6
    assert by_group["T^2"].dimension < 6
    assert not feasibility.generic_spaces_have_endpoint_volume_candidate()


def test_n11_product_action_is_cohomogeneity_one() -> None:
    """The exceptional N11 product action should give six-dimensional orbits."""
    assert feasibility.cp2_real_so3_generic_orbit_dimension() == 3
    assert feasibility.n11_fiber_group_dimension() == 3
    assert feasibility.n11_product_action_generic_orbit_dimension() == 6
    assert feasibility.n11_product_action_cohomogeneity() == 1
    assert feasibility.n11_has_endpoint_volume_candidate()


def test_n11_candidate_is_promising_but_not_current_q_system_ready() -> None:
    """The audit should distinguish viability from current q-system readiness."""
    candidates = feasibility.n11_action_candidates()
    by_group = {candidate.group: candidate for candidate in candidates}
    product = by_group["SO(3)_real x SO(3)_fiber"]
    assert product.dimension == 6
    assert product.generic_stabilizer_dimension == 0
    assert product.cohomogeneity == 1
    assert product.verdict == "viable-new-ode-candidate"
    assert not product.current_q_system_ready


def test_aloff_wallach_summary_records_the_recommended_next_step() -> None:
    """The command summary should make the N11 next step explicit."""
    summary = feasibility.build_summary()
    assert summary["version"] == "aloff-wallach-feasibility-v1"
    assert summary["generic_Nkl"]["endpoint_volume_candidate"] is False
    assert summary["N11"]["endpoint_volume_candidate"] is True
    assert summary["N11"]["product_action_cohomogeneity"] == 1
    assert summary["current_q_system"]["aloff_wallach_ready"] is False
    assert "derive the SO(3)_real x SO(3)_fiber invariant forms" in summary["recommended_next_step"]
    assert experiments.aloff_wallach_feasibility.main is feasibility.main

