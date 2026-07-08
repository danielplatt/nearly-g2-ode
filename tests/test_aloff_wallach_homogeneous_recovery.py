"""Tests for direct Aloff-Wallach homogeneous recovery."""

from __future__ import annotations

import experiments.aloff_wallach_homogeneous_recovery
from experiments.aloff_wallach import homogeneous_recovery as recovery
from experiments.aloff_wallach.evolution import AWSettings, algebraic_residual


def test_squashed_trajectory_satisfies_action_scaled_constraint() -> None:
    """The derived product-action scales should make the squashed path algebraic."""
    trajectory = recovery.homogeneous_trajectory("squashed")
    state = recovery.trajectory_state(trajectory, 0.3)
    residual = algebraic_residual(
        state,
        4.0,
        None,
        base_structure_scale=recovery.BASE_STRUCTURE_SCALE,
        fiber_structure_scale=recovery.FIBER_STRUCTURE_SCALE,
    )
    assert max(abs(value) for value in residual) < 1e-10


def test_homogeneous_recovery_recovers_squashed_and_rejects_tri_sasakian() -> None:
    """The exact-trajectory recovery should recover only action-invariant targets."""
    settings = AWSettings(max_step=0.01, rtol=1e-8, atol=1e-10)
    squashed = recovery.recover_target("squashed", settings, epsilon=2e-2, match_fraction=0.5)
    tri = recovery.recover_target("tri_sasakian", settings, epsilon=2e-2, match_fraction=0.5)
    assert squashed["classification"] == "recovered_homogeneous"
    assert squashed["match_error"] < 1e-6
    assert tri["classification"] == "not_invariant_under_fiber_action"
    assert experiments.aloff_wallach_homogeneous_recovery.main is recovery.main
