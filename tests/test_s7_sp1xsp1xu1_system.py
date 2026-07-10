"""Tests for the Sp(1) x Sp(1) x U(1) S7 system."""

from __future__ import annotations

from mpmath import mp

import experiments.s7_sp1xsp1xu1_system
from experiments.s7 import sp1xsp1xu1_system as system
from experiments.s7 import su2_cubed_action_audit


def test_invariant_basis_and_podesta_embedding_are_recorded() -> None:
    """The new chart should have the census-predicted 5+8 functions."""
    summary = system.invariant_basis_summary()
    assert len(summary["omega_basis"]) == 5
    assert len(summary["gamma_basis"]) == 8
    assert summary["maurer_cartan"]["da1"] == "6 a2^a3"
    assert summary["podesta_embedding"]["omega"] == "f0*(a3b3 + delta)"
    assert "f3*(b3a12 + a3_epsilon)" in summary["podesta_embedding"]["gamma"]
    constraints = summary["polynomial_constraints"]
    assert constraints["domega_components"]["a3b12"] == "-6*x3"
    assert "-lambda*x1*x3 = 0" in constraints["d_gamma_minus_lambda_omega2_over_2"]
    assert "x1=x2=0" in constraints["regular_branch_note"]
    assert "six algebraic degrees" in constraints["regular_branch_algebraic_dimension"]
    assert experiments.s7_sp1xsp1xu1_system.main is system.main


def test_maurer_cartan_scale_recovers_podesta_algebraic_constraint() -> None:
    """The exterior algebra should induce f3+f4+lambda*f0^2/6=0."""
    f0 = mp.mpf("1.7")
    lam = mp.mpf("2.3")
    f1 = mp.mpf("0.4")
    f2 = mp.mpf("-0.8")
    f3 = mp.mpf("0.9")
    f4 = -f3 - lam * f0**2 / 6
    state = system.podesta_state_from_f(f0, f1, f2, f3, f4)
    residual = system.algebraic_residual(state, lam)
    assert residual["d_gamma_minus_lambda_omega2_over_2"] < mp.mpf("1e-40")


def test_round_and_squashed_targets_satisfy_full_u1_system() -> None:
    """Both homogeneous S7 targets should be recovered in the larger chart."""
    with mp.workdps(80):
        for target in (
            su2_cubed_action_audit.round_target(),
            su2_cubed_action_audit.squashed_target(),
        ):
            embedded = system.embedded_target(target)
            rows = system.target_residuals(embedded, (mp.mpf("0.37"), mp.mpf("0.91")))
            assert max(mp.mpf(row["max_abs_residual"]) for row in rows) < mp.mpf("1e-60")


def test_hitchin_dual_orientation_is_selected_by_omega_volume() -> None:
    """The squashed target needs the opposite Hitchin-dual branch from round."""
    with mp.workdps(80):
        signs = {}
        for target in (
            su2_cubed_action_audit.round_target(),
            su2_cubed_action_audit.squashed_target(),
        ):
            embedded = system.embedded_target(target)
            state = tuple(function(mp.mpf("0.37")) for function in embedded.state_functions)
            omega, gamma = system.state_omega_gamma(state)
            raw_hat = system.hitchin_dual(gamma)
            oriented_hat = system.oriented_hitchin_dual(omega, gamma)
            raw_volume = system.volume_coefficient(system.wedge(gamma, raw_hat))
            oriented_volume = system.volume_coefficient(system.wedge(gamma, oriented_hat))
            signs[target.name] = mp.sign(oriented_volume / raw_volume)
            assert abs(oriented_volume - 4 * system.omega_volume(omega)) < mp.mpf("1e-60")
        assert signs == {"round": 1, "squashed": -1}


def test_endpoint_weight_summary_contains_podesta_subchart() -> None:
    """Endpoint data should expose the smooth weights and known subfamily."""
    weights = system.endpoint_weight_table()
    assert weights["left_K_plus"]["collapsing"] == "a1,a2,a3"
    assert "y6=Y6" in weights["left_K_plus"]["regular_variables"]
    assert "y1=Y1" in weights["right_K_minus"]["regular_variables"]
    assert "y2=y8=f4" in weights["left_K_plus"]["podesta_subchart"]
    dimensions = system.endpoint_jet_dimensions(4)
    assert [row["allowed_dimension"] for row in dimensions if row["endpoint"] == "left_K_plus"] == [1, 1, 5, 9, 17]
