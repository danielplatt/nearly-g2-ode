"""Tests for the S7 SU(2)^3 action audit."""

from __future__ import annotations

from mpmath import mp

import experiments.s7_su2_cubed_action_audit
from experiments.s7 import su2_cubed_action_audit as audit


def test_group_diagram_has_s7_cohomogeneity_one_dimensions() -> None:
    """The SU2^3 action should have 6D principal and 3D singular orbits."""
    diagram = audit.group_diagram()
    assert diagram["cohomogeneity"] == 1
    assert diagram["dimensions"]["manifold"] == 7
    assert diagram["dimensions"]["principal_orbit"] == 6
    assert diagram["dimensions"]["singular_orbit"] == 3
    assert diagram["dimensions"]["slice"] == 4
    assert diagram["principal_orbit"] == "G/H ~= S3 x S3"


def test_invariant_form_basis_is_podesta_five_function_chart() -> None:
    """Podesta's chart should expose one invariant 2-form and five 3-form components."""
    basis = audit.invariant_form_basis()
    assert basis["invariant_2_forms_on_principal_orbit"] == ["omega = e25 + e36 + e47"]
    assert len(basis["invariant_3_form_basis"]) == 5
    assert "phi = f0" in basis["general_invariant_3_form"]


def test_endpoint_smoothness_records_left_and_right_conditions() -> None:
    """Both singular S3 endpoints should be represented in the audit."""
    conditions = audit.endpoint_smoothness_conditions()
    assert "6 f0'(0)=f3''(0)" in conditions["left_K_plus"]["derivative_conditions"]
    assert "f2(0) f0'(0) < 0" in conditions["left_K_plus"]["nondegeneracy"]
    assert "g1(s)=f2(pi/2-s)" in conditions["right_K_minus"]["transform_to_left_conditions"]


def test_known_solutions_satisfy_np_system_and_endpoint_smoothness() -> None:
    """Round and squashed S7 should pass the chart smoke checks."""
    with mp.workdps(audit.DEFAULT_DPS):
        for target in (audit.round_target(), audit.squashed_target()):
            for t in (mp.mpf("0.37"), mp.mpf("0.91")):
                residuals = audit.np_residuals(target, t)
                assert max(abs(value) for value in residuals.values()) < mp.mpf("1e-60")
            for side in ("left", "right"):
                residuals = audit.endpoint_residuals(target, side)
                assert abs(residuals["g1_value"]) < mp.mpf("1e-60")
                assert abs(residuals["g3_value"]) < mp.mpf("1e-60")
                assert abs(residuals["g4_value"]) < mp.mpf("1e-60")
                assert abs(residuals["g1_second"]) < mp.mpf("1e-60")
                assert abs(residuals["six_g0_prime_minus_g3_second"]) < mp.mpf("1e-60")
                assert residuals["nondegenerate_product"] < 0


def test_summary_and_top_level_shim() -> None:
    """The audit summary should be import-safe and mark the smoke test passed."""
    summary = audit.build_summary()
    assert summary["version"] == audit.AUDIT_VERSION
    assert summary["topology"]["verified"] is True
    assert summary["smoke_status"]["passed"] is True
    assert summary["verdict"].startswith("new-viable-s7-action")
    assert experiments.s7_su2_cubed_action_audit.main is audit.main
