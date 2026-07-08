"""Tests for the Berger S3-bundle action audit."""

from __future__ import annotations

import experiments.berger_s3_bundle_action_audit
from experiments.berger_space import s3_bundle_action_audit as audit


def test_berger_topology_matches_grove_ziller_sphere_bundle() -> None:
    """The Berger bundle invariants should match the GZ M_{6,4} model."""
    ce_model = audit.berger_crowley_escher_model()
    gz_model = audit.crowley_escher_to_grove_ziller(ce_model)
    assert ce_model.euler_class == 10
    assert ce_model.tangent_p1_mod_euler == 6
    assert (gz_model.k, gz_model.l) == (6, 4)
    assert gz_model.euler_class == 10
    assert gz_model.tangent_p1_mod_euler == 6


def test_grove_ziller_slopes_realize_berger_bundle_labels() -> None:
    """The small deterministic GZ slopes should realize k=6 and l=4."""
    slopes = audit.slopes_for_grove_ziller_bundle(audit.GroveZillerBundle(6, 4))
    assert slopes.all_congruent_one_mod_four()
    assert slopes.k == 6
    assert slopes.l == 4
    assert (slopes.p_minus**2 - slopes.p_plus**2) // 8 == 6
    assert -(slopes.q_minus**2 - slopes.q_plus**2) // 8 == 4


def test_associated_action_is_not_cohomogeneity_one() -> None:
    """The induced action on the Berger-sized bundle has 3D generic orbits."""
    summary = audit.build_summary()
    associated = summary["associated_s3_bundle_action"]
    assert associated["dimension"] == 7
    assert associated["principal_orbit_dimension"] == 3
    assert associated["cohomogeneity"] == 4
    assert associated["g2_endpoint_problem"] is False
    assert "D10" in associated["orbit_types"]


def test_principal_action_is_cohomogeneity_one_but_wrong_dimension() -> None:
    """GZ supplies a coh1 principal-bundle action, not a 7D G2 action."""
    summary = audit.build_summary()
    principal = summary["principal_so4_bundle_action"]
    assert principal["dimension"] == 10
    assert principal["cohomogeneity"] == 1
    assert principal["g2_target"] is False
    assert summary["endpoint_smoothness"]["status"] == "not_applicable_to_new_7d_action"


def test_top_level_shim_imports_impl() -> None:
    """The top-level module should expose the implementation module."""
    assert experiments.berger_s3_bundle_action_audit.main is audit.main
    assert experiments.berger_s3_bundle_action_audit.AUDIT_VERSION == audit.AUDIT_VERSION
