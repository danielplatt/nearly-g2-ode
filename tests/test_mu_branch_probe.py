"""Tests for the Berger mu/square-root branch probe."""

from __future__ import annotations

from experiments.berger_space import mu_branch_probe as probe


def test_mu_branch_probe_finds_only_the_default_mirrored_branch() -> None:
    """The global-p branch check should only find the already explored default."""
    summary = probe.build_summary(dps=60)
    compatible = summary["compatible_two_sided"]
    assert len(compatible) == 1
    assert compatible[0]["p_signs"] == [-1, 1, 1]
    assert compatible[0]["left_mu"] == -1
    assert compatible[0]["right_mu"] == -1
    assert summary["nondefault_compatible_two_sided"] == []
    assert summary["global_scout_ready"] is False


def test_mu_branch_probe_finds_the_mixed_endpoint_opposite_mu_branch() -> None:
    """The opposite-mu scout should use endpoint-local p-signs."""
    summary = probe.build_summary(dps=60)
    compatible = summary["mixed_opposite_mu_compatible"]
    assert len(compatible) == 1
    assert compatible[0]["p_signs"] == [1, 1, 1]
    assert compatible[0]["right_p_signs"] == [-1, 1, -1]
    assert compatible[0]["left_mu"] == 1
    assert compatible[0]["right_mu"] == 1
    assert summary["mixed_endpoint_scout_ready"] is True
    assert summary["scout_ready"] is True


def test_mu_branch_probe_records_opposite_mu_one_sided_obstruction() -> None:
    """The flipped branch should cancel one endpoint but not both."""
    summary = probe.build_summary(dps=60)
    left_records = summary["left_opposite_mu_one_sided"]
    right_records = summary["right_opposite_mu_one_sided"]
    assert any(record["p_signs"] == [1, 1, 1] and record["left_residual"] == "0.0" for record in left_records)
    assert any(record["p_signs"] == [-1, 1, -1] and record["right_residual"] == "0.0" for record in right_records)
    assert all(record["right_residual"] != "0.0" for record in left_records)
    assert all(record["left_residual"] != "0.0" for record in right_records)
