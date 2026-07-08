"""Tests for the S7 SU(2)^3 next-sprint audit."""

from __future__ import annotations

import json

import experiments.s7_su2_cubed_next_sprint_audit
from experiments.s7 import su2_cubed_next_sprint_audit as audit


def test_psection_audit_selects_proof_friendly_cone() -> None:
    """The p=0.33 section should use the old proof-friendly K=1.23 wall."""
    report = audit.run_audit(step_size=0.001)
    psection = report["psection_audit"]
    chosen = psection["chosen_section"]
    assert chosen["p"] == 0.33
    assert chosen["status"] == "promising"
    assert chosen["best_K"] >= 2.0
    assert chosen["recommended_cone"]["c_lower"] == 1.23
    assert psection["normalized_c_cone"]["status"] == "promising"
    assert psection["finite_b_event_map_stability"]["status"] == "inconclusive, with exact blocker"


def test_cif_route_is_ranked_above_incomplete_psection_route() -> None:
    """The route ranking should not call p-section complete without event stability."""
    report = audit.run_audit(step_size=0.001)
    recommendations = report["recommendations"]
    assert recommendations[0]["route"] == "D_C_IF integral route"
    assert recommendations[0]["status"] == "promising"
    assert recommendations[1]["route"] == "p-section cone route"
    assert recommendations[1]["status"] == "inconclusive, with exact blocker"
    assert report["CIF_integral_audit"]["limit_summary"]["Itotal_at_p_0.33"] > 1.0


def test_l_formula_matches_chain_rule_on_sections() -> None:
    """The structured L derivative formula should match chain-rule sampling."""
    report = audit.run_audit(step_size=0.001)
    l_audit = report["L_scalar_audit"]
    assert l_audit["max_formula_chain_rule_error"] < 1e-5
    assert l_audit["status"] in {"promising", "not promising"}


def test_cli_writes_requested_artifacts(tmp_path) -> None:
    """The CLI should write the sprint report and the three requested JSON files."""
    output_dir = tmp_path / "out"
    report_path = tmp_path / "report.md"
    audit.main(["--step-size", "0.001", "--output-dir", str(output_dir), "--report-path", str(report_path)])
    assert report_path.read_text(encoding="utf-8").startswith("# S7 SU(2)^3 Next-Sprint Audit")
    for name in ("psection_audit.json", "CIF_integral_audit.json", "terminal_separator_audit.json"):
        payload = json.loads((output_dir / name).read_text(encoding="utf-8"))
        assert payload["version"] == audit.NEXT_SPRINT_AUDIT_VERSION


def test_top_level_shim_exports_impl() -> None:
    """The top-level module should be a compatibility shim."""
    assert experiments.s7_su2_cubed_next_sprint_audit.main is audit.main
