"""Tests for the S7 SU(2)^3 scalar defect audit."""

from __future__ import annotations

import json

import experiments.s7_su2_cubed_defect_audit
from experiments.s7 import su2_cubed_defect_audit as audit


def test_defect_registry_contains_shortlist() -> None:
    """The first-pass registry should include the handoff shortlist."""
    names = {spec.name for spec in audit.defect_specs()}
    assert {
        "D_x3",
        "D_x2",
        "D_C",
        "D_x3_C_norm2",
        "D_x3_x2_norm2",
        "D_S1",
        "D_S3",
        "D_W_over_b",
        "D_C_IF",
        "D_3_IF",
    } <= names
    assert "D_C_plus_p1_x3" in names
    assert "D_x3_plus_m10_x2" in names


def test_known_compact_crossings_make_endpoint_defects_small() -> None:
    """The direct round and squashed values should calibrate endpoint defects."""
    for a in (-36.0, 108.0 / 5.0):
        sample = audit.integrate_terminal_sample("exact", a, step_size=1e-3)
        assert sample.status == "crossed"
        rows = {row["defect_name"]: row for row in audit.evaluate_defects(sample)}
        assert rows["D_x3"]["abs_value"] < 1e-3
        assert rows["D_x2"]["abs_value"] < 1e-3
        assert rows["D_x3_C_norm2"]["abs_value"] < 1e-6


def test_limiting_sample_gives_large_x3_and_s1_defects() -> None:
    """The b=0 limiting crossing should strongly miss the terminal conditions."""
    sample = audit.integrate_terminal_sample("limit", step_size=1e-3)
    rows = {row["defect_name"]: row for row in audit.evaluate_defects(sample)}
    assert sample.status == "crossed"
    assert rows["D_x3"]["value"] < -0.8
    assert rows["D_S1"]["value"] < -0.5
    assert rows["D_x3_C_norm2"]["value"] > 0.5


def test_audit_selects_structurally_distinct_top_candidates() -> None:
    """Top candidates should avoid selecting only rescaled x3 duplicates."""
    report = audit.run_audit(
        (-36.0, 108.0 / 5.0, -250.0, 250.0),
        step_size=1e-3,
        barrier_grid_subdivisions=2,
    )
    selected = [item["defect_name"] for item in report["selected_top_candidates"]]
    assert selected == ["D_x3", "D_x3_C_norm2", "D_S1"]
    assert report["asymptotic_reductions"]["status"] == "reduced_to_singular_endpoint_convergence"
    assert report["dx3_asymptotic_tail_proof"]["status"] == "terminal_tail_bound"
    assert report["uniform_exclusion_attempt"]["barrier_report"]["status"] == "scalar_margins_positive"


def test_dx3_asymptotic_tail_proof_bounds_endpoint_change() -> None:
    """The p-time terminal layer should keep the limiting x3 endpoint negative."""
    report = audit.dx3_asymptotic_tail_proof_report()
    assert report["status"] == "terminal_tail_bound"
    assert report["regularized_p_prime_coefficient_bounds"][1] < 0.0
    assert report["tail_variation_bounds"]["x3"] < 1e-4
    assert report["x3_endpoint_interval_from_limit_p0"][1] < -1.0
    assert all(item["inside_box"] for item in report["sample_states"] if item["status"] == "sampled")


def test_render_markdown_contains_reductions() -> None:
    """The rendered report should expose the ranking and proof-reduction sections."""
    report = audit.run_audit(
        (-36.0, 108.0 / 5.0, -250.0, 250.0),
        step_size=1e-3,
        barrier_grid_subdivisions=2,
    )
    markdown = audit.render_markdown(report)
    assert "# S7 SU(2)^3 Defect Audit" in markdown
    assert "## Infinity Reduction" in markdown
    assert "### D_x3 Terminal Tail Proof" in markdown
    assert "`D_x3`" in markdown
    assert "support-entry/containment lemma" in markdown


def test_cli_writes_json_markdown_and_csv(tmp_path) -> None:
    """The CLI should write reusable audit artifacts."""
    json_path = tmp_path / "audit.json"
    markdown_path = tmp_path / "audit.md"
    csv_path = tmp_path / "audit.csv"
    audit.main(
        [
            "--a-values=-36,21.6,-250,250",
            "--step-size",
            "0.001",
            "--barrier-grid-subdivisions",
            "2",
            "--write-json",
            str(json_path),
            "--write-markdown",
            str(markdown_path),
            "--write-csv",
            str(csv_path),
        ]
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["version"] == audit.DEFECT_AUDIT_VERSION
    assert markdown_path.read_text(encoding="utf-8").startswith("# S7 SU(2)^3 Defect Audit")
    assert "defect_name" in csv_path.read_text(encoding="utf-8")


def test_top_level_shim_exports_impl() -> None:
    """The top-level module should be a compatibility shim."""
    assert experiments.s7_su2_cubed_defect_audit.main is audit.main
