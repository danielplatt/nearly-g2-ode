"""Tests for the Berger branch closure audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.berger_branch_audit as audit_shim
from experiments.berger_space import berger_branch_audit as audit


BRANCH_PROBE = {
    "compatible_two_sided": [
        {
            "p_signs": [-1, 1, 1],
            "left_mu": -1,
            "right_mu": -1,
            "left_residual": "0.0",
            "right_residual": "0.0",
            "left_failure": None,
            "right_failure": None,
        }
    ],
    "mixed_opposite_mu_compatible": [
        {
            "p_signs": [1, 1, 1],
            "right_p_signs": [-1, 1, -1],
            "left_mu": 1,
            "right_mu": 1,
            "left_residual": "0.0",
            "right_residual": "0.0",
            "left_failure": None,
            "right_failure": None,
        }
    ],
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _summary_for_jsonl(root: Path, jsonl_name: str) -> Path:
    path = root / jsonl_name
    return path.with_name(f"{path.stem}-summary.json")


def _base_params(*, a: str = "1", c: str = "-2", left_mu: int = -1, right_mu: int = -1, p_signs=None, right_p_signs=None) -> dict:
    payload = {
        "interval_end": "10",
        "lambda": "1",
        "left": {"a": a, "c": c, "alpha": "-0.1", "mu": left_mu},
        "right": {"d": f"{-float(a)}", "f": f"{-float(c)}", "omega": "0.1", "mu": right_mu},
    }
    if p_signs is not None:
        payload["p_signs"] = p_signs
        payload["right_p_signs"] = right_p_signs
    return payload


def _write_scout(
    root: Path,
    jsonl_name: str,
    *,
    norm: str,
    seed: int = 7,
    seed_count: int = 10,
    failures: int = 0,
    region: str = "synthetic",
    base_params: dict | None = None,
) -> Path:
    summary_path = _summary_for_jsonl(root, jsonl_name)
    _write_json(
        summary_path,
        {
            "search_version": "grid-v1",
            "scout_count": seed_count,
            "classification_counts": {"scout_success": seed_count - failures, "scout_failure": failures},
            "grid": {
                "region": region,
                "shift": "vertex",
                "bounds": [["0", "1"]],
                "axis_counts": [seed_count],
                "seed_count": seed_count,
                "full_seed_count": seed_count,
                "limit": None,
                "base_params": base_params or _base_params(),
            },
            "best_scouts": [
                {
                    "seed_index": seed,
                    "region": region,
                    "source": "test",
                    "residual_norm": norm,
                    "distance": "0.5",
                    "asymmetry": "0.25",
                    "failure": None,
                    "seed_point": {},
                }
            ],
        },
    )
    return summary_path


def _write_refinement(
    root: Path,
    summary_name: str,
    *,
    counts: dict,
    final_norm: str,
    final_s: str = "0",
    physical_t: str = "10",
    verifications: int = 0,
) -> Path:
    path = root / summary_name
    track = {
        "seed_index": 11,
        "classification": next(iter(counts)),
        "region": "synthetic",
        "scout": {"residual_norm": "0.1"},
        "stages": [{"final": {"residual_norm": final_norm, "point": {"s": final_s}}}],
        "verifications": [{} for _ in range(verifications)],
    }
    _write_json(
        path,
        {
            "refinement_version": "grid-refine-v1",
            "scout_jsonl": "output/full.jsonl",
            "selection_count": 3,
            "classified_count": 1,
            "classification_counts": counts,
            "tracks": [track],
            "best_verified_tracks": [
                {
                    "seed_index": 11,
                    "final_residual_norm": final_norm,
                    "final_point": {"s": final_s},
                    "physical_parameters": {"T": physical_t},
                    "verification_norms": [],
                }
            ],
        },
    )
    return path


def test_strict_manifest_validation_and_allow_missing(tmp_path: Path) -> None:
    """Strict mode should fail on missing canonical summaries."""
    manifest = (audit.CanonicalAuditEntry("missing", ("output/missing.jsonl",), None),)
    with pytest.raises(FileNotFoundError, match="Missing canonical Berger audit artifacts"):
        audit.build_audit(root=tmp_path, manifest=manifest, branch_probe_summary=BRANCH_PROBE)

    payload = audit.build_audit(root=tmp_path, manifest=manifest, allow_missing=True, branch_probe_summary=BRANCH_PROBE)
    assert payload["runs"] == []
    assert payload["missing_artifacts"]


def test_manifest_ignores_unlisted_smoke_runs(tmp_path: Path) -> None:
    """Only manifest-listed artifacts should contribute to the audit facts."""
    _write_scout(tmp_path, "output/full.jsonl", norm="0.4", seed=22, seed_count=100)
    _write_scout(tmp_path, "output/smoke.jsonl", norm="1e-9", seed=1, seed_count=8)
    manifest = (audit.CanonicalAuditEntry("full", ("output/full.jsonl",), None),)

    payload = audit.build_audit(root=tmp_path, manifest=manifest, branch_probe_summary=BRANCH_PROBE)
    run = payload["runs"][0]
    assert run["best_scout_residual"] == "0.4"
    assert run["best_scout_seed"] == 22
    assert run["total_scout_count"] == 100
    assert run["outcome"] == "no-low-scout-signal"


def test_refinement_extraction_and_best_final_residual(tmp_path: Path) -> None:
    """Refinement summaries should expose counts, verification totals, and best final residuals."""
    _write_scout(tmp_path, "output/full.jsonl", norm="0.05")
    _write_refinement(
        tmp_path,
        "output/refine-summary.json",
        counts={"failed": 1},
        final_norm="0.0125",
        final_s="0",
        verifications=2,
    )
    manifest = (audit.CanonicalAuditEntry("refined", ("output/full.jsonl",), "output/refine-summary.json"),)

    run = audit.build_audit(root=tmp_path, manifest=manifest, branch_probe_summary=BRANCH_PROBE)["runs"][0]
    assert run["refinement"]["selection_count"] == 3
    assert run["refinement"]["classification_counts"] == {"failed": 1}
    assert run["refinement"]["best_final_residual"] == "0.0125"
    assert run["refinement"]["verification_count"] == 2
    assert run["outcome"] == "finite-residual-tail"


def test_branch_descriptor_derives_signs_mu_and_p_choices(tmp_path: Path) -> None:
    """The physical branch descriptor should come from persisted base parameters."""
    base = _base_params(a="1", c="2", left_mu=1, right_mu=1, p_signs=[1, 1, 1], right_p_signs=[-1, 1, -1])
    _write_scout(tmp_path, "output/full.jsonl", norm="0.3", base_params=base)
    manifest = (audit.CanonicalAuditEntry("branch", ("output/full.jsonl",), None),)

    branch = audit.build_audit(root=tmp_path, manifest=manifest, branch_probe_summary=BRANCH_PROBE)["runs"][0]["branch"]
    assert branch["a_sign"] == ">0"
    assert branch["c_sign"] == ">0"
    assert branch["left_mu"] == 1
    assert branch["right_mu"] == 1
    assert branch["p_signs"] == [1, 1, 1]
    assert branch["right_p_signs"] == [-1, 1, -1]


def test_outcome_labels_cover_recovered_not_refined_and_collapsed(tmp_path: Path) -> None:
    """Synthetic summaries should exercise the nontrivial audit outcome labels."""
    _write_scout(tmp_path, "output/recovered.jsonl", norm="0.02")
    _write_refinement(tmp_path, "output/recovered-refine-summary.json", counts={"recovered_berger": 2}, final_norm="1e-12")
    _write_scout(tmp_path, "output/not-refined.jsonl", norm="0.02")
    _write_scout(tmp_path, "output/collapsed.jsonl", norm="0.05")
    _write_refinement(
        tmp_path,
        "output/collapsed-refine-summary.json",
        counts={"failed": 1},
        final_norm="0.003",
        final_s="-3",
        physical_t="0.2",
    )
    manifest = (
        audit.CanonicalAuditEntry("recovered", ("output/recovered.jsonl",), "output/recovered-refine-summary.json"),
        audit.CanonicalAuditEntry("not-refined", ("output/not-refined.jsonl",), None),
        audit.CanonicalAuditEntry("collapsed", ("output/collapsed.jsonl",), "output/collapsed-refine-summary.json"),
    )

    runs = audit.build_audit(root=tmp_path, manifest=manifest, branch_probe_summary=BRANCH_PROBE)["runs"]
    assert {run["label"]: run["outcome"] for run in runs} == {
        "recovered": "berger-recovered",
        "not-refined": "not-refined",
        "collapsed": "collapsed-tail",
    }


def test_markdown_rendering_contains_sections_and_conclusion(tmp_path: Path) -> None:
    """The rendered report should contain the durable audit sections."""
    _write_scout(tmp_path, "output/full.jsonl", norm="0.4")
    manifest = (audit.CanonicalAuditEntry("full", ("output/full.jsonl",), None),)
    payload = audit.build_audit(root=tmp_path, manifest=manifest, branch_probe_summary=BRANCH_PROBE)

    rendered = audit.render_markdown(payload)
    assert "# Berger Branch Closure Audit" in rendered
    assert "## Canonical Runs" in rendered
    assert "## Branch Coverage" in rendered
    assert "## Conclusion" in rendered
    assert "No non-Berger nearly G2 candidate survived" in rendered


def test_top_level_shim_imports_audit_module() -> None:
    """The top-level compatibility shim should expose the implementation module."""
    assert audit_shim.AUDIT_VERSION == audit.AUDIT_VERSION
