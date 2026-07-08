"""Tests for Aloff-Wallach known-solution recovery calibration."""

from __future__ import annotations

import json
from pathlib import Path

from mpmath import mp

import experiments.aloff_wallach_recovery_calibration
from experiments.aloff_wallach import recovery_calibration as recovery
from experiments.aloff_wallach.evolution import EndpointConstants


def test_reference_constants_are_lambda_four_scaled() -> None:
    """Known homogeneous constants should be normalized by lambda_known / 4."""
    with mp.workdps(80):
        tri = recovery.reference_constants("tri_sasakian")
        squashed = recovery.reference_constants("squashed")
    assert abs(tri.A - 2**0.5 / 2) < 1e-12
    assert abs(tri.B - 0.5) < 1e-12
    assert abs(tri.D - 2**0.5 / 2) < 1e-12
    assert abs(squashed.A - 0.4242640687119285) < 1e-12
    assert abs(squashed.B - 0.6708203932499369) < 1e-12
    assert abs(squashed.D + 0.4242640687119285) < 1e-12


def test_branch_variants_are_deterministic() -> None:
    """The recovery branch lists should keep the canonical branch first."""
    assert [variant.label for variant in recovery.branch_variants("canonical")] == ["++++"]
    assert [variant.label for variant in recovery.branch_variants("paired-signs")] == ["++++", "+--+", "-++-", "----"]
    all_signs = recovery.branch_variants("all-signs")
    assert len(all_signs) == 16
    assert all_signs[0].label == "++++"


def test_recovery_seed_count_includes_left_right_branches() -> None:
    """The calibration grid is target x left-branch x right-branch."""
    assert recovery.recovery_seed_count(("tri_sasakian",), "canonical") == 1
    assert recovery.recovery_seed_count(("tri_sasakian", "squashed"), "paired-signs") == 32
    assert recovery.recovery_seed_count(("tri_sasakian",), "all-signs", (1.0, 1.25), (0.5, 1.0)) == 1024


def test_dry_run_prints_references_without_writing(tmp_path: Path, capsys, monkeypatch) -> None:
    """Dry-run should print deterministic seed metadata and avoid output files."""
    monkeypatch.setattr(recovery, "OUTPUT_DIR", tmp_path)
    recovery.main(["--dry-run", "--branch-mode", "paired-signs", "--limit", "3"])
    output = capsys.readouterr().out
    assert "known-solution recovery dry run" in output
    assert "tri_sasakian" in output
    assert "branch mode: paired-signs" in output
    assert not list(tmp_path.glob("*"))


def test_runner_writes_fake_recovery_summary(tmp_path: Path, monkeypatch) -> None:
    """The resumable JSONL runner should work with a synthetic evaluator."""
    monkeypatch.setattr(recovery, "OUTPUT_DIR", tmp_path)

    class FakeGerm:
        normal_weight = 1
        constants = EndpointConstants(1.0, 2.0, 3.0, 4.0)
        residual_norm = 1e-8
        success = True
        message = "ok"

    class FakeSide:
        status = "max_volume"
        tau = 0.5
        volume = 1.25
        volume_dot = 0.0
        message = None
        germ = FakeGerm()

    class FakeMatch:
        failure = None
        residual_norm = 1e-8
        residual = (1e-8,)
        left = FakeSide()
        right = FakeSide()

        @property
        def reconstructed_interval(self):
            return self.left.tau + self.right.tau

    def fake_match(left, right, settings):
        return FakeMatch()

    monkeypatch.setattr(recovery, "max_volume_match", fake_match)
    recovery.main(["--targets", "tri_sasakian", "--limit", "1", "--workers", "1", "--no-resume"])
    summaries = list(tmp_path.glob("*summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["classification_counts"] == {"recovered_tri_sasakian": 1}
    assert summary["best_recovery_matches"][0]["target"] == "tri_sasakian"
    assert experiments.aloff_wallach_recovery_calibration.main is recovery.main
