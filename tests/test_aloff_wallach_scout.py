"""Tests for the endpoint-reduced Aloff-Wallach scout runner."""

from __future__ import annotations

import json
from pathlib import Path

from mpmath import mp

import experiments.aloff_wallach_scout
from experiments.aloff_wallach import scout
from experiments.aloff_wallach.evolution import (
    AWSettings,
    EndpointConstants,
    endpoint_zero_state,
    hitchin_dual,
    subtract_forms,
    wedge,
)


def test_scout_grid_is_endpoint_reduced_eight_dimensional() -> None:
    """The honest scout should expose four smooth constants at each endpoint."""
    axes = scout.scout_axes(mp.mpf("1"), mp.mpf("1"), "vertex")
    assert len(axes) == 8
    assert [len(axis) for axis in axes] == [3] * 8
    assert scout.scout_seed_count(mp.mpf("1"), mp.mpf("1"), "vertex") == 3**8
    first_seed = scout.scout_seeds(mp.mpf("1"), mp.mpf("1"), "vertex", limit=1)[0]
    left, right = scout._endpoint_constants(first_seed.point)
    assert left.B * left.C < 0
    assert right.B != 0 or right.C != 0
    assert left.A == left.D == right.A == right.D == 0
    assert scout.COORDINATE_NAMES == (
        "left_A",
        "left_B",
        "left_C",
        "left_D",
        "right_A",
        "right_B",
        "right_C",
        "right_D",
    )


def test_endpoint_zero_state_matches_smoothness_parameterization() -> None:
    """The four constants should expand to the derived zero-order chart."""
    state = endpoint_zero_state(EndpointConstants(1.0, 2.0, 3.0, 4.0))
    assert list(state[:7]) == [0.0] * 7
    assert list(state[7:]) == [1.0, -1.0, 2.0, 3.0, 2.0, 3.0, -3.0, 2.0, -3.0, 2.0, 4.0, -4.0]


def test_hitchin_dual_matches_standard_su3_sign() -> None:
    """The ODE layer should use the usual Im(Omega) sign."""
    gamma = {
        (1, 3, 5): 1.0,
        (1, 4, 6): -1.0,
        (2, 3, 6): -1.0,
        (2, 4, 5): -1.0,
    }
    expected_hat = {
        (1, 3, 6): 1.0,
        (1, 4, 5): 1.0,
        (2, 3, 5): 1.0,
        (2, 4, 6): -1.0,
    }
    residual = subtract_forms(hitchin_dual(gamma), expected_hat)
    assert max((abs(value) for value in residual.values()), default=0.0) < 1e-12
    omega = {(1, 2): 1.0, (3, 4): 1.0, (5, 6): 1.0}
    assert wedge(omega, gamma) == {}


def test_dry_run_prints_parameterization_without_writing(tmp_path: Path, capsys, monkeypatch) -> None:
    """Dry-run should be safe and side-effect-light."""
    monkeypatch.setattr(scout, "OUTPUT_DIR", tmp_path)
    scout.main(["--dry-run", "--radius", "1", "--spacing", "1", "--limit", "2"])
    output = capsys.readouterr().out
    assert "endpoint-reduced scout dry run" in output
    assert "left_A, left_B, left_C, left_D, right_A, right_B, right_C, right_D" in output
    assert not list(tmp_path.glob("*"))


def test_runner_writes_fake_result_summary(tmp_path: Path, monkeypatch) -> None:
    """The resumable JSONL runner should work with a synthetic evaluator."""
    monkeypatch.setattr(scout, "OUTPUT_DIR", tmp_path)

    class FakeGerm:
        normal_weight = 1
        constants = EndpointConstants(1.0, 2.0, 3.0, 4.0)
        residual_norm = 0.25
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
        residual_norm = 0.125
        residual = (0.125,)
        left = FakeSide()
        right = FakeSide()

    def fake_match(left, right, settings):
        return FakeMatch()

    monkeypatch.setattr(scout, "max_volume_match", fake_match)
    scout.main(["--radius", "1", "--spacing", "2", "--limit", "1", "--workers", "1", "--no-resume"])
    summaries = list(tmp_path.glob("*summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["classification_counts"] == {"ok": 1}
    assert summary["best_scouts"][0]["reconstructed_interval"] == 1.0
    assert experiments.aloff_wallach_scout.main is scout.main


def test_settings_payload_records_germ_controls() -> None:
    """Checkpoint compatibility should include the germ-fitting controls."""
    settings = AWSettings(endpoint_order=3, max_germ_evaluations=7)
    payload = scout._settings_payload(settings)
    assert payload["endpoint_order"] == 3
    assert payload["max_germ_evaluations"] == 7
