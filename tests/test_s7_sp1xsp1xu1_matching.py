"""Tests for Sp(1)xSp(1)xU(1) endpoint matching and scout wiring."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.s7 import sp1xsp1xu1_matching as matching
from experiments.s7 import sp1xsp1xu1_scout as scout
from experiments.s7 import su2_cubed_action_audit
import experiments.s7_sp1xsp1xu1_matching
import experiments.s7_sp1xsp1xu1_scout


def test_target_endpoint_parameters_recover_podesta_leading_data() -> None:
    """Round and squashed centers should expose the known endpoint constants."""
    round_target = su2_cubed_action_audit.round_target()
    left = matching.target_endpoint_parameters(round_target, "left")
    right = matching.target_endpoint_parameters(round_target, "right")
    assert left == matching.EndpointParameters(-9.0, -9.0, -27.0, -27.0, 27.0)
    assert right == left

    squashed = su2_cubed_action_audit.squashed_target()
    left_squashed = matching.target_endpoint_parameters(squashed, "left")
    right_squashed = matching.target_endpoint_parameters(squashed, "right")
    assert left_squashed.A3 == left_squashed.A4
    assert right_squashed.A3 == right_squashed.A4
    assert left_squashed.B2 == right_squashed.B4
    assert left_squashed.B4 == right_squashed.B2


def test_endpoint_initial_values_satisfy_leading_algebraic_relations() -> None:
    """The five-parameter chart should build in the leading d_gamma relations."""
    params = matching.EndpointParameters(2.0, 3.0, 5.0, 7.0, 11.0)
    values = matching.endpoint_initial_regular_values("left", params, lam=4.0)
    assert values[2] == 2.0
    assert values[3] == 3.0
    assert values[10] == 11.0
    assert abs(6 * (values[6] + values[9]) + 4.0 * values[3] ** 2) < 1e-12
    assert abs(6 * (values[8] + values[12]) + 4.0 * values[2] * values[3]) < 1e-12


def test_scout_dry_run_is_side_effect_light(tmp_path: Path, monkeypatch, capsys) -> None:
    """Dry run should print seed metadata without writing output."""
    monkeypatch.setattr(scout, "OUTPUT_DIR", tmp_path)
    scout.main(["--dry-run", "--samples", "2", "--radius", "0.1"])
    output = capsys.readouterr().out
    assert "max-volume scout dry run" in output
    assert "left_A3" in output
    assert "target: none" in output
    assert not list(tmp_path.glob("*"))


def test_target_independent_scout_samples_absolute_box_with_optional_controls() -> None:
    """The default scout should not be centered at round or squashed."""
    seeds = scout.scout_seeds(None, samples=3, radius=2.0, lam=4.0, include_known_controls=False)
    assert len(seeds) == 3
    assert all(seed.source == "random_box" for seed in seeds)
    assert all(max(abs(value) for value in scout._coordinates(seed.point)) <= 2.0 for seed in seeds)

    controlled = scout.scout_seeds(None, samples=3, radius=40.0, lam=4.0, include_known_controls=True)
    assert controlled[0].source == "round_control"
    assert controlled[1].source == "squashed_control"
    assert controlled[0].point.left.A3 == -9.0


def test_scout_writes_summary_with_fake_match(tmp_path: Path, monkeypatch) -> None:
    """The scout runner should serialize fitted max-volume results."""
    monkeypatch.setattr(scout, "OUTPUT_DIR", tmp_path)

    class FakeGerm:
        side = "left"
        source = "fit"
        residual_norm = 0.25
        success = True
        message = "ok"
        parameters = matching.EndpointParameters(1.0, 2.0, 3.0, 4.0, 5.0)

    class FakeSide:
        status = "max_volume"
        tau = 0.5
        volume = 2.0
        volume_dot = 0.0
        volume_sign = 1.0
        message = None
        germ = FakeGerm()

    class FakeMatch:
        failure = None
        residual_norm = 0.125
        residual = (0.125,)
        reconstructed_interval = 1.0
        left = FakeSide()
        right = FakeSide()

    def fake_match(left, right, settings):
        return FakeMatch()

    monkeypatch.setattr(scout.matching, "max_volume_match", fake_match)
    scout.main(["--target", "round", "--samples", "1", "--radius", "0", "--workers", "1"])
    summaries = list(tmp_path.glob("*summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["classification_counts"] == {"ok": 1}
    assert summary["best_scouts"][0]["reconstructed_interval"] == 1.0
    assert experiments.s7_sp1xsp1xu1_scout.main is scout.main
    assert experiments.s7_sp1xsp1xu1_matching.main
