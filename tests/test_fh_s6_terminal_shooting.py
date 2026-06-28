"""Tests for naive Foscolo-Haskins S6 terminal shooting."""

from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.foscolo_haskins import s6_terminal_shooting as shooting


def test_round_terminal_recovery() -> None:
    """The naive terminal shooter should recover the round S6 parameters."""
    run = shooting.recover_terminal_shooting("round", 1.6, 1.4, 0.75, "round-terminal")
    assert run.classification == "recovered_round_s6"
    assert abs(run.final.a - math.sqrt(3.0)) < 1e-7
    assert abs(run.final.b - 1.5) < 1e-7
    assert abs(run.final.match_time - 0.77289845) < 1e-6
    assert run.final.residual_norm < 1e-7


def test_exotic_terminal_recovery() -> None:
    """The naive terminal shooter should recover the inhomogeneous FH S6 parameters."""
    run = shooting.recover_terminal_shooting("exotic", 0.55, 0.6, 1.2, "exotic-terminal")
    assert run.classification == "recovered_exotic_s6"
    assert abs(run.final.a - 0.564550017) < 1e-7
    assert abs(run.final.b - 0.599013546) < 1e-7
    assert abs(run.final.match_time - 1.22271453) < 1e-6
    assert run.final.residual_norm < 1e-7


def test_wrong_terminal_transform_does_not_recover_exotic() -> None:
    """The exotic root is sensitive to the terminal chart symmetry."""
    evaluation = shooting.evaluate_terminal_shooting(0.564550017, 0.599013546, 1.22271453, "round-terminal")
    assert evaluation.residual_norm > 0.1


def test_cli_writes_terminal_summary(tmp_path: Path, monkeypatch) -> None:
    """The terminal shooting CLI should write JSONL and summary artifacts."""
    monkeypatch.setattr(shooting, "OUTPUT_DIR", tmp_path)
    shooting.main(["--recover-round"])
    summaries = sorted(tmp_path.glob("*-summary.json"))
    jsonls = sorted(tmp_path.glob("*.jsonl"))
    assert len(summaries) == 1
    assert len(jsonls) == 1
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert payload["classification"] == "recovered_round_s6"
    assert payload["transform"] == "round-terminal"
