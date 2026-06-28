"""Tests for Foscolo-Haskins S6 scout grid runners."""

from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.foscolo_haskins import s6_scout


def test_axis_values_include_bounds_with_max_spacing() -> None:
    """Scout axes should include both endpoints and respect max spacing."""
    axis = s6_scout._axis_values(math.log(0.5), math.log(2.0), 0.3)
    assert axis[0] == math.log(0.5)
    assert axis[-1] == math.log(2.0)
    assert max(b - a for a, b in zip(axis, axis[1:])) <= 0.3


def test_terminal_seed_count_includes_transform_axis() -> None:
    """Terminal scout counts should include a,b,h and transform choices."""
    grid = s6_scout.ScoutGrid(
        "terminal",
        (0.0, 0.1),
        (0.0, 0.1),
        0.1,
        (0.0, 0.1),
        ("round-terminal", "exotic-terminal"),
    )
    assert [len(axis) for axis in s6_scout.scout_axes(grid)] == [2, 2, 2]
    assert s6_scout.scout_seed_count(grid) == 16


def test_local_minima_are_detected_per_transform() -> None:
    """Nearest-neighbor minima should be computed separately by terminal transform."""
    grid = s6_scout.ScoutGrid(
        "terminal",
        (0.0, 0.2),
        (0.0, 0.0),
        0.1,
        (0.0, 0.0),
        ("round-terminal", "exotic-terminal"),
    )
    payloads = [
        {"seed_index": 0, "method": "terminal", "grid_index": [0, 0, 0, 0], "status": "ok", "residual_norm": 3.0, "parameters": {}, "transform": "round-terminal"},
        {"seed_index": 2, "method": "terminal", "grid_index": [1, 0, 0, 0], "status": "ok", "residual_norm": 1.0, "parameters": {}, "transform": "round-terminal"},
        {"seed_index": 4, "method": "terminal", "grid_index": [2, 0, 0, 0], "status": "ok", "residual_norm": 2.0, "parameters": {}, "transform": "round-terminal"},
        {"seed_index": 1, "method": "terminal", "grid_index": [0, 0, 0, 1], "status": "ok", "residual_norm": 1.5, "parameters": {}, "transform": "exotic-terminal"},
        {"seed_index": 3, "method": "terminal", "grid_index": [1, 0, 0, 1], "status": "ok", "residual_norm": 2.5, "parameters": {}, "transform": "exotic-terminal"},
    ]
    minima = s6_scout._local_minima(payloads, grid)
    assert [payload["seed_index"] for payload in minima] == [2, 1]


def test_max_volume_scout_cli_tiny_run(tmp_path: Path, monkeypatch) -> None:
    """Tiny max-volume scout should write compact JSONL and summary output."""
    monkeypatch.setattr(s6_scout, "MAX_VOLUME_OUTPUT_DIR", tmp_path)
    s6_scout.main_max_volume(["--limit", "3", "--workers", "1", "--no-resume", "--progress-every", "1"])
    summaries = sorted(tmp_path.glob("*-summary.json"))
    assert len(summaries) == 1
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert payload["grid"]["method"] == "max-volume"
    assert payload["scout_count"] == 3
    assert payload["best_scouts"]


def test_parallel_scout_falls_back_to_threads(tmp_path: Path, monkeypatch) -> None:
    """Restricted environments that block process pools should still run scouts."""

    def broken_process_pool(*args, **kwargs):
        raise PermissionError("sandbox blocks process semaphores")

    monkeypatch.setattr(s6_scout, "MAX_VOLUME_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(s6_scout, "ProcessPoolExecutor", broken_process_pool)
    s6_scout.main_max_volume(["--limit", "2", "--workers", "2", "--no-resume", "--progress-every", "1"])
    summaries = sorted(tmp_path.glob("*-summary.json"))
    assert len(summaries) == 1
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert payload["scout_count"] == 2
    assert payload["classification_counts"]["ok"] == 2


def test_terminal_scout_cli_tiny_run(tmp_path: Path, monkeypatch) -> None:
    """Tiny terminal scout should write compact JSONL and summary output."""
    monkeypatch.setattr(s6_scout, "TERMINAL_OUTPUT_DIR", tmp_path)
    s6_scout.main_terminal(["--limit", "3", "--workers", "1", "--no-resume", "--progress-every", "1"])
    summaries = sorted(tmp_path.glob("*-summary.json"))
    assert len(summaries) == 1
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert payload["grid"]["method"] == "terminal"
    assert payload["scout_count"] == 3
    assert payload["best_scouts"]
