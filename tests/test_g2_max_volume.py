"""Tests for G2 maximal-volume matching and scout runners."""

from __future__ import annotations

import json
from pathlib import Path

from mpmath import mp

from problem import DEFAULT_CONFIG, DEFAULT_PARAMS, SolverConfig
from solver.max_volume import MAX_VOLUME_VERSION, MaxVolumeSettings, max_volume_match

from experiments.berger_space import max_volume_calibration as berger_calibration
from experiments.berger_space import max_volume_scout as berger_scout
from experiments.s7 import max_volume_calibration as s7_calibration
from experiments.s7 import max_volume_scout as s7_scout
from experiments.s7.search_common import TARGETS


def _quick_settings() -> MaxVolumeSettings:
    """Return a cheap setting that still finds the known max-volume orbits."""
    config = SolverConfig(8, 50, 20, mp.mpf("0.8"), 1, DEFAULT_CONFIG.match_t)
    return MaxVolumeSettings(config, bisection_steps=36, event_tolerance=mp.mpf("1e-20"))


def _payload(seed_index: int, grid_index: list[int], residual: str | None, target: str | None = None) -> dict:
    """Return a synthetic scout payload."""
    failure = None if residual is not None else "failed"
    payload = {
        "seed_index": seed_index,
        "region": "toy",
        "grid_index": grid_index,
        "seed_point": {},
        "physical": {},
        "result": {
            "failure": failure,
            "residual_norm": residual,
            "reconstructed_interval": None,
            "interval_error": None,
            "left": {"status": "max_volume"},
            "right": {"status": "max_volume" if failure is None else "branch_exit"},
        },
    }
    if target is not None:
        payload["target"] = target
    return payload


def test_known_solutions_have_matching_max_volume_orbits_at_low_order() -> None:
    """The event finder should approximately recover Berger, round S7, and squashed S7."""
    settings = _quick_settings()
    with mp.workdps(settings.config.working_dps):
        berger = max_volume_match(DEFAULT_PARAMS, settings)
        round_s7 = max_volume_match(TARGETS["round"].params_builder(), settings)
        squashed_s7 = max_volume_match(TARGETS["squashed"].params_builder(), settings)
    assert MAX_VOLUME_VERSION == "g2-max-volume-v1"
    for match in (berger, round_s7, squashed_s7):
        assert match.failure is None
        assert match.residual_norm < mp.mpf("0.003")
        assert abs(match.interval_error) < mp.mpf("0.003")


def test_calibration_classifiers_cover_known_successes() -> None:
    """High-accuracy classifications should use stable recovered labels."""
    assert berger_calibration.classify_berger_match(mp.mpf("1e-10"), mp.mpf("1e-10")) == "recovered_berger"
    assert berger_calibration.classify_berger_match(mp.mpf("1e-4"), mp.mpf("1e-4")) == "failed"
    assert s7_calibration.classify_s7_match("round", mp.mpf("1e-10"), mp.mpf("1e-10")) == "recovered_round_s7"
    assert s7_calibration.classify_s7_match("squashed", mp.mpf("1e-10"), mp.mpf("1e-10")) == "recovered_squashed_s7"


def test_berger_local_minima_ignore_failed_neighbors() -> None:
    """Failed scouts should not suppress successful local minima."""
    payloads = [
        _payload(0, [0], "3"),
        _payload(1, [1], None),
        _payload(2, [2], "2"),
    ]
    minima = berger_scout._local_minima(payloads, [3])
    assert [payload["seed_index"] for payload in minima] == [2, 0]


def test_s7_local_minima_are_computed_per_target() -> None:
    """S7 local minima should compare only scouts from the same fixed right chart."""
    payloads = [
        _payload(0, [0], "3", "round"),
        _payload(1, [1], "1", "round"),
        _payload(2, [2], "2", "round"),
        _payload(3, [0], "1.5", "squashed"),
        _payload(4, [1], "2.5", "squashed"),
    ]
    minima = s7_scout._local_minima(payloads, [3])
    assert [payload["seed_index"] for payload in minima] == [1, 3]


def test_berger_max_volume_scout_cli_tiny_run(tmp_path: Path, monkeypatch) -> None:
    """A tiny Berger scout should write resumable JSONL and a summary."""
    monkeypatch.setattr(berger_scout, "OUTPUT_DIR", tmp_path)
    berger_scout.main(["--limit", "2", "--workers", "1", "--no-resume", "--progress-every", "1"])
    summaries = sorted(tmp_path.glob("*-summary.json"))
    jsonls = sorted(tmp_path.glob("*.jsonl"))
    assert len(summaries) == 1
    assert len(jsonls) == 1
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert payload["max_volume_version"] == MAX_VOLUME_VERSION
    assert payload["scout_version"] == berger_scout.SCOUT_VERSION
    assert payload["grid"]["coordinate_names"] == list(berger_scout.COORDINATE_NAMES)
    assert payload["scout_count"] == 2


def test_s7_max_volume_scout_cli_tiny_run(tmp_path: Path, monkeypatch) -> None:
    """A tiny S7 scout should write compact best-scout output."""
    monkeypatch.setattr(s7_scout, "OUTPUT_DIR", tmp_path)
    s7_scout.main(["--limit", "2", "--workers", "1", "--no-resume", "--progress-every", "1"])
    summaries = sorted(tmp_path.glob("*-summary.json"))
    assert len(summaries) == 1
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert payload["max_volume_version"] == MAX_VOLUME_VERSION
    assert payload["scout_version"] == s7_scout.SCOUT_VERSION
    assert payload["scout_count"] == 2
    assert payload["best_scouts"]


def test_parallel_s7_scout_falls_back_to_threads(tmp_path: Path, monkeypatch) -> None:
    """Restricted environments that block process pools should still run scouts."""

    def broken_process_pool(*args, **kwargs):
        raise PermissionError("sandbox blocks process semaphores")

    monkeypatch.setattr(s7_scout, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(s7_scout, "ProcessPoolExecutor", broken_process_pool)
    s7_scout.main(["--limit", "2", "--workers", "2", "--no-resume", "--progress-every", "1"])
    summaries = sorted(tmp_path.glob("*-summary.json"))
    assert len(summaries) == 1
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert payload["scout_count"] == 2
    assert payload["classification_counts"]["ok"] == 2
