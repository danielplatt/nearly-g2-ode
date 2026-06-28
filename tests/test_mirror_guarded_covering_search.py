"""Tests for the guarded mirrored covering search runner."""

from __future__ import annotations

from datetime import datetime

from mpmath import mp

from experiments.berger_space import mirror_guarded_covering_search as guarded
from experiments.shared.mirror_sweep_common import SearchSeed, _event, _write_jsonl_event
from solver.mirror_shooting import MirrorSearchPoint


def test_halton_points_are_deterministic_and_guarded() -> None:
    """The wide deterministic scouts should stay inside the guarded box."""
    with mp.workdps(80):
        points = [guarded._halton_point(guarded.HALTON_SKIP + index) for index in range(1, 40)]
        repeated = [guarded._halton_point(guarded.HALTON_SKIP + index) for index in range(1, 40)]
        assert points == repeated
        for point in points:
            assert guarded.S_MIN < point.s < guarded.MAX_REFINEMENT_COORDINATE
            assert max(abs(point.u), abs(point.v), abs(point.r), abs(point.s)) <= guarded.MAX_REFINEMENT_COORDINATE


def test_search_seed_counts_match_recipe() -> None:
    """The runner should contain the 10k covering grid plus 40k wide scouts."""
    assert len(guarded._core_grid_seeds()) == guarded.CORE_SEEDS
    assert len(guarded._halton_seeds()) == guarded.HALTON_SAMPLES


def test_newton_settings_include_both_guardrails() -> None:
    """Both guarded Newton stages should enforce coordinate and midpoint floors."""
    for settings in (guarded.CONTRACTION_SETTINGS, guarded.ORDER10_SETTINGS, guarded.ORDER14_SETTINGS):
        assert settings.max_abs_coordinate == guarded.MAX_REFINEMENT_COORDINATE
        assert settings.min_s_coordinate == guarded.S_MIN


def test_midpoint_floor_is_rejected_without_solving() -> None:
    """A scout exactly at the forbidden floor should produce a diagnostic result."""
    point = MirrorSearchPoint(mp.zero, mp.zero, mp.zero, guarded.S_MIN)
    seed = SearchSeed(123456, "floor_test", "synthetic", point)
    candidate = guarded._evaluate_seed(seed)
    assert candidate.result.failure == "m_floor_rejected"
    assert candidate.result.patch_count == 0


def test_checkpoint_compatibility_round_trips(tmp_path) -> None:
    """A run-start event with matching metadata should be resumable."""
    jsonl_path, summary_path = guarded._output_paths(datetime(2026, 5, 14, 12, 0, 0))
    jsonl_path = tmp_path / jsonl_path.name
    summary_path = tmp_path / summary_path.name
    _write_jsonl_event(jsonl_path, _event("run_start", guarded._run_start_payload(jsonl_path, summary_path)))
    assert guarded._checkpoint_is_compatible(jsonl_path)
