"""Tests for the V3 deep mirror sweep with a midpoint floor."""

from __future__ import annotations

import json
from random import Random

from mpmath import mp

from experiments import mirror_sweep, mirror_sweep_v2, mirror_sweep_v3
from problem import DEFAULT_CONFIG, DEFAULT_PARAMS
from solver.mirror_refinement import CandidateTrack, NewtonSettings, RefinementStageReport
from solver.mirror_shooting import MirrorResidualResult
from solver.mirror_shooting import MirrorSearchPoint


def _fake_result(point: MirrorSearchPoint, norm: str) -> MirrorResidualResult:
    """Build a synthetic residual result."""
    return MirrorResidualResult(point, DEFAULT_PARAMS, DEFAULT_CONFIG, (), mp.mpf(norm), None, None, 0, {}, None)


def _fake_track(point: MirrorSearchPoint, verification_norms) -> CandidateTrack:
    """Build a synthetic verified track."""
    settings = NewtonSettings("fake", DEFAULT_CONFIG, mp.mpf("1e-3"), mp.mpf("1e-8"), 1)
    final = _fake_result(point, "1e-8")
    stage = RefinementStageReport(settings, _fake_result(point, "1"), final, (), "max_steps")
    verifications = tuple(_fake_result(point, norm) for norm in verification_norms)
    return CandidateTrack(1, "test", point, _fake_result(point, "1"), (stage,), verifications, "inconclusive")


def test_v3_midpoint_floor_constant_matches_definition() -> None:
    """The V3 s-floor should be exactly the scaled form of m > 0.01."""
    with mp.workdps(70):
        expected = mp.log(mp.mpf("0.01") / DEFAULT_CONFIG.match_t)
        assert abs(mirror_sweep_v3.S_MIN - expected) < mp.mpf("1e-60")


def test_v3_search_seed_count_is_expected() -> None:
    """V3 should generate 14000 random box seeds plus 16 corners."""
    with mp.workdps(50):
        assert len(mirror_sweep_v3._search_seeds()) == 14016


def test_v3_generated_seeds_respect_midpoint_floor() -> None:
    """All V3 seeds should have m strictly above the configured floor."""
    with mp.workdps(50):
        for seed in mirror_sweep_v3._search_seeds():
            match_t = DEFAULT_CONFIG.match_t * mp.exp(seed.point.s)
            assert seed.point.s > mirror_sweep_v3.S_MIN
            assert match_t > mirror_sweep_v3.MIN_MATCH_T


def test_v3_box_sampling_is_reproducible_and_respects_ranges() -> None:
    """Asymmetric V3 boxes should be deterministic and obey coordinate ranges."""
    spec = mirror_sweep_v3.REGIONS[2]
    left = mirror_sweep_v3._box_region_seeds(spec, Random(1729), 0)[:10]
    right = mirror_sweep_v3._box_region_seeds(spec, Random(1729), 0)[:10]
    assert left == right
    for seed in left:
        for value, (lower, upper) in zip((seed.point.u, seed.point.v, seed.point.r, seed.point.s), spec.ranges):
            assert lower <= value <= upper
        assert seed.point.s > mirror_sweep_v3.S_MIN


def test_previous_sweep_seed_counts_are_unchanged() -> None:
    """V3 helper additions should not alter the completed V1/V2 recipes."""
    with mp.workdps(50):
        assert len(mirror_sweep._search_seeds()) == 5416
        assert len(mirror_sweep_v2._search_seeds()) == 9016


def test_v3_strong_lead_requires_midpoint_floor() -> None:
    """A point below the m-floor should not be classified as a strong lead."""
    point = MirrorSearchPoint(mp.mpf("0.1"), mp.zero, mp.zero, mirror_sweep_v3.S_MIN)
    berger_refs = (_fake_result(MirrorSearchPoint(mp.zero, mp.zero, mp.zero, mp.zero), "1e-8"),)
    assert mirror_sweep_v3._classify_track_v3(_fake_track(point, ("2e-5", "3e-5")), berger_refs) != "strong_lead"


def test_v3_loads_cached_scout_candidates(tmp_path) -> None:
    """V3 resume should reconstruct scout candidates from JSONL output."""
    seed = mirror_sweep_v3.SearchSeed(7, "near_control", "test", MirrorSearchPoint(mp.mpf("0.5"), mp.zero, mp.zero, mp.zero))
    candidate = mirror_sweep_v3.SearchCandidate(seed, _fake_result(seed.point, "0.125"))
    path = tmp_path / "cached.jsonl"
    event = {"event": "scout_result", **mirror_sweep_v3._candidate_payload(candidate)}
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    loaded = mirror_sweep_v3._load_scout_candidates(path)
    assert loaded[7].seed == seed
    assert loaded[7].result.residual_norm == mp.mpf("0.125")


def test_v3_finds_latest_incomplete_checkpoint(tmp_path) -> None:
    """Only V3 JSONL files without run summaries should be resumed."""
    older = tmp_path / "20260425-010000-seed1729-v3.jsonl"
    newer = tmp_path / "20260425-020000-seed1729-v3.jsonl"
    complete = tmp_path / "20260425-030000-seed1729-v3.jsonl"
    older.write_text(json.dumps({"event": "run_start"}) + "\n", encoding="utf-8")
    newer.write_text(json.dumps({"event": "run_start"}) + "\n", encoding="utf-8")
    complete.write_text(json.dumps({"event": "run_summary"}) + "\n", encoding="utf-8")
    assert mirror_sweep_v3._latest_incomplete_jsonl(tmp_path) == newer


def test_v3_detects_post_scout_events(tmp_path) -> None:
    """Resume should know when a checkpoint is no longer scout-only."""
    path = tmp_path / "events.jsonl"
    rows = [{"event": "run_start"}, {"event": "scout_result"}, {"event": "region_selection"}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert mirror_sweep_v3._has_post_scout_events(path)
