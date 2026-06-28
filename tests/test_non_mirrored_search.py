"""Tests for non-mirrored seeded search helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from random import Random

from mpmath import mp

from experiments import non_mirrored_search
from solver.two_sided_refinement import TwoSidedCandidateTrack
from solver.two_sided_shooting import BASE_TWO_SIDED_POINT, TwoSidedResidualResult, TwoSidedSearchPoint


def _fake_result(point: TwoSidedSearchPoint, norm: str, failure: str | None = None) -> TwoSidedResidualResult:
    """Return a synthetic two-sided residual result."""
    params, config = non_mirrored_search.params_from_two_sided_scaled(point)
    return TwoSidedResidualResult(point, params, config, tuple(mp.mpf(norm) for _ in range(8)), mp.mpf(norm), None, None, None, None, (0, 0), {}, failure)


def _fake_track(point: TwoSidedSearchPoint, norms: tuple[str, ...], failure: str | None = None) -> TwoSidedCandidateTrack:
    """Return a synthetic non-mirrored candidate track."""
    scout = _fake_result(point, "1", failure)
    return TwoSidedCandidateTrack(1, "test", point, scout, (), tuple(_fake_result(point, norm) for norm in norms), "inconclusive")


def test_non_mirrored_seed_generation_is_reproducible() -> None:
    """The seeded scout list should be deterministic."""
    with mp.workdps(50):
        left = non_mirrored_search._search_seeds(1729)
        right = non_mirrored_search._search_seeds(1729)
        assert left == right
        assert len(left) == 484


def test_region_filters_enforce_asymmetry() -> None:
    """The asymmetric region should not emit nearly mirrored points."""
    with mp.workdps(50):
        spec = next(item for item in non_mirrored_search.REGIONS if item.name == "asymmetric")
        seeds = non_mirrored_search._region_seeds(spec, Random(1729), 0)[:10]
        for seed in seeds:
            assert non_mirrored_search._asymmetry_distance(seed.point) >= spec.min_asymmetry
            assert seed.point.s > non_mirrored_search.S_MIN


def test_sort_key_places_failures_after_successes() -> None:
    """Branch failures should sort after successful scout residuals."""
    seed = non_mirrored_search.SearchSeed(1, "test", "test", BASE_TWO_SIDED_POINT)
    success = non_mirrored_search.SearchCandidate(seed, _fake_result(BASE_TWO_SIDED_POINT, "10"))
    failure = non_mirrored_search.SearchCandidate(seed, _fake_result(BASE_TWO_SIDED_POINT, "0", "branch failure"))
    assert sorted([failure, success], key=non_mirrored_search._sort_key) == [success, failure]


def test_classification_labels_synthetic_tracks() -> None:
    """Synthetic tracks should hit the intended classification branches."""
    refs = tuple(_fake_result(BASE_TWO_SIDED_POINT, "1e-12") for _ in range(2))
    asymmetric = TwoSidedSearchPoint(mp.mpf("0.2"), mp.zero, mp.zero, mp.mpf("-0.2"), mp.zero, mp.zero, mp.zero)
    symmetric = TwoSidedSearchPoint(mp.mpf("0.1"), mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.1"), mp.mpf("0.2"), mp.mpf("0.3"), mp.zero)
    assert non_mirrored_search._classify_track(_fake_track(asymmetric, ("1e-8", "5e-8")), refs) == "possible_non_mirrored_candidate"
    assert non_mirrored_search._classify_track(_fake_track(symmetric, ("1e-10", "2e-10")), refs) == "flows_to_symmetric"
    assert non_mirrored_search._classify_track(_fake_track(asymmetric, ("1e-3", "2e-3"), "branch failure"), refs) == "branch_failure"


def test_jsonl_writer_serializes_strings(tmp_path: Path) -> None:
    """JSONL events should be append-only and parseable."""
    path = tmp_path / "events.jsonl"
    event = non_mirrored_search._event("example", {"value": non_mirrored_search._mp_string(mp.mpf("1.25"))})
    non_mirrored_search._write_jsonl_event(path, event)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["event"] == "example"
    assert loaded["value"] == "1.25"


def test_output_paths_use_non_mirrored_suffix(tmp_path: Path, monkeypatch) -> None:
    """Output path stems should identify the non-mirrored search."""
    monkeypatch.setattr(non_mirrored_search, "OUTPUT_DIR", tmp_path)
    jsonl, summary = non_mirrored_search._output_paths(datetime(2026, 4, 27, 12, 0, 0))
    assert jsonl.name.endswith("-non-mirrored.jsonl")
    assert summary.name.endswith("-non-mirrored-summary.json")
