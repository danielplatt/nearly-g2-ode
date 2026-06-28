"""Tests for deterministic seeded mirror-search helpers."""

from __future__ import annotations

from random import Random

from mpmath import mp

from experiments import mirror_search
from problem import DEFAULT_CONFIG, DEFAULT_PARAMS
from solver.mirror_refinement import CandidateTrack, NewtonSettings, RefinementStageReport
from solver.mirror_shooting import BASE_POINT, MirrorResidualResult, MirrorSearchPoint


def _fake_result(point: MirrorSearchPoint, norm: str, failure: str | None = None) -> MirrorResidualResult:
    """Build a small synthetic residual result for classification tests."""
    return MirrorResidualResult(
        point=point,
        params=DEFAULT_PARAMS,
        config=DEFAULT_CONFIG,
        residual=(),
        residual_norm=mp.mpf(norm),
        match_q=None,
        l_value=None,
        patch_count=0,
        branch_diagnostics={},
        failure=failure,
    )


def _fake_track(point: MirrorSearchPoint, stage_norm: str, verification_norms, failure: str | None = None) -> CandidateTrack:
    """Build a synthetic candidate track with one completed stage."""
    settings = NewtonSettings("fake", DEFAULT_CONFIG, mp.mpf("1e-3"), mp.mpf("1e-8"), 1)
    final = _fake_result(point, stage_norm, failure)
    stage = RefinementStageReport(settings, _fake_result(point, "1"), final, (), "max_steps")
    verifications = tuple(_fake_result(point, norm) for norm in verification_norms)
    return CandidateTrack(1, "test", point, _fake_result(point, "1"), (stage,), verifications, "inconclusive")


def test_annular_seed_generation_is_reproducible_and_in_range() -> None:
    """Annular random seeds should be deterministic and obey max-distance bounds."""
    with mp.workdps(50):
        left = mirror_search._annular_seeds("near", mp.mpf("0.25"), mp.one, 5, Random(1729))
        right = mirror_search._annular_seeds("near", mp.mpf("0.25"), mp.one, 5, Random(1729))
        assert left == right
        for seed in left:
            distance = mirror_search._point_distance(seed.point)
            assert mp.mpf("0.25") <= distance <= mp.one


def test_sort_key_places_branch_failures_after_successes() -> None:
    """Successful scout candidates should sort before branch failures."""
    seed = mirror_search.SearchSeed("test", BASE_POINT)
    success = mirror_search.SearchCandidate(seed, _fake_result(BASE_POINT, "10"))
    failure = mirror_search.SearchCandidate(seed, _fake_result(BASE_POINT, "0", "branch failure"))
    assert sorted([failure, success], key=mirror_search._sort_key) == [success, failure]


def test_candidate_classification_labels_synthetic_tracks() -> None:
    """Classification should distinguish the main search outcomes."""
    berger_refs = (_fake_result(BASE_POINT, "1e-8"), _fake_result(BASE_POINT, "2e-8"))
    near = MirrorSearchPoint(mp.mpf("0.001"), mp.zero, mp.zero, mp.zero)
    far = MirrorSearchPoint(mp.mpf("0.1"), mp.zero, mp.zero, mp.zero)

    assert mirror_search._classify_track(_fake_track(near, "1e-12", ("1e-8", "2e-8")), berger_refs) == "flows_to_berger"
    assert mirror_search._classify_track(_fake_track(near, "1e-12", ("1e-3", "2e-3")), berger_refs) == "finite_order_artifact"
    assert mirror_search._classify_track(_fake_track(far, "1e-12", ("1e-8", "5e-8")), berger_refs) == "possible_candidate"
    assert mirror_search._classify_track(_fake_track(far, "1", (), "branch failure"), berger_refs) == "branch_failure"
