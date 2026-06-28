"""Tests for deterministic Berger covering calibration helpers."""

from __future__ import annotations

from mpmath import mp

from experiments.berger_space import mirror_covering_calibration as covering
from problem import DEFAULT_CONFIG, DEFAULT_PARAMS
from solver.mirror_refinement import CandidateTrack, RefinementStageReport
from solver.mirror_shooting import BASE_POINT, MirrorResidualResult, MirrorSearchPoint


def _fake_result(point: MirrorSearchPoint, norm: str, failure: str | None = None) -> MirrorResidualResult:
    """Build one synthetic mirror residual result."""
    return MirrorResidualResult(point, DEFAULT_PARAMS, DEFAULT_CONFIG, (), mp.mpf(norm), None, None, 0, {}, failure)


def _fake_candidate(index: int, point: MirrorSearchPoint, norm: str) -> covering.SearchCandidate:
    """Build one synthetic scout candidate."""
    seed = covering.SearchSeed(index, "test", "synthetic", point)
    return covering.SearchCandidate(seed, _fake_result(point, norm))


def _fake_probe(index: int, point: MirrorSearchPoint, initial: str, final: str) -> covering.ContractionProbe:
    """Build one synthetic contraction probe."""
    candidate = _fake_candidate(index, point, initial)
    stage = RefinementStageReport(covering.CONTRACTION_SETTINGS, _fake_result(point, initial), _fake_result(point, final), (), "max_steps")
    return covering.ContractionProbe(candidate, stage)


def test_covering_grid_has_expected_size_and_distance() -> None:
    """The covering grid should avoid Berger but come within 0.1."""
    seeds = covering._grid_seeds()
    assert len(seeds) == covering.GRID_SIZE
    assert all(seed.point != BASE_POINT for seed in seeds)
    assert min(covering._point_distance(seed.point) for seed in seeds) == mp.mpf("0.1")
    assert len(covering._oracle_seeds(seeds)) == 16
    assert all(covering._point_distance(seed.point) == mp.mpf("0.1") for seed in covering._oracle_seeds(seeds))


def test_covering_grid_covers_the_box() -> None:
    """The coordinate grid should cover [-1,1]^4 at max-radius 0.1."""
    assert covering._grid_covers_box()


def test_blind_selection_is_deterministic_and_not_distance_based() -> None:
    """Selection should follow residual/probe data rather than Berger distance."""
    near = MirrorSearchPoint(mp.mpf("0.1"), mp.mpf("0.1"), mp.mpf("0.1"), mp.mpf("0.1"))
    far = MirrorSearchPoint(mp.mpf("0.9"), mp.mpf("0.9"), mp.mpf("0.9"), mp.mpf("0.9"))
    probes = [_fake_probe(1, near, "1", "0.5"), _fake_probe(2, far, "1", "0.1")]
    left = covering._select_blind_candidates(probes, best_final_quota=1, best_ratio_quota=0, diverse_quota=0)
    right = covering._select_blind_candidates(probes, best_final_quota=1, best_ratio_quota=0, diverse_quota=0)
    assert left == right
    assert left[0].candidate.seed.index == 2


def test_selection_deduplicates_exact_scaled_coordinates() -> None:
    """The same coordinate selected by multiple rules should appear once."""
    point = MirrorSearchPoint(mp.mpf("0.3"), mp.zero, mp.zero, mp.zero)
    probes = [_fake_probe(1, point, "1", "0.1"), _fake_probe(2, point, "2", "0.2")]
    selected = covering._select_blind_candidates(probes, best_final_quota=2, best_ratio_quota=2, diverse_quota=0)
    assert len(selected) == 1


def test_diverse_selection_returns_separated_representatives() -> None:
    """Diverse selection should add candidates away from already chosen points."""
    points = [
        MirrorSearchPoint(mp.mpf("-0.9"), mp.mpf("-0.9"), mp.zero, mp.zero),
        MirrorSearchPoint(mp.mpf("0.9"), mp.mpf("0.9"), mp.zero, mp.zero),
        MirrorSearchPoint(mp.mpf("0.1"), mp.mpf("0.1"), mp.zero, mp.zero),
    ]
    candidates = [_fake_candidate(index, point, str(index + 1)) for index, point in enumerate(points)]
    diverse = covering._select_diverse(candidates, [candidates[0]], 1)
    assert diverse == [candidates[1]]


def test_checkpoint_and_probe_payload_round_trip(tmp_path) -> None:
    """Covering checkpoint metadata and contraction probes should reload."""
    jsonl_path = tmp_path / "covering.jsonl"
    summary_path = tmp_path / "covering-summary.json"
    start = covering._event("run_start", covering._run_start_payload(jsonl_path, summary_path))
    probe = _fake_probe(7, MirrorSearchPoint(mp.mpf("0.1"), mp.zero, mp.zero, mp.zero), "1", "0.25")
    covering._write_jsonl_event(jsonl_path, start)
    covering._write_jsonl_event(jsonl_path, covering._event("contraction_probe", covering._probe_payload(probe)))
    assert covering._checkpoint_is_compatible(jsonl_path)
    loaded = covering._load_contraction_probes(jsonl_path)
    assert loaded[7].candidate.seed.point == probe.candidate.seed.point
    assert loaded[7].stage.final.residual_norm == probe.stage.final.residual_norm


def test_synthetic_oracle_recovery_classification() -> None:
    """Synthetic near-Berger successful tracks should classify as recovered."""
    base = _fake_result(BASE_POINT, "1e-14")
    scout = _fake_result(BASE_POINT, "1e-14")
    track = CandidateTrack(0, "oracle", BASE_POINT, scout, (), (base, base), "inconclusive")
    assert covering.recovery._classify_track(track, (base, base)) == "recovered_berger"
