"""Tests for the wide non-mirrored surrogate search helpers."""

from __future__ import annotations

import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from mpmath import mp

from experiments import non_mirrored_surrogate_wide_search as wide
from experiments.non_mirrored_surrogate_common import (
    ConstantBranchModel,
    ConstantResidualModel,
    SurrogateModels,
    _candidate_payload,
    _event,
    _labels_from_candidates,
    _load_candidates_from_jsonl,
    _load_refinement_tracks_from_jsonl,
    _run_with_timeout,
    _score_points,
    _stage_payload,
    _write_jsonl_event,
)
from solver.two_sided_refinement import TwoSidedCandidateTrack, TwoSidedRefinementStageReport
from solver.two_sided_shooting import TwoSidedResidualResult, TwoSidedSearchPoint, params_from_two_sided_scaled
from experiments.non_mirrored_common import SearchCandidate, SearchSeed, _asymmetry_distance, _point_distance, _track_payload


class SequenceResidualModel:
    """Synthetic model returning a fixed sequence of predictions."""

    def __init__(self, values: tuple[float, ...]):
        self.values = np.asarray(values, dtype=np.float64)

    def predict(self, features):
        """Return one prediction per feature row."""
        return self.values[: features.shape[0]]


def _fake_result(point: TwoSidedSearchPoint, residual: tuple[str, ...], failure: str | None = None) -> TwoSidedResidualResult:
    """Return a synthetic residual result."""
    params, config = params_from_two_sided_scaled(point)
    values = tuple(mp.mpf(value) for value in residual) if failure is None else ()
    norm = max(abs(value) for value in values) if values else mp.inf
    return TwoSidedResidualResult(point, params, config, values, norm, None, None, None, None, (0, 0), {}, failure)


def test_wide_training_seed_generation_has_exact_quotas() -> None:
    """The V2 training design should be reproducible and exactly sized."""
    with mp.workdps(50):
        left = wide._training_seeds(1729)
        right = wide._training_seeds(1729)
        counts = Counter(seed.region for seed in left)
        expected = {
            "near_control": 1000,
            "medium_asymmetric": 3500,
            "far_asymmetric": 4500,
            "wide_asymmetric": 4500,
            "alpha_wide": 3000,
            "m_wide": 2500,
            "tail_negative_uv": 1000,
        }
        assert left == right
        assert len(left) == 20000
        assert counts == expected


def test_wide_training_seeds_satisfy_region_filters() -> None:
    """All generated V2 seeds should satisfy distance, asymmetry, and m filters."""
    specs = {spec.name: spec for spec in wide.TRAINING_REGIONS}
    with mp.workdps(50):
        for seed in wide._training_seeds(1729):
            spec = specs[seed.region]
            assert seed.point.s > wide.S_MIN
            assert spec.min_distance <= _point_distance(seed.point) <= spec.max_distance
            assert _asymmetry_distance(seed.point) >= spec.min_asymmetry


def test_vector_labels_have_expected_shapes_and_values() -> None:
    """Synthetic successful samples should produce finite norm and vector labels."""
    point = TwoSidedSearchPoint(mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero)
    seed = SearchSeed(1, "test", "synthetic", point)
    candidate = SearchCandidate(seed, _fake_result(point, tuple(str(index + 1) for index in range(8))))
    branch, logs, residuals = _labels_from_candidates([candidate])
    assert branch.shape == (1,)
    assert logs.shape == (1,)
    assert residuals.shape == (1, 8)
    assert np.isfinite(logs).all()
    assert np.allclose(residuals[0], np.arange(1, 9))


def test_vector_scoring_penalizes_component_large_predictions() -> None:
    """Vector-head scoring should prefer points with all components predicted small."""
    good = TwoSidedSearchPoint(mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero)
    bad = TwoSidedSearchPoint(mp.mpf("0.1"), mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero)
    vector_models = tuple((SequenceResidualModel((1e-8, 1e-1)),) for _ in range(8))
    models = SurrogateModels(ConstantBranchModel(1.0), (ConstantResidualModel(-3.0),), vector_models, {})
    proposals = _score_points(models, "test", [good, bad], 0, branch_cutoff=0.0, optimism_weight=0.0)
    assert proposals[0].score < proposals[1].score
    assert proposals[0].vector_log_norm < proposals[1].vector_log_norm


def test_timeout_wrapper_returns_nonfatal_timeout() -> None:
    """Slow synthetic work should become a timeout status instead of raising."""
    value, status = _run_with_timeout(lambda: time.sleep(0.2), 0.01)
    assert value is None
    assert status == "timeout"


def test_candidate_checkpoint_round_trip(tmp_path: Path) -> None:
    """Persisted training samples should reload as search candidates."""
    point = TwoSidedSearchPoint(mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero)
    seed = SearchSeed(7, "test", "synthetic", point)
    candidate = SearchCandidate(seed, _fake_result(point, tuple("0.1" for _ in range(8))))
    path = tmp_path / "checkpoint.jsonl"
    _write_jsonl_event(path, _event("training_sample", _candidate_payload(candidate)))
    loaded = _load_candidates_from_jsonl(path, "training_sample")
    assert len(loaded) == 1
    assert loaded[0].seed.index == 7
    assert loaded[0].result.residual_norm == mp.mpf("0.1")


def test_resume_path_chooses_incomplete_checkpoint(tmp_path: Path, monkeypatch) -> None:
    """The V2 runner should resume the newest incomplete checkpoint."""
    monkeypatch.setattr(wide, "OUTPUT_DIR", tmp_path)
    old_jsonl, _summary = wide._output_paths(datetime(2026, 4, 28, 18, 14, 47))
    old_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl_event(old_jsonl, _event("run_start", {"random_seed": 1729}))
    jsonl_path, summary_path, resumed = wide._resume_or_new_paths()
    assert resumed is True
    assert jsonl_path == old_jsonl
    assert summary_path.name.endswith("-summary.json")


def test_refinement_stage_checkpoint_round_trip(tmp_path: Path) -> None:
    """Completed order-6 tracks should reload from refinement-stage events."""
    point = TwoSidedSearchPoint(mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero)
    result = _fake_result(point, tuple("0.2" for _ in range(8)))
    stage = TwoSidedRefinementStageReport(wide.ORDER6_SETTINGS, result, result, (), "max_steps")
    track = TwoSidedCandidateTrack(123, "test", point, result, (stage,), (), "inconclusive")
    path = tmp_path / "checkpoint.jsonl"
    payload = {"stage": _stage_payload(stage), "track": _track_payload(track)}
    _write_jsonl_event(path, _event("refinement_stage", payload))
    loaded = _load_refinement_tracks_from_jsonl(path, {6: wide.ORDER6_SETTINGS, 10: wide.ORDER10_SETTINGS})
    assert len(loaded) == 1
    assert loaded[0].seed_rank == 123
    assert loaded[0].stages[-1].final.residual_norm == mp.mpf("0.2")
