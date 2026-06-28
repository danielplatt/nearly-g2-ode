"""Tests for the V2 deep mirror-sweep experiment."""

from __future__ import annotations

import json
from random import Random

from mpmath import mp

from experiments import mirror_sweep_v2
from problem import DEFAULT_CONFIG, DEFAULT_PARAMS
from solver.mirror_refinement import CandidateTrack, NewtonSettings, RefinementStageReport
from solver.mirror_shooting import MirrorResidualResult, MirrorSearchPoint


def _point(value: str) -> MirrorSearchPoint:
    """Build a simple one-coordinate point."""
    return MirrorSearchPoint(mp.mpf(value), mp.zero, mp.zero, mp.zero)


def _fake_result(point: MirrorSearchPoint, norm: str, failure: str | None = None) -> MirrorResidualResult:
    """Build a synthetic residual result."""
    return MirrorResidualResult(point, DEFAULT_PARAMS, DEFAULT_CONFIG, (), mp.mpf(norm), None, None, 0, {}, failure)


def _fake_track(point: MirrorSearchPoint, verification_norms) -> CandidateTrack:
    """Build a synthetic verified track."""
    settings = NewtonSettings("fake", DEFAULT_CONFIG, mp.mpf("1e-3"), mp.mpf("1e-8"), 1)
    final = _fake_result(point, "1e-8")
    stage = RefinementStageReport(settings, _fake_result(point, "1"), final, (), "max_steps")
    verifications = tuple(_fake_result(point, norm) for norm in verification_norms)
    return CandidateTrack(1, "test", point, _fake_result(point, "1"), (stage,), verifications, "inconclusive")


def test_v2_search_seed_count_is_expected() -> None:
    """V2 should generate 9000 random seeds plus 16 corners."""
    with mp.workdps(50):
        assert len(mirror_sweep_v2._search_seeds()) == 9016


def test_tail_focus_seeds_are_reproducible_and_inside_box() -> None:
    """Tail-focus sampling should use the explicit asymmetric box."""
    left = mirror_sweep_v2._tail_focus_seeds(Random(1729), 0)[:10]
    right = mirror_sweep_v2._tail_focus_seeds(Random(1729), 0)[:10]
    assert left == right
    for seed in left:
        point = seed.point
        assert mirror_sweep_v2.TAIL_FOCUS_BOX["u"][0] <= point.u <= mirror_sweep_v2.TAIL_FOCUS_BOX["u"][1]
        assert mirror_sweep_v2.TAIL_FOCUS_BOX["v"][0] <= point.v <= mirror_sweep_v2.TAIL_FOCUS_BOX["v"][1]
        assert mirror_sweep_v2.TAIL_FOCUS_BOX["r"][0] <= point.r <= mirror_sweep_v2.TAIL_FOCUS_BOX["r"][1]
        assert mirror_sweep_v2.TAIL_FOCUS_BOX["s"][0] <= point.s <= mirror_sweep_v2.TAIL_FOCUS_BOX["s"][1]


def test_v2_classification_adds_strong_lead() -> None:
    """Stable small far residuals should receive the strong-lead label."""
    far = _point("0.1")
    berger_refs = (_fake_result(_point("0"), "1e-8"), _fake_result(_point("0"), "1e-8"))
    assert mirror_sweep_v2._classify_track_v2(_fake_track(far, ("2e-5", "3e-5")), berger_refs) == "strong_lead"
    assert mirror_sweep_v2._classify_track_v2(_fake_track(far, ("2e-4", "3e-4")), berger_refs) == "inconclusive"


def test_refinement_stage_timeout_is_nonfatal(monkeypatch) -> None:
    """A timed-out refinement stage should become a failed stage report."""

    def raise_timeout(*args, **kwargs):
        raise TimeoutError("synthetic stage timeout")

    monkeypatch.setattr(mirror_sweep_v2, "newton_refine", raise_timeout)
    report = mirror_sweep_v2._newton_refine_with_timeout(_point("0.5"), mirror_sweep_v2.ORDER6_SETTINGS)
    assert report.status == "branch_failure"
    assert report.final.failure == "synthetic stage timeout"


def test_refinement_timeout_writes_json_event(monkeypatch, tmp_path) -> None:
    """A timed-out stage should still produce a refinement-stage JSON event."""

    def raise_timeout(*args, **kwargs):
        raise TimeoutError("synthetic stage timeout")

    monkeypatch.setattr(mirror_sweep_v2, "newton_refine", raise_timeout)
    point = _point("0.5")
    seed = mirror_sweep_v2.SearchSeed(1, "near", "test", point)
    candidate = mirror_sweep_v2.SearchCandidate(seed, _fake_result(point, "1"))
    selection = mirror_sweep_v2.SelectedCandidate(1, "best", candidate)
    path = tmp_path / "events.jsonl"
    track = mirror_sweep_v2._run_order6(selection, path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert track.stages[0].status == "branch_failure"
    assert rows[0]["event"] == "refinement_stage"
    assert rows[0]["stage"]["final"]["failure"] == "synthetic stage timeout"
