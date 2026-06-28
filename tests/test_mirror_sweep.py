"""Tests for the long mirror-sweep helper layer."""

from __future__ import annotations

import json
from random import Random

from mpmath import mp

from experiments import mirror_sweep
from experiments import mirror_sweep_common
from problem import DEFAULT_CONFIG, DEFAULT_PARAMS
from solver.mirror_shooting import MirrorResidualResult, MirrorSearchPoint


def _point(u: str) -> MirrorSearchPoint:
    """Build a simple one-coordinate search point."""
    return MirrorSearchPoint(mp.mpf(u), mp.zero, mp.zero, mp.zero)


def _fake_result(point: MirrorSearchPoint, norm: str, failure: str | None = None) -> MirrorResidualResult:
    """Build one synthetic residual result."""
    return MirrorResidualResult(
        point=point,
        params=DEFAULT_PARAMS,
        config=DEFAULT_CONFIG,
        residual=(mp.mpf(norm),),
        residual_norm=mp.mpf(norm),
        match_q=None,
        l_value=mp.mpf("0.25"),
        patch_count=2,
        branch_diagnostics={"min_sum27": mp.mpf("1.5")},
        failure=failure,
    )


def _candidate(index: int, region: str, point: MirrorSearchPoint, norm: str) -> mirror_sweep.SearchCandidate:
    """Build one synthetic scout candidate."""
    seed = mirror_sweep.SearchSeed(index, region, "test", point)
    return mirror_sweep.SearchCandidate(seed, _fake_result(point, norm))


def test_region_sampling_is_reproducible_and_respects_bounds() -> None:
    """Long-sweep annular seeds should be deterministic and inside their region."""
    spec = mirror_sweep.RegionSpec("test", mp.mpf("0.25"), mp.one, (mp.one, mp.one, mp.mpf("3"), mp.one), 8, 2, 2, 1)
    with mp.workdps(50):
        left = mirror_sweep._region_seeds(spec, Random(1729), 0)
        right = mirror_sweep._region_seeds(spec, Random(1729), 0)
        assert left == right
        for seed in left:
            distance = mirror_sweep._point_distance(seed.point)
            assert spec.lower <= distance <= spec.upper
            for value, bound in zip((seed.point.u, seed.point.v, seed.point.r, seed.point.s), spec.bounds):
                assert abs(value) <= bound


def test_diverse_selection_does_not_only_pick_lowest_residuals() -> None:
    """Diverse representatives should cover separated points in the region."""
    spec = mirror_sweep.RegionSpec("near", mp.mpf("0.25"), mp.one, (mp.one, mp.one, mp.mpf("3"), mp.one), 0, 2, 1, 1)
    candidates = [
        _candidate(1, "near", _point("0.30"), "0.01"),
        _candidate(2, "near", _point("0.31"), "0.02"),
        _candidate(3, "near", _point("0.32"), "0.03"),
        _candidate(4, "near", _point("0.90"), "0.50"),
    ]
    selected = mirror_sweep._select_region_candidates(spec, candidates)
    assert len(selected) == 3
    assert [item.reason for item in selected] == ["best", "best", "diverse"]
    assert selected[-1].candidate.seed.index == 4


def test_region_selection_enforces_quotas_when_available() -> None:
    """Selection should use both best and diverse quotas for a populated region."""
    spec = mirror_sweep.RegionSpec("middle", mp.one, mp.mpf("2"), (mp.mpf("2"),) * 4, 0, 2, 2, 1)
    candidates = [_candidate(index, "middle", _point(str(1 + index / 10)), str(index / 100)) for index in range(1, 7)]
    selected = mirror_sweep._select_region_candidates(spec, candidates)
    assert len(selected) == 4
    assert [item.reason for item in selected].count("best") == 2
    assert [item.reason for item in selected].count("diverse") == 2


def test_jsonl_writer_appends_readable_events(tmp_path) -> None:
    """JSONL checkpoint events should append one valid JSON object per line."""
    path = tmp_path / "events.jsonl"
    event = mirror_sweep._event("example", {"value": mirror_sweep._mp_string(mp.mpf("1.25"))})
    mirror_sweep._write_jsonl_event(path, event)
    mirror_sweep._write_jsonl_event(path, event)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["example", "example"]
    assert mp.mpf(rows[0]["value"]) == mp.mpf("1.25")


def test_result_payload_serializes_mpmath_numbers_as_strings() -> None:
    """Residual payloads should preserve mpmath values as decimal strings."""
    payload = mirror_sweep._result_payload(_fake_result(_point("0.5"), "0.125"))
    assert payload["residual_norm"] == "0.125"
    assert payload["residual"] == ["0.125"]
    assert payload["branch_diagnostics"]["min_sum27"] == "1.5"


def test_timed_out_scout_becomes_failure_candidate(monkeypatch) -> None:
    """Scout timeouts should be recorded as ordinary failed candidates."""

    def raise_timeout(*args, **kwargs):
        raise mirror_sweep.ScoutTimeoutError("synthetic timeout")

    monkeypatch.setattr(mirror_sweep_common, "mirror_residual", raise_timeout)
    seed = mirror_sweep.SearchSeed(1, "near", "test", _point("0.5"))
    candidate = mirror_sweep._evaluate_seed_with_timeout(seed)
    assert candidate.result.failure == "synthetic timeout"
    assert candidate.result.residual_norm == mp.inf
