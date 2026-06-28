"""Tests for S7 terminal-offset full-moduli local-minimum refinement."""

from __future__ import annotations

from pathlib import Path

from mpmath import mp

from experiments.s7 import full_moduli_offset_refine as refine
from experiments.s7 import full_moduli_offset_scout as scout
from experiments.shared.non_mirrored_common import _write_jsonl_event


def _point(value: str | float | int = 0) -> scout.FullModuliOffsetPoint:
    """Return a synthetic 7D terminal-offset point."""
    x = mp.mpf(value)
    return scout.FullModuliOffsetPoint(x, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero)


def _result(
    index: int,
    target: str,
    point: scout.FullModuliOffsetPoint,
    norm: str,
    failure: str | None = None,
) -> scout.FullModuliOffsetResult:
    """Return a synthetic terminal-offset scout result."""
    seed = scout.FullModuliOffsetSeed(index, target, point)
    return scout.FullModuliOffsetResult(
        seed=seed,
        residual=() if failure else tuple(mp.mpf(norm) for _ in range(8)),
        residual_norm=mp.inf if failure else mp.mpf(norm),
        raw_residual_norm=mp.inf if failure else mp.mpf(norm),
        germ_residual_norm=mp.mpf("1e-5"),
        germ_success=True,
        germ_evaluations=1,
        left_l=None,
        right_l=None,
        patch_counts=(0, 0),
        failure=failure,
    )


def _candidate(
    index: int,
    target: str,
    point: scout.FullModuliOffsetPoint,
    norm: str,
    failure: str | None = None,
) -> refine.OffsetScoutCandidate:
    """Return one synthetic terminal-offset candidate."""
    result = _result(index, target, point, norm, failure)
    return refine.OffsetScoutCandidate(result.seed, result)


def _write_scout_jsonl(path: Path, candidates: list[refine.OffsetScoutCandidate], axis_count: int) -> None:
    """Write a tiny completed terminal-offset scout checkpoint."""
    _write_jsonl_event(path, {"event": "run_start", "axis_count": axis_count, "targets": ["round", "squashed"]})
    for candidate in candidates:
        _write_jsonl_event(path, {"event": "scout_result", **scout._result_payload(candidate.result)})
    _write_jsonl_event(path, {"event": "run_summary", "done": True})


def test_reconstruct_offset_scout_candidates_from_payloads(tmp_path: Path) -> None:
    """Persisted offset scout payloads should rebuild 7D candidates."""
    path = tmp_path / "scouts.jsonl"
    candidate = _candidate(7, "round", _point("0.125"), "0.0125")
    _write_scout_jsonl(path, [candidate], 1)

    rebuilt = refine._load_scout_candidates(path)

    assert len(rebuilt) == 1
    assert rebuilt[0].seed.index == 7
    assert rebuilt[0].seed.target == "round"
    assert rebuilt[0].seed.point == candidate.seed.point
    assert rebuilt[0].result.residual_norm == mp.mpf("0.0125")


def test_target_local_minima_use_target_blocks_and_failures_as_infinite() -> None:
    """Local minima should be computed target-wise and skip failed centers."""
    shape = (3, 1, 1, 1, 1, 1, 1)
    candidates = [
        _candidate(0, "round", _point(0), "0.2"),
        _candidate(1, "round", _point(1), "0.05"),
        _candidate(2, "round", _point(2), "1", failure="branch_failure"),
        _candidate(3, "squashed", _point(0), "0.3"),
        _candidate(4, "squashed", _point(1), "0.07"),
        _candidate(5, "squashed", _point(2), "0.3"),
    ]

    selected = refine._select_local_minima(candidates, {"axis_count": 3}, mp.mpf("0.15"))

    assert [(item.candidate.seed.target, item.candidate.seed.index) for item in selected] == [("round", 1), ("squashed", 4)]
    assert [candidate.seed.index for candidate in refine._target_local_minima(candidates[:3], shape)] == [1]


def test_dry_run_reports_selected_offset_minima_without_output(tmp_path: Path, capsys, monkeypatch) -> None:
    """Dry-run should select minima and avoid writing refinement output."""
    path = tmp_path / "scouts.jsonl"
    candidates = [
        _candidate(0, "round", _point(0), "0.2"),
        _candidate(1, "round", _point(1), "0.05"),
        _candidate(2, "round", _point(2), "0.2"),
    ]
    _write_scout_jsonl(path, candidates, 3)
    monkeypatch.setattr(refine, "OUTPUT_DIR", tmp_path / "refinements")

    refine.main(["--dry-run", "--scout-jsonl", str(path)])

    out = capsys.readouterr().out
    assert "selected local minima: 1" in out
    assert "seed=1" in out
    assert not (tmp_path / "refinements").exists()


def test_synthetic_recovered_classification() -> None:
    """A near-center low-verification track should classify as target recovery."""
    point = scout.FullModuliOffsetPoint(*(mp.zero for _ in scout.COORDINATE_NAMES))
    result = _result(0, "round", point, "1e-12")
    residual = refine.OffsetResidualResult(
        target="round",
        point=point,
        config=refine.VERIFY14_CONFIG,
        residual=tuple(mp.mpf("1e-12") for _ in range(8)),
        residual_norm=mp.mpf("1e-12"),
        raw_residual_norm=mp.mpf("1e-12"),
        germ_residual_norm=mp.zero,
        germ_success=True,
        germ_evaluations=0,
        left_l=None,
        right_l=None,
        patch_counts=(0, 0),
    )
    settings = refine._settings_for_max_coordinate(mp.mpf("1"))[0]
    stage = refine.OffsetRefinementStageReport(settings, residual, residual, (), "tolerance_hit")
    track = refine.OffsetCandidateTrack(0, "round", point, result, (stage,), (residual, residual), "inconclusive")

    assert refine._classify_track(track) == "recovered_round_s7"
