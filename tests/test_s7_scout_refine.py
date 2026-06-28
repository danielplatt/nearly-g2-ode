"""Tests for refining selected S7 scout local minima."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mpmath import mp

from experiments.s7 import scout_refine as refine
from experiments.s7 import search_common as s7


def _point(u: str | float | int = 0, v: str | float | int = 0, r: str | float | int = 0) -> s7.S7SearchPoint:
    """Return a synthetic S7 search point."""
    return s7.S7SearchPoint(mp.mpf(u), mp.mpf(v), mp.mpf(r), mp.zero)


def _fake_result(
    target: str,
    point: s7.S7SearchPoint,
    norm: str,
    *,
    failure: str | None = None,
) -> s7.S7ResidualResult:
    """Return a synthetic S7 residual result."""
    params, config = s7.params_from_s7_scaled(
        point,
        base_params=s7.TARGETS[target].params_builder(),
        template_config=s7.SCOUT_CONFIG,
    )
    residual_norm = mp.inf if failure else mp.mpf(norm)
    residual = () if failure else tuple(mp.mpf(norm) for _ in range(8))
    return s7.S7ResidualResult(
        point,
        params,
        config,
        residual,
        residual_norm,
        None,
        None,
        (0, 0),
        {},
        failure,
    )


def _candidate(index: int, target: str, point: s7.S7SearchPoint, norm: str, failure: str | None = None) -> s7.S7ScoutCandidate:
    """Return one synthetic S7 scout candidate."""
    seed = s7.S7SearchSeed(index, target, "default", "s7_grid", point)
    return s7.S7ScoutCandidate(seed, _fake_result(target, point, norm, failure=failure))


def _write_scout_jsonl(path: Path, candidates: list[s7.S7ScoutCandidate], shape: tuple[int, ...]) -> None:
    """Write a tiny completed scout checkpoint."""
    s7._write_jsonl_event(path, s7._event("run_start", {"grid": {"axis_counts": list(shape)}}))
    for candidate in candidates:
        s7._write_jsonl_event(path, s7._event("scout_result", s7._candidate_payload(candidate)))
    s7._write_jsonl_event(path, s7._event("run_summary", {"done": True}))


def test_reconstruct_s7_scout_candidates_from_payloads(tmp_path: Path) -> None:
    """Persisted S7 scout payloads should rebuild target-aware candidates."""
    path = tmp_path / "scouts.jsonl"
    candidate = _candidate(7, "round", _point("0.1", "-0.2", "0.3"), "0.0125")
    _write_scout_jsonl(path, [candidate], (1, 1, 1))

    rebuilt = refine._load_scout_candidates(path)

    assert len(rebuilt) == 1
    assert rebuilt[0].seed.index == 7
    assert rebuilt[0].seed.target == "round"
    assert rebuilt[0].seed.point == candidate.seed.point
    assert rebuilt[0].result.residual_norm == mp.mpf("0.0125")


def test_target_local_minima_treat_failures_as_infinite_neighbors() -> None:
    """Failed scouts should not be selected but should count as infinite neighbors."""
    candidates = [
        _candidate(0, "round", _point(0), "0.2"),
        _candidate(1, "round", _point(1), "0.1"),
        _candidate(2, "round", _point(2), "1", failure="branch_failure"),
    ]

    minima = refine._target_local_minima(candidates, (3, 1, 1))

    assert [candidate.seed.index for candidate in minima] == [1]


def test_local_minima_are_computed_target_wise_with_seed_offsets() -> None:
    """Round and squashed target blocks should have independent local minima."""
    candidates = [
        _candidate(0, "round", _point(0), "0.2"),
        _candidate(1, "round", _point(1), "0.05"),
        _candidate(2, "round", _point(2), "0.2"),
        _candidate(3, "squashed", _point(0), "0.3"),
        _candidate(4, "squashed", _point(1), "0.07"),
        _candidate(5, "squashed", _point(2), "0.3"),
    ]

    selected = refine._select_local_minima(candidates, {"grid": {"axis_counts": [3, 1, 1]}}, mp.mpf("0.15"))

    assert [(item.candidate.seed.target, item.candidate.seed.index) for item in selected] == [("round", 1), ("squashed", 4)]
    assert [item.rank for item in selected] == [1, 2]


def test_local_minima_cutoff_and_limit_are_deterministic() -> None:
    """Residual cutoff and optional limit should apply after sorted local-minimum selection."""
    candidates = [
        _candidate(0, "round", _point(0), "0.3"),
        _candidate(1, "round", _point(1), "0.05"),
        _candidate(2, "round", _point(2), "0.3"),
        _candidate(3, "round", _point(3), "0.12"),
        _candidate(4, "round", _point(4), "0.3"),
    ]

    selected = refine._select_local_minima(candidates, {"grid": {"axis_counts": [5, 1, 1]}}, mp.mpf("0.15"), limit=1)

    assert [item.candidate.seed.index for item in selected] == [1]


def test_dry_run_reports_selected_minima_without_output(tmp_path: Path, capsys, monkeypatch) -> None:
    """Dry-run should print selections and return before writing refinement files."""
    scout_path = tmp_path / "scouts.jsonl"
    candidates = [
        _candidate(0, "round", _point(0), "0.2"),
        _candidate(1, "round", _point(1), "0.05"),
        _candidate(2, "round", _point(2), "0.2"),
    ]
    _write_scout_jsonl(scout_path, candidates, (3, 1, 1))
    monkeypatch.setattr(refine, "OUTPUT_DIR", tmp_path / "refinements")

    refine.main(["--dry-run", "--scout-jsonl", str(scout_path)])

    out = capsys.readouterr().out
    assert "selected local minima: 1" in out
    assert "seed=1" in out
    assert not (tmp_path / "refinements").exists()


def test_checkpoint_compatibility_distinguishes_cutoff_and_targets(tmp_path: Path, monkeypatch) -> None:
    """Resume matching should include the source scout and selection config."""
    monkeypatch.setattr(refine, "OUTPUT_DIR", tmp_path / "refinements")
    scout_path = tmp_path / "scouts.jsonl"
    jsonl_path, summary_path = refine._refinement_output_paths(datetime(2026, 6, 13, 12, 0, 0))
    timeout = float(refine.DEFAULT_CANDIDATE_TIMEOUT_SECONDS)
    targets = ("round",)
    s7._write_jsonl_event(
        jsonl_path,
        s7._event(
            "run_start",
            refine._run_start_payload(
                jsonl_path,
                summary_path,
                scout_path,
                mp.mpf("0.15"),
                timeout,
                targets,
                None,
            ),
        ),
    )

    assert refine._checkpoint_is_compatible(jsonl_path, scout_path, mp.mpf("0.15"), timeout, targets, None)
    assert not refine._checkpoint_is_compatible(jsonl_path, scout_path, mp.mpf("0.2"), timeout, targets, None)
    assert not refine._checkpoint_is_compatible(jsonl_path, scout_path, mp.mpf("0.15"), timeout, ("squashed",), None)


def test_timeout_track_is_nonfatal_failed_classification() -> None:
    """A timeout should produce a serializable failed track rather than raising."""
    selection = refine.SelectedS7Candidate(1, "local-minimum", _candidate(1, "round", _point("0.1"), "0.01"))

    track = refine._timeout_track(selection, "timeout")
    payload = refine._classification_payload(track)

    assert track.classification == "failed"
    assert payload["classification"] == "failed"
    assert payload["stages"][0]["status"] == "timeout"
