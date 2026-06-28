"""Tests for processing calibrated non-mirrored grid scouts."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from mpmath import mp

from experiments.berger_space import non_mirrored_grid_refine as refine
from experiments.shared.non_mirrored_common import SearchCandidate, SearchSeed, _candidate_payload, _event, _write_jsonl_event
from solver.two_sided_refinement import TwoSidedCandidateTrack, TwoSidedRefinementStageReport
from solver.two_sided_shooting import BASE_TWO_SIDED_POINT, TwoSidedResidualResult, TwoSidedSearchPoint, params_from_two_sided_scaled


def _point(*values: str | float | int) -> TwoSidedSearchPoint:
    """Return a synthetic 7D scaled search point."""
    padded = list(values) + [0] * (7 - len(values))
    return TwoSidedSearchPoint(*(mp.mpf(value) for value in padded))


def _fake_result(
    point: TwoSidedSearchPoint,
    norm: str = "1e-3",
    *,
    config=refine.SCOUT_CONFIG,
    failure: str | None = None,
    base_params=refine.DEFAULT_PARAMS,
) -> TwoSidedResidualResult:
    """Return a synthetic residual result for fast refinement tests."""
    params, concrete_config = params_from_two_sided_scaled(point, base_params=base_params, template_config=config)
    residual_norm = mp.inf if failure else mp.mpf(norm)
    residual = () if failure else tuple(mp.mpf(norm) for _ in range(8))
    return TwoSidedResidualResult(
        point,
        params,
        concrete_config,
        residual,
        residual_norm,
        None,
        None,
        None,
        None,
        (0, 0),
        {},
        failure,
    )


def _candidate(
    index: int,
    point: TwoSidedSearchPoint,
    norm: str,
    failure: str | None = None,
    *,
    region: str = "near",
    base_params=refine.DEFAULT_PARAMS,
) -> SearchCandidate:
    """Return a synthetic grid scout candidate."""
    seed = SearchSeed(index, region, "calibrated_grid", point)
    return SearchCandidate(seed, _fake_result(point, norm, failure=failure, base_params=base_params))


def _stage(
    point: TwoSidedSearchPoint,
    initial_norm: str,
    final_norm: str,
    status: str = "max_steps",
    *,
    base_params=refine.DEFAULT_PARAMS,
) -> TwoSidedRefinementStageReport:
    """Return a synthetic refinement stage."""
    return TwoSidedRefinementStageReport(
        refine.ORDER6_SETTINGS,
        _fake_result(point, initial_norm, config=refine.ORDER6_CONFIG, base_params=base_params),
        _fake_result(point, final_norm, config=refine.ORDER6_CONFIG, base_params=base_params),
        (),
        status,
    )


def _track(
    point: TwoSidedSearchPoint,
    *,
    scout_norm: str = "1",
    final_norm: str | None = None,
    verification_norms: tuple[str, ...] = (),
    region: str = "near",
    base_params=refine.DEFAULT_PARAMS,
) -> TwoSidedCandidateTrack:
    """Return a synthetic track with optional final stage and verifications."""
    scout = _fake_result(point, scout_norm, base_params=base_params)
    stages = () if final_norm is None else (_stage(point, scout_norm, final_norm, base_params=base_params),)
    verifications = tuple(_fake_result(point, norm, config=config, base_params=base_params) for norm, config in zip(verification_norms, refine.VERIFY_CONFIGS))
    return TwoSidedCandidateTrack(1, region, point, scout, stages, verifications, "inconclusive")


def _write_scout_jsonl(path: Path, candidates: list[SearchCandidate], shape: tuple[int, ...], region: str = "near") -> None:
    """Write a tiny scout checkpoint with persisted scout_result payloads."""
    _write_jsonl_event(path, _event("run_start", {"grid": {"axis_counts": list(shape), "region": region}}))
    for candidate in candidates:
        _write_jsonl_event(path, _event("scout_result", _candidate_payload(candidate)))


def test_reconstruct_candidates_from_scout_payloads(tmp_path: Path) -> None:
    """Persisted scout_result payloads should rebuild SearchCandidate objects."""
    candidate = _candidate(7, _point("0.1", "-0.2", "0.3", "0.4", "-0.5", "0.6", "-0.1"), "0.0125")
    path = tmp_path / "scouts.jsonl"
    _write_scout_jsonl(path, [candidate], (1, 1, 1, 1, 1, 1, 1))

    rebuilt = refine._load_scout_candidates(path)

    assert len(rebuilt) == 1
    assert rebuilt[0].seed.index == 7
    assert rebuilt[0].seed.point == candidate.seed.point
    assert rebuilt[0].result.residual_norm == mp.mpf("0.0125")


def test_reconstruct_positive_ac_candidate_uses_region_base(tmp_path: Path) -> None:
    """Positive-ac scout payloads should rebuild physical parameters in the positive-ac chart."""
    base_params = refine.grid_search.POSITIVE_AC_BASE_PARAMS
    point = _point("0.25", "-0.1", "0.5", "-0.2", "0.15", "-0.25", "-0.4")
    candidate = _candidate(9, point, "0.02", region="positive-ac", base_params=base_params)
    path = tmp_path / "positive_ac_scouts.jsonl"
    _write_scout_jsonl(path, [candidate], (1, 1, 1, 1, 1, 1, 1), region="positive-ac")

    rebuilt = refine._load_scout_candidates(path)[0]
    expected, _ = params_from_two_sided_scaled(point, base_params=base_params, template_config=refine.SCOUT_CONFIG)
    default, _ = params_from_two_sided_scaled(point, template_config=refine.SCOUT_CONFIG)

    assert rebuilt.result.params.left.a == expected.left.a
    assert rebuilt.result.params.right.d == expected.right.d
    assert rebuilt.result.params.left.c != default.left.c
    assert rebuilt.result.params.right.f != default.right.f


def test_local_minima_treat_failures_as_infinite() -> None:
    """Failed scouts should not be selected but should count as infinite neighbors."""
    candidates = [
        _candidate(0, _point(0), "0.2"),
        _candidate(1, _point(1), "0.1"),
        _candidate(2, _point(2), "1", failure="branch_failure"),
    ]

    minima = refine._local_minima(candidates, (3, 1, 1, 1, 1, 1, 1))

    assert [candidate.seed.index for candidate in minima] == [1]


def test_deduplicate_exact_left_right_swaps_keeps_lower_seed_index() -> None:
    """Exact left/right-swapped scouts should canonicalize to one candidate."""
    high = _candidate(5, _point(1, 2, 3, 4, 5, 6, 0), "0.01")
    low = _candidate(2, _point(4, 5, 6, 1, 2, 3, 0), "0.02")

    deduped = refine._dedupe_left_right([high, low])

    assert [candidate.seed.index for candidate in deduped] == [2]


def test_select_candidates_balances_local_best_and_diverse(monkeypatch) -> None:
    """The v1 selector should deterministically fill the balanced 50-candidate quota."""
    locals_best = [
        _candidate(index, _point("0", "0", "0", mp.mpf("0.1") + index * mp.mpf("1e-5"), "0", "0", "0"), mp.nstr(mp.mpf("0.001") + index * mp.mpf("0.001"), 30))
        for index in range(40)
    ]
    diverse = [
        _candidate(
            40 + index,
            _point(mp.mpf(index) / 20, "0", "0", mp.mpf("1.0") + index * mp.mpf("0.01"), "0", "0", "0"),
            mp.nstr(mp.mpf("0.08") + index * mp.mpf("0.001"), 30),
        )
        for index in range(20)
    ]
    candidates = locals_best + diverse
    monkeypatch.setattr(refine, "_local_minima", lambda _candidates, _shape: candidates)

    selected = refine._select_candidates(candidates, {"grid": {"axis_counts": [1, 1, 1, 1, 1, 1, 1]}}, quota=50)

    reasons = Counter(selection.reason for selection in selected)
    assert len(selected) == 50
    assert reasons == {"local-best": 36, "local-diverse": 14}
    assert [selection.rank for selection in selected] == list(range(1, 51))
    assert [selection.candidate.seed.index for selection in selected[:3]] == [0, 1, 2]


def test_local_minima_mode_selects_all_sorted_branch_valid_minima() -> None:
    """Local-minima mode should select all canonical branch-valid local minima."""
    candidates = [
        _candidate(0, _point(0), "0.3"),
        _candidate(1, _point(1), "0.1"),
        _candidate(2, _point(2), "1", failure="branch_failure"),
        _candidate(3, _point(3), "0.05"),
        _candidate(4, _point(4), "0.2"),
    ]

    selected = refine._select_for_mode(
        candidates,
        {"grid": {"axis_counts": [5, 1, 1, 1, 1, 1, 1]}},
        refine.LOCAL_MINIMA_SELECTION_MODE,
        None,
        None,
    )

    assert [selection.candidate.seed.index for selection in selected] == [3, 1]
    assert [selection.reason for selection in selected] == ["local-minimum", "local-minimum"]
    assert [selection.rank for selection in selected] == [1, 2]


def test_local_minima_mode_deduplicates_left_right_swaps(monkeypatch) -> None:
    """Local-minima mode should canonicalize exact left/right-swapped points."""
    high = _candidate(5, _point(1, 2, 3, 4, 5, 6, 0), "0.01")
    low = _candidate(2, _point(4, 5, 6, 1, 2, 3, 0), "0.02")
    monkeypatch.setattr(refine, "_local_minima", lambda _candidates, _shape: [high, low])

    selected = refine._select_for_mode(
        [high, low],
        {"grid": {"axis_counts": [1, 1, 1, 1, 1, 1, 1]}},
        refine.LOCAL_MINIMA_SELECTION_MODE,
        None,
        None,
    )

    assert [selection.candidate.seed.index for selection in selected] == [2]


def test_local_minima_mode_filters_by_residual_cutoff(monkeypatch) -> None:
    """The optional local-minimum residual cutoff should be deterministic."""
    candidates = [
        _candidate(0, _point(0), "0.05"),
        _candidate(1, _point(1), "0.12"),
        _candidate(2, _point(2), "0.21"),
    ]
    monkeypatch.setattr(refine, "_local_minima", lambda _candidates, _shape: candidates)

    selected = refine._select_for_mode(
        candidates,
        {"grid": {"axis_counts": [1, 1, 1, 1, 1, 1, 1]}},
        refine.LOCAL_MINIMA_SELECTION_MODE,
        None,
        mp.mpf("0.15"),
    )

    assert [selection.candidate.seed.index for selection in selected] == [0, 1]


def test_dry_run_reports_selected_seeds_without_refinement_output(tmp_path: Path, capsys, monkeypatch) -> None:
    """Dry-run should print the selected batch and return before writing refinement stages."""
    scout_path = tmp_path / "scouts.jsonl"
    _write_scout_jsonl(scout_path, [_candidate(0, BASE_TWO_SIDED_POINT, "0.01")], (1, 1, 1, 1, 1, 1, 1))
    monkeypatch.setattr(refine, "OUTPUT_DIR", tmp_path / "refinements")

    refine.main(["--dry-run", "--quota", "1", "--scout-jsonl", str(scout_path)])

    out = capsys.readouterr().out
    assert "selected candidates: 1" in out
    assert "seed=0" in out
    assert not (tmp_path / "refinements").exists()


def test_dry_run_local_minima_reports_selected_seeds_without_refinement_output(tmp_path: Path, capsys, monkeypatch) -> None:
    """Local-minima dry-run should print selected minima and avoid output writes."""
    scout_path = tmp_path / "scouts.jsonl"
    candidates = [
        _candidate(0, _point(0), "0.3"),
        _candidate(1, _point(1), "0.1"),
        _candidate(2, _point(2), "1", failure="branch_failure"),
    ]
    _write_scout_jsonl(scout_path, candidates, (3, 1, 1, 1, 1, 1, 1))
    monkeypatch.setattr(refine, "OUTPUT_DIR", tmp_path / "refinements")

    refine.main(["--dry-run", "--selection-mode", "local-minima", "--scout-jsonl", str(scout_path)])

    out = capsys.readouterr().out
    assert "selected candidates: 1" in out
    assert "seed=1" in out
    assert "reason=local-minimum" in out
    assert not (tmp_path / "refinements").exists()


def test_checkpoint_compatibility_distinguishes_selection_mode_and_cutoff(tmp_path: Path, monkeypatch) -> None:
    """Resume matching should include selection mode and local-minimum cutoff."""
    monkeypatch.setattr(refine, "OUTPUT_DIR", tmp_path / "refinements")
    scout_path = tmp_path / "scouts.jsonl"
    _write_scout_jsonl(scout_path, [_candidate(0, BASE_TWO_SIDED_POINT, "0.01")], (1, 1, 1, 1, 1, 1, 1))
    jsonl_path, summary_path = refine._output_paths(datetime(2026, 5, 24, 12, 0, 0))
    order6_timeout = float(refine.ORDER6_TIMEOUT_SECONDS)
    promotion_timeout = float(refine.PROMOTION_TIMEOUT_SECONDS)
    _write_jsonl_event(
        jsonl_path,
        _event(
            "run_start",
            refine._run_start_payload(
                jsonl_path,
                summary_path,
                scout_path,
                None,
                order6_timeout,
                promotion_timeout,
                refine.LOCAL_MINIMA_SELECTION_MODE,
                mp.mpf("0.15"),
            ),
        ),
    )

    assert refine._checkpoint_is_compatible(
        jsonl_path,
        scout_path,
        None,
        order6_timeout,
        promotion_timeout,
        refine.LOCAL_MINIMA_SELECTION_MODE,
        mp.mpf("0.15"),
    )
    assert not refine._checkpoint_is_compatible(
        jsonl_path,
        scout_path,
        50,
        order6_timeout,
        promotion_timeout,
        refine.BALANCED_SELECTION_MODE,
        None,
    )
    assert not refine._checkpoint_is_compatible(
        jsonl_path,
        scout_path,
        None,
        order6_timeout,
        promotion_timeout,
        refine.LOCAL_MINIMA_SELECTION_MODE,
        mp.mpf("0.2"),
    )


def test_checkpoint_compatibility_distinguishes_scout_region(tmp_path: Path, monkeypatch) -> None:
    """Resume matching should not mix default and positive-ac coordinate charts."""
    monkeypatch.setattr(refine, "OUTPUT_DIR", tmp_path / "refinements")
    scout_path = tmp_path / "positive_ac_scouts.jsonl"
    candidate = _candidate(0, BASE_TWO_SIDED_POINT, "0.01", region="positive-ac", base_params=refine.grid_search.POSITIVE_AC_BASE_PARAMS)
    _write_scout_jsonl(scout_path, [candidate], (1, 1, 1, 1, 1, 1, 1), region="positive-ac")
    jsonl_path, summary_path = refine._output_paths(datetime(2026, 5, 24, 12, 0, 0))
    order6_timeout = float(refine.ORDER6_TIMEOUT_SECONDS)
    promotion_timeout = float(refine.PROMOTION_TIMEOUT_SECONDS)
    _write_jsonl_event(
        jsonl_path,
        _event(
            "run_start",
            refine._run_start_payload(
                jsonl_path,
                summary_path,
                scout_path,
                1,
                order6_timeout,
                promotion_timeout,
                refine.BALANCED_SELECTION_MODE,
                None,
                "positive-ac",
            ),
        ),
    )

    assert refine._checkpoint_is_compatible(
        jsonl_path,
        scout_path,
        1,
        order6_timeout,
        promotion_timeout,
        refine.BALANCED_SELECTION_MODE,
        None,
        "positive-ac",
    )
    assert not refine._checkpoint_is_compatible(
        jsonl_path,
        scout_path,
        1,
        order6_timeout,
        promotion_timeout,
        refine.BALANCED_SELECTION_MODE,
        None,
        "near",
    )


def test_checkpoint_compatibility_distinguishes_newton_coordinate_guard(tmp_path: Path, monkeypatch) -> None:
    """Symmetric-alpha refinements should not resume checkpoints with the old guard."""
    monkeypatch.setattr(refine, "OUTPUT_DIR", tmp_path / "refinements")
    scout_path = tmp_path / "scouts.jsonl"
    _write_scout_jsonl(scout_path, [_candidate(0, BASE_TWO_SIDED_POINT, "0.01")], (1, 1, 1, 1, 1, 1, 1))
    jsonl_path, summary_path = refine._output_paths(datetime(2026, 5, 24, 12, 0, 0))
    order6_timeout = float(refine.ORDER6_TIMEOUT_SECONDS)
    promotion_timeout = float(refine.PROMOTION_TIMEOUT_SECONDS)
    try:
        refine._configure_newton_settings(mp.mpf("4"))
        _write_jsonl_event(
            jsonl_path,
            _event(
                "run_start",
                refine._run_start_payload(
                    jsonl_path,
                    summary_path,
                    scout_path,
                    None,
                    order6_timeout,
                    promotion_timeout,
                    refine.LOCAL_MINIMA_SELECTION_MODE,
                    None,
                ),
            ),
        )
        assert refine._checkpoint_is_compatible(
            jsonl_path,
            scout_path,
            None,
            order6_timeout,
            promotion_timeout,
            refine.LOCAL_MINIMA_SELECTION_MODE,
            None,
        )
        refine._configure_newton_settings(refine.DEFAULT_MAX_NEWTON_COORDINATE)
        assert not refine._checkpoint_is_compatible(
            jsonl_path,
            scout_path,
            None,
            order6_timeout,
            promotion_timeout,
            refine.LOCAL_MINIMA_SELECTION_MODE,
            None,
        )
    finally:
        refine._configure_newton_settings(refine.DEFAULT_MAX_NEWTON_COORDINATE)


def test_resume_skips_already_classified_seed(tmp_path: Path, capsys, monkeypatch) -> None:
    """A compatible incomplete refinement checkpoint should not rerun classified seeds."""
    scout_path = tmp_path / "scouts.jsonl"
    candidate = _candidate(0, BASE_TWO_SIDED_POINT, "0.01")
    _write_scout_jsonl(scout_path, [candidate], (1, 1, 1, 1, 1, 1, 1))
    monkeypatch.setattr(refine, "OUTPUT_DIR", tmp_path / "refinements")
    jsonl_path, summary_path = refine._output_paths(datetime(2026, 5, 24, 12, 0, 0))
    order6_timeout = float(refine.ORDER6_TIMEOUT_SECONDS)
    promotion_timeout = float(refine.PROMOTION_TIMEOUT_SECONDS)
    _write_jsonl_event(jsonl_path, _event("run_start", refine._run_start_payload(jsonl_path, summary_path, scout_path, 1, order6_timeout, promotion_timeout)))
    selection = refine.SelectedGridCandidate(1, "local-best", candidate)
    _write_jsonl_event(jsonl_path, _event("candidate_selected", refine._candidate_selected_payload(selection)))
    classified = TwoSidedCandidateTrack(0, "near", candidate.seed.point, candidate.result, (), (), "inconclusive")
    _write_jsonl_event(jsonl_path, _event("candidate_classification", refine._classification_payload(classified)))
    monkeypatch.setattr(refine, "_reference_residuals", lambda *_args, **_kwargs: (_fake_result(BASE_TWO_SIDED_POINT, "1e-12"),) * 2)
    monkeypatch.setattr(refine, "_run_order6", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should skip classified seeds")))

    refine.main(["--scout-jsonl", str(scout_path), "--quota", "1"])

    out = capsys.readouterr().out
    assert "reused completed classification" in out
    events = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events].count("candidate_classification") == 1
    assert events[-1]["event"] == "run_summary"


def test_promotion_timeout_writes_failed_classification(tmp_path: Path, monkeypatch) -> None:
    """Promotion timeout should be a persisted failed classification, not an exception."""
    point = BASE_TWO_SIDED_POINT
    scout = _fake_result(point, "1")
    stage = _stage(point, "1", "0.5")
    track = TwoSidedCandidateTrack(3, "near", point, scout, (stage,), (), "inconclusive")
    path = tmp_path / "refine.jsonl"
    refs = (_fake_result(point, "1e-12"), _fake_result(point, "1e-12"))
    monkeypatch.setattr(refine, "_run_with_timeout", lambda _callback, _timeout: (None, "timeout"))

    classified = refine._promote_track(track, refs, path, 1)

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert classified.classification == "failed"
    assert events[-1]["event"] == "candidate_classification"
    assert events[-1]["classification"] == "failed"
    assert events[-1]["stages"][-1]["status"] == "timeout"


def test_order6_refinement_receives_scout_base_params(tmp_path: Path, monkeypatch) -> None:
    """The first Newton stage should run in the scout region's physical chart."""
    base_params = refine.grid_search.POSITIVE_AC_BASE_PARAMS
    point = _point("0.1", "0", "0", "0", "0", "0", "0")
    candidate = _candidate(4, point, "0.5", region="positive-ac", base_params=base_params)
    selection = refine.SelectedGridCandidate(1, "local-minimum", candidate)
    path = tmp_path / "refine.jsonl"
    captured = {}

    def fake_newton(newton_point, settings, *, base_params):
        captured["point"] = newton_point
        captured["settings"] = settings
        captured["base_params"] = base_params
        return _stage(newton_point, "0.5", "0.25", base_params=base_params)

    monkeypatch.setattr(refine, "two_sided_newton_refine", fake_newton)

    refine._run_order6(selection, path, 1, base_params)

    assert captured["point"] == point
    assert captured["settings"] == refine.ORDER6_SETTINGS
    assert captured["base_params"] == base_params


def test_best_track_physical_payload_uses_scout_base() -> None:
    """Best-track summaries should report physical parameters in the selected chart."""
    base_params = refine.grid_search.POSITIVE_AC_BASE_PARAMS
    point = _point("0.2", "-0.1", "0.3", "0.4", "-0.2", "0.1", "-0.5")
    track = _track(point, final_norm="1e-9", verification_norms=("1e-9", "2e-9"), region="positive-ac", base_params=base_params)

    payload = refine._best_verified_tracks([track], base_params, limit=1)[0]
    expected, _ = params_from_two_sided_scaled(point, base_params=base_params)

    assert mp.mpf(payload["physical_parameters"]["a"]) == expected.left.a
    assert mp.mpf(payload["physical_parameters"]["d"]) == expected.right.d


def test_classification_labels_synthetic_tracks() -> None:
    """Synthetic tracks should exercise the planned classifier labels."""
    refs = (_fake_result(BASE_TWO_SIDED_POINT, "1e-12"), _fake_result(BASE_TWO_SIDED_POINT, "1e-12"))
    recovered = _track(BASE_TWO_SIDED_POINT, verification_norms=("1e-10", "2e-10"))
    artifact = _track(BASE_TWO_SIDED_POINT, scout_norm="1e-10", verification_norms=("1e-3", "2e-3"))
    far = _point("0.2", "0", "0", "-0.2", "0", "0", "0")
    possible = _track(far, verification_norms=("1e-9", "2e-9"))
    inconclusive = _track(BASE_TWO_SIDED_POINT)

    assert refine._classify_track(recovered, refs) == "recovered_berger"
    assert refine._classify_track(artifact, refs) == "finite_order_artifact"
    assert refine._classify_track(possible, refs) == "possible_non_mirrored_candidate"
    assert refine._classify_track(inconclusive, refs) == "inconclusive"


def test_non_default_chart_base_is_not_called_recovered_berger() -> None:
    """A non-default chart's zero point should not be labeled as recovered Berger."""
    refs = (_fake_result(BASE_TWO_SIDED_POINT, "1e-12"), _fake_result(BASE_TWO_SIDED_POINT, "1e-12"))
    base_track = _track(
        BASE_TWO_SIDED_POINT,
        verification_norms=("1e-10", "2e-10"),
        region="positive-ac",
        base_params=refine.grid_search.POSITIVE_AC_BASE_PARAMS,
    )

    assert refine._classify_track(base_track, refs, allow_recovered_berger=False) == "possible_symmetric_non_berger_candidate"
