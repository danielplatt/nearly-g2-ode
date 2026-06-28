"""Deep mirror sweep with a lower bound on the midpoint length."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from random import Random

from mpmath import mp

from problem import DEFAULT_CONFIG, SolverConfig
from solver.mirror_refinement import CandidateTrack, NewtonSettings, RefinementStageReport, newton_refine
from solver.mirror_shooting import BASE_POINT, MirrorResidualResult, MirrorSearchPoint, mirror_residual, params_from_scaled

from .mirror_search import _classify_track, _replace_track, _stable_within_factor, _track_final_result
from ..shared.mirror_sweep_common import (
    BoxRegionSpec,
    SearchCandidate,
    SearchSeed,
    SelectedCandidate,
    _box_region_seeds,
    _candidate_payload,
    _corner_seeds as _common_corner_seeds,
    _event,
    _evaluate_seed as _common_evaluate_seed,
    _mp_string,
    _output_paths as _common_output_paths,
    _point_distance,
    _point_payload,
    _region_summary as _common_region_summary,
    _result_payload,
    _select_region_candidates,
    _stage_payload,
    _track_payload as _common_track_payload,
    _with_timeout,
    _write_jsonl_event,
    _write_summary,
    _print_region_best as _common_print_region_best,
    _print_region_summary,
)


def _high_precision_mpf(value: str) -> mp.mpf:
    """Return one decimal literal initialized at high precision."""
    with mp.workdps(80):
        return mp.mpf(value)


MIN_MATCH_T = _high_precision_mpf("0.01")


def _midpoint_floor_s_coordinate() -> mp.mpf:
    """Return the scaled-coordinate value equivalent to m = 0.01."""
    with mp.workdps(80):
        return mp.log(MIN_MATCH_T / DEFAULT_CONFIG.match_t)


S_MIN = _midpoint_floor_s_coordinate()
S_MARGIN = _high_precision_mpf("1e-8")

SCOUT_CONFIG = SolverConfig(4, 30, 15, mp.mpf("0.95"), 0, DEFAULT_CONFIG.match_t)
ORDER6_CONFIG = SolverConfig(6, 40, 20, mp.mpf("0.9"), 0, DEFAULT_CONFIG.match_t)
ORDER10_CONFIG = SolverConfig(10, 70, 30, mp.mpf("0.7"), 1, DEFAULT_CONFIG.match_t)
VERIFY14_CONFIG = SolverConfig(14, 90, 35, mp.mpf("0.6"), 2, DEFAULT_CONFIG.match_t)
VERIFY18_CONFIG = SolverConfig(18, 110, 45, mp.mpf("0.55"), 2, DEFAULT_CONFIG.match_t)
VERIFY22_CONFIG = SolverConfig(22, 130, 50, mp.mpf("0.5"), 2, DEFAULT_CONFIG.match_t)
VERIFY_CONFIGS = (VERIFY14_CONFIG, VERIFY18_CONFIG)

RANDOM_SEED = 1729
SWEEP_VERSION = "v3"
OUTPUT_DIR = Path("output/mirror_sweeps")
SCOUT_TIMEOUT_SECONDS = 120
STAGE_TIMEOUT_SECONDS = 8 * 60
MAX_NEWTON_COORDINATE = mp.mpf("12")
ORDER6_SETTINGS = NewtonSettings(
    "order-6",
    ORDER6_CONFIG,
    mp.mpf("1e-3"),
    mp.mpf("1e-8"),
    3,
    max_abs_coordinate=MAX_NEWTON_COORDINATE,
    min_s_coordinate=S_MIN,
)
ORDER10_SETTINGS = NewtonSettings(
    "order-10",
    ORDER10_CONFIG,
    mp.mpf("3e-4"),
    mp.mpf("1e-10"),
    3,
    max_abs_coordinate=MAX_NEWTON_COORDINATE,
    min_s_coordinate=S_MIN,
)

S_FLOOR = S_MIN + S_MARGIN
REGIONS = (
    BoxRegionSpec("near_control", mp.mpf("0.25"), mp.one, ((-mp.one, mp.one), (-mp.one, mp.one), (mp.mpf("-3"), mp.mpf("3")), (-mp.one, mp.one)), 1000, 4, 4, 1),
    BoxRegionSpec("middle_control", mp.one, mp.mpf("4"), ((mp.mpf("-3"), mp.mpf("3")), (mp.mpf("-3"), mp.mpf("3")), (mp.mpf("-8"), mp.mpf("8")), (S_FLOOR, mp.mpf("2"))), 1800, 8, 8, 3),
    BoxRegionSpec("negative_uv_mid_m", mp.zero, mp.inf, ((mp.mpf("-12"), mp.mpf("-4")), (mp.mpf("-12"), mp.mpf("-4")), (mp.mpf("-12"), mp.mpf("12")), (S_FLOOR, mp.mpf("-1"))), 3500, 18, 18, 8),
    BoxRegionSpec("negative_uv_large_m", mp.zero, mp.inf, ((mp.mpf("-12"), mp.mpf("-4")), (mp.mpf("-12"), mp.mpf("-4")), (mp.mpf("-12"), mp.mpf("12")), (-mp.one, mp.one)), 3000, 14, 14, 6),
    BoxRegionSpec("negative_uv_wide_r", mp.zero, mp.inf, ((mp.mpf("-12"), mp.mpf("-4")), (mp.mpf("-12"), mp.mpf("-4")), (mp.mpf("-20"), mp.mpf("20")), (S_FLOOR, mp.one)), 2500, 12, 12, 5),
    BoxRegionSpec("mixed_far", mp.mpf("4"), mp.mpf("12"), ((mp.mpf("-12"), mp.mpf("4")), (mp.mpf("-12"), mp.mpf("4")), (mp.mpf("-14"), mp.mpf("14")), (S_FLOOR, mp.mpf("2"))), 2200, 10, 10, 4),
)


def _corner_seeds(start_index: int = 0) -> list[SearchSeed]:
    """Return the fixed corner seeds, assigned by V3 distance bands."""
    return _common_corner_seeds(REGIONS, start_index)


def _search_seeds(seed: int = RANDOM_SEED) -> list[SearchSeed]:
    """Return all deterministic V3 sweep seeds."""
    rng = Random(seed)
    seeds = _corner_seeds()
    for spec in REGIONS:
        seeds.extend(_box_region_seeds(spec, rng, len(seeds)))
    return seeds


def _output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped V3 JSONL and summary output paths."""
    return _common_output_paths(OUTPUT_DIR, RANDOM_SEED, f"-{SWEEP_VERSION}", now)


def _summary_path_for_jsonl(jsonl_path: Path) -> Path:
    """Return the conventional summary path for one V3 JSONL file."""
    return jsonl_path.with_name(f"{jsonl_path.stem}-summary.json")


def _jsonl_has_event(path: Path, event_type: str) -> bool:
    """Return whether one JSONL checkpoint contains a given event type."""
    if not path.exists():
        return False
    return any(json.loads(line).get("event") == event_type for line in path.read_text(encoding="utf-8").splitlines())


def _latest_incomplete_jsonl(output_dir: Path = OUTPUT_DIR) -> Path | None:
    """Return the newest V3 JSONL without a run summary."""
    paths = sorted(output_dir.glob(f"*-seed{RANDOM_SEED}-{SWEEP_VERSION}.jsonl"), reverse=True)
    incomplete = [path for path in paths if not _jsonl_has_event(path, "run_summary")]
    return incomplete[0] if incomplete else None


def _has_post_scout_events(path: Path) -> bool:
    """Return whether a checkpoint has advanced beyond scout evaluation."""
    allowed = {"run_start", "scout_result"}
    return any(json.loads(line).get("event") not in allowed for line in path.read_text(encoding="utf-8").splitlines())


def _resume_or_new_paths(now: datetime | None = None) -> tuple[Path, Path, bool]:
    """Return checkpoint paths, preferring an incomplete V3 run if present."""
    existing = _latest_incomplete_jsonl()
    if existing is not None:
        return existing, _summary_path_for_jsonl(existing), True
    jsonl_path, summary_path = _output_paths(now)
    return jsonl_path, summary_path, False


def _point_from_payload(payload: dict) -> MirrorSearchPoint:
    """Rebuild one scaled point from JSON checkpoint data."""
    return MirrorSearchPoint(*(mp.mpf(payload[name]) for name in ("u", "v", "r", "s")))


def _result_from_payload(point: MirrorSearchPoint, payload: dict) -> MirrorResidualResult:
    """Rebuild one scout residual result from JSON checkpoint data."""
    params, config = params_from_scaled(point, template_config=SCOUT_CONFIG)
    residual = tuple(mp.mpf(value) for value in payload["residual"])
    branch = {key: mp.mpf(value) for key, value in payload["branch_diagnostics"].items()}
    l_value = None if payload["l_value"] is None else mp.mpf(payload["l_value"])
    return MirrorResidualResult(point, params, config, residual, mp.mpf(payload["residual_norm"]), None, l_value, payload["patch_count"], branch, payload["failure"])


def _candidate_from_event(event: dict) -> SearchCandidate:
    """Rebuild one scout candidate from a JSONL event."""
    point = _point_from_payload(event["seed_point"])
    seed = SearchSeed(event["seed_index"], event["region"], event["source"], point)
    return SearchCandidate(seed, _result_from_payload(point, event["result"]))


def _load_scout_candidates(path: Path) -> dict[int, SearchCandidate]:
    """Load completed scout candidates from an incomplete V3 checkpoint."""
    candidates: dict[int, SearchCandidate] = {}
    if not path.exists():
        return candidates
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") == "scout_result":
            candidate = _candidate_from_event(event)
            candidates[candidate.seed.index] = candidate
    return candidates


def _m_floor_result(seed: SearchSeed, config: SolverConfig) -> SearchCandidate:
    """Return a failed scout candidate below the allowed midpoint floor."""
    params, local_config = params_from_scaled(seed.point, template_config=config)
    result = MirrorResidualResult(seed.point, params, local_config, (), mp.inf, None, None, 0, {}, "m_floor_rejected")
    return SearchCandidate(seed, result)


def _evaluate_seed(seed: SearchSeed, path: Path) -> SearchCandidate:
    """Evaluate and persist one scout seed, enforcing the midpoint floor."""
    if seed.point.s <= S_MIN:
        candidate = _m_floor_result(seed, SCOUT_CONFIG)
        _write_jsonl_event(path, _event("scout_result", _candidate_payload(candidate)))
        return candidate
    return _common_evaluate_seed(seed, path, SCOUT_CONFIG, SCOUT_TIMEOUT_SECONDS)


def _verify_point(point: MirrorSearchPoint, include_order22: bool = False) -> tuple[MirrorResidualResult, ...]:
    """Evaluate one point at the high-order verification configs."""
    configs = VERIFY_CONFIGS + ((VERIFY22_CONFIG,) if include_order22 else ())
    results = []
    for config in configs:
        with mp.workdps(config.working_dps):
            results.append(mirror_residual(point, config))
    return tuple(results)


def _stage_timeout_report(point: MirrorSearchPoint, settings: NewtonSettings, message: str) -> RefinementStageReport:
    """Return a nonfatal refinement-stage timeout report."""
    params, config = params_from_scaled(point, template_config=settings.config)
    result = MirrorResidualResult(point, params, config, (), mp.inf, None, None, 0, {}, message)
    return RefinementStageReport(settings, result, result, (), "branch_failure")


def _newton_refine_with_timeout(point: MirrorSearchPoint, settings: NewtonSettings) -> RefinementStageReport:
    """Run one Newton stage with a wall-clock timeout."""
    try:
        return _with_timeout(
            STAGE_TIMEOUT_SECONDS,
            f"{settings.name} refinement exceeded {STAGE_TIMEOUT_SECONDS} seconds",
            lambda: newton_refine(point, settings),
        )
    except TimeoutError as exc:
        return _stage_timeout_report(point, settings, str(exc))


def _promote_tracks(tracks: list[CandidateTrack], spec: BoxRegionSpec) -> set[int]:
    """Return seed ranks to promote for one V3 region."""
    viable = [track for track in tracks if track.seed_region == spec.name and track.stages[-1].final.failure is None]
    viable.sort(key=lambda track: track.stages[-1].final.residual_norm)
    return {track.seed_rank for track in viable[: spec.promote_quota]}


def _verification_norms(track: CandidateTrack) -> tuple[mp.mpf, ...]:
    """Return finite verification norms for one track."""
    return tuple(result.residual_norm for result in track.verifications if result.failure is None)


def _is_strong_lead(track: CandidateTrack) -> bool:
    """Return whether one verified track is a V3 strong lead."""
    if len(track.verifications) < 2 or any(result.failure for result in track.verifications[:2]):
        return False
    final = _track_final_result(track)
    norms = _verification_norms(track)[:2]
    return final.point.s > S_MIN and _point_distance(final.point) >= mp.mpf("0.05") and norms[-1] < mp.mpf("5e-5") and _stable_within_factor(norms, mp.mpf("10"))


def _classify_track_v3(track: CandidateTrack, berger_refs: tuple[MirrorResidualResult, ...]) -> str:
    """Classify one V3 track, adding a strong-lead label."""
    if _is_strong_lead(track):
        return "strong_lead"
    return _classify_track(track, berger_refs)


def _needs_order22(track: CandidateTrack) -> bool:
    """Return whether one promoted track deserves order-22 verification."""
    if len(track.verifications) < 2 or track.verifications[-1].failure:
        return False
    return _point_distance(_track_final_result(track).point) > mp.mpf("0.05") and track.verifications[-1].residual_norm < mp.mpf("5e-5")


def _physical_payload(point: MirrorSearchPoint) -> dict[str, str | None]:
    """Return JSON-ready physical parameters for one scaled point."""
    params, config = params_from_scaled(point)
    return {
        "a": _mp_string(params.left.a),
        "c": _mp_string(params.left.c),
        "alpha": _mp_string(params.left.alpha),
        "m": _mp_string(config.match_t),
        "interval_end": _mp_string(params.interval_end),
    }


def _track_payload(track: CandidateTrack) -> dict:
    """Return JSON-ready data for one V3 candidate track."""
    final = _track_final_result(track)
    payload = _common_track_payload(track, final)
    payload["final_physical"] = _physical_payload(final.point)
    return payload


def _run_order6(selection: SelectedCandidate, path: Path) -> CandidateTrack:
    """Run and persist the first refinement stage for one selected seed."""
    stage = _newton_refine_with_timeout(selection.candidate.seed.point, ORDER6_SETTINGS)
    track = CandidateTrack(selection.candidate.seed.index, selection.candidate.seed.region, selection.candidate.seed.point, selection.candidate.result, (stage,), (), "inconclusive")
    payload = {"seed_index": track.seed_rank, "region": track.seed_region, "selection_reason": selection.reason, "stage": _stage_payload(stage)}
    _write_jsonl_event(path, _event("refinement_stage", payload))
    return track


def _run_order10_and_verify(track: CandidateTrack, berger_refs: tuple[MirrorResidualResult, ...], path: Path) -> CandidateTrack:
    """Run promoted refinement, verification, classification, and JSON events."""
    stage = _newton_refine_with_timeout(track.stages[-1].final.point, ORDER10_SETTINGS)
    _write_jsonl_event(path, _event("refinement_stage", {"seed_index": track.seed_rank, "region": track.seed_region, "stage": _stage_payload(stage)}))
    if stage.final.failure:
        promoted = _replace_track(track, track.stages + (stage,), (), "inconclusive")
        classified = _replace_track(promoted, promoted.stages, promoted.verifications, _classify_track_v3(promoted, berger_refs))
        _write_jsonl_event(path, _event("candidate_classification", _track_payload(classified)))
        return classified
    verifications = _verify_point(stage.final.point)
    promoted = _replace_track(track, track.stages + (stage,), verifications, "inconclusive")
    if _needs_order22(promoted):
        verifications = verifications + _verify_point(stage.final.point, include_order22=True)[-1:]
        promoted = _replace_track(track, track.stages + (stage,), verifications, "inconclusive")
    classified = _replace_track(promoted, promoted.stages, promoted.verifications, _classify_track_v3(promoted, berger_refs))
    for result in classified.verifications:
        _write_jsonl_event(path, _event("verification", {"seed_index": track.seed_rank, "region": track.seed_region, "result": _result_payload(result)}))
    _write_jsonl_event(path, _event("candidate_classification", _track_payload(classified)))
    return classified


def _classify_unpromoted(track: CandidateTrack, berger_refs: tuple[MirrorResidualResult, ...], path: Path) -> CandidateTrack:
    """Classify and persist one unpromoted order-6 track."""
    classified = _replace_track(track, track.stages, (), _classify_track_v3(track, berger_refs))
    _write_jsonl_event(path, _event("candidate_classification", _track_payload(classified)))
    return classified


def _region_summary(candidates: list[SearchCandidate]) -> dict[str, dict[str, int]]:
    """Return scout success/failure counts by region."""
    return _common_region_summary(REGIONS, candidates)


def _print_region_best(candidates: list[SearchCandidate], limit: int = 5) -> None:
    """Print a small per-region best-residual table."""
    _common_print_region_best(REGIONS, candidates, limit)


def _print_track(track: CandidateTrack) -> None:
    """Print one final candidate-track summary."""
    final = _track_final_result(track)
    print(
        f"seed={track.seed_rank}, region={track.seed_region}, class={track.classification}, "
        f"norm={mp.nstr(final.residual_norm, 12)}, distance={mp.nstr(_point_distance(final.point), 8)}",
        flush=True,
    )
    if track.classification == "strong_lead":
        physical = _physical_payload(final.point)
        print(
            "  "
            f"scaled=(u={mp.nstr(final.point.u, 8)}, v={mp.nstr(final.point.v, 8)}, "
            f"r={mp.nstr(final.point.r, 8)}, s={mp.nstr(final.point.s, 8)})",
            flush=True,
        )
        print(
            "  "
            f"physical=(a={physical['a']}, c={physical['c']}, "
            f"alpha={physical['alpha']}, m={physical['m']})",
            flush=True,
        )


def _region_payload(spec: BoxRegionSpec) -> dict:
    """Return JSON metadata for one V3 region."""
    return {
        "name": spec.name,
        "lower": _mp_string(spec.lower),
        "upper": _mp_string(spec.upper),
        "ranges": [[_mp_string(lower), _mp_string(upper)] for lower, upper in spec.ranges],
        "samples": spec.samples,
        "best_quota": spec.best_quota,
        "diverse_quota": spec.diverse_quota,
        "promote_quota": spec.promote_quota,
    }


def _run_start_payload(jsonl_path: Path, summary_path: Path) -> dict:
    """Return JSON-ready run metadata."""
    return {
        "sweep_version": SWEEP_VERSION,
        "random_seed": RANDOM_SEED,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "min_match_t": _mp_string(MIN_MATCH_T),
        "s_min": _mp_string(S_MIN),
        "s_margin": _mp_string(S_MARGIN),
        "scout_order": SCOUT_CONFIG.series_order,
        "scout_timeout_seconds": SCOUT_TIMEOUT_SECONDS,
        "stage_timeout_seconds": STAGE_TIMEOUT_SECONDS,
        "regions": [_region_payload(spec) for spec in REGIONS],
    }


def main() -> None:
    """Run the V3 deep mirror sweep with JSON checkpointing."""
    jsonl_path, summary_path, resuming = _resume_or_new_paths()
    if resuming:
        if _has_post_scout_events(jsonl_path):
            raise NotImplementedError("V3 resume currently supports checkpoints stopped during scout evaluation.")
        print(f"resuming JSONL events from {jsonl_path}", flush=True)
    else:
        print(f"writing JSONL events to {jsonl_path}", flush=True)
        _write_jsonl_event(jsonl_path, _event("run_start", _run_start_payload(jsonl_path, summary_path)))
    seeds = _search_seeds()
    cached = _load_scout_candidates(jsonl_path) if resuming else {}
    if cached:
        print(f"loaded {len(cached)} cached scout seeds", flush=True)
    candidates = []
    for index, seed in enumerate(seeds, start=1):
        candidates.append(cached[seed.index] if seed.index in cached else _evaluate_seed(seed, jsonl_path))
        if index % 50 == 0 or index == len(seeds):
            print(f"processed {index}/{len(seeds)} scout seeds", flush=True)
    summary = _region_summary(candidates)
    print("\nscout-region counts:", flush=True)
    _print_region_summary(summary)
    _print_region_best(candidates)
    selections = [selection for spec in REGIONS for selection in _select_region_candidates(spec, candidates)]
    for selection in selections:
        payload = {"region": selection.candidate.seed.region, "reason": selection.reason, "candidate": _candidate_payload(selection.candidate)}
        _write_jsonl_event(jsonl_path, _event("region_selection", payload))
    berger_refs = _verify_point(BASE_POINT)
    tracks = [_run_order6(selection, jsonl_path) for selection in selections]
    promoted = set().union(*(_promote_tracks(tracks, spec) for spec in REGIONS))
    final_tracks = []
    for track in tracks:
        final = _run_order10_and_verify(track, berger_refs, jsonl_path) if track.seed_rank in promoted else _classify_unpromoted(track, berger_refs, jsonl_path)
        final_tracks.append(final)
        _print_track(final)
    run_summary = {
        "sweep_version": SWEEP_VERSION,
        "min_match_t": _mp_string(MIN_MATCH_T),
        "s_min": _mp_string(S_MIN),
        "region_summary": summary,
        "berger_references": [_result_payload(result) for result in berger_refs],
        "tracks": [_track_payload(track) for track in final_tracks],
    }
    _write_jsonl_event(jsonl_path, _event("run_summary", run_summary))
    _write_summary(summary_path, run_summary)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
