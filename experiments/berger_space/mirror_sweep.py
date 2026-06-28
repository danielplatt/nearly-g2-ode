"""Long seeded sweep for non-Berger mirror-complete candidates."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from random import Random

from mpmath import mp

from problem import DEFAULT_CONFIG, SolverConfig
from solver.mirror_refinement import CandidateTrack, NewtonSettings, newton_refine
from solver.mirror_shooting import BASE_POINT, MirrorResidualResult, MirrorSearchPoint, mirror_residual

from .mirror_search import _classify_track, _replace_track, _track_final_result
from ..shared.mirror_sweep_common import (
    RegionSpec,
    ScoutTimeoutError,
    SearchCandidate,
    SearchSeed,
    SelectedCandidate,
    _candidate_payload,
    _corner_seeds as _common_corner_seeds,
    _event,
    _evaluate_seed as _common_evaluate_seed,
    _evaluate_seed_with_timeout as _common_evaluate_seed_with_timeout,
    _mp_string,
    _output_paths as _common_output_paths,
    _point_distance,
    _point_from_values,
    _point_key,
    _point_payload,
    _region_for_point as _common_region_for_point,
    _region_seeds,
    _region_summary as _common_region_summary,
    _result_payload,
    _search_seeds as _common_search_seeds,
    _select_region_candidates,
    _sort_key,
    _stage_payload,
    _successful,
    _timeout_result as _common_timeout_result,
    _track_payload as _common_track_payload,
    _write_jsonl_event,
    _write_summary,
    _print_region_best as _common_print_region_best,
    _print_region_summary,
)


SCOUT_CONFIG = SolverConfig(4, 30, 15, mp.mpf("0.95"), 0, DEFAULT_CONFIG.match_t)
ORDER6_CONFIG = SolverConfig(6, 40, 20, mp.mpf("0.9"), 0, DEFAULT_CONFIG.match_t)
ORDER10_CONFIG = SolverConfig(10, 70, 30, mp.mpf("0.7"), 1, DEFAULT_CONFIG.match_t)
VERIFY14_CONFIG = SolverConfig(14, 90, 35, mp.mpf("0.6"), 2, DEFAULT_CONFIG.match_t)
VERIFY18_CONFIG = SolverConfig(18, 110, 45, mp.mpf("0.55"), 2, DEFAULT_CONFIG.match_t)
VERIFY_CONFIGS = (VERIFY14_CONFIG, VERIFY18_CONFIG)

RANDOM_SEED = 1729
OUTPUT_DIR = Path("output/mirror_sweeps")
SCOUT_TIMEOUT_SECONDS = 120
MAX_NEWTON_COORDINATE = mp.mpf("8")
ORDER6_SETTINGS = NewtonSettings("order-6", ORDER6_CONFIG, mp.mpf("1e-3"), mp.mpf("1e-8"), 3, max_abs_coordinate=MAX_NEWTON_COORDINATE)
ORDER10_SETTINGS = NewtonSettings("order-10", ORDER10_CONFIG, mp.mpf("3e-4"), mp.mpf("1e-10"), 3, max_abs_coordinate=MAX_NEWTON_COORDINATE)

REGIONS = (
    RegionSpec("near", mp.mpf("0.25"), mp.one, (mp.one, mp.one, mp.mpf("3"), mp.one), 1200, 6, 6, 2),
    RegionSpec("middle", mp.one, mp.mpf("2"), (mp.mpf("2"), mp.mpf("2"), mp.mpf("5"), mp.mpf("2")), 1600, 8, 8, 3),
    RegionSpec("far", mp.mpf("2"), mp.mpf("4"), (mp.mpf("4"), mp.mpf("4"), mp.mpf("8"), mp.mpf("4")), 1600, 8, 8, 4),
    RegionSpec("very_far", mp.mpf("4"), mp.mpf("6"), (mp.mpf("6"), mp.mpf("6"), mp.mpf("12"), mp.mpf("6")), 1000, 6, 6, 3),
)


def _region_for_point(point: MirrorSearchPoint) -> str:
    """Return the first V1 annular region containing one point."""
    return _common_region_for_point(point, REGIONS)


def _corner_seeds(start_index: int = 0) -> list[SearchSeed]:
    """Return the fixed corner seeds from the short scout search."""
    return _common_corner_seeds(REGIONS, start_index)


def _search_seeds(seed: int = RANDOM_SEED) -> list[SearchSeed]:
    """Return all deterministic V1 long-sweep seeds."""
    return _common_search_seeds(REGIONS, seed)


def _output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped JSONL and summary output paths."""
    return _common_output_paths(OUTPUT_DIR, RANDOM_SEED, "", now)


def _timeout_result(seed: SearchSeed, message: str) -> MirrorResidualResult:
    """Return a synthetic residual result for a timed-out scout seed."""
    return _common_timeout_result(seed, SCOUT_CONFIG, message)


def _evaluate_seed_with_timeout(seed: SearchSeed) -> SearchCandidate:
    """Evaluate one seed with the V1 scout timeout."""
    return _common_evaluate_seed_with_timeout(seed, SCOUT_CONFIG, SCOUT_TIMEOUT_SECONDS)


def _evaluate_seed(seed: SearchSeed, path: Path) -> SearchCandidate:
    """Evaluate and persist one scout seed."""
    return _common_evaluate_seed(seed, path, SCOUT_CONFIG, SCOUT_TIMEOUT_SECONDS)


def _verify_point(point: MirrorSearchPoint) -> tuple[MirrorResidualResult, ...]:
    """Evaluate one point at the high-order verification configs."""
    results = []
    for config in VERIFY_CONFIGS:
        with mp.workdps(config.working_dps):
            results.append(mirror_residual(point, config))
    return tuple(results)


def _promote_tracks(tracks: list[CandidateTrack], spec: RegionSpec) -> set[int]:
    """Return seed ranks to promote for one region."""
    viable = [track for track in tracks if track.seed_region == spec.name and track.stages[-1].final.failure is None]
    viable.sort(key=lambda track: track.stages[-1].final.residual_norm)
    return {track.seed_rank for track in viable[: spec.promote_quota]}


def _track_payload(track: CandidateTrack) -> dict:
    """Return JSON-ready data for one candidate track."""
    return _common_track_payload(track, _track_final_result(track))


def _run_order6(selection: SelectedCandidate, path: Path) -> CandidateTrack:
    """Run and persist the first refinement stage for one selected seed."""
    stage = newton_refine(selection.candidate.seed.point, ORDER6_SETTINGS)
    track = CandidateTrack(selection.candidate.seed.index, selection.candidate.seed.region, selection.candidate.seed.point, selection.candidate.result, (stage,), (), "inconclusive")
    payload = {"seed_index": track.seed_rank, "region": track.seed_region, "selection_reason": selection.reason, "stage": _stage_payload(stage)}
    _write_jsonl_event(path, _event("refinement_stage", payload))
    return track


def _run_order10_and_verify(track: CandidateTrack, berger_refs: tuple[MirrorResidualResult, ...], path: Path) -> CandidateTrack:
    """Run promoted refinement, verification, classification, and JSON events."""
    stage = newton_refine(track.stages[-1].final.point, ORDER10_SETTINGS)
    verifications = _verify_point(stage.final.point)
    promoted = _replace_track(track, track.stages + (stage,), verifications, "inconclusive")
    classified = _replace_track(promoted, promoted.stages, promoted.verifications, _classify_track(promoted, berger_refs))
    _write_jsonl_event(path, _event("refinement_stage", {"seed_index": track.seed_rank, "region": track.seed_region, "stage": _stage_payload(stage)}))
    for result in verifications:
        _write_jsonl_event(path, _event("verification", {"seed_index": track.seed_rank, "region": track.seed_region, "result": _result_payload(result)}))
    _write_jsonl_event(path, _event("candidate_classification", _track_payload(classified)))
    return classified


def _classify_unpromoted(track: CandidateTrack, berger_refs: tuple[MirrorResidualResult, ...], path: Path) -> CandidateTrack:
    """Classify and persist one unpromoted order-6 track."""
    classified = _replace_track(track, track.stages, (), _classify_track(track, berger_refs))
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


def _run_start_payload(jsonl_path: Path, summary_path: Path) -> dict:
    """Return JSON-ready run metadata."""
    return {
        "random_seed": RANDOM_SEED,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "scout_order": SCOUT_CONFIG.series_order,
        "scout_timeout_seconds": SCOUT_TIMEOUT_SECONDS,
        "regions": [spec.__dict__ | {"bounds": [_mp_string(value) for value in spec.bounds], "lower": _mp_string(spec.lower), "upper": _mp_string(spec.upper)} for spec in REGIONS],
    }


def main() -> None:
    """Run the long mirror sweep with JSON checkpointing."""
    jsonl_path, summary_path = _output_paths()
    print(f"writing JSONL events to {jsonl_path}", flush=True)
    _write_jsonl_event(jsonl_path, _event("run_start", _run_start_payload(jsonl_path, summary_path)))
    seeds = _search_seeds()
    candidates = []
    for index, seed in enumerate(seeds, start=1):
        candidates.append(_evaluate_seed(seed, jsonl_path))
        if index % 50 == 0 or index == len(seeds):
            print(f"evaluated {index}/{len(seeds)} scout seeds", flush=True)
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
    run_summary = {"region_summary": summary, "berger_references": [_result_payload(result) for result in berger_refs], "tracks": [_track_payload(track) for track in final_tracks]}
    _write_jsonl_event(jsonl_path, _event("run_summary", run_summary))
    _write_summary(summary_path, run_summary)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
