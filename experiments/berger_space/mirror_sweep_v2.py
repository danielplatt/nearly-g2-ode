"""Deep mirror sweep biased toward far and tail regions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from random import Random

from mpmath import mp

from problem import DEFAULT_CONFIG, SolverConfig
from solver.mirror_refinement import CandidateTrack, NewtonSettings, RefinementStageReport, newton_refine
from solver.mirror_shooting import BASE_POINT, MirrorResidualResult, MirrorSearchPoint, mirror_residual, params_from_scaled

from .mirror_search import _classify_track, _replace_track, _stable_within_factor, _track_final_result
from ..shared.mirror_sweep_common import (
    RegionSpec,
    SearchCandidate,
    SearchSeed,
    SelectedCandidate,
    _candidate_payload,
    _corner_seeds as _common_corner_seeds,
    _event,
    _evaluate_seed as _common_evaluate_seed,
    _mp_string,
    _output_paths as _common_output_paths,
    _point_distance,
    _point_from_values,
    _point_payload,
    _region_for_point as _common_region_for_point,
    _region_seeds,
    _region_summary as _common_region_summary,
    _result_payload,
    _select_region_candidates,
    _stage_payload,
    _successful,
    _track_payload as _common_track_payload,
    _with_timeout,
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
VERIFY22_CONFIG = SolverConfig(22, 130, 50, mp.mpf("0.5"), 2, DEFAULT_CONFIG.match_t)
VERIFY_CONFIGS = (VERIFY14_CONFIG, VERIFY18_CONFIG)

RANDOM_SEED = 1729
SWEEP_VERSION = "v2"
OUTPUT_DIR = Path("output/mirror_sweeps")
SCOUT_TIMEOUT_SECONDS = 120
STAGE_TIMEOUT_SECONDS = 8 * 60
MAX_NEWTON_COORDINATE = mp.mpf("12")
ORDER6_SETTINGS = NewtonSettings("order-6", ORDER6_CONFIG, mp.mpf("1e-3"), mp.mpf("1e-8"), 3, max_abs_coordinate=MAX_NEWTON_COORDINATE)
ORDER10_SETTINGS = NewtonSettings("order-10", ORDER10_CONFIG, mp.mpf("3e-4"), mp.mpf("1e-10"), 3, max_abs_coordinate=MAX_NEWTON_COORDINATE)

TAIL_FOCUS_BOX = {
    "u": (mp.mpf("-11"), mp.mpf("-6")),
    "v": (mp.mpf("-11"), mp.mpf("-6")),
    "r": (mp.mpf("-8"), mp.mpf("8")),
    "s": (mp.mpf("-9"), mp.mpf("-3")),
}

REGIONS = (
    RegionSpec("near", mp.mpf("0.25"), mp.one, (mp.one, mp.one, mp.mpf("3"), mp.one), 1000, 4, 4, 1),
    RegionSpec("middle", mp.one, mp.mpf("2.5"), (mp.mpf("2.5"), mp.mpf("2.5"), mp.mpf("6"), mp.mpf("2.5")), 1800, 8, 8, 3),
    RegionSpec("far", mp.mpf("2.5"), mp.mpf("5"), (mp.mpf("5"), mp.mpf("5"), mp.mpf("10"), mp.mpf("5")), 2200, 10, 10, 5),
    RegionSpec("very_far", mp.mpf("5"), mp.mpf("8"), (mp.mpf("8"), mp.mpf("8"), mp.mpf("14"), mp.mpf("8")), 1800, 10, 10, 5),
    RegionSpec("tail_focus", mp.zero, mp.inf, (mp.mpf("11"), mp.mpf("11"), mp.mpf("8"), mp.mpf("9")), 2200, 16, 16, 8),
)
ANNULAR_REGIONS = REGIONS[:4]
TAIL_REGION = REGIONS[-1]


def _region_for_point(point: MirrorSearchPoint) -> str:
    """Return the first V2 annular region containing one point."""
    return _common_region_for_point(point, ANNULAR_REGIONS)


def _corner_seeds(start_index: int = 0) -> list[SearchSeed]:
    """Return the fixed corner seeds from the short scout search."""
    return _common_corner_seeds(ANNULAR_REGIONS, start_index)


def _tail_point(rng: Random) -> MirrorSearchPoint:
    """Sample one point from the V2 tail-focus box."""
    u = rng.uniform(float(TAIL_FOCUS_BOX["u"][0]), float(TAIL_FOCUS_BOX["u"][1]))
    v = rng.uniform(float(TAIL_FOCUS_BOX["v"][0]), float(TAIL_FOCUS_BOX["v"][1]))
    r = rng.uniform(float(TAIL_FOCUS_BOX["r"][0]), float(TAIL_FOCUS_BOX["r"][1]))
    s = rng.uniform(float(TAIL_FOCUS_BOX["s"][0]), float(TAIL_FOCUS_BOX["s"][1]))
    return _point_from_values((u, v, r, s))


def _tail_focus_seeds(rng: Random, start_index: int) -> list[SearchSeed]:
    """Return reproducible tail-focus seeds."""
    return [SearchSeed(start_index + index, TAIL_REGION.name, "tail_focus", _tail_point(rng)) for index in range(TAIL_REGION.samples)]


def _search_seeds(seed: int = RANDOM_SEED) -> list[SearchSeed]:
    """Return all deterministic V2 long-sweep seeds."""
    rng = Random(seed)
    seeds = _corner_seeds()
    for spec in ANNULAR_REGIONS:
        seeds.extend(_region_seeds(spec, rng, len(seeds)))
    seeds.extend(_tail_focus_seeds(rng, len(seeds)))
    return seeds


def _output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped V2 JSONL and summary output paths."""
    return _common_output_paths(OUTPUT_DIR, RANDOM_SEED, f"-{SWEEP_VERSION}", now)


def _evaluate_seed(seed: SearchSeed, path: Path) -> SearchCandidate:
    """Evaluate and persist one scout seed."""
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


def _promote_tracks(tracks: list[CandidateTrack], spec: RegionSpec) -> set[int]:
    """Return seed ranks to promote for one region."""
    viable = [track for track in tracks if track.seed_region == spec.name and track.stages[-1].final.failure is None]
    viable.sort(key=lambda track: track.stages[-1].final.residual_norm)
    return {track.seed_rank for track in viable[: spec.promote_quota]}


def _verification_norms(track: CandidateTrack) -> tuple[mp.mpf, ...]:
    """Return finite verification norms for one track."""
    return tuple(result.residual_norm for result in track.verifications if result.failure is None)


def _is_strong_lead(track: CandidateTrack) -> bool:
    """Return whether one verified track is a V2 strong lead."""
    if len(track.verifications) < 2 or any(result.failure for result in track.verifications[:2]):
        return False
    final = _track_final_result(track)
    norms = _verification_norms(track)[:2]
    return _point_distance(final.point) >= mp.mpf("0.05") and norms[-1] < mp.mpf("5e-5") and _stable_within_factor(norms, mp.mpf("10"))


def _classify_track_v2(track: CandidateTrack, berger_refs: tuple[MirrorResidualResult, ...]) -> str:
    """Classify one V2 track, adding a strong-lead label."""
    if _is_strong_lead(track):
        return "strong_lead"
    return _classify_track(track, berger_refs)


def _needs_order22(track: CandidateTrack) -> bool:
    """Return whether one promoted track deserves order-22 verification."""
    if len(track.verifications) < 2 or track.verifications[-1].failure:
        return False
    return _point_distance(_track_final_result(track).point) > mp.mpf("0.05") and track.verifications[-1].residual_norm < mp.mpf("5e-5")


def _track_payload(track: CandidateTrack) -> dict:
    """Return JSON-ready data for one candidate track."""
    return _common_track_payload(track, _track_final_result(track))


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
        classified = _replace_track(promoted, promoted.stages, promoted.verifications, _classify_track_v2(promoted, berger_refs))
        _write_jsonl_event(path, _event("candidate_classification", _track_payload(classified)))
        return classified
    verifications = _verify_point(stage.final.point)
    promoted = _replace_track(track, track.stages + (stage,), verifications, "inconclusive")
    if _needs_order22(promoted):
        verifications = verifications + _verify_point(stage.final.point, include_order22=True)[-1:]
        promoted = _replace_track(track, track.stages + (stage,), verifications, "inconclusive")
    classified = _replace_track(promoted, promoted.stages, promoted.verifications, _classify_track_v2(promoted, berger_refs))
    for result in classified.verifications:
        _write_jsonl_event(path, _event("verification", {"seed_index": track.seed_rank, "region": track.seed_region, "result": _result_payload(result)}))
    _write_jsonl_event(path, _event("candidate_classification", _track_payload(classified)))
    return classified


def _classify_unpromoted(track: CandidateTrack, berger_refs: tuple[MirrorResidualResult, ...], path: Path) -> CandidateTrack:
    """Classify and persist one unpromoted order-6 track."""
    classified = _replace_track(track, track.stages, (), _classify_track_v2(track, berger_refs))
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


def _region_payload(spec: RegionSpec) -> dict:
    """Return JSON metadata for one V2 region."""
    payload = {
        "name": spec.name,
        "lower": _mp_string(spec.lower),
        "upper": _mp_string(spec.upper),
        "bounds": [_mp_string(value) for value in spec.bounds],
        "samples": spec.samples,
        "best_quota": spec.best_quota,
        "diverse_quota": spec.diverse_quota,
        "promote_quota": spec.promote_quota,
    }
    if spec.name == "tail_focus":
        payload["box"] = {key: [_mp_string(value[0]), _mp_string(value[1])] for key, value in TAIL_FOCUS_BOX.items()}
    return payload


def _run_start_payload(jsonl_path: Path, summary_path: Path) -> dict:
    """Return JSON-ready run metadata."""
    return {
        "sweep_version": SWEEP_VERSION,
        "random_seed": RANDOM_SEED,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "scout_order": SCOUT_CONFIG.series_order,
        "scout_timeout_seconds": SCOUT_TIMEOUT_SECONDS,
        "stage_timeout_seconds": STAGE_TIMEOUT_SECONDS,
        "regions": [_region_payload(spec) for spec in REGIONS],
    }


def main() -> None:
    """Run the V2 deep mirror sweep with JSON checkpointing."""
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
    run_summary = {
        "sweep_version": SWEEP_VERSION,
        "region_summary": summary,
        "berger_references": [_result_payload(result) for result in berger_refs],
        "tracks": [_track_payload(track) for track in final_tracks],
    }
    _write_jsonl_event(jsonl_path, _event("run_summary", run_summary))
    _write_summary(summary_path, run_summary)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
