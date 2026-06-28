"""Guarded mirrored covering search for non-Berger candidates."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

from mpmath import mp

from problem import DEFAULT_CONFIG
from solver.mirror_refinement import CandidateTrack, NewtonSettings, RefinementStageReport, newton_refine
from solver.mirror_shooting import MirrorResidualResult, MirrorSearchPoint

from ..shared.mirror_sweep_common import (
    SearchCandidate,
    SearchSeed,
    SelectedCandidate,
    _candidate_payload,
    _event,
    _mp_string,
    _output_paths as _common_output_paths,
    _point_distance,
    _result_payload,
    _stage_payload,
    _track_payload as _common_track_payload,
    _write_jsonl_event,
    _write_summary,
)
from . import mirror_covering_calibration as covering
from . import mirror_recovery_calibration as recovery


RANDOM_SEED = 1729
SEARCH_VERSION = "guarded_covering_v1"
OUTPUT_DIR = Path("output/mirror_guarded_covering_search")
OUTPUT_SUFFIX = "-guarded-covering-v1"

MAX_REFINEMENT_COORDINATE = mp.mpf("4")
with mp.workdps(80):
    MIN_MATCH_T = mp.mpf("0.01")
    S_MIN = mp.log(MIN_MATCH_T / DEFAULT_CONFIG.match_t)

SCOUT_CONFIG = recovery.SCOUT_CONFIG
ORDER10_CONFIG = recovery.ORDER10_CONFIG
VERIFY14_CONFIG = recovery.VERIFY14_CONFIG
VERIFY18_CONFIG = recovery.VERIFY18_CONFIG
VERIFY_CONFIGS = (VERIFY14_CONFIG, VERIFY18_CONFIG)

CONTRACTION_SETTINGS = NewtonSettings("order-6-one-step-guarded", recovery.ORDER6_CONFIG, mp.mpf("1e-3"), mp.mpf("-1"), 1, max_abs_coordinate=MAX_REFINEMENT_COORDINATE, min_s_coordinate=S_MIN)
ORDER10_SETTINGS = NewtonSettings("order-10-guarded", ORDER10_CONFIG, mp.mpf("3e-4"), mp.mpf("1e-10"), 3, max_abs_coordinate=MAX_REFINEMENT_COORDINATE, min_s_coordinate=S_MIN)
ORDER14_SETTINGS = NewtonSettings("order-14-guarded", VERIFY14_CONFIG, mp.mpf("1e-4"), mp.mpf("1e-12"), 2, max_abs_coordinate=MAX_REFINEMENT_COORDINATE, min_s_coordinate=S_MIN)

CORE_SEEDS = 10_000
HALTON_SAMPLES = 40_000
HALTON_SKIP = 1729
HALTON_PRIMES = (2, 3, 5, 7)
HALTON_BOUNDS = ((mp.mpf("-2"), mp.mpf("2")), (mp.mpf("-2"), mp.mpf("2")), (mp.mpf("-4"), mp.mpf("4")), (mp.mpf("-2"), mp.mpf("2")))
SCOUT_KEEP = 1500
SCOUT_PROGRESS_INTERVAL = 500
SELECT_BEST_FINAL = 100
SELECT_BEST_RATIO = 60
SELECT_DIVERSE = 40
LOW_ORDER_LEAD_THRESHOLD = mp.mpf("1e-8")


def _radical_inverse(index: int, base: int) -> mp.mpf:
    """Return the radical inverse of one positive integer."""
    value = mp.zero
    factor = mp.one / base
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor /= base
    return value


def _scale(value: mp.mpf, bounds: tuple[mp.mpf, mp.mpf]) -> mp.mpf:
    """Scale one unit interval value to the given bounds."""
    lower, upper = bounds
    return lower + (upper - lower) * value


def _halton_point(index: int) -> MirrorSearchPoint:
    """Return one deterministic 4D Halton point in the search box."""
    values = [_scale(_radical_inverse(index, base), bounds) for base, bounds in zip(HALTON_PRIMES, HALTON_BOUNDS)]
    return MirrorSearchPoint(*values)


def _core_grid_seeds() -> list[SearchSeed]:
    """Return the calibrated core covering grid."""
    return [SearchSeed(seed.index, "core_cover", seed.source, seed.point) for seed in covering._grid_seeds()]


def _halton_seeds(start_index: int = CORE_SEEDS) -> list[SearchSeed]:
    """Return deterministic wide Halton scout seeds."""
    seeds = []
    for offset in range(HALTON_SAMPLES):
        point = _halton_point(HALTON_SKIP + offset + 1)
        seeds.append(SearchSeed(start_index + offset, "wide_halton", "halton", point))
    return seeds


def _search_seeds() -> list[SearchSeed]:
    """Return all scout seeds for the guarded run."""
    return _core_grid_seeds() + _halton_seeds()


def _evaluate_seed(seed: SearchSeed) -> SearchCandidate:
    """Evaluate one scout seed, rejecting points below the midpoint floor."""
    if seed.point.s <= S_MIN:
        params, config = recovery.params_from_scaled(seed.point, template_config=SCOUT_CONFIG)
        result = MirrorResidualResult(seed.point, params, config, (), mp.inf, None, None, 0, {}, "m_floor_rejected")
        return SearchCandidate(seed, result)
    return covering._evaluate_seed(seed)


def _run_scouts(path: Path, seeds: list[SearchSeed], cached: dict[int, SearchCandidate]) -> list[SearchCandidate]:
    """Evaluate or reuse all guarded scout seeds."""
    candidates = []
    reused = 0
    for index, seed in enumerate(seeds, start=1):
        if seed.index in cached:
            candidate = cached[seed.index]
            reused += 1
        else:
            candidate = _evaluate_seed(seed)
            _write_jsonl_event(path, _event("scout_result", _candidate_payload(candidate)))
        candidates.append(candidate)
        if index % SCOUT_PROGRESS_INTERVAL == 0:
            print(f"scouts processed: {index}/{len(seeds)}", flush=True)
    if reused:
        print(f"scouts reused: {reused}/{len(seeds)}", flush=True)
    return candidates


def _successful(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    """Return successful scouts sorted by residual norm."""
    return [candidate for candidate in sorted(candidates, key=lambda item: item.result.residual_norm) if candidate.result.failure is None]


def _top_scouts(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    """Return scouts selected for guarded contraction probes."""
    return _successful(candidates)[:SCOUT_KEEP]


def _run_contraction_probe(candidate: SearchCandidate) -> covering.ContractionProbe:
    """Run one guarded order-6 contraction probe."""
    stage = newton_refine(candidate.seed.point, CONTRACTION_SETTINGS)
    return covering.ContractionProbe(candidate, stage)


def _probe_payload(probe: covering.ContractionProbe) -> dict:
    """Return JSON-ready contraction probe data."""
    return {
        "candidate": _candidate_payload(probe.candidate),
        "stage": _stage_payload(probe.stage),
        "contraction_ratio": _mp_string(covering._contraction_ratio(probe)),
    }


def _probe_from_payload(payload: dict) -> covering.ContractionProbe:
    """Deserialize one guarded contraction probe."""
    return covering.ContractionProbe(recovery._candidate_from_payload(payload["candidate"]), recovery._stage_from_payload(payload["stage"]))


def _run_contraction_probes(path: Path, candidates: list[SearchCandidate], cached: dict[int, covering.ContractionProbe]) -> list[covering.ContractionProbe]:
    """Evaluate or reuse contraction probes."""
    probes = []
    reused = 0
    for candidate in candidates:
        if candidate.seed.index in cached:
            probe = cached[candidate.seed.index]
            reused += 1
        else:
            probe = _run_contraction_probe(candidate)
            _write_jsonl_event(path, _event("contraction_probe", _probe_payload(probe)))
        probes.append(probe)
    if reused:
        print(f"contraction probes reused: {reused}/{len(candidates)}", flush=True)
    return probes


def _select_candidates(probes: list[covering.ContractionProbe]) -> list[SelectedCandidate]:
    """Select guarded refinement candidates from contraction data."""
    return covering._select_blind_candidates(
        probes,
        best_final_quota=SELECT_BEST_FINAL,
        best_ratio_quota=SELECT_BEST_RATIO,
        diverse_quota=SELECT_DIVERSE,
    )


def _verify_point(point: MirrorSearchPoint) -> tuple[MirrorResidualResult, ...]:
    """Verify one guarded refinement output at higher orders."""
    results = []
    for config in VERIFY_CONFIGS:
        with mp.workdps(config.working_dps):
            results.append(recovery.mirror_residual(point, config))
    return tuple(results)


def _needs_order14(stage: RefinementStageReport) -> bool:
    """Return whether a low-order lead deserves guarded order-14 correction."""
    return stage.final.failure is None and stage.final.residual_norm < LOW_ORDER_LEAD_THRESHOLD


def _run_guarded_track(candidate: SearchCandidate, berger_refs: tuple[MirrorResidualResult, ...], path: Path) -> CandidateTrack:
    """Run guarded order-10/order-14 refinement and verification."""
    order10 = newton_refine(candidate.seed.point, ORDER10_SETTINGS)
    stages = [order10]
    _write_jsonl_event(path, _event("refinement_stage", {"seed_index": candidate.seed.index, "stage": _stage_payload(order10)}))
    if _needs_order14(order10):
        order14 = newton_refine(order10.final.point, ORDER14_SETTINGS)
        stages.append(order14)
        _write_jsonl_event(path, _event("refinement_stage", {"seed_index": candidate.seed.index, "stage": _stage_payload(order14)}))
    final = stages[-1].final
    verifications = _verify_point(final.point)
    for result in verifications:
        _write_jsonl_event(path, _event("verification", {"seed_index": candidate.seed.index, "result": _result_payload(result)}))
    track = CandidateTrack(candidate.seed.index, candidate.seed.region, candidate.seed.point, candidate.result, tuple(stages), verifications, "inconclusive")
    classified = CandidateTrack(track.seed_rank, track.seed_region, track.seed_point, track.scout_result, track.stages, track.verifications, recovery._classify_track(track, berger_refs))
    _write_jsonl_event(path, _event("candidate_classification", _common_track_payload(classified, classified.stages[-1].final)))
    return classified


def _run_tracks(path: Path, selections: list[SelectedCandidate], berger_refs: tuple[MirrorResidualResult, ...], cached: dict[int, CandidateTrack]) -> list[CandidateTrack]:
    """Run or reuse guarded refinement tracks."""
    tracks = []
    for selection in selections:
        seed_index = selection.candidate.seed.index
        if seed_index in cached:
            track = cached[seed_index]
            print(f"guarded seed={seed_index}: reused {track.classification}", flush=True)
        else:
            track = _run_guarded_track(selection.candidate, berger_refs, path)
            cached[seed_index] = track
            print(f"guarded seed={seed_index}: {track.classification}", flush=True)
        tracks.append(track)
    return tracks


def _output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped guarded search output paths."""
    return _common_output_paths(OUTPUT_DIR, RANDOM_SEED, OUTPUT_SUFFIX, now)


def _summary_path_for_jsonl(path: Path) -> Path:
    """Return the summary path paired with a JSONL checkpoint."""
    return path.with_name(f"{path.stem}-summary.json")


def _run_start_payload(jsonl_path: Path, summary_path: Path) -> dict:
    """Return checkpoint metadata for the guarded run."""
    return {
        "random_seed": RANDOM_SEED,
        "search_version": SEARCH_VERSION,
        "min_match_t": _mp_string(MIN_MATCH_T),
        "s_min": _mp_string(S_MIN),
        "max_refinement_coordinate": _mp_string(MAX_REFINEMENT_COORDINATE),
        "core_seeds": CORE_SEEDS,
        "halton_samples": HALTON_SAMPLES,
        "halton_bounds": [[_mp_string(left), _mp_string(right)] for left, right in HALTON_BOUNDS],
        "scout_keep": SCOUT_KEEP,
        "select_best_final": SELECT_BEST_FINAL,
        "select_best_ratio": SELECT_BEST_RATIO,
        "select_diverse": SELECT_DIVERSE,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
    }


def _checkpoint_is_compatible(path: Path) -> bool:
    """Return whether one checkpoint can be resumed by this runner."""
    if recovery._jsonl_has_event(path, "run_summary"):
        return False
    starts = [event for event in recovery._jsonl_events(path) if event.get("event") == "run_start"]
    if not starts:
        return False
    expected = _run_start_payload(path, _summary_path_for_jsonl(path))
    return all(starts[-1].get(key) == value for key, value in expected.items() if key not in {"jsonl_path", "summary_path"})


def _latest_incomplete_checkpoint() -> Path | None:
    """Return the newest compatible unfinished checkpoint."""
    candidates = sorted(OUTPUT_DIR.glob(f"*-seed{RANDOM_SEED}{OUTPUT_SUFFIX}.jsonl"), reverse=True)
    return next((path for path in candidates if _checkpoint_is_compatible(path)), None)


def _resume_or_new_paths(now: datetime | None = None) -> tuple[Path, Path, bool]:
    """Return output paths, resuming an incomplete run when possible."""
    if now is None:
        checkpoint = _latest_incomplete_checkpoint()
        if checkpoint is not None:
            return checkpoint, _summary_path_for_jsonl(checkpoint), True
    jsonl_path, summary_path = _output_paths(now)
    return jsonl_path, summary_path, False


def _load_contraction_probes(path: Path) -> dict[int, covering.ContractionProbe]:
    """Load completed contraction probes from a checkpoint."""
    probes = {}
    for event in recovery._jsonl_events(path):
        if event.get("event") == "contraction_probe":
            probe = _probe_from_payload(event)
            probes[probe.candidate.seed.index] = probe
    return probes


def _classification_counts(tracks: list[CandidateTrack]) -> dict[str, int]:
    """Return classification counts for guarded tracks."""
    return dict(Counter(track.classification for track in tracks))


def _track_final(track: CandidateTrack) -> MirrorResidualResult:
    """Return the final residual result for one track."""
    return track.stages[-1].final if track.stages else track.scout_result


def _physical_payload(point: MirrorSearchPoint) -> dict:
    """Return physical parameters for one scaled point."""
    params, config = recovery.params_from_scaled(point, template_config=DEFAULT_CONFIG)
    return {
        "a": _mp_string(params.left.a),
        "c": _mp_string(params.left.c),
        "alpha": _mp_string(params.left.alpha),
        "m": _mp_string(config.match_t),
    }


def _lead_payload(track: CandidateTrack) -> dict:
    """Return compact data for one non-Berger lead."""
    final = _track_final(track)
    return {
        "seed_index": track.seed_rank,
        "classification": track.classification,
        "final": _result_payload(final),
        "verifications": [_result_payload(result) for result in track.verifications],
        "physical": _physical_payload(final.point),
    }


def _summary_payload(references, scouts, probes, selections, tracks) -> dict:
    """Return JSON-ready final summary."""
    possible = [track for track in tracks if track.classification == "possible_non_berger_root"]
    stable = [track for track in tracks if track.classification != "recovered_berger" and track.verifications and max(result.residual_norm for result in track.verifications) < mp.mpf("1e-4")]
    return {
        "berger_references": [_result_payload(result) for result in references],
        "scout_count": len(scouts),
        "scout_successes": sum(candidate.result.failure is None for candidate in scouts),
        "probe_count": len(probes),
        "selection_count": len(selections),
        "classification_counts": _classification_counts(tracks),
        "selected": [{"rank": item.rank, "reason": item.reason, "candidate": _candidate_payload(item.candidate)} for item in selections],
        "tracks": [_common_track_payload(track, _track_final(track)) for track in tracks],
        "possible_non_berger": [_lead_payload(track) for track in possible],
        "stable_non_berger_leads": [_lead_payload(track) for track in stable],
    }


def _print_summary(payload: dict) -> None:
    """Print a compact guarded search summary."""
    print(f"scouts: {payload['scout_successes']}/{payload['scout_count']} branch-valid", flush=True)
    print(f"probes: {payload['probe_count']}", flush=True)
    print(f"selected: {payload['selection_count']}", flush=True)
    print(f"classifications: {payload['classification_counts']}", flush=True)
    print(f"possible non-Berger roots: {len(payload['possible_non_berger'])}", flush=True)
    print(f"stable non-Berger leads (<1e-4): {len(payload['stable_non_berger_leads'])}", flush=True)


def main() -> None:
    """Run the guarded mirrored covering search."""
    jsonl_path, summary_path, resumed = _resume_or_new_paths()
    if resumed:
        print(f"resuming JSONL checkpoint {jsonl_path}", flush=True)
        scout_cache = recovery._load_scout_candidates(jsonl_path)
        probe_cache = _load_contraction_probes(jsonl_path)
        track_cache = recovery._load_classified_tracks(jsonl_path)
    else:
        print(f"writing JSONL events to {jsonl_path}", flush=True)
        _write_jsonl_event(jsonl_path, _event("run_start", _run_start_payload(jsonl_path, summary_path)))
        scout_cache = {}
        probe_cache = {}
        track_cache = {}
    references = recovery._reference_residuals()
    recovery._print_references(references)
    berger_refs = references[-2:]
    scouts = _run_scouts(jsonl_path, _search_seeds(), scout_cache)
    probes = _run_contraction_probes(jsonl_path, _top_scouts(scouts), probe_cache)
    selections = _select_candidates(probes)
    _write_jsonl_event(jsonl_path, _event("selection", covering._selection_payload("guarded_mirrored_search", selections)))
    tracks = _run_tracks(jsonl_path, selections, berger_refs, track_cache)
    payload = _summary_payload(references, scouts, probes, selections, tracks)
    _print_summary(payload)
    _write_jsonl_event(jsonl_path, _event("run_summary", payload))
    _write_summary(summary_path, payload)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
