"""Deterministic covering calibration for rediscovering Berger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

from mpmath import mp

from solver.mirror_refinement import CandidateTrack, NewtonSettings, RefinementStageReport, newton_refine
from solver.mirror_shooting import BASE_POINT, MirrorResidualResult, MirrorSearchPoint, mirror_residual

from ..shared.mirror_sweep_common import (
    SearchCandidate,
    SearchSeed,
    SelectedCandidate,
    _candidate_payload,
    _event,
    _mp_string,
    _output_paths as _common_output_paths,
    _point_distance,
    _point_payload,
    _result_payload,
    _stage_payload,
    _write_jsonl_event,
    _write_summary,
)
from . import mirror_recovery_calibration as recovery


RANDOM_SEED = recovery.RANDOM_SEED
CALIBRATION_VERSION = "covering_v1"
OUTPUT_DIR = Path("output/mirror_covering_calibration")
OUTPUT_SUFFIX = "-covering-v1"

GRID_COORDINATES = tuple(mp.mpf(value) for value in ("-0.9", "-0.7", "-0.5", "-0.3", "-0.1", "0.1", "0.3", "0.5", "0.7", "0.9"))
GRID_BOUND = mp.one
GRID_RADIUS = mp.mpf("0.1")
GRID_SIZE = 10_000

SCOUT_CONFIG = recovery.SCOUT_CONFIG
SCOUT_KEEP = 500
SCOUT_PROGRESS_INTERVAL = 250

ORACLE_CLOSEST_COUNT = 16
CONTRACTION_SETTINGS = NewtonSettings("order-6-one-step", recovery.ORDER6_CONFIG, mp.mpf("1e-3"), mp.mpf("-1"), 1, max_abs_coordinate=recovery.MAX_NEWTON_COORDINATE)
SELECT_BEST_FINAL = 40
SELECT_BEST_RATIO = 20
SELECT_DIVERSE = 20


@dataclass(frozen=True)
class ContractionProbe:
    """One order-6 contraction probe for a scout candidate."""

    candidate: SearchCandidate
    stage: RefinementStageReport


@dataclass(frozen=True)
class CoveringSummary:
    """Compact summary for one covering calibration run."""

    scout_successes: int
    oracle_recovered: bool
    blind_recovered: bool
    selected_count: int


def _grid_seeds(coords: tuple[mp.mpf, ...] = GRID_COORDINATES) -> list[SearchSeed]:
    """Return all cell-centered grid seeds in the mirrored 4D box."""
    seeds = []
    for index, values in enumerate(product(coords, repeat=4)):
        seeds.append(SearchSeed(index, "cover_grid", "cell_center", MirrorSearchPoint(*values)))
    return seeds


def _grid_covers_box(coords: tuple[mp.mpf, ...] = GRID_COORDINATES) -> bool:
    """Return whether coordinate cells cover [-1,1] with radius 0.1."""
    ordered = sorted(coords)
    tolerance = mp.mpf("1e-12")
    if ordered[0] - (-GRID_BOUND) > GRID_RADIUS + tolerance:
        return False
    if GRID_BOUND - ordered[-1] > GRID_RADIUS + tolerance:
        return False
    return all(right - left <= 2 * GRID_RADIUS + tolerance for left, right in zip(ordered, ordered[1:]))


def _oracle_seeds(seeds: list[SearchSeed]) -> list[SearchSeed]:
    """Return the grid points closest to the Berger point."""
    ordered = sorted(seeds, key=lambda seed: (_point_distance(seed.point), seed.index))
    return ordered[:ORACLE_CLOSEST_COUNT]


def _successful(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    """Return successful scout candidates sorted by residual norm."""
    return [candidate for candidate in sorted(candidates, key=lambda item: item.result.residual_norm) if candidate.result.failure is None]


def _evaluate_seed(seed: SearchSeed) -> SearchCandidate:
    """Evaluate one covering scout with the cheap residual config."""
    with mp.workdps(SCOUT_CONFIG.working_dps):
        return SearchCandidate(seed, mirror_residual(seed.point, SCOUT_CONFIG))


def _run_scouts(path: Path, seeds: list[SearchSeed], cached: dict[int, SearchCandidate]) -> list[SearchCandidate]:
    """Evaluate or reuse every grid scout."""
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


def _candidate_by_seed(candidates: list[SearchCandidate]) -> dict[int, SearchCandidate]:
    """Index scout candidates by seed index."""
    return {candidate.seed.index: candidate for candidate in candidates}


def _top_scouts(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    """Return branch-valid scouts selected for contraction probes."""
    return _successful(candidates)[:SCOUT_KEEP]


def _run_contraction_probe(candidate: SearchCandidate) -> ContractionProbe:
    """Run one order-6 one-step Newton contraction probe."""
    stage = newton_refine(candidate.seed.point, CONTRACTION_SETTINGS)
    return ContractionProbe(candidate, stage)


def _probe_payload(probe: ContractionProbe) -> dict:
    """Return JSON-ready data for one contraction probe."""
    return {
        "candidate": _candidate_payload(probe.candidate),
        "stage": _stage_payload(probe.stage),
        "contraction_ratio": _mp_string(_contraction_ratio(probe)),
    }


def _probe_from_payload(payload: dict) -> ContractionProbe:
    """Deserialize one contraction probe from a JSON event."""
    return ContractionProbe(recovery._candidate_from_payload(payload["candidate"]), recovery._stage_from_payload(payload["stage"]))


def _run_contraction_probes(path: Path, candidates: list[SearchCandidate], cached: dict[int, ContractionProbe]) -> list[ContractionProbe]:
    """Evaluate or reuse contraction probes for selected scouts."""
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


def _contraction_ratio(probe: ContractionProbe) -> mp.mpf:
    """Return final/initial residual ratio for one contraction probe."""
    initial = probe.stage.initial.residual_norm
    if initial == 0 or probe.stage.final.failure is not None:
        return mp.inf
    return probe.stage.final.residual_norm / initial


def _probe_successful(probe: ContractionProbe) -> bool:
    """Return whether one contraction probe produced a branch-valid final state."""
    return probe.stage.final.failure is None


def _point_key(point: MirrorSearchPoint) -> tuple[str, str, str, str]:
    """Return a stable key for deduplicating scaled points."""
    return tuple(_mp_string(value) or "" for value in (point.u, point.v, point.r, point.s))


def _dedupe_selected(items: list[SelectedCandidate]) -> list[SelectedCandidate]:
    """Remove duplicate selected coordinates while preserving order."""
    seen = set()
    output = []
    for item in items:
        key = _point_key(item.candidate.seed.point)
        if key not in seen:
            seen.add(key)
            output.append(SelectedCandidate(len(output) + 1, item.reason, item.candidate))
    return output


def _append_unique(
    selected: list[SelectedCandidate],
    probes: list[ContractionProbe],
    reason: str,
    quota: int,
) -> None:
    """Append up to quota probes whose coordinates are not already selected."""
    seen = {_point_key(item.candidate.seed.point) for item in selected}
    added = 0
    for probe in probes:
        key = _point_key(probe.candidate.seed.point)
        if key in seen:
            continue
        selected.append(SelectedCandidate(len(selected) + 1, reason, probe.candidate))
        seen.add(key)
        added += 1
        if added == quota:
            return


def _select_diverse(candidates: list[SearchCandidate], chosen: list[SearchCandidate], quota: int) -> list[SearchCandidate]:
    """Greedily select separated candidates without using Berger distance."""
    selected = list(chosen)
    output = []
    if not selected and candidates:
        selected.append(candidates[0])
        output.append(candidates[0])
    while len(output) < quota and len(selected) < len(candidates):
        keys = {_point_key(candidate.seed.point) for candidate in selected}
        remaining = [candidate for candidate in candidates if _point_key(candidate.seed.point) not in keys]
        picked = max(remaining, key=lambda item: min(recovery._distance_between(item.seed.point, other.seed.point) for other in selected))
        selected.append(picked)
        output.append(picked)
    return output


def _select_blind_candidates(
    probes: list[ContractionProbe],
    *,
    best_final_quota: int = SELECT_BEST_FINAL,
    best_ratio_quota: int = SELECT_BEST_RATIO,
    diverse_quota: int = SELECT_DIVERSE,
) -> list[SelectedCandidate]:
    """Select blind refinement seeds from contraction data only."""
    successful = [probe for probe in probes if _probe_successful(probe)]
    by_final = sorted(successful, key=lambda probe: probe.stage.final.residual_norm)
    by_ratio = sorted(successful, key=_contraction_ratio)
    selected: list[SelectedCandidate] = []
    _append_unique(selected, by_final, "best_final", best_final_quota)
    _append_unique(selected, by_ratio, "best_ratio", best_ratio_quota)
    chosen = [item.candidate for item in selected]
    diverse = _select_diverse([probe.candidate for probe in successful], chosen, diverse_quota) if successful else []
    offset = len(selected)
    selected += [SelectedCandidate(offset + index + 1, "diverse", candidate) for index, candidate in enumerate(diverse)]
    return _dedupe_selected(selected)


def _selection_payload(label: str, selections: list[SelectedCandidate]) -> dict:
    """Return JSON-ready data for one selection group."""
    return {
        "label": label,
        "selected_count": len(selections),
        "selections": [{"rank": item.rank, "reason": item.reason, "candidate": _candidate_payload(item.candidate)} for item in selections],
    }


def _run_refinement_set(
    label: str,
    candidates: list[SearchCandidate],
    berger_refs: tuple[MirrorResidualResult, ...],
    path: Path,
    completed_tracks: dict[int, CandidateTrack],
) -> list[CandidateTrack]:
    """Run or reuse high-order recovery tracks for selected candidates."""
    tracks = []
    for candidate in candidates:
        seed = candidate.seed
        if seed.index in completed_tracks:
            track = completed_tracks[seed.index]
            print(f"{label} seed={seed.index}: reused {track.classification}", flush=True)
        else:
            track = recovery._run_local_track(seed, candidate.result, berger_refs, path)
            completed_tracks[seed.index] = track
            print(f"{label} seed={seed.index}: {track.classification}", flush=True)
        tracks.append(track)
    return tracks


def _track_payload(track: CandidateTrack) -> dict:
    """Return JSON-ready data for one candidate track."""
    return recovery._track_payload(track)


def _output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped output paths for covering calibration."""
    return _common_output_paths(OUTPUT_DIR, RANDOM_SEED, OUTPUT_SUFFIX, now)


def _run_start_payload(jsonl_path: Path, summary_path: Path) -> dict:
    """Return metadata that must match to resume a covering run."""
    return {
        "random_seed": RANDOM_SEED,
        "calibration_version": CALIBRATION_VERSION,
        "grid_coordinates": [_mp_string(value) for value in GRID_COORDINATES],
        "grid_size": GRID_SIZE,
        "scout_keep": SCOUT_KEEP,
        "oracle_closest_count": ORACLE_CLOSEST_COUNT,
        "select_best_final": SELECT_BEST_FINAL,
        "select_best_ratio": SELECT_BEST_RATIO,
        "select_diverse": SELECT_DIVERSE,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
    }


def _summary_path_for_jsonl(path: Path) -> Path:
    """Return the summary path paired with one JSONL checkpoint."""
    return path.with_name(f"{path.stem}-summary.json")


def _checkpoint_is_compatible(path: Path) -> bool:
    """Return whether one checkpoint can be resumed by this recipe."""
    if recovery._jsonl_has_event(path, "run_summary"):
        return False
    starts = [event for event in recovery._jsonl_events(path) if event.get("event") == "run_start"]
    if not starts:
        return False
    expected = _run_start_payload(path, _summary_path_for_jsonl(path))
    return all(starts[-1].get(key) == value for key, value in expected.items() if key not in {"jsonl_path", "summary_path"})


def _latest_incomplete_checkpoint() -> Path | None:
    """Return the newest compatible unfinished covering checkpoint."""
    candidates = sorted(OUTPUT_DIR.glob(f"*-seed{RANDOM_SEED}{OUTPUT_SUFFIX}.jsonl"), reverse=True)
    return next((path for path in candidates if _checkpoint_is_compatible(path)), None)


def _resume_or_new_paths(now: datetime | None = None) -> tuple[Path, Path, bool]:
    """Return output paths and whether they resume an unfinished run."""
    if now is None:
        checkpoint = _latest_incomplete_checkpoint()
        if checkpoint is not None:
            return checkpoint, _summary_path_for_jsonl(checkpoint), True
    jsonl_path, summary_path = _output_paths(now)
    return jsonl_path, summary_path, False


def _load_contraction_probes(path: Path) -> dict[int, ContractionProbe]:
    """Load completed contraction probes from a JSONL checkpoint."""
    probes = {}
    for event in recovery._jsonl_events(path):
        if event.get("event") == "contraction_probe":
            probe = _probe_from_payload(event)
            probes[probe.candidate.seed.index] = probe
    return probes


def _build_summary(scouts: list[SearchCandidate], oracle_tracks: list[CandidateTrack], blind_tracks: list[CandidateTrack]) -> CoveringSummary:
    """Build the compact run summary."""
    successes = sum(candidate.result.failure is None for candidate in scouts)
    oracle = any(track.classification == "recovered_berger" for track in oracle_tracks)
    blind = any(track.classification == "recovered_berger" for track in blind_tracks)
    return CoveringSummary(successes, oracle, blind, len(blind_tracks))


def _summary_payload(
    summary: CoveringSummary,
    references,
    scouts,
    probes,
    oracle_selections,
    oracle_tracks,
    blind_selections,
    blind_tracks,
) -> dict:
    """Return JSON-ready final summary."""
    return {
        "berger_references": [_result_payload(result) for result in references],
        "grid_size": len(scouts),
        "scout_successes": summary.scout_successes,
        "top_scouts": [_candidate_payload(candidate) for candidate in _top_scouts(scouts)[:20]],
        "contraction_probes": [_probe_payload(probe) for probe in probes],
        "oracle_recovered": summary.oracle_recovered,
        "blind_recovered": summary.blind_recovered,
        "oracle_selections": [_candidate_payload(candidate) for candidate in oracle_selections],
        "blind_selections": [{"rank": item.rank, "reason": item.reason, "candidate": _candidate_payload(item.candidate)} for item in blind_selections],
        "oracle_tracks": [_track_payload(track) for track in oracle_tracks],
        "blind_tracks": [_track_payload(track) for track in blind_tracks],
    }


def _print_summary(summary: CoveringSummary, scouts: list[SearchCandidate], probes: list[ContractionProbe]) -> None:
    """Print a compact human-readable covering summary."""
    best = _successful(scouts)[:5]
    print(f"scout successes: {summary.scout_successes}/{len(scouts)}", flush=True)
    print(f"contraction probes: {len(probes)}", flush=True)
    for candidate in best:
        print(f"  scout seed={candidate.seed.index}, norm={mp.nstr(candidate.result.residual_norm, 12)}", flush=True)
    print(f"oracle recovered Berger: {summary.oracle_recovered}", flush=True)
    print(f"blind recovered Berger: {summary.blind_recovered}", flush=True)


def main() -> None:
    """Run the deterministic covering recovery calibration."""
    jsonl_path, summary_path, resumed = _resume_or_new_paths()
    if resumed:
        print(f"resuming JSONL checkpoint {jsonl_path}", flush=True)
        scouts_cache = recovery._load_scout_candidates(jsonl_path)
        probe_cache = _load_contraction_probes(jsonl_path)
        track_cache = recovery._load_classified_tracks(jsonl_path)
    else:
        print(f"writing JSONL events to {jsonl_path}", flush=True)
        _write_jsonl_event(jsonl_path, _event("run_start", _run_start_payload(jsonl_path, summary_path)))
        scouts_cache = {}
        probe_cache = {}
        track_cache = {}

    references = recovery._reference_residuals()
    recovery._print_references(references)
    berger_refs = references[-2:]
    seeds = _grid_seeds()
    scouts = _run_scouts(jsonl_path, seeds, scouts_cache)
    scout_by_seed = _candidate_by_seed(scouts)

    oracle_candidates = [scout_by_seed[seed.index] for seed in _oracle_seeds(seeds)]
    _write_jsonl_event(jsonl_path, _event("selection", _selection_payload("oracle_cover_track", [SelectedCandidate(index + 1, "closest_to_berger", candidate) for index, candidate in enumerate(oracle_candidates)])))
    oracle_tracks = _run_refinement_set("oracle_cover_track", oracle_candidates, berger_refs, jsonl_path, track_cache)

    top_scouts = _top_scouts(scouts)
    probes = _run_contraction_probes(jsonl_path, top_scouts, probe_cache)
    blind_selections = _select_blind_candidates(probes)
    _write_jsonl_event(jsonl_path, _event("selection", _selection_payload("blind_selection_track", blind_selections)))
    blind_tracks = _run_refinement_set("blind_selection_track", [item.candidate for item in blind_selections], berger_refs, jsonl_path, track_cache)

    summary = _build_summary(scouts, oracle_tracks, blind_tracks)
    payload = _summary_payload(summary, references, scouts, probes, oracle_candidates, oracle_tracks, blind_selections, blind_tracks)
    _print_summary(summary, scouts, probes)
    _write_jsonl_event(jsonl_path, _event("run_summary", payload))
    _write_summary(summary_path, payload)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
