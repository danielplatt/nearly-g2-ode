"""Calibrate whether the mirrored search can rediscover Berger."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random

from mpmath import mp

from problem import DEFAULT_CONFIG, SolverConfig
from solver.mirror_refinement import CandidateTrack, NewtonSettings, RefinementStageReport, newton_refine
from solver.mirror_shooting import BASE_POINT, MirrorResidualResult, MirrorSearchPoint, mirror_residual, params_from_scaled

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
    _track_payload as _common_track_payload,
    _write_jsonl_event,
    _write_summary,
)


SCOUT_CONFIG = SolverConfig(4, 30, 15, mp.mpf("0.95"), 0, DEFAULT_CONFIG.match_t)
ORDER6_CONFIG = SolverConfig(6, 40, 20, mp.mpf("0.9"), 0, DEFAULT_CONFIG.match_t)
ORDER10_CONFIG = SolverConfig(10, 70, 30, mp.mpf("0.7"), 1, DEFAULT_CONFIG.match_t)
VERIFY14_CONFIG = SolverConfig(14, 90, 35, mp.mpf("0.6"), 2, DEFAULT_CONFIG.match_t)
VERIFY18_CONFIG = SolverConfig(18, 110, 45, mp.mpf("0.55"), 2, DEFAULT_CONFIG.match_t)
VERIFY_CONFIGS = (VERIFY14_CONFIG, VERIFY18_CONFIG)
REFERENCE_CONFIGS = (SCOUT_CONFIG, ORDER6_CONFIG, ORDER10_CONFIG, VERIFY14_CONFIG, VERIFY18_CONFIG)

RANDOM_SEED = 1729
CALIBRATION_VERSION = "high_order_v2"
SHELL_RADII = ("1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2", "1e-1", "3e-1")
RANDOM_SHELL_SAMPLES = 4
BROAD_BOX_SAMPLES = 800
BROAD_BEST_QUOTA = 40
BROAD_DIVERSE_QUOTA = 40
OUTPUT_DIR = Path("output/mirror_calibration")

MAX_NEWTON_COORDINATE = mp.mpf("12")
BROAD_HIGH_ORDER_SEED_RADIUS = mp.mpf("0.05")
HIGH_ORDER_CORRECTION_RADIUS = mp.mpf("0.02")
ORDER6_SETTINGS = NewtonSettings("order-6", ORDER6_CONFIG, mp.mpf("1e-3"), mp.mpf("1e-8"), 3, max_abs_coordinate=MAX_NEWTON_COORDINATE)
ORDER10_SETTINGS = NewtonSettings("order-10", ORDER10_CONFIG, mp.mpf("3e-4"), mp.mpf("1e-10"), 3, max_abs_coordinate=MAX_NEWTON_COORDINATE)
ORDER14_SETTINGS = NewtonSettings("order-14", VERIFY14_CONFIG, mp.mpf("1e-4"), mp.mpf("1e-12"), 2, max_abs_coordinate=MAX_NEWTON_COORDINATE)


@dataclass(frozen=True)
class CalibrationSummary:
    """Compact summary data for one completed calibration run."""

    shell_counts: dict[str, dict[str, int]]
    largest_any_recovery: str | None
    largest_eighty_percent_recovery: str | None
    broad_recovered: bool


def _point_from_values(values) -> MirrorSearchPoint:
    """Build one search point from four numeric values."""
    return MirrorSearchPoint(*(mp.mpf(value) for value in values))


def _axis_shell_seeds(radius: mp.mpf, start_index: int) -> list[SearchSeed]:
    """Return the eight coordinate-axis seeds at max-norm radius."""
    seeds = []
    for coord in range(4):
        for sign in (-1, 1):
            values = [mp.zero, mp.zero, mp.zero, mp.zero]
            values[coord] = sign * radius
            seeds.append(SearchSeed(start_index + len(seeds), f"shell_{mp.nstr(radius, 8)}", "axis", _point_from_values(values)))
    return seeds


def _random_shell_point(radius: mp.mpf, rng: Random) -> MirrorSearchPoint:
    """Sample one point exactly on the max-norm shell of radius."""
    values = [mp.mpf(rng.uniform(-float(radius), float(radius))) for _ in range(4)]
    face = rng.randrange(4)
    values[face] = radius if rng.randrange(2) else -radius
    return _point_from_values(values)


def _shell_seeds(radius: mp.mpf, start_index: int, rng: Random) -> list[SearchSeed]:
    """Return deterministic axis plus random seeds for one shell."""
    seeds = _axis_shell_seeds(radius, start_index)
    region = f"shell_{mp.nstr(radius, 8)}"
    for _ in range(RANDOM_SHELL_SAMPLES):
        seeds.append(SearchSeed(start_index + len(seeds), region, "random_shell", _random_shell_point(radius, rng)))
    return seeds


def _local_shell_seeds(seed: int = RANDOM_SEED) -> list[SearchSeed]:
    """Return all deterministic local shell calibration seeds."""
    rng = Random(seed)
    seeds: list[SearchSeed] = []
    for radius_text in SHELL_RADII:
        seeds.extend(_shell_seeds(mp.mpf(radius_text), len(seeds), rng))
    return seeds


def _broad_box_seeds(seed: int = RANDOM_SEED, count: int = BROAD_BOX_SAMPLES, start_index: int = 0) -> list[SearchSeed]:
    """Return broad-box seeds in u,v,s in [-1,1] and r in [-3,3]."""
    rng = Random(seed)
    seeds = []
    for index in range(count):
        point = _point_from_values((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-3, 3), rng.uniform(-1, 1)))
        seeds.append(SearchSeed(start_index + index, "broad_box", "random_box", point))
    return seeds


def _evaluate_seed(seed: SearchSeed, config: SolverConfig = SCOUT_CONFIG) -> SearchCandidate:
    """Evaluate one scout seed with a residual configuration."""
    with mp.workdps(config.working_dps):
        return SearchCandidate(seed, mirror_residual(seed.point, config))


def _final_result(track: CandidateTrack) -> MirrorResidualResult:
    """Return the latest residual result carried by one track."""
    return track.stages[-1].final if track.stages else track.scout_result


def _verification_norms(track: CandidateTrack) -> tuple[mp.mpf, ...]:
    """Return successful verification norms for one track."""
    return tuple(result.residual_norm for result in track.verifications if result.failure is None)


def _stable_within_factor(norms: tuple[mp.mpf, ...], factor: mp.mpf) -> bool:
    """Return whether verification norms are stable within a factor."""
    positive = [norm for norm in norms if norm != 0]
    return len(norms) >= 2 and (not positive or max(positive) <= factor * min(positive))


def _verification_thresholds(berger_refs: tuple[MirrorResidualResult, ...]) -> tuple[mp.mpf, ...]:
    """Return Berger-comparison thresholds for order-14/order-18 verification."""
    return tuple(max(mp.mpf("1e-10"), mp.mpf("1000") * result.residual_norm) for result in berger_refs)


def _has_failed_stage(track: CandidateTrack) -> bool:
    """Return whether refinement ended with a fatal calibration status."""
    fatal = {"branch_failure", "jacobian_failure", "no_improvement"}
    return any(stage.status in fatal or stage.final.failure for stage in track.stages)


def _classify_track(track: CandidateTrack, berger_refs: tuple[MirrorResidualResult, ...]) -> str:
    """Classify one calibration track."""
    if track.scout_result.failure or any(result.failure for result in track.verifications):
        return "failed"
    final = _final_result(track)
    norms = _verification_norms(track)
    if final.residual_norm < mp.mpf("1e-8") and norms and max(norms) > mp.mpf("1e-4"):
        return "finite_order_artifact"
    if len(norms) == len(berger_refs) and _point_distance(final.point) < mp.mpf("1e-3"):
        if all(norm <= threshold for norm, threshold in zip(norms, _verification_thresholds(berger_refs))):
            return "recovered_berger"
    if len(norms) == 2 and _point_distance(final.point) >= mp.mpf("0.05"):
        if max(norms) < mp.mpf("1e-8") and _stable_within_factor(norms, mp.mpf("10")):
            return "possible_non_berger_root"
    return "failed" if _has_failed_stage(track) else "inconclusive"


def _failure_reason(track: CandidateTrack) -> str:
    """Return a short reason for a failed or inconclusive track."""
    if track.scout_result.failure:
        return track.scout_result.failure
    for stage in track.stages:
        if stage.status in {"branch_failure", "jacobian_failure", "no_improvement"}:
            return stage.status
        if stage.final.failure:
            return stage.final.failure
    return track.classification


def _verify_point(point: MirrorSearchPoint) -> tuple[MirrorResidualResult, ...]:
    """Evaluate one point with the high-order verification configs."""
    results = []
    for config in VERIFY_CONFIGS:
        with mp.workdps(config.working_dps):
            results.append(mirror_residual(point, config))
    return tuple(results)


def _should_promote(stage: RefinementStageReport, allow_far_candidate: bool) -> bool:
    """Return whether one order-6 output deserves order-10 verification."""
    final = stage.final
    if final.failure is not None:
        return False
    near_berger = _point_distance(final.point) < mp.mpf("0.05") and final.residual_norm < mp.mpf("1e-6")
    far_lead = allow_far_candidate and _point_distance(final.point) >= mp.mpf("0.05") and final.residual_norm < mp.mpf("1e-8")
    return near_berger or far_lead


def _needs_high_order_correction(track: CandidateTrack) -> bool:
    """Return whether a low-order attractor needs order-14 correction."""
    final = _final_result(track)
    if final.failure is not None:
        return False
    if _point_distance(final.point) >= HIGH_ORDER_CORRECTION_RADIUS:
        return False
    return final.residual_norm < mp.mpf("1e-8")


def _replace_track(track: CandidateTrack, stages, verifications, classification: str) -> CandidateTrack:
    """Return one track with updated refinement data."""
    return CandidateTrack(track.seed_rank, track.seed_region, track.seed_point, track.scout_result, tuple(stages), tuple(verifications), classification)


def _track_payload(track: CandidateTrack) -> dict:
    """Return JSON-ready data for one calibration track."""
    return _common_track_payload(track, _final_result(track))


def _run_track(
    seed: SearchSeed,
    scout: MirrorResidualResult,
    berger_refs: tuple[MirrorResidualResult, ...],
    path: Path,
    *,
    allow_far_candidate: bool,
) -> CandidateTrack:
    """Run staged refinement, verification, and classification for one seed."""
    order6 = newton_refine(seed.point, ORDER6_SETTINGS)
    track = CandidateTrack(seed.index, seed.region, seed.point, scout, (order6,), (), "inconclusive")
    _write_jsonl_event(path, _event("refinement_stage", {"seed": _seed_payload(seed), "stage": _stage_payload(order6)}))
    if _should_promote(order6, allow_far_candidate):
        track = _run_promoted_track(track, berger_refs, path)
    if _needs_high_order_correction(track):
        track = _run_high_order_correction(track, path)
    classified = _replace_track(track, track.stages, track.verifications, _classify_track(track, berger_refs))
    _write_jsonl_event(path, _event("candidate_classification", _track_payload(classified)))
    return classified


def _run_local_track(seed: SearchSeed, scout: MirrorResidualResult, berger_refs: tuple[MirrorResidualResult, ...], path: Path) -> CandidateTrack:
    """Run the stricter local recovery pipeline for one shell seed."""
    order10 = newton_refine(seed.point, ORDER10_SETTINGS)
    track = CandidateTrack(seed.index, seed.region, seed.point, scout, (order10,), (), "inconclusive")
    _write_jsonl_event(path, _event("refinement_stage", {"seed": _seed_payload(seed), "stage": _stage_payload(order10)}))
    if _needs_high_order_correction(track):
        track = _run_high_order_correction(track, path)
    if track.verifications:
        verifications = track.verifications
    else:
        verifications = _verify_point(_final_result(track).point)
        for result in verifications:
            _write_jsonl_event(path, _event("verification", {"seed_index": track.seed_rank, "result": _result_payload(result)}))
    track = _replace_track(track, track.stages, verifications, "inconclusive")
    classified = _replace_track(track, track.stages, track.verifications, _classify_track(track, berger_refs))
    _write_jsonl_event(path, _event("candidate_classification", _track_payload(classified)))
    return classified


def _run_promoted_track(track: CandidateTrack, berger_refs: tuple[MirrorResidualResult, ...], path: Path) -> CandidateTrack:
    """Run order-10 refinement and verification for one promoted track."""
    order10 = newton_refine(track.stages[-1].final.point, ORDER10_SETTINGS)
    verifications = _verify_point(order10.final.point)
    _write_jsonl_event(path, _event("refinement_stage", {"seed_index": track.seed_rank, "stage": _stage_payload(order10)}))
    for result in verifications:
        _write_jsonl_event(path, _event("verification", {"seed_index": track.seed_rank, "result": _result_payload(result)}))
    return _replace_track(track, track.stages + (order10,), verifications, "inconclusive")


def _run_high_order_correction(track: CandidateTrack, path: Path) -> CandidateTrack:
    """Run order-14 correction from a near-Berger low-order attractor."""
    order14 = newton_refine(_final_result(track).point, ORDER14_SETTINGS)
    verifications = _verify_point(order14.final.point)
    _write_jsonl_event(path, _event("refinement_stage", {"seed_index": track.seed_rank, "stage": _stage_payload(order14)}))
    for result in verifications:
        _write_jsonl_event(path, _event("verification", {"seed_index": track.seed_rank, "result": _result_payload(result)}))
    return _replace_track(track, track.stages + (order14,), verifications, "inconclusive")


def _seed_payload(seed: SearchSeed) -> dict:
    """Return JSON-ready seed data."""
    return {
        "seed_index": seed.index,
        "region": seed.region,
        "source": seed.source,
        "distance": _mp_string(_point_distance(seed.point)),
        "point": _point_payload(seed.point),
    }


def _run_local_shells(
    path: Path,
    berger_refs: tuple[MirrorResidualResult, ...],
    completed_tracks: dict[int, CandidateTrack],
) -> list[CandidateTrack]:
    """Run the local shell calibration and persist every event."""
    tracks = []
    for seed in _local_shell_seeds():
        if seed.index in completed_tracks:
            tracks.append(completed_tracks[seed.index])
            print(f"local {seed.region} seed={seed.index}: reused {tracks[-1].classification}", flush=True)
            continue
        _write_jsonl_event(path, _event("seed", _seed_payload(seed)))
        scout = mirror_residual(seed.point, ORDER10_CONFIG)
        tracks.append(_run_local_track(seed, scout, berger_refs, path))
        print(f"local {seed.region} seed={seed.index}: {tracks[-1].classification}", flush=True)
    return tracks


def _point_key(point: MirrorSearchPoint) -> tuple[str, str, str, str]:
    """Return a stable deduplication key for one point."""
    return tuple(_mp_string(value) or "" for value in (point.u, point.v, point.r, point.s))


def _distance_between(left: MirrorSearchPoint, right: MirrorSearchPoint) -> mp.mpf:
    """Return max-distance between two scaled points."""
    return max(abs(lval - rval) for lval, rval in zip((left.u, left.v, left.r, left.s), (right.u, right.v, right.r, right.s)))


def _select_diverse(candidates: list[SearchCandidate], chosen: list[SearchCandidate], quota: int) -> list[SearchCandidate]:
    """Greedily select max-min separated broad-box candidates."""
    selected = list(chosen)
    output = []
    while len(output) < quota and len(selected) < len(candidates):
        keys = {_point_key(candidate.seed.point) for candidate in selected}
        remaining = [candidate for candidate in candidates if _point_key(candidate.seed.point) not in keys]
        picked = max(remaining, key=lambda item: min(_distance_between(item.seed.point, other.seed.point) for other in selected))
        selected.append(picked)
        output.append(picked)
    return output


def _select_broad_candidates(candidates: list[SearchCandidate]) -> list[SelectedCandidate]:
    """Select best residual and diverse broad-box seeds for refinement."""
    successful = [candidate for candidate in sorted(candidates, key=lambda item: item.result.residual_norm) if candidate.result.failure is None]
    best = successful[:BROAD_BEST_QUOTA]
    diverse = _select_diverse(successful, best or successful[:1], BROAD_DIVERSE_QUOTA) if successful else []
    selected = [SelectedCandidate(index + 1, "best", candidate) for index, candidate in enumerate(best)]
    selected += [SelectedCandidate(len(selected) + index + 1, "diverse", candidate) for index, candidate in enumerate(diverse)]
    return selected


def _run_broad_box(
    path: Path,
    berger_refs: tuple[MirrorResidualResult, ...],
    completed_tracks: dict[int, CandidateTrack],
    completed_scouts: dict[int, SearchCandidate],
    start_index: int,
) -> tuple[list[SearchCandidate], list[SelectedCandidate], list[CandidateTrack]]:
    """Run broad-box scout selection and refinement."""
    candidates = []
    reused_scouts = 0
    for seed in _broad_box_seeds(start_index=start_index):
        if seed.index in completed_scouts:
            candidate = completed_scouts[seed.index]
            reused_scouts += 1
        else:
            _write_jsonl_event(path, _event("seed", _seed_payload(seed)))
            candidate = _evaluate_seed(seed, SCOUT_CONFIG)
            _write_jsonl_event(path, _event("scout_result", _candidate_payload(candidate)))
        candidates.append(candidate)
    if reused_scouts:
        print(f"broad-box scouts reused: {reused_scouts}/{len(candidates)}", flush=True)
    selections = _select_broad_candidates(candidates)
    tracks = []
    for selection in selections:
        seed = selection.candidate.seed
        if seed.index in completed_tracks:
            tracks.append(completed_tracks[seed.index])
            print(f"broad seed={seed.index}: reused {tracks[-1].classification}", flush=True)
            continue
        if _point_distance(seed.point) < BROAD_HIGH_ORDER_SEED_RADIUS:
            tracks.append(_run_local_track(seed, selection.candidate.result, berger_refs, path))
        else:
            tracks.append(_run_track(seed, selection.candidate.result, berger_refs, path, allow_far_candidate=True))
    return candidates, selections, tracks


def _reference_residuals() -> tuple[MirrorResidualResult, ...]:
    """Return Berger reference residuals for all calibration configs."""
    results = []
    for config in REFERENCE_CONFIGS:
        with mp.workdps(config.working_dps):
            results.append(mirror_residual(BASE_POINT, config))
    return tuple(results)


def _output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped JSONL and summary paths."""
    return _common_output_paths(OUTPUT_DIR, RANDOM_SEED, "-recovery-v2", now)


def _mp_from_payload(value):
    """Deserialize one optional mpmath value."""
    return None if value is None else mp.mpf(value)


def _point_from_payload(payload: dict) -> MirrorSearchPoint:
    """Deserialize one scaled point from JSON payload data."""
    return MirrorSearchPoint(*(mp.mpf(payload[name]) for name in ("u", "v", "r", "s")))


def _config_for_payload(payload: dict) -> SolverConfig:
    """Return the closest baked-in config for a serialized result."""
    order = int(payload.get("config_order", DEFAULT_CONFIG.series_order))
    dps = int(payload.get("config_dps", DEFAULT_CONFIG.working_dps))
    for config in REFERENCE_CONFIGS:
        if config.series_order == order:
            return config
    return SolverConfig(order, dps, DEFAULT_CONFIG.target_dps, DEFAULT_CONFIG.step_safety, DEFAULT_CONFIG.sample_points, DEFAULT_CONFIG.match_t)


def _result_from_payload(payload: dict) -> MirrorResidualResult:
    """Deserialize a residual result from a JSONL event payload."""
    point = _point_from_payload(payload["point"])
    config = _config_for_payload(payload)
    params, local_config = params_from_scaled(point, template_config=config)
    branch = {key: mp.mpf(value) for key, value in payload.get("branch_diagnostics", {}).items()}
    return MirrorResidualResult(
        point=point,
        params=params,
        config=local_config,
        residual=tuple(mp.mpf(value) for value in payload.get("residual", ())),
        residual_norm=mp.mpf(payload["residual_norm"]),
        match_q=None,
        l_value=_mp_from_payload(payload.get("l_value")),
        patch_count=int(payload.get("patch_count", 0)),
        branch_diagnostics=branch,
        failure=payload.get("failure"),
    )


def _settings_from_payload(payload: dict) -> NewtonSettings:
    """Deserialize Newton settings from a refinement-stage payload."""
    config = _config_for_payload({"config_order": payload["order"], "config_dps": payload["working_dps"]})
    return NewtonSettings(
        payload["name"],
        config,
        mp.mpf(payload["fd_step"]),
        mp.mpf(payload["tolerance"]),
        int(payload["max_steps"]),
        max_abs_coordinate=_mp_from_payload(payload.get("max_abs_coordinate")),
        min_s_coordinate=_mp_from_payload(payload.get("min_s_coordinate")),
    )


def _stage_from_payload(payload: dict) -> RefinementStageReport:
    """Deserialize one refinement stage, omitting per-step details."""
    return RefinementStageReport(
        _settings_from_payload(payload["settings"]),
        _result_from_payload(payload["initial"]),
        _result_from_payload(payload["final"]),
        (),
        payload["status"],
    )


def _track_from_payload(payload: dict) -> CandidateTrack:
    """Deserialize one completed calibration track."""
    return CandidateTrack(
        int(payload["seed_index"]),
        payload["region"],
        _point_from_payload(payload["seed_point"]),
        _result_from_payload(payload["scout"]),
        tuple(_stage_from_payload(item) for item in payload.get("stages", ())),
        tuple(_result_from_payload(item) for item in payload.get("verifications", ())),
        payload["classification"],
    )


def _candidate_from_payload(payload: dict) -> SearchCandidate:
    """Deserialize one completed broad-box scout candidate."""
    seed = SearchSeed(
        int(payload["seed_index"]),
        payload["region"],
        payload["source"],
        _point_from_payload(payload["seed_point"]),
    )
    return SearchCandidate(seed, _result_from_payload(payload["result"]))


def _jsonl_events(path: Path):
    """Yield complete JSONL events, ignoring a possible partial final line."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _jsonl_has_event(path: Path, event_type: str) -> bool:
    """Return whether a JSONL checkpoint already contains an event type."""
    return any(event.get("event") == event_type for event in _jsonl_events(path))


def _run_start_payload(jsonl_path: Path, summary_path: Path) -> dict:
    """Return checkpoint metadata that must match for resume."""
    return {
        "random_seed": RANDOM_SEED,
        "calibration_version": CALIBRATION_VERSION,
        "radii": list(SHELL_RADII),
        "random_shell_samples": RANDOM_SHELL_SAMPLES,
        "broad_box_samples": BROAD_BOX_SAMPLES,
        "broad_best_quota": BROAD_BEST_QUOTA,
        "broad_diverse_quota": BROAD_DIVERSE_QUOTA,
        "broad_high_order_seed_radius": _mp_string(BROAD_HIGH_ORDER_SEED_RADIUS),
        "high_order_correction_radius": _mp_string(HIGH_ORDER_CORRECTION_RADIUS),
        "local_first_newton_order": ORDER10_CONFIG.series_order,
        "high_order_correction_order": VERIFY14_CONFIG.series_order,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
    }


def _checkpoint_is_compatible(path: Path) -> bool:
    """Return whether a JSONL file can be resumed by this calibration recipe."""
    if _jsonl_has_event(path, "run_summary"):
        return False
    starts = [event for event in _jsonl_events(path) if event.get("event") == "run_start"]
    if not starts:
        return False
    expected = _run_start_payload(path, _summary_path_for_jsonl(path))
    return all(starts[-1].get(key) == value for key, value in expected.items() if key not in {"jsonl_path", "summary_path"})


def _summary_path_for_jsonl(path: Path) -> Path:
    """Return the summary path paired with one JSONL checkpoint."""
    return path.with_name(f"{path.stem}-summary.json")


def _latest_incomplete_checkpoint() -> Path | None:
    """Return the newest compatible unfinished checkpoint, if one exists."""
    candidates = sorted(OUTPUT_DIR.glob(f"*-seed{RANDOM_SEED}-recovery-v2.jsonl"), reverse=True)
    return next((path for path in candidates if _checkpoint_is_compatible(path)), None)


def _resume_or_new_paths(now: datetime | None = None) -> tuple[Path, Path, bool]:
    """Return output paths and whether they resume an unfinished run."""
    if now is None:
        checkpoint = _latest_incomplete_checkpoint()
        if checkpoint is not None:
            return checkpoint, _summary_path_for_jsonl(checkpoint), True
    jsonl_path, summary_path = _output_paths(now)
    return jsonl_path, summary_path, False


def _load_classified_tracks(path: Path) -> dict[int, CandidateTrack]:
    """Load completed candidate tracks from a checkpoint."""
    tracks = {}
    for event in _jsonl_events(path):
        if event.get("event") == "candidate_classification":
            track = _track_from_payload(event)
            tracks[track.seed_rank] = track
    return tracks


def _load_scout_candidates(path: Path) -> dict[int, SearchCandidate]:
    """Load completed broad-box scout evaluations from a checkpoint."""
    candidates = {}
    for event in _jsonl_events(path):
        if event.get("event") == "scout_result":
            candidate = _candidate_from_payload(event)
            candidates[candidate.seed.index] = candidate
    return candidates


def _shell_counts(tracks: list[CandidateTrack]) -> dict[str, dict[str, int]]:
    """Return classification counts by shell radius."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for track in tracks:
        counts[track.seed_region][track.classification] += 1
    return {region: dict(counter) for region, counter in counts.items()}


def _largest_recovery_radius(counts: dict[str, dict[str, int]], fraction: mp.mpf) -> str | None:
    """Return largest shell radius whose recovery fraction meets a threshold."""
    recovered = None
    for radius_text in SHELL_RADII:
        region = f"shell_{mp.nstr(mp.mpf(radius_text), 8)}"
        total = sum(counts.get(region, {}).values())
        if total and counts.get(region, {}).get("recovered_berger", 0) / total >= fraction:
            recovered = radius_text
    return recovered


def _summarize(tracks: list[CandidateTrack], broad_tracks: list[CandidateTrack]) -> CalibrationSummary:
    """Build the final calibration summary."""
    counts = _shell_counts(tracks)
    any_recovery = _largest_recovery_radius(counts, mp.mpf("1") / 24)
    eighty = _largest_recovery_radius(counts, mp.mpf("0.8"))
    broad_recovered = any(track.classification == "recovered_berger" for track in broad_tracks)
    return CalibrationSummary(counts, any_recovery, eighty, broad_recovered)


def _print_references(references: tuple[MirrorResidualResult, ...]) -> None:
    """Print Berger reference residuals by configuration."""
    print("Berger reference residuals:", flush=True)
    for result in references:
        print(f"  order {result.config.series_order}: {mp.nstr(result.residual_norm, 12)}", flush=True)


def _print_shell_summary(summary: CalibrationSummary, tracks: list[CandidateTrack]) -> None:
    """Print local shell recovery rates and failure reasons."""
    print("\nlocal shell recovery:", flush=True)
    for radius_text in SHELL_RADII:
        region = f"shell_{mp.nstr(mp.mpf(radius_text), 8)}"
        subset = [track for track in tracks if track.seed_region == region]
        recovered = summary.shell_counts.get(region, {}).get("recovered_berger", 0)
        failures = Counter(_failure_reason(track) for track in subset if track.classification == "failed")
        print(
            f"  {radius_text}: recovered {recovered}/{len(subset)}, "
            f"classifications={summary.shell_counts.get(region, {})}, failures={dict(failures)}",
            flush=True,
        )
    print(f"largest radius with any recovery: {summary.largest_any_recovery}", flush=True)
    print(f"largest radius with >=80% recovery: {summary.largest_eighty_percent_recovery}", flush=True)


def _print_broad_summary(candidates: list[SearchCandidate], selections: list[SelectedCandidate], tracks: list[CandidateTrack]) -> None:
    """Print broad-box scout and recovery summary."""
    successful = [candidate for candidate in sorted(candidates, key=lambda item: item.result.residual_norm) if candidate.result.failure is None]
    ranks = {candidate.seed.index: index + 1 for index, candidate in enumerate(successful)}
    print("\nbroad-box selected seeds:", flush=True)
    for selection in selections:
        rank = ranks.get(selection.candidate.seed.index, "failed")
        print(f"  seed={selection.candidate.seed.index}, reason={selection.reason}, scout_rank={rank}", flush=True)
    print(f"broad-box recovered Berger: {any(track.classification == 'recovered_berger' for track in tracks)}", flush=True)


def _summary_payload(summary: CalibrationSummary, references, shell_tracks, broad_candidates, selections, broad_tracks) -> dict:
    """Return JSON-ready final summary."""
    return {
        "berger_references": [_result_payload(result) for result in references],
        "shell_counts": summary.shell_counts,
        "largest_any_recovery": summary.largest_any_recovery,
        "largest_eighty_percent_recovery": summary.largest_eighty_percent_recovery,
        "broad_recovered": summary.broad_recovered,
        "broad_candidates": [_candidate_payload(candidate) for candidate in broad_candidates],
        "broad_selections": [{"reason": item.reason, "candidate": _candidate_payload(item.candidate)} for item in selections],
        "shell_tracks": [_track_payload(track) for track in shell_tracks],
        "broad_tracks": [_track_payload(track) for track in broad_tracks],
    }


def main() -> None:
    """Run the Berger recovery calibration experiment."""
    jsonl_path, summary_path, resumed = _resume_or_new_paths()
    if resumed:
        print(f"resuming JSONL checkpoint {jsonl_path}", flush=True)
        completed_tracks = _load_classified_tracks(jsonl_path)
        completed_scouts = _load_scout_candidates(jsonl_path)
        print(
            f"loaded {len(completed_tracks)} completed tracks and {len(completed_scouts)} broad-box scouts",
            flush=True,
        )
    else:
        print(f"writing JSONL events to {jsonl_path}", flush=True)
        _write_jsonl_event(jsonl_path, _event("run_start", _run_start_payload(jsonl_path, summary_path)))
        completed_tracks = {}
        completed_scouts = {}
    references = _reference_residuals()
    _print_references(references)
    berger_refs = references[-2:]
    shell_tracks = _run_local_shells(jsonl_path, berger_refs, completed_tracks)
    broad_start_index = len(_local_shell_seeds())
    broad_candidates, selections, broad_tracks = _run_broad_box(
        jsonl_path,
        berger_refs,
        completed_tracks,
        completed_scouts,
        broad_start_index,
    )
    summary = _summarize(shell_tracks, broad_tracks)
    _print_shell_summary(summary, shell_tracks)
    _print_broad_summary(broad_candidates, selections, broad_tracks)
    payload = _summary_payload(summary, references, shell_tracks, broad_candidates, selections, broad_tracks)
    _write_jsonl_event(jsonl_path, _event("run_summary", payload))
    _write_summary(summary_path, payload)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
