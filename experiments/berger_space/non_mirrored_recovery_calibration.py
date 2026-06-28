"""Calibrate whether the non-mirrored search can rediscover Berger."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from random import Random

from mpmath import mp

from problem import DEFAULT_CONFIG, SolverConfig
from solver.two_sided_refinement import (
    TwoSidedCandidateTrack,
    TwoSidedNewtonSettings,
    two_sided_newton_refine,
)
from solver.two_sided_shooting import (
    BASE_TWO_SIDED_POINT,
    TwoSidedResidualResult,
    TwoSidedSearchPoint,
    two_sided_residual,
)

from ..shared.non_mirrored_common import (
    RANDOM_SEED,
    S_MIN,
    SearchCandidate,
    SearchSeed,
    _coordinates,
    _event,
    _mp_string,
    _output_paths as _common_output_paths,
    _point_distance,
    _point_from_values,
    _result_payload,
    _track_payload,
    _verification_norms,
    _write_jsonl_event,
    _write_summary,
)


OUTPUT_DIR = Path("output/non_mirrored_calibration")
CALIBRATION_VERSION = "recovery-v3"
OUTPUT_SUFFIX = "non-mirrored-recovery-v3"
SHELL_RADII = ("1e-4", "1e-3", "1e-2", "3e-2", "1e-1", "2e-1", "3e-1")
RANDOM_SHELL_SAMPLES = 2
LOCAL_BOX_RADIUS = mp.mpf("0.3")
LOCAL_BOX_SAMPLES = 20

SCOUT_CONFIG = SolverConfig(4, 30, 15, mp.mpf("0.95"), 0, DEFAULT_CONFIG.match_t)
ORDER6_CONFIG = SolverConfig(6, 40, 20, mp.mpf("0.9"), 0, DEFAULT_CONFIG.match_t)
ORDER10_CONFIG = SolverConfig(10, 70, 30, mp.mpf("0.7"), 1, DEFAULT_CONFIG.match_t)
VERIFY14_CONFIG = SolverConfig(14, 90, 35, mp.mpf("0.6"), 2, DEFAULT_CONFIG.match_t)
VERIFY18_CONFIG = SolverConfig(18, 110, 45, mp.mpf("0.55"), 2, DEFAULT_CONFIG.match_t)
VERIFY_CONFIGS = (VERIFY14_CONFIG, VERIFY18_CONFIG)
REFERENCE_CONFIGS = (SCOUT_CONFIG, ORDER6_CONFIG, ORDER10_CONFIG, VERIFY14_CONFIG, VERIFY18_CONFIG)

MAX_NEWTON_COORDINATE = mp.mpf("2")
ORDER6_SETTINGS = TwoSidedNewtonSettings(
    "order-6-calibration",
    ORDER6_CONFIG,
    mp.mpf("1e-3"),
    mp.mpf("1e-8"),
    3,
    max_abs_coordinate=MAX_NEWTON_COORDINATE,
    min_s_coordinate=S_MIN,
)
ORDER10_SETTINGS = TwoSidedNewtonSettings(
    "order-10-calibration",
    ORDER10_CONFIG,
    mp.mpf("3e-4"),
    mp.mpf("1e-10"),
    3,
    max_abs_coordinate=MAX_NEWTON_COORDINATE,
    min_s_coordinate=S_MIN,
)
ORDER14_SETTINGS = TwoSidedNewtonSettings(
    "order-14-correction",
    VERIFY14_CONFIG,
    mp.mpf("1e-4"),
    mp.mpf("1e-12"),
    2,
    max_abs_coordinate=MAX_NEWTON_COORDINATE,
    min_s_coordinate=S_MIN,
)


def _axis_shell_seeds(radius: mp.mpf, start_index: int) -> list[SearchSeed]:
    """Return the 14 coordinate-axis seeds at one max-norm radius."""
    seeds = []
    for coordinate in range(7):
        for sign in (-1, 1):
            values = [mp.zero for _ in range(7)]
            values[coordinate] = sign * radius
            point = _point_from_values(values)
            seeds.append(SearchSeed(start_index + len(seeds), f"shell_{mp.nstr(radius, 8)}", "axis", point))
    return seeds


def _random_shell_point(radius: mp.mpf, rng: Random) -> TwoSidedSearchPoint:
    """Return one reproducible random point on a 7D max-norm shell."""
    values = [mp.mpf(rng.uniform(-float(radius), float(radius))) for _ in range(7)]
    face = rng.randrange(7)
    values[face] = radius if rng.randrange(2) else -radius
    return _point_from_values(values)


def _shell_seeds(radius: mp.mpf, start_index: int, rng: Random) -> list[SearchSeed]:
    """Return axis and random shell seeds for one radius."""
    seeds = _axis_shell_seeds(radius, start_index)
    region = f"shell_{mp.nstr(radius, 8)}"
    for _ in range(RANDOM_SHELL_SAMPLES):
        seeds.append(SearchSeed(start_index + len(seeds), region, "random_shell", _random_shell_point(radius, rng)))
    return seeds


def _local_box_seeds(start_index: int, rng: Random) -> list[SearchSeed]:
    """Return blind local-box seeds around Berger."""
    seeds = []
    radius = float(LOCAL_BOX_RADIUS)
    for index in range(LOCAL_BOX_SAMPLES):
        values = [rng.uniform(-radius, radius) for _ in range(7)]
        seeds.append(SearchSeed(start_index + index, "local_box", "random_box", _point_from_values(values)))
    return seeds


def _calibration_seeds(seed: int = RANDOM_SEED) -> list[SearchSeed]:
    """Return all deterministic shell and local-box calibration seeds."""
    rng = Random(seed)
    seeds: list[SearchSeed] = []
    for radius_text in SHELL_RADII:
        seeds.extend(_shell_seeds(mp.mpf(radius_text), len(seeds), rng))
    seeds.extend(_local_box_seeds(len(seeds), rng))
    return seeds


def _evaluate_seed(seed: SearchSeed) -> SearchCandidate:
    """Evaluate one cheap scout residual."""
    with mp.workdps(SCOUT_CONFIG.working_dps):
        return SearchCandidate(seed, two_sided_residual(seed.point, SCOUT_CONFIG))


def _verify_point(point: TwoSidedSearchPoint) -> tuple[TwoSidedResidualResult, ...]:
    """Evaluate one point at the high-order verification configs."""
    results = []
    for config in VERIFY_CONFIGS:
        with mp.workdps(config.working_dps):
            results.append(two_sided_residual(point, config))
    return tuple(results)


def _reference_residuals() -> tuple[TwoSidedResidualResult, ...]:
    """Return Berger reference residuals at all calibration orders."""
    results = []
    for config in REFERENCE_CONFIGS:
        with mp.workdps(config.working_dps):
            results.append(two_sided_residual(BASE_TWO_SIDED_POINT, config))
    return tuple(results)


def _track_final(track: TwoSidedCandidateTrack) -> TwoSidedResidualResult:
    """Return the final residual carried by one track."""
    return track.stages[-1].final if track.stages else track.scout_result


def _verification_thresholds(references: tuple[TwoSidedResidualResult, ...]) -> tuple[mp.mpf, ...]:
    """Return Berger-relative order-14/order-18 recovery thresholds."""
    return tuple(max(mp.mpf("1e-8"), mp.mpf("1000") * result.residual_norm) for result in references)


def _stable_within_factor(norms: tuple[mp.mpf, ...], factor: mp.mpf) -> bool:
    """Return whether verification norms are stable within a multiplicative factor."""
    positive = [norm for norm in norms if norm != 0]
    return len(norms) >= 2 and (not positive or max(positive) <= factor * min(positive))


def _has_failed_stage(track: TwoSidedCandidateTrack) -> bool:
    """Return whether a refinement stage ended in a fatal status."""
    fatal = {"branch_failure", "jacobian_failure", "no_improvement"}
    return any(stage.status in fatal or stage.final.failure for stage in track.stages)


def _deserves_order10(stage) -> bool:
    """Return whether an order-6 stage deserves order-10 refinement."""
    return stage.final.failure is None and stage.final.residual_norm < stage.initial.residual_norm


def _deserves_order14(stage) -> bool:
    """Return whether a low-order attractor deserves order-14 correction."""
    final = stage.final
    return final.failure is None and (final.residual_norm < mp.mpf("1e-6") or _point_distance(final.point) < mp.mpf("0.02"))


def _deserves_verification(stage) -> bool:
    """Return whether a refined point deserves expensive high-order verification."""
    final = stage.final
    return final.failure is None and (final.residual_norm < mp.mpf("1e-4") or _point_distance(final.point) < mp.mpf("0.05"))


def _classify_track(track: TwoSidedCandidateTrack, references: tuple[TwoSidedResidualResult, ...]) -> str:
    """Classify one non-mirrored recovery calibration track."""
    if track.scout_result.failure or any(result.failure for result in track.verifications):
        return "failed"
    final = _track_final(track)
    norms = _verification_norms(track)
    if final.residual_norm < mp.mpf("1e-8") and norms and max(norms) > mp.mpf("1e-4"):
        return "finite_order_artifact"
    if len(norms) == 2 and _point_distance(final.point) < mp.mpf("1e-3"):
        if all(norm <= threshold for norm, threshold in zip(norms, _verification_thresholds(references))):
            return "recovered_berger"
    if len(norms) == 2 and _point_distance(final.point) >= mp.mpf("0.05"):
        if max(norms) < mp.mpf("1e-8") and _stable_within_factor(norms, mp.mpf("10")):
            return "possible_non_berger_root"
    return "failed" if _has_failed_stage(track) else "inconclusive"


def _run_track(seed: SearchSeed, references: tuple[TwoSidedResidualResult, ...]) -> TwoSidedCandidateTrack:
    """Run scout, order-6, order-10, and verification for one seed."""
    scout = _evaluate_seed(seed)
    if scout.result.failure:
        return TwoSidedCandidateTrack(seed.index, seed.region, seed.point, scout.result, (), (), "failed")
    order6 = two_sided_newton_refine(seed.point, ORDER6_SETTINGS)
    if order6.final.failure or order6.status in {"jacobian_failure", "no_improvement", "branch_failure"}:
        track = TwoSidedCandidateTrack(seed.index, seed.region, seed.point, scout.result, (order6,), (), "inconclusive")
        return TwoSidedCandidateTrack(seed.index, seed.region, seed.point, scout.result, (order6,), (), _classify_track(track, references))
    if not _deserves_order10(order6):
        track = TwoSidedCandidateTrack(seed.index, seed.region, seed.point, scout.result, (order6,), (), "inconclusive")
        return TwoSidedCandidateTrack(seed.index, seed.region, seed.point, scout.result, track.stages, (), _classify_track(track, references))
    order10 = two_sided_newton_refine(order6.final.point, ORDER10_SETTINGS)
    stages = [order6, order10]
    if _deserves_order14(order10):
        stages.append(two_sided_newton_refine(order10.final.point, ORDER14_SETTINGS))
    verifications = _verify_point(stages[-1].final.point) if _deserves_verification(stages[-1]) else ()
    track = TwoSidedCandidateTrack(seed.index, seed.region, seed.point, scout.result, tuple(stages), verifications, "inconclusive")
    return TwoSidedCandidateTrack(seed.index, seed.region, seed.point, scout.result, track.stages, verifications, _classify_track(track, references))


def _classification_payload(track: TwoSidedCandidateTrack) -> dict:
    """Return JSON-ready data for one classified calibration track."""
    payload = _track_payload(track)
    payload["source"] = "recovery_calibration"
    return payload


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
    """Return whether one JSONL checkpoint contains an event type."""
    return any(event.get("event") == event_type for event in _jsonl_events(path))


def _run_start_payload(jsonl_path: Path, summary_path: Path) -> dict:
    """Return checkpoint metadata for this calibration recipe."""
    return {
        "random_seed": RANDOM_SEED,
        "calibration_version": CALIBRATION_VERSION,
        "shell_radii": list(SHELL_RADII),
        "random_shell_samples": RANDOM_SHELL_SAMPLES,
        "local_box_radius": _mp_string(LOCAL_BOX_RADIUS),
        "local_box_samples": LOCAL_BOX_SAMPLES,
        "max_newton_coordinate": _mp_string(MAX_NEWTON_COORDINATE),
        "s_min": _mp_string(S_MIN),
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
    }


def _output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped output paths for the calibration."""
    return _common_output_paths(OUTPUT_DIR, OUTPUT_SUFFIX, now)


def _summary_path_for_jsonl(path: Path) -> Path:
    """Return the summary path paired with a JSONL checkpoint."""
    return path.with_name(f"{path.stem}-summary.json")


def _checkpoint_is_compatible(path: Path) -> bool:
    """Return whether one checkpoint can be resumed by this recipe."""
    if _jsonl_has_event(path, "run_summary"):
        return False
    starts = [event for event in _jsonl_events(path) if event.get("event") == "run_start"]
    if not starts:
        return False
    expected = _run_start_payload(path, _summary_path_for_jsonl(path))
    return all(starts[-1].get(key) == value for key, value in expected.items() if key not in {"jsonl_path", "summary_path"})


def _latest_incomplete_checkpoint() -> Path | None:
    """Return the newest compatible incomplete checkpoint, if present."""
    candidates = sorted(OUTPUT_DIR.glob(f"*-seed{RANDOM_SEED}-{OUTPUT_SUFFIX}.jsonl"), reverse=True)
    return next((path for path in candidates if _checkpoint_is_compatible(path)), None)


def _resume_or_new_paths(now: datetime | None = None) -> tuple[Path, Path, bool]:
    """Return output paths, resuming an incomplete checkpoint when possible."""
    if now is None:
        checkpoint = _latest_incomplete_checkpoint()
        if checkpoint is not None:
            return checkpoint, _summary_path_for_jsonl(checkpoint), True
    jsonl_path, summary_path = _output_paths(now)
    return jsonl_path, summary_path, False


def _completed_seed_indices(path: Path) -> set[int]:
    """Return seed indices already classified in a checkpoint."""
    return {int(event["seed_index"]) for event in _jsonl_events(path) if event.get("event") == "candidate_classification"}


def _classified_payloads(path: Path) -> list[dict]:
    """Return all classified track payloads from a checkpoint."""
    return [event for event in _jsonl_events(path) if event.get("event") == "candidate_classification"]


def _shell_counts(payloads: list[dict]) -> dict[str, dict[str, int]]:
    """Return classification counts by shell/local-box region."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for payload in payloads:
        counts[payload["region"]][payload["classification"]] += 1
    return {region: dict(counter) for region, counter in counts.items()}


def _largest_recovery_radius(counts: dict[str, dict[str, int]], fraction: mp.mpf) -> str | None:
    """Return the largest shell radius whose recovery fraction exceeds a threshold."""
    largest = None
    for radius_text in SHELL_RADII:
        region = f"shell_{mp.nstr(mp.mpf(radius_text), 8)}"
        region_counts = counts.get(region, {})
        total = sum(region_counts.values())
        recovered = region_counts.get("recovered_berger", 0)
        if total and mp.mpf(recovered) / total >= fraction:
            largest = radius_text
    return largest


def _summary_payload(path: Path, references: tuple[TwoSidedResidualResult, ...], seeds: list[SearchSeed]) -> dict:
    """Return JSON-ready final calibration summary."""
    payloads = _classified_payloads(path)
    counts = Counter(payload["classification"] for payload in payloads)
    shell_counts = _shell_counts(payloads)
    return {
        "reference_residuals": [_result_payload(result) for result in references],
        "seed_count": len(seeds),
        "classified_count": len(payloads),
        "classification_counts": dict(counts),
        "shell_counts": shell_counts,
        "largest_any_recovery_radius": _largest_recovery_radius(shell_counts, mp.mpf("1e-30")),
        "largest_eighty_percent_recovery_radius": _largest_recovery_radius(shell_counts, mp.mpf("0.8")),
        "local_box_recovered": shell_counts.get("local_box", {}).get("recovered_berger", 0) > 0,
        "tracks": payloads,
    }


def _print_references(references: tuple[TwoSidedResidualResult, ...]) -> None:
    """Print Berger reference residuals for the calibration configs."""
    for result in references:
        print(
            f"reference order {result.config.series_order}: norm={mp.nstr(result.residual_norm, 12)} failure={result.failure}",
            flush=True,
        )


def _print_summary(payload: dict) -> None:
    """Print a compact human-readable calibration summary."""
    print(f"classified: {payload['classified_count']}/{payload['seed_count']}", flush=True)
    print(f"classifications: {payload['classification_counts']}", flush=True)
    print(f"largest radius with any Berger recovery: {payload['largest_any_recovery_radius']}", flush=True)
    print(f"largest radius with >=80% recovery: {payload['largest_eighty_percent_recovery_radius']}", flush=True)
    print(f"local box recovered Berger: {payload['local_box_recovered']}", flush=True)


def main() -> None:
    """Run the non-mirrored Berger recovery calibration."""
    jsonl_path, summary_path, resumed = _resume_or_new_paths()
    if resumed:
        print(f"resuming JSONL checkpoint {jsonl_path}", flush=True)
    else:
        print(f"writing JSONL events to {jsonl_path}", flush=True)
        _write_jsonl_event(jsonl_path, _event("run_start", _run_start_payload(jsonl_path, summary_path)))
    references = _reference_residuals()
    _print_references(references)
    seeds = _calibration_seeds()
    completed = _completed_seed_indices(jsonl_path)
    for index, seed in enumerate(seeds, start=1):
        if seed.index in completed:
            continue
        track = _run_track(seed, references[-2:])
        _write_jsonl_event(jsonl_path, _event("candidate_classification", _classification_payload(track)))
        print(
            f"seed {index}/{len(seeds)} ({seed.region}, {seed.source}): {track.classification}",
            flush=True,
        )
    payload = _summary_payload(jsonl_path, references, seeds)
    _print_summary(payload)
    _write_jsonl_event(jsonl_path, _event("run_summary", payload))
    _write_summary(summary_path, payload)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
