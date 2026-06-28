"""Shared helpers for long mirror-sweep experiments."""

from __future__ import annotations

import json
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from types import FrameType

from mpmath import mp

from problem import SolverConfig
from solver.mirror_refinement import CandidateTrack, NewtonSettings, RefinementStageReport
from solver.mirror_shooting import MirrorResidualResult, MirrorSearchPoint, mirror_residual


@dataclass(frozen=True)
class RegionSpec:
    """One annular scout region with quotas."""

    name: str
    lower: mp.mpf
    upper: mp.mpf
    bounds: tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]
    samples: int
    best_quota: int
    diverse_quota: int
    promote_quota: int


@dataclass(frozen=True)
class BoxRegionSpec:
    """One rectangular scout region with an optional distance range."""

    name: str
    lower: mp.mpf
    upper: mp.mpf
    ranges: tuple[tuple[mp.mpf, mp.mpf], tuple[mp.mpf, mp.mpf], tuple[mp.mpf, mp.mpf], tuple[mp.mpf, mp.mpf]]
    samples: int
    best_quota: int
    diverse_quota: int
    promote_quota: int


@dataclass(frozen=True)
class SearchSeed:
    """One deterministic scout seed."""

    index: int
    region: str
    source: str
    point: MirrorSearchPoint


@dataclass(frozen=True)
class SearchCandidate:
    """One evaluated scout seed."""

    seed: SearchSeed
    result: MirrorResidualResult


@dataclass(frozen=True)
class SelectedCandidate:
    """One scout candidate selected for refinement."""

    rank: int
    reason: str
    candidate: SearchCandidate


class ScoutTimeoutError(TimeoutError):
    """Raised when one scout residual evaluation exceeds the per-seed limit."""


class StageTimeoutError(TimeoutError):
    """Raised when one refinement stage exceeds the per-stage limit."""


def _mp_string(value) -> str | None:
    """Serialize one mpmath value as a decimal string."""
    return None if value is None else mp.nstr(value, 80)


def _point_distance(point: MirrorSearchPoint) -> mp.mpf:
    """Return max-distance from Berger in scaled coordinates."""
    return max(abs(point.u), abs(point.v), abs(point.r), abs(point.s))


def _point_payload(point: MirrorSearchPoint) -> dict[str, str | None]:
    """Return JSON-ready scaled point coordinates."""
    return {name: _mp_string(value) for name, value in zip(("u", "v", "r", "s"), (point.u, point.v, point.r, point.s))}


def _point_from_values(values) -> MirrorSearchPoint:
    """Build one scaled point from four values."""
    return MirrorSearchPoint(*(mp.mpf(value) for value in values))


def _point_key(point: MirrorSearchPoint) -> tuple[str | None, str | None, str | None, str | None]:
    """Return a stable key for deduplicating points."""
    return tuple(_mp_string(value) for value in (point.u, point.v, point.r, point.s))


def _random_point(spec: RegionSpec, rng: Random) -> MirrorSearchPoint:
    """Sample one point from a region's rectangular bounds."""
    values = [rng.uniform(-float(bound), float(bound)) for bound in spec.bounds]
    return _point_from_values(values)


def _random_box_point(spec: BoxRegionSpec, rng: Random) -> MirrorSearchPoint:
    """Sample one point from an asymmetric rectangular region."""
    values = [rng.uniform(float(lower), float(upper)) for lower, upper in spec.ranges]
    return _point_from_values(values)


def _region_for_point(point: MirrorSearchPoint, regions: tuple[RegionSpec, ...]) -> str:
    """Return the first annular region containing one point."""
    distance = _point_distance(point)
    for spec in regions:
        if spec.lower <= distance <= spec.upper:
            return spec.name
    return "outside"


def _region_seeds(spec: RegionSpec, rng: Random, start_index: int) -> list[SearchSeed]:
    """Return reproducible random seeds in one annular region."""
    seeds: list[SearchSeed] = []
    attempts = 0
    while len(seeds) < spec.samples and attempts < spec.samples * 1000:
        attempts += 1
        point = _random_point(spec, rng)
        if spec.lower <= _point_distance(point) <= spec.upper:
            seeds.append(SearchSeed(start_index + len(seeds), spec.name, "random", point))
    if len(seeds) != spec.samples:
        raise RuntimeError(f"Could not sample enough points for region {spec.name!r}.")
    return seeds


def _box_region_seeds(spec: BoxRegionSpec, rng: Random, start_index: int) -> list[SearchSeed]:
    """Return reproducible random seeds in one asymmetric box region."""
    seeds: list[SearchSeed] = []
    attempts = 0
    while len(seeds) < spec.samples and attempts < spec.samples * 1000:
        attempts += 1
        point = _random_box_point(spec, rng)
        if spec.lower <= _point_distance(point) <= spec.upper:
            seeds.append(SearchSeed(start_index + len(seeds), spec.name, "box", point))
    if len(seeds) != spec.samples:
        raise RuntimeError(f"Could not sample enough points for box region {spec.name!r}.")
    return seeds


def _corner_seeds(regions: tuple[RegionSpec, ...], start_index: int = 0) -> list[SearchSeed]:
    """Return the fixed corner seeds from the short scout search."""
    seeds = []
    for u in (-0.7, 0.7):
        for v in (-0.7, 0.7):
            for r in (-2.0, 2.0):
                for s in (-0.7, 0.7):
                    point = _point_from_values((u, v, r, s))
                    seeds.append(SearchSeed(start_index + len(seeds), _region_for_point(point, regions), "corner", point))
    return seeds


def _search_seeds(regions: tuple[RegionSpec, ...], seed: int) -> list[SearchSeed]:
    """Return deterministic annular long-sweep seeds."""
    rng = Random(seed)
    seeds = _corner_seeds(regions)
    for spec in regions:
        seeds.extend(_region_seeds(spec, rng, len(seeds)))
    return seeds


def _result_payload(result: MirrorResidualResult) -> dict:
    """Return a compact JSON-ready residual result."""
    return {
        "point": _point_payload(result.point),
        "residual_norm": _mp_string(result.residual_norm),
        "residual": [_mp_string(value) for value in result.residual],
        "l_value": _mp_string(result.l_value),
        "failure": result.failure,
        "patch_count": result.patch_count,
        "branch_diagnostics": {key: _mp_string(value) for key, value in result.branch_diagnostics.items()},
        "config_order": result.config.series_order,
        "config_dps": result.config.working_dps,
    }


def _settings_payload(settings: NewtonSettings) -> dict:
    """Return JSON-ready Newton settings."""
    return {
        "name": settings.name,
        "order": settings.config.series_order,
        "working_dps": settings.config.working_dps,
        "fd_step": _mp_string(settings.fd_step),
        "tolerance": _mp_string(settings.tolerance),
        "max_steps": settings.max_steps,
        "max_abs_coordinate": _mp_string(settings.max_abs_coordinate),
        "min_s_coordinate": _mp_string(settings.min_s_coordinate),
    }


def _step_payload(step) -> dict:
    """Return JSON-ready data for one Newton step."""
    return {
        "index": step.index,
        "status": step.status,
        "damping": _mp_string(step.damping),
        "condition_number": _mp_string(step.condition_number),
        "delta": None if step.delta is None else [_mp_string(value) for value in step.delta],
        "before_norm": _mp_string(step.residual_before.residual_norm),
        "after_norm": _mp_string(step.residual_after.residual_norm),
        "trial_norms": [[_mp_string(damping), _mp_string(norm), failed, reason] for damping, norm, failed, reason in step.trial_norms],
    }


def _stage_payload(stage: RefinementStageReport) -> dict:
    """Return JSON-ready data for one refinement stage."""
    return {
        "settings": _settings_payload(stage.settings),
        "status": stage.status,
        "initial": _result_payload(stage.initial),
        "final": _result_payload(stage.final),
        "steps": [_step_payload(step) for step in stage.steps],
    }


def _event(event_type: str, payload: dict) -> dict:
    """Build one timestamped JSONL event."""
    return {"event": event_type, "time_utc": datetime.now(timezone.utc).isoformat(), **payload}


def _write_jsonl_event(path: Path, event: dict) -> None:
    """Append one JSON event and flush it immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()


def _output_paths(output_dir: Path, random_seed: int, suffix: str = "", now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped JSONL and summary output paths."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    stem = f"{stamp}-seed{random_seed}{suffix}"
    return output_dir / f"{stem}.jsonl", output_dir / f"{stem}-summary.json"


def _candidate_payload(candidate: SearchCandidate) -> dict:
    """Return JSON-ready scout candidate data."""
    return {
        "seed_index": candidate.seed.index,
        "region": candidate.seed.region,
        "source": candidate.seed.source,
        "distance": _mp_string(_point_distance(candidate.seed.point)),
        "seed_point": _point_payload(candidate.seed.point),
        "result": _result_payload(candidate.result),
    }


def _raise_timeout(message: str):
    """Build one signal handler that raises a timeout with a fixed message."""

    def handler(signum: int, frame: FrameType | None) -> None:
        raise TimeoutError(message)

    return handler


def _with_timeout(seconds: int, message: str, callback):
    """Run a callback with a wall-clock timeout."""
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout(message))
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return callback()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _timeout_result(seed: SearchSeed, config: SolverConfig, message: str) -> MirrorResidualResult:
    """Return a synthetic residual result for a timed-out scout seed."""
    from solver.mirror_shooting import params_from_scaled

    params, local_config = params_from_scaled(seed.point, template_config=config)
    return MirrorResidualResult(seed.point, params, local_config, (), mp.inf, None, None, 0, {}, message)


def _evaluate_seed_with_timeout(seed: SearchSeed, config: SolverConfig, timeout_seconds: int) -> SearchCandidate:
    """Evaluate one seed with a wall-clock timeout."""
    try:
        return _with_timeout(
            timeout_seconds,
            f"scout evaluation exceeded {timeout_seconds} seconds",
            lambda: SearchCandidate(seed, mirror_residual(seed.point, config)),
        )
    except TimeoutError as exc:
        return SearchCandidate(seed, _timeout_result(seed, config, str(exc)))


def _evaluate_seed(seed: SearchSeed, path: Path, config: SolverConfig, timeout_seconds: int) -> SearchCandidate:
    """Evaluate and persist one scout seed."""
    with mp.workdps(config.working_dps):
        candidate = _evaluate_seed_with_timeout(seed, config, timeout_seconds)
    _write_jsonl_event(path, _event("scout_result", _candidate_payload(candidate)))
    return candidate


def _sort_key(candidate: SearchCandidate) -> tuple[bool, mp.mpf]:
    """Sort successful candidates before failures by residual norm."""
    return candidate.result.failure is not None, candidate.result.residual_norm


def _successful(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    """Return successful candidates sorted by scout residual."""
    return [candidate for candidate in sorted(candidates, key=_sort_key) if candidate.result.failure is None]


def _point_distance_between(left: MirrorSearchPoint, right: MirrorSearchPoint) -> mp.mpf:
    """Return max-distance between two scaled points."""
    return max(abs(lval - rval) for lval, rval in zip((left.u, left.v, left.r, left.s), (right.u, right.v, right.r, right.s)))


def _diverse_candidates(
    candidates: list[SearchCandidate],
    selected: list[SearchCandidate],
    quota: int,
) -> list[SearchCandidate]:
    """Greedily select max-min separated candidates."""
    chosen = list(selected)
    output = []
    while len(output) < quota:
        chosen_keys = {_point_key(item.seed.point) for item in chosen}
        remaining = [candidate for candidate in candidates if _point_key(candidate.seed.point) not in chosen_keys]
        if not remaining:
            break
        picked = max(remaining, key=lambda candidate: min(_point_distance_between(candidate.seed.point, item.seed.point) for item in chosen))
        chosen.append(picked)
        output.append(picked)
    return output


def _select_region_candidates(spec: RegionSpec, candidates: list[SearchCandidate]) -> list[SelectedCandidate]:
    """Select best and diverse candidates for one region."""
    region_candidates = _successful([candidate for candidate in candidates if candidate.seed.region == spec.name])
    best = region_candidates[: spec.best_quota]
    diverse = _diverse_candidates(region_candidates, best, spec.diverse_quota)
    selected = [SelectedCandidate(index + 1, "best", candidate) for index, candidate in enumerate(best)]
    offset = len(selected)
    selected += [SelectedCandidate(offset + index + 1, "diverse", candidate) for index, candidate in enumerate(diverse)]
    return selected


def _track_payload(track: CandidateTrack, final_result: MirrorResidualResult) -> dict:
    """Return JSON-ready data for one candidate track."""
    return {
        "seed_index": track.seed_rank,
        "region": track.seed_region,
        "seed_point": _point_payload(track.seed_point),
        "final_point": _point_payload(final_result.point),
        "distance": _mp_string(_point_distance(final_result.point)),
        "classification": track.classification,
        "scout": _result_payload(track.scout_result),
        "stages": [_stage_payload(stage) for stage in track.stages],
        "verifications": [_result_payload(result) for result in track.verifications],
    }


def _region_summary(regions: tuple[RegionSpec, ...], candidates: list[SearchCandidate]) -> dict[str, dict[str, int]]:
    """Return scout success/failure counts by region."""
    summary = {}
    for spec in regions:
        subset = [candidate for candidate in candidates if candidate.seed.region == spec.name]
        successes = [candidate for candidate in subset if candidate.result.failure is None]
        summary[spec.name] = {"total": len(subset), "successes": len(successes), "failures": len(subset) - len(successes)}
    return summary


def _print_region_summary(summary: dict[str, dict[str, int]]) -> None:
    """Print scout counts by region."""
    for region, counts in summary.items():
        print(f"{region}: total={counts['total']}, successes={counts['successes']}, failures={counts['failures']}", flush=True)


def _print_region_best(regions: tuple[RegionSpec, ...], candidates: list[SearchCandidate], limit: int = 5) -> None:
    """Print a small per-region best-residual table."""
    for spec in regions:
        print(f"\nbest {spec.name} scouts:", flush=True)
        for candidate in _successful([item for item in candidates if item.seed.region == spec.name])[:limit]:
            print(
                f"  seed={candidate.seed.index}, norm={mp.nstr(candidate.result.residual_norm, 12)}, "
                f"distance={mp.nstr(_point_distance(candidate.seed.point), 8)}",
                flush=True,
            )


def _write_summary(path: Path, payload: dict) -> None:
    """Write the final summary JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
