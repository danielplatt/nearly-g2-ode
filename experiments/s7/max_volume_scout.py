"""Maximal-volume scout grids for fixed-chart S7 searches."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable

from mpmath import mp

from solver.max_volume import MAX_VOLUME_VERSION, MaxVolumeSettings, max_volume_match

from ..shared.g2_max_volume_common import SCOUT_SETTINGS, match_payload, settings_payload
from ..shared.non_mirrored_common import RANDOM_SEED, _event, _mp_string, _output_paths, _write_jsonl_event, _write_summary
from . import search_common as s7


SCOUT_VERSION = "s7-max-volume-scout-v1"
OUTPUT_DIR = Path("output/s7_max_volume_scouts")
OUTPUT_SUFFIX = SCOUT_VERSION


@dataclass(frozen=True)
class S7MaxVolumeSeed:
    """One deterministic S7 max-volume scout seed."""

    index: int
    target: str
    region: str
    source: str
    grid_index: tuple[int, ...]
    point: s7.S7SearchPoint


def scout_seed_count(
    targets: tuple[str, ...],
    spacing: mp.mpf,
    region: str,
    limit: int | None = None,
) -> int:
    """Return seed count after any debugging limit."""
    count = 1
    for axis in s7.scout_axes(spacing, region):
        count *= len(axis)
    count *= len(targets)
    return min(count, limit) if limit is not None else count


def scout_grid_metadata(
    targets: tuple[str, ...],
    spacing: mp.mpf,
    region: str,
    limit: int | None,
    settings: MaxVolumeSettings,
) -> dict:
    """Return JSON-ready S7 scout grid metadata."""
    axes = s7.scout_axes(spacing, region)
    scout_region = s7._scout_region(region)
    full_per_target = 1
    for axis in axes:
        full_per_target *= len(axis)
    full_count = full_per_target * len(targets)
    return {
        "region": scout_region.name,
        "targets": list(targets),
        "coordinate_names": list(scout_region.coordinate_names),
        "bounds": [[_mp_string(low), _mp_string(high)] for low, high in scout_region.bounds],
        "parameterization": scout_region.parameterization,
        "max_grid_spacing": _mp_string(spacing),
        "axis_counts": [len(axis) for axis in axes],
        "full_per_target": full_per_target,
        "full_seed_count": full_count,
        "seed_count": min(full_count, limit) if limit is not None else full_count,
        "limit": limit,
        "max_tau": _mp_string(settings.max_tau),
    }


def scout_seeds(
    targets: tuple[str, ...],
    spacing: mp.mpf,
    region: str,
    limit: int | None = None,
) -> list[S7MaxVolumeSeed]:
    """Return deterministic S7 max-volume scout seeds."""
    axes = s7.scout_axes(spacing, region)
    index_axes = tuple(range(len(axis)) for axis in axes)
    seeds: list[S7MaxVolumeSeed] = []
    for target in targets:
        s7._target(target)
        for grid_index in product(*index_axes):
            if limit is not None and len(seeds) >= limit:
                return seeds
            values = tuple(axis[axis_index] for axis, axis_index in zip(axes, grid_index))
            seeds.append(S7MaxVolumeSeed(len(seeds), target, region, "s7_max_volume_grid", grid_index, s7._point_from_values(values)))
    return seeds


def _evaluate_seed_payload(seed: S7MaxVolumeSeed, settings: MaxVolumeSettings = SCOUT_SETTINGS) -> dict:
    """Evaluate one S7 max-volume scout seed."""
    target = s7._target(seed.target)
    with mp.workdps(settings.config.working_dps):
        params, _config = s7.params_from_s7_scaled(
            seed.point,
            base_params=target.params_builder(),
            template_config=settings.config,
            region=seed.region,
        )
        match = max_volume_match(params, settings)
    return {
        "seed_index": seed.index,
        "target": seed.target,
        "region": seed.region,
        "source": seed.source,
        "grid_index": list(seed.grid_index),
        "seed_point": s7._point_payload(seed.point),
        "physical": {
            "left": {
                "a": _mp_string(params.left.a),
                "c": _mp_string(params.left.c),
                "alpha": _mp_string(params.left.alpha),
            },
            "right_chart": params.right_chart,
            "fixed_right_label": None if params.fixed_right is None else params.fixed_right.label,
        },
        "result": match_payload(match),
    }


def _evaluate_seed_payloads(
    seeds: list[S7MaxVolumeSeed],
    workers: int,
    settings: MaxVolumeSettings,
    chunksize: int | None = None,
) -> Iterable[dict]:
    """Yield JSON-ready S7 scout payloads."""
    if workers <= 1:
        for seed in seeds:
            yield _evaluate_seed_payload(seed, settings)
        return
    actual_chunksize = chunksize or 4
    try:
        executor = ProcessPoolExecutor(max_workers=workers)
    except (OSError, PermissionError):
        print("process workers unavailable; falling back to threads", flush=True)
        executor = ThreadPoolExecutor(max_workers=workers)
    with executor:
        yield from executor.map(_evaluate_seed_payload, seeds, [settings] * len(seeds), chunksize=actual_chunksize)


def _jsonl_events(path: Path):
    """Yield complete JSONL events."""
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


def _completed_seed_indices(path: Path) -> set[int]:
    """Return already persisted scout seed indices."""
    return {int(event["seed_index"]) for event in _jsonl_events(path) if event.get("event") == "scout_result"}


def _scout_payloads(path: Path) -> list[dict]:
    """Return completed scout payloads."""
    return [event for event in _jsonl_events(path) if event.get("event") == "scout_result"]


def _run_has_summary(path: Path) -> bool:
    """Return whether checkpoint has final summary."""
    return any(event.get("event") == "run_summary" for event in _jsonl_events(path))


def _summary_path_for_jsonl(path: Path) -> Path:
    """Return paired summary path."""
    return path.with_name(f"{path.stem}-summary.json")


def _new_output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped output paths."""
    return _output_paths(OUTPUT_DIR, OUTPUT_SUFFIX, now)


def _run_start_payload(
    jsonl_path: Path,
    summary_path: Path,
    targets: tuple[str, ...],
    spacing: mp.mpf,
    region: str,
    limit: int | None,
    settings: MaxVolumeSettings,
) -> dict:
    """Return JSON-ready run metadata."""
    return {
        "random_seed": RANDOM_SEED,
        "max_volume_version": MAX_VOLUME_VERSION,
        "scout_version": SCOUT_VERSION,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "grid": scout_grid_metadata(targets, spacing, region, limit, settings),
        "settings": settings_payload(settings),
    }


def _checkpoint_is_compatible(
    path: Path,
    targets: tuple[str, ...],
    spacing: mp.mpf,
    region: str,
    limit: int | None,
    settings: MaxVolumeSettings,
) -> bool:
    """Return whether an incomplete checkpoint can be resumed."""
    if _run_has_summary(path):
        return False
    starts = [event for event in _jsonl_events(path) if event.get("event") == "run_start"]
    if not starts:
        return False
    expected = _run_start_payload(path, _summary_path_for_jsonl(path), targets, spacing, region, limit, settings)
    ignored = {"jsonl_path", "summary_path"}
    return all(starts[-1].get(key) == value for key, value in expected.items() if key not in ignored)


def _latest_incomplete_checkpoint(
    targets: tuple[str, ...],
    spacing: mp.mpf,
    region: str,
    limit: int | None,
    settings: MaxVolumeSettings,
) -> Path | None:
    """Return newest compatible incomplete checkpoint."""
    candidates = sorted(OUTPUT_DIR.glob(f"*-seed{RANDOM_SEED}-{OUTPUT_SUFFIX}.jsonl"), reverse=True)
    return next((path for path in candidates if _checkpoint_is_compatible(path, targets, spacing, region, limit, settings)), None)


def _resume_or_new_paths(
    targets: tuple[str, ...],
    spacing: mp.mpf,
    region: str,
    limit: int | None,
    settings: MaxVolumeSettings,
    resume: bool,
) -> tuple[Path, Path, bool]:
    """Return output paths, resuming compatible work if requested."""
    if resume:
        checkpoint = _latest_incomplete_checkpoint(targets, spacing, region, limit, settings)
        if checkpoint is not None:
            return checkpoint, _summary_path_for_jsonl(checkpoint), True
    jsonl_path, summary_path = _new_output_paths()
    return jsonl_path, summary_path, False


def _payload_success(payload: dict) -> bool:
    """Return whether one payload matched both max-volume events."""
    return payload["result"]["failure"] is None


def _payload_status(payload: dict) -> str:
    """Return a compact status label for summary counts."""
    result = payload["result"]
    if result["failure"] is None:
        return "ok"
    left_status = result["left"]["status"]
    right_status = result["right"]["status"]
    if left_status != "max_volume":
        return f"left:{left_status}"
    return f"right:{right_status}"


def _payload_norm(payload: dict) -> mp.mpf:
    """Return one residual norm."""
    value = payload["result"]["residual_norm"]
    return mp.inf if value is None else mp.mpf(value)


def _compact_payload(payload: dict) -> dict:
    """Return compact summary data."""
    return {
        "seed_index": payload["seed_index"],
        "target": payload["target"],
        "region": payload["region"],
        "grid_index": payload["grid_index"],
        "seed_point": payload["seed_point"],
        "residual_norm": payload["result"]["residual_norm"],
        "failure": payload["result"]["failure"],
        "reconstructed_interval": payload["result"]["reconstructed_interval"],
        "interval_error": payload["result"]["interval_error"],
        "physical": payload["physical"],
    }


def _neighbor_offsets(dim: int) -> list[tuple[int, ...]]:
    """Return nearest-neighbor offsets."""
    return [offset for offset in product((-1, 0, 1), repeat=dim) if any(offset)]


def _local_minima(payloads: list[dict], axis_counts: list[int]) -> list[dict]:
    """Return target-wise nearest-neighbor local minima."""
    by_key = {(payload["target"], tuple(payload["grid_index"])): payload for payload in payloads}
    offsets = _neighbor_offsets(len(axis_counts))
    minima = []
    for payload in payloads:
        if not _payload_success(payload):
            continue
        key = tuple(payload["grid_index"])
        norm = _payload_norm(payload)
        is_minimum = True
        for offset in offsets:
            neighbor_key = tuple(index + delta for index, delta in zip(key, offset))
            if any(index < 0 or index >= axis_counts[axis] for axis, index in enumerate(neighbor_key)):
                continue
            neighbor = by_key.get((payload["target"], neighbor_key))
            if neighbor is not None and _payload_success(neighbor) and _payload_norm(neighbor) < norm:
                is_minimum = False
                break
        if is_minimum:
            minima.append(payload)
    return sorted(minima, key=lambda item: (_payload_norm(item), int(item["seed_index"])))


def _summary_payload(jsonl_path: Path, metadata: dict, best_limit: int = 30) -> dict:
    """Return final summary payload."""
    payloads = _scout_payloads(jsonl_path)
    successes = [payload for payload in payloads if _payload_success(payload)]
    counts = Counter(_payload_status(payload) for payload in payloads)
    best = sorted(successes, key=_payload_norm)[:best_limit]
    minima = _local_minima(payloads, metadata["grid"]["axis_counts"])[:best_limit]
    return {
        **metadata,
        "scout_count": len(payloads),
        "classification_counts": dict(counts),
        "best_scouts": [_compact_payload(payload) for payload in best],
        "best_local_minima": [_compact_payload(payload) for payload in minima],
    }


def _run_scouts(
    seeds: list[S7MaxVolumeSeed],
    jsonl_path: Path,
    workers: int,
    settings: MaxVolumeSettings,
    progress_every: int,
    chunksize: int | None,
) -> None:
    """Evaluate missing seeds and append JSONL events."""
    completed = _completed_seed_indices(jsonl_path)
    pending = [seed for seed in seeds if seed.index not in completed]
    total = len(seeds)
    print(f"loaded completed scouts: {len(completed)}/{total}", flush=True)
    print(f"pending scouts: {len(pending)} with workers={workers}", flush=True)
    done = len(completed)
    for payload in _evaluate_seed_payloads(pending, workers, settings, chunksize):
        _write_jsonl_event(jsonl_path, _event("scout_result", payload))
        done += 1
        if progress_every and (done == total or done % progress_every == 0):
            print(f"scouts complete: {done}/{total}", flush=True)


def _default_workers() -> int:
    """Return conservative worker count."""
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def _positive_int(value: str) -> int:
    """Parse positive int."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    """Parse nonnegative int."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_mpf(value: str) -> mp.mpf:
    """Parse positive mpf."""
    parsed = mp.mpf(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parse_targets(value: str) -> tuple[str, ...]:
    """Parse comma-separated target names."""
    targets = tuple(item.strip() for item in value.split(",") if item.strip())
    if not targets:
        raise argparse.ArgumentTypeError("at least one target is required")
    unknown = [target for target in targets if target not in s7.TARGETS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown S7 targets: {', '.join(unknown)}")
    return targets


def _settings_from_args(args: argparse.Namespace) -> MaxVolumeSettings:
    """Build max-volume scout settings from CLI args."""
    config = SCOUT_SETTINGS.config
    if args.order is not None:
        config = type(config)(
            args.order,
            args.dps,
            args.target_dps,
            mp.mpf(args.step_safety),
            args.sample_points,
            config.match_t,
        )
    max_tau = None if args.max_tau is None else mp.mpf(args.max_tau)
    return MaxVolumeSettings(config, max_tau=max_tau, bisection_steps=args.bisection_steps, event_tolerance=mp.mpf(args.event_tolerance))


def _print_dry_run(targets: tuple[str, ...], spacing: mp.mpf, region: str, limit: int | None, settings: MaxVolumeSettings) -> None:
    """Print grid size without writing output."""
    metadata = scout_grid_metadata(targets, spacing, region, limit, settings)
    print(f"region: {region}", flush=True)
    print(f"targets: {list(targets)}", flush=True)
    print(f"max spacing: {mp.nstr(spacing, 12)}", flush=True)
    print(f"axis counts: {metadata['axis_counts']}", flush=True)
    print(f"scout points: {metadata['seed_count']} (full {metadata['full_seed_count']})", flush=True)
    print(f"settings: order={settings.config.series_order}, dps={settings.config.working_dps}", flush=True)
    print("serial estimate at 0.15-0.5s/point: "
          f"{int(metadata['seed_count']) * 0.15 / 3600:.2f}-{int(metadata['seed_count']) * 0.5 / 3600:.2f}h", flush=True)


def main(argv: list[str] | None = None) -> None:
    """Run or resume an S7 maximal-volume scout grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=_parse_targets, default=s7.TARGET_NAMES)
    parser.add_argument("--region", choices=tuple(sorted(s7.SCOUT_REGIONS)), default=s7.DEFAULT_SCOUT_REGION.name)
    parser.add_argument("--spacing", type=_positive_mpf, default=s7.DEFAULT_SCOUT_SPACING)
    parser.add_argument("--workers", type=_positive_int, default=_default_workers())
    parser.add_argument("--limit", type=_nonnegative_int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--progress-every", type=_positive_int, default=1000)
    parser.add_argument("--chunksize", type=_positive_int, default=None)
    parser.add_argument("--order", type=int, default=None, help="override scout Taylor order")
    parser.add_argument("--dps", type=int, default=SCOUT_SETTINGS.config.working_dps)
    parser.add_argument("--target-dps", type=int, default=SCOUT_SETTINGS.config.target_dps)
    parser.add_argument("--step-safety", default=str(SCOUT_SETTINGS.config.step_safety))
    parser.add_argument("--sample-points", type=int, default=SCOUT_SETTINGS.config.sample_points)
    parser.add_argument("--bisection-steps", type=int, default=SCOUT_SETTINGS.bisection_steps)
    parser.add_argument("--event-tolerance", default=str(SCOUT_SETTINGS.event_tolerance))
    parser.add_argument("--max-tau", default=None, help="optional local-time event cap")
    args = parser.parse_args(argv)
    settings = _settings_from_args(args)
    if args.dry_run:
        _print_dry_run(args.targets, args.spacing, args.region, args.limit, settings)
        return
    jsonl_path, summary_path, resumed = _resume_or_new_paths(args.targets, args.spacing, args.region, args.limit, settings, not args.no_resume)
    metadata = _run_start_payload(jsonl_path, summary_path, args.targets, args.spacing, args.region, args.limit, settings)
    if resumed:
        print(f"resuming JSONL checkpoint {jsonl_path}", flush=True)
    else:
        print(f"writing JSONL events to {jsonl_path}", flush=True)
        _write_jsonl_event(jsonl_path, _event("run_start", metadata))
    seeds = scout_seeds(args.targets, args.spacing, args.region, args.limit)
    print(f"S7 max-volume scout seeds: {len(seeds)}", flush=True)
    _run_scouts(seeds, jsonl_path, args.workers, settings, args.progress_every, args.chunksize)
    summary = _summary_payload(jsonl_path, metadata)
    _write_jsonl_event(jsonl_path, _event("run_summary", summary))
    _write_summary(summary_path, summary)
    print(f"classification counts: {summary['classification_counts']}", flush=True)
    if summary["best_scouts"]:
        best = summary["best_scouts"][0]
        print(f"best scout: seed={best['seed_index']} target={best['target']} residual={best['residual_norm']}", flush=True)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
