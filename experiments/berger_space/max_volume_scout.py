"""Maximal-volume scout grids for Berger-space endpoint branches."""

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

from problem import LeftEndpointParameters, ProblemParameters, RightEndpointParameters
from solver.max_volume import MAX_VOLUME_VERSION, MaxVolumeSettings, max_volume_match

from ..shared.g2_max_volume_common import SCOUT_SETTINGS, match_payload, settings_payload
from ..shared.non_mirrored_common import RANDOM_SEED, _event, _mp_string, _output_paths as _common_output_paths, _write_jsonl_event, _write_summary
from . import non_mirrored_grid_search as grid_search


SCOUT_VERSION = "berger-max-volume-scout-v1"
OUTPUT_DIR = Path("output/berger_max_volume_scouts")
OUTPUT_SUFFIX = SCOUT_VERSION
COORDINATE_NAMES = ("u_left", "v_left", "r_left", "u_right", "v_right", "r_right")


@dataclass(frozen=True)
class BergerMaxVolumePoint:
    """Six scaled endpoint coordinates; the interval is reconstructed."""

    u_left: mp.mpf
    v_left: mp.mpf
    r_left: mp.mpf
    u_right: mp.mpf
    v_right: mp.mpf
    r_right: mp.mpf


@dataclass(frozen=True)
class BergerMaxVolumeSeed:
    """One deterministic max-volume scout seed."""

    index: int
    region: str
    source: str
    grid_index: tuple[int, ...]
    point: BergerMaxVolumePoint


def _coordinates(point: BergerMaxVolumePoint) -> tuple[mp.mpf, ...]:
    """Return scaled coordinates as a tuple."""
    return (point.u_left, point.v_left, point.r_left, point.u_right, point.v_right, point.r_right)


def _point_from_values(values) -> BergerMaxVolumePoint:
    """Build one max-volume scout point from numeric values."""
    parsed = tuple(mp.mpf(value) for value in values)
    if len(parsed) != 6:
        raise ValueError("Berger max-volume scout points need exactly six coordinates.")
    return BergerMaxVolumePoint(*parsed)


def params_from_max_volume_point(point: BergerMaxVolumePoint, base_params: ProblemParameters) -> ProblemParameters:
    """Convert six scaled coordinates into endpoint parameters."""
    left = LeftEndpointParameters(
        a=base_params.left.a * mp.exp(point.u_left),
        c=base_params.left.c * mp.exp(point.v_left),
        alpha=base_params.left.alpha * (1 + point.r_left),
    )
    right = RightEndpointParameters(
        d=base_params.right.d * mp.exp(point.u_right),
        f=base_params.right.f * mp.exp(point.v_right),
        omega=base_params.right.omega * (1 + point.r_right),
    )
    return ProblemParameters(
        lam=base_params.lam,
        interval_end=base_params.interval_end,
        left=left,
        right=right,
        right_chart=base_params.right_chart,
        fixed_right=base_params.fixed_right,
        left_mu=base_params.left_mu,
        right_mu=base_params.right_mu,
        p_signs=base_params.p_signs,
        right_p_signs=base_params.right_p_signs,
    )


def _base_params_payload(params: ProblemParameters) -> dict:
    """Return JSON-ready base parameters."""
    return grid_search._base_params_payload(params)


def _grid_region(region_name: str):
    """Return the underlying Berger 7D grid region."""
    return grid_search._grid_region(region_name)


def _grid_axes(region_name: str, spacing: mp.mpf, shift: str) -> tuple[tuple[mp.mpf, ...], ...]:
    """Return the six coordinate axes used by max-volume scouting."""
    return grid_search._grid_axes(_grid_region(region_name), spacing, shift)[:6]


def _axis_spacing(axis: tuple[mp.mpf, ...]) -> mp.mpf:
    """Return largest adjacent spacing on one axis."""
    return grid_search._axis_spacing(axis)


def scout_seed_count(region_name: str, spacing: mp.mpf, shift: str, limit: int | None = None) -> int:
    """Return seed count after any debugging limit."""
    count = 1
    for axis in _grid_axes(region_name, spacing, shift):
        count *= len(axis)
    return min(count, limit) if limit is not None else count


def scout_grid_metadata(
    region_name: str,
    spacing: mp.mpf,
    shift: str,
    limit: int | None,
    settings: MaxVolumeSettings,
) -> dict:
    """Return JSON-ready scout grid metadata."""
    region = _grid_region(region_name)
    axes = _grid_axes(region_name, spacing, shift)
    full_count = scout_seed_count(region_name, spacing, shift, None)
    return {
        "region": region.name,
        "shift": shift,
        "base_params": _base_params_payload(grid_search._base_params_for_region(region.name)),
        "coordinate_names": list(COORDINATE_NAMES),
        "bounds": [[_mp_string(low), _mp_string(high)] for low, high in region.bounds[:6]],
        "physical_odd_bounds": grid_search._physical_odd_bounds(region),
        "max_grid_spacing": _mp_string(spacing),
        "axis_counts": [len(axis) for axis in axes],
        "axis_spacings": [_mp_string(_axis_spacing(axis)) for axis in axes],
        "full_seed_count": full_count,
        "seed_count": min(full_count, limit) if limit is not None else full_count,
        "limit": limit,
        "max_tau": _mp_string(settings.max_tau),
    }


def scout_seeds(region_name: str, spacing: mp.mpf, shift: str, limit: int | None = None) -> list[BergerMaxVolumeSeed]:
    """Return deterministic max-volume scout seeds."""
    axes = _grid_axes(region_name, spacing, shift)
    seeds = []
    index_axes = tuple(range(len(axis)) for axis in axes)
    for index, grid_index in enumerate(product(*index_axes)):
        if limit is not None and index >= limit:
            break
        values = tuple(axis[axis_index] for axis, axis_index in zip(axes, grid_index))
        seeds.append(BergerMaxVolumeSeed(index, region_name, "max_volume_grid", grid_index, _point_from_values(values)))
    return seeds


def _point_payload(point: BergerMaxVolumePoint) -> dict:
    """Return JSON-ready scaled point coordinates."""
    return {name: _mp_string(value) for name, value in zip(COORDINATE_NAMES, _coordinates(point))}


def _physical_payload(point: BergerMaxVolumePoint, base_params: ProblemParameters) -> dict:
    """Return physical endpoint data for one scout point."""
    params = params_from_max_volume_point(point, base_params)
    return {
        "left": {
            "a": _mp_string(params.left.a),
            "c": _mp_string(params.left.c),
            "alpha": _mp_string(params.left.alpha),
        },
        "right": {
            "d": _mp_string(params.right.d),
            "f": _mp_string(params.right.f),
            "omega": _mp_string(params.right.omega),
        },
    }


def _evaluate_seed_payload(seed: BergerMaxVolumeSeed, settings: MaxVolumeSettings = SCOUT_SETTINGS) -> dict:
    """Evaluate one max-volume scout seed."""
    base_params = grid_search._base_params_for_region(seed.region)
    with mp.workdps(settings.config.working_dps):
        params = params_from_max_volume_point(seed.point, base_params)
        match = max_volume_match(params, settings)
    return {
        "seed_index": seed.index,
        "region": seed.region,
        "source": seed.source,
        "grid_index": list(seed.grid_index),
        "seed_point": _point_payload(seed.point),
        "physical": _physical_payload(seed.point, base_params),
        "result": match_payload(match),
    }


def _evaluate_seed_payloads(
    seeds: list[BergerMaxVolumeSeed],
    workers: int,
    settings: MaxVolumeSettings,
    chunksize: int | None = None,
) -> Iterable[dict]:
    """Yield JSON-ready scout payloads in stable order."""
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
    """Return whether the checkpoint is complete."""
    return any(event.get("event") == "run_summary" for event in _jsonl_events(path))


def _summary_path_for_jsonl(path: Path) -> Path:
    """Return paired summary path."""
    return path.with_name(f"{path.stem}-summary.json")


def _new_output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped output paths for this scout."""
    return _common_output_paths(OUTPUT_DIR, OUTPUT_SUFFIX, now)


def _run_start_payload(
    jsonl_path: Path,
    summary_path: Path,
    region_name: str,
    spacing: mp.mpf,
    shift: str,
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
        "grid": scout_grid_metadata(region_name, spacing, shift, limit, settings),
        "settings": settings_payload(settings),
    }


def _checkpoint_is_compatible(
    path: Path,
    region_name: str,
    spacing: mp.mpf,
    shift: str,
    limit: int | None,
    settings: MaxVolumeSettings,
) -> bool:
    """Return whether an incomplete checkpoint can be resumed."""
    if _run_has_summary(path):
        return False
    starts = [event for event in _jsonl_events(path) if event.get("event") == "run_start"]
    if not starts:
        return False
    expected = _run_start_payload(path, _summary_path_for_jsonl(path), region_name, spacing, shift, limit, settings)
    ignored = {"jsonl_path", "summary_path"}
    return all(starts[-1].get(key) == value for key, value in expected.items() if key not in ignored)


def _latest_incomplete_checkpoint(
    region_name: str,
    spacing: mp.mpf,
    shift: str,
    limit: int | None,
    settings: MaxVolumeSettings,
) -> Path | None:
    """Return newest compatible incomplete checkpoint."""
    candidates = sorted(OUTPUT_DIR.glob(f"*-seed{RANDOM_SEED}-{OUTPUT_SUFFIX}.jsonl"), reverse=True)
    return next((path for path in candidates if _checkpoint_is_compatible(path, region_name, spacing, shift, limit, settings)), None)


def _resume_or_new_paths(
    region_name: str,
    spacing: mp.mpf,
    shift: str,
    limit: int | None,
    settings: MaxVolumeSettings,
    resume: bool,
) -> tuple[Path, Path, bool]:
    """Return output paths, resuming compatible work if requested."""
    if resume:
        checkpoint = _latest_incomplete_checkpoint(region_name, spacing, shift, limit, settings)
        if checkpoint is not None:
            return checkpoint, _summary_path_for_jsonl(checkpoint), True
    jsonl_path, summary_path = _new_output_paths()
    return jsonl_path, summary_path, False


def _payload_success(payload: dict) -> bool:
    """Return whether one payload found both max-volume events."""
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
    """Return residual norm for one payload."""
    value = payload["result"]["residual_norm"]
    if value is None:
        return mp.inf
    return mp.mpf(value)


def _compact_payload(payload: dict) -> dict:
    """Return compact summary data for one scout."""
    return {
        "seed_index": payload["seed_index"],
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
    """Return nearest-neighbor offsets for a grid dimension."""
    return [offset for offset in product((-1, 0, 1), repeat=dim) if any(offset)]


def _local_minima(payloads: list[dict], axis_counts: list[int]) -> list[dict]:
    """Return nearest-neighbor local minima among successful scouts."""
    by_index = {tuple(payload["grid_index"]): payload for payload in payloads}
    minima = []
    offsets = _neighbor_offsets(len(axis_counts))
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
            neighbor = by_index.get(neighbor_key)
            if neighbor is not None and _payload_success(neighbor) and _payload_norm(neighbor) < norm:
                is_minimum = False
                break
        if is_minimum:
            minima.append(payload)
    return sorted(minima, key=lambda item: (_payload_norm(item), int(item["seed_index"])))


def _summary_payload(jsonl_path: Path, metadata: dict, best_limit: int = 30) -> dict:
    """Return final scout summary."""
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
    seeds: list[BergerMaxVolumeSeed],
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


def _print_dry_run(region_name: str, spacing: mp.mpf, shift: str, limit: int | None, settings: MaxVolumeSettings) -> None:
    """Print grid size without writing output."""
    metadata = scout_grid_metadata(region_name, spacing, shift, limit, settings)
    print(f"region: {region_name}", flush=True)
    print(f"shift: {shift}", flush=True)
    print(f"max spacing: {mp.nstr(spacing, 12)}", flush=True)
    print(f"axis counts: {metadata['axis_counts']}", flush=True)
    print(f"scout points: {metadata['seed_count']} (full {metadata['full_seed_count']})", flush=True)
    print(f"settings: order={settings.config.series_order}, dps={settings.config.working_dps}", flush=True)
    print("serial estimate at 0.15-0.5s/point: "
          f"{int(metadata['seed_count']) * 0.15 / 3600:.2f}-{int(metadata['seed_count']) * 0.5 / 3600:.2f}h", flush=True)


def main(argv: list[str] | None = None) -> None:
    """Run or resume a Berger maximal-volume scout grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=tuple(sorted(grid_search.GRID_REGIONS)), default="near")
    parser.add_argument("--shift", choices=grid_search.GRID_SHIFTS, default=grid_search.DEFAULT_GRID_SHIFT)
    parser.add_argument("--spacing", type=_positive_mpf, default=grid_search.DEFAULT_GRID_SPACING)
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
        _print_dry_run(args.region, args.spacing, args.shift, args.limit, settings)
        return
    jsonl_path, summary_path, resumed = _resume_or_new_paths(args.region, args.spacing, args.shift, args.limit, settings, not args.no_resume)
    metadata = _run_start_payload(jsonl_path, summary_path, args.region, args.spacing, args.shift, args.limit, settings)
    if resumed:
        print(f"resuming JSONL checkpoint {jsonl_path}", flush=True)
    else:
        print(f"writing JSONL events to {jsonl_path}", flush=True)
        _write_jsonl_event(jsonl_path, _event("run_start", metadata))
    seeds = scout_seeds(args.region, args.spacing, args.shift, args.limit)
    print(f"Berger max-volume scout seeds: {len(seeds)}", flush=True)
    _run_scouts(seeds, jsonl_path, args.workers, settings, args.progress_every, args.chunksize)
    summary = _summary_payload(jsonl_path, metadata)
    _write_jsonl_event(jsonl_path, _event("run_summary", summary))
    _write_summary(summary_path, summary)
    print(f"classification counts: {summary['classification_counts']}", flush=True)
    if summary["best_scouts"]:
        best = summary["best_scouts"][0]
        print(f"best scout: seed={best['seed_index']} residual={best['residual_norm']}", flush=True)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
