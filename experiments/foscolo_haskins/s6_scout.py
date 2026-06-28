"""Grid scouts for the Foscolo-Haskins S6 matching benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable, Iterator

from ..shared.non_mirrored_common import RANDOM_SEED, _event, _write_jsonl_event, _write_summary
from . import s6_common as maxvol
from . import s6_terminal_shooting as terminal


SCOUT_VERSION = "fh-s6-scout-v1"
MAX_VOLUME_OUTPUT_DIR = Path("output/fh_s6_max_volume_scouts")
TERMINAL_OUTPUT_DIR = Path("output/fh_s6_terminal_scouts")
DEFAULT_MIN_PARAMETER = 0.25
DEFAULT_MAX_PARAMETER = 2.4
DEFAULT_MIN_MATCH_TIME = 0.35
DEFAULT_MAX_MATCH_TIME = 1.65
DEFAULT_MAX_VOLUME_SPACING = 0.004
DEFAULT_TERMINAL_SPACING = 0.035
DEFAULT_TERMINAL_TRANSFORMS = ("round-terminal", "exotic-terminal")


@dataclass(frozen=True)
class ScoutGrid:
    """One rectangular log-coordinate scout grid."""

    method: str
    log_a_bounds: tuple[float, float]
    log_b_bounds: tuple[float, float]
    spacing: float
    log_match_time_bounds: tuple[float, float] | None = None
    transforms: tuple[str, ...] = ()
    limit: int | None = None


@dataclass(frozen=True)
class ScoutSeed:
    """One deterministic FH scout seed."""

    index: int
    grid_index: tuple[int, ...]
    log_a: float
    log_b: float
    log_match_time: float | None = None
    transform: str | None = None


def _axis_values(low: float, high: float, spacing: float) -> tuple[float, ...]:
    """Return an inclusive evenly spaced axis with max step at most spacing."""
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    if high < low:
        raise ValueError("axis high bound must be >= low bound")
    if high == low:
        return (low,)
    intervals = max(1, math.ceil((high - low) / spacing))
    step = (high - low) / intervals
    return tuple(low + index * step for index in range(intervals + 1))


def scout_axes(grid: ScoutGrid) -> tuple[tuple[float, ...], ...]:
    """Return log-coordinate axes for one grid."""
    axes = [_axis_values(*grid.log_a_bounds, grid.spacing), _axis_values(*grid.log_b_bounds, grid.spacing)]
    if grid.method == "terminal":
        if grid.log_match_time_bounds is None:
            raise ValueError("terminal scout needs match-time bounds")
        axes.append(_axis_values(*grid.log_match_time_bounds, grid.spacing))
    return tuple(axes)


def scout_seed_count(grid: ScoutGrid) -> int:
    """Return the number of generated scout seeds after any limit."""
    count = 1
    for axis in scout_axes(grid):
        count *= len(axis)
    if grid.method == "terminal":
        count *= len(grid.transforms)
    return min(count, grid.limit) if grid.limit is not None else count


def full_seed_count(grid: ScoutGrid) -> int:
    """Return the full unbounded seed count."""
    count = 1
    for axis in scout_axes(grid):
        count *= len(axis)
    if grid.method == "terminal":
        count *= len(grid.transforms)
    return count


def scout_grid_metadata(grid: ScoutGrid) -> dict:
    """Return JSON-ready scout grid metadata."""
    axes = scout_axes(grid)
    payload = {
        "method": grid.method,
        "spacing": grid.spacing,
        "log_a_bounds": list(grid.log_a_bounds),
        "log_b_bounds": list(grid.log_b_bounds),
        "a_bounds": [math.exp(value) for value in grid.log_a_bounds],
        "b_bounds": [math.exp(value) for value in grid.log_b_bounds],
        "axis_counts": [len(axis) for axis in axes],
        "seed_count": scout_seed_count(grid),
        "full_seed_count": full_seed_count(grid),
        "limit": grid.limit,
    }
    if grid.method == "terminal":
        assert grid.log_match_time_bounds is not None
        payload["log_match_time_bounds"] = list(grid.log_match_time_bounds)
        payload["match_time_bounds"] = [math.exp(value) for value in grid.log_match_time_bounds]
        payload["transforms"] = list(grid.transforms)
    return payload


def scout_seeds(grid: ScoutGrid) -> Iterator[ScoutSeed]:
    """Yield deterministic seeds for one scout grid."""
    axes = scout_axes(grid)
    emitted = 0
    if grid.method == "max-volume":
        for i, log_a in enumerate(axes[0]):
            for j, log_b in enumerate(axes[1]):
                index = i * len(axes[1]) + j
                if grid.limit is not None and emitted >= grid.limit:
                    return
                emitted += 1
                yield ScoutSeed(index, (i, j), log_a, log_b)
        return
    if grid.method == "terminal":
        transforms = grid.transforms
        for i, log_a in enumerate(axes[0]):
            for j, log_b in enumerate(axes[1]):
                for k, log_h in enumerate(axes[2]):
                    for transform_index, transform in enumerate(transforms):
                        index = (((i * len(axes[1]) + j) * len(axes[2]) + k) * len(transforms)) + transform_index
                        if grid.limit is not None and emitted >= grid.limit:
                            return
                        emitted += 1
                        yield ScoutSeed(index, (i, j, k, transform_index), log_a, log_b, log_h, transform)
        return
    raise ValueError(f"unknown scout method {grid.method!r}")


def _max_volume_payload(seed: ScoutSeed, settings: maxvol.FHMarchSettings) -> dict:
    """Evaluate one max-volume scout seed."""
    a = math.exp(seed.log_a)
    b = math.exp(seed.log_b)
    evaluation = maxvol.evaluate_match(a, b, None, settings)
    return {
        "method": "max-volume",
        "seed_index": seed.index,
        "grid_index": list(seed.grid_index),
        "log_coordinates": {"a": seed.log_a, "b": seed.log_b},
        "parameters": {"a": a, "b": b},
        "status": evaluation.status,
        "reflection": list(evaluation.reflection),
        "residual": list(evaluation.residual),
        "residual_norm": evaluation.residual_norm,
        "s2_status": evaluation.s2_orbit.status,
        "s3_status": evaluation.s3_orbit.status,
        "s2_volume": evaluation.s2_orbit.volume,
        "s3_volume": evaluation.s3_orbit.volume,
    }


def _terminal_payload(seed: ScoutSeed, settings: terminal.TerminalShootingSettings) -> dict:
    """Evaluate one terminal-shooting scout seed."""
    if seed.log_match_time is None or seed.transform is None:
        raise ValueError("terminal seed missing match-time or transform")
    a = math.exp(seed.log_a)
    b = math.exp(seed.log_b)
    match_time = math.exp(seed.log_match_time)
    evaluation = terminal.evaluate_terminal_shooting(a, b, match_time, seed.transform, settings)
    return {
        "method": "terminal",
        "seed_index": seed.index,
        "grid_index": list(seed.grid_index),
        "log_coordinates": {"a": seed.log_a, "b": seed.log_b, "match_time": seed.log_match_time},
        "parameters": {"a": a, "b": b, "match_time": match_time},
        "transform": seed.transform,
        "status": evaluation.status,
        "residual_norm": evaluation.residual_norm,
        "left_branch_valid": evaluation.diagnostics.get("left_branch_valid"),
        "right_branch_valid": evaluation.diagnostics.get("right_branch_valid"),
        "left_volume": evaluation.diagnostics.get("left_volume"),
        "right_volume": evaluation.diagnostics.get("right_volume"),
    }


def _evaluate_seed(args: tuple[str, ScoutSeed, object]) -> dict:
    """Evaluate one scout seed in a worker process."""
    method, seed, settings = args
    if method == "max-volume":
        return _max_volume_payload(seed, settings)  # type: ignore[arg-type]
    if method == "terminal":
        return _terminal_payload(seed, settings)  # type: ignore[arg-type]
    raise ValueError(f"unknown scout method {method!r}")


def _evaluate_seed_payloads(method: str, seeds: Iterable[ScoutSeed], settings: object, workers: int, chunksize: int | None) -> Iterable[dict]:
    """Yield scout payloads, optionally in parallel."""
    items = ((method, seed, settings) for seed in seeds)
    if workers <= 1:
        for item in items:
            yield _evaluate_seed(item)
        return
    actual_chunksize = chunksize or 8
    try:
        executor = ProcessPoolExecutor(max_workers=workers)
    except (OSError, PermissionError):
        print("process workers unavailable; falling back to threads", flush=True)
        executor = ThreadPoolExecutor(max_workers=workers)
    with executor:
        yield from executor.map(_evaluate_seed, items, chunksize=actual_chunksize)


def _jsonl_events(path: Path) -> Iterator[dict]:
    """Yield JSONL events from a checkpoint."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _completed_seed_indices(path: Path) -> set[int]:
    """Return completed scout seed indices in one checkpoint."""
    return {int(event["seed_index"]) for event in _jsonl_events(path) if event.get("event") == "scout_result"}


def _scout_payloads(path: Path) -> list[dict]:
    """Return scout_result payloads from one checkpoint."""
    return [event for event in _jsonl_events(path) if event.get("event") == "scout_result"]


def _run_has_summary(path: Path) -> bool:
    """Return whether one JSONL checkpoint has a final summary event."""
    return any(event.get("event") == "run_summary" for event in _jsonl_events(path))


def _output_paths(output_dir: Path, method: str, now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped output paths for one scout run."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S-%f")
    base = f"{stamp}-seed{RANDOM_SEED}-{method}-{SCOUT_VERSION}"
    return output_dir / f"{base}.jsonl", output_dir / f"{base}-summary.json"


def _summary_path_for_jsonl(path: Path) -> Path:
    """Return the summary path paired with one JSONL checkpoint."""
    return path.with_name(f"{path.stem}-summary.json")


def _run_start_payload(jsonl_path: Path, summary_path: Path, grid: ScoutGrid, settings: object) -> dict:
    """Return JSON-ready run metadata."""
    return {
        "scout_version": SCOUT_VERSION,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "grid": scout_grid_metadata(grid),
        "settings": settings.__dict__,
    }


def _compatible_checkpoint(path: Path, grid: ScoutGrid, settings: object) -> bool:
    """Return whether one incomplete checkpoint can be resumed."""
    if _run_has_summary(path):
        return False
    try:
        first = next(_jsonl_events(path))
    except StopIteration:
        return False
    expected = _run_start_payload(path, _summary_path_for_jsonl(path), grid, settings)
    return first.get("event") == "run_start" and first.get("scout_version") == expected["scout_version"] and first.get("grid") == expected["grid"] and first.get("settings") == expected["settings"]


def _latest_incomplete_checkpoint(output_dir: Path, method: str, grid: ScoutGrid, settings: object) -> Path | None:
    """Return the newest compatible incomplete checkpoint, if present."""
    candidates = sorted(output_dir.glob(f"*-{method}-{SCOUT_VERSION}.jsonl"), reverse=True)
    return next((path for path in candidates if _compatible_checkpoint(path, grid, settings)), None)


def _resume_or_new_paths(output_dir: Path, method: str, grid: ScoutGrid, settings: object, resume: bool) -> tuple[Path, Path, bool]:
    """Return output paths, resuming compatible incomplete work when requested."""
    if resume:
        checkpoint = _latest_incomplete_checkpoint(output_dir, method, grid, settings)
        if checkpoint is not None:
            return checkpoint, _summary_path_for_jsonl(checkpoint), True
    jsonl_path, summary_path = _output_paths(output_dir, method)
    return jsonl_path, summary_path, False


def _pending_seeds(grid: ScoutGrid, completed: set[int]) -> Iterator[ScoutSeed]:
    """Yield seeds not already completed."""
    for seed in scout_seeds(grid):
        if seed.index not in completed:
            yield seed


def _run_scouts(
    method: str,
    grid: ScoutGrid,
    settings: object,
    jsonl_path: Path,
    workers: int,
    progress_every: int,
    chunksize: int | None,
) -> None:
    """Evaluate missing scout seeds and append results to JSONL."""
    completed = _completed_seed_indices(jsonl_path)
    total = scout_seed_count(grid)
    pending_count = total - len(completed)
    print(f"loaded completed scouts: {len(completed)}/{total}", flush=True)
    print(f"pending scouts: {pending_count} with workers={workers}", flush=True)
    done = len(completed)
    for payload in _evaluate_seed_payloads(method, _pending_seeds(grid, completed), settings, workers, chunksize):
        _write_jsonl_event(jsonl_path, _event("scout_result", payload))
        done += 1
        if progress_every and (done % progress_every == 0 or done == total):
            print(f"scouts complete: {done}/{total}", flush=True)


def _finite_norm(payload: dict) -> float:
    """Return one payload residual norm, with failures sorted last."""
    if payload.get("status") != "ok":
        return math.inf
    value = float(payload.get("residual_norm", math.inf))
    return value if math.isfinite(value) else math.inf


def _compact_payload(payload: dict) -> dict:
    """Return a compact scout payload for summaries."""
    output = {
        "seed_index": payload["seed_index"],
        "grid_index": payload["grid_index"],
        "parameters": payload["parameters"],
        "status": payload["status"],
        "residual_norm": payload["residual_norm"],
    }
    if payload["method"] == "max-volume":
        output["reflection"] = payload["reflection"]
    if payload["method"] == "terminal":
        output["transform"] = payload["transform"]
    return output


def _local_minima(payloads: list[dict], grid: ScoutGrid) -> list[dict]:
    """Return nearest-neighbor local minima from completed payloads."""
    axes = scout_axes(grid)
    by_index = {tuple(payload["grid_index"]): payload for payload in payloads}
    minima: list[dict] = []
    spatial_dims = 2 if grid.method == "max-volume" else 3
    neighbor_offsets = list(product((-1, 0, 1), repeat=spatial_dims))
    for payload in payloads:
        key = tuple(payload["grid_index"])
        norm = _finite_norm(payload)
        if norm == math.inf:
            continue
        spatial_key = key[:spatial_dims]
        tail = key[spatial_dims:]
        is_minimum = True
        for offset in neighbor_offsets:
            if all(value == 0 for value in offset):
                continue
            neighbor_spatial = tuple(index + delta for index, delta in zip(spatial_key, offset))
            if any(index < 0 or index >= len(axes[axis]) for axis, index in enumerate(neighbor_spatial)):
                continue
            neighbor_key = neighbor_spatial + tail
            neighbor = by_index.get(neighbor_key)
            if neighbor is not None and _finite_norm(neighbor) < norm:
                is_minimum = False
                break
        if is_minimum:
            minima.append(payload)
    return sorted(minima, key=lambda item: (_finite_norm(item), int(item["seed_index"])))


def scout_summary_payload(jsonl_path: Path, grid: ScoutGrid, metadata: dict, best_limit: int = 30) -> dict:
    """Return final summary for a scout run."""
    payloads = _scout_payloads(jsonl_path)
    counts = Counter(payload.get("status", "unknown") for payload in payloads)
    successes = [payload for payload in payloads if payload.get("status") == "ok"]
    best = sorted(successes, key=lambda item: (_finite_norm(item), int(item["seed_index"])))[:best_limit]
    minima = _local_minima(successes, grid)[:best_limit]
    return {
        **metadata,
        "scout_count": len(payloads),
        "classification_counts": dict(counts),
        "best_scouts": [_compact_payload(payload) for payload in best],
        "best_local_minima": [_compact_payload(payload) for payload in minima],
    }


def _default_workers() -> int:
    """Return a conservative worker count."""
    return max(1, min(4, (os.cpu_count() or 2) - 1))


def _positive_float(value: str) -> float:
    """Parse a positive CLI float."""
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_int(value: str) -> int:
    """Parse a positive CLI integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    """Parse a nonnegative CLI integer."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _parse_transform_list(value: str) -> tuple[str, ...]:
    """Parse comma-separated terminal transform names."""
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise argparse.ArgumentTypeError("at least one transform is required")
    for name in names:
        if name not in terminal.TRANSFORMS:
            raise argparse.ArgumentTypeError(f"unknown transform {name!r}")
    return names


def _grid_from_common_args(args: argparse.Namespace, method: str) -> ScoutGrid:
    """Build a ScoutGrid from parsed CLI args."""
    return ScoutGrid(
        method=method,
        log_a_bounds=(math.log(args.a_min), math.log(args.a_max)),
        log_b_bounds=(math.log(args.b_min), math.log(args.b_max)),
        spacing=args.spacing,
        log_match_time_bounds=None if method == "max-volume" else (math.log(args.match_time_min), math.log(args.match_time_max)),
        transforms=() if method == "max-volume" else args.transforms,
        limit=args.limit,
    )


def _print_dry_run(grid: ScoutGrid) -> None:
    """Print grid size and rough runtime estimates."""
    metadata = scout_grid_metadata(grid)
    print(f"method: {grid.method}", flush=True)
    print(f"a bounds: {metadata['a_bounds']}", flush=True)
    print(f"b bounds: {metadata['b_bounds']}", flush=True)
    if grid.method == "terminal":
        print(f"match-time bounds: {metadata['match_time_bounds']}", flush=True)
        print(f"transforms: {metadata['transforms']}", flush=True)
    print(f"log spacing: {grid.spacing}", flush=True)
    print(f"axis counts: {metadata['axis_counts']}", flush=True)
    print(f"scout points: {metadata['seed_count']} (full {metadata['full_seed_count']})", flush=True)
    per_point = (0.01, 0.04) if grid.method == "max-volume" else (0.02, 0.08)
    count = int(metadata["seed_count"])
    print(f"serial estimate at {per_point[0]:.2f}-{per_point[1]:.2f}s/point: {count * per_point[0] / 3600:.2f}-{count * per_point[1] / 3600:.2f}h", flush=True)


def _add_common_args(parser: argparse.ArgumentParser, default_spacing: float) -> None:
    """Add common scout CLI options."""
    parser.add_argument("--workers", type=_positive_int, default=_default_workers(), help="parallel worker processes")
    parser.add_argument("--spacing", type=_positive_float, default=default_spacing, help="maximum log-coordinate grid spacing")
    parser.add_argument("--a-min", type=_positive_float, default=DEFAULT_MIN_PARAMETER, help="minimum a")
    parser.add_argument("--a-max", type=_positive_float, default=DEFAULT_MAX_PARAMETER, help="maximum a")
    parser.add_argument("--b-min", type=_positive_float, default=DEFAULT_MIN_PARAMETER, help="minimum b")
    parser.add_argument("--b-max", type=_positive_float, default=DEFAULT_MAX_PARAMETER, help="maximum b")
    parser.add_argument("--limit", type=_nonnegative_int, default=None, help="debug cap on generated seeds")
    parser.add_argument("--dry-run", action="store_true", help="print grid size without running")
    parser.add_argument("--no-resume", action="store_true", help="start a fresh checkpoint")
    parser.add_argument("--progress-every", type=_positive_int, default=1000, help="print progress every N completed scouts")
    parser.add_argument("--chunksize", type=_positive_int, default=None, help="multiprocessing chunksize")


def main_max_volume(argv: list[str] | None = None) -> None:
    """Run a max-volume matching scout."""
    parser = argparse.ArgumentParser(description="FH S6 max-volume matching scout grid.")
    _add_common_args(parser, DEFAULT_MAX_VOLUME_SPACING)
    parser.add_argument("--step-size", type=_positive_float, default=maxvol.FHMarchSettings.step_size, help="RK4 step size")
    parser.add_argument("--s2-epsilon", type=_positive_float, default=maxvol.FHMarchSettings.s2_epsilon, help="S2 Taylor seed epsilon")
    parser.add_argument("--s3-epsilon", type=_positive_float, default=maxvol.FHMarchSettings.s3_epsilon, help="S3 Taylor seed epsilon")
    parser.add_argument("--max-time", type=_positive_float, default=maxvol.FHMarchSettings.max_time, help="max march time")
    args = parser.parse_args(argv)
    grid = _grid_from_common_args(args, "max-volume")
    settings = maxvol.FHMarchSettings(args.s2_epsilon, args.s3_epsilon, args.step_size, args.max_time)
    if args.dry_run:
        _print_dry_run(grid)
        return
    _run_command("max-volume", grid, settings, MAX_VOLUME_OUTPUT_DIR, args.workers, args.progress_every, args.chunksize, not args.no_resume)


def main_terminal(argv: list[str] | None = None) -> None:
    """Run a terminal shooting scout."""
    parser = argparse.ArgumentParser(description="FH S6 naive terminal shooting scout grid.")
    _add_common_args(parser, DEFAULT_TERMINAL_SPACING)
    parser.add_argument("--match-time-min", type=_positive_float, default=DEFAULT_MIN_MATCH_TIME, help="minimum interior match time")
    parser.add_argument("--match-time-max", type=_positive_float, default=DEFAULT_MAX_MATCH_TIME, help="maximum interior match time")
    parser.add_argument("--transforms", type=_parse_transform_list, default=DEFAULT_TERMINAL_TRANSFORMS, help="comma-separated terminal transforms")
    parser.add_argument("--step-size", type=_positive_float, default=terminal.TerminalShootingSettings.step_size, help="RK4 step size")
    parser.add_argument("--s2-epsilon", type=_positive_float, default=terminal.TerminalShootingSettings.s2_epsilon, help="S2 Taylor seed epsilon")
    parser.add_argument("--s3-epsilon", type=_positive_float, default=terminal.TerminalShootingSettings.s3_epsilon, help="S3 Taylor seed epsilon")
    args = parser.parse_args(argv)
    grid = _grid_from_common_args(args, "terminal")
    settings = terminal.TerminalShootingSettings(args.s2_epsilon, args.s3_epsilon, args.step_size)
    if args.dry_run:
        _print_dry_run(grid)
        return
    _run_command("terminal", grid, settings, TERMINAL_OUTPUT_DIR, args.workers, args.progress_every, args.chunksize, not args.no_resume)


def _run_command(
    method: str,
    grid: ScoutGrid,
    settings: object,
    output_dir: Path,
    workers: int,
    progress_every: int,
    chunksize: int | None,
    resume: bool,
) -> None:
    """Run or resume one scout command."""
    jsonl_path, summary_path, resumed = _resume_or_new_paths(output_dir, method, grid, settings, resume)
    metadata = _run_start_payload(jsonl_path, summary_path, grid, settings)
    if resumed:
        print(f"resuming JSONL checkpoint {jsonl_path}", flush=True)
    else:
        print(f"writing JSONL events to {jsonl_path}", flush=True)
        _write_jsonl_event(jsonl_path, _event("run_start", metadata))
    print(f"FH S6 {method} scout seeds: {scout_seed_count(grid)}", flush=True)
    _run_scouts(method, grid, settings, jsonl_path, workers, progress_every, chunksize)
    summary = scout_summary_payload(jsonl_path, grid, metadata)
    _write_jsonl_event(jsonl_path, _event("run_summary", summary))
    _write_summary(summary_path, summary)
    print(f"classification counts: {summary['classification_counts']}", flush=True)
    if summary["best_scouts"]:
        best = summary["best_scouts"][0]
        print(f"best scout: seed={best['seed_index']} residual={best['residual_norm']}", flush=True)
    print(f"summary written to {summary_path}", flush=True)
