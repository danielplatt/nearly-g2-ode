"""Endpoint-reduced Aloff-Wallach N_{1,1} maximal-volume scout runner."""

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

from experiments.shared.non_mirrored_common import RANDOM_SEED, _event, _mp_string, _output_paths, _write_jsonl_event, _write_summary

from . import endpoint_smoothness
from .evolution import AWSettings, EVOLUTION_VERSION, EndpointConstants, MatchResult, max_volume_match


SCOUT_VERSION = "aloff-wallach-n11-scout-v1"
OUTPUT_DIR = Path("output/aloff_wallach_scouts")
OUTPUT_SUFFIX = SCOUT_VERSION
COORDINATE_NAMES = ("left_A", "left_B", "left_C", "left_D", "right_A", "right_B", "right_C", "right_D")
SEED_ORDER = "bc-branch-center-out-v1"
DEFAULT_RADIUS = mp.mpf("1.0")
DEFAULT_SPACING = mp.mpf("1.0")


@dataclass(frozen=True)
class AWScoutPoint:
    """Eight endpoint-reduced smooth constants."""

    left_A: mp.mpf
    left_B: mp.mpf
    left_C: mp.mpf
    left_D: mp.mpf
    right_A: mp.mpf
    right_B: mp.mpf
    right_C: mp.mpf
    right_D: mp.mpf


@dataclass(frozen=True)
class AWScoutSeed:
    """One deterministic Aloff-Wallach scout seed."""

    index: int
    source: str
    grid_index: tuple[int, ...]
    point: AWScoutPoint


def _coordinates(point: AWScoutPoint) -> tuple[mp.mpf, ...]:
    return (
        point.left_A,
        point.left_B,
        point.left_C,
        point.left_D,
        point.right_A,
        point.right_B,
        point.right_C,
        point.right_D,
    )


def _point_from_values(values) -> AWScoutPoint:
    parsed = tuple(mp.mpf(value) for value in values)
    if len(parsed) != 8:
        raise ValueError("Aloff-Wallach scout points need exactly eight coordinates.")
    return AWScoutPoint(*parsed)


def _axis(radius: mp.mpf, spacing: mp.mpf, shift: str) -> tuple[mp.mpf, ...]:
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    count = int(mp.floor((2 * radius) / spacing)) + 1
    start = -radius
    values = [start + index * spacing for index in range(count)]
    if values[-1] < radius:
        values.append(radius)
    if shift == "vertex":
        return tuple(mp.mpf(value) for value in values)
    if shift == "cell-center":
        shifted = [value + spacing / 2 for value in values[:-1]]
        return tuple(mp.mpf(value) for value in shifted if -radius <= value <= radius)
    raise ValueError(f"unknown grid shift {shift!r}")


def scout_axes(radius: mp.mpf = DEFAULT_RADIUS, spacing: mp.mpf = DEFAULT_SPACING, shift: str = "vertex") -> tuple[tuple[mp.mpf, ...], ...]:
    """Return the eight endpoint-reduced coordinate axes."""
    axis = _axis(radius, spacing, shift)
    return tuple(axis for _ in COORDINATE_NAMES)


def scout_seed_count(radius: mp.mpf = DEFAULT_RADIUS, spacing: mp.mpf = DEFAULT_SPACING, shift: str = "vertex", limit: int | None = None) -> int:
    """Return the deterministic grid size."""
    count = 1
    for axis in scout_axes(radius, spacing, shift):
        count *= len(axis)
    return min(count, limit) if limit is not None else count


def scout_seeds(
    radius: mp.mpf = DEFAULT_RADIUS,
    spacing: mp.mpf = DEFAULT_SPACING,
    shift: str = "vertex",
    limit: int | None = None,
) -> list[AWScoutSeed]:
    """Return deterministic endpoint-reduced scout seeds."""
    axes = scout_axes(radius, spacing, shift)
    index_axes = tuple(range(len(axis)) for axis in axes)
    grid_indices = sorted(
        product(*index_axes),
        key=lambda grid_index: _seed_sort_key(grid_index, axes),
    )
    seeds = []
    for grid_index in grid_indices:
        if limit is not None and len(seeds) >= limit:
            break
        values = tuple(axis[axis_index] for axis, axis_index in zip(axes, grid_index))
        seeds.append(AWScoutSeed(len(seeds), "endpoint_reduced_grid", grid_index, _point_from_values(values)))
    return seeds


def _endpoint_constants(point: AWScoutPoint) -> tuple[EndpointConstants, EndpointConstants]:
    left = EndpointConstants(float(point.left_A), float(point.left_B), float(point.left_C), float(point.left_D))
    right = EndpointConstants(float(point.right_A), float(point.right_B), float(point.right_C), float(point.right_D))
    return left, right


def _left_branch_score(constants: EndpointConstants) -> tuple[int, float]:
    """Prefer left endpoint branches with opposite nonzero B,C signs."""
    if constants.B * constants.C < 0:
        return (0, abs(constants.A) + abs(constants.D))
    if constants.B != 0 or constants.C != 0:
        return (1, abs(constants.A) + abs(constants.D))
    return (2, abs(constants.A) + abs(constants.D))


def _right_branch_score(constants: EndpointConstants) -> tuple[int, int, float]:
    """Prefer right endpoint branches with at least one active B/C direction."""
    active = int(constants.B != 0) + int(constants.C != 0)
    if active:
        return (0, -active, abs(constants.A) + abs(constants.D))
    return (1, 0, abs(constants.A) + abs(constants.D))


def _seed_sort_key(grid_index: tuple[int, ...], axes: tuple[tuple[mp.mpf, ...], ...]) -> tuple:
    """Prefer observed nondegenerate endpoint branches, then center-out order."""
    values = tuple(axis[axis_index] for axis, axis_index in zip(axes, grid_index))
    point = _point_from_values(values)
    left, right = _endpoint_constants(point)
    return (
        _left_branch_score(left),
        _right_branch_score(right),
        max(abs(value) for value in values),
        sum(abs(value) for value in values),
        grid_index,
    )


def _settings_payload(settings: AWSettings) -> dict:
    """Return JSON-ready numerical settings."""
    return {
        "lambda": settings.lam,
        "structure_scale": settings.structure_scale,
        "base_structure_scale": settings.base_structure_scale,
        "fiber_structure_scale": settings.fiber_structure_scale,
        "endpoint_order": settings.endpoint_order,
        "germ_epsilon": settings.germ_epsilon,
        "germ_samples": list(settings.germ_samples),
        "max_tau": settings.max_tau,
        "max_step": settings.max_step,
        "rtol": settings.rtol,
        "atol": settings.atol,
        "max_germ_evaluations": settings.max_germ_evaluations,
    }


def _point_payload(point: AWScoutPoint) -> dict:
    """Return JSON-ready scout coordinates."""
    return {name: _mp_string(value) for name, value in zip(COORDINATE_NAMES, _coordinates(point))}


def _germ_payload(germ) -> dict:
    return {
        "normal_weight": germ.normal_weight,
        "constants": {
            "A": germ.constants.A,
            "B": germ.constants.B,
            "C": germ.constants.C,
            "D": germ.constants.D,
        },
        "residual_norm": germ.residual_norm,
        "success": germ.success,
        "message": germ.message,
    }


def _side_payload(side) -> dict:
    return {
        "status": side.status,
        "tau": side.tau,
        "volume": side.volume,
        "volume_dot": side.volume_dot,
        "message": side.message,
        "germ": _germ_payload(side.germ),
    }


def _reconstructed_interval(match: MatchResult) -> float | None:
    """Return the reconstructed interval from the two max-volume events."""
    value = getattr(match, "reconstructed_interval", None)
    if value is not None:
        return value
    if match.left.status != "max_volume" or match.right.status != "max_volume":
        return None
    if match.left.tau is None or match.right.tau is None:
        return None
    return match.left.tau + match.right.tau


def _match_payload(match: MatchResult) -> dict:
    return {
        "failure": match.failure,
        "residual_norm": match.residual_norm,
        "residual": list(match.residual),
        "reconstructed_interval": _reconstructed_interval(match),
        "left": _side_payload(match.left),
        "right": _side_payload(match.right),
    }


def _evaluate_seed_payload(seed: AWScoutSeed, settings: AWSettings) -> dict:
    """Evaluate one scout seed."""
    left, right = _endpoint_constants(seed.point)
    match = max_volume_match(left, right, settings)
    return {
        "seed_index": seed.index,
        "source": seed.source,
        "grid_index": list(seed.grid_index),
        "seed_point": _point_payload(seed.point),
        "endpoint_constants": {
            "left": {"A": left.A, "B": left.B, "C": left.C, "D": left.D},
            "right": {"A": right.A, "B": right.B, "C": right.C, "D": right.D},
        },
        "result": _match_payload(match),
    }


def _evaluate_seed_payload_star(args) -> dict:
    seed, settings = args
    return _evaluate_seed_payload(seed, settings)


def _evaluate_seed_payloads(
    seeds: list[AWScoutSeed],
    workers: int,
    settings: AWSettings,
    chunksize: int | None = None,
) -> Iterable[dict]:
    """Yield JSON-ready payloads in stable order."""
    if workers <= 1:
        for seed in seeds:
            yield _evaluate_seed_payload(seed, settings)
        return
    actual_chunksize = chunksize or 1
    try:
        executor = ProcessPoolExecutor(max_workers=workers)
    except (OSError, PermissionError):
        print("process workers unavailable; falling back to threads", flush=True)
        executor = ThreadPoolExecutor(max_workers=workers)
    with executor:
        yield from executor.map(_evaluate_seed_payload_star, [(seed, settings) for seed in seeds], chunksize=actual_chunksize)


def _jsonl_events(path: Path):
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
    return {int(event["seed_index"]) for event in _jsonl_events(path) if event.get("event") == "scout_result"}


def _scout_payloads(path: Path) -> list[dict]:
    return [event for event in _jsonl_events(path) if event.get("event") == "scout_result"]


def _run_has_summary(path: Path) -> bool:
    return any(event.get("event") == "run_summary" for event in _jsonl_events(path))


def _summary_path_for_jsonl(path: Path) -> Path:
    return path.with_name(f"{path.stem}-summary.json")


def _new_output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    return _output_paths(OUTPUT_DIR, OUTPUT_SUFFIX, now)


def _grid_metadata(radius: mp.mpf, spacing: mp.mpf, shift: str, limit: int | None) -> dict:
    axes = scout_axes(radius, spacing, shift)
    full_count = scout_seed_count(radius, spacing, shift)
    return {
        "parameterization": "4 smooth endpoint constants at RP2 and 4 at CP1; higher endpoint layers fitted internally",
        "coordinate_names": list(COORDINATE_NAMES),
        "radius": _mp_string(radius),
        "spacing": _mp_string(spacing),
        "shift": shift,
        "seed_order": SEED_ORDER,
        "axis_counts": [len(axis) for axis in axes],
        "bounds": [[_mp_string(axis[0]), _mp_string(axis[-1])] for axis in axes],
        "full_seed_count": full_count,
        "seed_count": min(full_count, limit) if limit is not None else full_count,
        "limit": limit,
    }


def _run_start_payload(
    jsonl_path: Path,
    summary_path: Path,
    radius: mp.mpf,
    spacing: mp.mpf,
    shift: str,
    limit: int | None,
    settings: AWSettings,
) -> dict:
    return {
        "random_seed": RANDOM_SEED,
        "scout_version": SCOUT_VERSION,
        "evolution_version": EVOLUTION_VERSION,
        "endpoint_smoothness_version": endpoint_smoothness.ENDPOINT_SMOOTHNESS_VERSION,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "grid": _grid_metadata(radius, spacing, shift, limit),
        "settings": _settings_payload(settings),
    }


def _checkpoint_is_compatible(path: Path, radius: mp.mpf, spacing: mp.mpf, shift: str, limit: int | None, settings: AWSettings) -> bool:
    if _run_has_summary(path):
        return False
    starts = [event for event in _jsonl_events(path) if event.get("event") == "run_start"]
    if not starts:
        return False
    expected = _run_start_payload(path, _summary_path_for_jsonl(path), radius, spacing, shift, limit, settings)
    ignored = {"jsonl_path", "summary_path"}
    return all(starts[-1].get(key) == value for key, value in expected.items() if key not in ignored)


def _latest_incomplete_checkpoint(radius: mp.mpf, spacing: mp.mpf, shift: str, limit: int | None, settings: AWSettings) -> Path | None:
    candidates = sorted(OUTPUT_DIR.glob(f"*-seed{RANDOM_SEED}-{OUTPUT_SUFFIX}.jsonl"), reverse=True)
    return next((path for path in candidates if _checkpoint_is_compatible(path, radius, spacing, shift, limit, settings)), None)


def _resume_or_new_paths(
    radius: mp.mpf,
    spacing: mp.mpf,
    shift: str,
    limit: int | None,
    settings: AWSettings,
    resume: bool,
) -> tuple[Path, Path, bool]:
    if resume:
        checkpoint = _latest_incomplete_checkpoint(radius, spacing, shift, limit, settings)
        if checkpoint is not None:
            return checkpoint, _summary_path_for_jsonl(checkpoint), True
    jsonl_path, summary_path = _new_output_paths()
    return jsonl_path, summary_path, False


def _payload_success(payload: dict) -> bool:
    return payload["result"]["failure"] is None


def _payload_status(payload: dict) -> str:
    result = payload["result"]
    if result["failure"] is None:
        return "ok"
    return str(result["failure"])


def _payload_norm(payload: dict) -> mp.mpf:
    value = payload["result"]["residual_norm"]
    return mp.inf if value is None else mp.mpf(str(value))


def _compact_payload(payload: dict) -> dict:
    return {
        "seed_index": payload["seed_index"],
        "grid_index": payload["grid_index"],
        "seed_point": payload["seed_point"],
        "residual_norm": payload["result"]["residual_norm"],
        "reconstructed_interval": payload["result"].get("reconstructed_interval"),
        "failure": payload["result"]["failure"],
        "left_status": payload["result"]["left"]["status"],
        "right_status": payload["result"]["right"]["status"],
        "left_germ_residual": payload["result"]["left"]["germ"]["residual_norm"],
        "right_germ_residual": payload["result"]["right"]["germ"]["residual_norm"],
    }


def _summary_payload(jsonl_path: Path, metadata: dict, best_limit: int = 30) -> dict:
    payloads = _scout_payloads(jsonl_path)
    counts = Counter(_payload_status(payload) for payload in payloads)
    successes = [payload for payload in payloads if _payload_success(payload)]
    best = sorted(successes, key=_payload_norm)[:best_limit]
    best_germs = sorted(payloads, key=lambda payload: max(
        mp.mpf(str(payload["result"]["left"]["germ"]["residual_norm"])),
        mp.mpf(str(payload["result"]["right"]["germ"]["residual_norm"])),
    ))[:best_limit]
    return {
        **metadata,
        "scout_count": len(payloads),
        "classification_counts": dict(counts),
        "best_scouts": [_compact_payload(payload) for payload in best],
        "best_germ_fits": [_compact_payload(payload) for payload in best_germs],
    }


def _run_scouts(
    seeds: list[AWScoutSeed],
    jsonl_path: Path,
    workers: int,
    settings: AWSettings,
    progress_every: int,
    chunksize: int | None,
) -> None:
    completed = _completed_seed_indices(jsonl_path)
    pending = [seed for seed in seeds if seed.index not in completed]
    if completed:
        print(f"resuming: {len(completed)} completed, {len(pending)} pending", flush=True)
    for completed_count, payload in enumerate(_evaluate_seed_payloads(pending, workers, settings, chunksize), start=len(completed) + 1):
        _write_jsonl_event(jsonl_path, _event("scout_result", payload))
        if progress_every and (completed_count % progress_every == 0 or completed_count == len(seeds)):
            norm = payload["result"]["residual_norm"]
            print(
                f"[{completed_count}/{len(seeds)}] seed {payload['seed_index']} "
                f"status={_payload_status(payload)} residual={norm}",
                flush=True,
            )


def _settings_from_args(args: argparse.Namespace) -> AWSettings:
    return AWSettings(
        lam=float(args.lam),
        structure_scale=None if args.structure_scale is None else float(args.structure_scale),
        base_structure_scale=float(args.base_structure_scale),
        fiber_structure_scale=float(args.fiber_structure_scale),
        endpoint_order=args.endpoint_order,
        germ_epsilon=float(args.germ_epsilon),
        max_tau=float(args.max_tau),
        max_step=float(args.max_step),
        rtol=float(args.rtol),
        atol=float(args.atol),
        max_germ_evaluations=args.max_germ_evaluations,
    )


def main(argv: list[str] | None = None) -> None:
    """Run the endpoint-reduced Aloff-Wallach scout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=mp.mpf, default=DEFAULT_RADIUS, help="coordinate radius for each endpoint constant")
    parser.add_argument("--spacing", type=mp.mpf, default=DEFAULT_SPACING, help="grid spacing for each endpoint constant")
    parser.add_argument("--shift", choices=("vertex", "cell-center"), default="vertex", help="grid shift")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)), help="parallel workers")
    parser.add_argument("--chunksize", type=int, default=None, help="process-pool chunksize")
    parser.add_argument("--limit", type=int, default=None, help="debug limit on evaluated seeds")
    parser.add_argument("--dry-run", action="store_true", help="print grid metadata without evaluating seeds")
    parser.add_argument("--no-resume", action="store_true", help="start a fresh checkpoint even if a compatible incomplete run exists")
    parser.add_argument("--progress-every", type=int, default=10, help="print progress every N completed seeds")
    parser.add_argument("--lam", type=float, default=4.0, help="normalized nearly-parallel lambda")
    parser.add_argument("--structure-scale", type=float, default=None, help="legacy common principal-orbit structure-equation scale")
    parser.add_argument("--base-structure-scale", type=float, default=-1.0, help="base SO(3) Maurer-Cartan scale")
    parser.add_argument("--fiber-structure-scale", type=float, default=-2.0, help="fiber SO(3) Maurer-Cartan scale")
    parser.add_argument("--endpoint-order", type=int, default=2, help="endpoint Taylor order fitted internally")
    parser.add_argument("--germ-epsilon", type=float, default=1e-3, help="local time where fitted germs seed marching")
    parser.add_argument("--max-tau", type=float, default=2.0, help="maximum one-sided march time")
    parser.add_argument("--max-step", type=float, default=0.02, help="maximum solve_ivp step")
    parser.add_argument("--rtol", type=float, default=1e-7, help="solve_ivp relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-9, help="solve_ivp absolute tolerance")
    parser.add_argument("--max-germ-evaluations", type=int, default=250, help="least-squares evaluations per endpoint germ")
    args = parser.parse_args(argv)

    settings = _settings_from_args(args)
    seeds = scout_seeds(args.radius, args.spacing, args.shift, args.limit)
    metadata_grid = _grid_metadata(args.radius, args.spacing, args.shift, args.limit)
    if args.dry_run:
        print("Aloff-Wallach N_{1,1} endpoint-reduced scout dry run", flush=True)
        print(f"version: {SCOUT_VERSION}", flush=True)
        print(f"parameterization: {metadata_grid['parameterization']}", flush=True)
        print(f"coordinates: {', '.join(COORDINATE_NAMES)}", flush=True)
        print(f"axis counts: {metadata_grid['axis_counts']}", flush=True)
        print(f"seed count: {metadata_grid['seed_count']} of {metadata_grid['full_seed_count']}", flush=True)
        for seed in seeds[: min(10, len(seeds))]:
            print(f"  seed {seed.index}: {seed.grid_index} {_point_payload(seed.point)}", flush=True)
        return

    jsonl_path, summary_path, resumed = _resume_or_new_paths(
        args.radius,
        args.spacing,
        args.shift,
        args.limit,
        settings,
        not args.no_resume,
    )
    metadata = _run_start_payload(jsonl_path, summary_path, args.radius, args.spacing, args.shift, args.limit, settings)
    if not resumed:
        _write_jsonl_event(jsonl_path, _event("run_start", metadata))
    print(
        f"{'resuming' if resumed else 'starting'} Aloff-Wallach scout: "
        f"{len(seeds)} seeds, workers={args.workers}, output={jsonl_path}",
        flush=True,
    )
    _run_scouts(seeds, jsonl_path, args.workers, settings, args.progress_every, args.chunksize)
    summary = _summary_payload(jsonl_path, metadata)
    _write_jsonl_event(jsonl_path, _event("run_summary", summary))
    _write_summary(summary_path, summary)
    print(f"wrote summary: {summary_path}", flush=True)
    print(f"classification counts: {summary['classification_counts']}", flush=True)


if __name__ == "__main__":
    main()
