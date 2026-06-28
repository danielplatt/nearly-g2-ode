"""Long S7 scout using left data plus right terminal-offset moduli germs."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from itertools import islice, product
from pathlib import Path

from mpmath import mp

from problem import LeftEndpointParameters, ProblemParameters, SolverConfig, round_s7_candidate_parameters, squashed_s7_parameters
from solver.march import solve_two_sided
from solver.two_sided_shooting import config_with_match_t

from ..shared.non_mirrored_common import RANDOM_SEED, _mp_string, _write_jsonl_event
from .right_germ import S7RightGermPoint, params_with_right_offset_moduli_germ


OUTPUT_DIR = Path("output/s7_full_moduli_offset_scouts")
SCOUT_VERSION = "s7-full-moduli-offset-scout-v1"

with mp.workdps(80):
    DEFAULT_MATCH_T = mp.pi / 6
    SCOUT_CONFIG = SolverConfig(6, 50, 24, mp.mpf("0.7"), 0, DEFAULT_MATCH_T)
    DEFAULT_BOUNDS = (
        (mp.mpf("-0.15"), mp.mpf("0.15")),
        (mp.mpf("-0.15"), mp.mpf("0.15")),
        (mp.mpf("-0.15"), mp.mpf("0.15")),
        (mp.mpf("-0.15"), mp.mpf("0.15")),
        (mp.mpf("-0.15"), mp.mpf("0.15")),
        (mp.mpf("-0.15"), mp.mpf("0.15")),
        (mp.mpf("-0.08"), mp.mpf("0.08")),
    )

COORDINATE_NAMES = ("u_left", "v_left", "r_left", "u_right", "v_right", "r_right", "s")
RIGHT_COORDINATE_DESCRIPTION = "u_right,v_right,r_right scale the terminal offset moduli A,B,C"
DEFAULT_AXIS_COUNT = 4


@dataclass(frozen=True)
class FullModuliOffsetPoint:
    """Seven scaled coordinates for the S7 offset-moduli scout."""

    u_left: mp.mpf
    v_left: mp.mpf
    r_left: mp.mpf
    u_right: mp.mpf
    v_right: mp.mpf
    r_right: mp.mpf
    s: mp.mpf


@dataclass(frozen=True)
class FullModuliOffsetSeed:
    """One deterministic offset-moduli scout seed."""

    index: int
    target: str
    point: FullModuliOffsetPoint


@dataclass(frozen=True)
class FullModuliOffsetResult:
    """One offset-moduli scout residual result."""

    seed: FullModuliOffsetSeed
    residual: tuple[mp.mpf, ...]
    residual_norm: mp.mpf
    raw_residual_norm: mp.mpf
    germ_residual_norm: mp.mpf
    germ_success: bool
    germ_evaluations: int
    left_l: mp.mpf | None
    right_l: mp.mpf | None
    patch_counts: tuple[int, int]
    failure: str | None = None


def _target_params(target: str) -> ProblemParameters:
    """Return the known S7 target parameters."""
    if target == "round":
        return round_s7_candidate_parameters()
    if target == "squashed":
        return squashed_s7_parameters()
    raise ValueError(f"Unknown S7 target {target!r}.")


def _parse_targets(value: str) -> tuple[str, ...]:
    """Parse a comma-separated target list."""
    targets = tuple(item.strip() for item in value.split(",") if item.strip())
    if not targets:
        raise argparse.ArgumentTypeError("must include at least one target")
    for target in targets:
        _target_params(target)
    return targets


def _positive_int(value: str) -> int:
    """Parse one positive integer CLI argument."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _axis_values(low: mp.mpf, high: mp.mpf, count: int) -> tuple[mp.mpf, ...]:
    """Return evenly spaced cell-center axis values."""
    if count < 1:
        raise ValueError("axis count must be positive")
    width = (high - low) / count
    return tuple(low + (mp.mpf(index) + mp.mpf("0.5")) * width for index in range(count))


def _grid_axes(axis_count: int) -> tuple[tuple[mp.mpf, ...], ...]:
    """Return all grid axes for the scout box."""
    return tuple(_axis_values(low, high, axis_count) for low, high in DEFAULT_BOUNDS)


def scout_seed_count(targets: tuple[str, ...] = ("round", "squashed"), axis_count: int = DEFAULT_AXIS_COUNT) -> int:
    """Return the number of deterministic scout seeds."""
    return len(targets) * (axis_count ** len(COORDINATE_NAMES))


def _point_payload(point: FullModuliOffsetPoint) -> dict[str, str]:
    """Return JSON-ready point coordinates."""
    values = (point.u_left, point.v_left, point.r_left, point.u_right, point.v_right, point.r_right, point.s)
    return {name: _mp_string(value) for name, value in zip(COORDINATE_NAMES, values)}


def _iter_seeds(targets: tuple[str, ...], axis_count: int) -> list[FullModuliOffsetSeed]:
    """Build deterministic scout seeds."""
    axes = _grid_axes(axis_count)
    seeds: list[FullModuliOffsetSeed] = []
    index = 0
    for target in targets:
        for values in product(*axes):
            seeds.append(FullModuliOffsetSeed(index, target, FullModuliOffsetPoint(*values)))
            index += 1
    return seeds


def _left_from_point(base: ProblemParameters, point: FullModuliOffsetPoint) -> LeftEndpointParameters:
    """Return left endpoint data from the scaled scout point."""
    return LeftEndpointParameters(
        a=base.left.a * mp.exp(point.u_left),
        c=base.left.c * mp.exp(point.v_left),
        alpha=base.left.alpha * (1 + point.r_left),
    )


def _local_config(point: FullModuliOffsetPoint) -> SolverConfig:
    """Return the interval/match config for one scaled scout point."""
    match_t = SCOUT_CONFIG.match_t * mp.exp(point.s)
    return config_with_match_t(SCOUT_CONFIG, match_t)


@lru_cache(maxsize=None)
def _reference_residual(target: str) -> tuple[mp.mpf, ...]:
    """Return the target's finite-order order-6 residual vector."""
    zero = FullModuliOffsetPoint(*(mp.zero for _ in COORDINATE_NAMES))
    seed = FullModuliOffsetSeed(-1, target, zero)
    result = evaluate_seed(seed, calibrate=False)
    if result.failure is not None:
        raise RuntimeError(f"Could not evaluate {target} reference: {result.failure}")
    return result.residual


def evaluate_seed(seed: FullModuliOffsetSeed, calibrate: bool = True) -> FullModuliOffsetResult:
    """Evaluate one full-moduli offset scout seed."""
    base = _target_params(seed.target)
    config = _local_config(seed.point)
    right_point = S7RightGermPoint(seed.point.u_right, seed.point.v_right, seed.point.r_right)
    try:
        params, germ = params_with_right_offset_moduli_germ(
            target=seed.target,
            point=right_point,
            left_params=_left_from_point(base, seed.point),
            interval_end=2 * config.match_t,
            order=config.series_order,
        )
        result = solve_two_sided(params, config)
    except (TypeError, ValueError, ZeroDivisionError, RuntimeError) as exc:
        return FullModuliOffsetResult(seed, (), mp.inf, mp.inf, mp.inf, False, 0, None, None, (0, 0), str(exc))

    residual = tuple(result.mismatch_q)
    raw_norm = max(abs(value) for value in residual)
    if calibrate:
        reference = _reference_residual(seed.target)
        residual = tuple(value - ref for value, ref in zip(residual, reference))
    norm = max(abs(value) for value in residual)
    return FullModuliOffsetResult(
        seed=seed,
        residual=residual,
        residual_norm=norm,
        raw_residual_norm=raw_norm,
        germ_residual_norm=germ.residual_norm,
        germ_success=germ.success,
        germ_evaluations=germ.evaluations,
        left_l=result.left_l,
        right_l=result.right_l,
        patch_counts=(len(result.left.patches), len(result.right.patches)),
    )


def _result_payload(result: FullModuliOffsetResult) -> dict:
    """Return JSON-ready result payload."""
    return {
        "seed_index": result.seed.index,
        "target": result.seed.target,
        "point": _point_payload(result.seed.point),
        "residual": [_mp_string(value) for value in result.residual],
        "residual_norm": _mp_string(result.residual_norm),
        "raw_residual_norm": _mp_string(result.raw_residual_norm),
        "germ_residual_norm": _mp_string(result.germ_residual_norm),
        "germ_success": result.germ_success,
        "germ_evaluations": result.germ_evaluations,
        "left_l": None if result.left_l is None else _mp_string(result.left_l),
        "right_l": None if result.right_l is None else _mp_string(result.right_l),
        "patch_counts": list(result.patch_counts),
        "failure": result.failure,
    }


def _output_path() -> Path:
    """Return a fresh scout JSONL output path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return OUTPUT_DIR / f"{stamp}-seed{RANDOM_SEED}-{SCOUT_VERSION}.jsonl"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Long S7 7D terminal-offset full-moduli scout.")
    parser.add_argument("--targets", type=_parse_targets, default=("round", "squashed"))
    parser.add_argument("--axis-count", type=_positive_int, default=DEFAULT_AXIS_COUNT)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--executor", choices=("process", "thread"), default="process")
    parser.add_argument("--max-points", type=_positive_int, default=None, help="debug cap for smoke tests")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the S7 offset-moduli full scout."""
    args = _parse_args()
    seeds = _iter_seeds(args.targets, args.axis_count)
    if args.max_points is not None:
        seeds = list(islice(seeds, args.max_points))
    count = len(seeds)
    total_count = scout_seed_count(args.targets, args.axis_count)
    print("S7 full-moduli terminal-offset scout", flush=True)
    print(f"version: {SCOUT_VERSION}", flush=True)
    print(f"targets: {','.join(args.targets)}", flush=True)
    print(f"axis count: {args.axis_count}", flush=True)
    print(f"full seed count: {total_count}", flush=True)
    print(f"scheduled seed count: {count}", flush=True)
    print(f"workers: {args.workers}", flush=True)
    print(f"executor: {args.executor}", flush=True)
    if args.dry_run:
        return

    path = _output_path()
    _write_jsonl_event(
        path,
        {
            "event": "run_start",
            "version": SCOUT_VERSION,
            "seed": RANDOM_SEED,
            "targets": list(args.targets),
            "axis_count": args.axis_count,
            "coordinate_names": list(COORDINATE_NAMES),
            "right_coordinate_description": RIGHT_COORDINATE_DESCRIPTION,
            "bounds": [[_mp_string(low), _mp_string(high)] for low, high in DEFAULT_BOUNDS],
            "config": {
                "order": SCOUT_CONFIG.series_order,
                "dps": SCOUT_CONFIG.working_dps,
                "match_t": _mp_string(SCOUT_CONFIG.match_t),
            },
            "scheduled_seed_count": count,
            "full_seed_count": total_count,
        },
    )
    counts: Counter[str] = Counter()
    best: FullModuliOffsetResult | None = None
    executor = None
    if args.workers == 1:
        iterator = map(evaluate_seed, seeds)
    else:
        executor_cls = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
        try:
            executor = executor_cls(max_workers=args.workers)
        except PermissionError:
            print("process executor unavailable; falling back to serial execution", flush=True)
            iterator = map(evaluate_seed, seeds)
        else:
            iterator = executor.map(evaluate_seed, seeds, chunksize=1)
    try:
        for result in iterator:
            counts["failed" if result.failure else "success"] += 1
            if result.failure is None and (best is None or result.residual_norm < best.residual_norm):
                best = result
            _write_jsonl_event(path, {"event": "scout_result", **_result_payload(result)})
            done = counts["failed"] + counts["success"]
            if done % 25 == 0 or done == count:
                best_text = "n/a" if best is None else mp.nstr(best.residual_norm, 8)
                print(f"completed {done}/{count}; failures={counts['failed']}; best={best_text}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
    summary = {
        "event": "run_summary",
        "version": SCOUT_VERSION,
        "scheduled_seed_count": count,
        "counts": dict(counts),
        "best": None if best is None else _result_payload(best),
    }
    _write_jsonl_event(path, summary)
    print(f"wrote {path}", flush=True)
    if best is not None:
        print(f"best target={best.seed.target} seed={best.seed.index} residual={mp.nstr(best.residual_norm, 12)}", flush=True)


if __name__ == "__main__":
    main()
