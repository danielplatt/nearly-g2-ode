"""Higher-order diagnostics for S7 full-moduli first-jet scout minima."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from itertools import repeat
from pathlib import Path

from mpmath import mp

from problem import ProblemParameters, SolverConfig
from solver.march import solve_two_sided
from solver.two_sided_shooting import config_with_match_t

from ..shared.non_mirrored_common import RANDOM_SEED, _event, _mp_string, _output_paths, _write_jsonl_event, _write_summary
from . import full_moduli_firstjet_scout as scout
from .right_germ import S7RightGermPoint, params_with_right_firstjet_germ


OUTPUT_DIR = Path("output/s7_full_moduli_firstjet_refinements")
REFINEMENT_VERSION = "s7-full-moduli-firstjet-refine-v1"
OUTPUT_SUFFIX = "s7-full-moduli-firstjet-refine-v1"
SELECTION_VERSION = "target-local-minima-v1"

with mp.workdps(80):
    EVALUATION_CONFIGS = {
        8: SolverConfig(8, 60, 28, mp.mpf("0.75"), 1, scout.SCOUT_CONFIG.match_t),
        10: SolverConfig(10, 80, 35, mp.mpf("0.65"), 2, scout.SCOUT_CONFIG.match_t),
        14: SolverConfig(14, 100, 42, mp.mpf("0.55"), 2, scout.SCOUT_CONFIG.match_t),
    }
    DEFAULT_ORDERS = (8, 10, 14)


@dataclass(frozen=True)
class FullModuliScoutCandidate:
    """One persisted full-moduli scout candidate."""

    seed: scout.FullModuliSeed
    result: scout.FullModuliResult


@dataclass(frozen=True)
class SelectedFullModuliCandidate:
    """One selected local minimum from the full-moduli scout grid."""

    rank: int
    reason: str
    candidate: FullModuliScoutCandidate


@dataclass(frozen=True)
class FullModuliEvaluation:
    """One calibrated higher-order evaluation of a selected 7D point."""

    seed: scout.FullModuliSeed
    order: int
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


@dataclass(frozen=True)
class FullModuliDiagnosticTrack:
    """Higher-order diagnostic track for one selected scout minimum."""

    selection: SelectedFullModuliCandidate
    evaluations: tuple[FullModuliEvaluation, ...]
    classification: str


def _iter_jsonl_events(path: Path):
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
    """Return whether one JSONL checkpoint contains a specific event."""
    return any(event.get("event") == event_type for event in _iter_jsonl_events(path))


def _latest_completed_scout_jsonl() -> Path:
    """Return the newest completed full-moduli first-jet scout checkpoint."""
    pattern = f"*-seed{RANDOM_SEED}-{scout.SCOUT_VERSION}.jsonl"
    candidates = sorted(scout.OUTPUT_DIR.glob(pattern), reverse=True)
    for path in candidates:
        if _jsonl_has_event(path, "run_summary"):
            return path
    raise FileNotFoundError(f"No completed full-moduli first-jet scout found under {scout.OUTPUT_DIR}")


def _scout_run_start(path: Path) -> dict:
    """Return the run_start event from one scout checkpoint."""
    for event in _iter_jsonl_events(path):
        if event.get("event") == "run_start":
            return event
    raise ValueError(f"{path} does not contain a run_start event")


def _point_from_payload(payload: dict) -> scout.FullModuliPoint:
    """Rebuild one full-moduli point from persisted JSON strings."""
    return scout.FullModuliPoint(*(mp.mpf(payload[name]) for name in scout.COORDINATE_NAMES))


def _result_from_payload(payload: dict) -> scout.FullModuliResult:
    """Rebuild one scout result from a persisted scout_result event."""
    seed = scout.FullModuliSeed(int(payload["seed_index"]), payload["target"], _point_from_payload(payload["point"]))
    residual = tuple(mp.mpf(value) for value in payload.get("residual", ()))
    failure = payload.get("failure")
    return scout.FullModuliResult(
        seed=seed,
        residual=residual,
        residual_norm=mp.inf if failure else mp.mpf(payload["residual_norm"]),
        raw_residual_norm=mp.inf if failure else mp.mpf(payload["raw_residual_norm"]),
        germ_residual_norm=mp.inf if payload.get("germ_residual_norm") is None else mp.mpf(payload["germ_residual_norm"]),
        germ_success=bool(payload.get("germ_success")),
        germ_evaluations=int(payload.get("germ_evaluations", 0)),
        left_l=None if payload.get("left_l") is None else mp.mpf(payload["left_l"]),
        right_l=None if payload.get("right_l") is None else mp.mpf(payload["right_l"]),
        patch_counts=tuple(payload.get("patch_counts", (0, 0))),
        failure=failure,
    )


def _load_scout_candidates(path: Path, targets: tuple[str, ...] | None = None) -> list[FullModuliScoutCandidate]:
    """Load all full-moduli scout candidates from one completed JSONL."""
    target_set = None if targets is None else set(targets)
    output = []
    for event in _iter_jsonl_events(path):
        if event.get("event") != "scout_result":
            continue
        if target_set is not None and event.get("target") not in target_set:
            continue
        result = _result_from_payload(event)
        output.append(FullModuliScoutCandidate(result.seed, result))
    return output


def _candidate_norm(candidate: FullModuliScoutCandidate) -> mp.mpf:
    """Return one candidate residual norm, treating failures as infinity."""
    return candidate.result.residual_norm if candidate.result.failure is None else mp.inf


def _grid_shape(run_start: dict) -> tuple[int, ...]:
    """Return the 7D scout grid shape from run_start metadata."""
    axis_count = int(run_start["axis_count"])
    return tuple(axis_count for _ in scout.COORDINATE_NAMES)


def _grid_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return row-major strides for the persisted full-moduli grid order."""
    strides = []
    product = 1
    for size in reversed(shape):
        strides.append(product)
        product *= size
    return tuple(reversed(strides))


def _grid_coordinate(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return the grid coordinate tuple for one local seed index."""
    remaining = index
    coordinates = []
    for stride, size in zip(_grid_strides(shape), shape):
        coordinates.append(remaining // stride)
        remaining %= stride
        if coordinates[-1] >= size:
            raise ValueError(f"Seed index {index} is outside grid shape {shape}")
    return tuple(coordinates)


def _grid_index(coordinates: tuple[int, ...], shape: tuple[int, ...]) -> int:
    """Return the local seed index for one grid coordinate tuple."""
    return sum(value * stride for value, stride in zip(coordinates, _grid_strides(shape)))


def _neighbor_indices(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return nearest-neighbor local grid indices for one local seed index."""
    coordinates = _grid_coordinate(index, shape)
    output = []
    for dimension, size in enumerate(shape):
        for offset in (-1, 1):
            neighbor = list(coordinates)
            neighbor[dimension] += offset
            if 0 <= neighbor[dimension] < size:
                output.append(_grid_index(tuple(neighbor), shape))
    return tuple(output)


def _local_index(candidate: FullModuliScoutCandidate, per_target: int) -> int:
    """Return the seed index local to its target block."""
    return candidate.seed.index % per_target


def _target_local_minima(
    candidates: list[FullModuliScoutCandidate],
    shape: tuple[int, ...],
) -> list[FullModuliScoutCandidate]:
    """Return target-wise nearest-neighbor local minima in scout residual."""
    per_target = 1
    for size in shape:
        per_target *= size

    grouped: dict[str, list[FullModuliScoutCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.seed.target].append(candidate)

    minima: list[FullModuliScoutCandidate] = []
    for group in grouped.values():
        by_local = {_local_index(candidate, per_target): candidate for candidate in group}
        for candidate in group:
            if candidate.result.failure is not None:
                continue
            local = _local_index(candidate, per_target)
            norm = candidate.result.residual_norm
            if all(norm <= _candidate_norm(by_local[neighbor]) for neighbor in _neighbor_indices(local, shape) if neighbor in by_local):
                minima.append(candidate)
    return sorted(minima, key=lambda candidate: (candidate.result.residual_norm, candidate.seed.target, candidate.seed.index))


def _select_local_minima(
    candidates: list[FullModuliScoutCandidate],
    run_start: dict,
    max_residual: mp.mpf | None = None,
    limit: int | None = None,
) -> list[SelectedFullModuliCandidate]:
    """Select target-wise full-moduli scout local minima."""
    minima = _target_local_minima(candidates, _grid_shape(run_start))
    if max_residual is not None:
        minima = [candidate for candidate in minima if candidate.result.residual_norm < max_residual]
    if limit is not None:
        minima = minima[:limit]
    return [SelectedFullModuliCandidate(index + 1, "local-minimum", candidate) for index, candidate in enumerate(minima)]


def _target_params(target: str) -> ProblemParameters:
    """Return known S7 target parameters."""
    return scout._target_params(target)


def _local_config(point: scout.FullModuliPoint, template_config: SolverConfig) -> SolverConfig:
    """Return the point-specific interval/match config."""
    match_t = template_config.match_t * mp.exp(point.s)
    return config_with_match_t(template_config, match_t)


def _evaluate_raw(seed: scout.FullModuliSeed, config: SolverConfig) -> FullModuliEvaluation:
    """Evaluate one selected seed at a requested finite order."""
    base = _target_params(seed.target)
    local_config = _local_config(seed.point, config)
    right_point = S7RightGermPoint(seed.point.u_right, seed.point.v_right, seed.point.r_right)
    try:
        params, germ = params_with_right_firstjet_germ(
            target=seed.target,
            point=right_point,
            left_params=scout._left_from_point(base, seed.point),
            interval_end=2 * local_config.match_t,
            order=local_config.series_order,
        )
        result = solve_two_sided(params, local_config)
    except (TypeError, ValueError, ZeroDivisionError, RuntimeError) as exc:
        return FullModuliEvaluation(seed, config.series_order, (), mp.inf, mp.inf, mp.inf, False, 0, None, None, (0, 0), str(exc))

    residual = tuple(result.mismatch_q)
    raw_norm = max(abs(value) for value in residual)
    return FullModuliEvaluation(
        seed=seed,
        order=config.series_order,
        residual=residual,
        residual_norm=raw_norm,
        raw_residual_norm=raw_norm,
        germ_residual_norm=germ.residual_norm,
        germ_success=germ.success,
        germ_evaluations=germ.evaluations,
        left_l=result.left_l,
        right_l=result.right_l,
        patch_counts=(len(result.left.patches), len(result.right.patches)),
    )


@lru_cache(maxsize=None)
def _reference_residual(target: str, order: int) -> tuple[mp.mpf, ...]:
    """Return the known-target raw residual vector for one finite order."""
    zero = scout.FullModuliPoint(*(mp.zero for _ in scout.COORDINATE_NAMES))
    seed = scout.FullModuliSeed(-1, target, zero)
    result = _evaluate_raw(seed, EVALUATION_CONFIGS[order])
    if result.failure is not None:
        raise RuntimeError(f"Could not evaluate {target} order-{order} reference: {result.failure}")
    return result.residual


def _calibrated_evaluation(seed: scout.FullModuliSeed, order: int) -> FullModuliEvaluation:
    """Evaluate one seed and subtract the finite-order known-target bias."""
    config = EVALUATION_CONFIGS[order]
    try:
        with mp.workdps(config.working_dps):
            raw = _evaluate_raw(seed, config)
            if raw.failure is not None:
                return raw
            reference = _reference_residual(seed.target, order)
            residual = tuple(value - ref for value, ref in zip(raw.residual, reference))
            norm = max(abs(value) for value in residual)
            return FullModuliEvaluation(
                seed=seed,
                order=order,
                residual=residual,
                residual_norm=norm,
                raw_residual_norm=raw.raw_residual_norm,
                germ_residual_norm=raw.germ_residual_norm,
                germ_success=raw.germ_success,
                germ_evaluations=raw.germ_evaluations,
                left_l=raw.left_l,
                right_l=raw.right_l,
                patch_counts=raw.patch_counts,
                failure=None,
            )
    except (TypeError, ValueError, ZeroDivisionError, RuntimeError) as exc:
        return FullModuliEvaluation(seed, order, (), mp.inf, mp.inf, mp.inf, False, 0, None, None, (0, 0), str(exc))


def _point_distance(point: scout.FullModuliPoint) -> mp.mpf:
    """Return max coordinate distance from the known target chart center."""
    return max(abs(value) for value in (point.u_left, point.v_left, point.r_left, point.u_right, point.v_right, point.r_right, point.s))


def _classify_track(selection: SelectedFullModuliCandidate, evaluations: tuple[FullModuliEvaluation, ...]) -> str:
    """Classify one higher-order diagnostic track."""
    if selection.candidate.result.failure or any(result.failure for result in evaluations):
        return "failed"
    norms = tuple(result.residual_norm for result in evaluations)
    if not norms:
        return "inconclusive"
    if max(norms) < mp.mpf("1e-8") and _point_distance(selection.candidate.seed.point) >= mp.mpf("0.05"):
        return "possible_other_s7_root"
    if min(norms) > mp.mpf("1e-2"):
        return "high_order_nonzero_at_scout_point"
    return "inconclusive"


def _run_selection(selection: SelectedFullModuliCandidate, orders: tuple[int, ...]) -> FullModuliDiagnosticTrack:
    """Evaluate one selected local minimum at all requested orders."""
    evaluations = tuple(_calibrated_evaluation(selection.candidate.seed, order) for order in orders)
    return FullModuliDiagnosticTrack(selection, evaluations, _classify_track(selection, evaluations))


def _point_payload(point: scout.FullModuliPoint) -> dict[str, str]:
    """Return JSON-ready 7D point coordinates."""
    return scout._point_payload(point)


def _evaluation_payload(result: FullModuliEvaluation) -> dict:
    """Return JSON-ready data for one higher-order evaluation."""
    return {
        "seed_index": result.seed.index,
        "target": result.seed.target,
        "order": result.order,
        "point": _point_payload(result.seed.point),
        "residual": [_mp_string(value) for value in result.residual],
        "residual_norm": _mp_string(result.residual_norm),
        "raw_residual_norm": _mp_string(result.raw_residual_norm),
        "germ_residual_norm": _mp_string(result.germ_residual_norm),
        "germ_success": result.germ_success,
        "germ_evaluations": result.germ_evaluations,
        "left_l": _mp_string(result.left_l),
        "right_l": _mp_string(result.right_l),
        "l_gap": None if result.left_l is None or result.right_l is None else _mp_string(abs(result.left_l - result.right_l)),
        "patch_counts": list(result.patch_counts),
        "failure": result.failure,
    }


def _selected_payload(selection: SelectedFullModuliCandidate) -> dict:
    """Return JSON-ready data for one selected local minimum."""
    candidate = selection.candidate
    return {
        "rank": selection.rank,
        "reason": selection.reason,
        "seed_index": candidate.seed.index,
        "target": candidate.seed.target,
        "distance": _mp_string(_point_distance(candidate.seed.point)),
        "scout_residual_norm": _mp_string(candidate.result.residual_norm),
        "scout_raw_residual_norm": _mp_string(candidate.result.raw_residual_norm),
        "scout_germ_residual_norm": _mp_string(candidate.result.germ_residual_norm),
        "point": _point_payload(candidate.seed.point),
    }


def _track_payload(track: FullModuliDiagnosticTrack) -> dict:
    """Return JSON-ready data for one classified diagnostic track."""
    return {
        **_selected_payload(track.selection),
        "classification": track.classification,
        "evaluations": [_evaluation_payload(result) for result in track.evaluations],
        "evaluation_norms": [_mp_string(result.residual_norm) for result in track.evaluations],
        "evaluation_orders": [result.order for result in track.evaluations],
    }


def _selection_config_payload(
    orders: tuple[int, ...],
    max_residual: mp.mpf | None,
    targets: tuple[str, ...] | None,
    limit: int | None,
) -> dict:
    """Return JSON-ready selection/evaluation config."""
    return {
        "selection_version": SELECTION_VERSION,
        "selection_mode": "target-local-minima",
        "local_minimum_max_residual": _mp_string(max_residual),
        "targets": None if targets is None else list(targets),
        "limit": limit,
        "orders": list(orders),
        "settings": [
            {
                "order": order,
                "dps": EVALUATION_CONFIGS[order].working_dps,
                "target_dps": EVALUATION_CONFIGS[order].target_dps,
                "step_safety": _mp_string(EVALUATION_CONFIGS[order].step_safety),
                "sample_points": EVALUATION_CONFIGS[order].sample_points,
            }
            for order in orders
        ],
    }


def _run_start_payload(
    jsonl_path: Path,
    summary_path: Path,
    scout_jsonl: Path,
    orders: tuple[int, ...],
    max_residual: mp.mpf | None,
    targets: tuple[str, ...] | None,
    limit: int | None,
) -> dict:
    """Return checkpoint metadata for one diagnostic run."""
    return {
        "random_seed": RANDOM_SEED,
        "refinement_version": REFINEMENT_VERSION,
        "scout_jsonl": str(scout_jsonl),
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "selection_config": _selection_config_payload(orders, max_residual, targets, limit),
    }


def _summary_payload(
    scout_jsonl: Path,
    scout_start: dict,
    selections: list[SelectedFullModuliCandidate],
    tracks: list[FullModuliDiagnosticTrack],
    selection_config: dict,
) -> dict:
    """Return final summary data for one diagnostic run."""
    counts = Counter(track.classification for track in tracks)
    best = sorted(tracks, key=lambda track: (max(track.evaluations[-1].residual_norm for _ in (0,)), track.selection.candidate.seed.index))
    return {
        "random_seed": RANDOM_SEED,
        "refinement_version": REFINEMENT_VERSION,
        "scout_jsonl": str(scout_jsonl),
        "scout_axis_count": scout_start["axis_count"],
        "selection_config": selection_config,
        "selection_count": len(selections),
        "classified_count": len(tracks),
        "classification_counts": dict(counts),
        "selections": [_selected_payload(selection) for selection in selections],
        "best_tracks": [_track_payload(track) for track in best[:20]],
        "tracks": [_track_payload(track) for track in tracks],
    }


def _parse_targets(value: str) -> tuple[str, ...]:
    """Parse a comma-separated target list."""
    targets = tuple(item.strip() for item in value.split(",") if item.strip())
    if not targets:
        raise argparse.ArgumentTypeError("must include at least one target")
    for target in targets:
        _target_params(target)
    return targets


def _parse_orders(value: str) -> tuple[int, ...]:
    """Parse a comma-separated order list."""
    orders = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not orders:
        raise argparse.ArgumentTypeError("must include at least one order")
    unknown = [order for order in orders if order not in EVALUATION_CONFIGS]
    if unknown:
        known = ", ".join(str(order) for order in sorted(EVALUATION_CONFIGS))
        raise argparse.ArgumentTypeError(f"unknown order(s) {unknown}; choose from {known}")
    return orders


def _optional_positive_mpf(value: str) -> mp.mpf | None:
    """Parse a positive mpmath value, or `none` for no cutoff."""
    if value.lower() in {"none", "all"}:
        return None
    parsed = mp.mpf(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive, or 'none'")
    return parsed


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI argument."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Higher-order diagnostics for S7 full-moduli first-jet scout minima.")
    parser.add_argument("--scout-jsonl", type=Path, default=None)
    parser.add_argument("--orders", type=_parse_orders, default=DEFAULT_ORDERS)
    parser.add_argument("--max-residual", type=_optional_positive_mpf, default=None)
    parser.add_argument("--targets", type=_parse_targets, default=None)
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--executor", choices=("process", "thread"), default="process")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _print_selection(selection: SelectedFullModuliCandidate) -> None:
    """Print one selected candidate."""
    point = selection.candidate.seed.point
    values = ", ".join(f"{name}={mp.nstr(value, 8)}" for name, value in zip(scout.COORDINATE_NAMES, (point.u_left, point.v_left, point.r_left, point.u_right, point.v_right, point.r_right, point.s)))
    print(
        f"  {selection.rank:02d}: seed={selection.candidate.seed.index} target={selection.candidate.seed.target} "
        f"scout={mp.nstr(selection.candidate.result.residual_norm, 12)} {values}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    """Run higher-order diagnostics for selected full-moduli scout minima."""
    args = _parse_args(argv)
    scout_jsonl = args.scout_jsonl or _latest_completed_scout_jsonl()
    scout_start = _scout_run_start(scout_jsonl)
    candidates = _load_scout_candidates(scout_jsonl, args.targets)
    selections = _select_local_minima(candidates, scout_start, args.max_residual, args.limit)

    print("S7 full-moduli first-jet local-minimum diagnostics", flush=True)
    print(f"version: {REFINEMENT_VERSION}", flush=True)
    print(f"scout: {scout_jsonl}", flush=True)
    print(f"orders: {','.join(str(order) for order in args.orders)}", flush=True)
    print(f"selected local minima: {len(selections)}", flush=True)
    for selection in selections:
        _print_selection(selection)
    if args.dry_run:
        return

    jsonl_path, summary_path = _output_paths(OUTPUT_DIR, OUTPUT_SUFFIX)
    selection_config = _selection_config_payload(args.orders, args.max_residual, args.targets, args.limit)
    _write_jsonl_event(
        jsonl_path,
        _event("run_start", _run_start_payload(jsonl_path, summary_path, scout_jsonl, args.orders, args.max_residual, args.targets, args.limit)),
    )
    for selection in selections:
        _write_jsonl_event(jsonl_path, _event("candidate_selected", _selected_payload(selection)))

    if args.workers == 1:
        iterator = map(_run_selection, selections, repeat(args.orders))
        executor = None
    else:
        executor_cls = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
        try:
            executor = executor_cls(max_workers=args.workers)
        except PermissionError:
            print("process executor unavailable; falling back to serial execution", flush=True)
            executor = None
            iterator = map(_run_selection, selections, repeat(args.orders))
        else:
            iterator = executor.map(_run_selection, selections, repeat(args.orders), chunksize=1)

    tracks: list[FullModuliDiagnosticTrack] = []
    try:
        for index, track in enumerate(iterator, start=1):
            tracks.append(track)
            for evaluation in track.evaluations:
                _write_jsonl_event(
                    jsonl_path,
                    _event(
                        "candidate_evaluation",
                        {
                            "selection_rank": track.selection.rank,
                            "selection_reason": track.selection.reason,
                            **_evaluation_payload(evaluation),
                        },
                    ),
                )
            _write_jsonl_event(jsonl_path, _event("candidate_classification", _track_payload(track)))
            final = track.evaluations[-1] if track.evaluations else None
            final_text = "n/a" if final is None else mp.nstr(final.residual_norm, 12)
            print(
                f"classified {index}/{len(selections)} seed={track.selection.candidate.seed.index} "
                f"target={track.selection.candidate.seed.target} class={track.classification} final={final_text}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown()

    summary = _summary_payload(scout_jsonl, scout_start, selections, tracks, selection_config)
    _write_jsonl_event(jsonl_path, _event("run_summary", summary))
    _write_summary(summary_path, summary)
    print(f"wrote {jsonl_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    print(f"classifications: {dict(Counter(track.classification for track in tracks))}", flush=True)


if __name__ == "__main__":
    main()
