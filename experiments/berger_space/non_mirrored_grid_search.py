"""Calibrated parallel 7D near-grid scout search."""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, localcontext
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable

import json
from mpmath import mp

from problem import DEFAULT_CONFIG, DEFAULT_PARAMS, ProblemParameters, SolverConfig, mirrored_problem_parameters
from solver.two_sided_shooting import two_sided_residual

from ..shared.non_mirrored_common import (
    RANDOM_SEED,
    SearchCandidate,
    SearchSeed,
    _candidate_payload,
    _event,
    _mp_string,
    _output_paths as _common_output_paths,
    _point_from_values,
    _write_jsonl_event,
    _write_summary,
)


OUTPUT_DIR = Path("output/non_mirrored_grid_searches")
SEARCH_VERSION = "grid-v1"
OUTPUT_SUFFIX = "non-mirrored-grid-v1"
TOP_SCOUTS = 20
DEFAULT_GRID_SHIFT = "vertex"
GRID_SHIFTS = (DEFAULT_GRID_SHIFT, "cell-center")

with mp.workdps(80):
    CALIBRATED_RECOVERY_RADIUS = mp.mpf("0.2")
    DEFAULT_GRID_SPACING = mp.mpf("0.4")
    SCOUT_STEP_SAFETY = mp.mpf("0.95")
    SYMMETRIC_ODD_RADIUS = mp.sqrt(5) / 20

COORDINATE_NAMES = ("u_left", "v_left", "r_left", "u_right", "v_right", "r_right", "s")

SCOUT_CONFIG = SolverConfig(4, 30, 15, SCOUT_STEP_SAFETY, 0, DEFAULT_CONFIG.match_t)
MIXED_MU_SCOUT_CONFIG = SolverConfig(6, 50, 15, mp.mpf("0.5"), 0, DEFAULT_CONFIG.match_t)


def _positive_ac_base_params() -> ProblemParameters:
    """Return the exploratory base point on the real ac > 0 endpoint branch."""
    with mp.workdps(80):
        sqrt5 = mp.sqrt(5)
        a = sqrt5 / 20
        c = 3 * sqrt5 / 100
        alpha = -sqrt5 / 50
        return mirrored_problem_parameters(a, c, alpha, DEFAULT_PARAMS.lam, DEFAULT_PARAMS.interval_end)


POSITIVE_AC_BASE_PARAMS = _positive_ac_base_params()


def _negative_ac_base_params() -> ProblemParameters:
    """Return an exploratory base point on the a < 0, c < 0, ac > 0 branch."""
    with mp.workdps(80):
        sqrt5 = mp.sqrt(5)
        a = -sqrt5 / 20
        c = -3 * sqrt5 / 5
        alpha = sqrt5 / 50
        return mirrored_problem_parameters(a, c, alpha, DEFAULT_PARAMS.lam, DEFAULT_PARAMS.interval_end)


NEGATIVE_AC_BASE_PARAMS = _negative_ac_base_params()


def _mixed_mu_base_params() -> ProblemParameters:
    """Return the exploratory opposite-mu base with validated endpoint-local p-signs."""
    with mp.workdps(80):
        return mirrored_problem_parameters(
            DEFAULT_PARAMS.left.a,
            DEFAULT_PARAMS.left.c,
            DEFAULT_PARAMS.left.alpha,
            DEFAULT_PARAMS.lam,
            DEFAULT_PARAMS.interval_end,
            left_mu=1,
            right_mu=1,
            p_signs=(1, 1, 1),
            right_p_signs=(-1, 1, -1),
        )


MIXED_MU_BASE_PARAMS = _mixed_mu_base_params()


@dataclass(frozen=True)
class GridRegion:
    """One rectangular region for calibrated 7D scout grids."""

    name: str
    bounds: tuple[tuple[mp.mpf, mp.mpf], ...]


def _r_bounds_for_symmetric_physical_odd(base_value: mp.mpf, radius: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    """Return scaled r-bounds for physical odd coefficient interval [-radius, radius]."""
    lower = -radius / base_value - 1
    upper = radius / base_value - 1
    return (min(lower, upper), max(lower, upper))


with mp.workdps(80):
    symmetric_alpha_bounds = _r_bounds_for_symmetric_physical_odd(DEFAULT_PARAMS.left.alpha, SYMMETRIC_ODD_RADIUS)
    symmetric_omega_bounds = _r_bounds_for_symmetric_physical_odd(DEFAULT_PARAMS.right.omega, SYMMETRIC_ODD_RADIUS)
    positive_ac_alpha_bounds = _r_bounds_for_symmetric_physical_odd(
        POSITIVE_AC_BASE_PARAMS.left.alpha, SYMMETRIC_ODD_RADIUS
    )
    positive_ac_omega_bounds = _r_bounds_for_symmetric_physical_odd(
        POSITIVE_AC_BASE_PARAMS.right.omega, SYMMETRIC_ODD_RADIUS
    )
    negative_ac_alpha_bounds = _r_bounds_for_symmetric_physical_odd(
        NEGATIVE_AC_BASE_PARAMS.left.alpha, SYMMETRIC_ODD_RADIUS
    )
    negative_ac_omega_bounds = _r_bounds_for_symmetric_physical_odd(
        NEGATIVE_AC_BASE_PARAMS.right.omega, SYMMETRIC_ODD_RADIUS
    )
    mixed_mu_alpha_bounds = _r_bounds_for_symmetric_physical_odd(
        MIXED_MU_BASE_PARAMS.left.alpha, SYMMETRIC_ODD_RADIUS
    )
    mixed_mu_omega_bounds = _r_bounds_for_symmetric_physical_odd(
        MIXED_MU_BASE_PARAMS.right.omega, SYMMETRIC_ODD_RADIUS
    )
    NEAR_GRID = GridRegion(
        "near",
        (
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            (mp.mpf("-1.5"), mp.mpf("1.5")),
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            (mp.mpf("-1.5"), mp.mpf("1.5")),
            (mp.mpf("-0.8"), mp.mpf("0.8")),
        ),
    )
    SYMMETRIC_ALPHA_OMEGA_GRID = GridRegion(
        "symmetric-alpha-omega",
        (
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            symmetric_alpha_bounds,
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            symmetric_omega_bounds,
            (mp.mpf("-0.8"), mp.mpf("0.8")),
        ),
    )
    POSITIVE_AC_GRID = GridRegion(
        "positive-ac",
        (
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            positive_ac_alpha_bounds,
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            positive_ac_omega_bounds,
            (mp.mpf("-0.8"), mp.mpf("0.8")),
        ),
    )
    NEGATIVE_AC_GRID = GridRegion(
        "negative-ac",
        (
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            negative_ac_alpha_bounds,
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            negative_ac_omega_bounds,
            (mp.mpf("-0.8"), mp.mpf("0.8")),
        ),
    )
    MIXED_MU_SHORT_GRID = GridRegion(
        "mixed-mu-short",
        (
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            mixed_mu_alpha_bounds,
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            (mp.mpf("-0.6"), mp.mpf("0.6")),
            mixed_mu_omega_bounds,
            (mp.mpf("-2.0"), mp.mpf("-0.4")),
        ),
    )
    MIXED_MU_BOUNDARY_GRID = GridRegion(
        "mixed-mu-boundary",
        (
            (mp.mpf("-1.8"), mp.mpf("-0.6")),
            (mp.mpf("-1.8"), mp.mpf("-0.6")),
            mixed_mu_alpha_bounds,
            (mp.mpf("-1.8"), mp.mpf("-0.6")),
            (mp.mpf("-1.8"), mp.mpf("-0.6")),
            mixed_mu_omega_bounds,
            (mp.mpf("-3.2"), mp.mpf("-1.6")),
        ),
    )
    POSITIVE_AC_BOUNDARY_GRID = GridRegion(
        "positive-ac-boundary",
        (
            (mp.mpf("-1.4"), mp.mpf("-0.2")),
            (mp.mpf("-1.4"), mp.mpf("-0.2")),
            (mp.mpf("-4.3"), mp.mpf("1.5")),
            (mp.mpf("-1.4"), mp.mpf("-0.2")),
            (mp.mpf("-1.4"), mp.mpf("-0.2")),
            (mp.mpf("-4.3"), mp.mpf("1.5")),
            (mp.mpf("-1.2"), mp.mpf("0.8")),
        ),
    )
    POSITIVE_AC_BOUNDARY_V2_GRID = GridRegion(
        "positive-ac-boundary-v2",
        (
            (mp.mpf("-2.2"), mp.mpf("-1.0")),
            (mp.mpf("-2.2"), mp.mpf("-1.0")),
            (mp.mpf("-4.3"), mp.mpf("1.5")),
            (mp.mpf("-2.2"), mp.mpf("-1.0")),
            (mp.mpf("-2.2"), mp.mpf("-1.0")),
            (mp.mpf("-4.3"), mp.mpf("1.5")),
            (mp.mpf("-2.0"), mp.mpf("-0.8")),
        ),
    )
GRID_REGIONS = {
    region.name: region
    for region in (
        NEAR_GRID,
        SYMMETRIC_ALPHA_OMEGA_GRID,
        POSITIVE_AC_GRID,
        NEGATIVE_AC_GRID,
        MIXED_MU_SHORT_GRID,
        MIXED_MU_BOUNDARY_GRID,
        POSITIVE_AC_BOUNDARY_GRID,
        POSITIVE_AC_BOUNDARY_V2_GRID,
    )
}


def _default_workers() -> int:
    """Return a conservative default process count for scout evaluation."""
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count - 1))


def _positive_int(value: str) -> int:
    """Parse one positive integer CLI argument."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    """Parse one nonnegative integer CLI argument."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_mpf(value: str) -> mp.mpf:
    """Parse one positive mpmath decimal CLI argument."""
    with mp.workdps(80):
        parsed = mp.mpf(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _grid_region(name: str) -> GridRegion:
    """Return one named grid region."""
    try:
        return GRID_REGIONS[name]
    except KeyError as exc:
        known = ", ".join(sorted(GRID_REGIONS))
        raise ValueError(f"Unknown grid region {name!r}; choose one of: {known}") from exc


def _base_params_for_region(region_name: str) -> ProblemParameters:
    """Return the physical base endpoint parameters for one grid region."""
    _grid_region(region_name)
    if region_name.startswith("positive-ac"):
        return POSITIVE_AC_BASE_PARAMS
    if region_name.startswith("negative-ac"):
        return NEGATIVE_AC_BASE_PARAMS
    if region_name.startswith("mixed-mu"):
        return MIXED_MU_BASE_PARAMS
    return DEFAULT_PARAMS


def _scout_config_for_region(region_name: str) -> SolverConfig:
    """Return the scout Taylor settings for one grid region."""
    _grid_region(region_name)
    if region_name.startswith("mixed-mu"):
        return MIXED_MU_SCOUT_CONFIG
    return SCOUT_CONFIG


def _base_params_payload(params: ProblemParameters) -> dict[str, dict[str, str | None] | str | None]:
    """Return JSON-ready base endpoint parameters for run metadata."""
    payload = {
        "lambda": _mp_string(params.lam),
        "interval_end": _mp_string(params.interval_end),
        "left": {
            "a": _mp_string(params.left.a),
            "c": _mp_string(params.left.c),
            "alpha": _mp_string(params.left.alpha),
            "mu": params.left_mu,
        },
        "right": {
            "d": _mp_string(params.right.d),
            "f": _mp_string(params.right.f),
            "omega": _mp_string(params.right.omega),
            "mu": params.right_mu,
        },
    }
    if params.p_signs != DEFAULT_PARAMS.p_signs or params.right_p_signs is not None:
        payload["p_signs"] = list(params.p_signs)
        payload["right_p_signs"] = None if params.right_p_signs is None else list(params.right_p_signs)
    return payload


def _axis_values(low: mp.mpf, high: mp.mpf, max_spacing: mp.mpf) -> tuple[mp.mpf, ...]:
    """Return inclusive axis values with mesh width no larger than max_spacing."""
    with localcontext() as context:
        context.prec = 100
        low_decimal = Decimal(mp.nstr(low, 80))
        high_decimal = Decimal(mp.nstr(high, 80))
        spacing_decimal = Decimal(mp.nstr(max_spacing, 80))
        if spacing_decimal <= 0:
            raise ValueError("max_spacing must be positive")
        if high_decimal < low_decimal:
            raise ValueError("axis upper bound must be at least the lower bound")
        if high_decimal == low_decimal:
            return (mp.mpf(str(low_decimal)),)
        intervals = max(1, int(((high_decimal - low_decimal) / spacing_decimal).to_integral_value(rounding=ROUND_CEILING)))
        step = (high_decimal - low_decimal) / Decimal(intervals)
        values = [low_decimal + step * index for index in range(intervals + 1)]
        values[-1] = high_decimal
    with mp.workdps(max(mp.dps, 80)):
        return tuple(mp.mpf(str(value)) for value in values)


def _cell_center_axis(axis: tuple[mp.mpf, ...]) -> tuple[mp.mpf, ...]:
    """Return midpoints between adjacent vertex grid values."""
    if len(axis) < 2:
        return axis
    return tuple((axis[index] + axis[index + 1]) / 2 for index in range(len(axis) - 1))


def _grid_axes(region: GridRegion, spacing: mp.mpf, shift: str = DEFAULT_GRID_SHIFT) -> tuple[tuple[mp.mpf, ...], ...]:
    """Return all coordinate axes for one calibrated grid."""
    vertex_axes = tuple(_axis_values(low, high, spacing) for low, high in region.bounds)
    if shift == DEFAULT_GRID_SHIFT:
        return vertex_axes
    if shift == "cell-center":
        return tuple(_cell_center_axis(axis) for axis in vertex_axes)
    raise ValueError(f"Unknown grid shift {shift!r}; choose one of: {', '.join(GRID_SHIFTS)}")


def _axis_spacing(axis: tuple[mp.mpf, ...]) -> mp.mpf:
    """Return the largest adjacent spacing on one axis."""
    with mp.workdps(max(mp.dps, 80)):
        if len(axis) < 2:
            return mp.zero
        return max(axis[index + 1] - axis[index] for index in range(len(axis) - 1))


def _physical_odd_bounds(region: GridRegion) -> dict[str, list[str | None]]:
    """Return physical alpha/omega intervals induced by one scaled grid region."""
    base_params = _base_params_for_region(region.name)
    with mp.workdps(max(mp.dps, 80)):
        alpha_values = [base_params.left.alpha * (1 + value) for value in region.bounds[2]]
        omega_values = [base_params.right.omega * (1 + value) for value in region.bounds[5]]
    return {
        "alpha": [_mp_string(value) for value in sorted(alpha_values)],
        "omega": [_mp_string(value) for value in sorted(omega_values)],
    }


def _grid_seed_count(region_name: str = "near", spacing: mp.mpf = DEFAULT_GRID_SPACING, shift: str = DEFAULT_GRID_SHIFT) -> int:
    """Return the full seed count for one grid before any debugging limit."""
    axes = _grid_axes(_grid_region(region_name), spacing, shift)
    count = 1
    for axis in axes:
        count *= len(axis)
    return count


def _grid_metadata(region_name: str, spacing: mp.mpf, limit: int | None = None, shift: str = DEFAULT_GRID_SHIFT) -> dict:
    """Return JSON-ready metadata for one grid recipe."""
    region = _grid_region(region_name)
    axes = _grid_axes(region, spacing, shift)
    full_seed_count = _grid_seed_count(region_name, spacing, shift)
    seed_count = min(full_seed_count, limit) if limit is not None else full_seed_count
    return {
        "region": region.name,
        "shift": shift,
        "base_params": _base_params_payload(_base_params_for_region(region.name)),
        "coordinate_names": list(COORDINATE_NAMES),
        "bounds": [[_mp_string(low), _mp_string(high)] for low, high in region.bounds],
        "physical_odd_bounds": _physical_odd_bounds(region),
        "max_grid_spacing": _mp_string(spacing),
        "calibrated_recovery_radius": _mp_string(CALIBRATED_RECOVERY_RADIUS),
        "axis_counts": [len(axis) for axis in axes],
        "axis_spacings": [_mp_string(_axis_spacing(axis)) for axis in axes],
        "full_seed_count": full_seed_count,
        "seed_count": seed_count,
        "limit": limit,
    }


def _grid_seeds(
    region_name: str = "near",
    spacing: mp.mpf = DEFAULT_GRID_SPACING,
    limit: int | None = None,
    shift: str = DEFAULT_GRID_SHIFT,
) -> list[SearchSeed]:
    """Return deterministic grid seeds in stable coordinate order."""
    with mp.workdps(max(mp.dps, 80)):
        region = _grid_region(region_name)
        axes = _grid_axes(region, spacing, shift)
        seeds = []
        for index, values in enumerate(product(*axes)):
            if limit is not None and index >= limit:
                break
            seeds.append(SearchSeed(index, region.name, "calibrated_grid", _point_from_values(values)))
        return seeds


def _scout_config_payload(region_name: str = "near") -> dict:
    """Return JSON-ready scout solver settings."""
    scout_config = _scout_config_for_region(region_name)
    return {
        "series_order": scout_config.series_order,
        "working_dps": scout_config.working_dps,
        "target_dps": scout_config.target_dps,
        "step_safety": _mp_string(scout_config.step_safety),
        "sample_points": scout_config.sample_points,
        "match_t": _mp_string(scout_config.match_t),
    }


def _output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped output paths for the grid search."""
    return _common_output_paths(OUTPUT_DIR, OUTPUT_SUFFIX, now)


def _summary_path_for_jsonl(path: Path) -> Path:
    """Return the summary path paired with one grid checkpoint."""
    return path.with_name(f"{path.stem}-summary.json")


def _run_start_payload(
    jsonl_path: Path,
    summary_path: Path,
    region_name: str = "near",
    spacing: mp.mpf = DEFAULT_GRID_SPACING,
    limit: int | None = None,
    shift: str = DEFAULT_GRID_SHIFT,
) -> dict:
    """Return JSON-ready metadata identifying one resumable grid run."""
    return {
        "random_seed": RANDOM_SEED,
        "search_version": SEARCH_VERSION,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "grid": _grid_metadata(region_name, spacing, limit, shift),
        "scout_config": _scout_config_payload(region_name),
    }


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
    """Return whether one checkpoint contains an event type."""
    return any(event.get("event") == event_type for event in _jsonl_events(path))


def _checkpoint_is_compatible(
    path: Path,
    region_name: str = "near",
    spacing: mp.mpf = DEFAULT_GRID_SPACING,
    limit: int | None = None,
    shift: str = DEFAULT_GRID_SHIFT,
) -> bool:
    """Return whether one incomplete checkpoint can be resumed."""
    if _jsonl_has_event(path, "run_summary"):
        return False
    starts = [event for event in _jsonl_events(path) if event.get("event") == "run_start"]
    if not starts:
        return False
    expected = _run_start_payload(path, _summary_path_for_jsonl(path), region_name, spacing, limit, shift)
    ignored = {"jsonl_path", "summary_path"}
    return all(starts[-1].get(key) == value for key, value in expected.items() if key not in ignored)


def _latest_incomplete_checkpoint(
    region_name: str = "near",
    spacing: mp.mpf = DEFAULT_GRID_SPACING,
    limit: int | None = None,
    shift: str = DEFAULT_GRID_SHIFT,
) -> Path | None:
    """Return the newest compatible incomplete checkpoint, if any."""
    candidates = sorted(OUTPUT_DIR.glob(f"*-seed{RANDOM_SEED}-{OUTPUT_SUFFIX}.jsonl"), reverse=True)
    return next((path for path in candidates if _checkpoint_is_compatible(path, region_name, spacing, limit, shift)), None)


def _resume_or_new_paths(
    *,
    region_name: str = "near",
    spacing: mp.mpf = DEFAULT_GRID_SPACING,
    limit: int | None = None,
    shift: str = DEFAULT_GRID_SHIFT,
    resume: bool = True,
    now: datetime | None = None,
) -> tuple[Path, Path, bool]:
    """Return output paths, resuming a compatible incomplete checkpoint when possible."""
    if resume and now is None:
        checkpoint = _latest_incomplete_checkpoint(region_name, spacing, limit, shift)
        if checkpoint is not None:
            return checkpoint, _summary_path_for_jsonl(checkpoint), True
    jsonl_path, summary_path = _output_paths(now)
    return jsonl_path, summary_path, False


def _completed_seed_indices(path: Path) -> set[int]:
    """Return scout seed indices already persisted in one checkpoint."""
    return {int(event["seed_index"]) for event in _jsonl_events(path) if event.get("event") == "scout_result"}


def _scout_payloads(path: Path) -> list[dict]:
    """Return all completed scout payloads from a checkpoint."""
    return [event for event in _jsonl_events(path) if event.get("event") == "scout_result"]


def _evaluate_seed_payload(seed: SearchSeed) -> dict:
    """Evaluate one grid scout seed and return a JSON-ready payload."""
    base_params = _base_params_for_region(seed.region)
    scout_config = _scout_config_for_region(seed.region)
    with mp.workdps(scout_config.working_dps):
        if base_params == DEFAULT_PARAMS:
            result = two_sided_residual(seed.point, scout_config)
        else:
            result = two_sided_residual(seed.point, scout_config, base_params=base_params)
    return _candidate_payload(SearchCandidate(seed, result))


def _evaluate_seed_payloads(
    seeds: list[SearchSeed],
    workers: int,
    chunksize: int | None = None,
) -> Iterable[dict]:
    """Yield JSON-ready scout payloads, using worker processes when requested."""
    if workers <= 1:
        for seed in seeds:
            yield _evaluate_seed_payload(seed)
        return
    actual_chunksize = chunksize or 8
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(_evaluate_seed_payload, seeds, chunksize=actual_chunksize)


def _run_scouts(
    seeds: list[SearchSeed],
    jsonl_path: Path,
    workers: int,
    *,
    progress_every: int = 1000,
    chunksize: int | None = None,
) -> None:
    """Evaluate missing scout seeds and append results to the checkpoint."""
    completed = _completed_seed_indices(jsonl_path)
    pending = [seed for seed in seeds if seed.index not in completed]
    total = len(seeds)
    print(f"loaded completed scouts: {len(completed)}/{total}", flush=True)
    print(f"pending scouts: {len(pending)} with workers={workers}", flush=True)
    if not pending:
        return
    done = len(completed)
    for payload in _evaluate_seed_payloads(pending, workers, chunksize):
        _write_jsonl_event(jsonl_path, _event("scout_result", payload))
        done += 1
        if progress_every and (done == total or done % progress_every == 0):
            print(f"scouts complete: {done}/{total}", flush=True)


def _payload_success(payload: dict) -> bool:
    """Return whether one scout payload is branch-valid."""
    return payload["result"]["failure"] is None


def _payload_norm(payload: dict) -> mp.mpf:
    """Return the residual norm for sorting one scout payload."""
    return mp.mpf(payload["result"]["residual_norm"])


def _compact_scout_payload(payload: dict) -> dict:
    """Return a compact summary record for one scout result."""
    return {
        "seed_index": payload["seed_index"],
        "region": payload["region"],
        "source": payload["source"],
        "distance": payload["distance"],
        "asymmetry": payload["asymmetry"],
        "residual_norm": payload["result"]["residual_norm"],
        "failure": payload["result"]["failure"],
        "seed_point": payload["seed_point"],
    }


def _scout_summary(payloads: list[dict]) -> dict[str, dict[str, int]]:
    """Return scout success/failure counts by grid region."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for payload in payloads:
        counts[payload["region"]]["total"] += 1
        counts[payload["region"]]["successes" if _payload_success(payload) else "failures"] += 1
    return {region: dict(counter) for region, counter in sorted(counts.items())}


def _summary_payload(
    jsonl_path: Path,
    jsonl_metadata: dict,
    *,
    best_limit: int = TOP_SCOUTS,
) -> dict:
    """Return a compact JSON-ready summary for completed scout results."""
    payloads = _scout_payloads(jsonl_path)
    successes = [payload for payload in payloads if _payload_success(payload)]
    failures = [payload for payload in payloads if not _payload_success(payload)]
    best = sorted(successes, key=_payload_norm)[:best_limit]
    failure_messages = Counter(payload["result"]["failure"] for payload in failures)
    return {
        "random_seed": RANDOM_SEED,
        "search_version": SEARCH_VERSION,
        "grid": jsonl_metadata["grid"],
        "scout_config": jsonl_metadata["scout_config"],
        "scout_count": len(payloads),
        "scout_summary": _scout_summary(payloads),
        "classification_counts": {"scout_success": len(successes), "scout_failure": len(failures)},
        "failure_messages": dict(failure_messages),
        "best_scouts": [_compact_scout_payload(payload) for payload in best],
    }


def _print_dry_run(region_name: str, spacing: mp.mpf, limit: int | None, shift: str) -> None:
    """Print grid size and rough runtime estimates without creating files."""
    metadata = _grid_metadata(region_name, spacing, limit, shift)
    count = metadata["seed_count"]
    scout_config = _scout_config_for_region(region_name)
    per_point = (mp.mpf("2.5"), mp.mpf("3.5")) if region_name.startswith("mixed-mu") else (mp.mpf("0.65"), mp.mpf("0.9"))
    print(f"region: {region_name}", flush=True)
    print(f"shift: {shift}", flush=True)
    print(f"max spacing: {mp.nstr(spacing, 12)}", flush=True)
    print(f"axis counts: {metadata['axis_counts']}", flush=True)
    print(
        f"scout config: order={scout_config.series_order}, dps={scout_config.working_dps}, step_safety={mp.nstr(scout_config.step_safety, 12)}",
        flush=True,
    )
    print(f"scout points: {count}", flush=True)
    print(
        f"serial scout estimate at {float(per_point[0])}-{float(per_point[1])} s/point: "
        f"{float(count * per_point[0] / 3600):.2f}-{float(count * per_point[1] / 3600):.2f} h",
        flush=True,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the grid scout runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=_positive_int, default=_default_workers(), help="parallel worker processes")
    parser.add_argument("--spacing", type=_positive_mpf, default=DEFAULT_GRID_SPACING, help="maximum grid spacing")
    parser.add_argument("--shift", choices=GRID_SHIFTS, default=DEFAULT_GRID_SHIFT, help="grid placement policy")
    parser.add_argument("--region", choices=sorted(GRID_REGIONS), default="near", help="grid region to scout")
    parser.add_argument("--limit", type=_nonnegative_int, default=None, help="debugging cap on generated seeds")
    parser.add_argument("--dry-run", action="store_true", help="print grid size and runtime estimate without running")
    parser.add_argument("--no-resume", action="store_true", help="start a fresh checkpoint instead of resuming")
    parser.add_argument("--progress-every", type=_positive_int, default=1000, help="print progress every N completed scouts")
    parser.add_argument("--chunksize", type=_positive_int, default=None, help="multiprocessing map chunk size")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the calibrated non-mirrored near-grid scout search."""
    args = _parse_args(argv)
    if args.dry_run:
        _print_dry_run(args.region, args.spacing, args.limit, args.shift)
        return

    jsonl_path, summary_path, resumed = _resume_or_new_paths(
        region_name=args.region,
        spacing=args.spacing,
        limit=args.limit,
        shift=args.shift,
        resume=not args.no_resume,
    )
    metadata = _run_start_payload(jsonl_path, summary_path, args.region, args.spacing, args.limit, args.shift)
    if resumed:
        print(f"resuming JSONL checkpoint {jsonl_path}", flush=True)
    else:
        print(f"writing JSONL events to {jsonl_path}", flush=True)
        _write_jsonl_event(jsonl_path, _event("run_start", metadata))

    seeds = _grid_seeds(args.region, args.spacing, args.limit, args.shift)
    print(f"grid seeds: {len(seeds)}", flush=True)
    _run_scouts(seeds, jsonl_path, args.workers, progress_every=args.progress_every, chunksize=args.chunksize)
    payload = _summary_payload(jsonl_path, metadata)
    print(f"scout summary: {payload['scout_summary']}", flush=True)
    if payload["best_scouts"]:
        best = payload["best_scouts"][0]
        print(f"best scout: seed={best['seed_index']} norm={best['residual_norm']}", flush=True)
    _write_jsonl_event(jsonl_path, _event("run_summary", payload))
    _write_summary(summary_path, payload)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
