"""Local grid refinement around the best V3 near-floor candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

from mpmath import mp

from problem import DEFAULT_CONFIG, DEFAULT_PARAMS, SolverConfig
from solver.mirror_shooting import MirrorResidualResult, MirrorSearchPoint, mirror_residual, params_from_scaled

from ..shared.mirror_sweep_common import _event, _mp_string, _output_paths, _point_payload, _result_payload, _with_timeout, _write_jsonl_event
from .mirror_sweep_v3 import MIN_MATCH_T, S_MIN, _physical_payload


@dataclass(frozen=True)
class GridCandidate:
    """One V3 candidate center for local grid refinement."""

    seed_index: int
    label: str
    point: MirrorSearchPoint


OUTPUT_DIR = Path("output/mirror_local_grids")
GRID_TIMEOUT_SECONDS = 120
VERIFY_TOP_PER_CANDIDATE = 5
GRID_CONFIG = SolverConfig(10, 70, 30, mp.mpf("0.7"), 1, DEFAULT_CONFIG.match_t)
VERIFY_CONFIGS = (
    SolverConfig(14, 90, 35, mp.mpf("0.6"), 2, DEFAULT_CONFIG.match_t),
    SolverConfig(18, 110, 45, mp.mpf("0.55"), 2, DEFAULT_CONFIG.match_t),
)
UVR_OFFSETS = (mp.mpf("-0.25"), mp.zero, mp.mpf("0.25"))
S_FLOOR_OFFSETS = (mp.mpf("0.002"), mp.mpf("0.005"), mp.mpf("0.01"), mp.mpf("0.02"))
S_CENTER_OFFSETS = (mp.zero, mp.mpf("0.025"), mp.mpf("0.05"), mp.mpf("0.1"))


def _point(u: str, v: str, r: str, s: str) -> MirrorSearchPoint:
    """Build one high-precision search point from decimal strings."""
    with mp.workdps(90):
        return MirrorSearchPoint(mp.mpf(u), mp.mpf(v), mp.mpf(r), mp.mpf(s))


CANDIDATES = (
    GridCandidate(
        13407,
        "best-v3-mixed-far",
        _point(
            "-8.34611856286786490954909822903573513031005859375",
            "-9.3767479287648978214519956964068114757537841796875",
            "-1.1612963984966189201486486126668751239776611328125",
            "-3.93750947418776764408221424673683941364288330078125",
        ),
    ),
    GridCandidate(
        8604,
        "best-v3-negative-large-m",
        _point(
            "-7.1630714236140912409344908828153811385248476208373208962875898970880054645871611",
            "-8.1418242184040801661077243819841119986395585875250069797853772743238329634303541",
            "-0.79409165583605334550852397501798455916877865474989064266026884125249167151884343",
            "-3.9339863679102288446277097755632232879032406326223524696865098960224142519628859",
        ),
    ),
)


def _local_output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped local-grid JSONL and summary paths."""
    return _output_paths(OUTPUT_DIR, 1729, "-v3-local-grid", now)


def _s_values(center: MirrorSearchPoint) -> tuple[mp.mpf, ...]:
    """Return floor-aware absolute s-values for one local grid."""
    values = [S_MIN + offset for offset in S_FLOOR_OFFSETS]
    values.extend(center.s + offset for offset in S_CENTER_OFFSETS)
    unique = {_mp_string(value): value for value in values if value > S_MIN}
    return tuple(sorted(unique.values()))


def _grid_points(center: MirrorSearchPoint) -> tuple[MirrorSearchPoint, ...]:
    """Return all local grid points around one candidate center."""
    points = []
    for du, dv, dr, s in product(UVR_OFFSETS, UVR_OFFSETS, UVR_OFFSETS, _s_values(center)):
        points.append(MirrorSearchPoint(center.u + du, center.v + dv, center.r + dr, s))
    return tuple(points)


def _timeout_result(point: MirrorSearchPoint, config: SolverConfig, message: str) -> MirrorResidualResult:
    """Return a synthetic residual for a timed-out local evaluation."""
    params, local_config = params_from_scaled(point, template_config=config)
    return MirrorResidualResult(point, params, local_config, (), mp.inf, None, None, 0, {}, message)


def _evaluate(point: MirrorSearchPoint, config: SolverConfig) -> MirrorResidualResult:
    """Evaluate one residual with a wall-clock timeout."""
    try:
        with mp.workdps(config.working_dps):
            return _with_timeout(GRID_TIMEOUT_SECONDS, f"local grid evaluation exceeded {GRID_TIMEOUT_SECONDS} seconds", lambda: mirror_residual(point, config))
    except TimeoutError as exc:
        return _timeout_result(point, config, str(exc))


def _grid_event(candidate: GridCandidate, index: int, result: MirrorResidualResult) -> dict:
    """Return one JSONL event for a local grid residual."""
    payload = {
        "candidate_seed": candidate.seed_index,
        "candidate_label": candidate.label,
        "grid_index": index,
        "distance_from_center": _mp_string(_point_distance(candidate.point, result.point)),
        "result": _result_payload(result),
        "physical": _physical_payload(result.point),
    }
    return _event("grid_result", payload)


def _point_distance(left: MirrorSearchPoint, right: MirrorSearchPoint) -> mp.mpf:
    """Return max-distance between two scaled points."""
    return max(abs(a - b) for a, b in zip((left.u, left.v, left.r, left.s), (right.u, right.v, right.r, right.s)))


def _successful(results: list[tuple[GridCandidate, int, MirrorResidualResult]]) -> list[tuple[GridCandidate, int, MirrorResidualResult]]:
    """Return successful grid evaluations sorted by residual."""
    valid = [item for item in results if item[2].failure is None]
    return sorted(valid, key=lambda item: item[2].residual_norm)


def _verification_event(candidate: GridCandidate, grid_index: int, result: MirrorResidualResult) -> dict:
    """Return one JSONL event for a high-order verification residual."""
    payload = {
        "candidate_seed": candidate.seed_index,
        "candidate_label": candidate.label,
        "grid_index": grid_index,
        "result": _result_payload(result),
        "physical": _physical_payload(result.point),
    }
    return _event("verification", payload)


def _verify_best(results: list[tuple[GridCandidate, int, MirrorResidualResult]], path: Path) -> list[dict]:
    """Verify the best local-grid results per candidate at higher order."""
    verified = []
    for candidate in CANDIDATES:
        subset = [item for item in _successful(results) if item[0] == candidate][:VERIFY_TOP_PER_CANDIDATE]
        for _, grid_index, grid_result in subset:
            checks = [_evaluate(grid_result.point, config) for config in VERIFY_CONFIGS]
            for check in checks:
                _write_jsonl_event(path, _verification_event(candidate, grid_index, check))
            verified.append({"candidate": candidate.seed_index, "grid_index": grid_index, "grid": _result_payload(grid_result), "verifications": [_result_payload(check) for check in checks]})
    return verified


def _summary_payload(results: list[tuple[GridCandidate, int, MirrorResidualResult]], verified: list[dict]) -> dict:
    """Return the final local-grid summary payload."""
    best = []
    for candidate in CANDIDATES:
        subset = [item for item in _successful(results) if item[0] == candidate][:10]
        best.append({"candidate": candidate.seed_index, "label": candidate.label, "best_grid": [{"grid_index": index, "result": _result_payload(result), "physical": _physical_payload(result.point)} for _, index, result in subset]})
    return {"min_match_t": _mp_string(MIN_MATCH_T), "s_min": _mp_string(S_MIN), "grid_order": GRID_CONFIG.series_order, "candidates": best, "verified": verified}


def main() -> None:
    """Run the local grid search around the two selected V3 candidates."""
    jsonl_path, summary_path = _local_output_paths()
    print(f"writing local-grid JSONL events to {jsonl_path}", flush=True)
    results: list[tuple[GridCandidate, int, MirrorResidualResult]] = []
    for candidate in CANDIDATES:
        points = _grid_points(candidate.point)
        print(f"candidate {candidate.seed_index}: evaluating {len(points)} grid points", flush=True)
        for index, point in enumerate(points):
            result = _evaluate(point, GRID_CONFIG)
            results.append((candidate, index, result))
            _write_jsonl_event(jsonl_path, _grid_event(candidate, index, result))
            if (index + 1) % 25 == 0 or index + 1 == len(points):
                print(f"  processed {index + 1}/{len(points)}", flush=True)
    verified = _verify_best(results, jsonl_path)
    summary = _summary_payload(results, verified)
    _write_jsonl_event(jsonl_path, _event("run_summary", summary))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
