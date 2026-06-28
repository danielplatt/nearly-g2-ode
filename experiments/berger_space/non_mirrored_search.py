"""Seeded search for non-mirrored two-ended nearly G2 candidates."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mpmath import mp

from problem import DEFAULT_CONFIG, SolverConfig
from solver.two_sided_shooting import BASE_TWO_SIDED_POINT, params_from_two_sided_scaled
from solver.two_sided_refinement import TwoSidedNewtonSettings

from ..shared.non_mirrored_common import (
    MIN_MATCH_T,
    RANDOM_SEED,
    S_MIN,
    RegionSpec,
    SearchCandidate,
    SearchSeed,
    SelectedCandidate,
    _asymmetry_distance,
    _candidate_payload,
    _classify_track,
    _control_seeds,
    _coordinates,
    _diverse_candidates,
    _evaluate_seed as _evaluate_seed_common,
    _event,
    _in_region,
    _initial_track as _initial_track_common,
    _mp_string,
    _output_paths as _common_output_paths,
    _physical_payload,
    _point_distance,
    _point_distance_between,
    _point_from_values,
    _point_key,
    _point_payload,
    _print_best,
    _promote_track as _promote_track_common,
    _random_point,
    _region_seeds,
    _region_summary,
    _replace_track,
    _result_payload,
    _search_seeds as _common_search_seeds,
    _select_region_candidates,
    _sort_key,
    _stage_payload,
    _successful,
    _track_final_result,
    _track_payload,
    _verification_norms,
    _verify_point as _verify_point_common,
    _write_jsonl_event,
    _write_summary,
)


OUTPUT_DIR = Path("output/non_mirrored_searches")
SCOUT_CONFIG = SolverConfig(4, 30, 15, mp.mpf("0.95"), 0, DEFAULT_CONFIG.match_t)
ORDER6_CONFIG = SolverConfig(6, 40, 20, mp.mpf("0.9"), 0, DEFAULT_CONFIG.match_t)
ORDER10_CONFIG = SolverConfig(10, 70, 30, mp.mpf("0.7"), 1, DEFAULT_CONFIG.match_t)
VERIFY14_CONFIG = SolverConfig(14, 90, 35, mp.mpf("0.6"), 2, DEFAULT_CONFIG.match_t)
VERIFY18_CONFIG = SolverConfig(18, 110, 45, mp.mpf("0.55"), 2, DEFAULT_CONFIG.match_t)
VERIFY_CONFIGS = (VERIFY14_CONFIG, VERIFY18_CONFIG)

ORDER6_SETTINGS = TwoSidedNewtonSettings(
    "order-6",
    ORDER6_CONFIG,
    mp.mpf("1e-3"),
    mp.mpf("1e-8"),
    3,
    max_abs_coordinate=mp.mpf("12"),
    min_s_coordinate=S_MIN,
)
ORDER10_SETTINGS = TwoSidedNewtonSettings(
    "order-10",
    ORDER10_CONFIG,
    mp.mpf("3e-4"),
    mp.mpf("1e-10"),
    3,
    max_abs_coordinate=mp.mpf("12"),
    min_s_coordinate=S_MIN,
)

REGIONS = (
    RegionSpec("near", ((-0.6, 0.6), (-0.6, 0.6), (-1.5, 1.5), (-0.6, 0.6), (-0.6, 0.6), (-1.5, 1.5), (-0.8, 0.8)), 80, 4, 4, 2, mp.mpf("0.05"), mp.mpf("1.5")),
    RegionSpec("medium", ((-1.5, 1.5), (-1.5, 1.5), (-3, 3), (-1.5, 1.5), (-1.5, 1.5), (-3, 3), (-1.5, 1.5)), 120, 6, 6, 3, mp.mpf("1"), mp.mpf("3")),
    RegionSpec("far", ((-3, 3), (-3, 3), (-6, 6), (-3, 3), (-3, 3), (-6, 6), (-2.5, 2)), 120, 6, 6, 3, mp.mpf("3"), mp.mpf("6")),
    RegionSpec("asymmetric", ((-2, 2), (-2, 2), (-4, 4), (-2, 2), (-2, 2), (-4, 4), (-1.5, 1.5)), 160, 8, 8, 4, mp.mpf("0.5"), mp.mpf("4"), mp.mpf("0.75")),
)


def _search_seeds(seed: int = RANDOM_SEED) -> list[SearchSeed]:
    """Return the full deterministic non-mirrored scout seed list."""
    return _common_search_seeds(REGIONS, seed)


def _output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped JSONL and summary output paths."""
    return _common_output_paths(OUTPUT_DIR, "non-mirrored", now)


def _evaluate_seed(seed: SearchSeed, path: Path, config: SolverConfig = SCOUT_CONFIG) -> SearchCandidate:
    """Evaluate and persist one scout seed."""
    return _evaluate_seed_common(seed, path, config)


def _verify_point(point) -> tuple:
    """Evaluate one point at high-order verification configs."""
    return _verify_point_common(point, VERIFY_CONFIGS)


def _initial_track(selection: SelectedCandidate, path: Path):
    """Run order-6 refinement for one selected scout."""
    return _initial_track_common(selection, path, ORDER6_SETTINGS)


def _promote_track(track, refs: tuple, path: Path):
    """Run order-10 refinement and high-order verification for one track."""
    return _promote_track_common(track, refs, path, ORDER10_SETTINGS, VERIFY_CONFIGS)


def main() -> None:
    """Run the non-mirrored seeded search with JSON checkpointing."""
    jsonl_path, summary_path = _output_paths()
    print(f"writing JSONL to {jsonl_path}", flush=True)
    _write_jsonl_event(jsonl_path, _event("run_start", {"random_seed": RANDOM_SEED, "regions": [spec.name for spec in REGIONS]}))
    candidates = [_evaluate_seed(seed, jsonl_path) for seed in _search_seeds()]
    summary = _region_summary(candidates)
    print("scout summary:", summary, flush=True)
    _print_best(candidates)
    refs = _verify_point(BASE_TWO_SIDED_POINT)
    selected = [item for spec in REGIONS for item in _select_region_candidates(spec, candidates)]
    tracks = [_initial_track(selection, jsonl_path) for selection in selected]
    promoted = []
    for spec in REGIONS:
        region_tracks = [track for track in tracks if track.seed_region == spec.name]
        region_tracks.sort(key=lambda track: _track_final_result(track).residual_norm)
        promoted.extend(_promote_track(track, refs, jsonl_path) for track in region_tracks[: spec.promote_quota])
    payload = {"scout_summary": summary, "symmetric_references": [_result_payload(result) for result in refs], "tracks": [_track_payload(track) for track in promoted]}
    _write_jsonl_event(jsonl_path, _event("run_summary", payload))
    _write_summary(summary_path, payload | {"physical_parameters": [_physical_payload(_track_final_result(track).point) for track in promoted]})
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
