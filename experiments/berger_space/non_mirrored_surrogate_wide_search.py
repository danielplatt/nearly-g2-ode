"""Wide XGBoost-assisted non-mirrored search with vector residual heads."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mpmath import mp

from problem import DEFAULT_CONFIG, SolverConfig
from solver.two_sided_refinement import TwoSidedNewtonSettings
from solver.two_sided_shooting import BASE_TWO_SIDED_POINT

from ..shared.non_mirrored_common import RANDOM_SEED, S_MIN, RegionSpec, _event, _region_summary, _result_payload, _track_final_result, _track_payload, _verify_point, _write_jsonl_event, _write_summary
from ..shared.non_mirrored_surrogate_common import (
    FEATURE_NAMES,
    _acquire_surrogate_proposals,
    _best_payload,
    _best_stable_by_region,
    _classification_counts,
    _evaluate_proposals,
    _evaluate_training_set,
    _has_event,
    _load_candidates_from_jsonl,
    _load_classified_tracks_from_jsonl,
    _load_proposals_from_jsonl,
    _load_refinement_tracks_from_jsonl,
    _model_metrics_payload,
    _physical_payloads,
    _promote_track_with_timeout,
    _select_refinement_candidates_by_region,
    _train_models,
    _training_seeds as _common_training_seeds,
    _initial_track_with_timeout,
)


OUTPUT_DIR = Path("output/non_mirrored_surrogate_wide")
SCOUT_CONFIG = SolverConfig(4, 30, 15, mp.mpf("0.95"), 0, DEFAULT_CONFIG.match_t)
ORDER6_CONFIG = SolverConfig(6, 40, 20, mp.mpf("0.9"), 0, DEFAULT_CONFIG.match_t)
ORDER10_CONFIG = SolverConfig(10, 70, 30, mp.mpf("0.7"), 1, DEFAULT_CONFIG.match_t)
VERIFY14_CONFIG = SolverConfig(14, 90, 35, mp.mpf("0.6"), 2, DEFAULT_CONFIG.match_t)
VERIFY18_CONFIG = SolverConfig(18, 110, 45, mp.mpf("0.55"), 2, DEFAULT_CONFIG.match_t)
VERIFY_CONFIGS = (VERIFY14_CONFIG, VERIFY18_CONFIG)

BRANCH_PROBABILITY_CUTOFF = 0.65
OPTIMISM_WEIGHT = 0.75
ACQUISITION_CHUNK_SIZE = 50_000
ORDER6_TIMEOUT_SECONDS = 12 * 60
PROMOTION_TIMEOUT_SECONDS = 20 * 60

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

TRAINING_REGIONS = (
    RegionSpec("near_control", ((-0.8, 0.8), (-0.8, 0.8), (-2, 2), (-0.8, 0.8), (-0.8, 0.8), (-2, 2), (-1, 1)), 1000, 0, 0, 0, mp.mpf("0.05"), mp.mpf("1.5")),
    RegionSpec("medium_asymmetric", ((-2.5, 2.5), (-2.5, 2.5), (-5, 5), (-2.5, 2.5), (-2.5, 2.5), (-5, 5), (S_MIN + mp.mpf("0.05"), 2)), 3500, 0, 0, 0, mp.mpf("1"), mp.mpf("4"), mp.mpf("0.75")),
    RegionSpec("far_asymmetric", ((-4.5, 4.5), (-4.5, 4.5), (-8, 8), (-4.5, 4.5), (-4.5, 4.5), (-8, 8), (S_MIN + mp.mpf("0.05"), 2.5)), 4500, 0, 0, 0, mp.mpf("3"), mp.mpf("8"), mp.mpf("1.0")),
    RegionSpec("wide_asymmetric", ((-8, 8), (-8, 8), (-12, 12), (-8, 8), (-8, 8), (-12, 12), (S_MIN + mp.mpf("0.05"), 3)), 4500, 0, 0, 0, mp.mpf("5"), mp.mpf("12"), mp.mpf("1.5")),
    RegionSpec("alpha_wide", ((-3, 3), (-3, 3), (-12, 12), (-3, 3), (-3, 3), (-12, 12), (S_MIN + mp.mpf("0.05"), 2)), 3000, 0, 0, 0, mp.mpf("3"), mp.mpf("12"), mp.mpf("1.0")),
    RegionSpec("m_wide", ((-4, 4), (-4, 4), (-8, 8), (-4, 4), (-4, 4), (-8, 8), (S_MIN + mp.mpf("0.05"), 3.5)), 2500, 0, 0, 0, mp.mpf("3"), mp.mpf("8"), mp.mpf("1.0")),
    RegionSpec("tail_negative_uv", ((-10, -4), (-10, -4), (-10, 10), (-10, -4), (-10, -4), (-10, 10), (-2.5, 1)), 1000, 0, 0, 0, mp.mpf("4"), mp.mpf("12"), mp.mpf("1.0")),
)

ACQUISITION_REGIONS = (
    RegionSpec("near_control", TRAINING_REGIONS[0].ranges, 250_000, 0, 0, 0, TRAINING_REGIONS[0].min_distance, TRAINING_REGIONS[0].max_distance, TRAINING_REGIONS[0].min_asymmetry),
    RegionSpec("medium_asymmetric", TRAINING_REGIONS[1].ranges, 650_000, 0, 0, 0, TRAINING_REGIONS[1].min_distance, TRAINING_REGIONS[1].max_distance, TRAINING_REGIONS[1].min_asymmetry),
    RegionSpec("far_asymmetric", TRAINING_REGIONS[2].ranges, 900_000, 0, 0, 0, TRAINING_REGIONS[2].min_distance, TRAINING_REGIONS[2].max_distance, TRAINING_REGIONS[2].min_asymmetry),
    RegionSpec("wide_asymmetric", TRAINING_REGIONS[3].ranges, 1_050_000, 0, 0, 0, TRAINING_REGIONS[3].min_distance, TRAINING_REGIONS[3].max_distance, TRAINING_REGIONS[3].min_asymmetry),
    RegionSpec("alpha_wide", TRAINING_REGIONS[4].ranges, 650_000, 0, 0, 0, TRAINING_REGIONS[4].min_distance, TRAINING_REGIONS[4].max_distance, TRAINING_REGIONS[4].min_asymmetry),
    RegionSpec("m_wide", TRAINING_REGIONS[5].ranges, 400_000, 0, 0, 0, TRAINING_REGIONS[5].min_distance, TRAINING_REGIONS[5].max_distance, TRAINING_REGIONS[5].min_asymmetry),
    RegionSpec("tail_negative_uv", TRAINING_REGIONS[6].ranges, 100_000, 0, 0, 0, TRAINING_REGIONS[6].min_distance, TRAINING_REGIONS[6].max_distance, TRAINING_REGIONS[6].min_asymmetry),
)

PROPOSAL_QUOTAS = {
    "near_control": 50,
    "medium_asymmetric": 120,
    "far_asymmetric": 170,
    "wide_asymmetric": 190,
    "alpha_wide": 140,
    "m_wide": 100,
    "tail_negative_uv": 30,
}
REFINE_QUOTAS = {
    "near_control": 4,
    "medium_asymmetric": 12,
    "far_asymmetric": 18,
    "wide_asymmetric": 20,
    "alpha_wide": 12,
    "m_wide": 10,
    "tail_negative_uv": 4,
}
PROMOTE_QUOTAS = {
    "near_control": 1,
    "medium_asymmetric": 5,
    "far_asymmetric": 7,
    "wide_asymmetric": 8,
    "alpha_wide": 5,
    "m_wide": 4,
    "tail_negative_uv": 2,
}


def _output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped JSONL and summary paths for the V2 sweep."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    stem = f"{stamp}-seed{RANDOM_SEED}-v2"
    return OUTPUT_DIR / f"{stem}-training.jsonl", OUTPUT_DIR / f"{stem}-summary.json"


def _resume_or_new_paths() -> tuple[Path, Path, bool]:
    """Return the newest incomplete checkpoint or a fresh V2 output path."""
    candidates = sorted(OUTPUT_DIR.glob(f"*-seed{RANDOM_SEED}-v2-training.jsonl"))
    for jsonl_path in reversed(candidates):
        summary_path = jsonl_path.with_name(jsonl_path.name.replace("-training.jsonl", "-summary.json"))
        if not summary_path.exists() and not _has_event(jsonl_path, "run_summary"):
            return jsonl_path, summary_path, True
    jsonl_path, summary_path = _output_paths()
    return jsonl_path, summary_path, False


def _training_seeds(seed: int = RANDOM_SEED):
    """Return the deterministic 20,000-point V2 training seed set."""
    return _common_training_seeds(TRAINING_REGIONS, seed)


def _promote_tracks_by_region(tracks, refs, path):
    """Promote the best order-6 tracks using per-region quotas."""
    existing = {track.seed_rank: track for track in _load_classified_tracks_from_jsonl(path, _settings_by_order())}
    promoted = []
    for region, quota in PROMOTE_QUOTAS.items():
        region_tracks = [track for track in tracks if track.seed_region == region]
        region_tracks.sort(key=lambda track: _track_final_result(track).residual_norm)
        for track in region_tracks[:quota]:
            if track.seed_rank in existing:
                promoted.append(existing[track.seed_rank])
            else:
                promoted.append(_promote_track_with_timeout(track, refs, path, ORDER10_SETTINGS, VERIFY_CONFIGS, PROMOTION_TIMEOUT_SECONDS))
    return promoted


def main() -> None:
    """Run the wide XGBoost-assisted non-mirrored search."""
    jsonl_path, summary_path, resumed = _resume_or_new_paths()
    print(f"{'resuming' if resumed else 'writing'} V2 training JSONL at {jsonl_path}", flush=True)
    if not resumed:
        _write_jsonl_event(jsonl_path, _event("run_start", _run_start_payload()))

    existing_training = _load_candidates_from_jsonl(jsonl_path, "training_sample")
    print(f"  loaded existing training samples: {len(existing_training)}/20000", flush=True)
    training_candidates = _evaluate_training_set(_training_seeds(), jsonl_path, SCOUT_CONFIG, "training samples evaluated", existing_training)
    initial_models = _train_models(training_candidates, seed=RANDOM_SEED, branch_cutoff=BRANCH_PROBABILITY_CUTOFF, include_vector_models=True)
    _write_jsonl_event(jsonl_path, _event("model_metrics", _model_metrics_payload(initial_models, "initial")))

    proposals = _load_or_acquire_proposals(initial_models, jsonl_path)
    existing_proposals = _load_candidates_from_jsonl(jsonl_path, "true_proposal_evaluation")
    print(f"  loaded existing true proposal evaluations: {len(existing_proposals)}/{sum(PROPOSAL_QUOTAS.values())}", flush=True)
    proposal_candidates = _evaluate_proposals(proposals, jsonl_path, SCOUT_CONFIG, existing_proposals)
    final_models = _train_models(training_candidates + proposal_candidates, seed=RANDOM_SEED + 1, branch_cutoff=BRANCH_PROBABILITY_CUTOFF, include_vector_models=True)
    _write_jsonl_event(jsonl_path, _event("model_metrics", _model_metrics_payload(final_models, "after_proposals")))

    refs = _verify_point(BASE_TWO_SIDED_POINT, VERIFY_CONFIGS)
    selections = _select_refinement_candidates_by_region(proposal_candidates, REFINE_QUOTAS)
    tracks = _load_or_run_order6_tracks(selections, jsonl_path)
    promoted = _promote_tracks_by_region(tracks, refs, jsonl_path)
    _write_outputs(summary_path, jsonl_path, training_candidates, proposal_candidates, initial_models, final_models, refs, promoted)
    print(f"summary written to {summary_path}", flush=True)


def _load_or_acquire_proposals(initial_models, jsonl_path):
    """Load selected proposals from checkpoint or acquire them from models."""
    proposals = _load_proposals_from_jsonl(jsonl_path)
    if len(proposals) >= sum(PROPOSAL_QUOTAS.values()):
        print(f"  loaded existing surrogate proposals: {len(proposals)}/{sum(PROPOSAL_QUOTAS.values())}", flush=True)
        return proposals
    if proposals:
        print(f"  resuming surrogate proposal selection: {len(proposals)}/{sum(PROPOSAL_QUOTAS.values())}", flush=True)
    return _acquire_surrogate_proposals(initial_models, jsonl_path, ACQUISITION_REGIONS, PROPOSAL_QUOTAS, chunk_size=ACQUISITION_CHUNK_SIZE, branch_cutoff=BRANCH_PROBABILITY_CUTOFF, optimism_weight=OPTIMISM_WEIGHT, existing=proposals)


def _load_or_run_order6_tracks(selections, jsonl_path):
    """Load completed order-6 tracks and run only missing selections."""
    existing = {track.seed_rank: track for track in _load_refinement_tracks_from_jsonl(jsonl_path, _settings_by_order())}
    if existing:
        print(f"  loaded existing order-6 refinement tracks: {len(existing)}/{sum(REFINE_QUOTAS.values())}", flush=True)
    tracks = []
    for selection in selections:
        seed_index = selection.candidate.seed.index
        if seed_index in existing:
            tracks.append(existing[seed_index])
        else:
            tracks.append(_initial_track_with_timeout(selection, jsonl_path, ORDER6_SETTINGS, ORDER6_TIMEOUT_SECONDS))
    return tracks


def _settings_by_order() -> dict[int, TwoSidedNewtonSettings]:
    """Return refinement settings keyed by Taylor order."""
    return {ORDER6_CONFIG.series_order: ORDER6_SETTINGS, ORDER10_CONFIG.series_order: ORDER10_SETTINGS}


def _run_start_payload() -> dict:
    """Return the V2 run-start metadata."""
    return {
        "random_seed": RANDOM_SEED,
        "sweep_version": "non_mirrored_surrogate_v2",
        "feature_names": FEATURE_NAMES,
        "training_regions": _region_specs_payload(TRAINING_REGIONS),
        "acquisition_regions": _region_specs_payload(ACQUISITION_REGIONS),
        "proposal_quotas": PROPOSAL_QUOTAS,
        "refine_quotas": REFINE_QUOTAS,
        "promote_quotas": PROMOTE_QUOTAS,
        "timeouts_seconds": {"order6": ORDER6_TIMEOUT_SECONDS, "promotion": PROMOTION_TIMEOUT_SECONDS},
    }


def _region_specs_payload(regions: tuple[RegionSpec, ...]) -> list[dict]:
    """Return JSON-ready region metadata."""
    return [
        {
            "name": spec.name,
            "samples": spec.samples,
            "min_distance": str(spec.min_distance),
            "max_distance": str(spec.max_distance),
            "min_asymmetry": str(spec.min_asymmetry),
        }
        for spec in regions
    ]


def _write_outputs(summary_path, jsonl_path, training, proposals, initial_models, final_models, refs, promoted) -> None:
    """Persist final V2 JSON summary and run-summary event."""
    payload = {
        "training_summary": _region_summary(training),
        "proposal_summary": _region_summary(proposals),
        "model_metrics": {"initial": initial_models.metrics, "after_proposals": final_models.metrics},
        "best_training_candidates": _best_payload(training),
        "best_proposal_candidates": _best_payload(proposals),
        "symmetric_references": [_result_payload(result) for result in refs],
        "classification_counts": _classification_counts(promoted),
        "best_stable_by_region": _best_stable_by_region(promoted),
        "tracks": [_track_payload(track) for track in promoted],
        "physical_parameters": _physical_payloads(promoted),
    }
    _write_jsonl_event(jsonl_path, _event("run_summary", payload))
    _write_summary(summary_path, payload)


if __name__ == "__main__":
    main()
