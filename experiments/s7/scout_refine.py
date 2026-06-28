"""Refine selected local minima from a completed fixed-chart S7 scout."""

from __future__ import annotations

import argparse
import json
import signal
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

from mpmath import mp

from problem import SolverConfig

from ..shared.non_mirrored_common import RANDOM_SEED, _event, _mp_string, _output_paths, _write_jsonl_event, _write_summary
from . import search_common as s7


OUTPUT_DIR = Path("output/s7_scout_refinements")
REFINEMENT_VERSION = "s7-scout-refine-v1"
OUTPUT_SUFFIX = "s7-scout-refine-v1"
SELECTION_VERSION = "local-minima-v1"
DEFAULT_CANDIDATE_TIMEOUT_SECONDS = 30 * 60

with mp.workdps(80):
    DEFAULT_MAX_RESIDUAL = mp.mpf("0.15")


@dataclass(frozen=True)
class SelectedS7Candidate:
    """One selected S7 scout local minimum."""

    rank: int
    reason: str
    candidate: s7.S7ScoutCandidate


class _TimeoutExpired(Exception):
    """Raised by the signal timeout guard."""


T = TypeVar("T")


@contextmanager
def _time_limit(seconds: float | None):
    """Raise `_TimeoutExpired` if a block exceeds the requested wall time."""
    if seconds is None:
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(_TimeoutExpired()))
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _run_with_timeout(callback: Callable[[], T], timeout_seconds: float | None) -> tuple[T | None, str | None]:
    """Run one callback and convert timeout into a nonfatal status."""
    try:
        with _time_limit(timeout_seconds):
            return callback(), None
    except _TimeoutExpired:
        return None, "timeout"


def _positive_float(value: str) -> float:
    """Parse one positive float CLI argument."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_int(value: str) -> int:
    """Parse one positive integer CLI argument."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _optional_positive_mpf(value: str) -> mp.mpf | None:
    """Parse a positive mpmath value, or `none` for no cutoff."""
    if value.lower() in {"none", "all"}:
        return None
    parsed = mp.mpf(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive, or 'none'")
    return parsed


def _parse_targets(value: str) -> tuple[str, ...]:
    """Parse a comma-separated target filter."""
    targets = tuple(item.strip() for item in value.split(",") if item.strip())
    if not targets:
        raise argparse.ArgumentTypeError("must include at least one target")
    for target in targets:
        s7._target(target)
    return targets


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
    """Return whether one JSONL checkpoint contains an event type."""
    return any(event.get("event") == event_type for event in _iter_jsonl_events(path))


def _latest_completed_scout_jsonl() -> Path:
    """Return the newest completed S7 scout checkpoint."""
    pattern = f"*-seed{RANDOM_SEED}-{s7.SCOUT_SUFFIX}.jsonl"
    candidates = sorted(s7.SCOUT_OUTPUT_DIR.glob(pattern), reverse=True)
    for path in candidates:
        if _jsonl_has_event(path, "run_summary"):
            return path
    raise FileNotFoundError(f"No completed S7 scout checkpoint found under {s7.SCOUT_OUTPUT_DIR}")


def _scout_run_start(path: Path) -> dict:
    """Return the run_start event from one scout checkpoint."""
    for event in _iter_jsonl_events(path):
        if event.get("event") == "run_start":
            return event
    raise ValueError(f"{path} does not contain a run_start event")


def _point_from_payload(payload: dict) -> s7.S7SearchPoint:
    """Rebuild one scaled S7 point from persisted JSON strings."""
    return s7.S7SearchPoint(
        mp.mpf(payload["u"]),
        mp.mpf(payload["v"]),
        mp.mpf(payload["r"]),
        mp.mpf(payload.get("s", "0")),
    )


def _config_from_payload(payload: dict) -> SolverConfig:
    """Return a minimal solver config matching a serialized residual payload."""
    order = int(payload.get("config_order", s7.SCOUT_CONFIG.series_order))
    dps = int(payload.get("config_dps", s7.SCOUT_CONFIG.working_dps))
    templates = {
        s7.SCOUT_CONFIG.series_order: s7.SCOUT_CONFIG,
        s7.ORDER8_CONFIG.series_order: s7.ORDER8_CONFIG,
        s7.ORDER10_CONFIG.series_order: s7.ORDER10_CONFIG,
        s7.VERIFY14_CONFIG.series_order: s7.VERIFY14_CONFIG,
        s7.VERIFY18_CONFIG.series_order: s7.VERIFY18_CONFIG,
    }
    template = templates.get(order, s7.SCOUT_CONFIG)
    return SolverConfig(order, dps, template.target_dps, template.step_safety, template.sample_points, template.match_t)


def _result_from_payload(payload: dict, target_name: str, region: str = s7.DEFAULT_SCOUT_REGION.name) -> s7.S7ResidualResult:
    """Rebuild one residual result from a persisted scout payload."""
    point = _point_from_payload(payload["point"])
    config = _config_from_payload(payload)
    params, local_config = s7.params_from_s7_scaled(
        point,
        base_params=s7.TARGETS[target_name].params_builder(),
        template_config=config,
        region=region,
    )
    residual = tuple(mp.mpf(value) for value in payload.get("residual", ()))
    branch = {key: mp.mpf(value) for key, value in payload.get("branch_diagnostics", {}).items()}
    failure = payload.get("failure")
    return s7.S7ResidualResult(
        point,
        params,
        local_config,
        residual,
        mp.inf if failure else mp.mpf(payload["residual_norm"]),
        None if payload.get("left_l") is None else mp.mpf(payload["left_l"]),
        None if payload.get("right_l") is None else mp.mpf(payload["right_l"]),
        tuple(payload.get("patch_counts", (0, 0))),
        branch,
        failure,
    )


def _candidate_from_payload(payload: dict) -> s7.S7ScoutCandidate:
    """Rebuild one S7 scout candidate from a persisted JSONL payload."""
    point = _point_from_payload(payload["seed_point"])
    seed = s7.S7SearchSeed(
        int(payload["seed_index"]),
        payload["target"],
        payload["region"],
        payload["source"],
        point,
    )
    return s7.S7ScoutCandidate(seed, _result_from_payload(payload["result"], payload["target"], payload.get("region", s7.DEFAULT_SCOUT_REGION.name)))


def _load_scout_candidates(path: Path, targets: tuple[str, ...] | None = None) -> list[s7.S7ScoutCandidate]:
    """Load all scout_result candidates from one S7 scout JSONL checkpoint."""
    target_set = None if targets is None else set(targets)
    return [
        _candidate_from_payload(event)
        for event in _iter_jsonl_events(path)
        if event.get("event") == "scout_result" and (target_set is None or event.get("target") in target_set)
    ]


def _candidate_norm(candidate: s7.S7ScoutCandidate) -> mp.mpf:
    """Return the scout residual norm, or infinity for failures."""
    return candidate.result.residual_norm if candidate.result.failure is None else mp.inf


def _grid_shape(run_start: dict) -> tuple[int, ...]:
    """Return axis counts from a scout run_start event."""
    return tuple(int(value) for value in run_start["grid"]["axis_counts"])


def _grid_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return row-major strides for the persisted S7 grid order."""
    strides = []
    product = 1
    for size in reversed(shape):
        strides.append(product)
        product *= size
    return tuple(reversed(strides))


def _grid_coordinate(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return the grid coordinate tuple for one local seed index."""
    strides = _grid_strides(shape)
    remaining = index
    coordinates = []
    for stride, size in zip(strides, shape):
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


def _local_index(candidate: s7.S7ScoutCandidate, per_target: int) -> int:
    """Return the seed index local to its target block."""
    return candidate.seed.index % per_target


def _target_local_minima(candidates: list[s7.S7ScoutCandidate], shape: tuple[int, ...]) -> list[s7.S7ScoutCandidate]:
    """Return target-wise branch-valid nearest-neighbor local minima."""
    per_target = 1
    for size in shape:
        per_target *= size

    grouped: dict[str, list[s7.S7ScoutCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.seed.target].append(candidate)

    minima: list[s7.S7ScoutCandidate] = []
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
    candidates: list[s7.S7ScoutCandidate],
    run_start: dict,
    max_residual: mp.mpf | None = DEFAULT_MAX_RESIDUAL,
    limit: int | None = None,
) -> list[SelectedS7Candidate]:
    """Select target-wise local minima, optionally filtering by scout residual."""
    minima = _target_local_minima(candidates, _grid_shape(run_start))
    if max_residual is not None:
        minima = [candidate for candidate in minima if candidate.result.residual_norm < max_residual]
    if limit is not None:
        minima = minima[:limit]
    return [SelectedS7Candidate(index + 1, "local-minimum", candidate) for index, candidate in enumerate(minima)]


def _selected_payload(selection: SelectedS7Candidate) -> dict:
    """Return JSON-ready data for one selected local minimum."""
    candidate = selection.candidate
    return {
        "rank": selection.rank,
        "reason": selection.reason,
        "seed_index": candidate.seed.index,
        "target": candidate.seed.target,
        "region": candidate.seed.region,
        "source": candidate.seed.source,
        "distance": _mp_string(s7._point_distance(candidate.seed.point)),
        "residual_norm": _mp_string(candidate.result.residual_norm),
        "seed_point": s7._point_payload(candidate.seed.point),
        "physical": {
            "interval_end": _mp_string(candidate.result.params.interval_end),
            "left": {
                "a": _mp_string(candidate.result.params.left.a),
                "c": _mp_string(candidate.result.params.left.c),
                "alpha": _mp_string(candidate.result.params.left.alpha),
            },
        },
    }


def _selection_config_payload(
    max_residual: mp.mpf | None,
    candidate_timeout: float | None,
    targets: tuple[str, ...] | None,
    limit: int | None,
) -> dict:
    """Return JSON-ready selection/refinement config."""
    return {
        "selection_version": SELECTION_VERSION,
        "selection_mode": "local-minima",
        "local_minimum_max_residual": _mp_string(max_residual),
        "candidate_timeout_seconds": candidate_timeout,
        "targets": None if targets is None else list(targets),
        "limit": limit,
        "settings": [
            {
                "name": settings.name,
                "order": settings.config.series_order,
                "dps": settings.config.working_dps,
                "fd_step": _mp_string(settings.fd_step),
                "max_steps": settings.max_steps,
                "max_abs_coordinate": _mp_string(settings.max_abs_coordinate),
                "min_s_coordinate": _mp_string(settings.min_s_coordinate),
            }
            for settings in s7._newton_settings()
        ],
        "verify_orders": [config.series_order for config in s7.VERIFY_CONFIGS],
    }


def _run_start_payload(
    jsonl_path: Path,
    summary_path: Path,
    scout_jsonl: Path,
    max_residual: mp.mpf | None,
    candidate_timeout: float | None,
    targets: tuple[str, ...] | None,
    limit: int | None,
) -> dict:
    """Return checkpoint metadata for one S7 scout-refinement run."""
    return {
        "random_seed": RANDOM_SEED,
        "refinement_version": REFINEMENT_VERSION,
        "scout_jsonl": str(scout_jsonl),
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "selection_config": _selection_config_payload(max_residual, candidate_timeout, targets, limit),
    }


def _summary_path_for_jsonl(path: Path) -> Path:
    """Return the summary path paired with one refinement JSONL."""
    return path.with_name(f"{path.stem}-summary.json")


def _refinement_output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped output paths for S7 scout refinement."""
    return _output_paths(OUTPUT_DIR, OUTPUT_SUFFIX, now)


def _checkpoint_is_compatible(
    path: Path,
    scout_jsonl: Path,
    max_residual: mp.mpf | None,
    candidate_timeout: float | None,
    targets: tuple[str, ...] | None,
    limit: int | None,
) -> bool:
    """Return whether an incomplete checkpoint can be resumed."""
    if _jsonl_has_event(path, "run_summary"):
        return False
    starts = [event for event in _iter_jsonl_events(path) if event.get("event") == "run_start"]
    if not starts:
        return False
    expected = _run_start_payload(
        path,
        _summary_path_for_jsonl(path),
        scout_jsonl,
        max_residual,
        candidate_timeout,
        targets,
        limit,
    )
    return all(starts[-1].get(key) == value for key, value in expected.items() if key not in {"jsonl_path", "summary_path"})


def _latest_incomplete_checkpoint(
    scout_jsonl: Path,
    max_residual: mp.mpf | None,
    candidate_timeout: float | None,
    targets: tuple[str, ...] | None,
    limit: int | None,
) -> Path | None:
    """Return the newest compatible incomplete refinement checkpoint."""
    pattern = f"*-seed{RANDOM_SEED}-{OUTPUT_SUFFIX}.jsonl"
    candidates = sorted(OUTPUT_DIR.glob(pattern), reverse=True)
    return next(
        (
            path
            for path in candidates
            if _checkpoint_is_compatible(path, scout_jsonl, max_residual, candidate_timeout, targets, limit)
        ),
        None,
    )


def _resume_or_new_paths(
    scout_jsonl: Path,
    max_residual: mp.mpf | None,
    candidate_timeout: float | None,
    targets: tuple[str, ...] | None,
    limit: int | None,
    *,
    resume: bool = True,
    now: datetime | None = None,
) -> tuple[Path, Path, bool]:
    """Return refinement paths, resuming a compatible incomplete checkpoint."""
    if resume and now is None:
        checkpoint = _latest_incomplete_checkpoint(scout_jsonl, max_residual, candidate_timeout, targets, limit)
        if checkpoint is not None:
            return checkpoint, _summary_path_for_jsonl(checkpoint), True
    jsonl_path, summary_path = _refinement_output_paths(now)
    return jsonl_path, summary_path, False


def _selected_seed_indices(path: Path) -> set[int]:
    """Return seed indices already written as selected events."""
    return {int(event["seed_index"]) for event in _iter_jsonl_events(path) if event.get("event") == "candidate_selected"}


def _classified_seed_indices(path: Path) -> set[int]:
    """Return seed indices already classified in one refinement checkpoint."""
    return {int(event["seed_index"]) for event in _iter_jsonl_events(path) if event.get("event") == "candidate_classification"}


def _write_missing_selections(path: Path, selections: list[SelectedS7Candidate]) -> None:
    """Persist any selected-local-minimum events not already present."""
    existing = _selected_seed_indices(path)
    for selection in selections:
        if selection.candidate.seed.index not in existing:
            _write_jsonl_event(path, _event("candidate_selected", _selected_payload(selection)))


def _references_by_target(targets: tuple[str, ...]) -> dict[str, tuple[s7.S7ResidualResult, ...]]:
    """Compute known-target reference residuals for selected targets."""
    return {target: s7._reference_residuals(s7.TARGETS[target]) for target in targets}


def _timeout_track(selection: SelectedS7Candidate, message: str) -> s7.S7CandidateTrack:
    """Return a failed candidate track for timeout diagnostics."""
    settings = s7._newton_settings()[0]
    target = s7.TARGETS[selection.candidate.seed.target]
    params, local_config = s7.params_from_s7_scaled(
        selection.candidate.seed.point,
        base_params=target.params_builder(),
        template_config=settings.config,
        region=s7._parameter_region_for_seed(selection.candidate.seed),
    )
    timeout_result = s7.S7ResidualResult(
        selection.candidate.seed.point,
        params,
        local_config,
        (),
        mp.inf,
        None,
        None,
        (0, 0),
        {},
        message,
    )
    stage = s7.S7RefinementStageReport(settings, timeout_result, timeout_result, (), message)
    return s7.S7CandidateTrack(
        selection.candidate.seed.index,
        selection.candidate.seed.target,
        selection.candidate.seed.region,
        selection.candidate.seed.point,
        selection.candidate.result,
        (stage,),
        (),
        "failed",
    )


def _run_selection(
    selection: SelectedS7Candidate,
    references: dict[str, tuple[s7.S7ResidualResult, ...]],
    timeout_seconds: float | None,
) -> s7.S7CandidateTrack:
    """Run one selected local minimum through the calibrated S7 recovery ladder."""
    seed = selection.candidate.seed
    target = s7.TARGETS[seed.target]
    track, status = _run_with_timeout(lambda: s7._run_recovery_track(seed, target, references[seed.target]), timeout_seconds)
    if track is None:
        return _timeout_track(selection, status or "timeout")
    return track


def _track_from_payload(payload: dict) -> dict:
    """Keep persisted track payloads as dicts for summary rebuilds."""
    return payload


def _load_classified_payloads(path: Path) -> list[dict]:
    """Load classified track payloads from one refinement checkpoint."""
    return [_track_from_payload(event) for event in _iter_jsonl_events(path) if event.get("event") == "candidate_classification"]


def _track_final_payload(track_payload: dict) -> dict:
    """Return the final residual-result payload for one persisted track."""
    stages = track_payload.get("stages") or ()
    if stages:
        return stages[-1]["final"]
    return track_payload["scout"]


def _classify_counts(track_payloads: list[dict]) -> dict[str, int]:
    """Return classification counts for refined tracks."""
    return dict(Counter(payload["classification"] for payload in track_payloads))


def _best_verified_tracks(track_payloads: list[dict], limit: int = 20) -> list[dict]:
    """Return compact best-track summaries by final verification norm."""
    def sort_key(payload: dict):
        norms = [mp.mpf(value) for value in payload.get("verification_norms", ())]
        final = _track_final_payload(payload)
        return (max(norms) if norms else mp.inf, mp.mpf(final["residual_norm"]), payload["seed_index"])

    output = []
    for payload in sorted(track_payloads, key=sort_key)[:limit]:
        final = _track_final_payload(payload)
        point = _point_from_payload(final["point"])
        output.append(
            {
                "seed_index": payload["seed_index"],
                "target": payload["target"],
                "region": payload["region"],
                "classification": payload["classification"],
                "final_residual_norm": final["residual_norm"],
                "verification_norms": payload.get("verification_norms", []),
                "distance": _mp_string(s7._point_distance(point)),
                "final_point": s7._point_payload(point),
                "physical": {
                    "interval_end": final["interval_end"],
                    "left": final["left"],
                },
            }
        )
    return output


def _classification_payload(track: s7.S7CandidateTrack) -> dict:
    """Return JSON-ready classification data for one refined S7 candidate."""
    payload = s7._track_payload(track)
    payload["source"] = "s7_scout_refine"
    return payload


def _summary_payload(
    scout_jsonl: Path,
    run_start: dict,
    selections: list[SelectedS7Candidate],
    track_payloads: list[dict],
    references: dict[str, tuple[s7.S7ResidualResult, ...]],
    selection_config: dict,
) -> dict:
    """Return JSON-ready final S7 scout-refinement summary."""
    return {
        "random_seed": RANDOM_SEED,
        "refinement_version": REFINEMENT_VERSION,
        "scout_jsonl": str(scout_jsonl),
        "scout_grid": run_start["grid"],
        "selection_config": selection_config,
        "selection_count": len(selections),
        "classified_count": len(track_payloads),
        "classification_counts": _classify_counts(track_payloads),
        "selections": [_selected_payload(selection) for selection in selections],
        "reference_residuals": {
            target: [s7._result_payload(result) for result in target_references]
            for target, target_references in references.items()
        },
        "best_verified_tracks": _best_verified_tracks(track_payloads),
        "tracks": track_payloads,
    }


def _print_selections(selections: list[SelectedS7Candidate]) -> None:
    """Print a compact selected-local-minimum table."""
    print(f"selected local minima: {len(selections)}", flush=True)
    for selection in selections:
        candidate = selection.candidate
        point = candidate.seed.point
        print(
            "  "
            f"{selection.rank:02d} target={candidate.seed.target} seed={candidate.seed.index} "
            f"norm={mp.nstr(candidate.result.residual_norm, 8)} "
            f"dist={mp.nstr(s7._point_distance(point), 6)} "
            f"point=({mp.nstr(point.u, 6)}, {mp.nstr(point.v, 6)}, {mp.nstr(point.r, 6)})",
            flush=True,
        )


def _resolve_scout_jsonl(path_text: str | None) -> Path:
    """Return the requested or newest completed S7 scout checkpoint."""
    return Path(path_text) if path_text else _latest_completed_scout_jsonl()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse S7 scout-refinement CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scout-jsonl", default=None, help="completed S7 scout JSONL to process")
    parser.add_argument("--max-residual", type=_optional_positive_mpf, default=DEFAULT_MAX_RESIDUAL, help="local-minimum scout residual cutoff; use 'none' for all")
    parser.add_argument("--targets", type=_parse_targets, default=None, help="optional comma-separated target filter, e.g. round,squashed")
    parser.add_argument("--limit", type=_positive_int, default=None, help="debugging cap after selection sorting")
    parser.add_argument("--candidate-timeout", type=_positive_float, default=float(DEFAULT_CANDIDATE_TIMEOUT_SECONDS), help="per-candidate timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="print selected local minima without refining")
    parser.add_argument("--no-resume", action="store_true", help="start a fresh refinement checkpoint")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run S7 scout local-minimum refinement."""
    args = _parse_args(argv)
    scout_jsonl = _resolve_scout_jsonl(args.scout_jsonl)
    run_start = _scout_run_start(scout_jsonl)
    candidates = _load_scout_candidates(scout_jsonl, args.targets)
    selections = _select_local_minima(candidates, run_start, args.max_residual, args.limit)
    _print_selections(selections)
    if args.dry_run:
        return

    jsonl_path, summary_path, resumed = _resume_or_new_paths(
        scout_jsonl,
        args.max_residual,
        args.candidate_timeout,
        args.targets,
        args.limit,
        resume=not args.no_resume,
    )
    run_payload = _run_start_payload(
        jsonl_path,
        summary_path,
        scout_jsonl,
        args.max_residual,
        args.candidate_timeout,
        args.targets,
        args.limit,
    )
    if resumed:
        print(f"resuming S7 refinement checkpoint {jsonl_path}", flush=True)
    else:
        print(f"writing S7 refinement JSONL to {jsonl_path}", flush=True)
        _write_jsonl_event(jsonl_path, _event("run_start", run_payload))
    _write_missing_selections(jsonl_path, selections)

    selected_targets = tuple(dict.fromkeys(selection.candidate.seed.target for selection in selections))
    references = _references_by_target(selected_targets)
    completed = _classified_seed_indices(jsonl_path)
    for selection in selections:
        seed_index = selection.candidate.seed.index
        if seed_index in completed:
            print(f"seed {seed_index}: reused completed classification", flush=True)
            continue
        track = _run_selection(selection, references, args.candidate_timeout)
        _write_jsonl_event(jsonl_path, _event("candidate_classification", _classification_payload(track)))
        print(f"seed {seed_index}: {track.classification}", flush=True)

    track_payloads = _load_classified_payloads(jsonl_path)
    payload = _summary_payload(scout_jsonl, run_start, selections, track_payloads, references, run_payload["selection_config"])
    _write_jsonl_event(jsonl_path, _event("run_summary", payload))
    _write_summary(summary_path, payload)
    print(f"classifications: {payload['classification_counts']}", flush=True)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
