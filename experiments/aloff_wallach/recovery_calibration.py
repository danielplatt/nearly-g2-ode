"""Known-solution recovery calibration for Aloff-Wallach ``N_{1,1}``.

This command deliberately starts from the two known homogeneous
Aloff-Wallach nearly-parallel structures and tests whether the current
endpoint-reduced max-volume machinery can recover them after allowing simple
left/right endpoint sign branches.  It is a calibration and branch-audit layer,
not a proof of nonexistence.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable

from mpmath import mp

from experiments.shared.non_mirrored_common import RANDOM_SEED, _event, _mp_string, _output_paths, _write_jsonl_event, _write_summary

from . import ansatz, endpoint_smoothness
from .evolution import AWSettings, EVOLUTION_VERSION, EndpointConstants, MatchResult, max_volume_match
from .scout import _settings_payload, _side_payload


RECOVERY_VERSION = "aloff-wallach-n11-recovery-calibration-v1"
OUTPUT_DIR = Path("output/aloff_wallach_recovery_calibrations")
OUTPUT_SUFFIX = RECOVERY_VERSION
TARGET_NAMES = tuple(solution.label for solution in ansatz.n11_known_solutions())


@dataclass(frozen=True)
class BranchVariant:
    """One discrete endpoint chart sign branch."""

    label: str
    signs: tuple[int, int, int, int]


@dataclass(frozen=True)
class RecoverySeed:
    """One deterministic known-solution recovery seed."""

    index: int
    target: str
    left_branch: BranchVariant
    right_branch: BranchVariant
    scale_multiplier: float
    structure_scale: float
    left_constants: EndpointConstants
    right_constants: EndpointConstants


def _parse_csv(value: str, allowed: tuple[str, ...] | None = None) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    if allowed is not None:
        unknown = [item for item in items if item not in allowed]
        if unknown:
            raise argparse.ArgumentTypeError(f"unknown value(s): {', '.join(unknown)}")
    return items


def _parse_float_csv(value: str) -> tuple[float, ...]:
    try:
        items = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated number")
    if any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("all values must be positive")
    return items


def _target_solution(target: str) -> ansatz.N11KnownSolution:
    for solution in ansatz.n11_known_solutions():
        if solution.label == target:
            return solution
    raise ValueError(f"unknown Aloff-Wallach target {target!r}")


def reference_constants(target: str, lam: float = 4.0, scale_multiplier: float = 1.0) -> EndpointConstants:
    """Return the homogeneous ``A,B,C,D`` reference scaled to the requested lambda."""
    solution = _target_solution(target)
    scale = float(solution.lambda_value / mp.mpf(str(lam))) * scale_multiplier
    return EndpointConstants(
        float(solution.A) * scale,
        float(solution.B) * scale,
        float(solution.C) * scale,
        float(solution.D) * scale,
    )


def _apply_branch(constants: EndpointConstants, branch: BranchVariant) -> EndpointConstants:
    a, b, c, d = branch.signs
    return EndpointConstants(a * constants.A, b * constants.B, c * constants.C, d * constants.D)


def branch_variants(mode: str) -> tuple[BranchVariant, ...]:
    """Return deterministic endpoint sign branches for calibration."""
    if mode == "canonical":
        return (BranchVariant("++++", (1, 1, 1, 1)),)
    if mode == "paired-signs":
        signs = (
            (1, 1, 1, 1),
            (1, -1, -1, 1),
            (-1, 1, 1, -1),
            (-1, -1, -1, -1),
        )
    elif mode == "all-signs":
        signs = tuple(product((1, -1), repeat=4))
    else:
        raise ValueError(f"unknown branch mode {mode!r}")
    return tuple(BranchVariant("".join("+" if sign > 0 else "-" for sign in item), item) for item in signs)


def recovery_seeds(
    targets: tuple[str, ...] = TARGET_NAMES,
    branch_mode: str = "canonical",
    scale_multipliers: tuple[float, ...] = (1.0,),
    structure_scales: tuple[float, ...] = (1.0,),
    *,
    lam: float = 4.0,
    limit: int | None = None,
) -> list[RecoverySeed]:
    """Return deterministic known-reference branch recovery seeds."""
    variants = branch_variants(branch_mode)
    seeds: list[RecoverySeed] = []
    for target in targets:
        _target_solution(target)
        for scale_multiplier in scale_multipliers:
            reference = reference_constants(target, lam, scale_multiplier)
            for structure_scale in structure_scales:
                for left_branch in variants:
                    for right_branch in variants:
                        if limit is not None and len(seeds) >= limit:
                            return seeds
                        seeds.append(
                            RecoverySeed(
                                len(seeds),
                                target,
                                left_branch,
                                right_branch,
                                scale_multiplier,
                                structure_scale,
                                _apply_branch(reference, left_branch),
                                _apply_branch(reference, right_branch),
                            )
                        )
    return seeds


def recovery_seed_count(
    targets: tuple[str, ...] = TARGET_NAMES,
    branch_mode: str = "canonical",
    scale_multipliers: tuple[float, ...] = (1.0,),
    structure_scales: tuple[float, ...] = (1.0,),
    limit: int | None = None,
) -> int:
    """Return the deterministic seed count after an optional debug limit."""
    count = len(targets) * len(branch_variants(branch_mode)) ** 2 * len(scale_multipliers) * len(structure_scales)
    return min(count, limit) if limit is not None else count


def _constants_payload(constants: EndpointConstants) -> dict[str, float]:
    return {"A": constants.A, "B": constants.B, "C": constants.C, "D": constants.D}


def _match_payload(match: MatchResult) -> dict:
    return {
        "failure": match.failure,
        "residual_norm": match.residual_norm,
        "residual": list(match.residual),
        "reconstructed_interval": match.reconstructed_interval,
        "left": _side_payload(match.left),
        "right": _side_payload(match.right),
    }


def classify_match(seed: RecoverySeed, match: MatchResult) -> str:
    """Classify one known-solution recovery attempt."""
    if match.failure is not None:
        return str(match.failure)
    assert match.residual_norm is not None
    interval = match.reconstructed_interval
    left_germ = match.left.germ.residual_norm
    right_germ = match.right.germ.residual_norm
    max_germ = max(left_germ, right_germ)
    if interval is not None and interval > 0.05 and match.residual_norm < 1e-6 and max_germ < 1e-4:
        return f"recovered_{seed.target}"
    if interval is not None and interval > 0.02 and match.residual_norm < 1e-3:
        return f"near_{seed.target}"
    if interval is not None and interval < 0.05 and match.residual_norm < 0.05:
        return "collapsed_tail"
    return "finite_residual"


def _evaluate_seed_payload(seed: RecoverySeed, settings: AWSettings) -> dict:
    """Evaluate one recovery seed."""
    seed_settings = replace(settings, structure_scale=seed.structure_scale)
    match = max_volume_match(seed.left_constants, seed.right_constants, seed_settings)
    classification = classify_match(seed, match)
    return {
        "seed_index": seed.index,
        "target": seed.target,
        "source": "known_homogeneous_branch_reference",
        "left_branch": seed.left_branch.label,
        "right_branch": seed.right_branch.label,
        "scale_multiplier": seed.scale_multiplier,
        "structure_scale": seed.structure_scale,
        "left_constants": _constants_payload(seed.left_constants),
        "right_constants": _constants_payload(seed.right_constants),
        "classification": classification,
        "result": _match_payload(match),
    }


def _evaluate_seed_payload_star(args) -> dict:
    seed, settings = args
    return _evaluate_seed_payload(seed, settings)


def _evaluate_seed_payloads(
    seeds: list[RecoverySeed],
    workers: int,
    settings: AWSettings,
    chunksize: int | None = None,
) -> Iterable[dict]:
    """Yield recovery payloads in stable order."""
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
    return {int(event["seed_index"]) for event in _jsonl_events(path) if event.get("event") == "recovery_result"}


def _result_payloads(path: Path) -> list[dict]:
    return [event for event in _jsonl_events(path) if event.get("event") == "recovery_result"]


def _run_has_summary(path: Path) -> bool:
    return any(event.get("event") == "run_summary" for event in _jsonl_events(path))


def _summary_path_for_jsonl(path: Path) -> Path:
    return path.with_name(f"{path.stem}-summary.json")


def _new_output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    return _output_paths(OUTPUT_DIR, OUTPUT_SUFFIX, now)


def _settings_from_args(args: argparse.Namespace) -> AWSettings:
    return AWSettings(
        lam=float(args.lam),
        structure_scale=None,
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


def _run_config(
    targets: tuple[str, ...],
    branch_mode: str,
    scale_multipliers: tuple[float, ...],
    structure_scales: tuple[float, ...],
    limit: int | None,
    settings: AWSettings,
) -> dict:
    return {
        "targets": list(targets),
        "branch_mode": branch_mode,
        "branch_count": len(branch_variants(branch_mode)),
        "scale_multipliers": list(scale_multipliers),
        "structure_scales": list(structure_scales),
        "seed_count": recovery_seed_count(targets, branch_mode, scale_multipliers, structure_scales, limit),
        "full_seed_count": recovery_seed_count(targets, branch_mode, scale_multipliers, structure_scales, None),
        "limit": limit,
        "normalization": "homogeneous A,B,C,D scaled by lambda_known / lambda_requested",
        "reference_constants": {
            target: {
                "lambda": _mp_string(_target_solution(target).lambda_value),
                "lambda4_constants": _constants_payload(reference_constants(target, settings.lam)),
            }
            for target in targets
        },
    }


def _run_start_payload(
    jsonl_path: Path,
    summary_path: Path,
    targets: tuple[str, ...],
    branch_mode: str,
    scale_multipliers: tuple[float, ...],
    structure_scales: tuple[float, ...],
    limit: int | None,
    settings: AWSettings,
) -> dict:
    return {
        "random_seed": RANDOM_SEED,
        "recovery_version": RECOVERY_VERSION,
        "ansatz_version": ansatz.N11_ANSATZ_VERSION,
        "evolution_version": EVOLUTION_VERSION,
        "endpoint_smoothness_version": endpoint_smoothness.ENDPOINT_SMOOTHNESS_VERSION,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "run_config": _run_config(targets, branch_mode, scale_multipliers, structure_scales, limit, settings),
        "settings": _settings_payload(settings),
    }


def _checkpoint_is_compatible(
    path: Path,
    targets: tuple[str, ...],
    branch_mode: str,
    scale_multipliers: tuple[float, ...],
    structure_scales: tuple[float, ...],
    limit: int | None,
    settings: AWSettings,
) -> bool:
    if _run_has_summary(path):
        return False
    starts = [event for event in _jsonl_events(path) if event.get("event") == "run_start"]
    if not starts:
        return False
    expected = _run_start_payload(path, _summary_path_for_jsonl(path), targets, branch_mode, scale_multipliers, structure_scales, limit, settings)
    ignored = {"jsonl_path", "summary_path"}
    return all(starts[-1].get(key) == value for key, value in expected.items() if key not in ignored)


def _latest_incomplete_checkpoint(
    targets: tuple[str, ...],
    branch_mode: str,
    scale_multipliers: tuple[float, ...],
    structure_scales: tuple[float, ...],
    limit: int | None,
    settings: AWSettings,
) -> Path | None:
    candidates = sorted(OUTPUT_DIR.glob(f"*-seed{RANDOM_SEED}-{OUTPUT_SUFFIX}.jsonl"), reverse=True)
    return next(
        (
            path
            for path in candidates
            if _checkpoint_is_compatible(path, targets, branch_mode, scale_multipliers, structure_scales, limit, settings)
        ),
        None,
    )


def _resume_or_new_paths(
    targets: tuple[str, ...],
    branch_mode: str,
    scale_multipliers: tuple[float, ...],
    structure_scales: tuple[float, ...],
    limit: int | None,
    settings: AWSettings,
    resume: bool,
) -> tuple[Path, Path, bool]:
    if resume:
        checkpoint = _latest_incomplete_checkpoint(targets, branch_mode, scale_multipliers, structure_scales, limit, settings)
        if checkpoint is not None:
            return checkpoint, _summary_path_for_jsonl(checkpoint), True
    jsonl_path, summary_path = _new_output_paths()
    return jsonl_path, summary_path, False


def _payload_norm(payload: dict) -> mp.mpf:
    value = payload["result"]["residual_norm"]
    return mp.inf if value is None else mp.mpf(str(value))


def _compact_payload(payload: dict) -> dict:
    return {
        "seed_index": payload["seed_index"],
        "target": payload["target"],
        "classification": payload["classification"],
        "left_branch": payload["left_branch"],
        "right_branch": payload["right_branch"],
        "scale_multiplier": payload["scale_multiplier"],
        "structure_scale": payload["structure_scale"],
        "left_constants": payload["left_constants"],
        "right_constants": payload["right_constants"],
        "failure": payload["result"]["failure"],
        "residual_norm": payload["result"]["residual_norm"],
        "reconstructed_interval": payload["result"].get("reconstructed_interval"),
        "left_status": payload["result"]["left"]["status"],
        "right_status": payload["result"]["right"]["status"],
        "left_germ_residual": payload["result"]["left"]["germ"]["residual_norm"],
        "right_germ_residual": payload["result"]["right"]["germ"]["residual_norm"],
    }


def _summary_payload(jsonl_path: Path, metadata: dict, best_limit: int = 30) -> dict:
    payloads = _result_payloads(jsonl_path)
    counts = Counter(payload["classification"] for payload in payloads)
    successes = [payload for payload in payloads if payload["result"]["failure"] is None]
    best = sorted(successes, key=_payload_norm)[:best_limit]
    best_by_germ = sorted(
        payloads,
        key=lambda payload: max(
            mp.mpf(str(payload["result"]["left"]["germ"]["residual_norm"])),
            mp.mpf(str(payload["result"]["right"]["germ"]["residual_norm"])),
        ),
    )[:best_limit]
    return {
        **metadata,
        "result_count": len(payloads),
        "classification_counts": dict(counts),
        "best_recovery_matches": [_compact_payload(payload) for payload in best],
        "best_germ_fits": [_compact_payload(payload) for payload in best_by_germ],
    }


def _run_recovery(
    seeds: list[RecoverySeed],
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
        _write_jsonl_event(jsonl_path, _event("recovery_result", payload))
        if progress_every and (completed_count % progress_every == 0 or completed_count == len(seeds)):
            print(
                f"[{completed_count}/{len(seeds)}] seed {payload['seed_index']} "
                f"target={payload['target']} class={payload['classification']} "
                f"residual={payload['result']['residual_norm']}",
                flush=True,
            )


def _print_dry_run(seeds: list[RecoverySeed], metadata: dict) -> None:
    print("Aloff-Wallach N_{1,1} known-solution recovery dry run", flush=True)
    print(f"version: {RECOVERY_VERSION}", flush=True)
    print(f"targets: {', '.join(metadata['run_config']['targets'])}", flush=True)
    print(f"branch mode: {metadata['run_config']['branch_mode']}", flush=True)
    print(f"scale multipliers: {metadata['run_config']['scale_multipliers']}", flush=True)
    print(f"structure scales: {metadata['run_config']['structure_scales']}", flush=True)
    print(f"seed count: {metadata['run_config']['seed_count']} of {metadata['run_config']['full_seed_count']}", flush=True)
    for seed in seeds[: min(12, len(seeds))]:
        print(
            f"  seed {seed.index}: target={seed.target} "
            f"L={seed.left_branch.label} R={seed.right_branch.label} "
            f"scale={seed.scale_multiplier} structure={seed.structure_scale} "
            f"left={_constants_payload(seed.left_constants)} right={_constants_payload(seed.right_constants)}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    """Run the known-solution recovery calibration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=lambda value: _parse_csv(value, TARGET_NAMES), default=TARGET_NAMES)
    parser.add_argument("--branch-mode", choices=("canonical", "paired-signs", "all-signs"), default="canonical")
    parser.add_argument("--scale-multipliers", type=_parse_float_csv, default=(1.0,))
    parser.add_argument("--structure-scales", type=_parse_float_csv, default=(1.0,))
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)), help="parallel workers")
    parser.add_argument("--chunksize", type=int, default=None, help="process-pool chunksize")
    parser.add_argument("--limit", type=int, default=None, help="debug limit on evaluated seeds")
    parser.add_argument("--dry-run", action="store_true", help="print seed metadata without evaluating")
    parser.add_argument("--no-resume", action="store_true", help="start a fresh checkpoint even if a compatible incomplete run exists")
    parser.add_argument("--progress-every", type=int, default=10, help="print progress every N completed seeds")
    parser.add_argument("--lam", type=float, default=4.0, help="normalized nearly-parallel lambda")
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
    seeds = recovery_seeds(
        args.targets,
        args.branch_mode,
        args.scale_multipliers,
        args.structure_scales,
        lam=settings.lam,
        limit=args.limit,
    )
    placeholder_jsonl = Path("<dry-run>")
    placeholder_summary = Path("<dry-run-summary>")
    metadata = _run_start_payload(
        placeholder_jsonl,
        placeholder_summary,
        args.targets,
        args.branch_mode,
        args.scale_multipliers,
        args.structure_scales,
        args.limit,
        settings,
    )
    if args.dry_run:
        _print_dry_run(seeds, metadata)
        return

    jsonl_path, summary_path, resumed = _resume_or_new_paths(
        args.targets,
        args.branch_mode,
        args.scale_multipliers,
        args.structure_scales,
        args.limit,
        settings,
        not args.no_resume,
    )
    metadata = _run_start_payload(
        jsonl_path,
        summary_path,
        args.targets,
        args.branch_mode,
        args.scale_multipliers,
        args.structure_scales,
        args.limit,
        settings,
    )
    if not resumed:
        _write_jsonl_event(jsonl_path, _event("run_start", metadata))
    print(
        f"{'resuming' if resumed else 'starting'} Aloff-Wallach recovery calibration: "
        f"{len(seeds)} seeds, workers={args.workers}, output={jsonl_path}",
        flush=True,
    )
    _run_recovery(seeds, jsonl_path, args.workers, settings, args.progress_every, args.chunksize)
    summary = _summary_payload(jsonl_path, metadata)
    _write_jsonl_event(jsonl_path, _event("run_summary", summary))
    _write_summary(summary_path, summary)
    print(f"wrote summary: {summary_path}", flush=True)
    print(f"classification counts: {summary['classification_counts']}", flush=True)


if __name__ == "__main__":
    main()
