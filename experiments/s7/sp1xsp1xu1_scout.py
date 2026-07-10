"""Target-centered max-volume scout for the S7 Sp(1)xSp(1)xU(1) action."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Iterable

from mpmath import mp

from experiments.shared.non_mirrored_common import RANDOM_SEED, _event, _mp_string, _output_paths, _write_jsonl_event, _write_summary

from . import sp1xsp1xu1_matching as matching
from . import su2_cubed_action_audit


SCOUT_VERSION = "s7-sp1xsp1xu1-scout-v1"
OUTPUT_DIR = Path("output/s7_sp1xsp1xu1_scouts")
OUTPUT_SUFFIX = SCOUT_VERSION
COORDINATE_NAMES = (
    "left_A3",
    "left_A4",
    "left_B2",
    "left_B4",
    "left_C",
    "right_A3",
    "right_A4",
    "right_B2",
    "right_B4",
    "right_C",
)


@dataclass(frozen=True)
class ScoutPoint:
    """A pair of five-parameter endpoint charts."""

    left: matching.EndpointParameters
    right: matching.EndpointParameters


@dataclass(frozen=True)
class ScoutSeed:
    """One deterministic target-centered scout seed."""

    index: int
    source: str
    offsets: tuple[float, ...]
    point: ScoutPoint


def target_by_name(name: str) -> su2_cubed_action_audit.PodestaTarget:
    """Return a known calibration target by name."""
    if name == "round":
        return su2_cubed_action_audit.round_target()
    if name == "squashed":
        return su2_cubed_action_audit.squashed_target()
    raise ValueError(f"unknown target {name!r}")


def center_point(target_name: str, lam: float | None = None) -> ScoutPoint:
    """Return the endpoint-parameter center induced by a known target."""
    target = target_by_name(target_name)
    target_lam = float(target.lam if lam is None else lam)
    return ScoutPoint(
        matching.target_endpoint_parameters_at_lambda(target, "left", target_lam),
        matching.target_endpoint_parameters_at_lambda(target, "right", target_lam),
    )


def _coordinates(point: ScoutPoint) -> tuple[float, ...]:
    return (
        point.left.A3,
        point.left.A4,
        point.left.B2,
        point.left.B4,
        point.left.C,
        point.right.A3,
        point.right.A4,
        point.right.B2,
        point.right.B4,
        point.right.C,
    )


def _point_from_coordinates(values: Iterable[float]) -> ScoutPoint:
    parsed = tuple(float(value) for value in values)
    if len(parsed) != 10:
        raise ValueError("Sp(1)xSp(1)xU(1) scout points need 10 coordinates")
    return ScoutPoint(
        matching.EndpointParameters(*parsed[:5]),
        matching.EndpointParameters(*parsed[5:]),
    )


def _point_payload(point: ScoutPoint) -> dict[str, str]:
    return {name: _mp_string(mp.mpf(value)) for name, value in zip(COORDINATE_NAMES, _coordinates(point))}


def scout_seeds(
    target_name: str | None,
    samples: int,
    radius: float,
    *,
    lam: float = 4.0,
    seed: int = RANDOM_SEED,
    include_axis_controls: bool = True,
    include_known_controls: bool = False,
) -> list[ScoutSeed]:
    """Return deterministic scout seeds.

    With ``target_name=None`` this samples an absolute box in the 10 endpoint
    parameters.  With ``target_name`` set to ``round`` or ``squashed`` it samples
    offsets around that known target center.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    seeds: list[ScoutSeed] = []
    rng = Random(seed)
    if target_name is None:
        if radius <= 0:
            raise ValueError("target-independent scout needs a positive radius")
        if include_known_controls:
            for known_name in ("round", "squashed"):
                if len(seeds) >= samples:
                    return seeds
                point = center_point(known_name, lam)
                seeds.append(ScoutSeed(len(seeds), f"{known_name}_control", (0.0,) * 10, point))
        while len(seeds) < samples:
            values = tuple(rng.uniform(-radius, radius) for _ in range(10))
            seeds.append(ScoutSeed(len(seeds), "random_box", values, _point_from_coordinates(values)))
        return seeds

    center = center_point(target_name, lam)
    center_coordinates = _coordinates(center)
    seeds.append(ScoutSeed(0, "center", (0.0,) * 10, center))
    if include_axis_controls and radius > 0:
        for axis in range(10):
            for sign in (-1.0, 1.0):
                if len(seeds) >= samples:
                    return seeds
                offsets = tuple(sign * radius if index == axis else 0.0 for index in range(10))
                seeds.append(
                    ScoutSeed(
                        len(seeds),
                        "axis_control",
                        offsets,
                        _point_from_coordinates(value + offset for value, offset in zip(center_coordinates, offsets)),
                    )
                )
    while len(seeds) < samples:
        offsets = tuple(rng.uniform(-radius, radius) for _ in range(10))
        seeds.append(
            ScoutSeed(
                len(seeds),
                "random_box",
                offsets,
                _point_from_coordinates(value + offset for value, offset in zip(center_coordinates, offsets)),
            )
        )
    return seeds


def _settings_payload(settings: matching.MatchingSettings) -> dict[str, object]:
    return {
        "lambda": settings.lam,
        "endpoint_order": settings.endpoint_order,
        "germ_epsilon": settings.germ_epsilon,
        "germ_samples": list(settings.germ_samples),
        "max_tau": settings.max_tau,
        "max_step": settings.max_step,
        "rtol": settings.rtol,
        "atol": settings.atol,
        "max_germ_evaluations": settings.max_germ_evaluations,
    }


def _germ_payload(germ: matching.EndpointGerm) -> dict[str, object]:
    return {
        "side": germ.side,
        "source": germ.source,
        "residual_norm": germ.residual_norm,
        "success": germ.success,
        "message": germ.message,
        "parameters": {
            "A3": germ.parameters.A3,
            "A4": germ.parameters.A4,
            "B2": germ.parameters.B2,
            "B4": germ.parameters.B4,
            "C": germ.parameters.C,
        },
    }


def _side_payload(side: matching.MarchResult) -> dict[str, object]:
    return {
        "status": side.status,
        "tau": side.tau,
        "volume": side.volume,
        "volume_dot": side.volume_dot,
        "volume_sign": side.volume_sign,
        "message": side.message,
        "germ": _germ_payload(side.germ),
    }


def _match_payload(match: matching.MatchResult) -> dict[str, object]:
    return {
        "failure": match.failure,
        "residual_norm": match.residual_norm,
        "residual": list(match.residual),
        "reconstructed_interval": match.reconstructed_interval,
        "left": _side_payload(match.left),
        "right": _side_payload(match.right),
    }


def _evaluate_seed(seed: ScoutSeed, settings: matching.MatchingSettings) -> dict[str, object]:
    match = matching.max_volume_match(seed.point.left, seed.point.right, settings)
    return {
        "seed_index": seed.index,
        "source": seed.source,
        "offsets": list(seed.offsets),
        "seed_point": _point_payload(seed.point),
        "result": _match_payload(match),
    }


def _evaluate_seed_star(args) -> dict[str, object]:
    seed, settings = args
    return _evaluate_seed(seed, settings)


def _evaluate_seeds(
    seeds: list[ScoutSeed],
    settings: matching.MatchingSettings,
    workers: int,
    chunksize: int | None,
) -> Iterable[dict[str, object]]:
    if workers <= 1:
        for seed in seeds:
            yield _evaluate_seed(seed, settings)
        return
    try:
        executor = ProcessPoolExecutor(max_workers=workers)
    except (OSError, PermissionError):
        executor = ThreadPoolExecutor(max_workers=workers)
    with executor:
        yield from executor.map(_evaluate_seed_star, [(seed, settings) for seed in seeds], chunksize=chunksize or 1)


def _payload_status(payload: dict[str, object]) -> str:
    result = payload["result"]
    assert isinstance(result, dict)
    return "ok" if result["failure"] is None else str(result["failure"])


def _payload_norm(payload: dict[str, object]) -> float:
    result = payload["result"]
    assert isinstance(result, dict)
    value = result["residual_norm"]
    return float("inf") if value is None else float(value)


def _compact_payload(payload: dict[str, object]) -> dict[str, object]:
    result = payload["result"]
    assert isinstance(result, dict)
    left = result["left"]
    right = result["right"]
    assert isinstance(left, dict)
    assert isinstance(right, dict)
    return {
        "seed_index": payload["seed_index"],
        "source": payload["source"],
        "residual_norm": result["residual_norm"],
        "reconstructed_interval": result["reconstructed_interval"],
        "failure": result["failure"],
        "left_status": left["status"],
        "right_status": right["status"],
        "left_germ_residual": left["germ"]["residual_norm"],  # type: ignore[index]
        "right_germ_residual": right["germ"]["residual_norm"],  # type: ignore[index]
        "offsets": payload["offsets"],
    }


def _jsonl_payloads(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payloads = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "scout_result":
                payloads.append(event)
    return payloads


def _summary_payload(jsonl_path: Path, metadata: dict[str, object], best_limit: int = 20) -> dict[str, object]:
    payloads = _jsonl_payloads(jsonl_path)
    counts = Counter(_payload_status(payload) for payload in payloads)
    best = sorted(payloads, key=_payload_norm)[:best_limit]
    return {
        **metadata,
        "scout_count": len(payloads),
        "classification_counts": dict(counts),
        "best_scouts": [_compact_payload(payload) for payload in best],
    }


def _settings_from_args(args: argparse.Namespace) -> matching.MatchingSettings:
    if args.lam is None:
        lam = 4.0 if args.target == "none" else float(target_by_name(args.target).lam)
    else:
        lam = float(args.lam)
    return matching.MatchingSettings(
        lam=lam,
        endpoint_order=args.endpoint_order,
        germ_epsilon=args.germ_epsilon,
        max_tau=args.max_tau,
        max_step=args.max_step,
        rtol=args.rtol,
        atol=args.atol,
        max_germ_evaluations=args.max_germ_evaluations,
    )


def _metadata(
    target_name: str,
    samples: int,
    radius: float,
    settings: matching.MatchingSettings,
    jsonl_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, object]:
    target = None if target_name == "none" else target_by_name(target_name)
    return {
        "random_seed": RANDOM_SEED,
        "scout_version": SCOUT_VERSION,
        "matching_version": matching.MATCHING_VERSION,
        "target": target_name,
        "target_lambda": None if target is None else float(target.lam),
        "lambda": settings.lam,
        "center": None if target is None else _point_payload(center_point(target_name, settings.lam)),
        "coordinate_names": list(COORDINATE_NAMES),
        "samples": samples,
        "radius": radius,
        "settings": _settings_payload(settings),
        "jsonl_path": None if jsonl_path is None else str(jsonl_path),
        "summary_path": None if summary_path is None else str(summary_path),
    }


def main(argv: list[str] | None = None) -> int:
    """Run a target-centered max-volume scout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("none", "round", "squashed"),
        default="none",
        help="none samples an absolute box; round/squashed sample around a known center",
    )
    parser.add_argument("--samples", type=int, default=1, help="number of seeds, including the center seed")
    parser.add_argument("--radius", type=float, default=1.0, help="absolute half-width, or offset radius around a target center")
    parser.add_argument("--include-known-controls", action="store_true", help="prepend round/squashed controls in target-independent mode")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)), help="parallel workers")
    parser.add_argument("--chunksize", type=int, default=None, help="process-pool chunksize")
    parser.add_argument("--dry-run", action="store_true", help="print seed metadata without evaluating")
    parser.add_argument("--recover-known", action="store_true", help="run exact-germ round/squashed max-volume recovery and exit")
    parser.add_argument("--progress-every", type=int, default=1, help="print progress every N completed seeds")
    parser.add_argument("--lam", type=float, default=None, help="override lambda; by default use the target lambda")
    parser.add_argument("--endpoint-order", type=int, default=3, help="regular endpoint Taylor order fitted internally")
    parser.add_argument("--germ-epsilon", type=float, default=1e-3, help="local time where endpoint germs seed marching")
    parser.add_argument("--max-tau", type=float, default=1.2, help="maximum one-sided march time")
    parser.add_argument("--max-step", type=float, default=0.02, help="maximum solve_ivp step")
    parser.add_argument("--rtol", type=float, default=1e-7, help="solve_ivp relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-9, help="solve_ivp absolute tolerance")
    parser.add_argument("--max-germ-evaluations", type=int, default=80, help="least-squares evaluations per endpoint germ")
    args = parser.parse_args(argv)

    settings = _settings_from_args(args)
    if args.recover_known:
        print(json.dumps(matching.known_recovery_summary(), indent=2, sort_keys=True))
        return 0

    target_name = None if args.target == "none" else args.target
    seeds = scout_seeds(
        target_name,
        args.samples,
        args.radius,
        lam=settings.lam,
        include_known_controls=args.include_known_controls,
    )
    metadata = _metadata(args.target, args.samples, args.radius, settings)
    if args.dry_run:
        print("S7 Sp(1)xSp(1)xU(1) max-volume scout dry run", flush=True)
        print(f"version: {SCOUT_VERSION}", flush=True)
        print(f"target: {args.target}", flush=True)
        print(f"lambda: {settings.lam}", flush=True)
        print(f"coordinates: {', '.join(COORDINATE_NAMES)}", flush=True)
        print(f"samples: {len(seeds)}", flush=True)
        print(f"radius: {args.radius}", flush=True)
        for seed in seeds[: min(10, len(seeds))]:
            print(f"  seed {seed.index}: source={seed.source} offsets={seed.offsets}", flush=True)
        return 0

    jsonl_path, summary_path = _output_paths(OUTPUT_DIR, OUTPUT_SUFFIX, datetime.now())
    metadata = _metadata(args.target, args.samples, args.radius, settings, jsonl_path, summary_path)
    _write_jsonl_event(jsonl_path, _event("run_start", metadata))
    print(
        f"starting {SCOUT_VERSION}: target={args.target}, lambda={settings.lam}, samples={len(seeds)}, "
        f"workers={args.workers}, output={jsonl_path}",
        flush=True,
    )
    for completed, payload in enumerate(_evaluate_seeds(seeds, settings, args.workers, args.chunksize), start=1):
        _write_jsonl_event(jsonl_path, _event("scout_result", payload))
        if args.progress_every and (completed % args.progress_every == 0 or completed == len(seeds)):
            result = payload["result"]
            assert isinstance(result, dict)
            print(
                f"[{completed}/{len(seeds)}] seed {payload['seed_index']} "
                f"status={_payload_status(payload)} residual={result['residual_norm']}",
                flush=True,
            )
    summary = _summary_payload(jsonl_path, metadata)
    _write_jsonl_event(jsonl_path, _event("run_summary", summary))
    _write_summary(summary_path, summary)
    print(f"wrote summary: {summary_path}", flush=True)
    print(f"classification counts: {summary['classification_counts']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
