"""Known round/squashed S7 calibration for G2 maximal-volume matching."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from mpmath import mp

from problem import ProblemParameters, SolverConfig
from solver.max_volume import MAX_VOLUME_VERSION, MaxVolumeSettings, max_volume_match

from ..shared.g2_max_volume_common import CALIBRATION_SETTINGS, match_payload, settings_payload
from ..shared.non_mirrored_common import RANDOM_SEED, _event, _output_paths, _write_jsonl_event, _write_summary
from .search_common import TARGETS, TARGET_NAMES


OUTPUT_DIR = Path("output/g2_max_volume_calibrations")
OUTPUT_SUFFIX = "s7-max-volume-calibration-v1"


def _settings_from_args(args: argparse.Namespace) -> MaxVolumeSettings:
    """Build max-volume settings from CLI args."""
    if args.order is None:
        return CALIBRATION_SETTINGS
    config = SolverConfig(
        args.order,
        args.dps,
        args.target_dps,
        mp.mpf(args.step_safety),
        args.sample_points,
        CALIBRATION_SETTINGS.config.match_t,
    )
    return MaxVolumeSettings(config, bisection_steps=args.bisection_steps, event_tolerance=mp.mpf(args.event_tolerance))


def classify_s7_match(target: str, residual_norm: mp.mpf, interval_error: mp.mpf | None) -> str:
    """Classify one known-S7 max-volume match."""
    if interval_error is None:
        return "failed"
    if residual_norm < mp.mpf("1e-8") and abs(interval_error) < mp.mpf("1e-8"):
        return f"recovered_{target}_s7"
    if residual_norm < mp.mpf("1e-5") and abs(interval_error) < mp.mpf("1e-5"):
        return f"near_{target}_s7"
    return "failed"


def _target_params(name: str) -> ProblemParameters:
    """Return target parameters by name."""
    return TARGETS[name].params_builder()


def run_target(name: str, settings: MaxVolumeSettings = CALIBRATION_SETTINGS) -> dict:
    """Run one known-S7 max-volume calibration target."""
    with mp.workdps(settings.config.working_dps):
        match = max_volume_match(_target_params(name), settings)
    classification = classify_s7_match(name, match.residual_norm, match.interval_error)
    return {
        "target": name,
        "classification": classification,
        "match": match_payload(match),
    }


def run_calibration(targets: tuple[str, ...] = TARGET_NAMES, settings: MaxVolumeSettings = CALIBRATION_SETTINGS) -> dict:
    """Run S7 max-volume calibration targets."""
    results = [run_target(name, settings) for name in targets]
    counts = Counter(result["classification"] for result in results)
    return {
        "max_volume_version": MAX_VOLUME_VERSION,
        "calibration": "s7",
        "random_seed": RANDOM_SEED,
        "targets": list(targets),
        "settings": settings_payload(settings),
        "classification_counts": dict(counts),
        "results": results,
    }


def _parse_targets(value: str) -> tuple[str, ...]:
    """Parse comma-separated S7 target names."""
    targets = tuple(item.strip() for item in value.split(",") if item.strip())
    if not targets:
        raise argparse.ArgumentTypeError("at least one target is required")
    unknown = [target for target in targets if target not in TARGETS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown S7 targets: {', '.join(unknown)}")
    return targets


def _print_summary(payload: dict) -> None:
    """Print a compact calibration summary."""
    print(f"classifications: {payload['classification_counts']}", flush=True)
    for result in payload["results"]:
        match = result["match"]
        print()
        print(result["target"], flush=True)
        print(f"  classification: {result['classification']}", flush=True)
        print(f"  residual norm: {match['residual_norm']}", flush=True)
        print(f"  reconstructed interval: {match['reconstructed_interval']}", flush=True)
        print(f"  interval error: {match['interval_error']}", flush=True)
        print(f"  left tau: {match['left']['max_tau']}  right tau: {match['right']['max_tau']}", flush=True)


def main(argv: list[str] | None = None) -> None:
    """Run known-S7 maximal-volume calibrations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=_parse_targets, default=TARGET_NAMES, help="comma-separated subset of round,squashed")
    parser.add_argument("--order", type=int, default=None, help="override Taylor order")
    parser.add_argument("--dps", type=int, default=120, help="working precision for --order")
    parser.add_argument("--target-dps", type=int, default=45)
    parser.add_argument("--step-safety", default="0.55")
    parser.add_argument("--sample-points", type=int, default=2)
    parser.add_argument("--bisection-steps", type=int, default=56)
    parser.add_argument("--event-tolerance", default="1e-30")
    parser.add_argument("--no-write", action="store_true", help="print only, without writing output files")
    args = parser.parse_args(argv)
    settings = _settings_from_args(args)
    payload = run_calibration(args.targets, settings)
    _print_summary(payload)
    if args.no_write:
        return
    jsonl_path, summary_path = _output_paths(OUTPUT_DIR, OUTPUT_SUFFIX)
    payload = {**payload, "jsonl_path": str(jsonl_path), "summary_path": str(summary_path)}
    _write_jsonl_event(jsonl_path, _event("run_summary", payload))
    _write_summary(summary_path, payload)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
