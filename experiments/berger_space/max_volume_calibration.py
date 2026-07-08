"""Known-Berger calibration for G2 maximal-volume matching."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from mpmath import mp

from problem import DEFAULT_PARAMS, SolverConfig
from solver.max_volume import MAX_VOLUME_VERSION, MaxVolumeSettings, max_volume_match

from ..shared.g2_max_volume_common import CALIBRATION_SETTINGS, match_payload, settings_payload
from ..shared.non_mirrored_common import RANDOM_SEED, _event, _mp_string, _output_paths, _write_jsonl_event, _write_summary


OUTPUT_DIR = Path("output/g2_max_volume_calibrations")
OUTPUT_SUFFIX = "berger-max-volume-calibration-v1"


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


def classify_berger_match(residual_norm: mp.mpf, interval_error: mp.mpf | None) -> str:
    """Classify one known-Berger max-volume match."""
    if interval_error is None:
        return "failed"
    if residual_norm < mp.mpf("1e-8") and abs(interval_error) < mp.mpf("1e-8"):
        return "recovered_berger"
    if residual_norm < mp.mpf("1e-5") and abs(interval_error) < mp.mpf("1e-5"):
        return "near_berger"
    return "failed"


def run_calibration(settings: MaxVolumeSettings = CALIBRATION_SETTINGS) -> dict:
    """Run the known-Berger max-volume calibration and return a summary payload."""
    with mp.workdps(settings.config.working_dps):
        match = max_volume_match(DEFAULT_PARAMS, settings)
    classification = classify_berger_match(match.residual_norm, match.interval_error)
    return {
        "max_volume_version": MAX_VOLUME_VERSION,
        "calibration": "berger",
        "random_seed": RANDOM_SEED,
        "classification": classification,
        "classification_counts": dict(Counter([classification])),
        "settings": settings_payload(settings),
        "match": match_payload(match),
    }


def _print_summary(payload: dict) -> None:
    """Print a compact calibration summary."""
    match = payload["match"]
    left = match["left"]
    right = match["right"]
    print(f"classification: {payload['classification']}", flush=True)
    print(f"residual norm: {match['residual_norm']}", flush=True)
    print(f"reconstructed interval: {match['reconstructed_interval']}", flush=True)
    print(f"interval error: {match['interval_error']}", flush=True)
    print(f"left tau: {left['max_tau']}  right tau: {right['max_tau']}", flush=True)


def main(argv: list[str] | None = None) -> None:
    """Run the known-Berger maximal-volume calibration."""
    parser = argparse.ArgumentParser(description=__doc__)
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
    payload = run_calibration(settings)
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
