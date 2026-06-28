"""Probe the missing Berger mu/square-root branch before launching a scout."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

from mpmath import mp

from problem import DEFAULT_PARAMS, LEFT_CHART, RIGHT_CHART, ProblemParameters, mirrored_problem_parameters
from problem.taylor_seed import weighted_m_minus_one_residual


OUTPUT_DIR = Path("output/berger_mu_branch_probes")
PROBE_VERSION = "mu-branch-probe-v1"
DEFAULT_P_SIGNS = (-1, 1, 1)
OPPOSITE_LEFT_MU = 1
DEFAULT_TOLERANCE = "1e-30"


@dataclass(frozen=True)
class BranchRecord:
    """One singular-end residual check for a mirrored Berger endpoint package."""

    p_signs: tuple[int, int, int]
    left_mu: int
    right_mu: int
    left_residual: mp.mpf | None
    right_residual: mp.mpf | None
    left_failure: str | None
    right_failure: str | None
    right_p_signs: tuple[int, int, int] | None = None

    def two_sided_ok(self, tolerance: mp.mpf) -> bool:
        """Return whether both singular-end residuals vanish to tolerance."""
        if self.left_residual is None or self.right_residual is None:
            return False
        return self.left_residual < tolerance and self.right_residual < tolerance

    def nondefault(self) -> bool:
        """Return whether this differs from the already explored Berger branch."""
        return (
            self.p_signs != DEFAULT_P_SIGNS
            or self.right_p_signs is not None
            or self.left_mu != -1
            or self.right_mu != -1
        )


def _mp_string(value: mp.mpf | None) -> str | None:
    """Return a stable JSON string for one mpmath value."""
    if value is None:
        return None
    if value == mp.inf:
        return "Infinity"
    return mp.nstr(value, 80)


def _max_abs(values) -> mp.mpf:
    """Return the largest absolute value in an iterable."""
    return max(abs(value) for value in values)


def _branch_params(
    p_signs: tuple[int, int, int],
    left_mu: int,
    right_mu: int,
    right_p_signs: tuple[int, int, int] | None = None,
) -> ProblemParameters:
    """Return mirrored Berger endpoint data on one explicit p/mu branch."""
    return mirrored_problem_parameters(
        DEFAULT_PARAMS.left.a,
        DEFAULT_PARAMS.left.c,
        DEFAULT_PARAMS.left.alpha,
        DEFAULT_PARAMS.lam,
        DEFAULT_PARAMS.interval_end,
        left_mu=left_mu,
        right_mu=right_mu,
        p_signs=p_signs,
        right_p_signs=right_p_signs,
    )


def _side_residual(chart, params: ProblemParameters) -> tuple[mp.mpf | None, str | None]:
    """Return one endpoint M_-1 residual norm, converting failures into data."""
    try:
        residual = weighted_m_minus_one_residual(chart, params)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return None, str(exc)
    return _max_abs(residual), None


def _branch_record(
    p_signs: tuple[int, int, int],
    left_mu: int,
    right_mu: int,
    right_p_signs: tuple[int, int, int] | None = None,
) -> BranchRecord:
    """Evaluate both singular endpoints for one mirrored branch choice."""
    params = _branch_params(p_signs, left_mu, right_mu, right_p_signs)
    left_residual, left_failure = _side_residual(LEFT_CHART, params)
    right_residual, right_failure = _side_residual(RIGHT_CHART, params)
    return BranchRecord(
        p_signs=p_signs,
        left_mu=left_mu,
        right_mu=right_mu,
        left_residual=left_residual,
        right_residual=right_residual,
        left_failure=left_failure,
        right_failure=right_failure,
        right_p_signs=right_p_signs,
    )


def enumerate_branch_records() -> list[BranchRecord]:
    """Check all mirrored Berger p-sign and left/right mu choices."""
    records = []
    for p_signs in product((-1, 1), repeat=3):
        for left_mu in (-1, 1):
            for right_mu in (-1, 1):
                records.append(_branch_record(tuple(p_signs), left_mu, right_mu))
    return records


def enumerate_mixed_opposite_mu_records() -> list[BranchRecord]:
    """Check endpoint-local p-sign choices for the opposite-mu Berger branch."""
    records = []
    for left_p_signs in product((-1, 1), repeat=3):
        for right_p_signs in product((-1, 1), repeat=3):
            records.append(_branch_record(tuple(left_p_signs), 1, 1, tuple(right_p_signs)))
    return records


def _record_payload(record: BranchRecord) -> dict:
    """Return JSON-ready data for one branch record."""
    payload = {
        "p_signs": list(record.p_signs),
        "left_mu": record.left_mu,
        "right_mu": record.right_mu,
        "left_residual": _mp_string(record.left_residual),
        "right_residual": _mp_string(record.right_residual),
        "left_failure": record.left_failure,
        "right_failure": record.right_failure,
    }
    if record.right_p_signs is not None:
        payload["right_p_signs"] = list(record.right_p_signs)
    return payload


def build_summary(*, dps: int = 80, tolerance: mp.mpf | str = DEFAULT_TOLERANCE) -> dict:
    """Return the complete branch-probe summary."""
    with mp.workdps(dps):
        tolerance = mp.mpf(tolerance)
        records = enumerate_branch_records()
        compatible = [record for record in records if record.two_sided_ok(tolerance)]
        nondefault_compatible = [record for record in compatible if record.nondefault()]
        mixed_records = enumerate_mixed_opposite_mu_records()
        mixed_compatible = [
            record
            for record in mixed_records
            if record.two_sided_ok(tolerance) and record.p_signs != record.right_p_signs
        ]
        left_opposite = [
            record
            for record in records
            if record.left_mu == OPPOSITE_LEFT_MU
            and record.left_residual is not None
            and record.left_residual < tolerance
        ]
        right_opposite = [
            record
            for record in records
            if record.right_mu == OPPOSITE_LEFT_MU
            and record.right_residual is not None
            and record.right_residual < tolerance
        ]
        return {
            "probe_version": PROBE_VERSION,
            "working_dps": dps,
            "tolerance": _mp_string(tolerance),
            "default_p_signs": list(DEFAULT_P_SIGNS),
            "records": [_record_payload(record) for record in records],
            "compatible_two_sided": [_record_payload(record) for record in compatible],
            "nondefault_compatible_two_sided": [_record_payload(record) for record in nondefault_compatible],
            "mixed_opposite_mu_compatible": [_record_payload(record) for record in mixed_compatible],
            "left_opposite_mu_one_sided": [_record_payload(record) for record in left_opposite],
            "right_opposite_mu_one_sided": [_record_payload(record) for record in right_opposite],
            "global_scout_ready": bool(nondefault_compatible),
            "mixed_endpoint_scout_ready": bool(mixed_compatible),
            "scout_ready": bool(nondefault_compatible or mixed_compatible),
        }


def _output_path(now: datetime | None = None) -> Path:
    """Return a timestamped output path for one branch probe."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return OUTPUT_DIR / f"{timestamp}-{PROBE_VERSION}.json"


def write_summary(summary: dict, path: Path | None = None) -> Path:
    """Write one branch-probe summary and return its path."""
    if path is None:
        path = _output_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _format_record(record: dict) -> str:
    """Return one compact record line for terminal output."""
    p_label = f"p={tuple(record['p_signs'])}"
    if "right_p_signs" in record:
        p_label = f"left_p={tuple(record['p_signs'])}, right_p={tuple(record['right_p_signs'])}"
    return (
        f"{p_label}, left_mu={record['left_mu']}, right_mu={record['right_mu']}, "
        f"left={record['left_residual']}, right={record['right_residual']}"
    )


def print_summary(summary: dict, *, output_path: Path | None = None) -> None:
    """Print the human-facing branch-probe verdict."""
    print("Berger mu/square-root branch probe", flush=True)
    print(f"version: {summary['probe_version']}", flush=True)
    print(f"working dps: {summary['working_dps']}", flush=True)
    print(f"tolerance: {summary['tolerance']}", flush=True)
    print("", flush=True)
    print("two-sided singular-compatible mirrored branches:", flush=True)
    for record in summary["compatible_two_sided"]:
        print(f"  {_format_record(record)}", flush=True)
    if not summary["compatible_two_sided"]:
        print("  none", flush=True)
    print("", flush=True)
    print("mixed endpoint-local opposite-mu compatible branches:", flush=True)
    for record in summary["mixed_opposite_mu_compatible"]:
        print(f"  {_format_record(record)}", flush=True)
    if not summary["mixed_opposite_mu_compatible"]:
        print("  none", flush=True)
    print("", flush=True)
    print("opposite-mu one-sided cancellations:", flush=True)
    print("  left endpoint:", flush=True)
    for record in summary["left_opposite_mu_one_sided"]:
        print(f"    {_format_record(record)}", flush=True)
    if not summary["left_opposite_mu_one_sided"]:
        print("    none", flush=True)
    print("  right endpoint:", flush=True)
    for record in summary["right_opposite_mu_one_sided"]:
        print(f"    {_format_record(record)}", flush=True)
    if not summary["right_opposite_mu_one_sided"]:
        print("    none", flush=True)
    print("", flush=True)
    if summary["global_scout_ready"]:
        print("verdict: a non-default global-p two-sided branch is scout-ready.", flush=True)
    elif summary["mixed_endpoint_scout_ready"]:
        print("verdict: no non-default global-p branch is ready, but a mixed endpoint-local branch is scout-ready.", flush=True)
    else:
        print("verdict: no non-default mirrored two-sided mu branch is scout-ready.", flush=True)
    if output_path is not None:
        print(f"summary: {output_path}", flush=True)


def _positive_int(value: str) -> int:
    """Parse one positive integer CLI argument."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_mpf_string(value: str) -> str:
    """Validate one positive mpmath value while preserving its decimal text."""
    with mp.workdps(80):
        parsed = mp.mpf(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse branch-probe CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dps", type=_positive_int, default=80, help="working decimal precision")
    parser.add_argument(
        "--tolerance",
        type=_positive_mpf_string,
        default=DEFAULT_TOLERANCE,
        help="zero residual tolerance",
    )
    parser.add_argument("--no-write", action="store_true", help="print only; do not write a JSON summary")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the branch probe from the command line."""
    args = parse_args(argv)
    summary = build_summary(dps=args.dps, tolerance=args.tolerance)
    output_path = None if args.no_write else write_summary(summary)
    print_summary(summary, output_path=output_path)
    return summary


if __name__ == "__main__":
    main()
