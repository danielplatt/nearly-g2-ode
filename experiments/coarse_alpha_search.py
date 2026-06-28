"""Run a small hand-picked alpha search before any automated minimisation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

from mpmath import mp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problem import DEFAULT_CONFIG, DEFAULT_PARAMS, ProblemParameters, SolverConfig
from solver import solve_to_midpoint


SEARCH_CONFIG = SolverConfig(
    series_order=DEFAULT_CONFIG.series_order,
    working_dps=DEFAULT_CONFIG.working_dps,
    target_dps=DEFAULT_CONFIG.target_dps,
    step_safety=DEFAULT_CONFIG.step_safety,
    sample_points=DEFAULT_CONFIG.sample_points,
    target_t=DEFAULT_CONFIG.target_t,
)


def alpha_candidates() -> list[tuple[str, mp.mpf]]:
    """Return the four guessed alpha values discussed before search."""
    unit = mp.sqrt(5) / 50
    return [
        ("+sqrt(5)/50", unit),
        ("0", mp.zero),
        ("-2*sqrt(5)/25", -4 * unit),
        ("+2*sqrt(5)/25", 4 * unit),
    ]


def defect_norm(values) -> mp.mpf:
    """Return the infinity norm of one midpoint defect vector."""
    return max(abs(value) for value in values)


def run_candidate(label: str, alpha: mp.mpf, config: SolverConfig) -> tuple[str, mp.mpf, mp.mpf | None, object]:
    """Solve to the midpoint for one alpha candidate and return diagnostics."""
    params = replace(DEFAULT_PARAMS, alpha=alpha)
    try:
        result = solve_to_midpoint(params, config)
    except Exception as exc:  # noqa: BLE001
        return label, alpha, None, exc
    return label, alpha, defect_norm(result.midpoint_ydot), result.midpoint_ydot


def print_candidate(label: str, alpha: mp.mpf, norm: mp.mpf | None, defect) -> None:
    """Print the midpoint defect data for one candidate."""
    print(f"{label}: alpha={alpha}")
    if norm is None:
        print(f"  failed: {defect}")
        return
    print(f"  ||y'(pi/6)||_inf = {norm}")
    for index, value in enumerate(defect, start=1):
        print(f"  y{index}'(pi/6) = {value}")


def print_summary(rows: list[tuple[str, mp.mpf, mp.mpf | None, object]]) -> None:
    """Print a short ranking by midpoint defect size."""
    print("summary by ||y'(pi/6)||_inf:")
    successes = [row for row in rows if row[2] is not None]
    failures = [row for row in rows if row[2] is None]
    for label, alpha, norm, _ in sorted(successes, key=lambda row: row[2]):
        print(f"  {label}: alpha={alpha}, norm={norm}")
    for label, alpha, _, error in failures:
        print(f"  {label}: alpha={alpha}, failed={error}")


def main() -> None:
    """Run the four coarse alpha candidates at the refined config."""
    rows = []
    for label, alpha in alpha_candidates():
        rows.append(run_candidate(label, alpha, SEARCH_CONFIG))
    for row in rows:
        print_candidate(*row)
    print_summary(rows)


if __name__ == "__main__":
    main()
