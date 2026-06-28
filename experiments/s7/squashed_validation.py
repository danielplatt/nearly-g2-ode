"""Validate the derived squashed-S7 p2-terminal chart."""

from __future__ import annotations

from mpmath import mp

from problem import DEFAULT_CONFIG, REFINED_CONFIG, ProblemParameters, squashed_s7_parameters
from run_exploration import run_experiment


def build_params() -> ProblemParameters:
    """Return the derived squashed-S7 two-ended parameter package."""
    return squashed_s7_parameters()


def main() -> None:
    """Run baseline and refined squashed-S7 matching diagnostics."""
    with mp.workdps(max(DEFAULT_CONFIG.working_dps, REFINED_CONFIG.working_dps)):
        run_experiment("squashed S7", build_params(), DEFAULT_CONFIG, REFINED_CONFIG)


if __name__ == "__main__":
    main()
