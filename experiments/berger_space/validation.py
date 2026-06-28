"""Run the validated homogeneous Berger two-sided matching experiment."""

from __future__ import annotations

from problem import DEFAULT_CONFIG, REFINED_CONFIG, ProblemParameters, berger_parameters
from run_exploration import run_experiment


def build_params() -> ProblemParameters:
    """Return the validated Berger parameter package."""
    return berger_parameters()


def main() -> None:
    """Run baseline and refined Berger diagnostics."""
    run_experiment("berger", build_params(), DEFAULT_CONFIG, REFINED_CONFIG)


if __name__ == "__main__":
    main()
