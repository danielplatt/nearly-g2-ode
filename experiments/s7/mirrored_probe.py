"""Historical mirrored round-S7 probe, not the official S7 validation."""

from __future__ import annotations

from mpmath import mp

from problem import DEFAULT_CONFIG, REFINED_CONFIG, ProblemParameters, mirrored_problem_parameters, round_s7_left_parameters
from run_exploration import run_experiment


def build_params() -> ProblemParameters:
    """Return the old mirrored round-S7 probe parameter package."""
    with mp.workdps(80):
        preset = round_s7_left_parameters()
        return mirrored_problem_parameters(
            preset.left.a,
            preset.left.c,
            preset.left.alpha,
            preset.lam,
            mp.pi / 3,
        )


def main() -> None:
    """Run the mirrored probe and allow mathematical failures."""
    run_experiment("mirrored round-S7 probe", build_params(), DEFAULT_CONFIG, REFINED_CONFIG, allow_failure=True)


if __name__ == "__main__":
    main()
