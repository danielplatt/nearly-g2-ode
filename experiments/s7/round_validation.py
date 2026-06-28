"""Validate the derived round-S7 p3-terminal chart."""

from __future__ import annotations

from mpmath import mp

from problem import (
    DEFAULT_CONFIG,
    REFINED_CONFIG,
    LeftEndpointPreset,
    ProblemParameters,
    left_first_jet_from_values,
    left_zero_jet_from_values,
    round_s7_candidate_parameters,
    round_s7_left_parameters,
    source_alpha,
)
from run_exploration import run_experiment


def build_left_preset() -> LeftEndpointPreset:
    """Return the trusted left-end round-S7 data."""
    return round_s7_left_parameters()


def build_params() -> ProblemParameters:
    """Return the derived two-ended round-S7 data."""
    return round_s7_candidate_parameters()


def _format_state(label: str, state) -> str:
    """Format one eight-component state for terminal diagnostics."""
    lines = [label]
    lines.extend(f"  y{index}: {value}" for index, value in enumerate(state, start=1))
    return "\n".join(lines)


def _validate_left_stage(preset: LeftEndpointPreset) -> None:
    """Check the source formulas for the left round-S7 endpoint."""
    sqrt5 = mp.sqrt(5)
    left = preset.left
    alpha_from_source = source_alpha(left.a, left.c, preset.lam)
    if abs(alpha_from_source + left.alpha) > mp.mpf("1e-40"):
        raise ValueError("round-S7 left alpha is not the sign-flipped source branch")
    if abs(left.alpha - sqrt5 / 50) > mp.mpf("1e-40"):
        raise ValueError("round-S7 left alpha is not sqrt(5)/50")


def _print_left_stage(preset: LeftEndpointPreset) -> None:
    """Print the currently trusted left-end round-S7 data."""
    left = preset.left
    y0 = left_zero_jet_from_values(left.a, left.c, preset.lam)
    y1 = left_first_jet_from_values(left.a, left.c, left.alpha, preset.lam)
    print("round S7 stage 1: trusted left endpoint")
    print(f"  lambda={preset.lam}")
    print(f"  a={left.a}")
    print(f"  c={left.c}")
    print(f"  alpha={left.alpha}")
    print(_format_state("left zero jet:", y0))
    print(_format_state("left first jet:", y1))


def main() -> None:
    """Validate the round-S7 endpoints and run the two-sided marcher."""
    with mp.workdps(max(DEFAULT_CONFIG.working_dps, REFINED_CONFIG.working_dps)):
        preset = build_left_preset()
        _validate_left_stage(preset)
        _print_left_stage(preset)
        print("round S7 stage 2: right endpoint")
        params = build_params()
        run_experiment("round S7", params, DEFAULT_CONFIG, REFINED_CONFIG, allow_failure=True)


if __name__ == "__main__":
    main()
