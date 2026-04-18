"""Run the q-driven exploratory midpoint solver with baked-in configs."""

from __future__ import annotations

from mpmath import mp

from problem import DEFAULT_CONFIG, DEFAULT_PARAMS, REFINED_CONFIG
from solver import agreement_digits, solve_to_midpoint


def _format_state(label: str, state) -> str:
    """Format one 8-component state with a short heading."""
    lines = [label]
    lines.extend(f"  {idx}: {value}" for idx, value in enumerate(state, start=1))
    return "\n".join(lines)


def _run(label, params, config):
    """Run one configuration and print the midpoint diagnostics."""
    print(f"{label} params: a={params.a}, c={params.c}, alpha={params.alpha}, lambda={params.lam}")
    print(f"{label} config: order={config.series_order}, dps={config.working_dps}, target_dps={config.target_dps}")
    result = solve_to_midpoint(params, config)
    print(f"{label} patches: {len(result.patches)}")
    print(f"{label} centres: {result.diagnostics['patch_centres']}")
    print(
        f"{label} branch minima: "
        f"sum27={result.diagnostics['min_sum27']}, "
        f"sum36={result.diagnostics['min_sum36']}, "
        f"gap={result.diagnostics['max_gap']}, "
        f"product={result.diagnostics['min_product']}"
    )
    print(_format_state(f"{label} midpoint q:", result.midpoint_q))
    print(_format_state(f"{label} midpoint y:", result.midpoint_y))
    print(_format_state(f"{label} midpoint y':", result.midpoint_ydot))
    return result


def _compare(left, right) -> None:
    """Print componentwise agreement for the recovered midpoint defect."""
    print("refinement agreement:")
    for idx, (left_value, right_value) in enumerate(zip(left.midpoint_ydot, right.midpoint_ydot), start=1):
        print(f"  {idx}: {agreement_digits(left_value, right_value)} digits")


def main() -> None:
    """Run the baseline and refined exploratory midpoint solves."""
    with mp.workdps(max(DEFAULT_CONFIG.working_dps, REFINED_CONFIG.working_dps)):
        baseline = _run("baseline", DEFAULT_PARAMS, DEFAULT_CONFIG)
        refined = _run("refined", DEFAULT_PARAMS, REFINED_CONFIG)
    _compare(baseline, refined)


if __name__ == "__main__":
    main()
