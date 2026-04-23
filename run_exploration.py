"""Run the two-sided weighted Berger validation with baked-in configs."""

from __future__ import annotations

from mpmath import mp

from problem import DEFAULT_CONFIG, DEFAULT_PARAMS, REFINED_CONFIG
from solver import agreement_digits, solve_two_sided


def _format_state(label: str, state) -> str:
    """Format one 8-component state with a short heading."""
    lines = [label]
    lines.extend(f"  {idx}: {value}" for idx, value in enumerate(state, start=1))
    return "\n".join(lines)


def _format_side(label: str, side) -> str:
    """Format one side's match-point diagnostics."""
    lines = [f"{label} patches: {len(side.patches)}", f"{label} local centres: {side.diagnostics['patch_centres']}"]
    lines.append(
        f"{label} branch extrema: "
        f"sum27={side.diagnostics['min_sum27']}, "
        f"sum36={side.diagnostics['min_sum36']}, "
        f"gap={side.diagnostics['max_gap']}, "
        f"product={side.diagnostics['min_product']}"
    )
    lines.append(_format_state(f"{label} midpoint y:", side.match_y))
    lines.append(_format_state(f"{label} midpoint q:", side.match_q))
    lines.append(_format_state(f"{label} midpoint qdot:", side.match_qdot))
    return "\n".join(lines)


def _run(label, params, config):
    """Run one two-sided configuration and print the matching diagnostics."""
    print(
        f"{label} left params: a={params.left.a}, c={params.left.c}, alpha={params.left.alpha}; "
        f"right params: d={params.right.d}, f={params.right.f}, omega={params.right.omega}; "
        f"lambda={params.lam}"
    )
    print(
        f"{label} config: order={config.series_order}, dps={config.working_dps}, "
        f"target_dps={config.target_dps}, match_t={config.match_t}"
    )
    result = solve_two_sided(params, config)
    print(_format_side(f"{label} left", result.left))
    print(_format_side(f"{label} right", result.right))
    print(_format_state(f"{label} midpoint q mismatch:", result.mismatch_q))
    print(f"{label} mismatch norm: {result.mismatch_norm}")
    print(f"{label} l(match_t): left={result.left_l}, right={result.right_l}")
    return result


def _compare(baseline, refined) -> None:
    """Print componentwise refinement agreement for the two-sided diagnostics."""
    print("refinement agreement:")
    for idx, (left_value, right_value) in enumerate(zip(baseline.mismatch_q, refined.mismatch_q), start=1):
        print(f"  mismatch q[{idx}]: {agreement_digits(left_value, right_value)} digits")
    print(f"  left l: {agreement_digits(baseline.left_l, refined.left_l)} digits")
    print(f"  right l: {agreement_digits(baseline.right_l, refined.right_l)} digits")


def main() -> None:
    """Run the baseline and refined two-sided Berger validation."""
    with mp.workdps(max(DEFAULT_CONFIG.working_dps, REFINED_CONFIG.working_dps)):
        baseline = _run("baseline", DEFAULT_PARAMS, DEFAULT_CONFIG)
        refined = _run("refined", DEFAULT_PARAMS, REFINED_CONFIG)
    _compare(baseline, refined)


if __name__ == "__main__":
    main()
