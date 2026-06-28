"""Reusable two-sided weighted experiment reporter."""

from __future__ import annotations

from mpmath import mp

from problem import DEFAULT_CONFIG, DEFAULT_PARAMS, REFINED_CONFIG, ProblemParameters, SolverConfig
from solver import TwoSidedResult, agreement_digits, solve_two_sided


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


def _run(label: str, params: ProblemParameters, config: SolverConfig) -> TwoSidedResult:
    """Run one two-sided configuration and print the matching diagnostics."""
    right_label = f"right chart={params.right_chart}"
    if params.fixed_right is not None:
        right_label += f", fixed offset={params.fixed_right.offset}"
    else:
        right_label += f", d={params.right.d}, f={params.right.f}, omega={params.right.omega}"
    print(
        f"{label} left params: a={params.left.a}, c={params.left.c}, alpha={params.left.alpha}; "
        f"{right_label}; "
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


def _compare(baseline: TwoSidedResult, refined: TwoSidedResult) -> None:
    """Print componentwise refinement agreement for the two-sided diagnostics."""
    print("refinement agreement:")
    for idx, (left_value, right_value) in enumerate(zip(baseline.mismatch_q, refined.mismatch_q), start=1):
        print(f"  mismatch q[{idx}]: {agreement_digits(left_value, right_value)} digits")
    print(f"  left l: {agreement_digits(baseline.left_l, refined.left_l)} digits")
    print(f"  right l: {agreement_digits(baseline.right_l, refined.right_l)} digits")


def _run_or_report_failure(
    label: str,
    params: ProblemParameters,
    config: SolverConfig,
    allow_failure: bool,
) -> TwoSidedResult | None:
    """Run one configuration, optionally reporting branch failures as diagnostics."""
    try:
        return _run(label, params, config)
    except ValueError as exc:
        if not allow_failure:
            raise
        print(f"{label} failed before reaching the match point: {exc}")
        return None


def run_experiment(
    label: str,
    params: ProblemParameters,
    baseline_config: SolverConfig,
    refined_config: SolverConfig,
    *,
    allow_failure: bool = False,
) -> tuple[TwoSidedResult | None, TwoSidedResult | None]:
    """Run baseline and refined two-sided diagnostics for one parameter point."""
    with mp.workdps(max(baseline_config.working_dps, refined_config.working_dps)):
        baseline = _run_or_report_failure(f"{label} baseline", params, baseline_config, allow_failure)
        if baseline is None:
            return None, None
        refined = _run_or_report_failure(f"{label} refined", params, refined_config, allow_failure)
    if refined is not None:
        _compare(baseline, refined)
    return baseline, refined


def main() -> None:
    """Run the baseline and refined two-sided Berger validation."""
    run_experiment("berger", DEFAULT_PARAMS, DEFAULT_CONFIG, REFINED_CONFIG)


if __name__ == "__main__":
    main()
