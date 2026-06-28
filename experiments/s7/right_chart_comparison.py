"""Compare Berger and round-S7 left marches against the Berger right chart."""

from __future__ import annotations

from dataclasses import dataclass

from mpmath import mp

from problem import (
    DEFAULT_PARAMS,
    LEFT_CHART,
    RIGHT_CHART,
    ProblemParameters,
    SolverConfig,
    State,
    right_first_jet,
    right_zero_jet,
    round_s7_left_parameters,
)
from solver.march import _march_side


EPSILONS = ("0.2", "0.1", "0.05", "0.02")
SERIES_ORDER = 18
WORKING_DPS = 100
TARGET_DPS = 40
STEP_SAFETY = "0.55"


@dataclass(frozen=True)
class ComparisonCase:
    """One left-end solution to compare against the Berger right chart."""

    label: str
    params: ProblemParameters
    scale: mp.mpf
    expected_ratio: tuple[str, ...]
    expectation: str


def _round_s7_left_problem() -> ProblemParameters:
    """Build the round-S7 left problem; the right endpoint is intentionally unused."""
    preset = round_s7_left_parameters()
    return ProblemParameters(
        lam=preset.lam,
        interval_end=DEFAULT_PARAMS.interval_end,
        left=preset.left,
        right=DEFAULT_PARAMS.right,
    )


def _config(target_tau: mp.mpf) -> SolverConfig:
    """Return the cheap-but-stable config for endpoint comparison."""
    return SolverConfig(SERIES_ORDER, WORKING_DPS, TARGET_DPS, mp.mpf(STEP_SAFETY), 0, target_tau)


def _max_abs(state: State[mp.mpf]) -> mp.mpf:
    """Return the infinity norm of one state."""
    return max(abs(value) for value in state)


def _format_state(state: State[mp.mpf], digits: int = 10) -> str:
    """Format an eight-component state compactly."""
    return "[" + ", ".join(mp.nstr(value, digits) for value in state) + "]"


def _terminal_q(params: ProblemParameters, eps: mp.mpf) -> State[mp.mpf]:
    """March from the left endpoint to T - eps and return raw q."""
    side = _march_side(LEFT_CHART, params.interval_end - eps, params, _config(params.interval_end - eps))
    return side.match_q


def _berger_one_jet_model(eps: mp.mpf) -> State[mp.mpf]:
    """Return the validated Berger right endpoint model through first weighted jet."""
    y = right_zero_jet(DEFAULT_PARAMS) + eps * right_first_jet(DEFAULT_PARAMS)
    return RIGHT_CHART.y_to_q(eps, y, DEFAULT_PARAMS)


def _berger_right_form_defect(q: State[mp.mpf]) -> mp.mpf:
    """Measure failure to have any Berger-right offset form at leading order."""
    q1, q2, q3, q4, q5, q6, q7, q8 = q
    return max(abs(value) for value in (q1 - 3 * q2, q3, q4, q5, q6, q7 - 3 * q8))


def _print_berger_chart_expectation() -> None:
    """Print the right-chart asymptotic shape being tested."""
    weights = State.from_iterable(RIGHT_CHART.weights)
    offset = RIGHT_CHART.offsets(DEFAULT_PARAMS)
    print("Berger right-chart asymptotic expectation")
    print("  q1 = 3f + O(s^2), q2 = f + O(s^2)")
    print("  q3,q4,q5,q6 = O(s)")
    print("  q7 = -3d + O(s^2), q8 = -d + O(s^2)")
    print("  chart-form constraints at s=0:")
    print("    q1 - 3q2 = 0, q3=q4=q5=q6=0, q7 - 3q8 = 0")
    print(f"  validated Berger right offset = {_format_state(offset, 16)}")
    print(f"  right-chart weights = {_format_state(weights, 1)}")


def _print_case(case: ComparisonCase) -> None:
    """Print the endpoint comparison table for one case."""
    offset = RIGHT_CHART.offsets(DEFAULT_PARAMS)
    print()
    print(case.label)
    print(f"  expectation: {case.expectation}")
    print(f"  scale used in q/scale: {mp.nstr(case.scale, 20)}")
    print(f"  expected terminal q / scale: [{', '.join(case.expected_ratio)}]")
    print("  eps     form defect     fixed Berger offset error     Berger one-jet error      observed q / scale")
    for eps_text in EPSILONS:
        eps = mp.mpf(eps_text)
        q = _terminal_q(case.params, eps)
        ratios = q.map(lambda value: value / case.scale)
        form_defect = _berger_right_form_defect(q)
        offset_error = _max_abs(q - offset)
        one_jet_error = _max_abs(q - _berger_one_jet_model(eps))
        print(
            f"  {eps_text:<7}"
            f"{mp.nstr(form_defect, 10):>16}"
            f"{mp.nstr(offset_error, 10):>30}"
            f"{mp.nstr(one_jet_error, 10):>24}"
            f"      {_format_state(ratios, 8)}"
        )
    print(f"  conclusion: {case.expectation}")


def _comparison_cases() -> tuple[ComparisonCase, ComparisonCase]:
    """Return the Berger and round-S7 cases for this diagnostic."""
    round_s7 = _round_s7_left_problem()
    berger = DEFAULT_PARAMS
    return (
        ComparisonCase(
            label="Case 1: Berger left data",
            params=berger,
            scale=berger.left.a,
            expected_ratio=("9/5", "3/5", "0", "0", "0", "0", "3", "1"),
            expectation="should match the validated Berger right chart",
        ),
        ComparisonCase(
            label="Case 2: round-S7 left data tested against the Berger right chart",
            params=round_s7,
            scale=round_s7.left.a,
            expected_ratio=("1", "-1", "-2", "2", "-2", "2", "19", "-19"),
            expectation="should not have Berger-right chart form; q3,q4,q5,q6 do not tend to zero",
        ),
    )


def main() -> None:
    """Run the right-chart comparison diagnostic."""
    mp.dps = WORKING_DPS
    _print_berger_chart_expectation()
    for case in _comparison_cases():
        _print_case(case)


if __name__ == "__main__":
    main()
