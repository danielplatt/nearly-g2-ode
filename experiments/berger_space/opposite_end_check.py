"""Check Berger agreement away from the singular endpoints."""

from __future__ import annotations

from mpmath import mp

from problem import (
    DEFAULT_CONFIG,
    DEFAULT_PARAMS,
    LEFT_CHART,
    RIGHT_CHART,
    ProblemParameters,
    SolverConfig,
    State,
    left_first_jet,
    left_zero_jet,
    right_first_jet,
    right_zero_jet,
)
from solver.march import _march_side


EPSILON = "0.1"


def _config(target_tau: mp.mpf) -> SolverConfig:
    """Return the moderately accurate config used for this diagnostic."""
    return SolverConfig(
        DEFAULT_CONFIG.series_order,
        DEFAULT_CONFIG.working_dps,
        DEFAULT_CONFIG.target_dps,
        DEFAULT_CONFIG.step_safety,
        DEFAULT_CONFIG.sample_points,
        target_tau,
    )


def _max_abs(state: State[mp.mpf]) -> mp.mpf:
    """Return the infinity norm of one eight-component state."""
    return max(abs(value) for value in state)


def _format_state(state: State[mp.mpf], digits: int = 16) -> str:
    """Format an eight-component vector compactly for terminal output."""
    return "[" + ", ".join(mp.nstr(value, digits) for value in state) + "]"


def _one_jet_model(side: str, eps: mp.mpf, params: ProblemParameters) -> State[mp.mpf]:
    """Return the raw q endpoint model through the first weighted jet."""
    if side == "left":
        return LEFT_CHART.y_to_q(eps, left_zero_jet(params) + eps * left_first_jet(params), params)
    if side == "right":
        return RIGHT_CHART.y_to_q(eps, right_zero_jet(params) + eps * right_first_jet(params), params)
    raise ValueError(f"Unknown endpoint side {side!r}.")


def run_check(eps: mp.mpf | str = EPSILON, params: ProblemParameters = DEFAULT_PARAMS) -> None:
    """Print direct and endpoint-asymptotic Berger agreement checks."""
    mp.dps = 110
    eps = mp.mpf(eps)
    near_left = _direct_pair(eps, params)
    near_right = _direct_pair(params.interval_end - eps, params)

    print(f"Berger opposite-end diagnostic with eps = {mp.nstr(eps, 20)}")
    print(f"T = pi/3 = {mp.nstr(params.interval_end, 30)}")
    _print_direct_agreement(eps, *near_left)
    _print_direct_agreement(params.interval_end - eps, *near_right)
    _print_endpoint_models(eps, params, near_right[0], near_left[1])


def _direct_pair(t: mp.mpf, params: ProblemParameters) -> tuple[State[mp.mpf], State[mp.mpf], int, int]:
    """March independently from both ends and return raw q at physical time t."""
    left = _march_side(LEFT_CHART, t, params, _config(t))
    right_tau = params.interval_end - t
    right = _march_side(RIGHT_CHART, right_tau, params, _config(right_tau))
    return left.match_q, right.match_q, len(left.patches), len(right.patches)


def _print_direct_agreement(
    t: mp.mpf,
    left_q: State[mp.mpf],
    right_q: State[mp.mpf],
    left_patches: int,
    right_patches: int,
) -> None:
    """Print direct raw q agreement at one physical time."""
    mismatch = left_q - right_q
    print()
    print(f"Direct raw q agreement at t = {mp.nstr(t, 30)}")
    print(f"  left/right patch counts = {left_patches} / {right_patches}")
    print(f"  ||q_left(t) - q_right(t)||_inf = {mp.nstr(_max_abs(mismatch), 30)}")
    print(f"  mismatch = {_format_state(mismatch)}")


def _print_endpoint_models(
    eps: mp.mpf,
    params: ProblemParameters,
    left_terminal_q: State[mp.mpf],
    right_terminal_q: State[mp.mpf],
) -> None:
    """Print comparison with the opposite endpoint asymptotic models."""
    left_offset_error = _max_abs(left_terminal_q - RIGHT_CHART.offsets(params))
    right_offset_error = _max_abs(right_terminal_q - LEFT_CHART.offsets(params))
    left_model_error = _max_abs(left_terminal_q - _one_jet_model("right", eps, params))
    right_model_error = _max_abs(right_terminal_q - _one_jet_model("left", eps, params))
    print()
    print("Endpoint asymptotic checks in raw q")
    print(f"  left -> right offset error = {mp.nstr(left_offset_error, 30)}")
    print(f"  left -> right one-jet error = {mp.nstr(left_model_error, 30)}")
    print(f"  right -> left offset error = {mp.nstr(right_offset_error, 30)}")
    print(f"  right -> left one-jet error = {mp.nstr(right_model_error, 30)}")


def main() -> None:
    """Run the default Berger opposite-end diagnostic."""
    run_check()


if __name__ == "__main__":
    main()
