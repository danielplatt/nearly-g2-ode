"""Generic Taylor-engine tests on textbook analytic ODEs."""

from __future__ import annotations

from mpmath import mp

from problem.types import State
from solver.series import evaluate_coefficients, state_to_coefficients, state_to_series


def _oscillator_state(u, v, zero) -> State:
    """Embed the 2D oscillator into the project-wide 8-component state type."""
    return State(u, v, zero, zero, zero, zero, zero, zero)


def _oscillator_rhs(state):
    """Return the harmonic-oscillator right-hand side u' = v, v' = -u."""
    zero = state.y1 * 0
    return _oscillator_state(state.y2, -state.y1, zero)


def _build_regular_patch(centre: mp.mpf, initial: State[mp.mpf], order: int):
    """Build one ordinary Taylor patch for a regular first-order system."""
    coeffs = State.from_iterable([[value] + [mp.zero] * order for value in initial])
    for degree in range(order):
        rhs_coeffs = state_to_coefficients(_oscillator_rhs(state_to_series(coeffs)))
        for component, rhs in zip(coeffs, rhs_coeffs):
            component[degree + 1] = rhs[degree] / (degree + 1)
    return coeffs


def _evaluate_patch(coeffs, t: mp.mpf, centre: mp.mpf) -> State[mp.mpf]:
    """Evaluate one regular patch at time t."""
    local = t - centre
    return State.from_iterable(evaluate_coefficients(component, local) for component in coeffs)


def test_series_engine_recovers_cosine_and_sine_on_one_patch() -> None:
    """A single Taylor patch should reproduce cos and -sin to high precision."""
    with mp.workdps(80):
        coeffs = _build_regular_patch(mp.zero, _oscillator_state(mp.one, mp.zero, mp.zero), 20)
        value = _evaluate_patch(coeffs, mp.mpf("0.5"), mp.zero)
        assert abs(value.y1 - mp.cos(mp.mpf("0.5"))) < mp.mpf("1e-18")
        assert abs(value.y2 + mp.sin(mp.mpf("0.5"))) < mp.mpf("1e-18")


def test_series_engine_reexpands_cleanly_on_a_second_patch() -> None:
    """Re-expanding from one patch to the next should stay close to the exact solution."""
    with mp.workdps(80):
        first = _build_regular_patch(mp.zero, _oscillator_state(mp.one, mp.zero, mp.zero), 16)
        mid = _evaluate_patch(first, mp.mpf("0.6"), mp.zero)
        second = _build_regular_patch(mp.mpf("0.6"), mid, 16)
        value = _evaluate_patch(second, mp.mpf("1.2"), mp.mpf("0.6"))
        assert abs(value.y1 - mp.cos(mp.mpf("1.2"))) < mp.mpf("1e-15")
        assert abs(value.y2 + mp.sin(mp.mpf("1.2"))) < mp.mpf("1e-15")
