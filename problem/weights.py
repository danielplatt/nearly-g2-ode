"""Weighted changes of variables between the q- and y-systems."""

from __future__ import annotations

from typing import Iterable

from mpmath import mp

from .initial_data import ProblemParameters
from .types import State


WEIGHTS = (2, 1, 1, 2, 2, 1, 1, 2)


def _shift_up(coeffs: Iterable, power: int, order: int) -> list:
    """Multiply one coefficient list by t**power and truncate."""
    items = list(coeffs)
    return ([mp.zero] * power + items)[: order + 1]


def _shift_down(coeffs: Iterable, power: int, order: int) -> list:
    """Divide one coefficient list by t**power, assuming divisibility."""
    items = list(coeffs)
    return (items[power:] + [mp.zero] * power)[: order + 1]


def _base_q_values(params: ProblemParameters) -> tuple:
    """Return the constant weighted offsets in the q variables."""
    return (params.a, 0, 0, params.c, -3 * params.a, 0, 0, -3 * params.c)


def y_to_q(t, y: State, params: ProblemParameters) -> State:
    """Convert one pointwise y-state into the corresponding q-state."""
    return State(
        params.a + t**2 * y.y1,
        t * y.y2,
        t * y.y3,
        params.c + t**2 * y.y4,
        -3 * params.a + t**2 * y.y5,
        t * y.y6,
        t * y.y7,
        -3 * params.c + t**2 * y.y8,
    )


def q_to_y(t, q: State, params: ProblemParameters) -> State:
    """Recover one pointwise y-state from q at one positive time t."""
    return State(
        (q.y1 - params.a) / t**2,
        q.y2 / t,
        q.y3 / t,
        (q.y4 - params.c) / t**2,
        (q.y5 + 3 * params.a) / t**2,
        q.y6 / t,
        q.y7 / t,
        (q.y8 + 3 * params.c) / t**2,
    )


def qdot_to_ydot(t, q: State, qdot: State, params: ProblemParameters) -> State:
    """Recover one pointwise y'-state from q and q' at one positive time."""
    y = q_to_y(t, q, params)
    return State(
        (qdot.y1 - 2 * t * y.y1) / t**2,
        (qdot.y2 - y.y2) / t,
        (qdot.y3 - y.y3) / t,
        (qdot.y4 - 2 * t * y.y4) / t**2,
        (qdot.y5 - 2 * t * y.y5) / t**2,
        (qdot.y6 - y.y6) / t,
        (qdot.y7 - y.y7) / t,
        (qdot.y8 - 2 * t * y.y8) / t**2,
    )


def y_series_to_q_series(y_series: State[list], params: ProblemParameters, order: int) -> State[list]:
    """Convert origin-based y Taylor coefficients into q coefficients."""
    base = _base_q_values(params)
    q_coeffs = [
        _shift_up(y_series.y1, 2, order),
        _shift_up(y_series.y2, 1, order),
        _shift_up(y_series.y3, 1, order),
        _shift_up(y_series.y4, 2, order),
        _shift_up(y_series.y5, 2, order),
        _shift_up(y_series.y6, 1, order),
        _shift_up(y_series.y7, 1, order),
        _shift_up(y_series.y8, 2, order),
    ]
    for index, value in enumerate(base):
        q_coeffs[index][0] += value
    return State.from_iterable(q_coeffs)


def q_series_to_y_series(q_series: State[list], params: ProblemParameters, order: int) -> State[list]:
    """Recover origin-based y Taylor coefficients from q coefficients."""
    q_coeffs = [list(component) for component in q_series]
    q_coeffs[0][0] -= params.a
    q_coeffs[3][0] -= params.c
    q_coeffs[4][0] += 3 * params.a
    q_coeffs[7][0] += 3 * params.c
    return State(
        _shift_down(q_coeffs[0], 2, order),
        _shift_down(q_coeffs[1], 1, order),
        _shift_down(q_coeffs[2], 1, order),
        _shift_down(q_coeffs[3], 2, order),
        _shift_down(q_coeffs[4], 2, order),
        _shift_down(q_coeffs[5], 1, order),
        _shift_down(q_coeffs[6], 1, order),
        _shift_down(q_coeffs[7], 2, order),
    )
