"""Endpoint-adapted weighted charts derived mechanically from the q-system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from mpmath import mp

from .initial_data import ProblemParameters
from .q_system import q_rhs
from .types import State


def _shift_up(coeffs: Iterable, power: int, order: int) -> list:
    """Multiply one coefficient list by tau**power and truncate."""
    items = list(coeffs)
    return ([mp.zero] * power + items)[: order + 1]


def _shift_down(coeffs: Iterable, power: int, order: int) -> list:
    """Divide one coefficient list by tau**power, assuming divisibility."""
    items = list(coeffs)
    return (items[power:] + [mp.zero] * power)[: order + 1]


def _tau_power(tau: Any, power: int) -> Any:
    """Raise one scalar or truncated series to a small nonnegative power."""
    if power == 0:
        return mp.one
    result = tau
    for _ in range(power - 1):
        result = result * tau
    return result


def _regularized_division(numerator: Any, tau: Any, weight: int) -> Any:
    """Divide by tau**weight after cleaning tiny forbidden low-order terms."""
    if hasattr(numerator, "coeffs"):
        coeffs = list(numerator.coeffs)
        scale = max(mp.one, max(abs(value) for value in coeffs))
        tolerance = mp.power(10, -(max(10, mp.dps // 2 - 10)))
        for degree in range(weight):
            if abs(coeffs[degree]) <= tolerance * scale:
                coeffs[degree] = mp.zero
        numerator = type(numerator)(tuple(coeffs))
    return numerator / _tau_power(tau, weight)


@dataclass(frozen=True)
class WeightedChart:
    """One endpoint-adapted weighted chart for the singular q-system."""

    name: str
    weights: tuple[int, ...]
    local_sign: int
    offset_builder: Callable[[ProblemParameters], State[Any]]
    time_builder: Callable[[Any, ProblemParameters], Any]

    def physical_t(self, tau: Any, params: ProblemParameters) -> Any:
        """Convert the local chart time into the physical t variable."""
        return self.time_builder(tau, params)

    def offsets(self, params: ProblemParameters) -> State[Any]:
        """Return the constant q-offsets for this chart."""
        return self.offset_builder(params)

    def local_q_rhs(self, tau: Any, q: State[Any], params: ProblemParameters) -> State[Any]:
        """Return dq/dtau in this chart."""
        return self.local_sign * q_rhs(self.physical_t(tau, params), q, params)

    def physical_qdot(self, local_qdot: State[Any]) -> State[Any]:
        """Convert dq/dtau into dq/dt."""
        return self.local_sign * local_qdot

    def y_to_q(self, tau: Any, y: State[Any], params: ProblemParameters) -> State[Any]:
        """Convert one local weighted y-state into the corresponding q-state."""
        offsets = self.offsets(params)
        components = []
        for offset, weight, value in zip(offsets, self.weights, y):
            components.append(offset + _tau_power(tau, weight) * value)
        return State.from_iterable(components)

    def q_to_y(self, tau: Any, q: State[Any], params: ProblemParameters) -> State[Any]:
        """Recover one local weighted y-state from q at a positive local time."""
        offsets = self.offsets(params)
        components = []
        for offset, weight, value in zip(offsets, self.weights, q):
            components.append((value - offset) / _tau_power(tau, weight))
        return State.from_iterable(components)

    def local_qdot_to_ydot(self, tau: Any, y: State[Any], qdot: State[Any]) -> State[Any]:
        """Recover dy/dtau from y and dq/dtau in this chart."""
        components = []
        for weight, y_value, qdot_value in zip(self.weights, y, qdot):
            correction = weight * _tau_power(tau, weight - 1) * y_value
            components.append(_regularized_division(qdot_value - correction, tau, weight))
        return State.from_iterable(components)

    def y_rhs(self, tau: Any, y: State[Any], params: ProblemParameters) -> State[Any]:
        """Return the weighted y-system derived from q by the chain rule."""
        q = self.y_to_q(tau, y, params)
        qdot = self.local_q_rhs(tau, q, params)
        return self.local_qdot_to_ydot(tau, y, qdot)

    def y_series_to_q_series(self, y_series: State[list], params: ProblemParameters, order: int) -> State[list]:
        """Convert local y Taylor coefficients into q Taylor coefficients."""
        q_coeffs = []
        for weight, coeffs in zip(self.weights, y_series):
            q_coeffs.append(_shift_up(coeffs, weight, order))
        for component, offset in zip(q_coeffs, self.offsets(params)):
            component[0] += offset
        return State.from_iterable(q_coeffs)

    def q_series_to_y_series(self, q_series: State[list], params: ProblemParameters, order: int) -> State[list]:
        """Recover local y Taylor coefficients from q Taylor coefficients."""
        q_coeffs = [list(component) for component in q_series]
        for component, offset in zip(q_coeffs, self.offsets(params)):
            component[0] -= offset
        y_coeffs = []
        for weight, coeffs in zip(self.weights, q_coeffs):
            y_coeffs.append(_shift_down(coeffs, weight, order))
        return State.from_iterable(y_coeffs)


def _left_offsets(params: ProblemParameters) -> State[Any]:
    """Return the left-end q offsets."""
    return State(
        params.left.a,
        mp.zero,
        mp.zero,
        params.left.c,
        -3 * params.left.a,
        mp.zero,
        mp.zero,
        -3 * params.left.c,
    )


def _right_offsets(params: ProblemParameters) -> State[Any]:
    """Return the right-end q offsets."""
    return State(
        3 * params.right.f,
        params.right.f,
        mp.zero,
        mp.zero,
        mp.zero,
        mp.zero,
        -3 * params.right.d,
        -params.right.d,
    )


LEFT_CHART = WeightedChart(
    name="left",
    weights=(2, 1, 1, 2, 2, 1, 1, 2),
    local_sign=1,
    offset_builder=_left_offsets,
    time_builder=lambda tau, _params: tau,
)


RIGHT_CHART = WeightedChart(
    name="right",
    weights=(2, 2, 1, 1, 1, 1, 2, 2),
    local_sign=-1,
    offset_builder=_right_offsets,
    time_builder=lambda tau, params: params.interval_end - tau,
)
