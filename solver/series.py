"""Generic truncated power-series utilities for high-precision exploration."""

from __future__ import annotations

from dataclasses import dataclass

from mpmath import mp

from problem.types import State


@dataclass(frozen=True)
class Series:
    """A truncated power series in one local variable."""

    coeffs: tuple[mp.mpf, ...]

    def __post_init__(self) -> None:
        """Clean tiny roundoff coefficients introduced by cancellation."""
        threshold = mp.power(10, -(mp.dps // 2 + 10))
        cleaned = tuple(mp.zero if abs(value) < threshold else mp.mpf(value) for value in self.coeffs)
        object.__setattr__(self, "coeffs", cleaned)

    @classmethod
    def zero(cls, order: int) -> "Series":
        """Create the zero series through the requested order."""
        return cls(tuple(mp.zero for _ in range(order + 1)))

    @classmethod
    def constant(cls, value: mp.mpf, order: int) -> "Series":
        """Create one constant series."""
        coeffs = [mp.zero for _ in range(order + 1)]
        coeffs[0] = value
        return cls(tuple(coeffs))

    @classmethod
    def variable(cls, order: int) -> "Series":
        """Create the local variable x with x + O(x^(order+1))."""
        coeffs = [mp.zero for _ in range(order + 1)]
        if order >= 1:
            coeffs[1] = mp.one
        return cls(tuple(coeffs))

    @property
    def order(self) -> int:
        """Return the truncation order."""
        return len(self.coeffs) - 1

    def coeff(self, degree: int) -> mp.mpf:
        """Return one coefficient, or zero beyond the truncation order."""
        return self.coeffs[degree] if 0 <= degree <= self.order else mp.zero

    def valuation(self) -> int:
        """Return the first degree with a nonzero coefficient."""
        for degree, value in enumerate(self.coeffs):
            if value != 0:
                return degree
        return self.order + 1

    def shift(self, power: int) -> "Series":
        """Multiply by x**power, or divide when power is negative."""
        if power >= 0:
            coeffs = [mp.zero] * power + list(self.coeffs)
            return Series(tuple(coeffs[: self.order + 1]))
        if self.valuation() < -power:
            raise ZeroDivisionError("Series is not divisible by the requested power.")
        coeffs = list(self.coeffs[-power :]) + [mp.zero] * (-power)
        return Series(tuple(coeffs[: self.order + 1]))

    def _coerce(self, other: object) -> "Series":
        """Convert one scalar or series to the current truncation order."""
        if isinstance(other, Series):
            return other if other.order == self.order else other.truncate(self.order)
        return Series.constant(mp.mpf(other), self.order)

    def truncate(self, order: int) -> "Series":
        """Truncate or extend the series to a new order."""
        coeffs = list(self.coeffs[: order + 1])
        coeffs.extend(mp.zero for _ in range(order + 1 - len(coeffs)))
        return Series(tuple(coeffs))

    def __add__(self, other: object) -> "Series":
        """Add one scalar or series."""
        rhs = self._coerce(other)
        return Series(tuple(left + right for left, right in zip(self.coeffs, rhs.coeffs)))

    def __radd__(self, other: object) -> "Series":
        """Add one scalar on the left."""
        return self + other

    def __sub__(self, other: object) -> "Series":
        """Subtract one scalar or series."""
        rhs = self._coerce(other)
        return Series(tuple(left - right for left, right in zip(self.coeffs, rhs.coeffs)))

    def __rsub__(self, other: object) -> "Series":
        """Subtract this series from one scalar or series."""
        return self._coerce(other) - self

    def __neg__(self) -> "Series":
        """Negate the series."""
        return Series(tuple(-value for value in self.coeffs))

    def __mul__(self, other: object) -> "Series":
        """Multiply by one scalar or series via truncated Cauchy product."""
        rhs = self._coerce(other)
        coeffs = [mp.zero for _ in range(self.order + 1)]
        for degree in range(self.order + 1):
            coeffs[degree] = sum(self.coeff(k) * rhs.coeff(degree - k) for k in range(degree + 1))
        return Series(tuple(coeffs))

    def __rmul__(self, other: object) -> "Series":
        """Multiply by one scalar on the left."""
        return self * other

    def _unit_quotient(self, other: "Series") -> list[mp.mpf]:
        """Divide two series with nonzero constant terms."""
        coeffs = [mp.zero for _ in range(self.order + 1)]
        coeffs[0] = self.coeff(0) / other.coeff(0)
        for degree in range(1, self.order + 1):
            total = sum(other.coeff(k) * coeffs[degree - k] for k in range(1, degree + 1))
            coeffs[degree] = (self.coeff(degree) - total) / other.coeff(0)
        return coeffs

    def __truediv__(self, other: object) -> "Series":
        """Divide by one scalar or series when the quotient is regular."""
        rhs = self._coerce(other)
        if rhs.valuation() > rhs.order:
            raise ZeroDivisionError("Cannot divide by the zero series.")
        if not isinstance(other, Series):
            return Series(tuple(value / rhs.coeff(0) for value in self.coeffs))
        num_val = self.valuation()
        den_val = rhs.valuation()
        if num_val < den_val:
            raise ZeroDivisionError("Series quotient would have negative valuation.")
        num = self.shift(-num_val)
        den = rhs.shift(-den_val)
        return Series(tuple(num._unit_quotient(den))).shift(num_val - den_val)

    def __rtruediv__(self, other: object) -> "Series":
        """Divide one scalar or series by this series."""
        return self._coerce(other) / self

    def sqrt(self) -> "Series":
        """Take the square root on the chosen real branch."""
        val = self.valuation()
        if val > self.order:
            return Series.zero(self.order)
        if val % 2:
            raise ValueError("Series square root requires even valuation.")
        shifted = self.shift(-val)
        lead = shifted.coeff(0)
        coeffs = [mp.zero for _ in range(self.order - val + 1)]
        coeffs[0] = mp.sqrt(lead)
        for degree in range(1, len(coeffs)):
            total = sum(coeffs[k] * coeffs[degree - k] for k in range(1, degree))
            coeffs[degree] = (shifted.coeff(degree) - total) / (2 * coeffs[0])
        return Series(tuple(coeffs)).shift(val // 2).truncate(self.order)

    def derivative(self) -> "Series":
        """Differentiate the series."""
        coeffs = [mp.zero for _ in range(self.order + 1)]
        for degree in range(self.order):
            coeffs[degree] = (degree + 1) * self.coeff(degree + 1)
        return Series(tuple(coeffs))

    def integral(self) -> "Series":
        """Integrate the series with zero constant of integration."""
        coeffs = [mp.zero for _ in range(self.order + 1)]
        for degree in range(1, self.order + 1):
            coeffs[degree] = self.coeff(degree - 1) / degree
        return Series(tuple(coeffs))

    def evaluate(self, x: mp.mpf) -> mp.mpf:
        """Evaluate the series polynomial at one point."""
        value = mp.zero
        for coeff in reversed(self.coeffs):
            value = value * x + coeff
        return value


def state_to_series(coeffs: State[list[mp.mpf]]) -> State[Series]:
    """Convert coefficient lists into truncated series objects."""
    return State.from_iterable(Series(tuple(component)) for component in coeffs)


def state_to_coefficients(series: State[Series]) -> State[list[mp.mpf]]:
    """Convert truncated series objects back to mutable coefficient lists."""
    return State.from_iterable([coeff for coeff in component.coeffs] for component in series)


def differentiate_coefficients(coeffs: State[list[mp.mpf]]) -> State[list[mp.mpf]]:
    """Differentiate one coefficient state componentwise."""
    return state_to_coefficients(state_to_series(coeffs).map(lambda series: series.derivative()))


def evaluate_coefficients(coeffs: list[mp.mpf], x: mp.mpf) -> mp.mpf:
    """Evaluate one coefficient list at one point."""
    return Series(tuple(coeffs)).evaluate(x)
