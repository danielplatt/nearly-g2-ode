"""Shared typed containers for the 8-component ODE states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Iterator, TypeVar


T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class State(Generic[T]):
    """Store one indexed 8-component state in the project-brief order."""

    y1: T
    y2: T
    y3: T
    y4: T
    y5: T
    y6: T
    y7: T
    y8: T

    def __iter__(self) -> Iterator[T]:
        """Iterate over the eight components in order."""
        yield from (
            self.y1,
            self.y2,
            self.y3,
            self.y4,
            self.y5,
            self.y6,
            self.y7,
            self.y8,
        )

    @classmethod
    def from_iterable(cls, values: Iterable[T]) -> "State[T]":
        """Build one state from exactly eight values."""
        items = tuple(values)
        if len(items) != 8:
            raise ValueError("State expects exactly 8 components.")
        return cls(*items)

    def map(self, func: Callable[[T], U]) -> "State[U]":
        """Apply one function componentwise."""
        return State.from_iterable(func(value) for value in self)

    def zip_map(self, other: "State[T]", func: Callable[[T, T], U]) -> "State[U]":
        """Apply one binary function componentwise."""
        return State.from_iterable(func(left, right) for left, right in zip(self, other))

    def __add__(self, other: "State[T]") -> "State[T]":
        """Add two states componentwise."""
        return self.zip_map(other, lambda left, right: left + right)

    def __sub__(self, other: "State[T]") -> "State[T]":
        """Subtract two states componentwise."""
        return self.zip_map(other, lambda left, right: left - right)

    def __mul__(self, scalar: object) -> "State[T]":
        """Multiply every component by one scalar on the right."""
        return State.from_iterable(value * scalar for value in self)

    def __rmul__(self, scalar: object) -> "State[T]":
        """Multiply every component by one scalar on the left."""
        return State.from_iterable(scalar * value for value in self)
