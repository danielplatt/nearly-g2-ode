"""Public solver exports for the two-sided weighted exploration code."""

from __future__ import annotations

from importlib import import_module


__all__ = ["BranchSample", "SeriesPatch", "SideResult", "TwoSidedResult", "agreement_digits", "solve_two_sided"]


def __getattr__(name: str):
    """Load the march exports lazily to avoid package import cycles."""
    if name not in __all__:
        raise AttributeError(f"module 'solver' has no attribute {name!r}")
    march = import_module(".march", __name__)
    return getattr(march, name)
