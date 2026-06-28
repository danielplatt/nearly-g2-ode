"""Public solver exports for the two-sided weighted exploration code."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "BranchSample",
    "BASE_TWO_SIDED_POINT",
    "SeriesPatch",
    "SideResult",
    "TwoSidedJacobianResult",
    "TwoSidedNewtonSettings",
    "TwoSidedResidualResult",
    "TwoSidedResult",
    "TwoSidedSearchPoint",
    "agreement_digits",
    "finite_difference_two_sided_jacobian",
    "solve_left_side",
    "solve_two_sided",
    "two_sided_newton_refine",
    "two_sided_residual",
]


_EXPORT_MODULES = {
    "BranchSample": ".march",
    "SeriesPatch": ".march",
    "SideResult": ".march",
    "TwoSidedResult": ".march",
    "agreement_digits": ".march",
    "solve_left_side": ".march",
    "solve_two_sided": ".march",
    "BASE_TWO_SIDED_POINT": ".two_sided_shooting",
    "TwoSidedJacobianResult": ".two_sided_shooting",
    "TwoSidedResidualResult": ".two_sided_shooting",
    "TwoSidedSearchPoint": ".two_sided_shooting",
    "finite_difference_two_sided_jacobian": ".two_sided_shooting",
    "two_sided_residual": ".two_sided_shooting",
    "TwoSidedNewtonSettings": ".two_sided_refinement",
    "two_sided_newton_refine": ".two_sided_refinement",
}


def __getattr__(name: str):
    """Load public solver exports lazily to avoid package import cycles."""
    if name not in __all__:
        raise AttributeError(f"module 'solver' has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name], __name__)
    return getattr(module, name)
