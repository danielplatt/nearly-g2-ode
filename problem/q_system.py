"""Geometric q-equations and branch quantities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mpmath import mp

from .initial_data import ProblemParameters
from .types import State


@dataclass(frozen=True)
class BranchQuantities:
    """Signed branch quantities used by the q-system square roots."""

    sum27: Any
    sum36: Any
    gap: Any
    product: Any


@dataclass(frozen=True)
class PValues:
    """The auxiliary p-values reconstructed from q."""

    p1: Any
    p2: Any
    p3: Any


@dataclass(frozen=True)
class AlphaBetaTerms:
    """Repeated alpha/beta combinations in the q-equations."""

    alpha1: Any
    alpha2: Any
    alpha3: Any
    alpha4: Any
    beta1: Any
    beta2: Any
    alpha_sum: Any


def _sqrt(value: Any) -> Any:
    """Take one square root for scalars or truncated series."""
    return value.sqrt() if hasattr(value, "sqrt") else mp.sqrt(value)


def branch_quantities(_t: Any, q: State[Any], _params: ProblemParameters) -> BranchQuantities:
    """Return the four branch quantities that control the square roots."""
    q1, q2, q3, q4, q5, q6, q7, q8 = q
    sum27 = q2 + q7
    sum36 = q3 + q6
    gap = q4 + q5
    return BranchQuantities(sum27=sum27, sum36=sum36, gap=gap, product=-(sum27 * sum36 * gap))


def p_values(t: Any, q: State[Any], params: ProblemParameters) -> PValues:
    """Return the p-values determined by q and the chosen square-root branch."""
    branch = branch_quantities(t, q, params)
    p1 = -_sqrt(-(branch.sum27 * branch.sum36) / (params.lam * branch.gap))
    p2 = _sqrt(-(branch.sum27 * branch.gap) / (params.lam * branch.sum36))
    p3 = _sqrt(-(branch.sum36 * branch.gap) / (params.lam * branch.sum27))
    return PValues(p1=p1, p2=p2, p3=p3)


def alpha_beta_terms(q: State[Any]) -> AlphaBetaTerms:
    """Build the repeated alpha_i and beta_i combinations."""
    q1, q2, q3, q4, q5, q6, q7, q8 = q
    alpha1 = q1 * q8
    alpha2 = q2 * q7
    alpha3 = q3 * q6
    alpha4 = q4 * q5
    beta1 = q1 * q4 * q6 * q7
    beta2 = q2 * q3 * q5 * q8
    return AlphaBetaTerms(
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        alpha4=alpha4,
        beta1=beta1,
        beta2=beta2,
        alpha_sum=alpha1 + alpha2 + alpha3 + alpha4,
    )


def _core(alpha: Any, alpha_sum: Any, beta: Any) -> Any:
    """Build the repeated alpha_i(2 alpha_i - sum alpha) + 2 beta term."""
    return alpha * (2 * alpha - alpha_sum) + 2 * beta


def q1_rhs(q: State[Any], params: ProblemParameters, p: PValues, terms: AlphaBetaTerms) -> Any:
    """Return q1'(t)."""
    return params.lam * _core(terms.alpha1, terms.alpha_sum, terms.beta2) / (2 * p.p1 * p.p2 * p.p3 * q.y8)


def q2_rhs(q: State[Any], params: ProblemParameters, p: PValues, terms: AlphaBetaTerms) -> Any:
    """Return q2'(t)."""
    return p.p3 - params.lam * _core(terms.alpha2, terms.alpha_sum, terms.beta1) / (2 * p.p1 * p.p2 * p.p3 * q.y7)


def q3_rhs(q: State[Any], params: ProblemParameters, p: PValues, terms: AlphaBetaTerms) -> Any:
    """Return q3'(t)."""
    return p.p2 - params.lam * _core(terms.alpha3, terms.alpha_sum, terms.beta1) / (2 * p.p1 * p.p2 * p.p3 * q.y6)


def q4_rhs(q: State[Any], params: ProblemParameters, p: PValues, terms: AlphaBetaTerms) -> Any:
    """Return q4'(t)."""
    return -p.p1 + params.lam * _core(terms.alpha4, terms.alpha_sum, terms.beta2) / (2 * p.p1 * p.p2 * p.p3 * q.y5)


def q5_rhs(q: State[Any], params: ProblemParameters, p: PValues, terms: AlphaBetaTerms) -> Any:
    """Return q5'(t)."""
    return p.p1 - params.lam * _core(terms.alpha4, terms.alpha_sum, terms.beta1) / (2 * p.p1 * p.p2 * p.p3 * q.y4)


def q6_rhs(q: State[Any], params: ProblemParameters, p: PValues, terms: AlphaBetaTerms) -> Any:
    """Return q6'(t)."""
    return -p.p2 + params.lam * _core(terms.alpha3, terms.alpha_sum, terms.beta2) / (2 * p.p1 * p.p2 * p.p3 * q.y3)


def q7_rhs(q: State[Any], params: ProblemParameters, p: PValues, terms: AlphaBetaTerms) -> Any:
    """Return q7'(t)."""
    return -p.p3 + params.lam * _core(terms.alpha2, terms.alpha_sum, terms.beta2) / (2 * p.p1 * p.p2 * p.p3 * q.y2)


def q8_rhs(q: State[Any], params: ProblemParameters, p: PValues, terms: AlphaBetaTerms) -> Any:
    """Return q8'(t)."""
    return -params.lam * _core(terms.alpha1, terms.alpha_sum, terms.beta1) / (2 * p.p1 * p.p2 * p.p3 * q.y1)


def q_rhs(t: Any, q: State[Any], params: ProblemParameters) -> State[Any]:
    """Return the geometric q-system right-hand side."""
    p = p_values(t, q, params)
    terms = alpha_beta_terms(q)
    return State(
        q1_rhs(q, params, p, terms),
        q2_rhs(q, params, p, terms),
        q3_rhs(q, params, p, terms),
        q4_rhs(q, params, p, terms),
        q5_rhs(q, params, p, terms),
        q6_rhs(q, params, p, terms),
        q7_rhs(q, params, p, terms),
        q8_rhs(q, params, p, terms),
    )
