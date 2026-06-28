"""Algebraic S7 p2/p3 right-end offset moduli charts."""

from __future__ import annotations

from mpmath import mp

from problem.q_system import alpha_beta_terms
from problem.types import State


def p3_offset(a: mp.mpf, b: mp.mpf, c: mp.mpf) -> State[mp.mpf]:
    """Return the p3-collapse terminal offset family.

    The collapsed branch sums are q3 + q6 and q4 + q5.  The non-collapsing
    branch sum is q2 + q7 = c - a.
    """
    return State(a, -a, -b, b, -b, b, c, -c)


def p2_offset(a: mp.mpf, b: mp.mpf, c: mp.mpf) -> State[mp.mpf]:
    """Return the p2-collapse terminal offset family.

    The collapsed branch sums are q2 + q7 and q4 + q5.  The non-collapsing
    branch sum is q3 + q6 = c - a.
    """
    return State(a, -b, -a, b, -b, c, b, -c)


def branch_sums(q: State[mp.mpf]) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """Return the three branch sums (q2+q7, q3+q6, q4+q5)."""
    return q.y2 + q.y7, q.y3 + q.y6, q.y4 + q.y5


def p3_offset_defect(q: State[mp.mpf]) -> mp.mpf:
    """Return max defect from the derived p3 offset family."""
    return max(
        abs(value)
        for value in (
            q.y2 + q.y1,
            q.y3 + q.y4,
            q.y5 - q.y3,
            q.y6 - q.y4,
            q.y8 + q.y7,
        )
    )


def p2_offset_defect(q: State[mp.mpf]) -> mp.mpf:
    """Return max defect from the derived p2 offset family."""
    return max(
        abs(value)
        for value in (
            q.y3 + q.y1,
            q.y2 + q.y4,
            q.y5 + q.y4,
            q.y6 + q.y8,
            q.y7 - q.y4,
        )
    )


def leading_core_residual(q: State[mp.mpf]) -> State[mp.mpf]:
    """Return the eight leading numerator cores at a terminal offset.

    These are the polynomial factors multiplying the singular
    ``1 / (p1 p2 p3)`` terms in the raw q-system.  They must vanish at a
    collapsing terminal offset before the weighted p2/p3 chart can be regular.
    """
    terms = alpha_beta_terms(q)
    alpha1, alpha2, alpha3, alpha4 = terms.alpha1, terms.alpha2, terms.alpha3, terms.alpha4
    alpha_sum = terms.alpha_sum

    def core(alpha: mp.mpf, beta: mp.mpf) -> mp.mpf:
        return alpha * (2 * alpha - alpha_sum) + 2 * beta

    return State(
        core(alpha1, terms.beta2),
        core(alpha2, terms.beta1),
        core(alpha3, terms.beta1),
        core(alpha4, terms.beta2),
        core(alpha4, terms.beta1),
        core(alpha3, terms.beta2),
        core(alpha2, terms.beta2),
        core(alpha1, terms.beta1),
    )
