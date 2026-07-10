"""Sp(1) x Sp(1) x U(1) cohomogeneity-one system on S7.

The action is the intermediate-symmetry S7 action from the census:

    G = Sp(1) x Sp(1) x U(1),    H = diagonal U(1).

The principal orbit is ``S3 x S3``.  We use a coframe

    a1, a2, a3, b1, b2, b3

where the diagonal U(1) rotates ``(a1,a2)`` and ``(b1,b2)`` with the same
orientation and fixes ``a3,b3``.  The Maurer-Cartan scale is fixed by the
Podesta five-function subchart: with this normalization the Podesta algebraic
constraint is ``f3 + f4 + lambda*f0^2/6 = 0``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from typing import TypeAlias

import numpy as np
from mpmath import mp

from . import su2_cubed_action_audit


SYSTEM_VERSION = "s7-sp1xsp1xu1-system-v1"
MC_SCALE = mp.mpf(6)
DEFAULT_DPS = 80

Index: TypeAlias = tuple[int, ...]
Form: TypeAlias = dict[Index, mp.mpf]

PRINCIPAL_LABELS = ("a1", "a2", "a3", "b1", "b2", "b3")
OMEGA_BASIS: tuple[Index, ...] = (
    (1, 2),  # x1 = a12
    (4, 5),  # x2 = b12
    (3, 6),  # x3 = a3 b3
    (1, 4),  # x4 starts a1 b1 + a2 b2
    (1, 5),  # x5 starts a1 b2 - a2 b1
)
GAMMA_BASIS: tuple[Form, ...] = (
    {(1, 2, 3): mp.one},  # y1 = a123
    {(3, 4, 5): mp.one},  # y2 = a3 b12
    {(1, 3, 4): -mp.one, (2, 3, 5): -mp.one},  # y3 = a3 ^ (a1b1+a2b2)
    {(1, 3, 5): -mp.one, (2, 3, 4): mp.one},  # y4 = a3 ^ (a1b2-a2b1)
    {(1, 2, 6): mp.one},  # y5 = b3 a12
    {(4, 5, 6): mp.one},  # y6 = b123
    {(1, 4, 6): mp.one, (2, 5, 6): mp.one},  # y7 = b3 ^ (a1b1+a2b2)
    {(1, 5, 6): mp.one, (2, 4, 6): -mp.one},  # y8 = b3 ^ (a1b2-a2b1)
)
GAMMA_BASIS_LABELS = (
    "a123",
    "a3b12",
    "a3_delta",
    "a3_epsilon",
    "b3a12",
    "b123",
    "b3_delta",
    "b3_epsilon",
)
FOUR_BASIS: tuple[Index, ...] = tuple(combinations(range(1, 7), 4))
THREE_BASIS_ALL: tuple[Index, ...] = tuple(combinations(range(1, 7), 3))
TWO_BASIS_ALL: tuple[Index, ...] = tuple(combinations(range(1, 7), 2))


@dataclass(frozen=True)
class EmbeddedTarget:
    """A known Podesta target embedded in the 13-variable U(1) chart."""

    name: str
    lam: mp.mpf
    state_functions: tuple
    state_derivative_functions: tuple


def _clean(value: mp.mpf, tolerance: mp.mpf = mp.mpf("1e-70")) -> mp.mpf:
    return mp.zero if abs(value) < tolerance else value


def _add_to(item: Form, key: Index, value: mp.mpf) -> None:
    new_value = _clean(item.get(key, mp.zero) + value)
    if new_value == 0:
        item.pop(key, None)
    else:
        item[key] = new_value


def basis(indices: Iterable[int], coefficient: mp.mpf | int | float = 1) -> Form:
    """Return a sorted basis wedge form."""
    key = tuple(indices)
    if tuple(sorted(key)) != key or len(set(key)) != len(key):
        raise ValueError(f"indices must be strictly increasing: {key}")
    return {key: mp.mpf(coefficient)} if coefficient != 0 else {}


def scale(value: mp.mpf | int | float, item: Form) -> Form:
    """Scale one sparse form."""
    scalar = mp.mpf(value)
    return {key: _clean(scalar * coefficient) for key, coefficient in item.items() if scalar * coefficient != 0}


def add(*items: Form) -> Form:
    """Add sparse forms."""
    result: Form = {}
    for item in items:
        for key, coefficient in item.items():
            _add_to(result, key, coefficient)
    return result


def subtract(left: Form, right: Form) -> Form:
    """Subtract two sparse forms."""
    return add(left, scale(-1, right))


def _merge_sign(left: Index, right: Index) -> tuple[Index, int] | None:
    if set(left).intersection(right):
        return None
    combined = left + right
    inversions = sum(1 for i, left_index in enumerate(combined) for right_index in combined[i + 1 :] if left_index > right_index)
    return tuple(sorted(combined)), -1 if inversions % 2 else 1


def wedge(left: Form, right: Form) -> Form:
    """Wedge two sparse forms."""
    result: Form = {}
    for left_key, left_coefficient in left.items():
        for right_key, right_coefficient in right.items():
            merged = _merge_sign(left_key, right_key)
            if merged is None:
                continue
            key, sign = merged
            _add_to(result, key, left_coefficient * right_coefficient * sign)
    return result


def contract(index: int, item: Form) -> Form:
    """Contract with the basis vector dual to one principal coform."""
    result: Form = {}
    for key, coefficient in item.items():
        if index not in key:
            continue
        position = key.index(index)
        _add_to(result, key[:position] + key[position + 1 :], coefficient * (mp.mpf(-1) ** position))
    return result


def one_form_differentials(scale_value: mp.mpf = MC_SCALE) -> dict[int, Form]:
    """Return the principal-orbit Maurer-Cartan table."""
    c = mp.mpf(scale_value)
    return {
        1: scale(c, basis((2, 3))),
        2: scale(-c, basis((1, 3))),
        3: scale(c, basis((1, 2))),
        4: scale(c, basis((5, 6))),
        5: scale(-c, basis((4, 6))),
        6: scale(c, basis((4, 5))),
    }


def exterior_derivative(item: Form, scale_value: mp.mpf = MC_SCALE) -> Form:
    """Compute the principal-orbit exterior derivative."""
    differentials = one_form_differentials(scale_value)
    result: Form = {}
    for key, coefficient in item.items():
        for position, index in enumerate(key):
            prefix = basis(key[:position])
            suffix = basis(key[position + 1 :])
            term = wedge(wedge(prefix, differentials[index]), suffix)
            result = add(result, scale((mp.mpf(-1) ** position) * coefficient, term))
    return result


def omega_basis_forms() -> tuple[Form, ...]:
    """Return the five U(1)-invariant 2-form basis elements."""
    return (
        basis((1, 2)),
        basis((4, 5)),
        basis((3, 6)),
        add(basis((1, 4)), basis((2, 5))),
        add(basis((1, 5)), scale(-1, basis((2, 4)))),
    )


def omega_form(x: Iterable[mp.mpf | int | float]) -> Form:
    """Return omega from five coefficients."""
    return add(*(scale(mp.mpf(value), item) for value, item in zip(x, omega_basis_forms())))


def gamma_form(y: Iterable[mp.mpf | int | float]) -> Form:
    """Return gamma=Re(Omega) from eight coefficients."""
    return add(*(scale(mp.mpf(value), item) for value, item in zip(y, GAMMA_BASIS)))


def state_omega_gamma(state: Iterable[mp.mpf | int | float]) -> tuple[Form, Form]:
    """Split a 13-vector into ``omega`` and ``gamma`` forms."""
    values = tuple(mp.mpf(value) for value in state)
    if len(values) != 13:
        raise ValueError("state must have 13 entries")
    return omega_form(values[:5]), gamma_form(values[5:])


def coefficients(item: Form, basis_keys: tuple[Index, ...]) -> tuple[mp.mpf, ...]:
    """Return coefficients against a tuple of basis keys."""
    return tuple(item.get(key, mp.zero) for key in basis_keys)


def gamma_coefficients(item: Form) -> tuple[mp.mpf, ...]:
    """Return coefficients in the eight-element U(1)-invariant 3-form basis."""
    matrix = []
    rhs = []
    for key in THREE_BASIS_ALL:
        matrix.append([basis_item.get(key, mp.zero) for basis_item in GAMMA_BASIS])
        rhs.append(item.get(key, mp.zero))
    a = np.array([[float(value) for value in row] for row in matrix], dtype=float)
    b = np.array([float(value) for value in rhs], dtype=float)
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    residual = np.linalg.norm(a @ solution - b, ord=np.inf)
    if residual > 1e-8:
        raise ValueError(f"3-form is not in the invariant gamma span: residual={residual:g}")
    return tuple(mp.mpf(str(value)) for value in solution)


def _matrix_multiply(left: tuple[tuple[mp.mpf, ...], ...], right: tuple[tuple[mp.mpf, ...], ...]) -> tuple[tuple[mp.mpf, ...], ...]:
    size = len(left)
    return tuple(tuple(sum(left[row][inner] * right[inner][column] for inner in range(size)) for column in range(size)) for row in range(size))


def hitchin_k_matrix(gamma: Form) -> tuple[tuple[mp.mpf, ...], ...]:
    """Return Hitchin's K matrix in the ordered volume trivialization."""
    rows = [[mp.zero for _ in range(6)] for _ in range(6)]
    full_key = (1, 2, 3, 4, 5, 6)
    for column in range(1, 7):
        alpha = wedge(contract(column, gamma), gamma)
        for key, coefficient in alpha.items():
            missing = [index for index in full_key if index not in key]
            if len(missing) != 1:
                continue
            row = missing[0]
            expected_key = tuple(index for index in full_key if index != row)
            if key != expected_key:
                raise ValueError(f"unexpected 5-form key: {key}")
            rows[row - 1][column - 1] = coefficient * (mp.mpf(-1) ** (row - 1))
    return tuple(tuple(row) for row in rows)


def hitchin_lambda(gamma: Form) -> mp.mpf:
    """Return Hitchin's quartic scalar for a 3-form."""
    matrix = hitchin_k_matrix(gamma)
    squared = _matrix_multiply(matrix, matrix)
    return sum(squared[index][index] for index in range(6)) / 6


def _pullback_form(item: Form, coform_matrix: tuple[tuple[mp.mpf, ...], ...]) -> Form:
    """Pull back a form by a linear map on coforms."""
    result: Form = {}
    for key, coefficient in item.items():
        terms: Form = {(): coefficient}
        for index in key:
            one_form = {
                (target + 1,): coform_matrix[index - 1][target]
                for target in range(6)
                if coform_matrix[index - 1][target] != 0
            }
            terms = wedge(terms, one_form)
        result = add(result, terms)
    return result


def hitchin_dual(gamma: Form) -> Form:
    """Return ``hat(gamma)=Im(Omega)`` for the negative stable component."""
    matrix = hitchin_k_matrix(gamma)
    lambda_value = hitchin_lambda(gamma)
    if lambda_value >= 0:
        raise ValueError("gamma is not in the negative stable component")
    root = mp.sqrt(-lambda_value)
    almost_complex = tuple(tuple(value / root for value in row) for row in matrix)
    return scale(-1, _pullback_form(gamma, almost_complex))


def oriented_hitchin_dual(omega: Form, gamma: Form) -> Form:
    """Return the Hitchin dual with the orientation selected by ``omega``.

    Hitchin's negative-stable 3-form determines the dual only up to the
    orientation component.  The cohomogeneity-one SU(3) pair fixes that choice
    by requiring ``gamma wedge hat(gamma)`` to have the same oriented volume as
    ``omega^3``.  This matters for the squashed S7 target, whose principal
    SU(3) orientation is opposite to the round target in this coframe.
    """
    hat = hitchin_dual(gamma)
    gamma_hat_volume = volume_coefficient(wedge(gamma, hat))
    omega_vol = omega_volume(omega)
    if gamma_hat_volume == 0 or omega_vol == 0:
        raise ValueError("cannot orient Hitchin dual from zero volume")
    if gamma_hat_volume * omega_vol < 0:
        return scale(-1, hat)
    return hat


def volume_coefficient(item: Form, dimension: int = 6) -> mp.mpf:
    """Return the coefficient of the ordered volume form."""
    return item.get(tuple(range(1, dimension + 1)), mp.zero)


def omega_volume(omega: Form) -> mp.mpf:
    """Return the coefficient of omega^3/6."""
    return volume_coefficient(wedge(wedge(omega, omega), omega)) / 6


def max_abs_coefficient(item: Form) -> mp.mpf:
    """Return the largest absolute coefficient of a sparse form."""
    return max((abs(value) for value in item.values()), default=mp.zero)


def _solve_omega_dot(omega: Form, rhs_form: Form) -> tuple[np.ndarray, float]:
    """Solve ``omega wedge omega_dot = rhs_form`` in the invariant 2-form basis."""
    basis_forms = omega_basis_forms()
    matrix = np.zeros((len(FOUR_BASIS), len(basis_forms)), dtype=float)
    for column, basis_form in enumerate(basis_forms):
        matrix[:, column] = [float(value) for value in coefficients(wedge(omega, basis_form), FOUR_BASIS)]
    rhs = np.array([float(value) for value in coefficients(rhs_form, FOUR_BASIS)], dtype=float)
    solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    residual = float(np.linalg.norm(matrix @ solution - rhs, ord=np.inf))
    return solution, residual


def rhs(state: Iterable[mp.mpf | int | float], lam: mp.mpf | int | float) -> tuple[mp.mpf, ...]:
    """Return the 13-dimensional nearly-parallel vector field.

    The component equations are

        dot(gamma) = d omega - lambda * hat(gamma),
        omega wedge dot(omega) = -d hat(gamma),

    together with the algebraic constraints recorded by
    :func:`algebraic_residual`.
    """
    omega, gamma = state_omega_gamma(state)
    lambda_value = mp.mpf(lam)
    hat = oriented_hitchin_dual(omega, gamma)
    gamma_dot_form = subtract(exterior_derivative(omega), scale(lambda_value, hat))
    omega_dot, residual = _solve_omega_dot(omega, scale(-1, exterior_derivative(hat)))
    if residual > 1e-7:
        raise ValueError(f"omega-dot solve residual too large: {residual:g}")
    return tuple(mp.mpf(str(value)) for value in omega_dot) + gamma_coefficients(gamma_dot_form)


def algebraic_residual(state: Iterable[mp.mpf | int | float], lam: mp.mpf | int | float) -> dict[str, mp.mpf]:
    """Return SU(3) and nearly-parallel algebraic residuals."""
    omega, gamma = state_omega_gamma(state)
    lambda_value = mp.mpf(lam)
    dgamma_minus = subtract(exterior_derivative(gamma), scale(lambda_value / 2, wedge(omega, omega)))
    compatibility = wedge(omega, gamma)
    hitchin_value = hitchin_lambda(gamma)
    volume_residual = mp.inf
    if hitchin_value < 0:
        volume_residual = abs(abs(omega_volume(omega)) - mp.sqrt(-hitchin_value) / 2)
    return {
        "d_gamma_minus_lambda_omega2_over_2": max_abs_coefficient(dgamma_minus),
        "omega_wedge_gamma": max_abs_coefficient(compatibility),
        "volume_normalization": volume_residual,
    }


def ode_residual(state: Iterable[mp.mpf | int | float], state_dot: Iterable[mp.mpf | int | float], lam: mp.mpf | int | float) -> dict[str, mp.mpf]:
    """Return residuals for the two evolution equations."""
    values = tuple(mp.mpf(value) for value in state)
    dots = tuple(mp.mpf(value) for value in state_dot)
    omega, gamma = state_omega_gamma(values)
    omega_dot = omega_form(dots[:5])
    gamma_dot = gamma_form(dots[5:])
    hat = oriented_hitchin_dual(omega, gamma)
    lambda_value = mp.mpf(lam)
    return {
        "gamma_dot_minus_domega_plus_lambda_hat": max_abs_coefficient(
            subtract(gamma_dot, subtract(exterior_derivative(omega), scale(lambda_value, hat)))
        ),
        "omega_wedge_omega_dot_plus_dhat": max_abs_coefficient(add(wedge(omega, omega_dot), exterior_derivative(hat))),
    }


def podesta_state_from_f(f0: mp.mpf, f1: mp.mpf, f2: mp.mpf, f3: mp.mpf, f4: mp.mpf) -> tuple[mp.mpf, ...]:
    """Embed Podesta's five functions into the 13-variable U(1) chart."""
    return (
        mp.zero,
        mp.zero,
        f0,
        f0,
        mp.zero,
        f1,
        f4,
        mp.zero,
        f3,
        f3,
        f2,
        mp.zero,
        f4,
    )


def embedded_target(target: su2_cubed_action_audit.PodestaTarget) -> EmbeddedTarget:
    """Return a Podesta target embedded in this larger chart."""

    def value_at(index: int, t: mp.mpf) -> mp.mpf:
        f_values = tuple(function(t) for function in target.functions)
        return podesta_state_from_f(*f_values)[index]

    def derivative_at(index: int, t: mp.mpf) -> mp.mpf:
        return mp.diff(lambda variable: value_at(index, variable), t)

    state_functions = tuple(lambda t, index=index: value_at(index, t) for index in range(13))
    derivative_functions = tuple(lambda t, index=index: derivative_at(index, t) for index in range(13))
    return EmbeddedTarget(target.name, target.lam, state_functions, derivative_functions)


def target_residuals(target: EmbeddedTarget, sample_times: Iterable[mp.mpf]) -> list[dict[str, str]]:
    """Evaluate full-system residuals for an embedded known target."""
    rows = []
    for t in sample_times:
        state = tuple(function(t) for function in target.state_functions)
        state_dot = tuple(function(t) for function in target.state_derivative_functions)
        residuals = ode_residual(state, state_dot, target.lam)
        residuals.update(algebraic_residual(state, target.lam))
        rows.append(
            {
                "t": _mp_string(t),
                "max_abs_residual": _mp_string(max(abs(value) for value in residuals.values())),
                "residuals": {key: _mp_string(value) for key, value in residuals.items()},
            }
        )
    return rows


def endpoint_weight_table() -> dict[str, dict[str, str]]:
    """Return leading endpoint weights in the calibrated Podesta-compatible chart.

    These are the raw regular-variable weights before solving the nearly
    parallel ODE recurrence.  The left endpoint collapses the ``a`` directions;
    the right endpoint collapses the ``b`` directions.
    """
    return {
        "left_K_plus": {
            "collapsing": "a1,a2,a3",
            "surviving": "b1,b2,b3",
            "regular_variables": (
                "x1=t^4 X1, x2=t X2, x3=t X3, x4=t X4, x5=t^3 X5; "
                "y1=t^4 Y1, y2=t^2 Y2, y3=t^3 Y3, y4=t^2 Y4, "
                "y5=t^2 Y5, y6=Y6, y7=t^2 Y7, y8=t^2 Y8"
            ),
            "podesta_subchart": "x3=x4=f0, y1=f1, y2=y8=f4, y4=y5=f3, y6=f2, other variables zero",
            "podesta_leading_relation": "Y4(0)=Y5(0)=3 X3(0)=3 X4(0)",
        },
        "right_K_minus": {
            "collapsing": "b1,b2,b3",
            "surviving": "a1,a2,a3",
            "regular_variables": (
                "x1=t X1, x2=t^4 X2, x3=t X3, x4=t X4, x5=t^3 X5; "
                "y1=Y1, y2=t^2 Y2, y3=t^2 Y3, y4=t^2 Y4, "
                "y5=t^2 Y5, y6=t^4 Y6, y7=t^3 Y7, y8=t^2 Y8"
            ),
            "podesta_subchart": "same conditions after swapping a<->b: y1 is the surviving volume",
            "podesta_leading_relation": "Y2(0)=Y8(0)=-3 X3(0)=-3 X4(0) in the inward right coordinate convention",
        },
    }


def endpoint_jet_dimensions(max_order: int = 6) -> list[dict[str, int | str]]:
    """Return the first formal smooth-jet dimensions for the endpoint model.

    The numbers come from the Eschenburg-Wang homogeneous-polynomial test for
    the normal quaternionic slice and the surviving adjoint ``S3`` directions.
    They are used as a consistency guide; the ODE recurrence must still be
    imposed after these linear smoothness restrictions.
    """
    rows = []
    dimensions = {
        0: 1,
        1: 1,
        2: 5,
        3: 9,
        4: 17,
        5: 25,
        6: 41,
    }
    previous = 0
    for side in ("left_K_plus", "right_K_minus"):
        previous = 0
        for order in range(max_order + 1):
            dimension = dimensions.get(order)
            if dimension is None:
                break
            rows.append(
                {
                    "endpoint": side,
                    "max_order": order,
                    "allowed_dimension": dimension,
                    "new_dimension_at_order": dimension - previous,
                }
            )
            previous = dimension
    return rows


def invariant_basis_summary() -> dict[str, object]:
    """Return the explicit invariant basis and exterior derivative summary."""
    omega_labels = ("a12", "b12", "a3b3", "delta=a1b1+a2b2", "epsilon=a1b2-a2b1")
    return {
        "coframe": PRINCIPAL_LABELS,
        "maurer_cartan": {
            "da1": "6 a2^a3",
            "da2": "6 a3^a1",
            "da3": "6 a1^a2",
            "db1": "6 b2^b3",
            "db2": "6 b3^b1",
            "db3": "6 b1^b2",
        },
        "omega_basis": omega_labels,
        "gamma_basis": GAMMA_BASIS_LABELS,
        "podesta_embedding": {
            "omega": "f0*(a3b3 + delta)",
            "gamma": "f1*a123 + f2*b123 + f3*(b3a12 + a3_epsilon) + f4*(a3b12 + b3_epsilon)",
        },
        "component_equations": {
            "constraint": "d_gamma = (lambda/2) omega^2, omega^gamma=0, omega^3/6 = sqrt(-lambda(gamma))/2",
            "gamma_evolution": "dot(gamma) = d_6 omega - lambda * hat(gamma)",
            "omega_evolution": "omega ^ dot(omega) = -d_6 hat(gamma)",
        },
        "polynomial_constraints": polynomial_constraint_summary(),
    }


def polynomial_constraint_summary() -> dict[str, object]:
    """Return the component polynomial constraints before the Hitchin dual."""
    return {
        "domega_components": {
            "a3b12": "-6*x3",
            "a3_delta": "-6*x5",
            "a3_epsilon": "6*x4",
            "b3a12": "6*x3",
            "b3_delta": "6*x5",
            "b3_epsilon": "-6*x4",
        },
        "d_gamma_minus_lambda_omega2_over_2": [
            "-lambda*x1*x3 = 0",
            "6*(y2+y5) - lambda*x1*x2 + lambda*(x4^2+x5^2) = 0",
            "6*(y4+y8) + lambda*x3*x4 = 0",
            "6*(y3+y7) - lambda*x3*x5 = 0",
            "-lambda*x2*x3 = 0",
        ],
        "omega_wedge_gamma": [
            "x1*y2 + x2*y1 - 2*x4*y3 - 2*x5*y4 = 0",
            "x1*y6 + x2*y5 - 2*x4*y7 - 2*x5*y8 = 0",
        ],
        "regular_branch_note": (
            "For lambda != 0 and x3 != 0, the algebraic nearly-parallel "
            "constraint forces x1=x2=0 before the differential evolution is used."
        ),
        "regular_branch_algebraic_dimension": (
            "After x1=x2=0, the d_gamma equation gives three further sum "
            "relations among y2,y5; y4,y8; y3,y7. The two compatibility "
            "equations then collapse to one independent equation, and the "
            "volume normalization gives one more scalar condition. Thus the "
            "regular branch has six algebraic degrees before endpoint Taylor "
            "recurrences are imposed."
        ),
    }


def build_summary() -> dict[str, object]:
    """Return a JSON-serializable derivation/calibration summary."""
    with mp.workdps(DEFAULT_DPS):
        sample_times = (mp.mpf("0.37"), mp.mpf("0.91"))
        targets = (
            embedded_target(su2_cubed_action_audit.round_target()),
            embedded_target(su2_cubed_action_audit.squashed_target()),
        )
        return {
            "version": SYSTEM_VERSION,
            "action": "Sp(1) x Sp(1) x U(1) on S7",
            "principal_orbit": "S3 x S3",
            "singular_orbits": ["S3", "S3"],
            "raw_function_count": {"omega": 5, "gamma": 8, "total": 13},
            "invariant_basis": invariant_basis_summary(),
            "endpoint_conditions": {
                "weights": endpoint_weight_table(),
                "jet_dimensions": endpoint_jet_dimensions(),
                "status": (
                    "linear smoothness weights and Podesta-compatible leading relations are recorded; "
                    "experiments.s7.sp1xsp1xu1_matching fits higher endpoint germs against the ODE"
                ),
            },
            "known_solution_checks": {
                target.name: target_residuals(target, sample_times)
                for target in targets
            },
        }


def _mp_string(value: mp.mpf) -> str:
    return mp.nstr(value, 80)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    summary = build_summary()
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    print(f"{summary['version']}: {summary['action']}")
    print(f"raw functions: {summary['raw_function_count']}")
    for name, rows in summary["known_solution_checks"].items():
        max_residual = max(mp.mpf(row["max_abs_residual"]) for row in rows)
        print(f"{name}: max residual {mp.nstr(max_residual, 12)}")
    print("endpoint weights:")
    for side, data in summary["endpoint_conditions"]["weights"].items():
        print(f"  {side}: {data['regular_variables']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
