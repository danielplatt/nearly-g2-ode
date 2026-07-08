"""Endpoint smoothness diagnostics for the N_{1,1} cohomogeneity-one ansatz.

The principal-orbit ansatz has 19 raw coefficients.  This module keeps that
full chart and derives the first singular-orbit restrictions by the
Eschenburg-Wang representation test: homogeneous normal-polynomial maps into
``Lambda^3(V+n)^*`` are evaluated on the chosen normal ray and compared with
the 19 principal coefficients.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from . import ansatz


ENDPOINT_SMOOTHNESS_VERSION = "aloff-wallach-n11-endpoint-smoothness-v1"
SMOOTH_COFRAME = ("normal_radial", "normal_angular", "base_1", "base_2", "fiber_1", "fiber_2", "surviving_axis")
PRINCIPAL_VARIABLES = tuple(f"x{index}" for index in range(1, 8)) + tuple(f"y{index}" for index in range(1, 13))
THREE_FORM_BASIS = tuple(combinations(range(7), 3))
THREE_FORM_INDEX = {key: index for index, key in enumerate(THREE_FORM_BASIS)}


@dataclass(frozen=True)
class EndpointModel:
    """One singular endpoint representation model."""

    label: str
    singular_base_orbit: str
    normal_weight: int
    collapse_combination: str
    surviving_combination: str
    note: str


def endpoint_models() -> tuple[EndpointModel, ...]:
    """Return the two N_{1,1} singular endpoint models."""
    return (
        EndpointModel(
            label="real_rp2",
            singular_base_orbit="RP^2 real-line orbit in CP^2",
            normal_weight=1,
            collapse_combination="theta = base_3 + fiber_3",
            surviving_combination="zeta = base_3 - fiber_3",
            note=(
                "The connected singular isotropy rotates the normal disk with "
                "weight 1; the disconnected stabilizer accounts for the "
                "principal half-turn already built into the 19-variable chart."
            ),
        ),
        EndpointModel(
            label="null_conic_cp1",
            singular_base_orbit="CP^1 null-conic orbit in CP^2",
            normal_weight=2,
            collapse_combination="theta = base_3 + fiber_3",
            surviving_combination="zeta = base_3 - fiber_3",
            note="The connected singular isotropy rotates the normal disk with weight 2.",
        ),
    )


def _merge_sign(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[tuple[int, ...], int] | None:
    if set(left).intersection(right):
        return None
    combined = left + right
    inversions = sum(1 for i, left_index in enumerate(combined) for right_index in combined[i + 1 :] if left_index > right_index)
    return tuple(sorted(combined)), -1 if inversions % 2 else 1


def _wedge_terms(
    left: Iterable[tuple[int, tuple[int, ...], float]],
    right: Iterable[tuple[int, tuple[int, ...], float]],
) -> list[tuple[int, tuple[int, ...], float]]:
    result: dict[tuple[int, tuple[int, ...]], float] = {}
    for left_shift, left_key, left_coefficient in left:
        for right_shift, right_key, right_coefficient in right:
            merged = _merge_sign(left_key, right_key)
            if merged is None:
                continue
            key, sign = merged
            result[(left_shift + right_shift, key)] = result.get((left_shift + right_shift, key), 0.0) + (
                sign * left_coefficient * right_coefficient
            )
    return [
        (shift, key, coefficient)
        for (shift, key), coefficient in sorted(result.items())
        if abs(coefficient) > 1e-12
    ]


def _principal_one_form(label: str, collapse_sign: int = 1) -> list[tuple[int, tuple[int, ...], float]]:
    """Return one principal coform in the smooth endpoint coframe.

    ``shift`` records how many collapsing angular coforms occur.  Since
    ``normal_angular = r * theta`` on the normal ray, a principal coefficient
    multiplying one ``theta`` contributes one order lower to the smooth
    coefficient.
    """
    if label == "B1":
        return [(0, (2,), 1.0)]
    if label == "B2":
        return [(0, (3,), 1.0)]
    if label == "F1":
        return [(0, (4,), 1.0)]
    if label == "F2":
        return [(0, (5,), 1.0)]
    if label == "B3":
        return [(1, (1,), 0.5), (0, (6,), 0.5)]
    if label == "F3":
        sign = float(collapse_sign)
        return [(1, (1,), 0.5 * sign), (0, (6,), -0.5 * sign)]
    raise ValueError(f"unknown principal coform label {label!r}")


def principal_variable_smooth_terms(collapse_sign: int = 1) -> dict[str, list[dict[str, object]]]:
    """Return the 19 principal variables rewritten in the endpoint coframe."""
    omega = (
        ("x1", ("B1", "B2")),
        ("x2", ("B1", "F1")),
        ("x3", ("B1", "F2")),
        ("x4", ("B2", "F1")),
        ("x5", ("B2", "F2")),
        ("x6", ("B3", "F3")),
        ("x7", ("F1", "F2")),
    )
    gamma = (
        ("y1", ("B1", "B2", "B3")),
        ("y2", ("B1", "B2", "F3")),
        ("y3", ("B1", "B3", "F1")),
        ("y4", ("B1", "B3", "F2")),
        ("y5", ("B1", "F1", "F3")),
        ("y6", ("B1", "F2", "F3")),
        ("y7", ("B2", "B3", "F1")),
        ("y8", ("B2", "B3", "F2")),
        ("y9", ("B2", "F1", "F3")),
        ("y10", ("B2", "F2", "F3")),
        ("y11", ("B3", "F1", "F2")),
        ("y12", ("F1", "F2", "F3")),
    )
    raw_terms: dict[str, list[tuple[int, tuple[int, ...], float]]] = {}
    radial = [(0, (0,), 1.0)]
    for name, labels in omega:
        terms = _principal_one_form(labels[0], collapse_sign)
        for label in labels[1:]:
            terms = _wedge_terms(terms, _principal_one_form(label, collapse_sign))
        raw_terms[name] = _wedge_terms(radial, terms)
    for name, labels in gamma:
        terms = _principal_one_form(labels[0], collapse_sign)
        for label in labels[1:]:
            terms = _wedge_terms(terms, _principal_one_form(label, collapse_sign))
        raw_terms[name] = terms
    return {
        name: [
            {
                "coefficient": coefficient,
                "basis": "^".join(SMOOTH_COFRAME[index] for index in key),
                "collapse_shift": shift,
            }
            for shift, key, coefficient in terms
        ]
        for name, terms in raw_terms.items()
    }


def _principal_terms_for_linear_algebra(collapse_sign: int = 1) -> tuple[tuple[str, tuple[tuple[int, tuple[int, ...], float], ...]], ...]:
    """Return compact terms for the jet-dimension linear algebra."""
    payload = principal_variable_smooth_terms(collapse_sign)
    result = []
    for name in PRINCIPAL_VARIABLES:
        terms = []
        for item in payload[name]:
            key = tuple(SMOOTH_COFRAME.index(label) for label in str(item["basis"]).split("^"))
            terms.append((int(item["collapse_shift"]), key, float(item["coefficient"])))
        result.append((name, tuple(terms)))
    return tuple(result)


def _rotation_generator(normal_weight: int) -> np.ndarray:
    """Return the infinitesimal generator on the endpoint smooth coframe."""
    matrix = np.zeros((7, 7))

    def block(first: int, second: int, weight: int) -> None:
        matrix[second, first] = float(weight)
        matrix[first, second] = -float(weight)

    block(0, 1, normal_weight)
    block(2, 3, 1)
    block(4, 5, 1)
    return matrix


def _exterior_generator(one_form_generator: np.ndarray) -> np.ndarray:
    """Return the induced generator on 3-forms."""
    matrix = np.zeros((len(THREE_FORM_BASIS), len(THREE_FORM_BASIS)))
    for column, key in enumerate(THREE_FORM_BASIS):
        for position, old_index in enumerate(key):
            for new_index, coefficient in enumerate(one_form_generator[:, old_index]):
                if abs(coefficient) < 1e-12 or new_index in key:
                    continue
                new_key = list(key)
                new_key[position] = new_index
                inversions = sum(
                    1
                    for i, left_index in enumerate(new_key)
                    for right_index in new_key[i + 1 :]
                    if left_index > right_index
                )
                sorted_key = tuple(sorted(new_key))
                matrix[THREE_FORM_INDEX[sorted_key], column] += coefficient * (-1 if inversions % 2 else 1)
    return matrix


def _domain_polynomial_generator(normal_weight: int, degree: int) -> np.ndarray:
    """Return the generator on homogeneous normal polynomials of one degree."""
    matrix = np.zeros((degree + 1, degree + 1))
    for y_power in range(degree + 1):
        x_power = degree - y_power
        if x_power:
            matrix[y_power + 1, y_power] += -x_power * normal_weight
        if y_power:
            matrix[y_power - 1, y_power] += y_power * normal_weight
    return matrix


def _evaluation_subspace_orthogonal(normal_weight: int, degree: int, tolerance: float = 1e-10) -> np.ndarray:
    """Return rows spanning the orthogonal complement of allowed degree jets."""
    if degree < 0:
        return np.eye(len(THREE_FORM_BASIS))
    exterior = _exterior_generator(_rotation_generator(normal_weight))
    domain = _domain_polynomial_generator(normal_weight, degree)
    equivariance = np.kron(np.eye(degree + 1), exterior) - np.kron(domain, np.eye(len(THREE_FORM_BASIS)))
    _, singular_values, vh = np.linalg.svd(equivariance)
    rank = int(np.sum(singular_values > tolerance))
    nullspace = vh[rank:].T
    evaluation = nullspace[: len(THREE_FORM_BASIS), :]
    if evaluation.size == 0:
        return np.eye(len(THREE_FORM_BASIS))
    left, eval_singular_values, _ = np.linalg.svd(evaluation, full_matrices=True)
    eval_rank = int(np.sum(eval_singular_values > tolerance))
    return left[:, eval_rank:].T


def allowed_principal_jet_dimension(
    normal_weight: int,
    max_order: int,
    *,
    collapse_sign: int = 1,
    tolerance: float = 1e-10,
) -> int:
    """Return the dimension of smooth principal jets through ``max_order``."""
    terms_by_variable = _principal_terms_for_linear_algebra(collapse_sign)
    unknown_count = len(terms_by_variable) * (max_order + 1)
    rows = []
    for smooth_degree in range(-1, max_order + 1):
        degree_matrix = np.zeros((len(THREE_FORM_BASIS), unknown_count))
        for variable_index, (_name, terms) in enumerate(terms_by_variable):
            for collapse_shift, key, coefficient in terms:
                principal_order = smooth_degree + collapse_shift
                if 0 <= principal_order <= max_order:
                    column = variable_index * (max_order + 1) + principal_order
                    degree_matrix[THREE_FORM_INDEX[key], column] += coefficient
        rows.append(_evaluation_subspace_orthogonal(normal_weight, smooth_degree, tolerance) @ degree_matrix)
    constraint_matrix = np.vstack(rows)
    _, singular_values, _ = np.linalg.svd(constraint_matrix)
    rank = int(np.sum(singular_values > tolerance))
    return unknown_count - rank


def jet_dimension_table(max_order: int = 5) -> list[dict[str, int | str]]:
    """Return allowed 19-variable smooth jet dimensions for both endpoints."""
    rows = []
    for model in endpoint_models():
        previous = 0
        for order in range(max_order + 1):
            dimension = allowed_principal_jet_dimension(model.normal_weight, order)
            rows.append(
                {
                    "endpoint": model.label,
                    "normal_weight": model.normal_weight,
                    "max_order": order,
                    "allowed_dimension": dimension,
                    "new_dimension_at_order": dimension - previous,
                }
            )
            previous = dimension
    return rows


def zero_order_conditions() -> list[str]:
    """Return readable standard-graph zero-order smoothness conditions."""
    return [
        "x1(0)=x2(0)=x3(0)=x4(0)=x5(0)=x6(0)=x7(0)=0",
        "y2(0)=-y1(0)",
        "y5(0)=y3(0)",
        "y6(0)=y4(0)",
        "y7(0)=-y4(0)",
        "y8(0)=y3(0)",
        "y9(0)=-y4(0)",
        "y10(0)=y3(0)",
        "y12(0)=-y11(0)",
    ]


def zero_order_parameterization() -> dict[str, object]:
    """Return the four-parameter standard-graph endpoint value chart."""
    return {
        "free_constants": ["A = y1(0)", "B = y3(0)", "C = y4(0)", "D = y11(0)"],
        "x_values": {f"x{index}(0)": "0" for index in range(1, 8)},
        "y_values": {
            "y1(0)": "A",
            "y2(0)": "-A",
            "y3(0)": "B",
            "y4(0)": "C",
            "y5(0)": "B",
            "y6(0)": "C",
            "y7(0)": "-C",
            "y8(0)": "B",
            "y9(0)": "-C",
            "y10(0)": "B",
            "y11(0)": "D",
            "y12(0)": "-D",
        },
        "dimension": 4,
    }


def build_summary(max_order: int = 5) -> dict:
    """Return a JSON-ready endpoint smoothness summary."""
    variables = ansatz.principal_orbit_su3_variable_basis()
    return {
        "version": ENDPOINT_SMOOTHNESS_VERSION,
        "topology": "N_{1,1}=SU(3)/U(1)_{1,1}",
        "method": (
            "full 19-variable principal-orbit chart, rewritten in endpoint "
            "coframes and tested against K-equivariant homogeneous normal "
            "polynomial maps"
        ),
        "principal_variable_count": variables["raw_coefficient_count"],
        "algebraic_su3_constraint_count": len(variables["algebraic_constraints"]),
        "endpoint_models": [
            {
                "label": model.label,
                "singular_base_orbit": model.singular_base_orbit,
                "normal_weight": model.normal_weight,
                "collapse_combination": model.collapse_combination,
                "surviving_combination": model.surviving_combination,
                "note": model.note,
            }
            for model in endpoint_models()
        ],
        "zero_order_conditions": zero_order_conditions(),
        "zero_order_parameterization": zero_order_parameterization(),
        "jet_dimension_table": jet_dimension_table(max_order),
        "parameter_count_note": (
            "Smoothness alone reduces the 19 raw coefficient values at a "
            "singular endpoint to four zeroth-order constants.  Weighted higher "
            "layers remain: through first order the RP2 endpoint has 9 smooth "
            "jet parameters, while the CP1 endpoint has 13.  The nearly-G2 "
            "evolution equations and the SU(3) algebraic constraints must be "
            "imposed next before choosing scout coordinates."
        ),
        "verdict": (
            "endpoint smoothness gives a canonical weighted chart direction "
            "without reducing to the homogeneous A,B,C,D family"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Print the N_{1,1} endpoint smoothness summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--max-order", type=int, default=5, help="maximum Taylor order for the dimension table")
    args = parser.parse_args(argv)
    summary = build_summary(args.max_order)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return
    print("Aloff-Wallach N_{1,1} endpoint smoothness", flush=True)
    print(f"version: {summary['version']}", flush=True)
    print(f"principal variables: {summary['principal_variable_count']}", flush=True)
    print("zero-order endpoint values:", flush=True)
    for condition in summary["zero_order_conditions"]:
        print(f"  {condition}", flush=True)
    print("jet dimensions:", flush=True)
    for row in summary["jet_dimension_table"]:
        if row["max_order"] <= 2:
            print(
                f"  {row['endpoint']} through order {row['max_order']}: "
                f"{row['allowed_dimension']} "
                f"(new {row['new_dimension_at_order']})",
                flush=True,
            )
    print(f"verdict: {summary['verdict']}", flush=True)


if __name__ == "__main__":
    main()
