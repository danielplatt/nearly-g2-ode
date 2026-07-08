"""Aloff-Wallach N_{1,1} ansatz and known-solution verification.

This module records the first usable ansatz for the exceptional Aloff-Wallach
space N_{1,1}.  The cohomogeneity-one action identified by the feasibility audit
has principal orbits locally SO(3) x SO(3), while the known calibration
structures can be checked inside the SU(3)-homogeneous A,B,C,D family used by
Ball-Oliveira.
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


N11_ANSATZ_VERSION = "aloff-wallach-n11-ansatz-v4"
Index: TypeAlias = tuple[int, ...]
Form: TypeAlias = dict[Index, mp.mpf]

PRINCIPAL_ORBIT_LABELS = ("base_1", "base_2", "base_3", "fiber_1", "fiber_2", "fiber_3")
PRINCIPAL_ORBIT_NORMAL_7D_INDEX = 7
PRINCIPAL_ORBIT_7D_TO_6D = {
    2: 1,
    6: 2,
    3: 3,
    1: 4,
    5: 5,
    4: 6,
}


@dataclass(frozen=True)
class N11KnownSolution:
    """One known nearly-parallel point in the N_{1,1} A,B,C,D ansatz."""

    label: str
    A: mp.mpf
    B: mp.mpf
    C: mp.mpf
    D: mp.mpf
    lambda_value: mp.mpf
    metric_type: str


@dataclass(frozen=True)
class N11ExtraSasakiEinsteinSolution:
    """The extra N_{1,1} nearly-parallel structure outside A,B,C,D."""

    label: str
    source_formula: str
    lambda_statement: str
    metric_type: str
    in_abcd_family: bool
    current_action_invariant: bool
    needs_alternative_cohomogeneity_one_action: bool
    invariant_terms: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class SU3AlgebraicCheck:
    """Algebraic SU(3)-structure diagnostics for one invariant pair."""

    label: str
    omega_coefficients: dict[str, str]
    gamma_coefficients: dict[str, str]
    omega_wedge_gamma_residual: mp.mpf
    hitchin_lambda: mp.mpf
    hitchin_complex_residual: mp.mpf
    volume_normalization_residual: mp.mpf
    orientation_sign: int
    stable_negative: bool


def _clean(value: mp.mpf, tolerance: mp.mpf = mp.mpf("1e-70")) -> mp.mpf:
    """Remove harmless numerical noise from one scalar."""
    return mp.zero if abs(value) < tolerance else value


def _add_to(form: Form, key: Index, value: mp.mpf) -> None:
    """Accumulate one form coefficient, dropping exact numerical zeroes."""
    new_value = _clean(form.get(key, mp.zero) + value)
    if new_value == 0:
        form.pop(key, None)
    else:
        form[key] = new_value


def form(*terms: tuple[Iterable[int], mp.mpf | int | float]) -> Form:
    """Build a form from ``((indices), coefficient)`` terms."""
    result: Form = {}
    for indices, coefficient in terms:
        key = tuple(indices)
        if tuple(sorted(key)) != key or len(set(key)) != len(key):
            raise ValueError(f"indices must be strictly increasing: {key}")
        _add_to(result, key, mp.mpf(coefficient))
    return result


def basis(indices: Iterable[int], coefficient: mp.mpf | int | float = 1) -> Form:
    """Return one basis wedge form."""
    return form((tuple(indices), coefficient))


def scale(value: mp.mpf | int | float, item: Form) -> Form:
    """Scale one form."""
    scalar = mp.mpf(value)
    return {key: _clean(scalar * coefficient) for key, coefficient in item.items() if scalar * coefficient != 0}


def add(*items: Form) -> Form:
    """Add forms."""
    result: Form = {}
    for item in items:
        for key, coefficient in item.items():
            _add_to(result, key, coefficient)
    return result


def subtract(left: Form, right: Form) -> Form:
    """Subtract two forms."""
    return add(left, scale(-1, right))


def _merge_sign(left: Index, right: Index) -> tuple[Index, int] | None:
    """Return the sorted wedge key and sign, or None for repeated indices."""
    if set(left).intersection(right):
        return None
    combined = left + right
    inversions = sum(1 for i, left_index in enumerate(combined) for right_index in combined[i + 1 :] if left_index > right_index)
    return tuple(sorted(combined)), -1 if inversions % 2 else 1


def wedge(left: Form, right: Form) -> Form:
    """Wedge two forms."""
    result: Form = {}
    for left_key, left_coefficient in left.items():
        for right_key, right_coefficient in right.items():
            merged = _merge_sign(left_key, right_key)
            if merged is None:
                continue
            key, sign = merged
            _add_to(result, key, left_coefficient * right_coefficient * sign)
    return result


def max_abs_coefficient(item: Form) -> mp.mpf:
    """Return the max absolute coefficient of a form."""
    return max((abs(value) for value in item.values()), default=mp.zero)


def _basis_key_label(key: Index, labels: tuple[str, ...] = PRINCIPAL_ORBIT_LABELS) -> str:
    """Return a readable basis label for one form key."""
    return "^".join(labels[index - 1] for index in key)


def _form_coefficients_by_basis(item: Form, basis_keys: Iterable[Index]) -> dict[str, str]:
    """Return stable string coefficients against a named basis."""
    return {
        _basis_key_label(key): _mp_string(item.get(key, mp.zero))
        for key in basis_keys
        if item.get(key, mp.zero) != 0
    }


def _remap_key(key: Index, index_map: dict[int, int]) -> tuple[Index, int]:
    """Remap a sorted key through an index map and return the sorting sign."""
    mapped = tuple(index_map[index] for index in key)
    inversions = sum(1 for i, left_index in enumerate(mapped) for right_index in mapped[i + 1 :] if left_index > right_index)
    return tuple(sorted(mapped)), -1 if inversions % 2 else 1


def contract_vector(index: int, item: Form) -> Form:
    """Contract a form with the basis vector dual to one basis 1-form."""
    result: Form = {}
    for key, coefficient in item.items():
        if index not in key:
            continue
        position = key.index(index)
        _add_to(result, key[:position] + key[position + 1 :], (mp.mpf(-1) ** position) * coefficient)
    return result


def restrict_to_principal_orbit(item: Form) -> Form:
    """Restrict a 7D homogeneous form to the model 6D principal orbit."""
    result: Form = {}
    for key, coefficient in item.items():
        if PRINCIPAL_ORBIT_NORMAL_7D_INDEX in key:
            continue
        remapped_key, sign = _remap_key(key, PRINCIPAL_ORBIT_7D_TO_6D)
        _add_to(result, remapped_key, sign * coefficient)
    return result


def contract_principal_normal(item: Form) -> Form:
    """Contract a 7D homogeneous form with the model principal-orbit normal."""
    contracted = contract_vector(PRINCIPAL_ORBIT_NORMAL_7D_INDEX, item)
    result: Form = {}
    for key, coefficient in contracted.items():
        remapped_key, sign = _remap_key(key, PRINCIPAL_ORBIT_7D_TO_6D)
        _add_to(result, remapped_key, sign * coefficient)
    return result


def volume_coefficient(item: Form, dimension: int = 6) -> mp.mpf:
    """Return the coefficient of the ordered volume form."""
    return item.get(tuple(range(1, dimension + 1)), mp.zero)


def _matrix_multiply(left: tuple[tuple[mp.mpf, ...], ...], right: tuple[tuple[mp.mpf, ...], ...]) -> tuple[tuple[mp.mpf, ...], ...]:
    """Multiply two small square matrices."""
    size = len(left)
    return tuple(
        tuple(sum(left[row][inner] * right[inner][column] for inner in range(size)) for column in range(size))
        for row in range(size)
    )


def hitchin_k_matrix(gamma: Form, dimension: int = 6) -> tuple[tuple[mp.mpf, ...], ...]:
    """Return Hitchin's K_gamma matrix in the ordered volume trivialization.

    For the standard real part of a complex volume form this returns ``2J``;
    hence ``tr(K^2)/6`` is negative.
    """
    rows = [[mp.zero for _ in range(dimension)] for _ in range(dimension)]
    full_key = tuple(range(1, dimension + 1))
    for column in range(1, dimension + 1):
        alpha = wedge(contract_vector(column, gamma), gamma)
        for key, coefficient in alpha.items():
            missing = [index for index in full_key if index not in key]
            if len(missing) != 1:
                continue
            row = missing[0]
            expected_key = tuple(index for index in full_key if index != row)
            if key != expected_key:
                raise ValueError(f"unexpected 5-form key order: {key}")
            rows[row - 1][column - 1] = coefficient * (mp.mpf(-1) ** (row - 1))
    return tuple(tuple(row) for row in rows)


def hitchin_lambda(gamma: Form) -> mp.mpf:
    """Return the scalar lambda with K_gamma^2=lambda I, when stable."""
    k_matrix = hitchin_k_matrix(gamma)
    k_squared = _matrix_multiply(k_matrix, k_matrix)
    return sum(k_squared[index][index] for index in range(6)) / 6


def su3_algebraic_check(label: str, omega: Form, gamma: Form) -> SU3AlgebraicCheck:
    """Return algebraic diagnostics for a 6D invariant SU(3) pair."""
    compatibility_residual = max_abs_coefficient(wedge(omega, gamma))
    k_matrix = hitchin_k_matrix(gamma)
    k_squared = _matrix_multiply(k_matrix, k_matrix)
    lambda_value = sum(k_squared[index][index] for index in range(6)) / 6
    complex_residual = max(
        abs(k_squared[row][column] - (lambda_value if row == column else mp.zero))
        for row in range(6)
        for column in range(6)
    )
    omega_volume = volume_coefficient(wedge(wedge(omega, omega), omega)) / 6
    if lambda_value < 0:
        target_volume = mp.sqrt(-lambda_value) / 2
        volume_residual = abs(abs(omega_volume) - target_volume)
    else:
        volume_residual = mp.inf
    orientation_sign = 0
    if omega_volume > 0:
        orientation_sign = 1
    elif omega_volume < 0:
        orientation_sign = -1
    return SU3AlgebraicCheck(
        label=label,
        omega_coefficients=_form_coefficients_by_basis(omega, principal_orbit_form_basis_indices(2)),
        gamma_coefficients=_form_coefficients_by_basis(gamma, principal_orbit_form_basis_indices(3)),
        omega_wedge_gamma_residual=compatibility_residual,
        hitchin_lambda=lambda_value,
        hitchin_complex_residual=complex_residual,
        volume_normalization_residual=volume_residual,
        orientation_sign=orientation_sign,
        stable_negative=lambda_value < 0,
    )


def _matrix(entries: list[list[complex]]) -> np.ndarray:
    """Return a complex matrix."""
    return np.array(entries, dtype=np.complex128)


def n11_su3_basis_matrices() -> tuple[np.ndarray, ...]:
    """Return the orthonormal e_1,...,e_7,H basis for N_{1,1}=SU(3)/U(1)."""
    sqrt2 = np.sqrt(2.0)
    sqrt6 = np.sqrt(6.0)
    e1 = (1 / sqrt2) * _matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    e5 = (1j / sqrt2) * _matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    e2 = (1 / sqrt2) * _matrix([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
    e6 = (1j / sqrt2) * _matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    e3 = (1 / sqrt2) * _matrix([[0, 0, -1], [0, 0, 0], [1, 0, 0]])
    e7 = (1j / sqrt2) * _matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    e4 = (1j / sqrt2) * _matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    h = (1j / sqrt6) * _matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]])
    return e1, e2, e3, e4, e5, e6, e7, h


def _inner(left: np.ndarray, right: np.ndarray) -> float:
    """Return the real inner product -tr(left right)."""
    return float(np.real(-np.trace(left @ right)))


def n11_bracket_constants() -> dict[tuple[int, int, int], mp.mpf]:
    """Return reductive bracket constants for the N_{1,1} basis.

    Constants are indexed by ``(i,j,k)`` with ``i<j`` and
    ``[e_i,e_j]_m = sum_k c_{ij}^k e_k``.
    """
    matrices = n11_su3_basis_matrices()[:7]
    constants: dict[tuple[int, int, int], mp.mpf] = {}
    for i, j in combinations(range(7), 2):
        bracket = matrices[i] @ matrices[j] - matrices[j] @ matrices[i]
        for k, basis_matrix in enumerate(matrices):
            coefficient = _inner(bracket, basis_matrix)
            if abs(coefficient) > 1e-13:
                constants[(i + 1, j + 1, k + 1)] = mp.mpf(str(coefficient))
    return constants


def n11_one_form_differentials() -> dict[int, Form]:
    """Return d omega_i for the N_{1,1} homogeneous coframe.

    Ball-Oliveira's quotient coframe uses the fundamental-vector convention
    opposite to the raw left-invariant matrix bracket, so d omega^k has the
    same sign as the reductive bracket constants computed above.
    """
    differentials: dict[int, Form] = {index: {} for index in range(1, 8)}
    for (i, j, k), coefficient in n11_bracket_constants().items():
        _add_to(differentials[k], (i, j), coefficient)
    return differentials


def exterior_derivative(item: Form, one_form_differentials: dict[int, Form] | None = None) -> Form:
    """Compute the exterior derivative of a left-invariant form."""
    differentials = one_form_differentials or n11_one_form_differentials()
    result: Form = {}
    for key, coefficient in item.items():
        for position, index in enumerate(key):
            prefix = basis(key[:position])
            suffix = basis(key[position + 1 :])
            term = wedge(wedge(prefix, differentials[index]), suffix)
            result = add(result, scale((mp.mpf(-1) ** position) * coefficient, term))
    return result


def aloff_wallach_phi(A: mp.mpf, B: mp.mpf, C: mp.mpf, D: mp.mpf) -> Form:
    """Return Ball-Oliveira's homogeneous Aloff-Wallach G2 3-form."""
    return add(
        scale(A * B * C, basis((1, 2, 3))),
        scale(-A * B * C, basis((1, 6, 7))),
        scale(A * B * C, basis((2, 5, 7))),
        scale(-A * B * C, basis((3, 5, 6))),
        scale(-D, wedge(basis((4,)), add(scale(A**2, basis((1, 5))), scale(B**2, basis((2, 6))), scale(C**2, basis((3, 7)))))),
    )


def aloff_wallach_psi(A: mp.mpf, B: mp.mpf, C: mp.mpf, D: mp.mpf) -> Form:
    """Return the 4-form psi=*phi in Ball-Oliveira's conventions."""
    return add(
        scale(A * B * C * D, basis((4, 5, 6, 7))),
        scale(-A * B * C * D, basis((2, 3, 4, 5))),
        scale(A * B * C * D, basis((1, 3, 4, 6))),
        scale(-A * B * C * D, basis((1, 2, 4, 7))),
        scale(B**2 * C**2, basis((2, 3, 6, 7))),
        scale(A**2 * C**2, basis((1, 3, 5, 7))),
        scale(A**2 * B**2, basis((1, 2, 5, 6))),
    )


def ball_oliveira_dphi_formula(A: mp.mpf, B: mp.mpf, C: mp.mpf, D: mp.mpf, k: int = 1, l: int = 1) -> Form:
    """Return dphi from Ball-Oliveira's closed formula for comparison."""
    m = -k - l
    s = mp.sqrt(k * k + l * l + m * m) / mp.sqrt(6)
    sqrt2 = mp.sqrt(2)
    return scale(
        1 / sqrt2,
        add(
            scale(D * (A**2 + B**2 + C**2), add(basis((4, 5, 6, 7)), scale(-1, basis((2, 3, 4, 5))), basis((1, 3, 4, 6)), scale(-1, basis((1, 2, 4, 7))))),
            scale(4 * A * B * C * s - B**2 * D * l - C**2 * D * k, basis((2, 3, 6, 7))),
            scale(4 * A * B * C * s - C**2 * D * m - A**2 * D * l, basis((1, 3, 5, 7))),
            scale(4 * A * B * C * s - A**2 * D * k - B**2 * D * m, basis((1, 2, 5, 6))),
        ),
    )


def n11_known_solutions() -> tuple[N11KnownSolution, ...]:
    """Return the tri-Sasakian and squashed N_{1,1} calibration points."""
    return (
        N11KnownSolution(
            label="tri_sasakian",
            A=mp.sqrt(2),
            B=mp.one,
            C=mp.one,
            D=mp.sqrt(2),
            lambda_value=mp.mpf(2),
            metric_type="tri-Sasakian",
        ),
        N11KnownSolution(
            label="squashed",
            A=mp.sqrt(mp.mpf(2) / 5),
            B=mp.one,
            C=mp.one,
            D=-mp.sqrt(mp.mpf(2) / 5),
            lambda_value=mp.mpf(6) / mp.sqrt(5),
            metric_type="strict proper nearly parallel",
        ),
    )


def n11_extra_sasaki_einstein_solution() -> N11ExtraSasakiEinsteinSolution:
    """Return the extra Sasaki-Einstein nearly-parallel structure data.

    Ball-Oliveira write this structure in the 3-Sasakian SO(3)-bundle notation.
    The two displayed tensors are SO(3)-invariant contractions, so the form is
    invariant under the fiber SO(3) used by the cohomogeneity-one action.
    """
    return N11ExtraSasakiEinsteinSolution(
        label="sasaki_einstein_phi_ts",
        source_formula=(
            "phi_ts = -eta_123 + (s/48) * (eta_1 wedge omega_1 + eta_2 wedge omega_2 + eta_3 wedge omega_3)"
        ),
        lambda_statement="d phi_ts = 4 psi_ts in Ball-Oliveira's tri-Sasakian normalization",
        metric_type="Sasaki-Einstein, on the tri-Sasakian metric",
        in_abcd_family=False,
        current_action_invariant=True,
        needs_alternative_cohomogeneity_one_action=False,
        invariant_terms=(
            "eta_123 is the SO(3)-invariant vertical volume form",
            "sum_i eta_i wedge omega_i is the SO(3)-invariant contraction of the vertical and self-dual triples",
        ),
        explanation=(
            "The extra form is outside the A,B,C,D family because the family locks "
            "the vertical-volume coefficient to the mixed contraction coefficients. "
            "The extra Sasaki-Einstein form changes that relative sign independently. "
            "It is visible in the SO(3)_real x SO(3)_fiber principal-orbit "
            "SU(3)-structure variables after the Weyl-adjusted Geipel-to-Ball "
            "coframe conversion."
        ),
    )


def nearly_parallel_residual(solution: N11KnownSolution) -> Form:
    """Return dphi-lambda psi for one known solution."""
    phi = aloff_wallach_phi(solution.A, solution.B, solution.C, solution.D)
    psi = aloff_wallach_psi(solution.A, solution.B, solution.C, solution.D)
    return subtract(exterior_derivative(phi), scale(solution.lambda_value, psi))


def best_abcd_lambda_residual(A: mp.mpf, B: mp.mpf, C: mp.mpf, D: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    """Return the best scalar lambda and residual for one A,B,C,D form."""
    phi = aloff_wallach_phi(A, B, C, D)
    psi = aloff_wallach_psi(A, B, C, D)
    dphi = exterior_derivative(phi)
    keys = set(dphi).union(psi)
    numerator = sum(dphi.get(key, mp.zero) * psi.get(key, mp.zero) for key in keys)
    denominator = sum(psi.get(key, mp.zero) ** 2 for key in keys)
    if denominator == 0:
        raise ValueError("psi has zero norm")
    lambda_value = numerator / denominator
    residual = subtract(dphi, scale(lambda_value, psi))
    return lambda_value, max_abs_coefficient(residual)


def abcd_vertical_flip_probe() -> dict[str, str]:
    """Probe whether the extra sign pattern is hidden as a D-sign branch."""
    A = mp.sqrt(2)
    B = mp.one
    C = mp.one
    good_lambda, good_residual = best_abcd_lambda_residual(A, B, C, A)
    flipped_lambda, flipped_residual = best_abcd_lambda_residual(A, B, C, -A)
    return {
        "metric_square_pattern": "A^2=2B^2, C^2=B^2, D^2=A^2",
        "reference_D": _mp_string(A),
        "reference_best_lambda": _mp_string(good_lambda),
        "reference_best_residual": _mp_string(good_residual),
        "flipped_D": _mp_string(-A),
        "flipped_best_lambda": _mp_string(flipped_lambda),
        "flipped_best_residual": _mp_string(flipped_residual),
        "interpretation": (
            "Changing only the A,B,C,D D-sign does not recover the extra "
            "Sasaki-Einstein form; the best possible dphi=lambda psi residual "
            "is order one."
        ),
    }


def ball_formula_residual(A: mp.mpf, B: mp.mpf, C: mp.mpf, D: mp.mpf) -> Form:
    """Return the difference between computed dphi and Ball-Oliveira's formula."""
    return subtract(exterior_derivative(aloff_wallach_phi(A, B, C, D)), ball_oliveira_dphi_formula(A, B, C, D))


def hodge_star_diagonal(item: Form, scales: tuple[mp.mpf, ...]) -> Form:
    """Return the Hodge star for a diagonal metric in the given coframe."""
    dimension = len(scales)
    full_key = tuple(range(1, dimension + 1))
    result: Form = {}
    for key, coefficient in item.items():
        complement = tuple(index for index in full_key if index not in key)
        combined = key + complement
        inversions = sum(1 for i, left_index in enumerate(combined) for right_index in combined[i + 1 :] if left_index > right_index)
        numerator = mp.one
        denominator = mp.one
        for index in complement:
            numerator *= scales[index - 1]
        for index in key:
            denominator *= scales[index - 1]
        _add_to(result, complement, coefficient * ((-1) if inversions % 2 else 1) * numerator / denominator)
    return result


def principal_orbit_su3_from_g2(label: str, phi: Form, normal_scale: mp.mpf | int | float = 1) -> SU3AlgebraicCheck:
    """Restrict a 7D G2 form to the model principal orbit as an SU(3) pair."""
    omega = scale(1 / mp.mpf(normal_scale), contract_principal_normal(phi))
    gamma = restrict_to_principal_orbit(phi)
    return su3_algebraic_check(label, omega, gamma)


def known_solution_principal_su3_checks() -> tuple[SU3AlgebraicCheck, ...]:
    """Return principal-orbit SU(3) checks for the A,B,C,D known solutions."""
    checks = []
    for solution in n11_known_solutions():
        checks.append(principal_orbit_su3_from_g2(solution.label, aloff_wallach_phi(solution.A, solution.B, solution.C, solution.D)))
    return tuple(checks)


def _ordered_basis(seq: Iterable[int], coefficient: mp.mpf | int | float = 1) -> Form:
    """Return a basis form from possibly unordered indices."""
    key = tuple(seq)
    inversions = sum(1 for i, left_index in enumerate(key) for right_index in key[i + 1 :] if left_index > right_index)
    return basis(sorted(key), mp.mpf(coefficient) * ((-1) if inversions % 2 else 1))


def geipel_sasaki_einstein_differentials() -> dict[int, Form]:
    """Return Geipel's Sasaki-Einstein X_{1,1} structure equations.

    The ``e_8`` terms are connection terms for the quotient.  They cancel from
    the invariant Sasaki-Einstein forms below.  The ``d e_5`` sign follows by
    expanding Geipel's complex equation for ``d Theta^3`` and makes ``d^2 e_7``
    vanish.
    """
    sqrt3 = mp.sqrt(3)
    return {
        1: add(scale(sqrt3, _ordered_basis((8, 2))), scale(-1, _ordered_basis((7, 2))), scale(-1, basis((3, 5))), scale(-1, basis((4, 6)))),
        2: add(scale(-sqrt3, _ordered_basis((8, 1))), _ordered_basis((7, 1)), scale(-1, basis((3, 6))), basis((4, 5))),
        3: add(scale(-sqrt3, _ordered_basis((8, 4))), scale(-1, _ordered_basis((7, 4))), basis((1, 5)), basis((2, 6))),
        4: add(scale(sqrt3, _ordered_basis((8, 3))), _ordered_basis((7, 3)), basis((1, 6)), scale(-1, basis((2, 5)))),
        5: add(scale(-2, _ordered_basis((7, 6))), scale(-2, basis((1, 3))), scale(2, basis((2, 4)))),
        6: add(scale(2, _ordered_basis((7, 5))), scale(-2, basis((1, 4))), scale(-2, basis((2, 3)))),
        7: add(scale(2, basis((1, 2))), scale(2, basis((3, 4))), scale(2, basis((5, 6)))),
        8: {},
    }


def geipel_sasaki_einstein_su3_pair(phase: str = "real") -> tuple[Form, Form]:
    """Return the transverse SU(3) pair for Geipel's Sasaki-Einstein coframe."""
    omega = add(basis((1, 2)), basis((3, 4)), basis((5, 6)))
    real_gamma = add(
        basis((1, 3, 5)),
        scale(-1, basis((1, 4, 6))),
        scale(-1, basis((2, 3, 6))),
        scale(-1, basis((2, 4, 5))),
    )
    imag_gamma = add(
        scale(-1, basis((1, 3, 6))),
        scale(-1, basis((1, 4, 5))),
        scale(-1, basis((2, 3, 5))),
        basis((2, 4, 6)),
    )
    if phase == "real":
        return omega, real_gamma
    if phase == "imag":
        return omega, imag_gamma
    raise ValueError(f"unknown Sasaki-Einstein phase: {phase}")


def geipel_sasaki_einstein_phi(phase: str = "real", gamma_sign: int = 1) -> Form:
    """Return one Sasaki-Einstein nearly-parallel G2 form in Geipel's coframe."""
    omega, gamma = geipel_sasaki_einstein_su3_pair(phase)
    return add(wedge(basis((7,)), omega), scale(gamma_sign, gamma))


def geipel_to_ball_coframe_map() -> dict[int, tuple[int, mp.mpf]]:
    """Return the Weyl-adjusted diagonal map from Geipel coforms to Ball coforms."""
    return {
        1: (2, mp.mpf("0.5")),
        2: (6, mp.mpf("-0.5")),
        3: (3, mp.mpf("0.5")),
        4: (7, mp.mpf("-0.5")),
        5: (1, -1 / mp.sqrt(2)),
        6: (5, -1 / mp.sqrt(2)),
        7: (4, -1 / mp.sqrt(2)),
    }


def geipel_to_ball_form(item: Form) -> Form:
    """Rewrite a form from Geipel's Sasaki-Einstein coframe to Ball's coframe."""
    coframe_map = geipel_to_ball_coframe_map()
    result: Form = {}
    for key, coefficient in item.items():
        mapped_key = []
        mapped_coefficient = coefficient
        for index in key:
            ball_index, scale_factor = coframe_map[index]
            mapped_key.append(ball_index)
            mapped_coefficient *= scale_factor
        if len(set(mapped_key)) != len(mapped_key):
            continue
        inversions = sum(1 for i, left_index in enumerate(mapped_key) for right_index in mapped_key[i + 1 :] if left_index > right_index)
        _add_to(result, tuple(sorted(mapped_key)), mapped_coefficient * ((-1) if inversions % 2 else 1))
    return result


def ball_sasaki_einstein_metric_scales() -> tuple[mp.mpf, ...]:
    """Return metric coframe scales for the Sasaki-Einstein metric in Ball coforms."""
    scales = [mp.zero for _ in range(7)]
    for ball_index, scale_factor in geipel_to_ball_coframe_map().values():
        scales[ball_index - 1] = abs(scale_factor)
    return tuple(scales)


def ball_sasaki_einstein_phi(phase: str = "real", gamma_sign: int = 1) -> Form:
    """Return the extra Sasaki-Einstein G2 form in Ball's coframe."""
    return geipel_to_ball_form(geipel_sasaki_einstein_phi(phase, gamma_sign))


def ball_sasaki_einstein_principal_su3_check(phase: str = "real", gamma_sign: int = 1) -> SU3AlgebraicCheck:
    """Return the principal-orbit SU(3) check for the extra Sasaki-Einstein form."""
    normal_scale = ball_sasaki_einstein_metric_scales()[PRINCIPAL_ORBIT_NORMAL_7D_INDEX - 1]
    return principal_orbit_su3_from_g2(f"sasaki_einstein_{phase}_{'plus' if gamma_sign > 0 else 'minus'}", ball_sasaki_einstein_phi(phase, gamma_sign), normal_scale)


def geipel_sasaki_einstein_check(phase: str = "real", gamma_sign: int = 1) -> dict[str, str | bool]:
    """Verify the extra Sasaki-Einstein nearly-parallel structure."""
    phi = ball_sasaki_einstein_phi(phase, gamma_sign)
    metric_scales = ball_sasaki_einstein_metric_scales()
    psi = hodge_star_diagonal(phi, metric_scales)
    dphi = exterior_derivative(phi)
    source_phi = geipel_sasaki_einstein_phi(phase, gamma_sign)
    source_dphi = exterior_derivative(source_phi, geipel_sasaki_einstein_differentials())
    residual = subtract(dphi, scale(4, psi))
    e8_leak = max((abs(value) for key, value in source_dphi.items() if 8 in key), default=mp.zero)
    su3_check = ball_sasaki_einstein_principal_su3_check(phase, gamma_sign)
    normal_scale = metric_scales[PRINCIPAL_ORBIT_NORMAL_7D_INDEX - 1]
    return {
        "label": f"sasaki_einstein_{phase}_{'plus' if gamma_sign > 0 else 'minus'}",
        "source": "Geipel Sasaki-Einstein coframe pulled back to Ball-Oliveira coframe by a Weyl-adjusted diagonal map",
        "lambda": "4",
        "max_abs_dphi_minus_4psi": _mp_string(max_abs_coefficient(residual)),
        "max_abs_e8_terms_in_dphi": _mp_string(e8_leak),
        "su3_omega_wedge_gamma_residual": _mp_string(su3_check.omega_wedge_gamma_residual),
        "su3_hitchin_lambda": _mp_string(su3_check.hitchin_lambda),
        "su3_hitchin_complex_residual": _mp_string(su3_check.hitchin_complex_residual),
        "su3_volume_normalization_residual": _mp_string(su3_check.volume_normalization_residual),
        "principal_omega_coefficients": su3_check.omega_coefficients,
        "principal_gamma_coefficients": su3_check.gamma_coefficients,
        "ball_metric_scales": [_mp_string(value) for value in metric_scales],
        "principal_normal_scale": _mp_string(normal_scale),
        "tested_in_principal_orbit_coframe": True,
        "principal_orbit_note": "The model normal is omega_7, with unit coform (1/2) omega_7 for this metric.",
    }


def cohomogeneity_one_ansatz_summary() -> dict:
    """Return the N_{1,1} ansatz data needed before deriving a full ODE."""
    invariant_basis = principal_orbit_invariant_form_basis()
    return {
        "action": "SO(3)_real x SO(3)_fiber",
        "principal_orbit": "locally SO(3) x SO(3), modulo the principal Z_2 isotropy",
        "principal_isotropy_model": "diagonal half-turn with signs (-,-,+,-,-,+)",
        "singular_base_orbits": ["RP^2 real-line orbit in CP^2", "null-conic CP^1 orbit in CP^2"],
        "vertical_coframe": ["omega_1", "omega_4", "omega_5"],
        "horizontal_coframe": ["omega_2", "omega_6", "omega_3", "omega_7"],
        "principal_orbit_model_coframe": {
            "base_1": "omega_2",
            "base_2": "omega_6",
            "base_3": "omega_3",
            "fiber_1": "omega_1",
            "fiber_2": "omega_5",
            "fiber_3": "omega_4",
            "normal": "omega_7",
        },
        "principal_orbit_invariant_dimensions": {
            "one_forms": len(invariant_basis[1]),
            "two_forms": len(invariant_basis[2]),
            "three_forms": len(invariant_basis[3]),
        },
        "omega_t_basis": invariant_basis[2],
        "gamma_t_basis": invariant_basis[3],
        "homogeneous_calibration_family": (
            "phi(A,B,C,D)=ABC(omega_123-omega_167+omega_257-omega_356)"
            "-D omega_4 wedge (A^2 omega_15+B^2 omega_26+C^2 omega_37)"
        ),
        "extra_sasaki_einstein_readiness": (
            "the extra Sasaki-Einstein nearly-parallel form is outside A,B,C,D "
            "but invariant under the same SO(3)_real x SO(3)_fiber action"
        ),
        "search_readiness": (
            "principal-orbit SU(3)-structure variables and known targets verified; "
            "full cohomogeneity-one scout still needs evolution equations and "
            "endpoint smoothness"
        ),
    }


def principal_orbit_isotropy_signs() -> dict[str, int]:
    """Return the principal Z2 action on a local SO(3)xSO(3) coframe.

    The first three entries are the real-SO(3) base orbit coframe and the last
    three are the fiber SO(3) coframe.  A half-turn fixes the third direction in
    each factor and negates the transverse two-plane.
    """
    return {
        "base_1": -1,
        "base_2": -1,
        "base_3": 1,
        "fiber_1": -1,
        "fiber_2": -1,
        "fiber_3": 1,
    }


def principal_orbit_invariant_form_basis() -> dict[int, list[str]]:
    """Return Z2-invariant principal-orbit form monomials through degree three."""
    signs = principal_orbit_isotropy_signs()
    labels = tuple(signs)
    invariant: dict[int, list[str]] = {}
    for degree in (1, 2, 3):
        basis_labels = []
        for combo in combinations(labels, degree):
            sign = 1
            for label in combo:
                sign *= signs[label]
            if sign == 1:
                basis_labels.append("^".join(combo))
        invariant[degree] = basis_labels
    return invariant


def principal_orbit_form_basis_indices(degree: int) -> tuple[Index, ...]:
    """Return invariant principal-orbit basis keys in the fixed 6D order."""
    signs = tuple(principal_orbit_isotropy_signs().values())
    keys = []
    for combo in combinations(range(1, 7), degree):
        sign = 1
        for index in combo:
            sign *= signs[index - 1]
        if sign == 1:
            keys.append(combo)
    return tuple(keys)


def principal_orbit_su3_variable_basis() -> dict:
    """Return the full invariant principal-orbit SU(3) variable declaration."""
    omega_basis = principal_orbit_form_basis_indices(2)
    gamma_basis = principal_orbit_form_basis_indices(3)
    return {
        "coframe": list(PRINCIPAL_ORBIT_LABELS),
        "omega_variables": [
            {"name": f"x{index}", "basis": _basis_key_label(key)}
            for index, key in enumerate(omega_basis, start=1)
        ],
        "gamma_variables": [
            {"name": f"y{index}", "basis": _basis_key_label(key)}
            for index, key in enumerate(gamma_basis, start=1)
        ],
        "raw_coefficient_count": len(omega_basis) + len(gamma_basis),
        "algebraic_constraints": [
            {
                "name": "omega_wedge_gamma_missing_fiber_3",
                "basis": _basis_key_label((1, 2, 3, 4, 5)),
                "equation": "x1*y11 + x2*y8 - x3*y7 - x4*y4 + x5*y3 + x7*y1 = 0",
            },
            {
                "name": "omega_wedge_gamma_missing_base_3",
                "basis": _basis_key_label((1, 2, 4, 5, 6)),
                "equation": "x1*y12 - x2*y10 + x3*y9 + x4*y6 - x5*y5 + x7*y2 = 0",
            },
            {
                "name": "hitchin_volume_normalization",
                "equation": "K_gamma^2 = lambda(gamma) I with lambda(gamma)<0 and |omega^3/6| = sqrt(-lambda(gamma))/2",
            },
        ],
        "expected_su3_dimension": 16,
    }


def _mp_string(value: mp.mpf) -> str:
    """Return a stable high-precision scalar string."""
    return mp.nstr(value, 80)


def _su3_check_payload(check: SU3AlgebraicCheck) -> dict:
    """Return a JSON-ready SU(3) algebraic check payload."""
    return {
        "label": check.label,
        "omega_coefficients": check.omega_coefficients,
        "gamma_coefficients": check.gamma_coefficients,
        "omega_wedge_gamma_residual": _mp_string(check.omega_wedge_gamma_residual),
        "hitchin_lambda": _mp_string(check.hitchin_lambda),
        "hitchin_complex_residual": _mp_string(check.hitchin_complex_residual),
        "volume_normalization_residual": _mp_string(check.volume_normalization_residual),
        "orientation_sign": check.orientation_sign,
        "stable_negative": check.stable_negative,
    }


def build_summary() -> dict:
    """Return a JSON-ready N_{1,1} ansatz verification summary."""
    with mp.workdps(80):
        bracket_vs_formula = ball_formula_residual(mp.mpf("1.3"), mp.mpf("0.9"), mp.mpf("1.1"), mp.mpf("-0.7"))
        solutions = []
        for solution in n11_known_solutions():
            residual = nearly_parallel_residual(solution)
            solutions.append(
                {
                    "label": solution.label,
                    "metric_type": solution.metric_type,
                    "A": _mp_string(solution.A),
                    "B": _mp_string(solution.B),
                    "C": _mp_string(solution.C),
                    "D": _mp_string(solution.D),
                    "lambda": _mp_string(solution.lambda_value),
                    "max_abs_dphi_minus_lambda_psi": _mp_string(max_abs_coefficient(residual)),
                }
            )
        flip_probe = abcd_vertical_flip_probe()
        principal_su3_checks = [_su3_check_payload(check) for check in known_solution_principal_su3_checks()]
        sasaki_einstein_model_checks = [
            geipel_sasaki_einstein_check(phase, gamma_sign)
            for phase in ("real", "imag")
            for gamma_sign in (1, -1)
        ]
    extra_solution = n11_extra_sasaki_einstein_solution()
    return {
        "version": N11_ANSATZ_VERSION,
        "topology": "N_{1,1}=SU(3)/U(1)_{1,1}",
        "ansatz": cohomogeneity_one_ansatz_summary(),
        "principal_orbit_su3_variables": principal_orbit_su3_variable_basis(),
        "bracket_check": {
            "description": "computed exterior derivative of phi agrees with Ball-Oliveira formula",
            "max_abs_difference": _mp_string(max_abs_coefficient(bracket_vs_formula)),
        },
        "known_solutions": solutions,
        "known_solution_principal_su3_checks": principal_su3_checks,
        "sasaki_einstein_model_checks": sasaki_einstein_model_checks,
        "extra_sasaki_einstein_solution": {
            "label": extra_solution.label,
            "metric_type": extra_solution.metric_type,
            "source_formula": extra_solution.source_formula,
            "lambda_statement": extra_solution.lambda_statement,
            "in_abcd_family": extra_solution.in_abcd_family,
            "current_action_invariant": extra_solution.current_action_invariant,
            "needs_alternative_cohomogeneity_one_action": extra_solution.needs_alternative_cohomogeneity_one_action,
            "invariant_terms": list(extra_solution.invariant_terms),
            "explanation": extra_solution.explanation,
        },
        "abcd_vertical_flip_probe": flip_probe,
        "verdict": (
            "principal-orbit SU(3) variables are now explicit as 7 omega "
            "coefficients and 12 gamma coefficients with algebraic constraints; "
            "the two A,B,C,D known structures pass the model principal-orbit "
            "SU(3) checks, and the extra Sasaki-Einstein structures pass "
            "Ball-coframe nearly-parallel and model principal-orbit SU(3) checks"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Print the N_{1,1} ansatz verification summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)
    summary = build_summary()
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return
    print("Aloff-Wallach N_{1,1} ansatz verification", flush=True)
    print(f"version: {summary['version']}", flush=True)
    print(f"principal orbit: {summary['ansatz']['principal_orbit']}", flush=True)
    print(f"bracket/formula residual: {summary['bracket_check']['max_abs_difference']}", flush=True)
    for solution in summary["known_solutions"]:
        print(
            f"{solution['label']}: lambda={solution['lambda']}, "
            f"residual={solution['max_abs_dphi_minus_lambda_psi']}",
            flush=True,
        )
    extra = summary["extra_sasaki_einstein_solution"]
    print(
        f"{extra['label']}: in A,B,C,D={extra['in_abcd_family']}, "
        f"current-action-invariant={extra['current_action_invariant']}",
        flush=True,
    )
    print(
        "D-sign flip probe residual: "
        f"{summary['abcd_vertical_flip_probe']['flipped_best_residual']}",
        flush=True,
    )
    print("principal-orbit SU(3) checks:", flush=True)
    for check in summary["known_solution_principal_su3_checks"]:
        print(
            f"  {check['label']}: omega^gamma={check['omega_wedge_gamma_residual']}, "
            f"volume={check['volume_normalization_residual']}",
            flush=True,
        )
    for check in summary["sasaki_einstein_model_checks"]:
        print(
            f"  {check['label']}: dphi-4psi={check['max_abs_dphi_minus_4psi']}, "
            f"model-coframe={check['tested_in_principal_orbit_coframe']}",
            flush=True,
        )
    print(f"verdict: {summary['verdict']}", flush=True)


if __name__ == "__main__":
    main()
