"""Numerical evolution helpers for the N_{1,1} endpoint-reduced scout.

This is a first practical ODE layer for the full invariant principal-orbit
``SU(3)`` chart.  Endpoint values are reduced by smoothness, while higher
Taylor layers are fitted numerically against the cohomogeneity-one
nearly-parallel equations before one-sided marching.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TypeAlias

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from . import endpoint_smoothness


Index: TypeAlias = tuple[int, ...]
FloatForm: TypeAlias = dict[Index, float]

EVOLUTION_VERSION = "aloff-wallach-n11-evolution-v1"
OMEGA_BASIS: tuple[Index, ...] = ((1, 2), (1, 4), (1, 5), (2, 4), (2, 5), (3, 6), (4, 5))
GAMMA_BASIS: tuple[Index, ...] = (
    (1, 2, 3),
    (1, 2, 6),
    (1, 3, 4),
    (1, 3, 5),
    (1, 4, 6),
    (1, 5, 6),
    (2, 3, 4),
    (2, 3, 5),
    (2, 4, 6),
    (2, 5, 6),
    (3, 4, 5),
    (4, 5, 6),
)
FOUR_BASIS: tuple[Index, ...] = tuple(combinations(range(1, 7), 4))
THREE_BASIS_ALL: tuple[Index, ...] = tuple(combinations(range(1, 7), 3))
TWO_BASIS_ALL: tuple[Index, ...] = tuple(combinations(range(1, 7), 2))
SMOOTH_THREE_BASIS: tuple[tuple[int, ...], ...] = tuple(combinations(range(7), 3))
SMOOTH_THREE_INDEX = {key: index for index, key in enumerate(SMOOTH_THREE_BASIS)}


@dataclass(frozen=True)
class AWSettings:
    """Numerical controls for the first Aloff-Wallach scout."""

    lam: float = 4.0
    structure_scale: float | None = None
    base_structure_scale: float = -1.0
    fiber_structure_scale: float = -2.0
    endpoint_order: int = 2
    germ_epsilon: float = 1e-3
    germ_samples: tuple[float, ...] = (1e-3, 2e-3, 4e-3)
    max_tau: float = 2.0
    max_step: float = 0.02
    rtol: float = 1e-7
    atol: float = 1e-9
    residual_weight: float = 1.0
    constraint_weight: float = 10.0
    smoothness_weight: float = 10.0
    endpoint_weight: float = 100.0
    max_germ_evaluations: int = 250


@dataclass(frozen=True)
class EndpointConstants:
    """The four smooth zeroth-order constants at one singular endpoint."""

    A: float
    B: float
    C: float
    D: float


@dataclass(frozen=True)
class EndpointGerm:
    """A fitted smooth endpoint germ."""

    constants: EndpointConstants
    normal_weight: int
    coefficients: np.ndarray
    residual_norm: float
    success: bool
    message: str

    def state(self, tau: float) -> np.ndarray:
        """Evaluate the 19 principal coefficients at local time ``tau``."""
        powers = np.array([tau**degree for degree in range(self.coefficients.shape[1])], dtype=float)
        return self.coefficients @ powers

    def derivative(self, tau: float) -> np.ndarray:
        """Evaluate the derivative of the fitted coefficient polynomial."""
        values = np.zeros(19, dtype=float)
        for degree in range(1, self.coefficients.shape[1]):
            values += degree * self.coefficients[:, degree] * tau ** (degree - 1)
        return values


@dataclass(frozen=True)
class MarchResult:
    """One endpoint march stopped at maximal volume."""

    status: str
    tau: float | None
    state: np.ndarray | None
    volume: float | None
    volume_dot: float | None
    germ: EndpointGerm
    message: str | None = None


@dataclass(frozen=True)
class MatchResult:
    """Two-sided maximal-volume match for one reduced scout point."""

    left: MarchResult
    right: MarchResult
    residual_norm: float | None
    residual: tuple[float, ...]
    failure: str | None

    @property
    def reconstructed_interval(self) -> float | None:
        """Return the max-volume interval length, when both events exist."""
        if self.left.status != "max_volume" or self.right.status != "max_volume":
            return None
        if self.left.tau is None or self.right.tau is None:
            return None
        return self.left.tau + self.right.tau


def _merge(left: Index, right: Index) -> tuple[Index, int] | None:
    if set(left).intersection(right):
        return None
    combined = left + right
    inversions = sum(1 for i, left_index in enumerate(combined) for right_index in combined[i + 1 :] if left_index > right_index)
    return tuple(sorted(combined)), -1 if inversions % 2 else 1


def wedge(left: FloatForm, right: FloatForm) -> FloatForm:
    """Wedge two small forms."""
    result: FloatForm = {}
    for left_key, left_value in left.items():
        for right_key, right_value in right.items():
            merged = _merge(left_key, right_key)
            if merged is None:
                continue
            key, sign = merged
            result[key] = result.get(key, 0.0) + sign * left_value * right_value
    return {key: value for key, value in result.items() if abs(value) > 1e-14}


def scale_form(value: float, item: FloatForm) -> FloatForm:
    """Scale one form."""
    return {key: value * coefficient for key, coefficient in item.items() if abs(value * coefficient) > 1e-14}


def add_forms(*items: FloatForm) -> FloatForm:
    """Add forms."""
    result: FloatForm = {}
    for item in items:
        for key, value in item.items():
            result[key] = result.get(key, 0.0) + value
    return {key: value for key, value in result.items() if abs(value) > 1e-14}


def subtract_forms(left: FloatForm, right: FloatForm) -> FloatForm:
    """Subtract two forms."""
    return add_forms(left, scale_form(-1.0, right))


def contract(index: int, item: FloatForm) -> FloatForm:
    """Contract a form with the basis vector dual to ``e^index``."""
    result: FloatForm = {}
    for key, value in item.items():
        if index not in key:
            continue
        position = key.index(index)
        new_key = key[:position] + key[position + 1 :]
        result[new_key] = result.get(new_key, 0.0) + ((-1.0) ** position) * value
    return {key: value for key, value in result.items() if abs(value) > 1e-14}


def _basis_form(key: Index, coefficient: float = 1.0) -> FloatForm:
    return {key: coefficient}


def _principal_differentials(
    structure_scale: float | None = None,
    *,
    base_structure_scale: float = -1.0,
    fiber_structure_scale: float = -2.0,
) -> dict[int, FloatForm]:
    """Return the product ``so(3)+so(3)`` principal-orbit exterior algebra.

    The natural ``SO(3)_real x SO(3)_fiber`` action basis for ``N_{1,1}`` has
    different Maurer-Cartan scales on the base and fiber factors.  Passing
    ``structure_scale`` keeps the old common-scale debugging model.
    """
    if structure_scale is not None:
        base_structure_scale = structure_scale
        fiber_structure_scale = structure_scale
    b = base_structure_scale
    f = fiber_structure_scale
    return {
        1: _basis_form((2, 3), -b),
        2: _basis_form((3, 1), -b),
        3: _basis_form((1, 2), -b),
        4: _basis_form((5, 6), -f),
        5: _basis_form((6, 4), -f),
        6: _basis_form((4, 5), -f),
    }


def exterior_derivative(
    item: FloatForm,
    structure_scale: float | None = None,
    *,
    base_structure_scale: float = -1.0,
    fiber_structure_scale: float = -2.0,
) -> FloatForm:
    """Compute the principal-orbit exterior derivative."""
    differentials = _principal_differentials(
        structure_scale,
        base_structure_scale=base_structure_scale,
        fiber_structure_scale=fiber_structure_scale,
    )
    result: FloatForm = {}
    for key, coefficient in item.items():
        for position, index in enumerate(key):
            prefix = _basis_form(key[:position])
            suffix = _basis_form(key[position + 1 :])
            term = wedge(wedge(prefix, differentials[index]), suffix)
            result = add_forms(result, scale_form(((-1.0) ** position) * coefficient, term))
    return result


def _form_from_coefficients(values: np.ndarray, basis: tuple[Index, ...]) -> FloatForm:
    return {key: float(value) for key, value in zip(basis, values) if abs(float(value)) > 1e-14}


def omega_form(state: np.ndarray) -> FloatForm:
    """Return ``omega`` from a 19-vector."""
    return _form_from_coefficients(state[:7], OMEGA_BASIS)


def gamma_form(state: np.ndarray) -> FloatForm:
    """Return ``gamma=Re(Omega)`` from a 19-vector."""
    return _form_from_coefficients(state[7:], GAMMA_BASIS)


def _coefficients(item: FloatForm, basis: tuple[Index, ...]) -> np.ndarray:
    return np.array([item.get(key, 0.0) for key in basis], dtype=float)


def hitchin_k_matrix(gamma: FloatForm) -> np.ndarray:
    """Return Hitchin's K matrix in the ordered volume trivialization."""
    rows = np.zeros((6, 6), dtype=float)
    full_key = (1, 2, 3, 4, 5, 6)
    for column in range(1, 7):
        alpha = wedge(contract(column, gamma), gamma)
        for key, coefficient in alpha.items():
            missing = [index for index in full_key if index not in key]
            if len(missing) != 1:
                continue
            row = missing[0]
            expected_key = tuple(index for index in full_key if index != row)
            if key == expected_key:
                rows[row - 1, column - 1] = coefficient * ((-1.0) ** (row - 1))
    return rows


def hitchin_lambda(gamma: FloatForm) -> float:
    """Return Hitchin's quartic scalar."""
    matrix = hitchin_k_matrix(gamma)
    return float(np.trace(matrix @ matrix) / 6.0)


def _pullback_form(item: FloatForm, coform_matrix: np.ndarray) -> FloatForm:
    """Pull back a form by a linear map on coforms."""
    result: FloatForm = {}
    for key, coefficient in item.items():
        terms: FloatForm = {(): coefficient}
        for index in key:
            one_form = {
                (target + 1,): float(coform_matrix[index - 1, target])
                for target in range(coform_matrix.shape[1])
                if abs(float(coform_matrix[index - 1, target])) > 1e-14
            }
            terms = wedge(terms, one_form)
        result = add_forms(result, terms)
    return result


def hitchin_dual(gamma: FloatForm) -> FloatForm:
    """Return ``hat(gamma)=Im(Omega)`` for the negative stable component."""
    k_matrix = hitchin_k_matrix(gamma)
    lambda_value = float(np.trace(k_matrix @ k_matrix) / 6.0)
    if not np.isfinite(lambda_value) or lambda_value >= -1e-14:
        raise ValueError("gamma is not in the negative stable component")
    vector_j = k_matrix / np.sqrt(-lambda_value)
    return scale_form(-1.0, _pullback_form(gamma, vector_j))


def omega_volume(omega: FloatForm) -> float:
    """Return the coefficient of ``omega^3/6``."""
    return wedge(wedge(omega, omega), omega).get((1, 2, 3, 4, 5, 6), 0.0) / 6.0


def volume_dot(omega: FloatForm, omega_dot: FloatForm) -> float:
    """Return the derivative of the principal volume coefficient."""
    return wedge(wedge(omega, omega), omega_dot).get((1, 2, 3, 4, 5, 6), 0.0) / 2.0


def _solve_omega_dot(omega: FloatForm, rhs_form: FloatForm) -> tuple[np.ndarray, float]:
    """Solve ``omega wedge omega_dot = rhs_form`` in the invariant 2-form basis."""
    matrix = np.zeros((len(FOUR_BASIS), len(OMEGA_BASIS)), dtype=float)
    for column, key in enumerate(OMEGA_BASIS):
        matrix[:, column] = _coefficients(wedge(omega, _basis_form(key)), FOUR_BASIS)
    rhs = _coefficients(rhs_form, FOUR_BASIS)
    solution, residuals, _rank, _singular_values = np.linalg.lstsq(matrix, rhs, rcond=None)
    residual_norm = float(np.linalg.norm(matrix @ solution - rhs))
    return solution, residual_norm


def rhs(
    state: np.ndarray,
    lam: float = 4.0,
    structure_scale: float | None = None,
    *,
    base_structure_scale: float = -1.0,
    fiber_structure_scale: float = -2.0,
) -> np.ndarray:
    """Return the cohomogeneity-one nearly-parallel vector field."""
    omega = omega_form(state)
    gamma = gamma_form(state)
    hat = hitchin_dual(gamma)
    ydot_form = subtract_forms(
        exterior_derivative(
            omega,
            structure_scale,
            base_structure_scale=base_structure_scale,
            fiber_structure_scale=fiber_structure_scale,
        ),
        scale_form(lam, hat),
    )
    xdot, residual = _solve_omega_dot(
        omega,
        scale_form(
            -1.0,
            exterior_derivative(
                hat,
                structure_scale,
                base_structure_scale=base_structure_scale,
                fiber_structure_scale=fiber_structure_scale,
            ),
        ),
    )
    if residual > 1e-5:
        raise ValueError(f"omega-dot solve residual too large: {residual:g}")
    return np.concatenate([xdot, _coefficients(ydot_form, GAMMA_BASIS)])


def algebraic_residual(
    state: np.ndarray,
    lam: float = 4.0,
    structure_scale: float | None = None,
    *,
    base_structure_scale: float = -1.0,
    fiber_structure_scale: float = -2.0,
) -> np.ndarray:
    """Return algebraic nearly-parallel and SU(3) residuals."""
    omega = omega_form(state)
    gamma = gamma_form(state)
    residual = subtract_forms(
        exterior_derivative(
            gamma,
            structure_scale,
            base_structure_scale=base_structure_scale,
            fiber_structure_scale=fiber_structure_scale,
        ),
        scale_form(lam / 2.0, wedge(omega, omega)),
    )
    compatibility = wedge(omega, gamma)
    lambda_value = hitchin_lambda(gamma)
    omega_vol = omega_volume(omega)
    volume_residual = 0.0
    if lambda_value < 0:
        volume_residual = abs(abs(omega_vol) - np.sqrt(-lambda_value) / 2.0)
    else:
        volume_residual = 1e3 + abs(lambda_value)
    return np.concatenate(
        [
            _coefficients(residual, FOUR_BASIS),
            _coefficients(compatibility, ((1, 2, 3, 4, 5), (1, 2, 4, 5, 6))),
            np.array([volume_residual], dtype=float),
        ]
    )


def endpoint_zero_state(constants: EndpointConstants) -> np.ndarray:
    """Return the smoothness-forced zeroth-order principal coefficients."""
    state = np.zeros(19, dtype=float)
    state[7:] = np.array(
        [
            constants.A,
            -constants.A,
            constants.B,
            constants.C,
            constants.B,
            constants.C,
            -constants.C,
            constants.B,
            -constants.C,
            constants.B,
            constants.D,
            -constants.D,
        ],
        dtype=float,
    )
    return state


def _smoothness_residual(coefficients: np.ndarray, normal_weight: int) -> np.ndarray:
    """Return linear smoothness residuals for a coefficient matrix."""
    terms_by_variable = endpoint_smoothness._principal_terms_for_linear_algebra()
    order = coefficients.shape[1] - 1
    residuals = []
    for smooth_degree in range(-1, order + 1):
        vector = np.zeros(len(SMOOTH_THREE_BASIS), dtype=float)
        for variable_index, (_name, terms) in enumerate(terms_by_variable):
            for collapse_shift, key, coefficient in terms:
                principal_order = smooth_degree + collapse_shift
                if 0 <= principal_order <= order:
                    vector[SMOOTH_THREE_INDEX[key]] += coefficient * coefficients[variable_index, principal_order]
        orthogonal = endpoint_smoothness._evaluation_subspace_orthogonal(normal_weight, smooth_degree)
        residuals.append(orthogonal @ vector)
    return np.concatenate(residuals) if residuals else np.zeros(0, dtype=float)


def _poly_state(coefficients: np.ndarray, tau: float) -> np.ndarray:
    powers = np.array([tau**degree for degree in range(coefficients.shape[1])], dtype=float)
    return coefficients @ powers


def _poly_derivative(coefficients: np.ndarray, tau: float) -> np.ndarray:
    derivative = np.zeros(19, dtype=float)
    for degree in range(1, coefficients.shape[1]):
        derivative += degree * coefficients[:, degree] * tau ** (degree - 1)
    return derivative


def fit_endpoint_germ(
    constants: EndpointConstants,
    normal_weight: int,
    settings: AWSettings,
    *,
    structure_scale: float | None = None,
) -> EndpointGerm:
    """Fit a smooth endpoint Taylor germ for fixed four-constant data."""
    structure_scale = settings.structure_scale if structure_scale is None else structure_scale
    base_structure_scale = settings.base_structure_scale
    fiber_structure_scale = settings.fiber_structure_scale
    zero_state = endpoint_zero_state(constants)
    order = settings.endpoint_order
    initial = np.zeros((19, order + 1), dtype=float)
    initial[:, 0] = zero_state
    scale = max(1.0, float(np.linalg.norm(zero_state)))
    for index in range(7):
        initial[index, 1] = 0.05 * scale

    def objective(flat: np.ndarray) -> np.ndarray:
        coefficients = flat.reshape(19, order + 1)
        residuals = [
            settings.endpoint_weight * (coefficients[:, 0] - zero_state),
            settings.smoothness_weight * _smoothness_residual(coefficients, normal_weight),
        ]
        for tau in settings.germ_samples:
            state = _poly_state(coefficients, tau)
            derivative = _poly_derivative(coefficients, tau)
            try:
                vector = rhs(
                    state,
                    settings.lam,
                    structure_scale,
                    base_structure_scale=base_structure_scale,
                    fiber_structure_scale=fiber_structure_scale,
                )
                residuals.append(settings.residual_weight * (derivative - vector))
                residuals.append(
                    settings.constraint_weight
                    * algebraic_residual(
                        state,
                        settings.lam,
                        structure_scale,
                        base_structure_scale=base_structure_scale,
                        fiber_structure_scale=fiber_structure_scale,
                    )
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                residuals.append(np.full(19, 1e3, dtype=float))
                residuals.append(np.full(len(FOUR_BASIS) + 3, 1e3, dtype=float))
        return np.concatenate(residuals)

    result = least_squares(
        objective,
        initial.ravel(),
        max_nfev=settings.max_germ_evaluations,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    coefficients = result.x.reshape(19, order + 1)
    residual_norm = float(np.linalg.norm(objective(result.x), ord=np.inf))
    return EndpointGerm(constants, normal_weight, coefficients, residual_norm, bool(result.success), result.message)


def march_to_max_volume(
    germ: EndpointGerm,
    settings: AWSettings,
    *,
    structure_scale: float | None = None,
) -> MarchResult:
    """March one fitted endpoint germ until the principal volume is stationary."""
    structure_scale = settings.structure_scale if structure_scale is None else structure_scale
    base_structure_scale = settings.base_structure_scale
    fiber_structure_scale = settings.fiber_structure_scale
    tau0 = settings.germ_epsilon
    state0 = germ.state(tau0)

    def ode(_tau: float, state: np.ndarray) -> np.ndarray:
        return rhs(
            state,
            settings.lam,
            structure_scale,
            base_structure_scale=base_structure_scale,
            fiber_structure_scale=fiber_structure_scale,
        )

    def event(_tau: float, state: np.ndarray) -> float:
        vector = rhs(
            state,
            settings.lam,
            structure_scale,
            base_structure_scale=base_structure_scale,
            fiber_structure_scale=fiber_structure_scale,
        )
        return volume_dot(omega_form(state), _form_from_coefficients(vector[:7], OMEGA_BASIS))

    event.terminal = True  # type: ignore[attr-defined]
    event.direction = 0  # type: ignore[attr-defined]
    try:
        initial_event = event(tau0, state0)
        solution = solve_ivp(
            ode,
            (tau0, settings.max_tau),
            state0,
            method="RK45",
            rtol=settings.rtol,
            atol=settings.atol,
            max_step=settings.max_step,
            events=event,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return MarchResult("march_failure", None, None, None, None, germ, str(exc))
    if solution.t_events and len(solution.t_events[0]) > 0:
        tau = float(solution.t_events[0][0])
        state = solution.y_events[0][0]
        vector = rhs(
            state,
            settings.lam,
            structure_scale,
            base_structure_scale=base_structure_scale,
            fiber_structure_scale=fiber_structure_scale,
        )
        return MarchResult(
            "max_volume",
            tau,
            state,
            omega_volume(omega_form(state)),
            volume_dot(omega_form(state), _form_from_coefficients(vector[:7], OMEGA_BASIS)),
            germ,
        )
    if not solution.success:
        return MarchResult("march_failure", None, None, None, None, germ, solution.message)
    message = f"no max-volume event; initial volume_dot={initial_event:g}"
    return MarchResult("no_max_volume", None, None, None, None, germ, message)


def max_volume_match(
    left_constants: EndpointConstants,
    right_constants: EndpointConstants,
    settings: AWSettings | None = None,
    *,
    structure_scale: float | None = None,
) -> MatchResult:
    """Fit both endpoint germs, march to maximal volume, and compare states."""
    settings = settings or AWSettings()
    structure_scale = settings.structure_scale if structure_scale is None else structure_scale
    left_germ = fit_endpoint_germ(left_constants, 1, settings, structure_scale=structure_scale)
    right_germ = fit_endpoint_germ(right_constants, 2, settings, structure_scale=structure_scale)
    left = march_to_max_volume(left_germ, settings, structure_scale=structure_scale)
    right = march_to_max_volume(right_germ, settings, structure_scale=structure_scale)
    if left.status != "max_volume":
        return MatchResult(left, right, None, (), f"left:{left.status}")
    if right.status != "max_volume":
        return MatchResult(left, right, None, (), f"right:{right.status}")
    assert left.state is not None
    assert right.state is not None
    residual = tuple(float(value) for value in (left.state - right.state))
    return MatchResult(left, right, float(np.linalg.norm(left.state - right.state, ord=np.inf)), residual, None)
