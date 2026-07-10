"""Endpoint germs and max-volume matching for the S7 Sp(1)xSp(1)xU(1) system."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Literal

import numpy as np
from mpmath import mp
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from . import sp1xsp1xu1_system as system
from . import su2_cubed_action_audit


MATCHING_VERSION = "s7-sp1xsp1xu1-matching-v1"
Side = Literal["left", "right"]

STATE_LABELS = (
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "y1",
    "y2",
    "y3",
    "y4",
    "y5",
    "y6",
    "y7",
    "y8",
)
LEFT_WEIGHTS = np.array((4, 1, 1, 1, 3, 4, 2, 3, 2, 2, 0, 2, 2), dtype=int)
RIGHT_WEIGHTS = np.array((1, 4, 1, 1, 3, 0, 2, 2, 2, 2, 4, 3, 2), dtype=int)


@dataclass(frozen=True)
class MatchingSettings:
    """Numerical controls for endpoint fitting and max-volume marching."""

    lam: float = 4.0
    endpoint_order: int = 4
    germ_epsilon: float = 1e-4
    germ_samples: tuple[float, ...] = (1e-4, 2e-4, 4e-4, 8e-4)
    max_tau: float = 2.0
    max_step: float = 0.01
    rtol: float = 1e-8
    atol: float = 1e-10
    endpoint_weight: float = 100.0
    ode_weight: float = 1.0
    constraint_weight: float = 10.0
    regularization_weight: float = 1e-5
    max_germ_evaluations: int = 300


@dataclass(frozen=True)
class EndpointParameters:
    """Leading regular endpoint coefficients in the five-parameter chart."""

    A3: float
    A4: float
    B2: float
    B4: float
    C: float


@dataclass(frozen=True)
class EndpointGerm:
    """A regular-variable endpoint Taylor germ."""

    side: Side
    parameters: EndpointParameters
    weights: np.ndarray
    coefficients: np.ndarray
    residual_norm: float
    success: bool
    message: str
    source: str = "fit"

    def regular_state(self, tau: float) -> np.ndarray:
        """Evaluate the regular variables at local time tau."""
        powers = np.array([tau**degree for degree in range(self.coefficients.shape[1])], dtype=float)
        return self.coefficients @ powers

    def state(self, tau: float) -> np.ndarray:
        """Evaluate the raw 13-variable state at local time tau."""
        return (tau**self.weights) * self.regular_state(tau)

    def derivative(self, tau: float) -> np.ndarray:
        """Evaluate d/dtau of the raw state."""
        regular = self.regular_state(tau)
        regular_derivative = np.zeros(13, dtype=float)
        for degree in range(1, self.coefficients.shape[1]):
            regular_derivative += degree * self.coefficients[:, degree] * tau ** (degree - 1)
        values = np.zeros(13, dtype=float)
        for index, weight in enumerate(self.weights):
            if weight:
                values[index] += weight * tau ** (weight - 1) * regular[index]
            values[index] += tau**weight * regular_derivative[index]
        return values


@dataclass(frozen=True)
class MarchResult:
    """One side marched to a max-volume section."""

    status: str
    tau: float | None
    state: np.ndarray | None
    volume: float | None
    volume_dot: float | None
    volume_sign: float | None
    germ: EndpointGerm
    message: str | None = None


@dataclass(frozen=True)
class MatchResult:
    """Two endpoint marches compared at their max-volume sections."""

    left: MarchResult
    right: MarchResult
    residual_norm: float | None
    residual: tuple[float, ...]
    failure: str | None

    @property
    def reconstructed_interval(self) -> float | None:
        """Return the sum of the two one-sided max-volume times."""
        if self.left.tau is None or self.right.tau is None:
            return None
        return self.left.tau + self.right.tau


def weights_for_side(side: Side) -> np.ndarray:
    """Return regular-variable weights for one endpoint side."""
    if side == "left":
        return LEFT_WEIGHTS.copy()
    if side == "right":
        return RIGHT_WEIGHTS.copy()
    raise ValueError(f"unknown endpoint side {side!r}")


def local_rhs(side: Side, state: np.ndarray, lam: float) -> np.ndarray:
    """Return d/dtau in the local endpoint coordinate."""
    vector = np.array([float(value) for value in system.rhs(tuple(float(v) for v in state), lam)], dtype=float)
    return vector if side == "left" else -vector


def algebraic_residual_vector(state: np.ndarray, lam: float) -> np.ndarray:
    """Return the three scalar algebraic residuals as a float vector."""
    residual = system.algebraic_residual(tuple(float(value) for value in state), lam)
    return np.array(
        [
            float(residual["d_gamma_minus_lambda_omega2_over_2"]),
            float(residual["omega_wedge_gamma"]),
            float(residual["volume_normalization"]),
        ],
        dtype=float,
    )


def volume(state: np.ndarray) -> float:
    """Return the signed principal-orbit volume coefficient."""
    omega, _gamma = system.state_omega_gamma(tuple(float(value) for value in state))
    return float(system.omega_volume(omega))


def volume_dot_from_vector(state: np.ndarray, vector: np.ndarray) -> float:
    """Return d/dtau of the signed principal-orbit volume."""
    omega, _gamma = system.state_omega_gamma(tuple(float(value) for value in state))
    omega_dot = system.omega_form(tuple(float(value) for value in vector[:5]))
    return float(system.volume_coefficient(system.wedge(system.wedge(omega, omega), omega_dot)) / 2)


def endpoint_initial_regular_values(side: Side, parameters: EndpointParameters, lam: float) -> np.ndarray:
    """Return the fixed order-zero regular values for an endpoint chart."""
    values = np.zeros(13, dtype=float)
    values[2] = parameters.A3
    values[3] = parameters.A4
    values[6] = parameters.B2
    values[8] = parameters.B4
    values[9] = -lam * parameters.A4**2 / 6.0 - parameters.B2
    values[12] = -lam * parameters.A3 * parameters.A4 / 6.0 - parameters.B4
    if side == "left":
        values[10] = parameters.C
    else:
        values[5] = parameters.C
    return values


def _initial_coefficients(side: Side, parameters: EndpointParameters, settings: MatchingSettings) -> np.ndarray:
    coefficients = np.zeros((13, settings.endpoint_order + 1), dtype=float)
    coefficients[:, 0] = endpoint_initial_regular_values(side, parameters, settings.lam)
    scale = max(1.0, float(np.linalg.norm(coefficients[:, 0], ord=np.inf)))
    for index, weight in enumerate(weights_for_side(side)):
        if weight > 0 and settings.endpoint_order >= 1:
            coefficients[index, 1] = 0.01 * scale
    return coefficients


def _state_from_coefficients(coefficients: np.ndarray, weights: np.ndarray, tau: float) -> np.ndarray:
    powers = np.array([tau**degree for degree in range(coefficients.shape[1])], dtype=float)
    return (tau**weights) * (coefficients @ powers)


def _derivative_from_coefficients(coefficients: np.ndarray, weights: np.ndarray, tau: float) -> np.ndarray:
    regular = coefficients @ np.array([tau**degree for degree in range(coefficients.shape[1])], dtype=float)
    regular_derivative = np.zeros(13, dtype=float)
    for degree in range(1, coefficients.shape[1]):
        regular_derivative += degree * coefficients[:, degree] * tau ** (degree - 1)
    values = np.zeros(13, dtype=float)
    for index, weight in enumerate(weights):
        if weight:
            values[index] += weight * tau ** (weight - 1) * regular[index]
        values[index] += tau**weight * regular_derivative[index]
    return values


def fit_endpoint_germ(side: Side, parameters: EndpointParameters, settings: MatchingSettings | None = None) -> EndpointGerm:
    """Fit an endpoint germ using smooth weights and the full ODE residual."""
    settings = settings or MatchingSettings()
    weights = weights_for_side(side)
    initial = _initial_coefficients(side, parameters, settings)
    fixed_zero = endpoint_initial_regular_values(side, parameters, settings.lam)

    def objective(flat: np.ndarray) -> np.ndarray:
        coefficients = flat.reshape(13, settings.endpoint_order + 1)
        residuals = [
            settings.endpoint_weight * (coefficients[:, 0] - fixed_zero),
            settings.regularization_weight * coefficients[:, 1:].ravel(),
        ]
        for tau in settings.germ_samples:
            state = _state_from_coefficients(coefficients, weights, tau)
            derivative = _derivative_from_coefficients(coefficients, weights, tau)
            try:
                vector = local_rhs(side, state, settings.lam)
                residuals.append(settings.ode_weight * (derivative - vector))
                residuals.append(settings.constraint_weight * algebraic_residual_vector(state, settings.lam))
            except (ArithmeticError, FloatingPointError, ValueError, np.linalg.LinAlgError):
                residuals.append(np.full(13, 1e3, dtype=float))
                residuals.append(np.full(3, 1e3, dtype=float))
        return np.concatenate(residuals)

    result = least_squares(
        objective,
        initial.ravel(),
        max_nfev=settings.max_germ_evaluations,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    coefficients = result.x.reshape(13, settings.endpoint_order + 1)
    residual_norm = float(np.linalg.norm(objective(result.x), ord=np.inf))
    return EndpointGerm(side, parameters, weights, coefficients, residual_norm, bool(result.success), result.message)


def target_endpoint_germ(
    target: su2_cubed_action_audit.PodestaTarget,
    side: Side,
    endpoint_order: int = 8,
) -> EndpointGerm:
    """Return the exact regular Taylor germ induced by a known target."""
    embedded = system.embedded_target(target)
    weights = weights_for_side(side)
    endpoint = mp.zero if side == "left" else mp.pi / 2
    coefficients = np.zeros((13, endpoint_order + 1), dtype=float)

    with mp.workdps(80):
        for index, function in enumerate(embedded.state_functions):
            for degree in range(endpoint_order + 1):
                derivative_order = int(weights[index] + degree)
                if side == "left":
                    local_function = function
                else:
                    local_function = lambda tau, function=function: function(endpoint - tau)
                if derivative_order == 0:
                    coefficient = local_function(mp.zero)
                else:
                    coefficient = mp.diff(local_function, mp.zero, derivative_order) / mp.mpf(factorial(derivative_order))
                coefficients[index, degree] = float(coefficient)
    parameters = endpoint_parameters_from_germ(side, coefficients, float(target.lam))
    return EndpointGerm(side, parameters, weights, coefficients, 0.0, True, "exact known target", source=target.name)


def endpoint_parameters_from_germ(side: Side, coefficients: np.ndarray, lam: float) -> EndpointParameters:
    """Extract the five leading endpoint parameters from regular coefficients."""
    c0 = coefficients[:, 0]
    surviving = c0[10] if side == "left" else c0[5]
    return EndpointParameters(float(c0[2]), float(c0[3]), float(c0[6]), float(c0[8]), float(surviving))


def target_endpoint_parameters(target: su2_cubed_action_audit.PodestaTarget, side: Side) -> EndpointParameters:
    """Return the endpoint parameters induced by a known target."""
    return target_endpoint_germ(target, side, endpoint_order=0).parameters


def rescale_endpoint_parameters(
    parameters: EndpointParameters,
    source_lam: float,
    target_lam: float,
) -> EndpointParameters:
    """Rescale endpoint parameters from one nearly-parallel lambda to another.

    If the metric is scaled by ``r^2``, then ``lambda`` scales by ``1/r``.
    The endpoint parameters ``A3,A4,B2,B4`` have regular weight one after this
    homothety, while the surviving-volume coefficient ``C`` has weight three.
    """
    if target_lam == 0:
        raise ValueError("target_lam must be nonzero")
    ratio = float(source_lam) / float(target_lam)
    return EndpointParameters(
        parameters.A3 * ratio,
        parameters.A4 * ratio,
        parameters.B2 * ratio,
        parameters.B4 * ratio,
        parameters.C * ratio**3,
    )


def target_endpoint_parameters_at_lambda(
    target: su2_cubed_action_audit.PodestaTarget,
    side: Side,
    lam: float,
) -> EndpointParameters:
    """Return known-target endpoint parameters rescaled to a chosen lambda."""
    return rescale_endpoint_parameters(target_endpoint_parameters(target, side), float(target.lam), lam)


def march_to_max_volume(germ: EndpointGerm, settings: MatchingSettings | None = None) -> MarchResult:
    """March an endpoint germ to the first stationary positive-volume slice."""
    settings = settings or MatchingSettings()
    tau0 = settings.germ_epsilon
    state0 = germ.state(tau0)
    initial_volume = volume(state0)
    if not np.isfinite(initial_volume) or abs(initial_volume) < 1e-14:
        return MarchResult("bad_initial_volume", None, None, None, None, None, germ)
    volume_sign = 1.0 if initial_volume > 0 else -1.0

    def ode(_tau: float, state: np.ndarray) -> np.ndarray:
        return local_rhs(germ.side, state, settings.lam)

    def event(_tau: float, state: np.ndarray) -> float:
        vector = ode(_tau, state)
        return volume_sign * volume_dot_from_vector(state, vector)

    event.terminal = True  # type: ignore[attr-defined]
    event.direction = -1  # type: ignore[attr-defined]
    try:
        initial_event = event(tau0, state0)
        solution = solve_ivp(
            ode,
            (tau0, settings.max_tau),
            state0,
            rtol=settings.rtol,
            atol=settings.atol,
            max_step=settings.max_step,
            events=event,
        )
    except (ArithmeticError, FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        return MarchResult("march_failure", None, None, None, None, volume_sign, germ, str(exc))

    if solution.t_events and len(solution.t_events[0]) > 0:
        tau = float(solution.t_events[0][0])
        state = solution.y_events[0][0]
        vector = local_rhs(germ.side, state, settings.lam)
        return MarchResult(
            "max_volume",
            tau,
            state,
            volume_sign * volume(state),
            volume_sign * volume_dot_from_vector(state, vector),
            volume_sign,
            germ,
        )
    if not solution.success:
        return MarchResult("march_failure", None, None, None, None, volume_sign, germ, solution.message)
    return MarchResult("no_max_volume", None, None, None, None, volume_sign, germ, f"initial volume_dot={initial_event:g}")


def max_volume_match(
    left_parameters: EndpointParameters,
    right_parameters: EndpointParameters,
    settings: MatchingSettings | None = None,
) -> MatchResult:
    """Fit both endpoint germs, march to max volume, and compare states."""
    settings = settings or MatchingSettings()
    left = march_to_max_volume(fit_endpoint_germ("left", left_parameters, settings), settings)
    right = march_to_max_volume(fit_endpoint_germ("right", right_parameters, settings), settings)
    return match_marched_results(left, right)


def match_marched_results(left: MarchResult, right: MarchResult) -> MatchResult:
    """Compare two already marched endpoint solutions."""
    if left.status != "max_volume":
        return MatchResult(left, right, None, (), f"left:{left.status}")
    if right.status != "max_volume":
        return MatchResult(left, right, None, (), f"right:{right.status}")
    assert left.state is not None
    assert right.state is not None
    residual = tuple(float(value) for value in left.state - right.state)
    return MatchResult(left, right, float(np.linalg.norm(left.state - right.state, ord=np.inf)), residual, None)


def known_target_match(
    target: su2_cubed_action_audit.PodestaTarget,
    settings: MatchingSettings | None = None,
    endpoint_order: int = 10,
) -> MatchResult:
    """Recover one known target using exact endpoint Taylor germs."""
    settings = settings or MatchingSettings(lam=float(target.lam), endpoint_order=min(endpoint_order, 6))
    settings = MatchingSettings(
        lam=float(target.lam),
        endpoint_order=settings.endpoint_order,
        germ_epsilon=settings.germ_epsilon,
        germ_samples=settings.germ_samples,
        max_tau=settings.max_tau,
        max_step=settings.max_step,
        rtol=settings.rtol,
        atol=settings.atol,
        endpoint_weight=settings.endpoint_weight,
        ode_weight=settings.ode_weight,
        constraint_weight=settings.constraint_weight,
        regularization_weight=settings.regularization_weight,
        max_germ_evaluations=settings.max_germ_evaluations,
    )
    left = march_to_max_volume(target_endpoint_germ(target, "left", endpoint_order), settings)
    right = march_to_max_volume(target_endpoint_germ(target, "right", endpoint_order), settings)
    return match_marched_results(left, right)


def known_recovery_summary() -> dict[str, object]:
    """Return round and squashed max-volume recovery diagnostics."""
    rows = {}
    for target in (su2_cubed_action_audit.round_target(), su2_cubed_action_audit.squashed_target()):
        settings = MatchingSettings(lam=float(target.lam), endpoint_order=6, germ_epsilon=1e-3, max_step=0.005)
        match = known_target_match(target, settings, endpoint_order=14)
        rows[target.name] = {
            "lambda": float(target.lam),
            "failure": match.failure,
            "residual_norm": match.residual_norm,
            "reconstructed_interval": match.reconstructed_interval,
            "left_tau": match.left.tau,
            "right_tau": match.right.tau,
            "left_volume": match.left.volume,
            "right_volume": match.right.volume,
        }
    return {"version": MATCHING_VERSION, "known_recoveries": rows}
