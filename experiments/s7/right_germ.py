"""Numerical right-end Taylor germs for S7 p2/p3 terminal charts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from mpmath import mp
from scipy.optimize import least_squares

from problem import (
    FixedRightEndpointData,
    ProblemParameters,
    S7_P2_RIGHT_CHART,
    S7_P3_RIGHT_CHART,
    SolverConfig,
    State,
    initial_right_series,
    q_rhs,
    round_s7_candidate_parameters,
    squashed_s7_parameters,
)

from .right_moduli_chart import p2_offset, p3_offset


DEFAULT_SOLVER_MAX_NFEV = 240
DEFAULT_RESIDUAL_TOLERANCE = mp.mpf("1e-9")
SAMPLE_TAUS = ("0.001", "0.002", "0.005", "0.01", "0.02", "0.04")


@dataclass(frozen=True)
class S7RightGermPoint:
    """Scaled coordinates for a numerical S7 right-end germ."""

    u: mp.mpf
    v: mp.mpf
    r: mp.mpf


@dataclass(frozen=True)
class S7RightGermSpec:
    """Chart-specific gauge choices for one S7 right endpoint family."""

    target: str
    base_params: ProblemParameters
    chart_name: str
    offset_variable_names: tuple[str, ...]
    offset_values: Callable[[State[mp.mpf]], tuple[mp.mpf, ...]]
    offset_builder: Callable[[tuple[mp.mpf, ...]], State[mp.mpf]]
    offset_anchor_indices: tuple[int, int]
    first_anchor_component: int


@dataclass(frozen=True)
class S7RightGermSolution:
    """One solved S7 right-end germ."""

    point: S7RightGermPoint
    fixed_right: FixedRightEndpointData
    residual_norm: mp.mpf
    success: bool
    message: str
    evaluations: int


def firstjet_anchor_components(target: str) -> tuple[int, int, int]:
    """Return three right first-jet components used by the scout germ chart."""
    if target == "round":
        return 2, 3, 4
    if target == "squashed":
        return 1, 3, 4
    raise ValueError(f"Unknown S7 right-germ target {target!r}; choose 'round' or 'squashed'.")


def _base_offset_moduli(target: str) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """Return the known `(A, B, C)` offset-moduli triple for one S7 target."""
    if target == "round":
        fixed = round_s7_candidate_parameters().fixed_right
        if fixed is None:
            raise ValueError("Round S7 target has no fixed right endpoint.")
        return fixed.offset.y1, fixed.offset.y4, fixed.offset.y7
    if target == "squashed":
        fixed = squashed_s7_parameters().fixed_right
        if fixed is None:
            raise ValueError("Squashed S7 target has no fixed right endpoint.")
        return fixed.offset.y1, fixed.offset.y4, fixed.offset.y6
    raise ValueError(f"Unknown S7 right-germ target {target!r}; choose 'round' or 'squashed'.")


def offset_moduli_from_point(target: str, point: S7RightGermPoint) -> State[mp.mpf]:
    """Return a terminal offset from scaled `(A, B, C)` moduli coordinates."""
    a0, b0, c0 = _base_offset_moduli(target)
    a = a0 * mp.exp(point.u)
    b = b0 * mp.exp(point.v)
    c = c0 * mp.exp(point.r)
    if target == "round":
        return p3_offset(a, b, c)
    if target == "squashed":
        return p2_offset(a, b, c)
    raise ValueError(f"Unknown S7 right-germ target {target!r}; choose 'round' or 'squashed'.")


def right_germ_spec(target: str) -> S7RightGermSpec:
    """Return the numerical right-germ chart specification for one target."""
    if target == "round":
        base = round_s7_candidate_parameters()
        return S7RightGermSpec(
            target="round",
            base_params=base,
            chart_name=S7_P3_RIGHT_CHART.name,
            offset_variable_names=("q1", "q2", "q3", "q4", "q7", "q8"),
            offset_values=lambda q: (q.y1, q.y2, q.y3, q.y4, q.y7, q.y8),
            offset_builder=lambda values: State(values[0], values[1], values[2], values[3], -values[3], -values[2], values[4], values[5]),
            offset_anchor_indices=(0, 1),
            first_anchor_component=2,
        )
    if target == "squashed":
        base = squashed_s7_parameters()
        return S7RightGermSpec(
            target="squashed",
            base_params=base,
            chart_name=S7_P2_RIGHT_CHART.name,
            offset_variable_names=("q1", "q2", "q3", "q4", "q6", "q8"),
            offset_values=lambda q: (q.y1, q.y2, q.y3, q.y4, q.y6, q.y8),
            offset_builder=lambda values: State(values[0], values[1], values[2], values[3], -values[3], values[4], -values[1], values[5]),
            offset_anchor_indices=(0, 1),
            first_anchor_component=1,
        )
    raise ValueError(f"Unknown S7 right-germ target {target!r}; choose 'round' or 'squashed'.")


def _chart_for_spec(spec: S7RightGermSpec):
    """Return the weighted right chart for one spec."""
    if spec.chart_name == S7_P3_RIGHT_CHART.name:
        return S7_P3_RIGHT_CHART
    if spec.chart_name == S7_P2_RIGHT_CHART.name:
        return S7_P2_RIGHT_CHART
    raise ValueError(f"Unknown S7 right chart {spec.chart_name!r}.")


def _base_template(spec: S7RightGermSpec, order: int) -> tuple[tuple[mp.mpf, ...], State[list[mp.mpf]]]:
    """Return base offset variables and explicit known coefficient template."""
    config = SolverConfig(order, 80, 30, mp.mpf("0.5"), 0, spec.base_params.interval_end / 2)
    coefficients = initial_right_series(spec.base_params, config)
    if spec.base_params.fixed_right is None:
        raise ValueError("S7 right-germ target must provide fixed right endpoint data.")
    return spec.offset_values(spec.base_params.fixed_right.offset), coefficients


def _anchor_values(
    spec: S7RightGermSpec,
    point: S7RightGermPoint,
    base_offsets: tuple[mp.mpf, ...],
    base_coefficients: State[list[mp.mpf]],
) -> dict[tuple[str, int, int | None], mp.mpf]:
    """Return hard coordinate anchors for one scaled right-germ point."""
    first_component = spec.first_anchor_component
    first_value = list(base_coefficients)[first_component][1]
    return {
        ("offset", spec.offset_anchor_indices[0], None): base_offsets[spec.offset_anchor_indices[0]] * mp.exp(point.u),
        ("offset", spec.offset_anchor_indices[1], None): base_offsets[spec.offset_anchor_indices[1]] * mp.exp(point.v),
        ("coefficient", first_component, 1): first_value * (1 + point.r),
    }


def _pack_variables(
    spec: S7RightGermSpec,
    base_offsets: tuple[mp.mpf, ...],
    base_coefficients: State[list[mp.mpf]],
    anchors: dict[tuple[str, int, int | None], mp.mpf],
    order: int,
) -> tuple[np.ndarray, list[tuple[str, int, int | None]], np.ndarray]:
    """Pack all non-anchored offset and coefficient variables."""
    values: list[float] = []
    keys: list[tuple[str, int, int | None]] = []
    for index, value in enumerate(base_offsets):
        key = ("offset", index, None)
        if key in anchors:
            continue
        keys.append(key)
        values.append(float(value))
    for degree in range(order + 1):
        for component_index, component in enumerate(base_coefficients):
            key = ("coefficient", component_index, degree)
            if key in anchors:
                continue
            keys.append(key)
            values.append(float(component[degree]))
    initial = np.asarray(values, dtype=np.float64)
    return initial.copy(), keys, initial


def _unpack_solution(
    spec: S7RightGermSpec,
    values: np.ndarray,
    keys: list[tuple[str, int, int | None]],
    anchors: dict[tuple[str, int, int | None], mp.mpf],
    base_offsets: tuple[mp.mpf, ...],
    base_coefficients: State[list[mp.mpf]],
    order: int,
) -> tuple[State[mp.mpf], State[list[mp.mpf]]]:
    """Build offset and coefficient states from packed variables plus anchors."""
    offset_values = [mp.mpf(value) for value in base_offsets]
    coefficient_values = [list(component) for component in base_coefficients]
    for key, value in anchors.items():
        kind, index, degree = key
        if kind == "offset":
            offset_values[index] = value
        else:
            coefficient_values[index][degree] = value
    for key, value in zip(keys, values):
        kind, index, degree = key
        if kind == "offset":
            offset_values[index] = mp.mpf(value)
        else:
            coefficient_values[index][degree] = mp.mpf(value)
    for component in coefficient_values:
        del component[order + 1 :]
    return spec.offset_builder(tuple(offset_values)), State.from_iterable(coefficient_values)


def _evaluate_coefficients(coefficients: list[mp.mpf], tau: mp.mpf) -> mp.mpf:
    """Evaluate one Taylor coefficient list at a local positive tau."""
    value = mp.zero
    for coefficient in reversed(coefficients):
        value = value * tau + coefficient
    return value


def _evaluate_derivative(coefficients: list[mp.mpf], tau: mp.mpf) -> mp.mpf:
    """Evaluate the tau derivative of one Taylor coefficient list."""
    value = mp.zero
    for degree in range(len(coefficients) - 1, 0, -1):
        value = value * tau + degree * coefficients[degree]
    return value


def _sampled_residual_vector(
    spec: S7RightGermSpec,
    offset: State[mp.mpf],
    coefficients: State[list[mp.mpf]],
    sample_taus: tuple[mp.mpf, ...],
) -> np.ndarray:
    """Return raw ODE residuals sampled near the right endpoint."""
    chart = _chart_for_spec(spec)
    with mp.workdps(40):
        fixed = FixedRightEndpointData(
            label=f"{spec.target}_right_germ_trial",
            offset=offset,
            zero_jet=State.from_iterable(component[0] for component in coefficients),
            first_jet=State.from_iterable(component[1] for component in coefficients),
        )
        params = ProblemParameters(
            lam=spec.base_params.lam,
            interval_end=spec.base_params.interval_end,
            left=spec.base_params.left,
            right=spec.base_params.right,
            right_chart=spec.base_params.right_chart,
            fixed_right=fixed,
        )
        residuals: list[float] = []
        for tau in sample_taus:
            y = State.from_iterable(_evaluate_coefficients(list(component), tau) for component in coefficients)
            ydot = State.from_iterable(_evaluate_derivative(list(component), tau) for component in coefficients)
            q = chart.y_to_q(tau, y, params)
            local_qdot = State.from_iterable(
                weight * tau ** (weight - 1) * value + tau**weight * dot
                for weight, value, dot in zip(chart.weights, y, ydot)
            )
            rhs = chart.local_q_rhs(tau, q, params)
            raw = local_qdot - rhs
            scale = max(mp.one, max(abs(value) for value in rhs))
            residuals.extend(float(value / scale) for value in raw)
    return np.asarray(residuals, dtype=np.float64)


def solve_right_germ(
    target: str,
    point: S7RightGermPoint,
    *,
    order: int,
    max_nfev: int = DEFAULT_SOLVER_MAX_NFEV,
    residual_tolerance: mp.mpf = DEFAULT_RESIDUAL_TOLERANCE,
) -> S7RightGermSolution:
    """Solve a numerical S7 p2/p3 right-end Taylor germ near one known target."""
    spec = right_germ_spec(target)
    base_offsets, base_coefficients = _base_template(spec, order)
    if point.u == 0 and point.v == 0 and point.r == 0:
        if spec.base_params.fixed_right is None:
            raise ValueError("S7 right-germ target must provide fixed right endpoint data.")
        fixed = FixedRightEndpointData(
            label=f"{spec.target}_right_germ",
            offset=spec.base_params.fixed_right.offset,
            zero_jet=State.from_iterable(component[0] for component in base_coefficients),
            first_jet=State.from_iterable(component[1] for component in base_coefficients),
            series_coefficients=State.from_iterable(
                tuple(value for value in component[: order + 1]) for component in base_coefficients
            ),
        )
        return S7RightGermSolution(
            point=point,
            fixed_right=fixed,
            residual_norm=mp.zero,
            success=True,
            message="exact homogeneous right germ",
            evaluations=0,
        )
    anchors = _anchor_values(spec, point, base_offsets, base_coefficients)
    initial, keys, reference = _pack_variables(spec, base_offsets, base_coefficients, anchors, order)
    sample_taus = tuple(mp.mpf(value) for value in SAMPLE_TAUS)
    regularization_weight = 1e-8

    def build_fixed(values: np.ndarray, *, label: str = f"{target}_right_germ") -> tuple[FixedRightEndpointData, mp.mpf]:
        offset, coefficients = _unpack_solution(spec, values, keys, anchors, base_offsets, base_coefficients, order)
        final_residual = _sampled_residual_vector(spec, offset, coefficients, sample_taus)
        residual_norm = mp.mpf(float(np.max(np.abs(final_residual))))
        fixed = FixedRightEndpointData(
            label=label,
            offset=offset,
            zero_jet=State.from_iterable(component[0] for component in coefficients),
            first_jet=State.from_iterable(component[1] for component in coefficients),
            series_coefficients=State.from_iterable(
                tuple(value for value in component[: order + 1]) for component in coefficients
            ),
        )
        return fixed, residual_norm

    try:
        fixed, residual_norm = build_fixed(initial)
    except Exception:
        fixed = None
        residual_norm = mp.inf
    if residual_norm <= residual_tolerance and fixed is not None:
        return S7RightGermSolution(
            point=point,
            fixed_right=fixed,
            residual_norm=residual_norm,
            success=True,
            message="base template satisfies endpoint equations",
            evaluations=0,
        )

    best_values = initial.copy()
    best_norm = float(residual_norm) if residual_norm != mp.inf else np.inf

    def objective(values: np.ndarray) -> np.ndarray:
        nonlocal best_values, best_norm
        offset, coefficients = _unpack_solution(spec, values, keys, anchors, base_offsets, base_coefficients, order)
        try:
            residual = _sampled_residual_vector(spec, offset, coefficients, sample_taus)
        except Exception:
            return np.full(len(sample_taus) * 8 + len(reference), 1e12, dtype=np.float64)
        residual_norm_float = float(np.max(np.abs(residual)))
        if residual_norm_float < best_norm:
            best_norm = residual_norm_float
            best_values = values.copy()
        regularization = regularization_weight * (values - reference)
        return np.concatenate([residual, regularization])

    result = least_squares(
        objective,
        initial,
        method="trf",
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        max_nfev=max_nfev,
    )
    try:
        fixed, residual_norm = build_fixed(result.x)
    except Exception:
        fixed, residual_norm = build_fixed(best_values)
    success = bool(result.success and residual_norm <= residual_tolerance)
    return S7RightGermSolution(
        point=point,
        fixed_right=fixed,
        residual_norm=residual_norm,
        success=success,
        message=str(result.message),
        evaluations=int(result.nfev),
    )


def solve_right_firstjet_germ(
    target: str,
    point: S7RightGermPoint,
    *,
    order: int,
    max_nfev: int = 120,
    residual_tolerance: mp.mpf = mp.mpf("1e-6"),
) -> S7RightGermSolution:
    """Solve an S7 right germ with fixed terminal offset and three first-jet anchors."""
    spec = right_germ_spec(target)
    _base_offsets, base_coefficients = _base_template(spec, order)
    if spec.base_params.fixed_right is None:
        raise ValueError("S7 right-germ target must provide fixed right endpoint data.")
    offset = spec.base_params.fixed_right.offset
    if point.u == 0 and point.v == 0 and point.r == 0:
        fixed = FixedRightEndpointData(
            label=f"{spec.target}_right_firstjet_germ",
            offset=offset,
            zero_jet=State.from_iterable(component[0] for component in base_coefficients),
            first_jet=State.from_iterable(component[1] for component in base_coefficients),
            series_coefficients=State.from_iterable(
                tuple(value for value in component[: order + 1]) for component in base_coefficients
            ),
        )
        return S7RightGermSolution(point, fixed, mp.zero, True, "exact homogeneous right first-jet germ", 0)

    anchor_components = firstjet_anchor_components(target)
    coordinate_values = (point.u, point.v, point.r)
    anchors: dict[tuple[str, int, int | None], mp.mpf] = {}
    for component, coordinate in zip(anchor_components, coordinate_values):
        base_value = list(base_coefficients)[component][1]
        anchors[("coefficient", component, 1)] = base_value * (1 + coordinate)

    values: list[float] = []
    keys: list[tuple[str, int, int | None]] = []
    for degree in range(order + 1):
        for component_index, component in enumerate(base_coefficients):
            key = ("coefficient", component_index, degree)
            if key in anchors:
                continue
            keys.append(key)
            values.append(float(component[degree]))
    initial = np.asarray(values, dtype=np.float64)
    reference = initial.copy()
    sample_taus = tuple(mp.mpf(value) for value in SAMPLE_TAUS)
    regularization_weight = 1e-8

    def unpack(current: np.ndarray) -> State[list[mp.mpf]]:
        components = [list(component) for component in base_coefficients]
        for key, value in anchors.items():
            _kind, component, degree = key
            components[component][degree] = value
        for key, value in zip(keys, current):
            _kind, component, degree = key
            components[component][degree] = mp.mpf(value)
        return State.from_iterable(components)

    def build_fixed(current: np.ndarray) -> tuple[FixedRightEndpointData, mp.mpf]:
        coefficients = unpack(current)
        residual = _sampled_residual_vector(spec, offset, coefficients, sample_taus)
        residual_norm = mp.mpf(float(np.max(np.abs(residual))))
        fixed = FixedRightEndpointData(
            label=f"{spec.target}_right_firstjet_germ",
            offset=offset,
            zero_jet=State.from_iterable(component[0] for component in coefficients),
            first_jet=State.from_iterable(component[1] for component in coefficients),
            series_coefficients=State.from_iterable(
                tuple(value for value in component[: order + 1]) for component in coefficients
            ),
        )
        return fixed, residual_norm

    try:
        fixed, residual_norm = build_fixed(initial)
    except Exception:
        fixed = None
        residual_norm = mp.inf
    if fixed is not None and residual_norm <= residual_tolerance:
        return S7RightGermSolution(point, fixed, residual_norm, True, "first-jet template satisfies sampled ODE", 0)

    best_values = initial.copy()
    best_norm = float(residual_norm) if residual_norm != mp.inf else np.inf

    def objective(current: np.ndarray) -> np.ndarray:
        nonlocal best_values, best_norm
        coefficients = unpack(current)
        try:
            residual = _sampled_residual_vector(spec, offset, coefficients, sample_taus)
        except Exception:
            return np.full(len(sample_taus) * 8 + len(reference), 1e12, dtype=np.float64)
        residual_norm_float = float(np.max(np.abs(residual)))
        if residual_norm_float < best_norm:
            best_norm = residual_norm_float
            best_values = current.copy()
        return np.concatenate([residual, regularization_weight * (current - reference)])

    result = least_squares(
        objective,
        initial,
        method="trf",
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        max_nfev=max_nfev,
    )
    try:
        fixed, residual_norm = build_fixed(result.x)
    except Exception:
        fixed, residual_norm = build_fixed(best_values)
    success = bool(residual_norm <= residual_tolerance)
    return S7RightGermSolution(point, fixed, residual_norm, success, str(result.message), int(result.nfev))


def solve_right_offset_moduli_germ(
    target: str,
    point: S7RightGermPoint,
    *,
    order: int,
    max_nfev: int = 300,
    residual_tolerance: mp.mpf = mp.mpf("1e-8"),
) -> S7RightGermSolution:
    """Solve an S7 right germ with the derived terminal offset moduli chart.

    The scaled coordinates move the three terminal offset parameters `(A, B, C)`
    in the p2/p3 families derived in `docs/s7-right-endpoint-moduli.md`.  Unlike
    the older fixed-offset first-jet solver, this routine solves the full
    weighted Taylor coefficient block simultaneously.
    """
    spec = right_germ_spec(target)
    _base_offsets, base_coefficients = _base_template(spec, order)
    offset = offset_moduli_from_point(target, point)
    if point.u == 0 and point.v == 0 and point.r == 0:
        fixed = FixedRightEndpointData(
            label=f"{spec.target}_right_offset_moduli_germ",
            offset=offset,
            zero_jet=State.from_iterable(component[0] for component in base_coefficients),
            first_jet=State.from_iterable(component[1] for component in base_coefficients),
            series_coefficients=State.from_iterable(
                tuple(value for value in component[: order + 1]) for component in base_coefficients
            ),
        )
        return S7RightGermSolution(point, fixed, mp.zero, True, "exact homogeneous right offset-moduli germ", 0)

    values: list[float] = []
    keys: list[tuple[int, int]] = []
    for degree in range(order + 1):
        for component_index, component in enumerate(base_coefficients):
            keys.append((component_index, degree))
            values.append(float(component[degree]))
    initial = np.asarray(values, dtype=np.float64)
    reference = initial.copy()
    sample_taus = tuple(mp.mpf(value) for value in SAMPLE_TAUS)
    regularization_weight = 1e-9

    def unpack(current: np.ndarray) -> State[list[mp.mpf]]:
        components = [list(component) for component in base_coefficients]
        for (component, degree), value in zip(keys, current):
            components[component][degree] = mp.mpf(value)
        return State.from_iterable(components)

    def build_fixed(current: np.ndarray) -> tuple[FixedRightEndpointData, mp.mpf]:
        coefficients = unpack(current)
        residual = _sampled_residual_vector(spec, offset, coefficients, sample_taus)
        residual_norm = mp.mpf(float(np.max(np.abs(residual))))
        fixed = FixedRightEndpointData(
            label=f"{spec.target}_right_offset_moduli_germ",
            offset=offset,
            zero_jet=State.from_iterable(component[0] for component in coefficients),
            first_jet=State.from_iterable(component[1] for component in coefficients),
            series_coefficients=State.from_iterable(
                tuple(value for value in component[: order + 1]) for component in coefficients
            ),
        )
        return fixed, residual_norm

    try:
        fixed, residual_norm = build_fixed(initial)
    except Exception:
        fixed = None
        residual_norm = mp.inf
    if fixed is not None and residual_norm <= residual_tolerance:
        return S7RightGermSolution(point, fixed, residual_norm, True, "base template satisfies endpoint equations", 0)

    best_values = initial.copy()
    best_norm = float(residual_norm) if residual_norm != mp.inf else np.inf

    def objective(current: np.ndarray) -> np.ndarray:
        nonlocal best_values, best_norm
        coefficients = unpack(current)
        try:
            residual = _sampled_residual_vector(spec, offset, coefficients, sample_taus)
        except Exception:
            return np.full(len(sample_taus) * 8 + len(reference), 1e12, dtype=np.float64)
        residual_norm_float = float(np.max(np.abs(residual)))
        if residual_norm_float < best_norm:
            best_norm = residual_norm_float
            best_values = current.copy()
        return np.concatenate([residual, regularization_weight * (current - reference)])

    result = least_squares(
        objective,
        initial,
        method="trf",
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        max_nfev=max_nfev,
    )
    try:
        fixed, residual_norm = build_fixed(result.x)
    except Exception:
        fixed, residual_norm = build_fixed(best_values)
    success = bool(residual_norm <= residual_tolerance)
    return S7RightGermSolution(point, fixed, residual_norm, success, str(result.message), int(result.nfev))


def params_with_right_germ(
    *,
    target: str,
    point: S7RightGermPoint,
    left_params,
    interval_end: mp.mpf,
    order: int,
) -> tuple[ProblemParameters, S7RightGermSolution]:
    """Build problem parameters with a solved numerical S7 right germ."""
    spec = right_germ_spec(target)
    solution = solve_right_germ(target, point, order=order)
    params = ProblemParameters(
        lam=spec.base_params.lam,
        interval_end=interval_end,
        left=left_params,
        right=spec.base_params.right,
        right_chart=spec.base_params.right_chart,
        fixed_right=solution.fixed_right,
    )
    return params, solution


def params_with_right_firstjet_germ(
    *,
    target: str,
    point: S7RightGermPoint,
    left_params,
    interval_end: mp.mpf,
    order: int,
) -> tuple[ProblemParameters, S7RightGermSolution]:
    """Build problem parameters with a solved fixed-offset, first-jet right germ."""
    spec = right_germ_spec(target)
    solution = solve_right_firstjet_germ(target, point, order=order)
    params = ProblemParameters(
        lam=spec.base_params.lam,
        interval_end=interval_end,
        left=left_params,
        right=spec.base_params.right,
        right_chart=spec.base_params.right_chart,
        fixed_right=solution.fixed_right,
    )
    return params, solution


def params_with_right_offset_moduli_germ(
    *,
    target: str,
    point: S7RightGermPoint,
    left_params,
    interval_end: mp.mpf,
    order: int,
) -> tuple[ProblemParameters, S7RightGermSolution]:
    """Build problem parameters with a solved offset-moduli right germ."""
    spec = right_germ_spec(target)
    solution = solve_right_offset_moduli_germ(target, point, order=order)
    params = ProblemParameters(
        lam=spec.base_params.lam,
        interval_end=interval_end,
        left=left_params,
        right=spec.base_params.right,
        right_chart=spec.base_params.right_chart,
        fixed_right=solution.fixed_right,
    )
    return params, solution
