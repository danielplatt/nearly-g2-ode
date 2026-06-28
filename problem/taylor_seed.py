"""Singular-end and positive-centre Taylor series for the weighted charts."""

from __future__ import annotations

from mpmath import mp

from solver.series import Series, differentiate_coefficients, state_to_coefficients, state_to_series

from .charts import (
    LEFT_CHART,
    RIGHT_CHART,
    S7_P2_RIGHT_CHART,
    S7_P3_RIGHT_CHART,
    WeightedChart,
    _tau_power,
    right_chart_for_params,
)
from .initial_data import ProblemParameters, SolverConfig, endpoint_first_jet, endpoint_zero_jet
from .types import State


def _zero_coeffs(order: int) -> list[mp.mpf]:
    """Allocate one zero coefficient list through the requested order."""
    return [mp.zero for _ in range(order + 1)]


def _coefficient_vector(series_state: State, degree: int) -> list[mp.mpf]:
    """Extract one coefficient level from a state of truncated series."""
    return [component.coeff(degree) for component in series_state]


def _solve_linear_system(matrix_rows: list[list[mp.mpf]], rhs: list[mp.mpf]) -> list[mp.mpf]:
    """Solve one dense 8x8 linear system with LU factorization."""
    matrix = mp.matrix(matrix_rows)
    vector = mp.matrix([[value] for value in rhs])
    solution = mp.lu_solve(matrix, vector)
    return [solution[row] for row in range(solution.rows)]


def _next_coefficients(
    chart: WeightedChart,
    y_coeffs: State[list[mp.mpf]],
    centre: mp.mpf,
    degree: int,
    order: int,
    params: ProblemParameters,
) -> list[mp.mpf]:
    """Solve for the next weighted coefficient vector c_(degree+1)."""
    tau = Series.constant(centre, order) + Series.variable(order)
    base_state = state_to_series(y_coeffs)
    base_rhs = chart.y_rhs(tau, base_state, params)
    constant = _coefficient_vector(base_rhs, degree)
    columns = []
    for index in range(8):
        mutable = [component[:] for component in y_coeffs]
        mutable[index][degree + 1] += mp.one
        trial_rhs = chart.y_rhs(tau, state_to_series(State.from_iterable(mutable)), params)
        trial = _coefficient_vector(trial_rhs, degree)
        columns.append([left - right for left, right in zip(trial, constant)])
    matrix = []
    for row in range(8):
        entries = []
        for col in range(8):
            diagonal = degree + 1 if row == col else 0
            entries.append(diagonal - columns[col][row])
        matrix.append(entries)
    return _solve_linear_system(matrix, constant)


def weighted_m_minus_one_residual(chart: WeightedChart, params: ProblemParameters) -> State[mp.mpf]:
    """Return the singular coefficient that must vanish for one endpoint zero jet."""
    order = 3
    y_series = State.from_iterable([[value] + [mp.zero] * order for value in endpoint_zero_jet(chart.name, params)])
    tau = Series.variable(order)
    qdot = chart.local_q_rhs(tau, state_to_series(chart.y_series_to_q_series(y_series, params, order)), params)
    numerators = []
    y_state = state_to_series(y_series)
    for weight, y_value, qdot_value in zip(chart.weights, y_state, qdot):
        numerators.append(qdot_value - weight * _tau_power(tau, weight - 1) * y_value)
    degrees = tuple(weight - 1 for weight in chart.weights)
    return State.from_iterable(series.coeff(degree) for series, degree in zip(numerators, degrees))


def weighted_series_residual(
    chart: WeightedChart,
    y_coeffs: State[list[mp.mpf]],
    centre: mp.mpf,
    params: ProblemParameters,
) -> State[list[mp.mpf]]:
    """Return the weighted coefficient residual y' - y_rhs for one series patch."""
    order = len(y_coeffs.y1) - 1
    tau = Series.constant(centre, order) + Series.variable(order)
    y_state = state_to_series(y_coeffs)
    residual = state_to_series(differentiate_coefficients(y_coeffs)) - chart.y_rhs(tau, y_state, params)
    return state_to_coefficients(residual)


def build_weighted_series(
    chart: WeightedChart,
    centre: mp.mpf,
    y0: State[mp.mpf],
    order: int,
    params: ProblemParameters,
    first_jet: State[mp.mpf] | None = None,
) -> State[list[mp.mpf]]:
    """Build one weighted Taylor series about a local centre."""
    y_coeffs = State.from_iterable([[value] + [mp.zero] * order for value in y0])
    if first_jet is not None and order >= 1:
        for component, value in zip(y_coeffs, first_jet):
            component[1] = value
    start_degree = 1 if first_jet is not None else 0
    for degree in range(start_degree, order):
        predicted_coeffs = _next_coefficients(chart, y_coeffs, centre, degree, order, params)
        for component, predicted in zip(y_coeffs, predicted_coeffs):
            component[degree + 1] = predicted
    return y_coeffs


def _constant_series(value: mp.mpf, order: int) -> list[mp.mpf]:
    """Return a Taylor series for one constant value."""
    return [value] + [mp.zero for _ in range(order)]


def _cos_right_series(phase: mp.mpf, frequency: int, order: int) -> list[mp.mpf]:
    """Return coefficients for cos(phase - frequency*tau)."""
    return [
        (-frequency) ** degree * mp.cos(phase + degree * mp.pi / 2) / mp.factorial(degree)
        for degree in range(order + 1)
    ]


def _sin_right_series(phase: mp.mpf, frequency: int, order: int) -> list[mp.mpf]:
    """Return coefficients for sin(phase - frequency*tau)."""
    return [
        (-frequency) ** degree * mp.sin(phase + degree * mp.pi / 2) / mp.factorial(degree)
        for degree in range(order + 1)
    ]


def _linear_series(order: int, constant: mp.mpf, terms: list[tuple[mp.mpf, list[mp.mpf]]]) -> list[mp.mpf]:
    """Return a linear combination of scalar Taylor series."""
    coeffs = _constant_series(constant, order)
    for scale, series in terms:
        for degree, value in enumerate(series):
            coeffs[degree] += scale * value
    return coeffs


def _scaled_series(scale: mp.mpf, coeffs: list[mp.mpf]) -> list[mp.mpf]:
    """Scale one Taylor coefficient list."""
    return [scale * value for value in coeffs]


def _squashed_s7_right_q_series(params: ProblemParameters, order: int) -> State[list[mp.mpf]]:
    """Return q(t) coefficients at the right endpoint for the explicit squashed-S7 curve."""
    sqrt5_over_25 = mp.sqrt(5) / 25
    end = params.interval_end
    cos_t = _cos_right_series(end, 1, order)
    cos_t_plus_pi_over_3 = _cos_right_series(end + mp.pi / 3, 1, order)
    sin_t_plus_pi_over_6 = _sin_right_series(end + mp.pi / 6, 1, order)
    cos_2t = _cos_right_series(2 * end, 2, order)
    cos_2t_plus_pi_over_3 = _cos_right_series(2 * end + mp.pi / 3, 2, order)
    sin_2t_plus_pi_over_6 = _sin_right_series(2 * end + mp.pi / 6, 2, order)
    cos_3t = _cos_right_series(3 * end, 3, order)

    return State(
        _constant_series(sqrt5_over_25, order),
        _scaled_series(sqrt5_over_25, _linear_series(order, -mp.one, [(2, cos_t_plus_pi_over_3)])),
        _scaled_series(sqrt5_over_25, _linear_series(order, mp.one, [(-2, sin_t_plus_pi_over_6)])),
        _scaled_series(sqrt5_over_25, _linear_series(order, -5, [(2, cos_t), (-12, cos_2t)])),
        _scaled_series(sqrt5_over_25, _linear_series(order, -mp.one, [(-2, cos_t)])),
        _scaled_series(
            sqrt5_over_25,
            _linear_series(order, 5, [(2, sin_t_plus_pi_over_6), (-12, cos_2t_plus_pi_over_3)]),
        ),
        _scaled_series(
            sqrt5_over_25,
            _linear_series(order, -5, [(12, sin_2t_plus_pi_over_6), (-2, cos_t_plus_pi_over_3)]),
        ),
        _scaled_series(sqrt5_over_25, _linear_series(order, 13, [(32, cos_3t)])),
    )


def _squashed_s7_p2_right_series(params: ProblemParameters, order: int) -> State[list[mp.mpf]]:
    """Build the fixed p2-right weighted series from the explicit squashed-S7 q(t)."""
    if params.fixed_right is None or params.fixed_right.label != "squashed_s7":
        raise ValueError("Explicit S7 p2 series is only available for the squashed_s7 endpoint.")

    q_series = _squashed_s7_right_q_series(params, order)
    return S7_P2_RIGHT_CHART.q_series_to_y_series(q_series, params, order)


def _round_s7_p3_right_series(params: ProblemParameters, order: int) -> State[list[mp.mpf]]:
    """Build the fixed p3-right weighted series from the derived round-S7 q(t)."""
    if params.fixed_right is None or params.fixed_right.label != "round_s7":
        raise ValueError("Explicit S7 p3 series is only available for the round_s7 endpoint.")

    squashed = _squashed_s7_right_q_series(params, order)
    q_series = State(
        squashed.y1,
        squashed.y3,
        squashed.y2,
        squashed.y4,
        squashed.y5,
        squashed.y7,
        squashed.y6,
        squashed.y8,
    )
    return S7_P3_RIGHT_CHART.q_series_to_y_series(q_series, params, order)


def initial_weighted_series(chart: WeightedChart, params: ProblemParameters, config: SolverConfig) -> State[list[mp.mpf]]:
    """Build the singular-end weighted series for one endpoint chart."""
    order = config.series_order
    if chart.name in {"s7_p2_right", "s7_p3_right"} and params.fixed_right is not None:
        coefficients = params.fixed_right.series_coefficients
        if coefficients is not None:
            if any(len(component) < order + 1 for component in coefficients):
                raise ValueError("Precomputed fixed-right series is shorter than the requested Taylor order.")
            return State.from_iterable([value for value in component[: order + 1]] for component in coefficients)
    if chart.name == "s7_p2_right" and params.fixed_right is not None and params.fixed_right.label == "squashed_s7":
        return _squashed_s7_p2_right_series(params, order)
    if chart.name == "s7_p3_right" and params.fixed_right is not None and params.fixed_right.label == "round_s7":
        return _round_s7_p3_right_series(params, order)
    y0 = endpoint_zero_jet(chart.name, params)
    y1 = endpoint_first_jet(chart.name, params)
    return build_weighted_series(chart, mp.zero, y0, order, params, first_jet=y1)


def initial_left_series(params: ProblemParameters, config: SolverConfig) -> State[list[mp.mpf]]:
    """Build the left weighted singular-end Taylor series."""
    return initial_weighted_series(LEFT_CHART, params, config)


def initial_right_series(params: ProblemParameters, config: SolverConfig) -> State[list[mp.mpf]]:
    """Build the right weighted singular-end Taylor series."""
    return initial_weighted_series(right_chart_for_params(params), params, config)
