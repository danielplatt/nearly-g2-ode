"""Singular-end and positive-centre Taylor series for the weighted charts."""

from __future__ import annotations

from mpmath import mp

from solver.series import Series, differentiate_coefficients, state_to_coefficients, state_to_series

from .charts import LEFT_CHART, RIGHT_CHART, WeightedChart, _tau_power
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


def initial_weighted_series(chart: WeightedChart, params: ProblemParameters, config: SolverConfig) -> State[list[mp.mpf]]:
    """Build the singular-end weighted series for one endpoint chart."""
    order = config.series_order
    y0 = endpoint_zero_jet(chart.name, params)
    y1 = endpoint_first_jet(chart.name, params)
    return build_weighted_series(chart, mp.zero, y0, order, params, first_jet=y1)


def initial_left_series(params: ProblemParameters, config: SolverConfig) -> State[list[mp.mpf]]:
    """Build the left weighted singular-end Taylor series."""
    return initial_weighted_series(LEFT_CHART, params, config)


def initial_right_series(params: ProblemParameters, config: SolverConfig) -> State[list[mp.mpf]]:
    """Build the right weighted singular-end Taylor series."""
    return initial_weighted_series(RIGHT_CHART, params, config)
