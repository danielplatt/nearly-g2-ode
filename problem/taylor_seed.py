"""Singular-end Taylor seed generation from the geometric q-system."""

from __future__ import annotations

from mpmath import mp

from solver.series import Series, differentiate_coefficients, state_to_coefficients, state_to_series

from .initial_data import ProblemParameters, SolverConfig, y_first_jet, y_zero_jet
from .q_system import q_rhs
from .types import State
from .weights import q_series_to_y_series, y_series_to_q_series


SEED_DEGREES = (3, 2, 2, 3, 3, 2, 2, 3)


def _zero_coeffs(order: int) -> list[mp.mpf]:
    """Allocate one zero coefficient list."""
    return [mp.zero for _ in range(order + 1)]


def _seed_y_series(params: ProblemParameters, order: int) -> State[list[mp.mpf]]:
    """Build the y-series seed y0 + t y1 used at the singular endpoint."""
    y0 = y_zero_jet(params)
    y1 = y_first_jet(params)
    coeffs = []
    for value0, value1 in zip(y0, y1):
        component = _zero_coeffs(order)
        component[0] = value0
        if order >= 1:
            component[1] = value1
        coeffs.append(component)
    return State.from_iterable(coeffs)


def _check_seed(predicted: mp.mpf, seeded: mp.mpf, degree: int, index: int) -> None:
    """Ensure the stored zero/first jet is consistent with the q-system."""
    scale = max(mp.one, abs(predicted), abs(seeded))
    tolerance = mp.power(10, -(mp.dps // 2 + 10))
    if abs(predicted - seeded) <= tolerance * scale:
        return
    raise ValueError(f"Seed mismatch in q{index} coefficient t^{degree}: predicted {predicted}, stored {seeded}.")


def _coefficient_vector(series_state: State, degree: int) -> list[mp.mpf]:
    """Extract one coefficient level from a state of truncated series."""
    return [component.coeff(degree) for component in series_state]


def _solve_linear_system(matrix_rows: list[list[mp.mpf]], rhs: list[mp.mpf]) -> list[mp.mpf]:
    """Solve one dense 8x8 system using mpmath's LU solver."""
    matrix = mp.matrix(matrix_rows)
    vector = mp.matrix([[value] for value in rhs])
    solution = mp.lu_solve(matrix, vector)
    return [solution[row] for row in range(solution.rows)]


def _next_q_coefficients(
    q_coeffs: State[list[mp.mpf]],
    degree: int,
    order: int,
    params: ProblemParameters,
) -> list[mp.mpf]:
    """Solve for the next singular q coefficient vector c_(degree+1)."""
    t = Series.variable(order)
    base_state = state_to_series(q_coeffs)
    base_rhs = q_rhs(t, base_state, params)
    constant = _coefficient_vector(base_rhs, degree)
    columns = []
    for index in range(8):
        mutable = [component[:] for component in q_coeffs]
        mutable[index][degree + 1] += mp.one
        trial_rhs = q_rhs(t, state_to_series(State.from_iterable(mutable)), params)
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


def m_minus_one_residual(params: ProblemParameters) -> State[mp.mpf]:
    """Return the q-derived singular residual that depends only on y(0)."""
    order = 3
    y_series = State.from_iterable([[value] + [mp.zero] * order for value in y_zero_jet(params)])
    q_series = y_series_to_q_series(y_series, params, order)
    q_state = state_to_series(q_series)
    y_state = state_to_series(y_series)
    t = Series.variable(order)
    qdot = q_rhs(t, q_state, params)
    numerators = (
        qdot.y1 - 2 * t * y_state.y1,
        qdot.y2 - y_state.y2,
        qdot.y3 - y_state.y3,
        qdot.y4 - 2 * t * y_state.y4,
        qdot.y5 - 2 * t * y_state.y5,
        qdot.y6 - y_state.y6,
        qdot.y7 - y_state.y7,
        qdot.y8 - 2 * t * y_state.y8,
    )
    degrees = (1, 0, 0, 1, 1, 0, 0, 1)
    return State.from_iterable(series.coeff(degree) for series, degree in zip(numerators, degrees))


def series_residual(q_coeffs: State[list[mp.mpf]], params: ProblemParameters) -> State[list[mp.mpf]]:
    """Return the coefficient residual of q' - q_rhs(q) for one q-series."""
    order = len(q_coeffs.y1) - 1
    t = Series.variable(order)
    q_state = state_to_series(q_coeffs)
    residual = state_to_series(differentiate_coefficients(q_coeffs)) - q_rhs(t, q_state, params)
    return state_to_coefficients(residual)


def initial_q_series(params: ProblemParameters, config: SolverConfig) -> State[list[mp.mpf]]:
    """Build the singular-end q-series from the stored zero jet and alpha."""
    order = config.series_order
    q_coeffs = y_series_to_q_series(_seed_y_series(params, order), params, order)
    for degree in range(2, order):
        predicted_coeffs = _next_q_coefficients(q_coeffs, degree, order, params)
        for index, (component, predicted) in enumerate(zip(q_coeffs, predicted_coeffs), start=1):
            if degree + 1 <= SEED_DEGREES[index - 1]:
                _check_seed(predicted, component[degree + 1], degree + 1, index)
            else:
                component[degree + 1] = predicted
    return q_coeffs


def recovered_y_series(q_coeffs: State[list[mp.mpf]], params: ProblemParameters) -> State[list[mp.mpf]]:
    """Recover the y Taylor coefficients corresponding to one q-series."""
    order = len(q_coeffs.y1) - 1
    return q_series_to_y_series(q_coeffs, params, order)
