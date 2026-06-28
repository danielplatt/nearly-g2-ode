"""Full two-sided matching residuals for non-mirrored endpoint data."""

from __future__ import annotations

from dataclasses import dataclass

from mpmath import mp

from problem import (
    DEFAULT_CONFIG,
    DEFAULT_PARAMS,
    LeftEndpointParameters,
    ProblemParameters,
    RightEndpointParameters,
    SolverConfig,
    State,
)
from .march import solve_two_sided


@dataclass(frozen=True)
class TwoSidedSearchPoint:
    """Scaled coordinates for independent left and right endpoint data."""

    u_left: mp.mpf
    v_left: mp.mpf
    r_left: mp.mpf
    u_right: mp.mpf
    v_right: mp.mpf
    r_right: mp.mpf
    s: mp.mpf


@dataclass(frozen=True)
class TwoSidedResidualResult:
    """One full two-sided matching residual evaluation."""

    point: TwoSidedSearchPoint
    params: ProblemParameters
    config: SolverConfig
    residual: tuple[mp.mpf, ...]
    residual_norm: mp.mpf
    left_q: State[mp.mpf] | None
    right_q: State[mp.mpf] | None
    left_l: mp.mpf | None
    right_l: mp.mpf | None
    patch_counts: tuple[int, int]
    branch_diagnostics: dict[str, mp.mpf]
    failure: str | None = None


@dataclass(frozen=True)
class TwoSidedJacobianResult:
    """Finite-difference Jacobian for the full two-sided residual."""

    point: TwoSidedSearchPoint
    step: mp.mpf
    matrix: mp.matrix
    singular_values: tuple[mp.mpf, ...]
    condition_number: mp.mpf


BASE_TWO_SIDED_POINT = TwoSidedSearchPoint(mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero)


def config_with_match_t(template: SolverConfig, match_t: mp.mpf) -> SolverConfig:
    """Copy one solver config with a new matching time."""
    return SolverConfig(
        series_order=template.series_order,
        working_dps=template.working_dps,
        target_dps=template.target_dps,
        step_safety=template.step_safety,
        sample_points=template.sample_points,
        match_t=match_t,
    )


def params_from_two_sided_scaled(
    point: TwoSidedSearchPoint,
    *,
    base_params: ProblemParameters = DEFAULT_PARAMS,
    template_config: SolverConfig = DEFAULT_CONFIG,
) -> tuple[ProblemParameters, SolverConfig]:
    """Convert scaled non-mirrored coordinates into parameters and config."""
    left = LeftEndpointParameters(
        a=base_params.left.a * mp.exp(point.u_left),
        c=base_params.left.c * mp.exp(point.v_left),
        alpha=base_params.left.alpha * (1 + point.r_left),
    )
    right = RightEndpointParameters(
        d=base_params.right.d * mp.exp(point.u_right),
        f=base_params.right.f * mp.exp(point.v_right),
        omega=base_params.right.omega * (1 + point.r_right),
    )
    match_t = template_config.match_t * mp.exp(point.s)
    params = ProblemParameters(
        lam=base_params.lam,
        interval_end=2 * match_t,
        left=left,
        right=right,
        right_chart=base_params.right_chart,
        fixed_right=base_params.fixed_right,
        left_mu=base_params.left_mu,
        right_mu=base_params.right_mu,
        p_signs=base_params.p_signs,
        right_p_signs=base_params.right_p_signs,
    )
    return params, config_with_match_t(template_config, match_t)


def point_with_delta(point: TwoSidedSearchPoint, index: int, delta: mp.mpf) -> TwoSidedSearchPoint:
    """Return one scaled point with one coordinate shifted."""
    values = [
        point.u_left,
        point.v_left,
        point.r_left,
        point.u_right,
        point.v_right,
        point.r_right,
        point.s,
    ]
    values[index] += delta
    return TwoSidedSearchPoint(*values)


def _diagnostics(result) -> dict[str, mp.mpf]:
    """Flatten left and right branch diagnostics from one march result."""
    diagnostics = {}
    for side_name, side in (("left", result.left), ("right", result.right)):
        for key in ("min_sum27", "min_sum36", "max_gap", "min_product"):
            diagnostics[f"{side_name}_{key}"] = side.diagnostics[key]
    return diagnostics


def _failure_result(
    point: TwoSidedSearchPoint,
    config: SolverConfig,
    params: ProblemParameters,
    message: str,
) -> TwoSidedResidualResult:
    """Build a nonfatal failed residual result."""
    return TwoSidedResidualResult(point, params, config, (), mp.inf, None, None, None, None, (0, 0), {}, message)


def two_sided_residual(
    point: TwoSidedSearchPoint,
    config: SolverConfig,
    *,
    base_params: ProblemParameters = DEFAULT_PARAMS,
) -> TwoSidedResidualResult:
    """Evaluate the full raw q mismatch between independent endpoints."""
    params, local_config = params_from_two_sided_scaled(point, base_params=base_params, template_config=config)
    try:
        result = solve_two_sided(params, local_config)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return _failure_result(point, local_config, params, str(exc))
    residual = tuple(result.mismatch_q)
    norm = max(abs(value) for value in residual)
    return TwoSidedResidualResult(
        point=point,
        params=params,
        config=local_config,
        residual=residual,
        residual_norm=norm,
        left_q=result.left.match_q,
        right_q=result.right.match_q,
        left_l=result.left_l,
        right_l=result.right_l,
        patch_counts=(len(result.left.patches), len(result.right.patches)),
        branch_diagnostics=_diagnostics(result),
    )


def finite_difference_two_sided_jacobian(
    point: TwoSidedSearchPoint,
    config: SolverConfig,
    step: mp.mpf,
    *,
    base_params: ProblemParameters = DEFAULT_PARAMS,
) -> TwoSidedJacobianResult:
    """Compute the centered finite-difference Jacobian of the 8-vector residual."""
    rows = [[mp.zero for _ in range(7)] for _ in range(8)]
    for col in range(7):
        plus = two_sided_residual(point_with_delta(point, col, step), config, base_params=base_params)
        minus = two_sided_residual(point_with_delta(point, col, -step), config, base_params=base_params)
        if plus.failure or minus.failure:
            raise ValueError(f"Cannot difference failed residuals in column {col}.")
        for row, (left, right) in enumerate(zip(plus.residual, minus.residual)):
            rows[row][col] = (left - right) / (2 * step)
    matrix = mp.matrix(rows)
    _, singulars, _ = mp.svd(matrix)
    singular_values = tuple(singulars)
    condition = singular_values[0] / singular_values[-1] if singular_values[-1] != 0 else mp.inf
    return TwoSidedJacobianResult(point, step, matrix, singular_values, condition)


def matrix_max_difference(left: mp.matrix, right: mp.matrix) -> mp.mpf:
    """Return the max-norm difference between two same-shaped matrices."""
    return max(abs(left[row, col] - right[row, col]) for row in range(left.rows) for col in range(left.cols))
