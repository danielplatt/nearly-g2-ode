"""One-sided mirror-closing residuals and finite-difference Jacobians."""

from __future__ import annotations

from dataclasses import dataclass

from mpmath import mp

from problem import DEFAULT_CONFIG, DEFAULT_PARAMS, ProblemParameters, SolverConfig, State, mirrored_problem_parameters
from problem.q_system import mean_curvature
from .march import SideResult, solve_left_side


@dataclass(frozen=True)
class MirrorSearchPoint:
    """Scaled coordinates around the homogeneous Berger solution."""

    u: mp.mpf
    v: mp.mpf
    r: mp.mpf
    s: mp.mpf


@dataclass(frozen=True)
class MirrorResidualResult:
    """One mirror-closing residual evaluation."""

    point: MirrorSearchPoint
    params: ProblemParameters
    config: SolverConfig
    residual: tuple[mp.mpf, ...]
    residual_norm: mp.mpf
    match_q: State[mp.mpf] | None
    l_value: mp.mpf | None
    patch_count: int
    branch_diagnostics: dict[str, mp.mpf]
    failure: str | None = None


@dataclass(frozen=True)
class JacobianResult:
    """Finite-difference Jacobian and regularity diagnostics."""

    point: MirrorSearchPoint
    step: mp.mpf
    matrix: mp.matrix
    singular_values: tuple[mp.mpf, ...]
    determinant: mp.mpf
    condition_number: mp.mpf


BASE_POINT = MirrorSearchPoint(mp.zero, mp.zero, mp.zero, mp.zero)


def config_with_match_t(template: SolverConfig, match_t: mp.mpf) -> SolverConfig:
    """Copy one solver config with a new midpoint time."""
    return SolverConfig(
        series_order=template.series_order,
        working_dps=template.working_dps,
        target_dps=template.target_dps,
        step_safety=template.step_safety,
        sample_points=template.sample_points,
        match_t=match_t,
    )


def params_from_scaled(
    point: MirrorSearchPoint,
    *,
    base_params: ProblemParameters = DEFAULT_PARAMS,
    template_config: SolverConfig = DEFAULT_CONFIG,
) -> tuple[ProblemParameters, SolverConfig]:
    """Convert scaled search coordinates into mirrored parameters and config."""
    a = base_params.left.a * mp.exp(point.u)
    c = base_params.left.c * mp.exp(point.v)
    alpha = base_params.left.alpha * (1 + point.r)
    match_t = template_config.match_t * mp.exp(point.s)
    params = mirrored_problem_parameters(
        a,
        c,
        alpha,
        base_params.lam,
        2 * match_t,
        left_mu=base_params.left_mu,
        right_mu=base_params.right_mu,
        p_signs=base_params.p_signs,
        right_p_signs=base_params.right_p_signs,
    )
    return params, config_with_match_t(template_config, match_t)


def _closing_residual(q: State[mp.mpf]) -> tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    """Return q - mirror(q) on the four independent fixed-point equations."""
    return (q.y1 - q.y8, q.y2 + q.y4, q.y3 - q.y6, q.y5 + q.y7)


def _branch_diagnostics(side: SideResult) -> dict[str, mp.mpf]:
    """Extract branch extrema from one left-side march result."""
    return {
        "min_sum27": side.diagnostics["min_sum27"],
        "min_sum36": side.diagnostics["min_sum36"],
        "max_gap": side.diagnostics["max_gap"],
        "min_product": side.diagnostics["min_product"],
    }


def mirror_residual(
    point: MirrorSearchPoint,
    config: SolverConfig,
    *,
    base_params: ProblemParameters = DEFAULT_PARAMS,
) -> MirrorResidualResult:
    """Evaluate the one-sided mirror-closing residual."""
    params, local_config = params_from_scaled(point, base_params=base_params, template_config=config)
    try:
        side = solve_left_side(params, local_config)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return MirrorResidualResult(point, params, local_config, (), mp.inf, None, None, 0, {}, str(exc))
    residual = _closing_residual(side.match_q)
    norm = max(abs(value) for value in residual)
    return MirrorResidualResult(
        point=point,
        params=params,
        config=local_config,
        residual=residual,
        residual_norm=norm,
        match_q=side.match_q,
        l_value=mean_curvature(side.match_q, side.match_qdot),
        patch_count=len(side.patches),
        branch_diagnostics=_branch_diagnostics(side),
    )


def point_with_delta(point: MirrorSearchPoint, index: int, delta: mp.mpf) -> MirrorSearchPoint:
    """Return one scaled point with one coordinate shifted."""
    values = [point.u, point.v, point.r, point.s]
    values[index] += delta
    return MirrorSearchPoint(*values)


def finite_difference_jacobian(
    point: MirrorSearchPoint,
    config: SolverConfig,
    step: mp.mpf,
    *,
    base_params: ProblemParameters = DEFAULT_PARAMS,
) -> JacobianResult:
    """Compute the centered finite-difference Jacobian of the mirror residual."""
    rows = [[mp.zero for _ in range(4)] for _ in range(4)]
    for col in range(4):
        plus = mirror_residual(point_with_delta(point, col, step), config, base_params=base_params)
        minus = mirror_residual(point_with_delta(point, col, -step), config, base_params=base_params)
        if plus.failure or minus.failure:
            raise ValueError(f"Cannot difference failed residuals in column {col}.")
        for row, (left, right) in enumerate(zip(plus.residual, minus.residual)):
            rows[row][col] = (left - right) / (2 * step)
    matrix = mp.matrix(rows)
    _, singulars, _ = mp.svd(matrix)
    singular_values = tuple(singulars)
    condition = singular_values[0] / singular_values[-1] if singular_values[-1] != 0 else mp.inf
    return JacobianResult(point, step, matrix, singular_values, mp.det(matrix), condition)


def matrix_max_difference(left: mp.matrix, right: mp.matrix) -> mp.mpf:
    """Return the max-norm difference between two same-shaped matrices."""
    return max(abs(left[row, col] - right[row, col]) for row in range(left.rows) for col in range(left.cols))
