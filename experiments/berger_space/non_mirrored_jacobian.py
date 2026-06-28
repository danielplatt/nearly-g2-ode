"""Report the full non-mirrored two-sided Jacobian at Berger."""

from __future__ import annotations

from mpmath import mp

from problem import REFINED_CONFIG, SolverConfig
from solver.two_sided_shooting import (
    BASE_TWO_SIDED_POINT,
    finite_difference_two_sided_jacobian,
    two_sided_residual,
)


BASE_VERIFY_CONFIG = SolverConfig(14, 90, 35, mp.mpf("0.6"), 1, REFINED_CONFIG.match_t)
JACOBIAN_CONFIG = SolverConfig(6, 50, 20, mp.mpf("0.75"), 1, REFINED_CONFIG.match_t)
JACOBIAN_STEPS = (mp.mpf("1e-4"), mp.mpf("3e-5"))


def _format_matrix(matrix) -> str:
    """Format one matrix row-by-row for terminal diagnostics."""
    rows = []
    for row in range(matrix.rows):
        rows.append("[" + ", ".join(mp.nstr(matrix[row, col], 12) for col in range(matrix.cols)) + "]")
    return "\n".join(rows)


def main() -> None:
    """Print Berger residual and finite-difference Jacobian diagnostics."""
    mp.dps = BASE_VERIFY_CONFIG.working_dps
    base = two_sided_residual(BASE_TWO_SIDED_POINT, BASE_VERIFY_CONFIG)
    print("Non-mirrored Berger Jacobian", flush=True)
    print(f"base verification residual norm: {mp.nstr(base.residual_norm, 16)}", flush=True)
    print(f"base failure: {base.failure}", flush=True)
    print(f"left l: {mp.nstr(base.left_l, 16)}", flush=True)
    print(f"right l: {mp.nstr(base.right_l, 16)}", flush=True)
    mp.dps = JACOBIAN_CONFIG.working_dps
    for step in JACOBIAN_STEPS:
        jacobian = finite_difference_two_sided_jacobian(BASE_TWO_SIDED_POINT, JACOBIAN_CONFIG, step)
        rank = sum(1 for value in jacobian.singular_values if value > mp.mpf("1e-8"))
        print(f"\nh = {mp.nstr(step, 8)}", flush=True)
        print(_format_matrix(jacobian.matrix), flush=True)
        print("singular values:", [mp.nstr(value, 12) for value in jacobian.singular_values], flush=True)
        print(f"condition number: {mp.nstr(jacobian.condition_number, 12)}", flush=True)
        print(f"numerical rank (>1e-8): {rank}", flush=True)


if __name__ == "__main__":
    main()
