"""Report the mirror-closing Jacobian at the Berger solution."""

from __future__ import annotations

from mpmath import mp

from problem import DEFAULT_CONFIG, SolverConfig
from solver.mirror_shooting import BASE_POINT, finite_difference_jacobian, matrix_max_difference, mirror_residual


STEPS = (mp.mpf("1e-4"), mp.mpf("3e-5"), mp.mpf("1e-5"))
JACOBIAN_CONFIG = SolverConfig(10, 70, 30, DEFAULT_CONFIG.step_safety, 3, DEFAULT_CONFIG.match_t)


def _format_vector(values) -> str:
    """Format one short vector of high-precision scalars."""
    return "(" + ", ".join(mp.nstr(value, 14) for value in values) + ")"


def _format_matrix(matrix: mp.matrix) -> str:
    """Format one small matrix row by row."""
    rows = []
    for row in range(matrix.rows):
        rows.append("  " + _format_vector(matrix[row, col] for col in range(matrix.cols)))
    return "\n".join(rows)


def _regularity_label(result, previous) -> str:
    """Return a compact numerical regularity conclusion."""
    stable = previous is None or matrix_max_difference(result.matrix, previous.matrix) < mp.mpf("1e-5")
    nonsingular = result.singular_values[-1] > mp.mpf("1e-6")
    return "regular" if stable and nonsingular else "numerically inconclusive"


def main() -> None:
    """Compute finite-difference Jacobians at several step sizes."""
    with mp.workdps(JACOBIAN_CONFIG.working_dps):
        base = mirror_residual(BASE_POINT, JACOBIAN_CONFIG)
        print(
            "config: "
            f"order={JACOBIAN_CONFIG.series_order}, dps={JACOBIAN_CONFIG.working_dps}, "
            f"match_t={JACOBIAN_CONFIG.match_t}",
            flush=True,
        )
        print(f"base residual norm: {mp.nstr(base.residual_norm, 16)}", flush=True)
        print(f"base l(match_t): {mp.nstr(base.l_value, 16)}", flush=True)
        previous = None
        for step in STEPS:
            result = finite_difference_jacobian(BASE_POINT, JACOBIAN_CONFIG, step)
            print(f"\nh = {step}", flush=True)
            print("Jacobian:", flush=True)
            print(_format_matrix(result.matrix), flush=True)
            print(f"singular values: {_format_vector(result.singular_values)}", flush=True)
            print(f"determinant: {mp.nstr(result.determinant, 16)}", flush=True)
            print(f"condition number: {mp.nstr(result.condition_number, 16)}", flush=True)
            if previous is not None:
                diff = matrix_max_difference(result.matrix, previous.matrix)
                print(f"max change from previous h: {mp.nstr(diff, 16)}", flush=True)
            print(f"conclusion: {_regularity_label(result, previous)}", flush=True)
            previous = result


if __name__ == "__main__":
    main()
