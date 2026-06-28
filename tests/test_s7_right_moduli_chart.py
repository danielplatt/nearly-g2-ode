"""Tests for the derived S7 p2/p3 right-end offset moduli charts."""

from __future__ import annotations

from mpmath import mp

from experiments.s7.right_moduli_chart import (
    branch_sums,
    leading_core_residual,
    p2_offset,
    p2_offset_defect,
    p3_offset,
    p3_offset_defect,
)
from problem import (
    S7_P2_RIGHT_CHART,
    S7_P3_RIGHT_CHART,
    SolverConfig,
    initial_right_series,
    round_s7_candidate_parameters,
    squashed_s7_parameters,
)
from problem.taylor_seed import weighted_series_residual


def _max_abs(values) -> mp.mpf:
    """Return max absolute value in an iterable."""
    return max(abs(value) for value in values)


def test_p3_offset_family_has_collapsed_sums_and_zero_leading_cores() -> None:
    """The derived p3 family should satisfy the leading regularity equations."""
    with mp.workdps(80):
        offset = p3_offset(mp.mpf("0.07"), mp.mpf("0.19"), mp.mpf("0.83"))
        sum27, sum36, gap = branch_sums(offset)
        cores = leading_core_residual(offset)

        assert sum27 == mp.mpf("0.76")
        assert sum36 == 0
        assert gap == 0
        assert p3_offset_defect(offset) == 0
        assert _max_abs(cores) < mp.mpf("1e-70")


def test_p2_offset_family_has_collapsed_sums_and_zero_leading_cores() -> None:
    """The derived p2 family should satisfy the leading regularity equations."""
    with mp.workdps(80):
        offset = p2_offset(mp.mpf("0.07"), mp.mpf("0.19"), mp.mpf("0.83"))
        sum27, sum36, gap = branch_sums(offset)
        cores = leading_core_residual(offset)

        assert sum27 == 0
        assert sum36 == mp.mpf("0.76")
        assert gap == 0
        assert p2_offset_defect(offset) == 0
        assert _max_abs(cores) < mp.mpf("1e-70")


def test_known_s7_offsets_are_special_points_of_derived_families() -> None:
    """Round and squashed S7 should be the same offset parameters in p3/p2."""
    with mp.workdps(80):
        scale = mp.sqrt(5) / 25
        round_params = round_s7_candidate_parameters()
        squashed_params = squashed_s7_parameters()

        assert round_params.fixed_right is not None
        assert squashed_params.fixed_right is not None
        assert round_params.fixed_right.offset == p3_offset(scale, 2 * scale, 19 * scale)
        assert squashed_params.fixed_right.offset == p2_offset(scale, 2 * scale, 19 * scale)


def test_known_s7_series_still_solve_weighted_equations_in_derived_charts() -> None:
    """The known exact right series should remain valid at the special points."""
    with mp.workdps(100):
        config = SolverConfig(10, 80, 30, mp.mpf("0.5"), 0, mp.pi / 6)
        for params, chart in (
            (round_s7_candidate_parameters(), S7_P3_RIGHT_CHART),
            (squashed_s7_parameters(), S7_P2_RIGHT_CHART),
        ):
            coeffs = initial_right_series(params, config)
            residual = weighted_series_residual(chart, coeffs, mp.zero, params)
            assert _max_abs(value for component in residual for value in component[:7]) < mp.mpf("1e-60")


def test_fixed_offset_firstjet_chart_misses_terminal_moduli() -> None:
    """A genuine right-moduli perturbation changes the terminal offset itself."""
    with mp.workdps(80):
        scale = mp.sqrt(5) / 25
        base = p3_offset(scale, 2 * scale, 19 * scale)
        varied = p3_offset(scale * mp.mpf("1.01"), 2 * scale, 19 * scale)

        assert p3_offset_defect(varied) == 0
        assert _max_abs(leading_core_residual(varied)) < mp.mpf("1e-70")
        assert _max_abs(left - right for left, right in zip(base, varied)) > 0
