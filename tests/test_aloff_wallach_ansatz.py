"""Tests for the Aloff-Wallach N11 ansatz verification."""

from __future__ import annotations

from mpmath import mp

import experiments.aloff_wallach_ansatz
from experiments.aloff_wallach import ansatz


def test_n11_bracket_basis_is_orthonormal_enough() -> None:
    """The SU3 matrix basis should be orthonormal for -tr(XY)."""
    matrices = ansatz.n11_su3_basis_matrices()
    for i, left in enumerate(matrices):
        for j, right in enumerate(matrices):
            value = ansatz._inner(left, right)
            expected = 1.0 if i == j else 0.0
            assert abs(value - expected) < 1e-12


def test_exterior_derivative_matches_ball_oliveira_formula() -> None:
    """The computed dphi should agree with the published closed expression."""
    with mp.workdps(80):
        residual = ansatz.ball_formula_residual(mp.mpf("1.3"), mp.mpf("0.9"), mp.mpf("1.1"), mp.mpf("-0.7"))
        assert ansatz.max_abs_coefficient(residual) < mp.mpf("1e-12")


def test_known_n11_nearly_parallel_points_verify() -> None:
    """Tri-Sasakian and squashed N11 calibration points should satisfy dphi=lambda psi."""
    with mp.workdps(80):
        residuals = {
            solution.label: ansatz.max_abs_coefficient(ansatz.nearly_parallel_residual(solution))
            for solution in ansatz.n11_known_solutions()
        }
        assert residuals["tri_sasakian"] < mp.mpf("1e-12")
        assert residuals["squashed"] < mp.mpf("1e-12")


def test_n11_summary_records_ansatz_limits() -> None:
    """The summary should distinguish verified calibrations from full scout readiness."""
    summary = ansatz.build_summary()
    assert summary["version"] == "aloff-wallach-n11-ansatz-v4"
    assert summary["ansatz"]["action"] == "SO(3)_real x SO(3)_fiber"
    assert "SO(3) x SO(3)" in summary["ansatz"]["principal_orbit"]
    assert summary["ansatz"]["principal_orbit_invariant_dimensions"] == {
        "one_forms": 2,
        "two_forms": 7,
        "three_forms": 12,
    }
    assert summary["ansatz"]["principal_orbit_model_coframe"]["fiber_2"] == "omega_5"
    assert summary["ansatz"]["principal_orbit_model_coframe"]["fiber_3"] == "omega_4"
    assert len(summary["known_solutions"]) == 2
    assert summary["known_solutions"][0]["label"] == "tri_sasakian"
    assert summary["known_solutions"][1]["label"] == "squashed"
    assert summary["extra_sasaki_einstein_solution"]["label"] == "sasaki_einstein_phi_ts"
    assert not summary["extra_sasaki_einstein_solution"]["in_abcd_family"]
    assert summary["extra_sasaki_einstein_solution"]["current_action_invariant"]
    assert not summary["extra_sasaki_einstein_solution"]["needs_alternative_cohomogeneity_one_action"]
    assert summary["principal_orbit_su3_variables"]["raw_coefficient_count"] == 19
    assert summary["principal_orbit_su3_variables"]["expected_su3_dimension"] == 16
    assert len(summary["principal_orbit_su3_variables"]["algebraic_constraints"]) == 3
    assert len(summary["sasaki_einstein_model_checks"]) == 4
    assert all(check["tested_in_principal_orbit_coframe"] for check in summary["sasaki_einstein_model_checks"])
    assert "principal-orbit SU(3) variables are now explicit" in summary["verdict"]
    assert experiments.aloff_wallach_ansatz.main is ansatz.main


def test_extra_sasaki_einstein_is_not_a_d_sign_flip() -> None:
    """The extra form should not be confused with D -> -D in A,B,C,D."""
    with mp.workdps(80):
        probe = ansatz.abcd_vertical_flip_probe()
        assert mp.mpf(probe["reference_best_residual"]) < mp.mpf("1e-12")
        assert mp.mpf(probe["flipped_best_residual"]) > mp.mpf("1")
        extra = ansatz.n11_extra_sasaki_einstein_solution()
        assert extra.current_action_invariant
        assert not extra.in_abcd_family


def test_principal_orbit_z2_invariant_basis_dimensions() -> None:
    """The future omega/gamma ansatz lives in the Z2-even form spaces."""
    signs = ansatz.principal_orbit_isotropy_signs()
    invariant = ansatz.principal_orbit_invariant_form_basis()
    assert tuple(signs.values()) == (-1, -1, 1, -1, -1, 1)
    assert invariant[1] == ["base_3", "fiber_3"]
    assert len(invariant[2]) == 7
    assert len(invariant[3]) == 12


def test_principal_orbit_su3_variable_basis_records_equations() -> None:
    """The full invariant SU3 variables should include basis and constraints."""
    variables = ansatz.principal_orbit_su3_variable_basis()
    assert [item["basis"] for item in variables["omega_variables"]] == [
        "base_1^base_2",
        "base_1^fiber_1",
        "base_1^fiber_2",
        "base_2^fiber_1",
        "base_2^fiber_2",
        "base_3^fiber_3",
        "fiber_1^fiber_2",
    ]
    assert len(variables["gamma_variables"]) == 12
    equations = [constraint["equation"] for constraint in variables["algebraic_constraints"]]
    assert "x1*y11 + x2*y8 - x3*y7 - x4*y4 + x5*y3 + x7*y1 = 0" in equations
    assert "x1*y12 - x2*y10 + x3*y9 + x4*y6 - x5*y5 + x7*y2 = 0" in equations


def test_known_abcd_points_restrict_to_principal_orbit_su3_structures() -> None:
    """The two A,B,C,D known forms should produce valid model-orbit SU3 pairs."""
    with mp.workdps(80):
        checks = {check.label: check for check in ansatz.known_solution_principal_su3_checks()}
        assert set(checks) == {"tri_sasakian", "squashed"}
        for check in checks.values():
            assert check.omega_wedge_gamma_residual < mp.mpf("1e-12")
            assert check.hitchin_complex_residual < mp.mpf("1e-12")
            assert check.volume_normalization_residual < mp.mpf("1e-12")
            assert check.stable_negative
            assert len(check.omega_coefficients) == 3
            assert len(check.gamma_coefficients) == 4
        assert checks["tri_sasakian"].orientation_sign == 1
        assert checks["squashed"].orientation_sign == -1


def test_sasaki_einstein_ball_coframe_nearly_parallel_check() -> None:
    """The extra Sasaki-Einstein Ball-coframe G2 forms should satisfy dphi=4psi."""
    with mp.workdps(80):
        scales = ansatz.ball_sasaki_einstein_metric_scales()
        assert scales == (
            1 / mp.sqrt(2),
            mp.mpf("0.5"),
            mp.mpf("0.5"),
            1 / mp.sqrt(2),
            1 / mp.sqrt(2),
            mp.mpf("0.5"),
            mp.mpf("0.5"),
        )
        representative = ansatz.ball_sasaki_einstein_phi("real", 1)
        assert abs(representative[(1, 2, 3)] + 1 / (4 * mp.sqrt(2))) < mp.mpf("1e-70")
        assert abs(representative[(1, 4, 5)] - 1 / (2 * mp.sqrt(2))) < mp.mpf("1e-70")
        assert abs(representative[(1, 6, 7)] - 1 / (4 * mp.sqrt(2))) < mp.mpf("1e-70")
        for phase in ("real", "imag"):
            for gamma_sign in (1, -1):
                check = ansatz.geipel_sasaki_einstein_check(phase, gamma_sign)
                assert mp.mpf(check["max_abs_dphi_minus_4psi"]) < mp.mpf("1e-12")
                assert mp.mpf(check["max_abs_e8_terms_in_dphi"]) < mp.mpf("1e-12")
                assert mp.mpf(check["su3_omega_wedge_gamma_residual"]) < mp.mpf("1e-12")
                assert mp.mpf(check["su3_hitchin_complex_residual"]) < mp.mpf("1e-12")
                assert mp.mpf(check["su3_volume_normalization_residual"]) < mp.mpf("1e-12")
                assert check["tested_in_principal_orbit_coframe"] is True


def test_sasaki_einstein_principal_orbit_su3_coefficients_are_invariant() -> None:
    """The extra Sasaki-Einstein structures should land in the 19-variable chart."""
    with mp.workdps(80):
        for phase in ("real", "imag"):
            for gamma_sign in (1, -1):
                check = ansatz.ball_sasaki_einstein_principal_su3_check(phase, gamma_sign)
                assert check.omega_wedge_gamma_residual < mp.mpf("1e-12")
                assert check.hitchin_complex_residual < mp.mpf("1e-12")
                assert check.volume_normalization_residual < mp.mpf("1e-12")
                assert check.stable_negative
                assert len(check.omega_coefficients) == 3
                assert len(check.gamma_coefficients) == 4
