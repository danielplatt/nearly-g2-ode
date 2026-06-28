"""Tests for the S7 right-endpoint moduli readiness probe."""

from __future__ import annotations

from mpmath import mp

from experiments.s7 import right_endpoint_moduli_probe as probe


def test_round_probe_detects_fixed_seed_but_missing_generic_moduli() -> None:
    """Round S7 has a valid explicit p3 seed but no generic right family yet."""
    with mp.workdps(80):
        diagnostics = probe.target_diagnostics("round")
        assert diagnostics.right_chart == "s7_p3"
        assert diagnostics.collapse_defect < mp.mpf("1e-40")
        assert diagnostics.berger_form_defect > mp.mpf("0.1")
        assert diagnostics.explicit_seed_residual_norm < mp.mpf("1e-30")
        assert diagnostics.recurrence_seed_residual_norm > mp.mpf("0.1")
        assert diagnostics.global_solve_residual_norm < mp.mpf("1e-10")
        assert not diagnostics.search_ready


def test_squashed_probe_detects_fixed_seed_but_missing_generic_moduli() -> None:
    """Squashed S7 has a valid explicit p2 seed but no generic right family yet."""
    with mp.workdps(80):
        diagnostics = probe.target_diagnostics("squashed")
        assert diagnostics.right_chart == "s7_p2"
        assert diagnostics.collapse_defect < mp.mpf("1e-40")
        assert diagnostics.berger_form_defect > mp.mpf("0.1")
        assert diagnostics.explicit_seed_residual_norm < mp.mpf("1e-30")
        assert diagnostics.recurrence_seed_residual_norm > mp.mpf("0.1")
        assert diagnostics.global_solve_residual_norm < mp.mpf("1e-10")
        assert not diagnostics.search_ready
