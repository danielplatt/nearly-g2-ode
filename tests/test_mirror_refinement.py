"""Tests for damped Newton mirror-refinement helpers."""

from __future__ import annotations

from mpmath import mp

from problem import DEFAULT_CONFIG, DEFAULT_PARAMS, SolverConfig
from solver import mirror_refinement
from solver.mirror_refinement import NewtonSettings, newton_refine
from solver.mirror_shooting import BASE_POINT, MirrorResidualResult, MirrorSearchPoint


SMOKE_CONFIG = SolverConfig(4, 30, 15, mp.mpf("0.95"), 0, DEFAULT_CONFIG.match_t)


def test_newton_settings_can_run_zero_steps_without_mutating_point() -> None:
    """A zero-step stage should only evaluate the starting residual."""
    settings = NewtonSettings("zero", SMOKE_CONFIG, mp.mpf("1e-3"), mp.zero, 0)
    with mp.workdps(SMOKE_CONFIG.working_dps):
        report = newton_refine(BASE_POINT, settings)
        assert report.initial.point == BASE_POINT
        assert report.final.point == BASE_POINT
        assert report.steps == ()
        assert report.status in {"max_steps", "tolerance_hit"}


def test_one_newton_step_improves_or_reports_clean_stop() -> None:
    """A smoke Newton stage should either improve or report a nonfatal stop."""
    point = MirrorSearchPoint(mp.mpf("0.01"), mp.mpf("0.002"), mp.mpf("0.02"), mp.mpf("-0.003"))
    original = point
    settings = NewtonSettings("one", SMOKE_CONFIG, mp.mpf("1e-3"), mp.mpf("1e-12"), 1)
    with mp.workdps(SMOKE_CONFIG.working_dps):
        report = newton_refine(point, settings)
        assert point == original
        if report.status == "max_steps":
            assert report.final.residual_norm <= report.initial.residual_norm
        else:
            assert report.status in {"branch_failure", "jacobian_failure", "no_improvement", "tolerance_hit"}


def test_coordinate_guard_rejects_out_of_bounds_trial(monkeypatch) -> None:
    """Out-of-bounds damped Newton trials should not call the residual evaluator."""
    called = False

    def fake_residual(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("out-of-bounds trial should not be evaluated")

    monkeypatch.setattr(mirror_refinement, "mirror_residual", fake_residual)
    result = MirrorResidualResult(BASE_POINT, DEFAULT_PARAMS, SMOKE_CONFIG, (), mp.one, None, None, 0, {})
    settings = NewtonSettings("bounded", SMOKE_CONFIG, mp.mpf("1e-3"), mp.zero, 1, (mp.one,), mp.mpf("0.5"))
    trial, damping, trials = mirror_refinement._try_dampings(result, (mp.one, mp.zero, mp.zero, mp.zero), settings, DEFAULT_PARAMS)
    assert trial is None
    assert damping is None
    assert trials == ((mp.one, mp.inf, True, "coordinate_bound"),)
    assert called is False


def test_midpoint_floor_guard_rejects_low_s_trial(monkeypatch) -> None:
    """The midpoint floor should reject Newton trials before residual evaluation."""
    called = False

    def fake_residual(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("m-floor trial should not be evaluated")

    monkeypatch.setattr(mirror_refinement, "mirror_residual", fake_residual)
    result = MirrorResidualResult(BASE_POINT, DEFAULT_PARAMS, SMOKE_CONFIG, (), mp.one, None, None, 0, {})
    settings = NewtonSettings("bounded", SMOKE_CONFIG, mp.mpf("1e-3"), mp.zero, 1, (mp.one,), min_s_coordinate=mp.mpf("-0.5"))
    trial, damping, trials = mirror_refinement._try_dampings(result, (mp.zero, mp.zero, mp.zero, mp.mpf("-1")), settings, DEFAULT_PARAMS)
    assert trial is None
    assert damping is None
    assert trials == ((mp.one, mp.inf, True, "m_floor_rejected"),)
    assert called is False
