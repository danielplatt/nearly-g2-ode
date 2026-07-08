"""Tests for the S7 SU(2)^3 tail-defect diagnostics."""

from __future__ import annotations

import json

import pytest

import experiments.s7_su2_cubed_tail_defect
from experiments.s7 import su2_cubed_tail_defect as tail


def test_limiting_crossing_has_positive_x2_tail_defect() -> None:
    """The large-|a| limiting IVP should miss the right closure."""
    crossing = tail.first_scaled_crossing("limit")
    assert crossing.status == "crossed"
    assert 3.55 < crossing.time < 3.65
    assert crossing.x[2] > 0.004
    assert crossing.x[3] < -0.8


def test_exact_tail_samples_have_nonzero_x2_defect() -> None:
    """Conservative finite tails should already have X2 bounded away from zero."""
    for a in (-250.0, 250.0, -500.0, 500.0):
        crossing = tail.first_scaled_crossing("exact", a)
        assert crossing.status == "crossed"
        assert crossing.x[2] > 0.004


def test_finite_a_perturbation_polynomial_reconstructs_exact_rhs() -> None:
    """The explicit b=1/a perturbation formula should match the exact scaled RHS."""
    state = tail.limiting_state_at(2.5)
    for a in (-250.0, 250.0):
        direct = tail.exact_scaled_rhs(2.5, state, a)
        reconstructed = tail.perturbation_rhs_from_coefficients(2.5, state, a)
        assert max(abs(left - right) for left, right in zip(direct, reconstructed)) < 1e-12


def test_scaled_taylor_c2_matches_regular_singular_equations() -> None:
    """The c2 seed should make x' - F vanish to cubic order at the singular end."""
    for b in (-1e-8, 0.0, 1e-8):
        c2 = tail.scaled_taylor_c2(b)
        time = 1e-3
        state = tail.scaled_taylor_seed(time, b)
        derivative = tuple(2.0 * c2[index] * time for index in range(4))
        rhs = tail.scaled_rhs_with_b(time, state, b)
        residual = max(abs(left - right) for left, right in zip(derivative, rhs))
        assert residual < 1e-8


def test_high_order_taylor_coefficients_extend_c2_seed() -> None:
    """The recursive singular Taylor germ should reproduce the known c2 term."""
    for b in (-1e-8, 0.0, 1e-8):
        coefficients = tail.scaled_taylor_coefficients(6, b, working_dps=50)
        expected = tail.scaled_taylor_c2(b)
        assert max(abs(float(coefficients[index][2]) - expected[index]) for index in range(4)) < 1e-16
        assert max(abs(float(coefficients[index][1])) for index in range(4)) < 1e-30


def test_high_order_taylor_seed_matches_taylor_seeded_rk4() -> None:
    """A moderate Taylor germ should agree with the same IVP marched by RK4."""
    target_time = 1.0
    state = tail.scaled_taylor_seed(1e-3, 0.0)
    time = 1e-3
    while time < target_time:
        step = min(1e-4, target_time - time)
        state = tail._rk4_step_b(time, state, step, 0.0)
        time += step

    seed = tail.high_order_scaled_taylor_seed(target_time, 0.0, order=14, working_dps=60)
    assert max(abs(left - right) for left, right in zip(seed, state)) < 1e-8


def test_high_order_taylor_state_at_p_matches_taylor_seeded_rk4() -> None:
    """The high-order germ should locate fixed-p slices reproducibly."""
    target_p = 0.8
    state = tail.scaled_taylor_seed(1e-3, 0.0)
    time = 1e-3
    step = 1e-4
    while True:
        next_state = tail._rk4_step_b(time, state, step, 0.0)
        if next_state[0] <= target_p:
            alpha = (state[0] - target_p) / (state[0] - next_state[0])
            reference = (time + alpha * step,) + tuple(
                state[index] + alpha * (next_state[index] - state[index])
                for index in range(1, 4)
            )
            break
        state = next_state
        time += step

    seed = tail.high_order_scaled_taylor_state_at_p(
        target_p,
        0.0,
        order=20,
        working_dps=60,
        time_high=2.2,
    )
    assert max(abs(left - right) for left, right in zip(seed, reference)) < 1e-5


def test_report_and_top_level_shim() -> None:
    """The diagnostic report should be import-safe and JSON-friendly."""
    report = tail.build_report((-250.0, 250.0))
    assert report["version"] == tail.TAIL_DEFECT_VERSION
    assert report["limiting_crossing"]["x2_tail_defect"] > 0.004
    assert report["sign_support"]["x2_boundary_derivative_at_support"] > 0
    assert report["sign_support"]["x3_minus_0_3_boundary_derivative_at_support"] < 0
    assert report["sign_support"]["riccati"]["riccati_lower_bound_numeric"] > 0.004
    barrier = report["terminal_barrier"]
    assert barrier["limit"]["x3"] < -0.7
    assert barrier["limit"]["x1_margin"] > 5.0
    assert barrier["limit"]["x3_zero_boundary_derivative"] < -1.0
    for item in barrier["finite_candidate_A"]:
        assert item["x3"] < -0.7
        assert item["x1_margin"] > 5.0
        assert item["x3_zero_boundary_derivative"] < -1.0
    assert report["finite_a_perturbation"]["candidate_A"] == tail.DEFAULT_CANDIDATE_A
    assert len(report["proof_strategy"]) == 6
    assert experiments.s7_su2_cubed_tail_defect.main is tail.main


def test_moving_tube_certificate_certifies_short_conditional_slab() -> None:
    """The interval face checker should certify a small conditional tube."""
    certificate = tail.moving_tube_certificate(
        start_time=3.5,
        end_time=3.5001,
        step_size=1e-4,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        radius0=(1e-7, 1e-6, 1e-8, 1e-7),
        radius_growth=(0.02, 0.2, 0.002, 0.02),
        seed_step_size=1e-3,
    )
    assert certificate["status"] == "certified"
    assert certificate["worst_margin"] > 0.0
    assert certificate["conditional"] == "start_box_contains_true_state"
    assert certificate["end_box"]["low"][0] < certificate["end_box"]["high"][0]


def test_moving_tube_certificate_records_subdivision_settings() -> None:
    """Subdivided interval checks should be reproducible in the payload."""
    certificate = tail.moving_tube_certificate(
        start_time=3.5,
        end_time=3.5001,
        step_size=1e-4,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        subdivisions=(2, 1, 1, 2),
        time_subdivisions=2,
        seed_step_size=1e-3,
    )
    assert certificate["status"] == "certified"
    assert certificate["subdivisions"] == [2, 1, 1, 2]
    assert certificate["time_subdivisions"] == 2


def test_segmented_moving_tube_certificate_certifies_short_chain() -> None:
    """Segmented tubes should carry boxes across multiple short blocks."""
    certificate = tail.segmented_moving_tube_certificate(
        start_time=3.5,
        end_time=3.501,
        step_size=1e-4,
        block_steps=5,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        seed_step_size=1e-3,
    )
    assert certificate["status"] == "certified"
    assert certificate["certified_until"] == 3.501
    assert certificate["blocks_certified"] == 2
    assert certificate["conditional"] == "initial_start_box_contains_true_state"


def test_tuned_segmented_moving_tube_certificate_certifies_short_chain() -> None:
    """The tuned centered tube should certify short regular-time chains."""
    certificate = tail.tuned_segmented_moving_tube_certificate(
        start_time=2.0,
        end_time=2.002,
        step_size=0.001,
        block_steps=1,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
    )
    assert certificate["status"] == "certified"
    assert certificate["certified_until"] == 2.002
    assert certificate["blocks_certified"] == 2
    assert certificate["tuning_attempt_count"] >= 2
    assert certificate["worst_margin"] > 0.0


def test_restart_tuned_time_chain_certifies_short_restart() -> None:
    """Restart boxes should explicitly contain carried boxes between segments."""
    certificate = tail.restart_tuned_time_chain_certificate(
        start_time=2.0,
        end_time=2.004,
        restart_interval=0.002,
        step_size=0.001,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
    )
    assert certificate["status"] == "certified"
    assert certificate["segments_certified"] == 2
    assert len(certificate["restarts"]) == 1
    assert certificate["restarts"][0]["source_box_contained"] is True
    assert certificate["restarts"][0]["sample_source"] == "propagated"
    assert certificate["worst_margin"] > 0.0


def test_centered_restart_box_from_samples_contains_source_box() -> None:
    """Supplied nominal samples should define a containing restart box."""
    samples = ((1.0, 2.0, 3.0, 4.0), (1.1, 2.1, 3.1, 4.1))
    certificate = tail.centered_restart_box_from_samples(
        2.0,
        samples,
        (0.9, 1.95, 2.5, 3.8),
        (1.3, 2.2, 3.2, 4.4),
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
    )
    assert certificate["source_box_contained"] is True
    assert certificate["sample_source"] == "supplied"
    assert certificate["box"]["low"][2] <= 2.5
    assert certificate["box"]["high"][3] >= 4.4


def test_x3_zero_wall_certificate_certifies_late_one_way_wall() -> None:
    """The late ordinary-time x3=0 wall should point into x3<0."""
    certificate = tail.x3_zero_wall_certificate(
        time_range=(3.02, 3.2),
        x0_range=(0.45, 0.56),
        x1_range=(5.0, 6.2),
        x2_range=(0.005, 0.02),
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        subdivisions=(4, 4, 2, 1),
        time_subdivisions=2,
    )
    assert certificate["status"] == "certified"
    assert certificate["threshold_margin"] > 0.0
    assert certificate["rhs_x3_upper"] < 0.0
    assert certificate["inward_margin"] > 0.0


def test_x3_zero_wall_uses_exact_formula_for_tiny_x0() -> None:
    """The exact x3=0 formula should certify even when intervals overestimate."""
    certificate = tail.x3_zero_wall_certificate(
        time_range=(3.45, 3.7),
        x0_range=(1e-6, 0.4),
        x1_range=(4.8, 20.0),
        x2_range=(0.001, 0.1),
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
    )
    assert certificate["status"] == "certified"
    assert certificate["analytic_rhs_x3_upper"] < 0.0
    assert certificate["analytic_inward_margin"] > 0.0
    assert certificate["interval_status"] == "overconservative_or_failed"


def test_x2_zero_boundary_factor_certificate_certifies_small_p_wall() -> None:
    """The exact x2=0 factorization should give an explicit finite-A margin."""
    certificate = tail.x2_zero_boundary_factor_certificate(
        p_range=(3.5e-4, 0.29),
        x3_range=(-1.5, 0.0),
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
    )
    assert certificate["status"] == "certified"
    assert certificate["factor_lower_bound"] > 0.0
    assert certificate["x2_prime_lower_bound_on_wall"] > 0.0


def test_cli_x2_zero_factor_check_reports_certificate(capsys) -> None:
    """The CLI should expose the exact x2=0 factor certificate."""
    tail.main(
        [
            "--x2-zero-factor-check",
            "--x2-zero-factor-p-range",
            "0.001,0.29",
        ]
    )
    output = capsys.readouterr().out
    assert "x2=0 factor certificate: status=certified" in output


def test_late_x3_descent_certificate_reaches_negative_side() -> None:
    """The late bridge should force x3 below zero and x0 below 0.4."""
    certificate = tail.late_x3_descent_certificate(
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
    )
    assert certificate["status"] == "certified_conditional"
    assert certificate["x3_negative"] is True
    assert certificate["x0_below_target"] is True
    assert certificate["end_box"]["high"][3] < 0.0
    assert certificate["end_box"]["high"][0] < tail.DEFAULT_LATE_X3_DESCENT_X0_TARGET
    assert certificate["x3_zero_wall"]["status"] == "certified"


def test_affine_time_corridor_certificate_checks_short_slab() -> None:
    """The ordinary-time affine corridor verifier should certify a tiny slab."""
    start_time = 0.5
    end_time = 0.5001
    state = tail.scaled_state_at("limit", start_time, step_size=1e-4)
    rhs = tail.limiting_scaled_rhs(start_time, state)
    lower_start = tuple(value - 1e-7 for value in state)
    upper_start = tuple(value + 1e-7 for value in state)
    lower_slope = tuple(value - 0.1 for value in rhs)
    upper_slope = tuple(value + 0.1 for value in rhs)
    certificate = tail.affine_time_corridor_certificate(
        start_time=start_time,
        end_time=end_time,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        lower_start=lower_start,
        upper_start=upper_start,
        lower_slope=lower_slope,
        upper_slope=upper_slope,
    )
    assert certificate["status"] == "certified"
    assert certificate["worst_margin"] > 0.0
    assert certificate["source_box_contained"] is True


def test_automatic_time_barrier_corridor_certifies_short_chain() -> None:
    """The automatic ordinary-time corridor should produce verified slabs."""
    certificate = tail.automatic_time_barrier_corridor_certificate(
        start_time=0.5,
        end_time=0.502,
        step_size=0.001,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        safety=(0.005, 0.05, 0.0005, 0.005),
    )
    assert certificate["status"] == "certified"
    assert certificate["steps_certified"] == 2
    assert certificate["end_time"] == 0.502
    assert certificate["worst_margin"] > 0.0
    assert certificate["conditional"] == "start_box_contains_true_state"


def test_taylor_start_block_certificate_certifies_first_singular_step() -> None:
    """The c2 Taylor start box should support the first tiny t-time slab."""
    certificate = tail.taylor_start_block_certificate(
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
    )
    assert certificate["status"] == "certified_conditional"
    assert certificate["step_certificate"]["status"] == "certified"
    assert certificate["worst_margin"] > 0.0
    assert certificate["conditional"] == "taylor_remainder_is_inside_radius"


def test_taylor_time_bridge_certifies_short_taylor_start_chain() -> None:
    """The staged bridge should carry the Taylor start box beyond the first slab."""
    certificate = tail.taylor_time_bridge_certificate(
        end_time=0.01,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        max_attempts=120,
    )
    assert certificate["status"] == "certified_conditional"
    assert certificate["certified_until"] == pytest.approx(0.01)
    assert certificate["blocks_certified"] > 1
    assert certificate["end_width"][0] > 0.0
    assert certificate["conditional"] == "taylor_remainder_is_inside_radius"


def test_taylor_time_bridge_reports_optional_progress() -> None:
    """Long bridge runs should be able to report reusable progress events."""
    events = []
    certificate = tail.taylor_time_bridge_certificate(
        end_time=0.002,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        max_attempts=120,
        progress_callback=events.append,
        progress_every_blocks=1,
    )
    assert certificate["status"] == "certified_conditional"
    assert events
    assert events[-1]["certified_until"] == pytest.approx(0.002)
    assert events[-1]["blocks_certified"] == certificate["blocks_certified"]


def test_taylor_time_bridge_snaps_zero_step_roundoff_to_stage_end() -> None:
    """Endpoint roundoff should not create infinite zero-step bridge blocks."""
    tiny_end = tail.DEFAULT_TAYLOR_START_TIME + 2e-15
    certificate = tail.taylor_time_bridge_certificate(
        end_time=tiny_end,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        stages=(
            (
                tiny_end,
                5e-5,
                1,
                (5e-5, 0.0025, 2.5e-5, 0.00025),
                (0.0005, 0.025, 0.0025, 0.0025),
            ),
        ),
        max_attempts=1,
    )
    assert certificate["status"] == "certified_conditional"
    assert certificate["certified_until"] == pytest.approx(tiny_end)
    assert certificate["blocks_certified"] == 0


def test_taylor_frontier_continuation_composes_short_bridge() -> None:
    """The frontier continuation should start from the Taylor bridge endpoint."""
    certificate = tail.taylor_frontier_continuation_certificate(
        bridge_end_time=0.01,
        end_time=0.011,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        bridge_max_attempts=120,
        max_attempts=120,
    )
    assert certificate["status"] == "certified_conditional"
    assert certificate["certified_until"] == pytest.approx(0.011)
    assert certificate["steps_certified"] == 1
    assert certificate["taylor_bridge"]["status"] == "certified_conditional"
    assert certificate["conditional"] == "taylor_remainder_is_inside_radius"


def test_taylor_restart_chain_composes_short_bridge_with_restart() -> None:
    """The restart chain should preserve the Taylor-start proof conditional."""
    certificate = tail.taylor_restart_chain_certificate(
        bridge_end_time=0.01,
        end_time=0.012,
        restart_interval=0.001,
        step_size=0.001,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        bridge_max_attempts=120,
        max_attempts=120,
    )
    assert certificate["status"] == "certified_conditional"
    assert certificate["certified_until"] == pytest.approx(0.012)
    assert certificate["segments_certified"] == 2
    assert len(certificate["restarts"]) == 1
    assert certificate["taylor_bridge"]["status"] == "certified_conditional"
    assert certificate["conditional"] == "taylor_remainder_is_inside_radius"


def test_taylor_restart_chain_reports_optional_progress() -> None:
    """Restart-chain runs should report segment-level progress when requested."""
    events = []
    certificate = tail.taylor_restart_chain_certificate(
        bridge_end_time=0.01,
        end_time=0.012,
        restart_interval=0.001,
        step_size=0.001,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        bridge_max_attempts=120,
        max_attempts=120,
        progress_callback=events.append,
        progress_every_segments=1,
    )
    assert certificate["status"] == "certified_conditional"
    assert events
    assert events[-1]["certified_until"] == pytest.approx(0.012)
    assert events[-1]["segments_certified"] == certificate["segments_certified"]


def test_taylor_restart_chain_records_retry_subdivision(monkeypatch) -> None:
    """A failed coarse block rescued by retry subdivision should be auditable."""
    calls = []

    def fake_block(
        start_time,
        step_size,
        block_steps,
        candidate_a,
        start_samples,
        start_low,
        start_high,
        *,
        subdivisions,
        **_kwargs,
    ):
        calls.append(tuple(subdivisions))
        if tuple(subdivisions) == (1, 1, 1, 1):
            return {
                "status": "failed",
                "worst_margin": -1.0,
                "failing_face": {"side": "lower", "component": 2},
                "tuning_attempts": [{"status": "failed"}],
                "end_samples": start_samples,
                "end_box": {"low": list(start_low), "high": list(start_high)},
            }
        return {
            "status": "certified",
            "worst_margin": 0.5,
            "worst_face": {"side": "lower", "component": 2},
            "tuning_attempts": [{"status": "certified"}],
            "end_samples": start_samples,
            "end_box": {"low": list(start_low), "high": list(start_high)},
        }

    monkeypatch.setattr(tail, "tuned_tube_block_certificate", fake_block)
    bridge = {
        "status": "certified_conditional",
        "certified_until": 0.01,
        "end_samples": [[1.0, 2.0, 3.0, 4.0]],
        "end_box": {"low": [0.0, 1.0, 2.0, 3.0], "high": [2.0, 3.0, 4.0, 5.0]},
    }
    certificate = tail.taylor_restart_chain_certificate(
        bridge_end_time=0.01,
        end_time=0.011,
        restart_interval=0.001,
        step_size=0.001,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        bridge_certificate=bridge,
        subdivisions=(1, 1, 1, 1),
        retry_subdivisions=((2, 1, 2, 2),),
    )
    assert certificate["status"] == "certified_conditional"
    assert certificate["retry_count"] == 1
    assert certificate["retry_log"][0]["subdivisions"] == [2, 1, 2, 2]
    assert calls == [(1, 1, 1, 1), (2, 1, 2, 2)]


def test_taylor_restart_chain_reuses_saved_bridge_payload(tmp_path) -> None:
    """A saved Taylor bridge JSON should avoid recomputing the bridge prefix."""
    bridge = tail.taylor_time_bridge_certificate(
        end_time=0.01,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        max_attempts=120,
    )
    bridge_path = tmp_path / "bridge.json"
    bridge_path.write_text(json.dumps({"taylor_time_bridge_certificate": bridge}), encoding="utf-8")

    loaded = tail._load_taylor_bridge_certificate(bridge_path)
    certificate = tail.taylor_restart_chain_certificate(
        bridge_end_time=0.01,
        end_time=0.011,
        restart_interval=0.001,
        step_size=0.001,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        max_attempts=120,
        bridge_certificate=loaded,
    )
    assert certificate["status"] == "certified_conditional"
    assert certificate["certified_until"] == pytest.approx(0.011)
    assert certificate["taylor_bridge"]["certified_until"] == pytest.approx(0.01)


def test_cli_taylor_restart_chain_accepts_saved_bridge_json(tmp_path, capsys) -> None:
    """The CLI should accept a saved bridge report for restart-chain checks."""
    bridge = tail.taylor_time_bridge_certificate(
        end_time=0.01,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        max_attempts=120,
    )
    bridge_path = tmp_path / "bridge-report.json"
    bridge_path.write_text(json.dumps({"taylor_time_bridge_certificate": bridge}), encoding="utf-8")

    tail.main(
        [
            "--taylor-restart-chain-check",
            "--taylor-bridge-end",
            "0.01",
            "--taylor-restart-end",
            "0.011",
            "--taylor-restart-interval",
            "0.001",
            "--tube-step",
            "0.001",
            "--taylor-restart-bridge-json",
            str(bridge_path),
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor restart-chain certificate: status=certified_conditional" in output
    assert "certified_until=0.011" in output


def test_cli_taylor_bridge_progress_goes_to_stderr(capsys) -> None:
    """Progress lines should not contaminate stdout JSON/file output."""
    tail.main(
        [
            "--taylor-time-bridge-check",
            "--taylor-bridge-end",
            "0.002",
            "--taylor-progress-every-blocks",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert "Taylor bridge progress:" in captured.err
    assert "Taylor time-bridge certificate:" in captured.out


def test_cli_taylor_restart_progress_goes_to_stderr(tmp_path, capsys) -> None:
    """Cached restart-chain progress should be visible on stderr."""
    bridge = tail.taylor_time_bridge_certificate(
        end_time=0.01,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        max_attempts=120,
    )
    bridge_path = tmp_path / "bridge-report.json"
    bridge_path.write_text(json.dumps({"taylor_time_bridge_certificate": bridge}), encoding="utf-8")

    tail.main(
        [
            "--taylor-restart-chain-check",
            "--taylor-bridge-end",
            "0.01",
            "--taylor-restart-end",
            "0.011",
            "--taylor-restart-interval",
            "0.001",
            "--tube-step",
            "0.001",
            "--taylor-restart-bridge-json",
            str(bridge_path),
            "--taylor-restart-progress-every-segments",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert "Taylor restart progress:" in captured.err
    assert "Taylor restart-chain certificate:" in captured.out


def test_taylor_p_slice_convergence_audit_reports_radius_comparison() -> None:
    """The p-slice audit should compare Taylor orders against a 5D start radius."""
    audit = tail.taylor_p_slice_convergence_audit(
        target_p=0.95,
        low_order=4,
        high_order=6,
        working_dps=50,
        radius0=(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert audit["status"] == "observed_convergence_inside_start_radius"
    assert audit["target_p"] == pytest.approx(0.95)
    assert audit["low_order"] == 4
    assert audit["high_order"] == 6
    assert len(audit["rows"]) == 3
    assert len(audit["max_order_difference_5d"]) == 5
    assert all(ratio < 1.0 for ratio in audit["max_order_difference_over_radius"])


def test_cli_taylor_p_slice_audit_reports_summary(capsys) -> None:
    """The CLI should expose the p-slice convergence audit."""
    tail.main(
        [
            "--taylor-p-slice-audit",
            "--taylor-p-slice-target",
            "0.95",
            "--taylor-p-slice-low-order",
            "4",
            "--taylor-p-slice-high-order",
            "6",
            "--taylor-p-slice-working-dps",
            "50",
            "--taylor-p-slice-radius",
            "1,1,1,1,1",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor p-slice convergence audit: status=observed_convergence_inside_start_radius" in output
    assert "orders=4->6" in output


def test_taylor_p_slice_tail_ratio_audit_reports_formal_tail() -> None:
    """The tail-ratio audit should report formal tail/radius comparisons."""
    audit = tail.taylor_p_slice_tail_ratio_audit(
        target_p=0.95,
        order=8,
        tail_start=6,
        ratio_start=4,
        ratio_bound=0.99,
        b_sample_count=5,
        working_dps=50,
        radius0=(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert audit["status"] == "formal_geometric_tail_inside_start_radius"
    assert audit["order"] == 8
    assert audit["tail_start"] == 6
    assert audit["b_sample_count"] == 5
    assert len(audit["rows"]) == 5
    assert len(audit["max_tail_estimate_5d"]) == 5
    assert audit["observed_ratios_inside_bound"] is True
    assert audit["ratio_bound"] == pytest.approx(0.99)
    assert all(ratio < 1.0 for ratio in audit["max_tail_estimate_over_radius"])
    assert len(audit["max_observed_ratio_witness_4d"]) == 4


def test_cli_taylor_p_slice_tail_audit_reports_summary(capsys) -> None:
    """The CLI should expose the formal p-slice tail audit."""
    tail.main(
        [
            "--taylor-p-slice-tail-audit",
            "--taylor-p-slice-target",
            "0.95",
            "--taylor-p-slice-tail-order",
            "8",
            "--taylor-p-slice-tail-start",
            "6",
            "--taylor-p-slice-ratio-start",
            "4",
            "--taylor-p-slice-ratio-bound",
            "0.99",
            "--taylor-p-slice-b-samples",
            "5",
            "--taylor-p-slice-tail-working-dps",
            "50",
            "--taylor-p-slice-radius",
            "1,1,1,1,1",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor p-slice tail-ratio audit: status=formal_geometric_tail_inside_start_radius" in output
    assert "order=8, tail_start=6" in output
    assert "ratio_bound=0.99" in output
    assert "b_samples=5" in output


def test_taylor_p_slice_cauchy_budget_audit_reports_viability() -> None:
    """The Cauchy-budget audit should report per-radius proof budgets."""
    audit = tail.taylor_p_slice_cauchy_budget_audit(
        target_p=0.95,
        order=8,
        tail_start=6,
        b_sample_count=3,
        working_dps=50,
        analytic_radii=(2.0, 3.0),
        circle_sample_count=12,
        circle_tail_ratio_bound=0.99,
        radius0=(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert audit["status"] in {
        "observed_cauchy_budget_has_proof_relevant_viable_radius",
        "observed_cauchy_budget_only_viable_beyond_real_terminal",
        "observed_cauchy_budget_has_no_viable_radius",
    }
    assert audit["analytic_radii"] == [2.0, 3.0]
    assert len(audit["radius_rows"]) == 2
    assert "proof_relevant_viable_analytic_radii" in audit
    assert "limiting_crossing_time_reference" in audit
    assert "best_radius_by_observed_floor" in audit
    assert "best_radius_min_p_circle_abs_partial" in audit
    assert "best_radius_certified_min_p_circle_abs_partial" in audit
    assert "best_radius_p_circle_rouche_margin" in audit
    assert "best_radius_p_circle_tail_inside_ratio_bound" in audit


def test_cli_taylor_p_slice_cauchy_budget_audit_reports_summary(capsys) -> None:
    """The CLI should expose the Cauchy-budget p-slice audit."""
    tail.main(
        [
            "--taylor-p-slice-cauchy-budget-audit",
            "--taylor-p-slice-target",
            "0.95",
            "--taylor-p-slice-tail-order",
            "8",
            "--taylor-p-slice-tail-start",
            "6",
            "--taylor-p-slice-b-samples",
            "3",
            "--taylor-p-slice-tail-working-dps",
            "50",
            "--taylor-p-slice-cauchy-radii",
            "2,3",
            "--taylor-p-slice-cauchy-circle-samples",
            "12",
            "--taylor-p-slice-cauchy-circle-tail-ratio-bound",
            "0.99",
            "--taylor-p-slice-radius",
            "1,1,1,1,1",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor p-slice Cauchy-budget audit: status=observed_cauchy_budget_" in output
    assert "best_radius=" in output
    assert "best_p_circle_min=" in output
    assert "best_p_circle_certified_min=" in output
    assert "best_p_circle_rouche_margin=" in output
    assert "best_p_circle_ratio=" in output
    assert "best_p_circle_inside_ratio_bound=" in output


def test_taylor_ratio_profile_audit_reports_bounds() -> None:
    """The ratio-profile audit should expose circle and p-slice ratio windows."""
    audit = tail.taylor_ratio_profile_audit(
        target_p=0.95,
        order=8,
        ratio_start=4,
        b_sample_count=3,
        working_dps=50,
        circle_radius=2.0,
        circle_ratio_bound=0.99,
        p_slice_ratio_bound=0.99,
    )
    assert audit["status"] in {"observed_ratios_inside_bounds", "observed_ratios_exceed_bounds"}
    assert audit["circle_radius"] == pytest.approx(2.0)
    assert len(audit["max_circle_ratio_4d"]) == 4
    assert len(audit["max_p_slice_ratio_4d"]) == 4
    assert len(audit["rows"]) == 3


def test_taylor_ratio_profile_audit_can_use_limit_mode() -> None:
    """The ratio-profile audit should support a cheaper b=0-only mode."""
    audit = tail.taylor_ratio_profile_audit(
        target_p=0.95,
        order=8,
        ratio_start=4,
        b_sample_count=3,
        b_mode="limit",
        working_dps=50,
        circle_radius=2.0,
        circle_ratio_bound=0.99,
        p_slice_ratio_bound=0.99,
    )
    assert audit["b_mode"] == "limit"
    assert audit["b_sample_count"] == 1
    assert audit["requested_b_sample_count"] == 3
    assert audit["b_samples"] == [0.0]
    assert len(audit["rows"]) == 1


def test_cli_taylor_ratio_profile_audit_reports_summary(capsys) -> None:
    """The CLI should expose the ratio-profile audit."""
    tail.main(
        [
            "--taylor-ratio-profile-audit",
            "--taylor-p-slice-target",
            "0.95",
            "--taylor-p-slice-tail-order",
            "8",
            "--taylor-p-slice-ratio-start",
            "4",
            "--taylor-p-slice-b-samples",
            "3",
            "--taylor-p-slice-tail-working-dps",
            "50",
            "--taylor-ratio-profile-circle-radius",
            "2.0",
            "--taylor-ratio-profile-circle-ratio-bound",
            "0.99",
            "--taylor-ratio-profile-p-slice-ratio-bound",
            "0.99",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor ratio-profile audit: status=observed_ratios_" in output
    assert "b_mode=grid" in output
    assert "circle_inside=" in output
    assert "p_slice_inside=" in output


def test_taylor_geometric_envelope_audit_reports_usage() -> None:
    """The geometric-envelope audit should report finite-window envelope usage."""
    audit = tail.taylor_geometric_envelope_audit(
        target_p=0.95,
        order=8,
        tail_start=4,
        b_sample_count=3,
        b_mode="limit",
        working_dps=50,
        circle_radius=2.0,
        circle_ratio_bound=0.99,
        p_slice_ratio_bound=0.99,
    )
    assert audit["status"] in {
        "observed_terms_inside_geometric_envelopes",
        "observed_terms_exceed_geometric_envelopes",
    }
    assert audit["b_mode"] == "limit"
    assert audit["b_sample_count"] == 1
    assert len(audit["max_circle_envelope_usage_4d"]) == 4
    assert len(audit["max_p_slice_envelope_usage_4d"]) == 4
    assert len(audit["max_circle_strict_post_anchor_usage_4d"]) == 4
    assert len(audit["max_p_slice_strict_post_anchor_usage_4d"]) == 4
    assert len(audit["max_circle_tail_sum_usage_4d"]) == 4
    assert len(audit["max_p_slice_tail_sum_usage_4d"]) == 4
    assert len(audit["rows"]) == 1


def test_cli_taylor_geometric_envelope_audit_reports_summary(capsys) -> None:
    """The CLI should expose the geometric-envelope audit."""
    tail.main(
        [
            "--taylor-geometric-envelope-audit",
            "--taylor-ratio-profile-b-mode",
            "limit",
            "--taylor-p-slice-target",
            "0.95",
            "--taylor-p-slice-tail-order",
            "8",
            "--taylor-p-slice-tail-start",
            "4",
            "--taylor-p-slice-tail-working-dps",
            "50",
            "--taylor-ratio-profile-circle-radius",
            "2.0",
            "--taylor-ratio-profile-circle-ratio-bound",
            "0.99",
            "--taylor-ratio-profile-p-slice-ratio-bound",
            "0.99",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor geometric-envelope audit: status=observed_terms_" in output
    assert "max_circle_usage=" in output
    assert "max_circle_strict_usage=" in output
    assert "max_p_slice_usage=" in output
    assert "max_p_slice_strict_usage=" in output


def test_taylor_even_parity_audit_reports_zero_odd_coefficients() -> None:
    """The parity audit should check odd Taylor coefficients on real and complex b samples."""
    audit = tail.taylor_even_parity_audit(
        order=8,
        b_sample_count=3,
        working_dps=50,
        complex_b_radius=1e-7,
        complex_b_sample_count=4,
    )
    assert audit["status"] == "observed_odd_coefficients_zero"
    assert audit["max_odd_abs_4d"] == [0.0, 0.0, 0.0, 0.0]
    assert len(audit["rows"]) == 7


def test_cli_taylor_even_parity_audit_reports_summary(capsys) -> None:
    """The CLI should expose the even-parity audit."""
    tail.main(
        [
            "--taylor-even-parity-audit",
            "--taylor-p-slice-tail-order",
            "8",
            "--taylor-p-slice-tail-working-dps",
            "50",
            "--taylor-b-cauchy-enclosure-radius",
            "1e-7",
            "--taylor-b-cauchy-enclosure-samples",
            "4",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor even-parity audit: status=observed_odd_coefficients_zero" in output
    assert "max_odd_abs=" in output


def test_taylor_even_s_series_audit_reports_s_targets() -> None:
    """The even-s audit should report the ordinary s-series target."""
    audit = tail.taylor_even_s_series_audit(
        target_p=0.95,
        order=8,
        tail_start=4,
        b_sample_count=3,
        b_mode="limit",
        working_dps=50,
        circle_radius=2.0,
        circle_ratio_bound=0.99,
        p_slice_ratio_bound=0.99,
    )
    assert audit["status"] in {"observed_s_series_inside_targets", "observed_s_series_exceeds_targets"}
    assert audit["tail_start_t_degree"] == 4
    assert audit["tail_start_s_index"] == 2
    assert audit["circle_radius_s"] == pytest.approx(4.0)
    assert len(audit["max_circle_ratio_4d"]) == 4
    assert len(audit["min_inferred_circle_radius_s_4d"]) == 4
    assert len(audit["rows"]) == 1


def test_cli_taylor_even_s_series_audit_reports_summary(capsys) -> None:
    """The CLI should expose the even-s-series audit."""
    tail.main(
        [
            "--taylor-even-s-series-audit",
            "--taylor-ratio-profile-b-mode",
            "limit",
            "--taylor-p-slice-target",
            "0.95",
            "--taylor-p-slice-tail-order",
            "8",
            "--taylor-p-slice-tail-start",
            "4",
            "--taylor-p-slice-tail-working-dps",
            "50",
            "--taylor-ratio-profile-circle-radius",
            "2.0",
            "--taylor-ratio-profile-circle-ratio-bound",
            "0.99",
            "--taylor-ratio-profile-p-slice-ratio-bound",
            "0.99",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor even-s-series audit: status=observed_s_series_" in output
    assert "tail_start_s=" in output
    assert "terminal_s=" in output


def test_recurrence_inverse_apply_round_trips_matrix() -> None:
    """The closed-form recurrence inverse should invert the degree matrix."""
    with tail.mp.workdps(80):
        vector = (tail.mp.mpf("0.1"), tail.mp.mpf("-0.2"), tail.mp.mpf("0.03"), tail.mp.mpf("0.4"))
        forcing = tail._recurrence_matrix_apply(10, vector)
        recovered = tail._recurrence_inverse_apply(10, forcing)
        assert max(abs(left - right) for left, right in zip(vector, recovered)) < tail.mp.mpf("1e-40")


def test_taylor_recurrence_forcing_audit_reports_inverse_bounds() -> None:
    """The recurrence-forcing audit should expose inverse and forcing diagnostics."""
    audit = tail.taylor_recurrence_forcing_audit(
        order=8,
        tail_start=4,
        b_sample_count=3,
        b_mode="limit",
        working_dps=50,
        circle_radius=2.0,
        circle_ratio_bound=0.99,
    )
    assert audit["status"] in {
        "observed_recurrence_forcing_inside_targets",
        "observed_recurrence_forcing_exceeds_targets",
    }
    assert audit["matrix_determinant_formula"] == "d*(d+1)*(d+4)*(d+6)"
    assert audit["tail_start_s_index"] == 2
    assert max(audit["max_reconstruction_error_4d"]) < 1e-40
    assert len(audit["max_forcing_ratio_4d"]) == 4
    assert len(audit["rows"]) == 1


def test_cli_taylor_recurrence_forcing_audit_reports_summary(capsys) -> None:
    """The CLI should expose the recurrence-forcing audit."""
    tail.main(
        [
            "--taylor-recurrence-forcing-audit",
            "--taylor-ratio-profile-b-mode",
            "limit",
            "--taylor-p-slice-tail-order",
            "8",
            "--taylor-p-slice-tail-start",
            "4",
            "--taylor-p-slice-tail-working-dps",
            "50",
            "--taylor-ratio-profile-circle-radius",
            "2.0",
            "--taylor-ratio-profile-circle-ratio-bound",
            "0.99",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor recurrence-forcing audit: status=observed_recurrence_forcing_" in output
    assert "max_inverse_usage=" in output
    assert "max_forcing_ratio=" in output


def test_taylor_b_sensitivity_audit_reports_finite_b_deltas() -> None:
    """The finite-b sensitivity audit should compare samples to the b=0 germ."""
    audit = tail.taylor_b_sensitivity_audit(
        target_p=0.95,
        order=8,
        ratio_start=4,
        b_sample_count=3,
        working_dps=50,
        circle_radius=2.0,
        radius0=(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert audit["status"] == "finite_b_state_delta_inside_start_radius"
    assert audit["circle_radius"] == pytest.approx(2.0)
    assert len(audit["limit_state_5d"]) == 5
    assert len(audit["max_state_delta_5d"]) == 5
    assert len(audit["max_circle_delta_l1_4d"]) == 4
    assert len(audit["max_circle_tail_delta_l1_relative_to_limit_4d"]) == 4
    assert len(audit["rows"]) == 3


def test_cli_taylor_b_sensitivity_audit_reports_summary(capsys) -> None:
    """The CLI should expose the finite-b sensitivity audit."""
    tail.main(
        [
            "--taylor-b-sensitivity-audit",
            "--taylor-p-slice-target",
            "0.95",
            "--taylor-p-slice-tail-order",
            "8",
            "--taylor-p-slice-ratio-start",
            "4",
            "--taylor-p-slice-b-samples",
            "3",
            "--taylor-p-slice-tail-working-dps",
            "50",
            "--taylor-b-sensitivity-circle-radius",
            "2.0",
            "--taylor-p-slice-radius",
            "1,1,1,1,1",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor b-sensitivity audit: status=finite_b_state_delta_" in output
    assert "max_state_delta/radius=" in output
    assert "max_circle_tail_delta_l1_rel=" in output


def test_complex_taylor_coefficients_extend_real_recurrence() -> None:
    """Complex-b diagnostic coefficients should agree on the real axis."""
    real = tail.scaled_taylor_coefficients(6, 1e-8, 50)
    complex_coefficients = tail.complex_scaled_taylor_coefficients(6, 1e-8, 50)
    for component in range(4):
        for degree in range(7):
            assert abs(complex_coefficients[component][degree] - real[component][degree]) < 1e-40


def test_taylor_p_slice_b_cauchy_event_audit_reports_event_delta() -> None:
    """The complex-b Cauchy event audit should bound finite p-slice motion."""
    audit = tail.taylor_p_slice_b_cauchy_event_audit(
        target_p=0.95,
        order=8,
        working_dps=60,
        b_cauchy_radius=1e-7,
        b_circle_sample_count=4,
        radius0=(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert audit["status"] == "sampled_b_cauchy_event_delta_inside_start_radius"
    assert audit["b_cauchy_radius"] == pytest.approx(1e-7)
    assert audit["target_p"] == pytest.approx(0.95)
    assert len(audit["cauchy_delta_bound_5d"]) == 5
    assert len(audit["cauchy_delta_bound_over_radius"]) == 5
    assert len(audit["max_circle_delta_witness_5d"]) == 5
    assert len(audit["max_adjacent_angular_slope_5d"]) == 5
    assert len(audit["empirical_cauchy_delta_bound_over_radius"]) == 5
    assert len(audit["proof_cauchy_delta_bound_over_radius"]) == 5
    assert audit["proof_cauchy_source"] == "sampled_inner_circle_max"
    assert len(audit["sample_rows"]) == 4
    assert len(audit["direct_endpoint_rows"]) == 2
    assert audit["max_p_residual_abs"] < 1e-35
    assert audit["min_event_p_derivative_abs"] > 0.0
    assert audit["min_event_p_derivative_witness"]["circle"] == "inner"
    assert "event_p_derivative_abs" in audit["sample_rows"][0]


def test_taylor_p_slice_b_cauchy_event_audit_reports_outer_bound() -> None:
    """The event audit should optionally use an outer-circle angular bound."""
    audit = tail.taylor_p_slice_b_cauchy_event_audit(
        target_p=0.95,
        order=6,
        working_dps=50,
        b_cauchy_radius=1e-7,
        b_circle_sample_count=4,
        b_outer_cauchy_radius=2e-7,
        b_outer_circle_sample_count=4,
        radius0=(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert audit["status"] == "sampled_b_cauchy_event_delta_inside_start_radius"
    assert audit["proof_cauchy_source"] == "sampled_outer_circle_cauchy_angular_bound"
    assert len(audit["outer_sample_rows"]) == 4
    assert len(audit["outer_cauchy_delta_bound_over_radius"]) == 5
    assert audit["proof_cauchy_delta_bound_over_radius"] == audit["outer_cauchy_delta_bound_over_radius"]


def test_taylor_p_slice_b_cauchy_event_audit_reports_nested_enclosure_bound() -> None:
    """The event audit should optionally control the outer circle from an enclosing circle."""
    audit = tail.taylor_p_slice_b_cauchy_event_audit(
        target_p=0.95,
        order=6,
        working_dps=50,
        b_cauchy_radius=1e-7,
        b_circle_sample_count=4,
        b_outer_cauchy_radius=2e-7,
        b_outer_circle_sample_count=4,
        b_enclosure_cauchy_radius=4e-7,
        b_enclosure_circle_sample_count=4,
        radius0=(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert audit["status"] == "sampled_b_cauchy_event_delta_inside_start_radius"
    assert audit["proof_cauchy_source"] == "sampled_enclosure_circle_nested_cauchy"
    assert len(audit["enclosure_sample_rows"]) == 4
    assert len(audit["enclosure_cauchy_outer_circle_delta_bound_5d"]) == 5
    assert len(audit["outer_cauchy_delta_bound_over_radius"]) == 5
    assert audit["proof_cauchy_delta_bound_over_radius"] == audit["outer_cauchy_delta_bound_over_radius"]


def test_cli_taylor_p_slice_b_cauchy_event_audit_reports_summary(capsys) -> None:
    """The CLI should expose the p-slice b-Cauchy event diagnostic."""
    tail.main(
        [
            "--taylor-p-slice-b-cauchy-event-audit",
            "--taylor-p-slice-target",
            "0.95",
            "--taylor-p-slice-tail-order",
            "8",
            "--taylor-p-slice-tail-working-dps",
            "60",
            "--taylor-b-cauchy-radius",
            "1e-7",
            "--taylor-b-cauchy-samples",
            "4",
            "--taylor-p-slice-radius",
            "1,1,1,1,1",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor p-slice b-Cauchy event audit: status=sampled_b_cauchy_event_delta_" in output
    assert "direct_delta/radius=" in output
    assert "cauchy_delta/radius=" in output
    assert "empirical_cauchy_delta/radius=" in output
    assert "proof_cauchy_delta/radius=" in output
    assert "min_event_p_derivative=" in output


def test_taylor_p_slice_entry_budget_audit_combines_tail_and_event() -> None:
    """The entry-budget audit should combine tail and finite-b event budgets."""
    audit = tail.taylor_p_slice_entry_budget_audit(
        target_p=0.95,
        order=8,
        tail_start=6,
        ratio_start=4,
        ratio_bound=0.99,
        b_sample_count=3,
        working_dps=60,
        b_cauchy_radius=1e-7,
        b_circle_sample_count=4,
        radius0=(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert audit["status"] == "formal_entry_budget_inside_start_radius"
    assert audit["tail_status"] == "formal_geometric_tail_inside_start_radius"
    assert audit["event_cauchy_status"] == "sampled_b_cauchy_event_delta_inside_start_radius"
    assert len(audit["tail_budget_5d"]) == 5
    assert len(audit["finite_b_budget_5d"]) == 5
    assert len(audit["combined_budget_5d"]) == 5
    assert audit["max_combined_budget_over_radius"] < 1.0


def test_taylor_p_slice_required_a_audit_reports_threshold() -> None:
    """The required-A audit should expose the conditional entry threshold."""
    audit = tail.taylor_p_slice_required_a_audit(
        target_p=0.95,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        order=8,
        tail_start=6,
        ratio_start=4,
        ratio_bound=0.99,
        b_sample_count=3,
        working_dps=60,
        b_cauchy_radius=1e-7,
        b_circle_sample_count=4,
        radius0=(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert audit["status"] == "candidate_A_fits_conditional_entry_budget"
    assert audit["minimum_A_for_conditional_entry_budget"] < audit["candidate_A"]
    assert audit["candidate_A_headroom_factor"] > 1.0
    assert audit["event_cauchy_source"] == "sampled_inner_circle_max"
    assert len(audit["component_rows"]) == 5
    assert audit["entry_budget_audit"]["status"] == "formal_entry_budget_inside_start_radius"


def test_cli_taylor_p_slice_entry_budget_audit_reports_summary(capsys) -> None:
    """The CLI should expose the combined p-slice entry budget."""
    tail.main(
        [
            "--taylor-p-slice-entry-budget-audit",
            "--taylor-p-slice-target",
            "0.95",
            "--taylor-p-slice-tail-order",
            "8",
            "--taylor-p-slice-tail-start",
            "6",
            "--taylor-p-slice-ratio-start",
            "4",
            "--taylor-p-slice-ratio-bound",
            "0.99",
            "--taylor-p-slice-tail-working-dps",
            "60",
            "--taylor-b-cauchy-radius",
            "1e-7",
            "--taylor-b-cauchy-samples",
            "4",
            "--taylor-p-slice-radius",
            "1,1,1,1,1",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor p-slice entry-budget audit: status=formal_entry_budget_" in output
    assert "combined/radius=" in output
    assert "event_status=sampled_b_cauchy_event_delta_" in output
    assert "event_source=sampled_inner_circle_max" in output


def test_cli_taylor_p_slice_required_a_audit_reports_summary(capsys) -> None:
    """The CLI should print the conditional explicit-A threshold."""
    tail.main(
        [
            "--taylor-p-slice-required-a-audit",
            "--taylor-p-slice-target",
            "0.95",
            "--taylor-p-slice-tail-order",
            "8",
            "--taylor-p-slice-tail-start",
            "6",
            "--taylor-p-slice-ratio-start",
            "4",
            "--taylor-p-slice-ratio-bound",
            "0.99",
            "--taylor-p-slice-tail-working-dps",
            "60",
            "--taylor-b-cauchy-radius",
            "1e-7",
            "--taylor-b-cauchy-samples",
            "4",
            "--taylor-p-slice-radius",
            "1,1,1,1,1",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor p-slice required-A audit: status=candidate_A_fits_" in output
    assert "minimum_A=" in output
    assert "headroom=" in output
    assert "event_source=sampled_inner_circle_max" in output


def test_taylor_b_cauchy_coefficient_audit_reports_support_budget() -> None:
    """The b-Cauchy diagnostic should compare finite coefficient motion to a support radius."""
    audit = tail.taylor_b_cauchy_coefficient_audit(
        order=6,
        working_dps=50,
        time_radius=1.0,
        b_cauchy_radius=1e-7,
        b_circle_sample_count=4,
        support_radius0=(1.0, 1.0, 1.0, 1.0),
    )
    assert audit["status"] == "sampled_b_cauchy_delta_inside_support_radius"
    assert audit["b_cauchy_radius"] == pytest.approx(1e-7)
    assert audit["b_circle_sample_count"] == 4
    assert len(audit["cauchy_delta_bound_4d"]) == 4
    assert len(audit["cauchy_delta_bound_over_support_radius"]) == 4
    assert len(audit["sample_rows"]) == 4
    assert len(audit["direct_endpoint_rows"]) == 2


def test_cli_taylor_b_cauchy_coefficient_audit_reports_summary(capsys) -> None:
    """The CLI should expose the b-Cauchy coefficient diagnostic."""
    tail.main(
        [
            "--taylor-b-cauchy-coefficient-audit",
            "--taylor-p-slice-tail-order",
            "6",
            "--taylor-p-slice-tail-working-dps",
            "50",
            "--taylor-b-cauchy-time-radius",
            "1.0",
            "--taylor-b-cauchy-radius",
            "1e-7",
            "--taylor-b-cauchy-samples",
            "4",
            "--support-tail-support-radius",
            "1,1,1,1",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor b-Cauchy coefficient audit: status=sampled_b_cauchy_delta_" in output
    assert "direct_delta/radius=" in output
    assert "cauchy_delta/radius=" in output


def test_taylor_support_time_convergence_audit_reports_radius_comparison() -> None:
    """The support-time audit should compare order changes to a support radius."""
    audit = tail.taylor_support_time_convergence_audit(
        support_time=1.0,
        low_order=4,
        high_order=6,
        b_sample_count=3,
        working_dps=50,
        support_radius0=(1.0, 1.0, 1.0, 1.0),
    )
    assert audit["status"] == "observed_support_time_convergence_inside_radius"
    assert audit["support_time"] == pytest.approx(1.0)
    assert audit["low_order"] == 4
    assert audit["high_order"] == 6
    assert len(audit["rows"]) == 3
    assert len(audit["max_order_difference_4d"]) == 4
    assert all(value < 1.0 for value in audit["max_order_difference_over_support_radius"])


def test_cli_taylor_support_time_audit_reports_summary(capsys) -> None:
    """The CLI should expose the support-time convergence audit."""
    tail.main(
        [
            "--taylor-support-time-audit",
            "--taylor-support-time",
            "1.0",
            "--taylor-support-low-order",
            "4",
            "--taylor-support-high-order",
            "6",
            "--taylor-p-slice-b-samples",
            "3",
            "--taylor-p-slice-tail-working-dps",
            "50",
            "--support-tail-support-radius",
            "1,1,1,1",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor support-time convergence audit: status=observed_support_time_convergence_" in output
    assert "orders=4->6" in output
    assert "max_diff/radius=" in output


def test_taylor_circle_residual_audit_reports_residuals() -> None:
    """The circle-residual audit should sample equation defects."""
    audit = tail.taylor_circle_residual_audit(
        order=8,
        b_sample_count=3,
        working_dps=50,
        circle_radius=2.0,
        circle_sample_count=12,
    )
    assert audit["status"] in {"sampled_residual_small", "sampled_residual_not_small"}
    assert audit["circle_radius"] == pytest.approx(2.0)
    assert len(audit["max_residual_4d"]) == 4
    assert len(audit["rows"]) == 3
    assert audit["min_p_abs"] > 0.0


def test_cli_taylor_circle_residual_audit_reports_summary(capsys) -> None:
    """The CLI should expose the circle-residual audit."""
    tail.main(
        [
            "--taylor-circle-residual-audit",
            "--taylor-p-slice-tail-order",
            "8",
            "--taylor-p-slice-b-samples",
            "3",
            "--taylor-p-slice-tail-working-dps",
            "50",
            "--taylor-circle-residual-radius",
            "2.0",
            "--taylor-circle-residual-samples",
            "12",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor circle-residual audit: status=sampled_residual_" in output
    assert "max_residual=" in output
    assert "min_p_abs=" in output


def test_interval_taylor_finite_ratio_audit_reports_b_subdivision() -> None:
    """The interval coefficient audit should report finite ratio checks."""
    audit = tail.interval_taylor_finite_ratio_audit(
        target_p=0.99,
        order=8,
        ratio_start=2,
        ratio_bound=0.99,
        b_subdivisions=2,
        working_dps=50,
    )
    assert audit["status"] in {
        "interval_finite_ratios_inside_bound",
        "interval_finite_ratios_exceed_bound",
        "interval_finite_ratios_failed",
    }
    assert audit["b_subdivisions"] == 2
    assert audit["ratio_bound"] == pytest.approx(0.99)
    assert "checked_time_hull" in audit


def test_cli_interval_taylor_ratio_audit_reports_summary(capsys) -> None:
    """The CLI should expose the interval finite-ratio audit."""
    tail.main(
        [
            "--taylor-p-slice-interval-ratio-audit",
            "--taylor-p-slice-target",
            "0.99",
            "--taylor-p-slice-interval-order",
            "8",
            "--taylor-p-slice-ratio-start",
            "2",
            "--taylor-p-slice-ratio-bound",
            "0.99",
            "--taylor-p-slice-b-subdivisions",
            "2",
            "--taylor-p-slice-interval-working-dps",
            "50",
        ]
    )
    output = capsys.readouterr().out
    assert "Taylor p-slice interval-ratio audit: status=interval_finite_ratios_" in output
    assert "b_subdivisions=2" in output


def test_p_time_rhs_matches_t_time_ratio() -> None:
    """The p-time system should be the scaled system divided by x0'."""
    p, x1, x2, x3 = tail.limiting_state_at(3.58)
    transformed = tail.p_time_rhs(p, (3.58, x1, x2, x3), 0.0)
    rhs = tail.scaled_rhs_with_b(3.58, (p, x1, x2, x3), 0.0)
    expected = (1.0 / rhs[0], rhs[1] / rhs[0], rhs[2] / rhs[0], rhs[3] / rhs[0])
    assert max(abs(left - right) for left, right in zip(transformed, expected)) < 1e-12


def test_carried_c_p_time_rhs_extends_p_time_rhs_on_c_graph() -> None:
    """The augmented C system should restrict to the original p-time dynamics."""
    p, x1, x2, x3 = tail.limiting_state_at(3.58)
    y = (3.58, x1, x2, x3)
    c_value = tail.cancellation_c_value(p, y)
    base = tail.p_time_rhs(p, y, 0.0)
    carried = tail.p_time_rhs_carried_c(p, (*y, c_value), 0.0)
    expected_c = x2 * base[1] + x1 * base[2] - p * x3 / 3.0 - p * p * base[3] / 6.0
    assert max(abs(left - right) for left, right in zip(carried[:4], base)) < 1e-12
    assert carried[4] == pytest.approx(expected_c)


def test_augment_p_slice_box_with_c_contains_corner_values() -> None:
    """The algebraic C interval should contain all box-corner C values."""
    low4 = (3.2, 2.0, 0.008, -2.0)
    high4 = (3.9, 20.0, 0.03, 0.5)
    low5, high5 = tail.augment_p_slice_box_with_c(0.39, low4, high4)
    assert low5[:4] == low4
    assert high5[:4] == high4
    values = [
        tail.cancellation_c_value(0.39, (t, x1, x2, x3))
        for t in (low4[0], high4[0])
        for x1 in (low4[1], high4[1])
        for x2 in (low4[2], high4[2])
        for x3 in (low4[3], high4[3])
    ]
    assert low5[4] <= min(values)
    assert max(values) <= high5[4]


def test_sharpen_carried_c_p_slice_box_intersects_algebraic_c() -> None:
    """Outgoing carried-C boxes should be tightened by the algebraic identity."""
    low4 = (3.2, 2.0, 0.008, -2.0)
    high4 = (3.9, 20.0, 0.03, 0.5)
    algebraic_low, algebraic_high = tail.cancellation_c_bounds_for_p_slice_box(0.39, low4, high4)
    low5 = (*low4, algebraic_low - 10.0)
    high5 = (*high4, algebraic_high + 10.0)
    sharpened_low, sharpened_high, changed = tail.sharpen_carried_c_p_slice_box(0.39, low5, high5)
    assert changed is True
    assert sharpened_low[:4] == low4
    assert sharpened_high[:4] == high4
    assert sharpened_low[4] == pytest.approx(algebraic_low)
    assert sharpened_high[4] == pytest.approx(algebraic_high)


def test_cancellation_p_prime_bound_removes_fake_zero_denominator() -> None:
    """The C=x1*x2-p^2*x3/6 bound should sharpen p-time denominator checks."""
    from mpmath import iv

    p_interval = iv.mpf([0.39, 0.395])
    y_interval = (
        iv.mpf([3.2, 3.9]),
        iv.mpf([2.0, 20.0]),
        iv.mpf([0.008, 0.03]),
        iv.mpf([-2.0, 0.5]),
    )
    b_interval = iv.mpf([-1e-8, 1e-8])
    with pytest.raises(ZeroDivisionError):
        tail._subdivided_interval_p_time_rhs_component(
            p_interval,
            y_interval,
            b_interval,
            0,
        )
    lower, upper = tail._subdivided_interval_p_time_rhs_component(
        p_interval,
        y_interval,
        b_interval,
        0,
        use_cancellation_p_prime=True,
    )
    assert lower < upper < 0.0


def test_carried_c_p_prime_uses_expanded_graph_intersection() -> None:
    """Equivalent p-prime formulas should remove fake carried-C combinations."""
    from mpmath import iv

    p_interval = iv.mpf([0.3775, 0.378])
    z_interval = (
        iv.mpf([3.576551889045412, 3.7370067844247936]),
        iv.mpf([1.239287409189501, 3.870127447587529]),
        iv.mpf([0.006670120786241291, 0.010186708433343687]),
        iv.mpf([-0.02495132127097288, 0.5712381638266035]),
        iv.mpf([0.04160956041595057, 0.08855638975710586]),
    )
    b_interval = iv.mpf([-1e-8, 1e-8])
    p_prime = tail._interval_scaled_p_prime_with_carried_c(p_interval, z_interval, b_interval)
    assert -0.51 < tail._interval_lower(p_prime) < -0.50
    assert -0.37 < tail._interval_upper(p_prime) < -0.36
    reciprocal = 1.0 / p_prime
    assert tail._interval_upper(reciprocal) < -1.97


def test_cancellation_p_prime_bound_requires_nonnegative_x2() -> None:
    """The dropped limiting term is only nonpositive when the x2 box is nonnegative."""
    from mpmath import iv

    p_interval = iv.mpf([0.39, 0.395])
    y_interval = (
        iv.mpf([3.2, 3.9]),
        iv.mpf([2.0, 20.0]),
        iv.mpf([-0.001, 0.03]),
        iv.mpf([-2.0, 0.5]),
    )
    b_interval = iv.mpf([-1e-8, 1e-8])
    assert tail._interval_p_prime_cancellation_upper(p_interval, y_interval, b_interval) is None


def test_segmented_p_tube_certificate_certifies_short_chain() -> None:
    """The transformed p-time checker should certify a tiny late-tail slab."""
    certificate = tail.segmented_p_tube_certificate(
        start_p=0.305,
        end_p=0.304,
        step_size=1e-3,
        block_steps=1,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        seed_step_size=1e-3,
    )
    assert certificate["status"] == "certified"
    assert certificate["certified_to_p"] == 0.304
    assert certificate["blocks_certified"] == 1
    assert certificate["conditional"] == "start_p_slice_box_contains_true_state"


def test_asymmetric_p_tube_profiles_certify_short_chain() -> None:
    """Asymmetric p-time profiles should remain accepted by the checker."""
    certificate = tail.segmented_p_tube_certificate(
        start_p=0.305,
        end_p=0.304,
        step_size=1e-3,
        block_steps=1,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        profiles=tail.DEFAULT_ASYMMETRIC_P_TUBE_PROFILES,
        seed_step_size=1e-3,
    )
    assert certificate["status"] == "certified"
    block = certificate["last_certified_block"]
    assert "lower_radius_growth" in block
    assert "upper_radius_growth" in block


def test_tuned_segmented_p_tube_certificate_certifies_short_chain() -> None:
    """The tuned p-time checker should certify short transformed slabs."""
    certificate = tail.tuned_segmented_p_tube_certificate(
        start_p=0.305,
        end_p=0.3045,
        step_size=5e-4,
        block_steps=1,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        seed_step_size=1e-3,
    )
    assert certificate["status"] == "certified"
    assert certificate["certified_to_p"] == pytest.approx(0.3045)
    assert certificate["blocks_certified"] == 1
    assert certificate["tuning_attempt_count"] >= 1
    assert certificate["conditional"] == "start_p_slice_box_contains_true_state"


def test_staged_union_p_tube_certificate_certifies_short_split() -> None:
    """Finite-union p-time boxes should compose through a staged split."""
    start = tail.p_start_slice_box(
        start_p=0.305,
        entry_time=tail.DEFAULT_P_TUBE_ENTRY_TIME,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        seed_step_size=1e-3,
    )
    certificate = tail.staged_union_p_tube_certificate(
        start_p=0.305,
        source_box_low=tuple(start["box"]["low"]),
        source_box_high=tuple(start["box"]["high"]),
        stages=((0.3045, (2, 1, 1, 1)),),
        step_size=5e-4,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        max_attempts=80,
    )
    assert certificate["status"] == "certified"
    assert certificate["certified_to_p"] == pytest.approx(0.3045)
    assert certificate["leaf_box_count"] == 2
    assert certificate["stage_reports"][0]["output_boxes"] == 2
    assert certificate["worst_margin"] > 0.0


def test_load_union_leaf_boxes_accepts_staged_certificate(tmp_path) -> None:
    """Saved staged-union JSON should provide reusable leaf boxes."""
    path = tmp_path / "union.json"
    path.write_text(
        json.dumps(
            {
                "staged_union_p_tube_certificate": {
                    "leaf_boxes": [
                        {"low": [1.0, 2.0, 3.0, 4.0], "high": [1.5, 2.5, 3.5, 4.5]}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    boxes = tail._load_union_leaf_boxes(path)
    assert boxes == [((1.0, 2.0, 3.0, 4.0), (1.5, 2.5, 3.5, 4.5))]


def test_adaptive_union_p_tube_certificate_splits_failed_component(monkeypatch) -> None:
    """Adaptive finite unions should split failed boxes until they certify."""
    calls = []

    def fake_certificate(start_p, end_p, start_low, start_high, **kwargs):
        assert kwargs["use_cancellation_p_prime"] is True
        width = start_high[0] - start_low[0]
        calls.append((start_low, start_high))
        if width > 0.6:
            return {
                "status": "failed",
                "certified_to_p": start_p,
                "blocks_certified": 0,
                "tuning_attempt_count": 1,
                "worst_margin": -0.5,
                "failing_block": {"failing_face": {"side": "upper", "component": 0}},
            }
        return {
            "status": "certified",
            "blocks_certified": 1,
            "tuning_attempt_count": 2,
            "worst_margin": 0.25,
            "worst_face": {"side": "upper", "component": 0},
            "end_box": {"low": list(start_low), "high": list(start_high)},
        }

    monkeypatch.setattr(tail, "tuned_p_tube_from_box_certificate", fake_certificate)
    certificate = tail.adaptive_union_p_tube_certificate(
        start_p=0.4,
        end_p=0.39,
        source_boxes=(((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)),),
        max_depth=1,
        use_cancellation_p_prime=True,
    )
    assert certificate["status"] == "certified"
    assert certificate["use_cancellation_p_prime"] is True
    assert certificate["split_count"] == 1
    assert certificate["certified_leaf_box_count"] == 8
    assert certificate["failed_leaf_box_count"] == 0
    assert certificate["blocks_certified"] == 8
    assert certificate["worst_margin"] == 0.25
    assert certificate["worst_failed_attempt_margin"] == -0.5
    assert len(calls) == 9


def test_adaptive_union_p_tube_certificate_reports_depth_failure(monkeypatch) -> None:
    """Uncertified leaves at max depth should be ordinary failed payloads."""

    def fake_certificate(start_p, _end_p, _start_low, _start_high, **_kwargs):
        return {
            "status": "failed",
            "certified_to_p": start_p,
            "blocks_certified": 0,
            "tuning_attempt_count": 1,
            "worst_margin": -0.5,
            "failing_block": {"failing_face": {"side": "lower", "component": 2}},
        }

    monkeypatch.setattr(tail, "tuned_p_tube_from_box_certificate", fake_certificate)
    certificate = tail.adaptive_union_p_tube_certificate(
        start_p=0.4,
        end_p=0.39,
        source_boxes=(((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)),),
        max_depth=0,
    )
    assert certificate["status"] == "failed"
    assert certificate["failed_leaf_box_count"] == 1
    assert certificate["failed_leaves"][0]["failing_face"]["component"] == 2


def test_adaptive_union_p_tube_certificate_reports_processed_budget(monkeypatch) -> None:
    """Processed-box limits should produce a valid failed certificate."""

    def fake_certificate(start_p, _end_p, _start_low, _start_high, **_kwargs):
        return {
            "status": "failed",
            "certified_to_p": start_p,
            "blocks_certified": 0,
            "tuning_attempt_count": 1,
            "worst_margin": -0.5,
            "failing_block": {"failing_face": {"side": "upper", "component": 0}},
        }

    monkeypatch.setattr(tail, "tuned_p_tube_from_box_certificate", fake_certificate)
    certificate = tail.adaptive_union_p_tube_certificate(
        start_p=0.4,
        end_p=0.39,
        source_boxes=(((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)),),
        max_depth=4,
        max_processed_boxes=1,
    )
    assert certificate["status"] == "failed"
    assert certificate["stopped_reason"] == "max_processed_boxes_exceeded"
    assert certificate["remaining_queue_count"] == 8


def test_adaptive_carried_c_union_p_tube_certificate_preserves_5d_boxes(monkeypatch) -> None:
    """The carried-C adaptive union should emit reusable five-dimensional leaves."""
    calls = []

    def fake_certificate(start_p, end_p, start_low, start_high, **_kwargs):
        calls.append((start_p, end_p, start_low, start_high))
        return {
            "status": "certified",
            "blocks_certified": 1,
            "tuning_attempt_count": 2,
            "worst_margin": 0.5,
            "worst_face": {"side": "lower", "component": 4},
            "end_box_5d": {"low": list(start_low), "high": list(start_high)},
        }

    monkeypatch.setattr(tail, "tuned_carried_c_p_tube_from_box_certificate", fake_certificate)
    certificate = tail.adaptive_carried_c_union_p_tube_certificate(
        start_p=0.39,
        end_p=0.3885,
        source_boxes=(((3.2, 2.0, 0.008, -2.0), (3.9, 20.0, 0.03, 0.5)),),
        max_depth=0,
    )
    assert certificate["status"] == "certified"
    assert certificate["certified_leaf_box_count"] == 1
    assert len(certificate["leaf_boxes_5d"][0]["low"]) == 5
    assert certificate["leaf_boxes_5d"][0]["low"][4] > 0.0
    assert calls[0][2] == tuple(certificate["leaf_boxes_5d"][0]["low"])


def test_adaptive_carried_c_union_can_split_x3_on_x2_failure(monkeypatch) -> None:
    """The late-tail carried-C splitter can optionally include x3 for x2 failures."""

    def fake_certificate(start_p, _end_p, _start_low, _start_high, **_kwargs):
        return {
            "status": "failed",
            "certified_to_p": start_p,
            "blocks_certified": 0,
            "tuning_attempt_count": 1,
            "worst_margin": -0.5,
            "failing_block": {"failing_face": {"side": "upper", "component": 2}},
        }

    monkeypatch.setattr(tail, "tuned_carried_c_p_tube_from_box_certificate", fake_certificate)
    source_box = (((0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0, 1.0)),)
    default_certificate = tail.adaptive_carried_c_union_p_tube_certificate(
        start_p=0.39,
        end_p=0.3885,
        source_boxes=source_box,
        max_depth=1,
        max_processed_boxes=1,
    )
    split_x3_certificate = tail.adaptive_carried_c_union_p_tube_certificate(
        start_p=0.39,
        end_p=0.3885,
        source_boxes=source_box,
        max_depth=1,
        max_processed_boxes=1,
        split_x3_on_x2_failure=True,
    )
    assert default_certificate["remaining_queue_count"] == 8
    assert split_x3_certificate["remaining_queue_count"] == 16
    assert default_certificate["remaining_queue_preview"][0]["split_history"][0]["split_components"] == [1, 2, 4]
    assert split_x3_certificate["remaining_queue_preview"][0]["split_history"][0]["split_components"] == [1, 2, 3, 4]
    assert split_x3_certificate["split_x3_on_x2_failure"] is True


def test_adaptive_carried_c_union_can_split_x3_on_x0_failure(monkeypatch) -> None:
    """The late-tail carried-C splitter can optionally include x3 for p-face failures."""

    def fake_certificate(start_p, _end_p, _start_low, _start_high, **_kwargs):
        return {
            "status": "failed",
            "certified_to_p": start_p,
            "blocks_certified": 0,
            "tuning_attempt_count": 1,
            "worst_margin": -0.5,
            "failing_block": {"failing_face": {"side": "lower", "component": 0}},
        }

    monkeypatch.setattr(tail, "tuned_carried_c_p_tube_from_box_certificate", fake_certificate)
    source_box = (((0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0, 1.0)),)
    default_certificate = tail.adaptive_carried_c_union_p_tube_certificate(
        start_p=0.39,
        end_p=0.3885,
        source_boxes=source_box,
        max_depth=1,
        max_processed_boxes=1,
    )
    split_x3_certificate = tail.adaptive_carried_c_union_p_tube_certificate(
        start_p=0.39,
        end_p=0.3885,
        source_boxes=source_box,
        max_depth=1,
        max_processed_boxes=1,
        split_x3_on_x0_failure=True,
    )
    assert default_certificate["remaining_queue_count"] == 16
    assert split_x3_certificate["remaining_queue_count"] == 32
    assert default_certificate["remaining_queue_preview"][0]["split_history"][0]["split_components"] == [0, 1, 2, 4]
    assert split_x3_certificate["remaining_queue_preview"][0]["split_history"][0]["split_components"] == [0, 1, 2, 3, 4]
    assert split_x3_certificate["split_x3_on_x0_failure"] is True


def test_sampled_carried_c_p_tube_certificate_chains_blocks(monkeypatch) -> None:
    """The sampled carried-C certificate should compose block certificates."""
    calls = []

    def fake_state_at_p(source, target_p, a, **_kwargs):
        offset = 0.0 if source == "limit" else (1e-9 if a > 0 else -1e-9)
        return (3.0 + target_p, 6.0 + offset, 0.01, -0.2)

    def fake_block(start_p, step_size, _block_steps, _candidate_a, start_samples, start_low, start_high, **_kwargs):
        calls.append(start_p)
        end_samples = tuple(
            (
                sample[0] + 0.01,
                sample[1] + 0.02,
                sample[2],
                sample[3] - 0.03,
                sample[4] + 0.001,
            )
            for sample in start_samples
        )
        return {
            "status": "certified",
            "end_samples": end_samples,
            "end_box_5d": {"low": list(start_low), "high": list(start_high)},
            "worst_margin": 0.25,
            "worst_face": {"side": "lower", "component": 0},
            "tuning_attempts": [{"status": "certified"}],
        }

    events = []
    monkeypatch.setattr(tail, "scaled_state_at_p", fake_state_at_p)
    monkeypatch.setattr(tail, "tuned_carried_c_p_tube_block_certificate", fake_block)
    certificate = tail.sampled_carried_c_p_tube_certificate(
        start_p=0.65,
        end_p=0.649,
        step_size=0.0005,
        radius0=(1e-5, 1e-4, 1e-6, 1e-5, 1e-5),
        progress_callback=events.append,
        progress_every=1,
    )
    assert certificate["status"] == "certified"
    assert certificate["blocks_certified"] == 2
    assert certificate["certified_to_p"] == pytest.approx(0.649)
    assert certificate["tuning_attempt_count"] == 2
    assert calls == pytest.approx([0.65, 0.6495])
    assert len(events) == 2
    assert certificate["conditional"] == "sampled_start_box_contains_true_state_and_C"


def test_cli_sampled_carried_c_p_tube_accepts_tight_profile_set(monkeypatch, capsys) -> None:
    """The sampled carried-C CLI should expose the tight p=0.3255 profiles."""

    def fake_sampled(**kwargs):
        assert kwargs["profiles"] == tail.TIGHT_SAMPLED_CARRIED_C_P_TUBE_PROFILES
        return {
            "status": "certified",
            "candidate_A": tail.DEFAULT_TUBE_CANDIDATE_A,
            "start_p": 0.65,
            "end_p": 0.6495,
            "certified_to_p": 0.6495,
            "blocks_certified": 1,
            "tuning_attempt_count": 1,
            "worst_margin": 0.1,
            "conditional": "sampled_start_box_contains_true_state_and_C",
            "end_box_5d": {
                "low": [1.0, 2.0, 0.1, -0.2, 0.2],
                "high": [1.1, 2.1, 0.2, -0.1, 0.5],
            },
        }

    monkeypatch.setattr(tail, "sampled_carried_c_p_tube_certificate", fake_sampled)
    tail.main(
        [
            "--sampled-carried-c-p-tube-check",
            "--sampled-carried-c-p-tube-profile-set",
            "tight",
            "--sampled-carried-c-p-tube-end",
            "0.6495",
        ]
    )
    output = capsys.readouterr().out
    assert "sampled carried-C p-tube certificate: status=certified" in output


def test_cli_carried_c_p_tube_from_box_loads_source_json(tmp_path, monkeypatch, capsys) -> None:
    """The CLI should continue a tuned carried-C p-tube from a saved 5D box."""
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(
            {
                "sampled_carried_c_p_tube_certificate": {
                    "certified_to_p": 0.4,
                    "end_box_5d": {
                        "low": [1.0, 2.0, 0.1, -0.2, 0.2],
                        "high": [1.1, 2.1, 0.2, -0.1, 0.5],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_from_box(**kwargs):
        assert kwargs["start_p"] == pytest.approx(0.4)
        assert kwargs["end_p"] == pytest.approx(0.3995)
        assert kwargs["start_low"] == (1.0, 2.0, 0.1, -0.2, 0.2)
        assert kwargs["start_high"] == (1.1, 2.1, 0.2, -0.1, 0.5)
        return {
            "status": "certified",
            "candidate_A": tail.DEFAULT_TUBE_CANDIDATE_A,
            "start_p": kwargs["start_p"],
            "end_p": kwargs["end_p"],
            "certified_to_p": kwargs["end_p"],
            "blocks_certified": 1,
            "tuning_attempt_count": 3,
            "worst_margin": 0.1,
            "end_box_5d": {
                "low": [1.0, 2.0, 0.1, -0.2, 0.2],
                "high": [1.1, 2.1, 0.2, -0.1, 0.5],
            },
        }

    monkeypatch.setattr(tail, "tuned_carried_c_p_tube_from_box_certificate", fake_from_box)
    tail.main(
        [
            "--carried-c-p-tube-from-box-check",
            "--carried-c-p-tube-source-json",
            str(source_path),
            "--carried-c-p-tube-start",
            "0.4",
            "--carried-c-p-tube-end",
            "0.3995",
        ]
    )
    output = capsys.readouterr().out
    assert "carried-C p-tube from box certificate: status=certified" in output
    assert "source=sampled_carried_c_p_tube_certificate.end_box_5d" in output


def test_cli_adaptive_union_p_tube_check_reports_certificate(tmp_path, monkeypatch, capsys) -> None:
    """The CLI should load source boxes and report adaptive union summaries."""
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps({"leaf_boxes": [{"low": [0.0, 0.0, 0.0, 0.0], "high": [1.0, 1.0, 1.0, 1.0]}]}),
        encoding="utf-8",
    )

    def fake_adaptive(**kwargs):
        assert kwargs["use_cancellation_p_prime"] is True
        return {
            "status": "certified",
            "candidate_A": tail.DEFAULT_TUBE_CANDIDATE_A,
            "start_p": 0.4,
            "end_p": 0.39,
            "certified_to_p": 0.39,
            "source_box_count": 1,
            "certified_leaf_box_count": 1,
            "failed_leaf_box_count": 0,
            "remaining_queue_count": 0,
            "split_count": 0,
            "processed_boxes": 1,
            "blocks_certified": 20,
            "tuning_attempt_count": 20,
            "worst_margin": 1.0,
            "use_cancellation_p_prime": True,
            "leaf_boxes": [{"low": [0, 0, 0, 0], "high": [1, 1, 1, 1]}],
            "failed_leaves": [],
        }

    monkeypatch.setattr(tail, "adaptive_union_p_tube_certificate", fake_adaptive)
    tail.main(
        [
            "--adaptive-union-p-tube-check",
            "--adaptive-union-p-tube-source-json",
            str(source_path),
            "--p-tube-start",
            "0.4",
            "--p-tube-end",
            "0.39",
            "--p-tube-cancellation-prime",
        ]
    )
    output = capsys.readouterr().out
    assert "adaptive union p-tube certificate: status=certified" in output
    assert "source_boxes=1" in output


def test_affine_p_corridor_certificate_checks_short_slab() -> None:
    """The affine p-time corridor verifier should certify a small first slab."""
    certificate = tail.affine_p_corridor_certificate(
        start_p=0.25,
        end_p=0.2495,
        step_size=5e-4,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
    )
    assert certificate["status"] == "certified"
    assert certificate["source_box_contained"] is True
    assert certificate["certified_to_p"] == 0.2495
    assert certificate["worst_margin"] > 0.0


def test_affine_p_corridor_tuner_reports_best_run() -> None:
    """The corridor tuner should summarize slope-scan results."""
    tuning = tail.tune_affine_p_corridor(
        x2_lower_slopes=(tail.DEFAULT_P_CORRIDOR_LOWER_SLOPE[2],),
        x1_upper_slopes=(tail.DEFAULT_P_CORRIDOR_UPPER_SLOPE[1],),
        start_p=0.25,
        end_p=0.2495,
        step_size=5e-4,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        max_runs=1,
    )
    assert tuning["status"] == "completed"
    assert tuning["runs"] == 1
    assert tuning["best"]["status"] == "certified"
    assert tuning["best"]["certified_to_p"] == 0.2495


def test_affine_p_corridor_tuner_rejects_empty_run_budget() -> None:
    """The tuner should reject max_runs values that cannot run a scan."""
    with pytest.raises(ValueError, match="max_runs"):
        tail.tune_affine_p_corridor(max_runs=0)


def test_terminal_barrier_takeover_certifies_conditional_wall() -> None:
    """The terminal x3-wall takeover should certify a short conditional range."""
    certificate = tail.terminal_barrier_takeover_certificate(
        p_min=tail.DEFAULT_TERMINAL_TAKEOVER_P_MIN,
        p_step=tail.DEFAULT_TERMINAL_TAKEOVER_P_STEP,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
    )
    assert certificate["status"] == "certified_conditional"
    assert certificate["source_box_contained"] is True
    assert certificate["source_below_wall"] is True
    assert certificate["source_x2_floor"] is True
    assert certificate["x3_zero_margin"] > 0.0
    assert certificate["worst_margin"] > 0.0
    assert certificate["small_p_tail"]["p_prime_negative_coefficient_margin"] > 0.0
    assert certificate["small_p_tail"]["x3_prime_negative_margin_at_p_min"] > 0.0


def test_stronger_terminal_wall_allows_lower_x1_floor() -> None:
    """A negative x3 wall can certify even when the x3=0 threshold is negative."""
    certificate = tail.terminal_barrier_takeover_certificate(
        p_start=tail.DEFAULT_LATE_TAIL_TAKEOVER_START,
        box_low=tail.DEFAULT_LATE_TAIL_TAKEOVER_BOX_LOW,
        box_high=tail.DEFAULT_LATE_TAIL_TAKEOVER_BOX_HIGH,
        x3_wall=tail.DEFAULT_LATE_TAIL_TAKEOVER_X3_WALL,
        source_box_low=tail.DEFAULT_PIECEWISE_CORRIDOR_KNOTS[-1][1],
        source_box_high=tail.DEFAULT_PIECEWISE_CORRIDOR_KNOTS[-1][2],
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
    )
    assert certificate["status"] == "certified_conditional"
    assert certificate["x3_zero_margin"] < 0.0
    assert certificate["worst_margin"] > 0.0


def test_frontier_continuation_certifies_one_block() -> None:
    """Continuation from the certified p-frontier should support short checks."""
    certificate = tail.p_tube_frontier_continuation_certificate(
        end_p=tail.DEFAULT_P_TUBE_END - tail.DEFAULT_P_TUBE_STEP,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
    )
    assert certificate["status"] == "certified"
    assert certificate["blocks_certified"] == 1
    assert certificate["certified_to_p"] == tail.DEFAULT_P_TUBE_END - tail.DEFAULT_P_TUBE_STEP
    assert certificate["worst_margin"] > 0.0


def test_piecewise_affine_p_corridor_certifies_single_segment() -> None:
    """The piecewise corridor helper should certify a simple short segment."""
    knots = (
        (
            0.25,
            tail.DEFAULT_P_CORRIDOR_LOWER_START,
            tail.DEFAULT_P_CORRIDOR_UPPER_START,
        ),
        (
            0.2495,
            tail._affine_barrier_value(
                tail.DEFAULT_P_CORRIDOR_LOWER_START,
                tail.DEFAULT_P_CORRIDOR_LOWER_SLOPE,
                0.25,
                0.2495,
            ),
            tail._affine_barrier_value(
                tail.DEFAULT_P_CORRIDOR_UPPER_START,
                tail.DEFAULT_P_CORRIDOR_UPPER_SLOPE,
                0.25,
                0.2495,
            ),
        ),
    )
    certificate = tail.piecewise_affine_p_corridor_certificate(
        knots=knots,
        step_size=5e-4,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
    )
    assert certificate["status"] == "certified"
    assert certificate["segments_certified"] == 1
    assert certificate["certified_to_p"] == 0.2495


def test_hybrid_p_frontier_handoff_certifies_short_composition() -> None:
    """The hybrid p-tube plus affine-corridor handoff should be composable."""
    certificate = tail.hybrid_p_frontier_handoff_certificate(
        start_p=0.325,
        tube_end_p=0.32495,
        frontier_p=0.3249,
        entry_time=3.5,
        step_size=5e-5,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        frontier_low=(3.505, 6.52, 0.0101, -0.386),
        frontier_high=(3.507, 6.55, 0.0102, -0.384),
        seed_step_size=1e-3,
    )
    assert certificate["status"] == "certified_conditional"
    assert certificate["p_tube"]["status"] == "certified"
    assert certificate["affine_corridor"]["status"] == "certified"
    assert certificate["certified_to_p"] == 0.3249
    assert certificate["conditional"] == "p_start_slice_box_contains_true_state"


def test_automatic_p_barrier_corridor_certifies_short_step() -> None:
    """The automatic p-barrier update should produce verified affine steps."""
    certificate = tail.automatic_p_barrier_corridor_certificate(
        start_p=0.25,
        end_p=0.2499,
        source_box_low=tail.DEFAULT_HYBRID_HANDOFF_FRONTIER_LOW,
        source_box_high=tail.DEFAULT_HYBRID_HANDOFF_FRONTIER_HIGH,
        step_size=5e-5,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
    )
    assert certificate["status"] == "certified"
    assert certificate["steps_certified"] == 2
    assert certificate["certified_to_p"] == 0.2499
    assert certificate["worst_margin"] > 0.0


def test_automatic_p_barrier_corridor_accepts_component_safety() -> None:
    """The automatic p-corridor should allow component-specific slack."""
    certificate = tail.automatic_p_barrier_corridor_certificate(
        start_p=0.25,
        end_p=0.24995,
        source_box_low=tail.DEFAULT_HYBRID_HANDOFF_FRONTIER_LOW,
        source_box_high=tail.DEFAULT_HYBRID_HANDOFF_FRONTIER_HIGH,
        step_size=5e-5,
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        safety=(1e-4, 2e-4, 3e-4, 4e-4),
    )
    assert certificate["status"] == "certified"
    assert certificate["safety"] == [1e-4, 2e-4, 3e-4, 4e-4]


def test_affine_carried_c_p_corridor_certificate_checks_short_slab(monkeypatch) -> None:
    """The carried-C affine corridor should check all five component faces."""

    def fake_rhs(_p_interval, _z_interval, _b_interval, _component, _subdivisions, _p_subdivisions):
        return (0.0, 1.0)

    monkeypatch.setattr(tail, "_subdivided_interval_carried_c_p_time_rhs_component", fake_rhs)
    certificate = tail.affine_carried_c_p_corridor_certificate(
        start_p=0.4,
        end_p=0.3995,
        step_size=5e-4,
        lower_start=(1.0, 2.0, 0.1, -0.2, 0.2),
        upper_start=(1.1, 2.1, 0.2, -0.1, 0.5),
        lower_slope=(1.25, 1.25, 1.25, 1.25, 1.25),
        upper_slope=(-0.25, -0.25, -0.25, -0.25, -0.25),
        subdivisions=(1, 1, 1, 1, 1),
        p_subdivisions=1,
    )
    assert certificate["status"] == "certified"
    assert certificate["steps_certified"] == 1
    assert certificate["certified_to_p"] == pytest.approx(0.3995)
    assert certificate["worst_margin"] == pytest.approx(0.25)
    assert "end_box_5d" in certificate


def test_automatic_carried_c_p_barrier_corridor_certifies_short_chain(monkeypatch) -> None:
    """The automatic carried-C corridor should compose verified affine slabs."""
    calls = []

    def fake_rhs(_p_interval, _z_interval, _b_interval, component, _subdivisions, _p_subdivisions):
        calls.append(component)
        return (-1.0, 1.0)

    monkeypatch.setattr(tail, "_subdivided_interval_carried_c_p_time_rhs_component", fake_rhs)
    certificate = tail.automatic_carried_c_p_barrier_corridor_certificate(
        start_p=0.4,
        end_p=0.399,
        step_size=5e-4,
        source_box_low=(1.0, 2.0, 0.1, -0.2, 0.2),
        source_box_high=(1.1, 2.1, 0.2, -0.1, 0.5),
        safety=(0.1, 0.1, 0.1, 0.1, 0.1),
        subdivisions=(1, 1, 1, 1, 1),
        p_subdivisions=1,
    )
    assert certificate["status"] == "certified"
    assert certificate["steps_certified"] == 2
    assert certificate["certified_to_p"] == pytest.approx(0.399)
    assert certificate["safety"] == [0.1, 0.1, 0.1, 0.1, 0.1]
    assert set(calls) == {0, 1, 2, 3, 4}


def test_carried_c_p_wall_certificate_certifies_short_x2_wall() -> None:
    """The carried-C p-wall checker should certify the positive x2 wall locally."""
    certificate = tail.carried_c_p_wall_certificate(
        start_p=0.29,
        end_p=0.285,
        p_step=0.005,
        component=2,
        side="lower",
        wall_value=0.0,
    )
    assert certificate["status"] == "certified_conditional"
    assert certificate["certified_to_p"] == pytest.approx(0.285)
    assert certificate["worst_margin"] > 0.0


def test_carried_c_p_wall_certificate_accepts_c_wall(monkeypatch) -> None:
    """The carried-C p-wall checker should also handle the carried C component."""
    calls = []

    def fake_rhs(p_interval, z_interval, b_interval, component, subdivisions, p_subdivisions):
        calls.append((component, len(z_interval)))
        return (-0.2, -0.1)

    monkeypatch.setattr(tail, "_subdivided_interval_carried_c_p_time_rhs_component", fake_rhs)
    certificate = tail.carried_c_p_wall_certificate(
        start_p=0.29,
        end_p=0.285,
        p_step=0.005,
        box_low=(3.4, 1.0, 0.0, -1.6, 0.0),
        box_high=(3.8, 15.0, 0.02, 0.0, 0.3),
        source_box_low=(3.5, 2.0, 0.001, -1.0, 0.01),
        source_box_high=(3.6, 10.0, 0.01, -0.1, 0.2),
        component=4,
        side="lower",
        wall_value=0.0,
    )
    assert certificate["status"] == "certified_conditional"
    assert certificate["component"] == 4
    assert certificate["source_box_contained"] is True
    assert calls == [(4, 5)]


def test_cli_carried_c_p_wall_check_reports_certificate(monkeypatch, capsys) -> None:
    """The CLI should expose carried-C p-wall diagnostics."""

    def fake_wall(**kwargs):
        assert kwargs["component"] == 2
        assert kwargs["side"] == "lower"
        return {
            "status": "certified_conditional",
            "candidate_A": tail.DEFAULT_TUBE_CANDIDATE_A,
            "component": kwargs["component"],
            "side": kwargs["side"],
            "wall_value": kwargs["wall_value"],
            "start_p": kwargs["start_p"],
            "end_p": kwargs["end_p"],
            "certified_to_p": kwargs["end_p"],
            "steps_certified": 1,
            "worst_margin": 0.1,
            "source_box_contained": True,
        }

    monkeypatch.setattr(tail, "carried_c_p_wall_certificate", fake_wall)
    tail.main(
        [
            "--carried-c-p-wall-check",
            "--carried-c-p-wall-start",
            "0.29",
            "--carried-c-p-wall-end",
            "0.285",
            "--carried-c-p-wall-component",
            "2",
            "--carried-c-p-wall-side",
            "lower",
        ]
    )
    output = capsys.readouterr().out
    assert "carried-C p-wall certificate: status=certified_conditional" in output
    assert "component=2" in output


def test_cli_carried_c_p_wall_keeps_saved_c_source(tmp_path, monkeypatch, capsys) -> None:
    """The C-wall CLI path should keep five-dimensional source boxes."""
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(
            {
                "automatic_carried_c_p_corridor_certificate": {
                    "status": "certified",
                    "end_p": 0.29,
                    "end_box_5d": {
                        "low": [3.5, 2.0, 0.001, -1.0, 0.01],
                        "high": [3.6, 10.0, 0.01, -0.1, 0.2],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_wall(**kwargs):
        assert kwargs["component"] == 4
        assert len(kwargs["box_low"]) == 5
        assert len(kwargs["source_box_low"]) == 5
        return {
            "status": "certified_conditional",
            "candidate_A": tail.DEFAULT_TUBE_CANDIDATE_A,
            "component": kwargs["component"],
            "side": kwargs["side"],
            "wall_value": kwargs["wall_value"],
            "start_p": kwargs["start_p"],
            "end_p": kwargs["end_p"],
            "certified_to_p": kwargs["end_p"],
            "steps_certified": 1,
            "worst_margin": 0.1,
            "source_box_contained": True,
        }

    monkeypatch.setattr(tail, "carried_c_p_wall_certificate", fake_wall)
    tail.main(
        [
            "--carried-c-p-wall-check",
            "--carried-c-p-wall-start",
            "0.29",
            "--carried-c-p-wall-end",
            "0.285",
            "--carried-c-p-wall-component",
            "4",
            "--carried-c-p-wall-side",
            "lower",
            "--carried-c-p-wall-box-low",
            "3.4,1.0,0.0,-1.6,0.0",
            "--carried-c-p-wall-box-high",
            "3.8,15.0,0.02,0.0,0.3",
            "--carried-c-p-wall-source-json",
            str(source_path),
        ]
    )
    output = capsys.readouterr().out
    assert "carried-C p-wall certificate: status=certified_conditional" in output
    assert "component=4" in output


def test_load_carried_c_corridor_source_box_from_sampled_json(tmp_path) -> None:
    """The carried-C corridor source loader should accept sampled tube output."""
    source_path = tmp_path / "sampled.json"
    source_path.write_text(
        json.dumps(
            {
                "sampled_carried_c_p_tube_certificate": {
                    "certified_to_p": 0.325,
                    "end_box_5d": {
                        "low": [1.0, 2.0, 0.1, -0.2, 0.2],
                        "high": [1.1, 2.1, 0.2, -0.1, 0.5],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    low, high, source_p, source_kind = tail._load_carried_c_corridor_source_box(source_path)
    assert low == (1.0, 2.0, 0.1, -0.2, 0.2)
    assert high == (1.1, 2.1, 0.2, -0.1, 0.5)
    assert source_p == pytest.approx(0.325)
    assert source_kind == "sampled_carried_c_p_tube_certificate.end_box_5d"


def test_cli_automatic_carried_c_p_corridor_loads_source_json(tmp_path, monkeypatch, capsys) -> None:
    """The CLI should run the automatic carried-C corridor from a saved source."""
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(
            {
                "sampled_carried_c_p_tube_certificate": {
                    "certified_to_p": 0.4,
                    "end_box_5d": {
                        "low": [1.0, 2.0, 0.1, -0.2, 0.2],
                        "high": [1.1, 2.1, 0.2, -0.1, 0.5],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_automatic(**kwargs):
        assert kwargs["source_box_low"] == (1.0, 2.0, 0.1, -0.2, 0.2)
        assert kwargs["source_box_high"] == (1.1, 2.1, 0.2, -0.1, 0.5)
        return {
            "status": "certified",
            "candidate_A": tail.DEFAULT_TUBE_CANDIDATE_A,
            "start_p": 0.4,
            "end_p": 0.3995,
            "certified_to_p": 0.3995,
            "steps_certified": 1,
            "steps": 1,
            "worst_margin": 0.1,
            "end_box_5d": {
                "low": [1.0, 2.0, 0.1, -0.2, 0.2],
                "high": [1.1, 2.1, 0.2, -0.1, 0.5],
            },
        }

    monkeypatch.setattr(tail, "automatic_carried_c_p_barrier_corridor_certificate", fake_automatic)
    tail.main(
        [
            "--automatic-carried-c-p-corridor-check",
            "--carried-c-p-corridor-source-json",
            str(source_path),
            "--carried-c-p-corridor-start",
            "0.4",
            "--carried-c-p-corridor-end",
            "0.3995",
        ]
    )
    output = capsys.readouterr().out
    assert "automatic carried-C p-corridor certificate: status=certified" in output
    assert "source=sampled_carried_c_p_tube_certificate.end_box_5d" in output


def test_cli_tuned_p_tube_check_reports_certificate(capsys) -> None:
    """The CLI should expose the tuned transformed p-time tube."""
    tail.main(
        [
            "--tuned-p-tube-check",
            "--p-tube-start",
            "0.305",
            "--p-tube-end",
            "0.3045",
            "--p-tube-step",
            "0.0005",
        ]
    )
    output = capsys.readouterr().out
    assert "tuned segmented p-tube certificate: status=certified" in output
    assert "certified_to_p=0.3045" in output


def test_p_start_slice_from_support_certificate_reaches_start_box() -> None:
    """The t-time support tube should cross into the hybrid p-start slice."""
    certificate = tail.p_start_slice_from_support_certificate(
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
    )
    assert certificate["status"] == "certified_conditional"
    assert certificate["before_above_target"] is True
    assert certificate["after_below_target"] is True
    assert certificate["crossing_slab_contained_in_start_slice"] is True


def test_p_start_slice_from_support_allows_larger_support_radius() -> None:
    """The support bridge should tolerate the documented 10x support box."""
    support_radius = tuple(10.0 * value for value in tail.DEFAULT_SUPPORT_TUBE_RADIUS)
    certificate = tail.p_start_slice_from_support_certificate(
        candidate_a=tail.DEFAULT_TUBE_CANDIDATE_A,
        support_radius0=support_radius,
    )
    assert certificate["status"] == "certified_conditional"
    assert tuple(certificate["support_radius0"]) == support_radius
    assert certificate["before_above_target"] is True
    assert certificate["after_below_target"] is True
    assert certificate["crossing_slab_contained_in_start_slice"] is True
