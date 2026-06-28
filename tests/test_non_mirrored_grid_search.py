"""Tests for the calibrated non-mirrored grid scout search."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from mpmath import mp

from experiments.berger_space import non_mirrored_grid_search as grid
from experiments.shared.non_mirrored_common import _event, _write_jsonl_event
from solver.two_sided_shooting import BASE_TWO_SIDED_POINT, TwoSidedResidualResult, params_from_two_sided_scaled


def _assert_close(left: mp.mpf, right: str) -> None:
    """Assert high-precision numerical equality for grid coordinates."""
    assert abs(left - mp.mpf(right)) < mp.mpf("1e-70")


def _fake_result(
    point,
    config,
    norm: str = "1e-3",
    failure: str | None = None,
    *,
    base_params=grid.DEFAULT_PARAMS,
) -> TwoSidedResidualResult:
    """Return a synthetic residual result for fast scout tests."""
    params, concrete_config = params_from_two_sided_scaled(point, base_params=base_params, template_config=config)
    residual_norm = mp.mpf(norm)
    return TwoSidedResidualResult(
        point,
        params,
        concrete_config,
        tuple(residual_norm for _ in range(8)),
        residual_norm,
        None,
        None,
        None,
        None,
        (0, 0),
        {},
        failure,
    )


def test_near_grid_uses_calibrated_covering_spacing() -> None:
    """The default near grid should have max mesh width 0.4 and 103,680 seeds."""
    with mp.workdps(80):
        axes = grid._grid_axes(grid.NEAR_GRID, grid.DEFAULT_GRID_SPACING)
        assert [len(axis) for axis in axes] == [4, 4, 9, 4, 4, 9, 5]
        assert grid._grid_seed_count() == 103_680
        assert max(grid._axis_spacing(axis) for axis in axes) <= grid.DEFAULT_GRID_SPACING
        assert axes[0][0] == mp.mpf("-0.6")
        assert axes[0][-1] == mp.mpf("0.6")
        assert axes[2][0] == mp.mpf("-1.5")
        assert axes[2][-1] == mp.mpf("1.5")


def test_symmetric_alpha_omega_grid_is_symmetric_in_physical_odd_coefficients() -> None:
    """The follow-up grid should remove the alpha/omega sign bias in physical space."""
    with mp.workdps(80):
        region = grid.SYMMETRIC_ALPHA_OMEGA_GRID
        axes = grid._grid_axes(region, grid.DEFAULT_GRID_SPACING)
        metadata = grid._grid_metadata(region.name, grid.DEFAULT_GRID_SPACING)
        radius = mp.sqrt(5) / 20
        assert [len(axis) for axis in axes] == [4, 4, 14, 4, 4, 14, 5]
        assert grid._grid_seed_count(region.name) == 250_880
        assert axes[2][0] == mp.mpf("-3.5")
        assert axes[2][-1] == mp.mpf("1.5")
        assert axes[5][0] == mp.mpf("-3.5")
        assert axes[5][-1] == mp.mpf("1.5")
        for actual, expected in zip(metadata["physical_odd_bounds"]["alpha"], (-radius, radius)):
            assert abs(mp.mpf(actual) - expected) < mp.mpf("1e-70")
        for actual, expected in zip(metadata["physical_odd_bounds"]["omega"], (-radius, radius)):
            assert abs(mp.mpf(actual) - expected) < mp.mpf("1e-70")


def test_positive_ac_grid_uses_second_real_endpoint_region() -> None:
    """The exploratory region should scan ac > 0 with the same symmetric odd box."""
    with mp.workdps(80):
        region = grid.POSITIVE_AC_GRID
        axes = grid._grid_axes(region, grid.DEFAULT_GRID_SPACING)
        metadata = grid._grid_metadata(region.name, grid.DEFAULT_GRID_SPACING)
        left = metadata["base_params"]["left"]
        right = metadata["base_params"]["right"]
        radius = mp.sqrt(5) / 20
        assert [len(axis) for axis in axes] == [4, 4, 14, 4, 4, 14, 5]
        assert grid._grid_seed_count(region.name) == 250_880
        assert mp.mpf(left["a"]) > 0
        assert mp.mpf(left["c"]) > 0
        assert 3 * mp.mpf(left["a"]) - mp.mpf(left["c"]) > 0
        assert mp.mpf(right["d"]) < 0
        assert mp.mpf(right["f"]) < 0
        for actual, expected in zip(metadata["physical_odd_bounds"]["alpha"], (-radius, radius)):
            assert abs(mp.mpf(actual) - expected) < mp.mpf("1e-70")
        for actual, expected in zip(metadata["physical_odd_bounds"]["omega"], (-radius, radius)):
            assert abs(mp.mpf(actual) - expected) < mp.mpf("1e-70")


def test_negative_ac_grid_uses_remaining_real_endpoint_region() -> None:
    """The negative-ac region should scan a < 0, c < 0 while preserving 3a-c > 0."""
    with mp.workdps(80):
        region = grid.NEGATIVE_AC_GRID
        axes = grid._grid_axes(region, grid.DEFAULT_GRID_SPACING)
        metadata = grid._grid_metadata(region.name, grid.DEFAULT_GRID_SPACING)
        base = grid._base_params_for_region(region.name)
        left = metadata["base_params"]["left"]
        right = metadata["base_params"]["right"]
        radius = mp.sqrt(5) / 20
        assert [len(axis) for axis in axes] == [4, 4, 14, 4, 4, 14, 5]
        assert grid._grid_seed_count(region.name) == 250_880
        assert mp.mpf(left["a"]) < 0
        assert mp.mpf(left["c"]) < 0
        assert 3 * mp.mpf(left["a"]) - mp.mpf(left["c"]) > 0
        assert mp.mpf(right["d"]) > 0
        assert mp.mpf(right["f"]) > 0
        for u_left in region.bounds[0]:
            for v_left in region.bounds[1]:
                a = base.left.a * mp.exp(u_left)
                c = base.left.c * mp.exp(v_left)
                assert a < 0
                assert c < 0
                assert 3 * a - c > 0
        for u_right in region.bounds[3]:
            for v_right in region.bounds[4]:
                a = -base.right.d * mp.exp(u_right)
                c = -base.right.f * mp.exp(v_right)
                assert a < 0
                assert c < 0
                assert 3 * a - c > 0
        for actual, expected in zip(metadata["physical_odd_bounds"]["alpha"], (-radius, radius)):
            assert abs(mp.mpf(actual) - expected) < mp.mpf("1e-70")
        for actual, expected in zip(metadata["physical_odd_bounds"]["omega"], (-radius, radius)):
            assert abs(mp.mpf(actual) - expected) < mp.mpf("1e-70")


def test_mixed_mu_short_grid_uses_validated_opposite_mu_endpoint_branches() -> None:
    """The mixed-mu grid should record the endpoint-local p-sign branch and shorter interval box."""
    with mp.workdps(80):
        region = grid.MIXED_MU_SHORT_GRID
        axes = grid._grid_axes(region, grid.DEFAULT_GRID_SPACING)
        metadata = grid._grid_metadata(region.name, grid.DEFAULT_GRID_SPACING)
        base = grid._base_params_for_region(region.name)
        radius = mp.sqrt(5) / 20
        assert [len(axis) for axis in axes] == [4, 4, 14, 4, 4, 14, 5]
        assert grid._grid_seed_count(region.name) == 250_880
        assert axes[6][0] == mp.mpf("-2.0")
        assert axes[6][-1] == mp.mpf("-0.4")
        assert base.left_mu == 1
        assert base.right_mu == 1
        assert base.p_signs == (1, 1, 1)
        assert base.right_p_signs == (-1, 1, -1)
        assert metadata["base_params"]["p_signs"] == [1, 1, 1]
        assert metadata["base_params"]["right_p_signs"] == [-1, 1, -1]
        assert grid._scout_config_for_region(region.name).series_order == 6
        assert grid._scout_config_for_region(region.name).working_dps == 50
        for actual, expected in zip(metadata["physical_odd_bounds"]["alpha"], (-radius, radius)):
            assert abs(mp.mpf(actual) - expected) < mp.mpf("1e-70")
        for actual, expected in zip(metadata["physical_odd_bounds"]["omega"], (-radius, radius)):
            assert abs(mp.mpf(actual) - expected) < mp.mpf("1e-70")


def test_mixed_mu_boundary_grid_extends_low_scale_short_interval_tail() -> None:
    """The broader mixed-mu strip should follow the low-scale short-interval boundary signal."""
    with mp.workdps(80):
        region = grid.MIXED_MU_BOUNDARY_GRID
        spacing = mp.mpf("0.6")
        axes = grid._grid_axes(region, spacing)
        metadata = grid._grid_metadata(region.name, spacing)
        base = grid._base_params_for_region(region.name)
        radius = mp.sqrt(5) / 20
        assert [len(axis) for axis in axes] == [3, 3, 10, 3, 3, 10, 4]
        assert grid._grid_seed_count(region.name, spacing=spacing) == 32_400
        assert metadata["bounds"][0] == ["-1.8", "-0.6"]
        assert metadata["bounds"][2] == ["-3.5", "1.5"]
        assert metadata["bounds"][6] == ["-3.2", "-1.6"]
        assert axes[0][0] == mp.mpf("-1.8")
        assert axes[0][-1] == mp.mpf("-0.6")
        assert axes[6][0] == mp.mpf("-3.2")
        assert axes[6][-1] == mp.mpf("-1.6")
        assert base.left_mu == 1
        assert base.right_mu == 1
        assert base.p_signs == (1, 1, 1)
        assert base.right_p_signs == (-1, 1, -1)
        assert metadata["base_params"]["p_signs"] == [1, 1, 1]
        assert metadata["base_params"]["right_p_signs"] == [-1, 1, -1]
        assert grid._scout_config_for_region(region.name).series_order == 6
        assert grid._scout_config_for_region(region.name).working_dps == 50
        for actual, expected in zip(metadata["physical_odd_bounds"]["alpha"], (-radius, radius)):
            assert abs(mp.mpf(actual) - expected) < mp.mpf("1e-70")
        for actual, expected in zip(metadata["physical_odd_bounds"]["omega"], (-radius, radius)):
            assert abs(mp.mpf(actual) - expected) < mp.mpf("1e-70")


def test_positive_ac_boundary_grid_expands_safe_low_edge() -> None:
    """The expanded positive-ac strip should probe the observed low boundary without crossing 3a-c=0."""
    with mp.workdps(80):
        region = grid.POSITIVE_AC_BOUNDARY_GRID
        axes = grid._grid_axes(region, grid.DEFAULT_GRID_SPACING)
        metadata = grid._grid_metadata(region.name, grid.DEFAULT_GRID_SPACING)
        base = grid._base_params_for_region(region.name)
        assert [len(axis) for axis in axes] == [4, 4, 16, 4, 4, 16, 6]
        assert grid._grid_seed_count(region.name) == 393_216
        assert metadata["bounds"][0] == ["-1.4", "-0.2"]
        assert metadata["bounds"][2] == ["-4.3", "1.5"]
        assert metadata["bounds"][6] == ["-1.2", "0.8"]
        assert metadata["base_params"]["left"]["c"] == grid._base_params_payload(grid.POSITIVE_AC_BASE_PARAMS)["left"]["c"]
        for u_left in region.bounds[0]:
            for v_left in region.bounds[1]:
                a = base.left.a * mp.exp(u_left)
                c = base.left.c * mp.exp(v_left)
                assert a > 0
                assert c > 0
                assert 3 * a - c > 0
        for u_right in region.bounds[3]:
            for v_right in region.bounds[4]:
                a = -base.right.d * mp.exp(u_right)
                c = -base.right.f * mp.exp(v_right)
                assert a > 0
                assert c > 0
                assert 3 * a - c > 0


def test_positive_ac_boundary_v2_grid_extends_lower_tail() -> None:
    """The second positive-ac strip should follow the new low boundary while staying on the branch."""
    with mp.workdps(80):
        region = grid.POSITIVE_AC_BOUNDARY_V2_GRID
        axes = grid._grid_axes(region, grid.DEFAULT_GRID_SPACING)
        metadata = grid._grid_metadata(region.name, grid.DEFAULT_GRID_SPACING)
        base = grid._base_params_for_region(region.name)
        assert [len(axis) for axis in axes] == [4, 4, 16, 4, 4, 16, 4]
        assert grid._grid_seed_count(region.name) == 262_144
        assert metadata["bounds"][0] == ["-2.2", "-1.0"]
        assert metadata["bounds"][2] == ["-4.3", "1.5"]
        assert metadata["bounds"][6] == ["-2.0", "-0.8"]
        assert axes[0][2] == mp.mpf("-1.4")
        assert axes[6][2] == mp.mpf("-1.2")
        assert metadata["base_params"]["left"]["c"] == grid._base_params_payload(grid.POSITIVE_AC_BASE_PARAMS)["left"]["c"]
        for u_left in region.bounds[0]:
            for v_left in region.bounds[1]:
                a = base.left.a * mp.exp(u_left)
                c = base.left.c * mp.exp(v_left)
                assert a > 0
                assert c > 0
                assert 3 * a - c > 0
        for u_right in region.bounds[3]:
            for v_right in region.bounds[4]:
                a = -base.right.d * mp.exp(u_right)
                c = -base.right.f * mp.exp(v_right)
                assert a > 0
                assert c > 0
                assert 3 * a - c > 0


def test_cell_center_shift_uses_midpoints_inside_same_box() -> None:
    """The shifted near grid should use cell centers between default vertices."""
    with mp.workdps(80):
        axes = grid._grid_axes(grid.NEAR_GRID, grid.DEFAULT_GRID_SPACING, "cell-center")
        assert [len(axis) for axis in axes] == [3, 3, 8, 3, 3, 8, 4]
        assert grid._grid_seed_count(shift="cell-center") == 20_736
        for actual, expected in zip(axes[0], ("-0.4", "0", "0.4")):
            _assert_close(actual, expected)
        _assert_close(axes[2][0], "-1.3125")
        _assert_close(axes[2][-1], "1.3125")
        for actual, expected in zip(axes[6], ("-0.6", "-0.2", "0.2", "0.6")):
            _assert_close(actual, expected)


def test_grid_seed_order_is_stable_and_limitable() -> None:
    """Debug limits should preserve the stable full-grid seed indices."""
    with mp.workdps(80):
        seeds = grid._grid_seeds(limit=3)
        assert [seed.index for seed in seeds] == [0, 1, 2]
        assert [seed.region for seed in seeds] == ["near", "near", "near"]
        assert seeds[0].source == "calibrated_grid"
        assert seeds[0].point.u_left == mp.mpf("-0.6")
        assert seeds[0].point.s == mp.mpf("-0.8")
        assert seeds[1].point.s == mp.mpf("-0.4")
        assert seeds[2].point.s == mp.zero


def test_cell_center_seed_order_is_stable_and_marked_in_metadata() -> None:
    """Cell-center runs should keep stable indices and persist the shift policy."""
    with mp.workdps(80):
        seeds = grid._grid_seeds(limit=3, shift="cell-center")
        assert [seed.index for seed in seeds] == [0, 1, 2]
        _assert_close(seeds[0].point.u_left, "-0.4")
        _assert_close(seeds[0].point.r_left, "-1.3125")
        _assert_close(seeds[0].point.s, "-0.6")
        _assert_close(seeds[1].point.s, "-0.2")
        _assert_close(seeds[2].point.s, "0.2")
        metadata = grid._grid_metadata("near", grid.DEFAULT_GRID_SPACING, limit=3, shift="cell-center")
        assert metadata["shift"] == "cell-center"
        assert metadata["full_seed_count"] == 20_736


def test_checkpoint_compatibility_and_resume_selection(tmp_path: Path, monkeypatch) -> None:
    """A matching incomplete checkpoint should be resumed, but completed runs should not."""
    monkeypatch.setattr(grid, "OUTPUT_DIR", tmp_path)
    jsonl_path, summary_path = grid._output_paths(datetime(2026, 5, 23, 12, 0, 0))
    _write_jsonl_event(jsonl_path, _event("run_start", grid._run_start_payload(jsonl_path, summary_path, limit=3)))
    assert grid._checkpoint_is_compatible(jsonl_path, limit=3)
    resumed_jsonl, resumed_summary, resumed = grid._resume_or_new_paths(limit=3)
    assert resumed
    assert resumed_jsonl == jsonl_path
    assert resumed_summary == summary_path

    _write_jsonl_event(jsonl_path, _event("run_summary", {"done": True}))
    assert not grid._checkpoint_is_compatible(jsonl_path, limit=3)


def test_checkpoint_compatibility_distinguishes_grid_shift(tmp_path: Path, monkeypatch) -> None:
    """Vertex and cell-center checkpoints must not resume each other."""
    monkeypatch.setattr(grid, "OUTPUT_DIR", tmp_path)
    jsonl_path, summary_path = grid._output_paths(datetime(2026, 5, 23, 12, 0, 0))
    _write_jsonl_event(jsonl_path, _event("run_start", grid._run_start_payload(jsonl_path, summary_path, limit=3, shift="cell-center")))
    assert grid._checkpoint_is_compatible(jsonl_path, limit=3, shift="cell-center")
    assert not grid._checkpoint_is_compatible(jsonl_path, limit=3)

    resumed_jsonl, _resumed_summary, resumed = grid._resume_or_new_paths(limit=3, shift="cell-center")
    assert resumed
    assert resumed_jsonl == jsonl_path

    vertex_jsonl, _vertex_summary, vertex_resumed = grid._resume_or_new_paths(limit=3, shift="vertex")
    assert not vertex_resumed
    assert vertex_jsonl != jsonl_path


def test_jsonl_parser_ignores_partial_final_line(tmp_path: Path) -> None:
    """A crash-truncated final line should not break completed-seed parsing."""
    path = tmp_path / "events.jsonl"
    _write_jsonl_event(path, _event("scout_result", {"seed_index": 7}))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "scout_result", ')
    assert grid._completed_seed_indices(path) == {7}


def test_serial_scout_runner_writes_payloads_and_skips_completed(tmp_path: Path, monkeypatch) -> None:
    """The parent process should write scout events and resume should skip completed seeds."""
    calls = []

    def fake_residual(point, config):
        calls.append(point)
        return _fake_result(point, config)

    monkeypatch.setattr(grid, "two_sided_residual", fake_residual)
    path = tmp_path / "grid.jsonl"
    seeds = grid._grid_seeds(limit=3)
    grid._run_scouts(seeds, path, workers=1, progress_every=0)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["scout_result", "scout_result", "scout_result"]
    assert [event["seed_index"] for event in events] == [0, 1, 2]
    assert len(calls) == 3

    def unexpected_residual(point, config):
        raise AssertionError("completed seeds should not be evaluated again")

    monkeypatch.setattr(grid, "two_sided_residual", unexpected_residual)
    grid._run_scouts(seeds, path, workers=1, progress_every=0)
    assert len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]) == 3


def test_non_default_scouts_use_region_base_params_and_configs(monkeypatch) -> None:
    """Non-default scouts should evaluate residuals from their region base point and scout config."""
    calls = []

    def fake_residual(point, config, *, base_params=grid.DEFAULT_PARAMS):
        calls.append((base_params, config))
        return _fake_result(point, config, base_params=base_params)

    monkeypatch.setattr(grid, "two_sided_residual", fake_residual)
    for region_name in (
        "positive-ac",
        "positive-ac-boundary",
        "positive-ac-boundary-v2",
        "negative-ac",
        "mixed-mu-short",
        "mixed-mu-boundary",
    ):
        seed = grid._grid_seeds(region_name=region_name, limit=1)[0]
        payload = grid._evaluate_seed_payload(seed)
        assert payload["region"] == region_name
        assert abs(mp.mpf(payload["result"]["residual_norm"]) - mp.mpf("1e-3")) < mp.mpf("1e-30")
    assert [base for base, _config in calls] == [
        grid.POSITIVE_AC_BASE_PARAMS,
        grid.POSITIVE_AC_BASE_PARAMS,
        grid.POSITIVE_AC_BASE_PARAMS,
        grid.NEGATIVE_AC_BASE_PARAMS,
        grid.MIXED_MU_BASE_PARAMS,
        grid.MIXED_MU_BASE_PARAMS,
    ]
    assert [config.series_order for _base, config in calls] == [4, 4, 4, 4, 6, 6]


def test_positive_ac_base_scout_stays_on_a_real_branch() -> None:
    """The positive-ac branch should not be rejected by Berger-sign branch guards."""
    with mp.workdps(grid.SCOUT_CONFIG.working_dps):
        result = grid.two_sided_residual(
            BASE_TWO_SIDED_POINT,
            grid.SCOUT_CONFIG,
            base_params=grid.POSITIVE_AC_BASE_PARAMS,
        )
        assert result.failure is None
        assert result.residual_norm > 0
        assert result.branch_diagnostics["left_min_sum27"] < 0
        assert result.branch_diagnostics["left_min_product"] > 0


def test_negative_ac_base_scout_stays_on_a_real_branch() -> None:
    """The negative-ac branch should be accepted by the implemented ac-positive formulas."""
    with mp.workdps(grid.SCOUT_CONFIG.working_dps):
        result = grid.two_sided_residual(
            BASE_TWO_SIDED_POINT,
            grid.SCOUT_CONFIG,
            base_params=grid.NEGATIVE_AC_BASE_PARAMS,
        )
        assert result.failure is None
        assert result.residual_norm > 0
        assert result.branch_diagnostics["left_min_sum27"] < 0
        assert result.branch_diagnostics["left_min_product"] > 0


def test_mixed_mu_short_first_seed_stays_on_a_real_branch() -> None:
    """The mixed-mu scout box should start in a branch-valid shortened interval regime."""
    seed = grid._grid_seeds(region_name="mixed-mu-short", limit=1)[0]
    with mp.workdps(grid.MIXED_MU_SCOUT_CONFIG.working_dps):
        result = grid.two_sided_residual(
            seed.point,
            grid.MIXED_MU_SCOUT_CONFIG,
            base_params=grid.MIXED_MU_BASE_PARAMS,
        )
        assert seed.point.s == mp.mpf("-2.0")
        assert result.failure is None
        assert result.residual_norm > 0
        assert result.branch_diagnostics["left_min_product"] > 0
        assert result.branch_diagnostics["right_min_product"] > 0


def test_summary_payload_is_compact_and_json_serializable(tmp_path: Path, monkeypatch) -> None:
    """Run summaries should contain compact scout diagnostics instead of full result lists."""
    monkeypatch.setattr(grid, "two_sided_residual", lambda point, config: _fake_result(point, config))
    jsonl_path = tmp_path / "grid.jsonl"
    summary_path = tmp_path / "grid-summary.json"
    metadata = grid._run_start_payload(jsonl_path, summary_path, limit=2)
    _write_jsonl_event(jsonl_path, _event("run_start", metadata))
    grid._run_scouts(grid._grid_seeds(limit=2), jsonl_path, workers=1, progress_every=0)
    payload = grid._summary_payload(jsonl_path, metadata)
    assert payload["scout_count"] == 2
    assert payload["classification_counts"] == {"scout_success": 2, "scout_failure": 0}
    assert len(payload["best_scouts"]) == 2
    json.dumps(payload)
