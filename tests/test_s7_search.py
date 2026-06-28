"""Tests for fixed-chart S7 recovery/search helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mpmath import mp

from experiments.s7 import search_common as s7


def test_s7_scaled_params_preserve_fixed_right_chart() -> None:
    """Scaled S7 points should vary left data while preserving fixed right chart and interval."""
    with mp.workdps(80):
        base = s7.TARGETS["round"].params_builder()
        point = s7.S7SearchPoint(mp.mpf("0.1"), mp.mpf("-0.2"), mp.mpf("0.3"), mp.mpf("0.4"))
        params, config = s7.params_from_s7_scaled(point, base_params=base, template_config=s7.SCOUT_CONFIG)
        assert params.right_chart == "s7_p3"
        assert params.fixed_right == base.fixed_right
        assert abs(params.left.a - base.left.a * mp.exp(point.u)) < mp.mpf("1e-40")
        assert abs(params.left.c - base.left.c * mp.exp(point.v)) < mp.mpf("1e-40")
        assert abs(params.left.alpha - base.left.alpha * mp.mpf("1.3")) < mp.mpf("1e-40")
        assert abs(config.match_t - s7.SCOUT_CONFIG.match_t) < mp.mpf("1e-40")
        assert abs(params.interval_end - base.interval_end) < mp.mpf("1e-40")


def test_positive_ac_scout_region_uses_real_positive_branch() -> None:
    """The positive-ac region should cover the other real left endpoint chamber."""
    with mp.workdps(80):
        base = s7.TARGETS["round"].params_builder()
        point = s7.S7SearchPoint(mp.mpf("0.25"), mp.mpf("0.5"), mp.mpf("-2"), mp.zero)
        params, config = s7.params_from_s7_scaled(
            point,
            base_params=base,
            template_config=s7.SCOUT_CONFIG,
            region=s7.POSITIVE_AC_SCOUT_REGION.name,
        )
        expected_a = base.left.a * mp.exp(point.u)
        assert params.right_chart == "s7_p3"
        assert params.fixed_right == base.fixed_right
        assert abs(params.left.a - expected_a) < mp.mpf("1e-40")
        assert abs(params.left.c - 3 * expected_a * point.v) < mp.mpf("1e-40")
        assert abs(params.left.alpha - s7.POSITIVE_AC_ALPHA_SCALE * point.r) < mp.mpf("1e-40")
        assert params.left.a > 0
        assert params.left.c > 0
        assert 3 * params.left.a - params.left.c > 0
        assert abs(config.match_t - s7.SCOUT_CONFIG.match_t) < mp.mpf("1e-40")


def test_recovery_seed_count_and_targets() -> None:
    """Both recovery scripts should use a small deterministic 3D calibration set."""
    round_seeds = s7.recovery_seeds("round")
    squashed_seeds = s7.recovery_seeds("squashed")
    assert len(round_seeds) == 6
    assert len(squashed_seeds) == 6
    assert {seed.target for seed in round_seeds} == {"round"}
    assert {seed.target for seed in squashed_seeds} == {"squashed"}
    assert {seed.source for seed in round_seeds} == {"axis"}


def test_scout_grid_metadata_matches_expected_long_run_size() -> None:
    """The default S7 scout grid should be large enough for a terminal run."""
    with mp.workdps(80):
        metadata = s7.scout_grid_metadata()
        assert metadata["axis_counts"] == [33, 33, 68]
        assert metadata["full_per_target"] == 74_052
        assert metadata["full_seed_count"] == 148_104
        assert s7.scout_seed_count() == 148_104


def test_positive_ac_scout_grid_metadata_matches_expected_terminal_run_size() -> None:
    """The positive-ac S7 scout should be a substantial but smaller follow-up grid."""
    with mp.workdps(80):
        metadata = s7.scout_grid_metadata(region=s7.POSITIVE_AC_SCOUT_REGION.name)
        assert metadata["region"] == "positive-ac"
        assert metadata["coordinate_names"] == ["u", "rho", "r"]
        assert metadata["axis_counts"] == [33, 6, 95]
        assert metadata["full_per_target"] == 18_810
        assert metadata["full_seed_count"] == 37_620
        assert s7.scout_seed_count(region=s7.POSITIVE_AC_SCOUT_REGION.name) == 37_620


def test_scout_seeds_are_grouped_by_target_and_limitable() -> None:
    """Debug limits should preserve deterministic seed indices and target order."""
    seeds = s7.scout_seeds(limit=3)
    assert [seed.index for seed in seeds] == [0, 1, 2]
    assert [seed.target for seed in seeds] == ["round", "round", "round"]
    assert seeds[0].point.u == mp.mpf("-1.2")
    assert seeds[0].point.s == mp.zero


def test_positive_ac_scout_seeds_use_positive_ac_region() -> None:
    """Positive-ac scout seeds should carry the positive-ac parameterization label."""
    seeds = s7.scout_seeds(region=s7.POSITIVE_AC_SCOUT_REGION.name, limit=3)
    assert [seed.index for seed in seeds] == [0, 1, 2]
    assert {seed.region for seed in seeds} == {"positive-ac"}
    assert seeds[0].point.v == mp.mpf("0.05")
    assert seeds[0].point.r == mp.mpf("-3.5")


def test_scout_checkpoint_resume_selection(tmp_path: Path, monkeypatch) -> None:
    """A matching incomplete scout checkpoint should be resumed."""
    monkeypatch.setattr(s7, "SCOUT_OUTPUT_DIR", tmp_path)
    jsonl_path, summary_path = s7._scout_output_paths(datetime(2026, 6, 9, 12, 0, 0))
    targets = ("round", "squashed")
    metadata = s7._scout_run_start_payload(jsonl_path, summary_path, targets, s7.DEFAULT_SCOUT_SPACING, limit=3)
    s7._write_jsonl_event(jsonl_path, s7._event("run_start", metadata))
    resumed_jsonl, resumed_summary, resumed = s7._resume_or_new_scout_paths(targets, s7.DEFAULT_SCOUT_SPACING, 3)
    assert resumed
    assert resumed_jsonl == jsonl_path
    assert resumed_summary == summary_path

    s7._write_jsonl_event(jsonl_path, s7._event("run_summary", {"done": True}))
    fresh_jsonl, _fresh_summary, fresh_resumed = s7._resume_or_new_scout_paths(targets, s7.DEFAULT_SCOUT_SPACING, 3)
    assert not fresh_resumed
    assert fresh_jsonl != jsonl_path


def test_scout_checkpoint_resume_distinguishes_region(tmp_path: Path, monkeypatch) -> None:
    """Default and positive-ac scouts should not resume each other's checkpoints."""
    monkeypatch.setattr(s7, "SCOUT_OUTPUT_DIR", tmp_path)
    jsonl_path, summary_path = s7._scout_output_paths(datetime(2026, 6, 13, 12, 0, 0))
    targets = ("round", "squashed")
    metadata = s7._scout_run_start_payload(
        jsonl_path,
        summary_path,
        targets,
        s7.DEFAULT_SCOUT_SPACING,
        limit=3,
        region=s7.POSITIVE_AC_SCOUT_REGION.name,
    )
    s7._write_jsonl_event(jsonl_path, s7._event("run_start", metadata))

    assert s7._scout_checkpoint_is_compatible(
        jsonl_path,
        targets,
        s7.DEFAULT_SCOUT_SPACING,
        3,
        s7.POSITIVE_AC_SCOUT_REGION.name,
    )
    assert not s7._scout_checkpoint_is_compatible(
        jsonl_path,
        targets,
        s7.DEFAULT_SCOUT_SPACING,
        3,
        s7.DEFAULT_SCOUT_REGION.name,
    )


def test_known_targets_have_small_low_order_residuals() -> None:
    """The cheap S7 scout residual should already recognize both known targets."""
    with mp.workdps(s7.SCOUT_CONFIG.working_dps):
        base_point = s7.S7SearchPoint(mp.zero, mp.zero, mp.zero, mp.zero)
        for target in s7.TARGETS.values():
            result = s7.s7_residual(base_point, s7.SCOUT_CONFIG, base_params=target.params_builder())
            assert result.failure is None
            assert result.residual_norm < mp.mpf("0.05")
