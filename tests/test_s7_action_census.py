"""Tests for the S7 cohomogeneity-one action census."""

from __future__ import annotations

import json

import experiments.s7_action_census
from experiments.s7 import action_census as census


def test_candidate_list_contains_tested_and_new_action_families() -> None:
    """The census should cover the actions relevant to the next-search decision."""
    keys = {candidate.key for candidate in census.action_candidates()}
    assert {
        "so4_z2_q_system",
        "sp1_3_diag_podesta",
        "sp1_2_u1_intermediate",
        "sp1_2_left_sum",
        "su3_u1_complex_sum",
        "g2_principal_s6",
        "real_sum_s1_s5",
        "real_sum_s2_s4",
        "real_sum_s3_s3",
    } <= keys


def test_duplicate_filter_keeps_weaker_s3x_s3_actions() -> None:
    """Same orbit foliation as Podesta is not a duplicate when invariance is weaker."""
    summary = census.build_summary()
    filtered = summary["steps"]["2_remove_duplicates"]
    assert "sp1_3_diag_podesta" in filtered["already_tested"]
    assert "sp1_2_u1_intermediate" in filtered["new_viable"]
    assert "sp1_2_left_sum" in filtered["new_viable"]
    assert "real_sum_s3_s3" in filtered["discarded"]


def test_invariant_dimension_computations_match_representation_models() -> None:
    """The key dimension counts should be computed from the intended models."""
    assert census.s3x_s3_ladder_dimensions("sp1_diag") == census.InvariantDimensions(1, 4)
    assert census.s3x_s3_ladder_dimensions("u1_diag") == census.InvariantDimensions(5, 8)
    assert census.s3x_s3_ladder_dimensions("finite") == census.InvariantDimensions(15, 20)
    assert census.g2_principal_s6_dimensions() == census.InvariantDimensions(1, 2)
    assert census.su3_u1_s5_s1_dimensions() == census.InvariantDimensions(4, 6)
    assert census.real_sum_invariant_dimensions(1, 5) == census.InvariantDimensions(0, 0)
    assert census.real_sum_invariant_dimensions(2, 4) == census.InvariantDimensions(1, 0)
    assert census.real_sum_invariant_dimensions(3, 3) == census.InvariantDimensions(0, 2)


def test_known_solution_visibility_and_stable_room_are_recorded() -> None:
    """Step 4 should distinguish calibration strength from mere metric symmetry."""
    by_key = {candidate.key: candidate for candidate in census.action_candidates()}
    assert by_key["sp1_2_u1_intermediate"].stable_room is True
    assert by_key["sp1_2_u1_intermediate"].round_visible == "yes"
    assert by_key["sp1_2_u1_intermediate"].squashed_visible == "yes"
    assert by_key["su3_u1_complex_sum"].round_visible == "yes"
    assert by_key["su3_u1_complex_sum"].squashed_visible == "not-known"
    assert by_key["real_sum_s2_s4"].stable_room is False


def test_ranking_puts_u1_intermediate_action_first() -> None:
    """The recommended next sprint should be the moderate S3xS3 U(1) action."""
    ranked = census.ranked_new_actions()
    assert [candidate.key for candidate in ranked] == [
        "sp1_2_u1_intermediate",
        "su3_u1_complex_sum",
        "g2_principal_s6",
        "sp1_2_left_sum",
    ]
    assert ranked[0].total_functions == 13
    assert ranked[0].rank_score is not None
    assert ranked[0].rank_score > ranked[1].rank_score


def test_render_and_cli_write_report(tmp_path) -> None:
    """The report should expose all five steps and the CLI should write it."""
    summary = census.build_summary()
    markdown = census.render_markdown(summary)
    assert "# S7 Cohomogeneity-One Action Census" in markdown
    assert "## Step 1: Candidate actions" in markdown
    assert "## Step 5: Ranking" in markdown
    assert "`sp1_2_u1_intermediate`" in markdown

    report_path = tmp_path / "census.md"
    census.main(["--write-markdown", str(report_path)])
    assert report_path.read_text(encoding="utf-8").startswith("# S7 Cohomogeneity-One Action Census")

    payload = json.loads(json.dumps(summary))
    assert payload["version"] == census.ACTION_CENSUS_VERSION
    assert experiments.s7_action_census.main is census.main
