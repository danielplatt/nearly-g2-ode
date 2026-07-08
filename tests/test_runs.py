"""Smoke tests for named experiment modules."""

from __future__ import annotations

from mpmath import mp

import experiments.berger_jacobian
import experiments.berger_max_volume_calibration
import experiments.berger_max_volume_scout
import experiments.berger_opposite_end_check
import experiments.berger_branch_audit
import experiments.berger_s3_bundle_action_audit
import experiments.aloff_wallach_feasibility
import experiments.aloff_wallach_ansatz
import experiments.aloff_wallach_endpoint_smoothness
import experiments.aloff_wallach_homogeneous_recovery
import experiments.aloff_wallach_scout
import experiments.aloff_wallach_recovery_calibration
import experiments.aloff_wallach.ansatz
import experiments.aloff_wallach.endpoint_smoothness
import experiments.aloff_wallach.evolution
import experiments.aloff_wallach.feasibility
import experiments.aloff_wallach.homogeneous_recovery
import experiments.aloff_wallach.recovery_calibration
import experiments.aloff_wallach.scout
import experiments.berger_space.max_volume_calibration
import experiments.berger_space.max_volume_scout
import experiments.berger_space.s3_bundle_action_audit
import experiments.berger_space.validation
import experiments.fh_s6_max_volume_match
import experiments.fh_s6_max_volume_scout
import experiments.fh_s6_terminal_scout
import experiments.fh_s6_terminal_shooting
import experiments.foscolo_haskins.s6_common
import experiments.foscolo_haskins.s6_scout
import experiments.foscolo_haskins.s6_terminal_shooting
import experiments.mirror_search
import experiments.mirror_recovery_calibration
import experiments.mirror_covering_calibration
import experiments.mirror_guarded_covering_search
import experiments.mirror_local_grid_v3
import experiments.mirror_sweep
import experiments.mirror_sweep_v2
import experiments.mirror_sweep_v3
import experiments.mu_branch_probe
import experiments.non_mirrored_berger_jacobian
import experiments.non_mirrored_grid_refine
import experiments.non_mirrored_grid_search
import experiments.non_mirrored_recovery_calibration
import experiments.non_mirrored_search
import experiments.non_mirrored_surrogate_search
import experiments.non_mirrored_surrogate_wide_search
import experiments.s7.round_validation
import experiments.s7.right_chart_comparison
import experiments.s7.round_recovery_calibration
import experiments.s7.scout_search
import experiments.s7.scout_refine
import experiments.s7.search_common
import experiments.s7.right_endpoint_moduli_probe
import experiments.s7.full_moduli_firstjet_scout
import experiments.s7.full_moduli_firstjet_refine
import experiments.s7.full_moduli_offset_scout
import experiments.s7.full_moduli_offset_refine
import experiments.s7.max_volume_calibration
import experiments.s7.max_volume_scout
import experiments.s7.right_germ
import experiments.s7.right_moduli_chart
import experiments.s7.su2_cubed_action_audit
import experiments.s7.action_census
import experiments.s7.su2_cubed_defect_audit
import experiments.s7.su2_cubed_next_sprint_audit
import experiments.s7.su2_cubed_scout
import experiments.s7.su2_cubed_tail_defect
import experiments.s7.squashed_recovery_calibration
import experiments.s7.squashed_validation
import experiments.s7_max_volume_calibration
import experiments.s7_max_volume_scout
import experiments.s7_su2_cubed_action_audit
import experiments.s7_action_census
import experiments.s7_su2_cubed_defect_audit
import experiments.s7_su2_cubed_next_sprint_audit
import experiments.s7_su2_cubed_scout
import experiments.s7_su2_cubed_tail_defect
import experiments.stiefel_feasibility
import experiments.stiefel.feasibility
import solver.max_volume
from experiments.berger import build_params as berger_params
from experiments.doubled_sphere import build_params as doubled_sphere_params


def test_named_experiments_construct_parameter_packages_without_executing() -> None:
    """Named experiment modules should expose import-safe parameter builders."""
    with mp.workdps(80):
        berger = berger_params()
        berger_from_target_package = experiments.berger_space.validation.build_params()
        doubled = doubled_sphere_params()
        round_s7_left = experiments.s7.round_validation.build_left_preset()
        squashed_s7 = experiments.s7.squashed_validation.build_params()
        assert berger == berger_from_target_package
        assert berger.interval_end == doubled.interval_end
        assert abs(doubled.right.d + doubled.left.a) < mp.mpf("1e-40")
        assert abs(doubled.right.f + doubled.left.c) < mp.mpf("1e-40")
        assert abs(doubled.right.omega + doubled.left.alpha) < mp.mpf("1e-40")
        assert abs(round_s7_left.left.a - mp.sqrt(5) / 25) < mp.mpf("1e-40")
        assert experiments.s7.round_validation.build_params().right_chart == "s7_p3"
        assert squashed_s7.right_chart == "s7_p2"
        assert squashed_s7.fixed_right is not None
        assert experiments.berger_jacobian.STEPS
        assert solver.max_volume.MAX_VOLUME_VERSION == "g2-max-volume-v1"
        assert experiments.berger_space.max_volume_scout.SCOUT_VERSION == "berger-max-volume-scout-v1"
        assert experiments.s7.max_volume_scout.SCOUT_VERSION == "s7-max-volume-scout-v1"
        assert experiments.stiefel.feasibility.STIEFEL_FEASIBILITY_VERSION == "stiefel-feasibility-v1"
        assert experiments.mirror_search.RANDOM_SEED == 1729
        assert experiments.mirror_recovery_calibration.RANDOM_SEED == 1729
        assert experiments.mirror_recovery_calibration.BROAD_BOX_SAMPLES == 800
        assert experiments.mirror_covering_calibration.GRID_SIZE == 10000
        assert experiments.mirror_guarded_covering_search.HALTON_SAMPLES == 40000
        assert experiments.mirror_guarded_covering_search.MIN_MATCH_T == mp.mpf("0.01")
        assert experiments.mirror_sweep.RANDOM_SEED == 1729
        assert experiments.mirror_sweep_v2.RANDOM_SEED == 1729
        assert experiments.mirror_sweep_v3.RANDOM_SEED == 1729
        assert experiments.non_mirrored_search.RANDOM_SEED == 1729
        assert experiments.non_mirrored_grid_search.DEFAULT_GRID_SPACING == mp.mpf("0.4")
        assert experiments.non_mirrored_grid_search._grid_seed_count() == 103680
        assert experiments.non_mirrored_grid_search._grid_seed_count("negative-ac") == 250880
        assert experiments.non_mirrored_grid_search._grid_seed_count("mixed-mu-short") == 250880
        assert experiments.non_mirrored_grid_search._grid_seed_count("mixed-mu-boundary", mp.mpf("0.6")) == 32400
        assert experiments.non_mirrored_grid_search.MIXED_MU_SCOUT_CONFIG.series_order == 6
        assert experiments.non_mirrored_grid_search._grid_seed_count("positive-ac-boundary-v2") == 262144
        assert experiments.non_mirrored_grid_refine.DEFAULT_SELECTION_QUOTA == 50
        assert experiments.non_mirrored_recovery_calibration.RANDOM_SEED == 1729
        assert experiments.non_mirrored_recovery_calibration.LOCAL_BOX_SAMPLES == 20
        assert experiments.non_mirrored_surrogate_search.RANDOM_SEED == 1729
        assert experiments.non_mirrored_surrogate_wide_search.RANDOM_SEED == 1729
        assert experiments.mirror_sweep_v3.MIN_MATCH_T == mp.mpf("0.01")
        assert experiments.mu_branch_probe.PROBE_VERSION == "mu-branch-probe-v1"
        assert experiments.non_mirrored_search.MIN_MATCH_T == mp.mpf("0.01")
        assert sum(spec.samples for spec in experiments.non_mirrored_surrogate_search.TRAINING_REGIONS) == 5000
        assert sum(spec.samples for spec in experiments.non_mirrored_surrogate_wide_search.TRAINING_REGIONS) == 20000
        assert experiments.mirror_local_grid_v3.CANDIDATES
        assert experiments.mirror_search.SCOUT_CONFIG.series_order < experiments.mirror_search.REFINE_CONFIG.series_order
        assert experiments.non_mirrored_berger_jacobian.JACOBIAN_CONFIG.series_order == 6
        assert experiments.berger_opposite_end_check.EPSILON == "0.1"
        assert experiments.berger_branch_audit.AUDIT_VERSION == "berger-branch-audit-v1"
        assert (
            experiments.berger_space.s3_bundle_action_audit.AUDIT_VERSION
            == "berger-s3-bundle-action-audit-v1"
        )
        assert experiments.aloff_wallach.feasibility.ALOFF_WALLACH_FEASIBILITY_VERSION == "aloff-wallach-feasibility-v1"
        assert experiments.aloff_wallach.ansatz.N11_ANSATZ_VERSION == "aloff-wallach-n11-ansatz-v4"
        assert (
            experiments.aloff_wallach.endpoint_smoothness.ENDPOINT_SMOOTHNESS_VERSION
            == "aloff-wallach-n11-endpoint-smoothness-v1"
        )
        assert experiments.aloff_wallach.evolution.EVOLUTION_VERSION == "aloff-wallach-n11-evolution-v1"
        assert experiments.aloff_wallach.scout.SCOUT_VERSION == "aloff-wallach-n11-scout-v1"
        assert experiments.aloff_wallach.scout.scout_seed_count(mp.mpf("1"), mp.mpf("1")) == 3**8
        assert (
            experiments.aloff_wallach.homogeneous_recovery.HOMOGENEOUS_RECOVERY_VERSION
            == "aloff-wallach-n11-homogeneous-recovery-v1"
        )
        assert (
            experiments.aloff_wallach.recovery_calibration.RECOVERY_VERSION
            == "aloff-wallach-n11-recovery-calibration-v1"
        )
        assert experiments.aloff_wallach.recovery_calibration.recovery_seed_count(("tri_sasakian",), "all-signs") == 256
        assert experiments.foscolo_haskins.s6_common.MATCH_VERSION == "fh-s6-max-volume-match-v1"
        assert experiments.foscolo_haskins.s6_scout.SCOUT_VERSION == "fh-s6-scout-v1"
        assert experiments.foscolo_haskins.s6_terminal_shooting.SHOOTING_VERSION == "fh-s6-terminal-shooting-v1"
        assert experiments.s7.right_chart_comparison.EPSILONS[-1] == "0.02"
        assert experiments.s7.search_common.scout_seed_count() == 148104
        assert len(experiments.s7.search_common.recovery_seeds("round")) == 6
        assert experiments.s7.scout_refine.DEFAULT_MAX_RESIDUAL == mp.mpf("0.15")
        assert experiments.s7.right_endpoint_moduli_probe.DEFAULT_SERIES_ORDER == 10
        assert experiments.s7.full_moduli_firstjet_scout.scout_seed_count(("round",), 2) == 128
        assert experiments.s7.full_moduli_firstjet_refine.DEFAULT_ORDERS == (8, 10, 14)
        assert experiments.s7.full_moduli_offset_scout.scout_seed_count(("round",), 2) == 128
        assert experiments.s7.full_moduli_offset_refine.DEFAULT_MAX_RESIDUAL == mp.mpf("0.075")
        assert experiments.s7.right_germ.firstjet_anchor_components("round") == (2, 3, 4)
        assert experiments.s7.right_moduli_chart.p3_offset_defect(experiments.s7.right_moduli_chart.p3_offset(1, 2, 19)) == 0
        assert (
            experiments.s7.su2_cubed_action_audit.AUDIT_VERSION
            == "s7-su2-cubed-action-audit-v1"
        )
        assert experiments.s7_su2_cubed_action_audit.main is experiments.s7.su2_cubed_action_audit.main
        assert experiments.s7.action_census.ACTION_CENSUS_VERSION == "s7-action-census-v1"
        assert experiments.s7_action_census.main is experiments.s7.action_census.main
        assert experiments.s7.su2_cubed_defect_audit.DEFECT_AUDIT_VERSION == "s7-su2-cubed-defect-audit-v1"
        assert experiments.s7_su2_cubed_defect_audit.main is experiments.s7.su2_cubed_defect_audit.main
        assert (
            experiments.s7.su2_cubed_next_sprint_audit.NEXT_SPRINT_AUDIT_VERSION
            == "s7-su2-cubed-next-sprint-audit-v1"
        )
        assert experiments.s7_su2_cubed_next_sprint_audit.main is experiments.s7.su2_cubed_next_sprint_audit.main
        assert experiments.s7.su2_cubed_scout.SCOUT_VERSION == "s7-su2-cubed-podesta-scout-v1"
        assert experiments.s7_su2_cubed_scout.main is experiments.s7.su2_cubed_scout.main
        assert experiments.s7.su2_cubed_tail_defect.TAIL_DEFECT_VERSION == "s7-su2-cubed-tail-defect-v1"
        assert experiments.s7_su2_cubed_tail_defect.main is experiments.s7.su2_cubed_tail_defect.main
