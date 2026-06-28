"""Tests for Berger mirror-recovery calibration helpers."""

from __future__ import annotations

from random import Random

from mpmath import mp

from experiments.berger_space import mirror_recovery_calibration as calibration
from problem import DEFAULT_CONFIG, DEFAULT_PARAMS, SolverConfig
from solver.mirror_refinement import CandidateTrack, NewtonSettings, RefinementStageReport, newton_refine
from solver.mirror_shooting import BASE_POINT, MirrorResidualResult, MirrorSearchPoint, mirror_residual


SMOKE_CONFIG = SolverConfig(4, 30, 15, mp.mpf("0.95"), 0, DEFAULT_CONFIG.match_t)


def _fake_result(point: MirrorSearchPoint, norm: str, failure: str | None = None) -> MirrorResidualResult:
    """Build one synthetic mirror residual result."""
    return MirrorResidualResult(point, DEFAULT_PARAMS, DEFAULT_CONFIG, (), mp.mpf(norm), None, None, 0, {}, failure)


def _fake_stage(point: MirrorSearchPoint, norm: str, status: str = "max_steps") -> RefinementStageReport:
    """Build one synthetic refinement stage."""
    settings = NewtonSettings("fake", DEFAULT_CONFIG, mp.mpf("1e-3"), mp.mpf("1e-8"), 1)
    final = _fake_result(point, norm)
    return RefinementStageReport(settings, _fake_result(point, "1"), final, (), status)


def _fake_track(
    point: MirrorSearchPoint,
    norm: str,
    verification_norms=(),
    status: str = "max_steps",
    failure: str | None = None,
) -> CandidateTrack:
    """Build one synthetic calibration track."""
    stage = _fake_stage(point, norm, status)
    scout = _fake_result(point, "1", failure)
    verifications = tuple(_fake_result(point, item) for item in verification_norms)
    return CandidateTrack(1, "test", point, scout, (stage,), verifications, "inconclusive")


def test_shell_seed_generation_is_reproducible_and_on_shell() -> None:
    """Shell seeds should be deterministic and have exact max-norm radius."""
    with mp.workdps(50):
        radius = mp.mpf("0.01")
        left = calibration._shell_seeds(radius, 0, Random(1729))
        right = calibration._shell_seeds(radius, 0, Random(1729))
        assert left == right
        assert len(left) == 8 + calibration.RANDOM_SHELL_SAMPLES
        for seed in left:
            assert calibration._point_distance(seed.point) == radius


def test_broad_box_seeds_are_reproducible_and_bounded() -> None:
    """Broad-box seeds should satisfy the intended coordinate bounds."""
    with mp.workdps(50):
        left = calibration._broad_box_seeds(1729, 20)
        right = calibration._broad_box_seeds(1729, 20)
        assert left == right
        for seed in left:
            point = seed.point
            assert abs(point.u) <= 1
            assert abs(point.v) <= 1
            assert abs(point.r) <= 3
            assert abs(point.s) <= 1


def test_broad_box_seed_indices_can_be_offset() -> None:
    """Broad-box seeds should support globally unique checkpoint indices."""
    seeds = calibration._broad_box_seeds(1729, 3, start_index=100)
    assert [seed.index for seed in seeds] == [100, 101, 102]


def test_checkpoint_compatibility_requires_current_recipe(tmp_path) -> None:
    """Resume should ignore stale checkpoints with different baked constants."""
    jsonl_path = tmp_path / "stale-seed1729-recovery.jsonl"
    summary_path = tmp_path / "stale-seed1729-recovery-summary.json"
    compatible = calibration._event("run_start", calibration._run_start_payload(jsonl_path, summary_path))
    stale = calibration._event("run_start", {"random_seed": calibration.RANDOM_SEED, "radii": ["1e-4"]})

    calibration._write_jsonl_event(jsonl_path, stale)
    assert not calibration._checkpoint_is_compatible(jsonl_path)

    jsonl_path.write_text("", encoding="utf-8")
    calibration._write_jsonl_event(jsonl_path, compatible)
    assert calibration._checkpoint_is_compatible(jsonl_path)


def test_candidate_payload_round_trips_for_resume() -> None:
    """Scout candidates should reload from JSON payloads without changing keys."""
    seed = calibration._broad_box_seeds(1729, 1, start_index=200)[0]
    result = _fake_result(seed.point, "0.125")
    candidate = calibration.SearchCandidate(seed, result)
    payload = calibration._candidate_payload(candidate)
    loaded = calibration._candidate_from_payload(payload)
    assert loaded.seed.index == candidate.seed.index
    assert loaded.seed.region == candidate.seed.region
    assert loaded.seed.point == candidate.seed.point
    assert loaded.result.residual_norm == candidate.result.residual_norm


def test_classification_labels_synthetic_tracks() -> None:
    """Calibration classification should identify the main outcomes."""
    refs = (_fake_result(BASE_POINT, "1e-14"), _fake_result(BASE_POINT, "2e-14"))
    near = MirrorSearchPoint(mp.mpf("5e-4"), mp.zero, mp.zero, mp.zero)
    far = MirrorSearchPoint(mp.mpf("0.1"), mp.zero, mp.zero, mp.zero)
    assert calibration._classify_track(_fake_track(near, "1e-12", ("1e-12", "2e-12")), refs) == "recovered_berger"
    assert calibration._classify_track(_fake_track(far, "1e-12", ("1e-9", "5e-9")), refs) == "possible_non_berger_root"
    assert calibration._classify_track(_fake_track(near, "1e-12", ("1e-3", "2e-3")), refs) == "finite_order_artifact"
    assert calibration._classify_track(_fake_track(far, "1", (), "no_improvement"), refs) == "failed"
    assert calibration._classify_track(_fake_track(far, "1", ()), refs) == "inconclusive"


def test_high_order_correction_only_triggers_near_low_order_roots() -> None:
    """Only near-Berger low-order attractors should pay for order-14 Newton."""
    near = MirrorSearchPoint(mp.mpf("0.001"), mp.zero, mp.zero, mp.zero)
    far = MirrorSearchPoint(mp.mpf("0.1"), mp.zero, mp.zero, mp.zero)
    assert calibration._needs_high_order_correction(_fake_track(near, "1e-10"))
    assert not calibration._needs_high_order_correction(_fake_track(near, "1e-4"))
    assert not calibration._needs_high_order_correction(_fake_track(far, "1e-10"))


def test_smoke_newton_and_base_recovery_classification() -> None:
    """A tiny calibration path should run and classify BASE_POINT as recovered."""
    point = MirrorSearchPoint(mp.mpf("0.002"), mp.zero, mp.zero, mp.zero)
    settings = NewtonSettings("smoke", SMOKE_CONFIG, mp.mpf("1e-3"), mp.mpf("1e-10"), 1)
    with mp.workdps(SMOKE_CONFIG.working_dps):
        report = newton_refine(point, settings)
        assert report.final.residual_norm <= report.initial.residual_norm or report.status in {
            "branch_failure",
            "jacobian_failure",
            "no_improvement",
            "tolerance_hit",
        }
        base = mirror_residual(BASE_POINT, SMOKE_CONFIG)
    track = CandidateTrack(0, "base", BASE_POINT, base, (), (base, base), "inconclusive")
    assert calibration._classify_track(track, (base, base)) == "recovered_berger"
