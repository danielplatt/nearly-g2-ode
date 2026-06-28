"""Tests for the non-mirrored Berger recovery calibration."""

from __future__ import annotations

import json
from datetime import datetime

from mpmath import mp

from experiments.berger_space import non_mirrored_recovery_calibration as calibration
from experiments.shared.non_mirrored_common import _event, _write_jsonl_event
from solver.two_sided_refinement import TwoSidedCandidateTrack
from solver.two_sided_shooting import BASE_TWO_SIDED_POINT, TwoSidedResidualResult, TwoSidedSearchPoint, params_from_two_sided_scaled


def _fake_result(point: TwoSidedSearchPoint, norm: str, failure: str | None = None) -> TwoSidedResidualResult:
    """Return a synthetic residual result for classification tests."""
    params, config = params_from_two_sided_scaled(point, template_config=calibration.SCOUT_CONFIG)
    residual = tuple(mp.mpf(norm) for _ in range(8))
    return TwoSidedResidualResult(point, params, config, residual, mp.mpf(norm), None, None, None, None, (0, 0), {}, failure)


def _fake_track(point: TwoSidedSearchPoint, norms: tuple[str, ...], scout_norm: str = "1") -> TwoSidedCandidateTrack:
    """Return a synthetic classified-track input."""
    scout = _fake_result(point, scout_norm)
    verifications = tuple(_fake_result(point, norm) for norm in norms)
    return TwoSidedCandidateTrack(1, "test", point, scout, (), verifications, "inconclusive")


def test_shell_seeds_are_reproducible_and_on_requested_radius() -> None:
    """Shell seeds should sit exactly on the intended max-norm radius."""
    with mp.workdps(60):
        radius = mp.mpf("0.03")
        left = calibration._shell_seeds(radius, 0, calibration.Random(1729))
        right = calibration._shell_seeds(radius, 0, calibration.Random(1729))
        assert left == right
        assert len(left) == 14 + calibration.RANDOM_SHELL_SAMPLES
        for seed in left:
            assert calibration._point_distance(seed.point) == radius


def test_calibration_seed_count_and_regions() -> None:
    """The calibration recipe should emit shells plus the blind local box."""
    seeds = calibration._calibration_seeds()
    assert len(seeds) == len(calibration.SHELL_RADII) * (14 + calibration.RANDOM_SHELL_SAMPLES) + calibration.LOCAL_BOX_SAMPLES
    assert any(seed.region == "local_box" for seed in seeds)


def test_classification_labels_synthetic_recovery_and_artifacts() -> None:
    """Synthetic tracks should exercise the recovery branches."""
    refs = tuple(_fake_result(BASE_TWO_SIDED_POINT, "1e-12") for _ in range(2))
    recovered = _fake_track(BASE_TWO_SIDED_POINT, ("1e-10", "2e-10"))
    artifact = _fake_track(BASE_TWO_SIDED_POINT, ("1e-3", "2e-3"), "1e-10")
    far = TwoSidedSearchPoint(mp.mpf("0.2"), mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero)
    possible = _fake_track(far, ("1e-9", "2e-9"))
    assert calibration._classify_track(recovered, refs) == "recovered_berger"
    assert calibration._classify_track(artifact, refs) == "finite_order_artifact"
    assert calibration._classify_track(possible, refs) == "possible_non_berger_root"


def test_checkpoint_compatibility_round_trips(tmp_path) -> None:
    """A matching run-start event should be resumable."""
    jsonl_path, summary_path = calibration._output_paths(datetime(2026, 5, 17, 12, 0, 0))
    jsonl_path = tmp_path / jsonl_path.name
    summary_path = tmp_path / summary_path.name
    _write_jsonl_event(jsonl_path, _event("run_start", calibration._run_start_payload(jsonl_path, summary_path)))
    assert calibration._checkpoint_is_compatible(jsonl_path)
    assert json.loads(jsonl_path.read_text())["calibration_version"] == calibration.CALIBRATION_VERSION
