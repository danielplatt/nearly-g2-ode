"""Tests for the honest S7 full-moduli terminal-offset scout."""

from __future__ import annotations

from mpmath import mp

from experiments.s7 import full_moduli_offset_scout as scout


def test_full_moduli_offset_scout_count() -> None:
    """The default 7D offset-moduli grid count should be deterministic."""
    assert scout.scout_seed_count(("round", "squashed"), 4) == 32768
    assert scout.scout_seed_count(("round",), 2) == 128


def test_zero_offset_moduli_seed_calibrates_to_known_targets() -> None:
    """The exact round/squashed points should have zero calibrated residual."""
    with mp.workdps(80):
        zero = scout.FullModuliOffsetPoint(*(mp.zero for _ in scout.COORDINATE_NAMES))
        for target in ("round", "squashed"):
            result = scout.evaluate_seed(scout.FullModuliOffsetSeed(-1, target, zero), calibrate=True)
            assert result.failure is None
            assert result.germ_success
            assert result.germ_residual_norm == 0
            assert result.residual_norm < mp.mpf("1e-40")


def test_first_cell_center_seed_is_finite_or_nonfatal() -> None:
    """A genuine scout grid seed should produce a serializable non-crashing result."""
    seed = scout._iter_seeds(("round",), 2)[0]

    result = scout.evaluate_seed(seed)

    assert result.seed == seed
    assert result.failure is None or result.residual_norm == mp.inf
