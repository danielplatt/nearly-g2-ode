"""Tests for the V3 local grid refinement experiment."""

from __future__ import annotations

from mpmath import mp

from experiments import mirror_local_grid_v3
from problem import DEFAULT_CONFIG


def test_local_grid_has_expected_size() -> None:
    """Each local candidate should get a small fixed 4D grid."""
    for candidate in mirror_local_grid_v3.CANDIDATES:
        assert len(mirror_local_grid_v3._grid_points(candidate.point)) == 216


def test_local_grid_respects_midpoint_floor() -> None:
    """The floor-aware s-values should keep all local grid points above m=0.01."""
    for candidate in mirror_local_grid_v3.CANDIDATES:
        for point in mirror_local_grid_v3._grid_points(candidate.point):
            assert point.s > mirror_local_grid_v3.S_MIN
            assert DEFAULT_CONFIG.match_t * mp.exp(point.s) > mirror_local_grid_v3.MIN_MATCH_T


def test_local_grid_uses_higher_order_verification() -> None:
    """Verification configs should be stronger than the exploratory grid config."""
    assert mirror_local_grid_v3.GRID_CONFIG.series_order == 10
    assert all(config.series_order > mirror_local_grid_v3.GRID_CONFIG.series_order for config in mirror_local_grid_v3.VERIFY_CONFIGS)
