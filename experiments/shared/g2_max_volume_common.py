"""Shared payload helpers for G2 maximal-volume experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mpmath import mp

from problem import DEFAULT_CONFIG, SolverConfig, State
from solver.max_volume import MaxVolumeMatchResult, MaxVolumeSettings, MaxVolumeSideResult

from .non_mirrored_common import _mp_string


CALIBRATION_CONFIG = SolverConfig(18, 120, 45, mp.mpf("0.55"), 2, DEFAULT_CONFIG.match_t)
SCOUT_CONFIG = SolverConfig(8, 50, 20, mp.mpf("0.8"), 1, DEFAULT_CONFIG.match_t)
CALIBRATION_SETTINGS = MaxVolumeSettings(CALIBRATION_CONFIG, bisection_steps=56, event_tolerance=mp.mpf("1e-30"))
SCOUT_SETTINGS = MaxVolumeSettings(SCOUT_CONFIG, bisection_steps=36, event_tolerance=mp.mpf("1e-20"))


def state_payload(state: State[mp.mpf] | None) -> list[str | None] | None:
    """Return JSON-ready state components."""
    return None if state is None else [_mp_string(value) for value in state]


def jsonify(value: Any) -> Any:
    """Recursively convert mpmath-heavy payloads into JSON-ready values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, mp.mpf):
        return _mp_string(value)
    if isinstance(value, Mapping):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [jsonify(item) for item in value]
    return str(value)


def settings_payload(settings: MaxVolumeSettings) -> dict:
    """Return JSON-ready max-volume settings."""
    config = settings.config
    return {
        "series_order": config.series_order,
        "working_dps": config.working_dps,
        "target_dps": config.target_dps,
        "step_safety": _mp_string(config.step_safety),
        "sample_points": config.sample_points,
        "match_t": _mp_string(config.match_t),
        "max_tau": _mp_string(settings.max_tau),
        "bisection_steps": settings.bisection_steps,
        "event_tolerance": _mp_string(settings.event_tolerance),
    }


def side_payload(side: MaxVolumeSideResult) -> dict:
    """Return JSON-ready one-sided max-volume result."""
    return {
        "chart_name": side.chart_name,
        "status": side.status,
        "failure": side.failure,
        "max_tau": _mp_string(side.max_tau),
        "physical_t": _mp_string(side.diagnostics.get("physical_t")),
        "volume": _mp_string(side.volume),
        "mean_curvature": _mp_string(side.mean_curvature),
        "max_q": state_payload(side.max_q),
        "patch_count": len(side.patches),
        "diagnostics": jsonify(side.diagnostics),
    }


def params_payload(params) -> dict:
    """Return JSON-ready endpoint parameters."""
    payload = {
        "lambda": _mp_string(params.lam),
        "interval_end": _mp_string(params.interval_end),
        "right_chart": params.right_chart,
        "fixed_right_label": None if params.fixed_right is None else params.fixed_right.label,
        "left": {
            "a": _mp_string(params.left.a),
            "c": _mp_string(params.left.c),
            "alpha": _mp_string(params.left.alpha),
            "mu": params.left_mu,
        },
        "right": {
            "d": _mp_string(params.right.d),
            "f": _mp_string(params.right.f),
            "omega": _mp_string(params.right.omega),
            "mu": params.right_mu,
        },
        "p_signs": list(params.p_signs),
        "right_p_signs": None if params.right_p_signs is None else list(params.right_p_signs),
    }
    return payload


def match_payload(match: MaxVolumeMatchResult) -> dict:
    """Return JSON-ready two-ended max-volume match result."""
    return {
        "failure": match.failure,
        "residual_norm": _mp_string(match.residual_norm),
        "residual": [_mp_string(value) for value in match.residual],
        "reconstructed_interval": _mp_string(match.reconstructed_interval),
        "interval_error": _mp_string(match.interval_error),
        "params": params_payload(match.params),
        "settings": settings_payload(match.settings),
        "left": side_payload(match.left),
        "right": side_payload(match.right),
    }
