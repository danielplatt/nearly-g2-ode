"""Tests for the S7 SU(2)^3 Podesta scout."""

from __future__ import annotations

import math

import experiments.s7_su2_cubed_scout
from experiments.s7 import su2_cubed_scout as scout


def test_endpoint_initial_data_satisfies_smoothness_equations() -> None:
    """The regular h data should encode Podesta's smooth singular endpoint."""
    h = scout.initial_h(scout.SQUASHED_A_DIRECT)
    h4 = scout.h4_from_h(h)
    assert h[1] == 27.0 / 4.0
    assert abs(h[2] + scout.SQUASHED_A_DIRECT**3 / 27.0) < 1e-12
    assert abs(h[3] - 3.0 * scout.SQUASHED_A_DIRECT) < 1e-12
    assert abs(h4 + h[3] + h[0] ** 2 / 6.0) < 1e-12


def test_known_direct_parameters_recover_round_and_squashed() -> None:
    """The one-parameter closure proxy should recover the known compact S7 cases."""
    settings = scout.PodestaSettings(step_size=1e-3)
    round_eval = scout.evaluate_a(scout.ROUND_A_DIRECT, settings)
    squashed_eval = scout.evaluate_a(scout.SQUASHED_A_DIRECT, settings)
    assert round_eval.classification == "recovered_round_s7"
    assert squashed_eval.classification == "recovered_squashed_s7"
    assert round_eval.endpoint_loss < 1e-3
    assert squashed_eval.endpoint_loss < 1e-3
    assert abs(round_eval.endpoint_time - 2.0 * math.pi) < 0.05
    assert abs(squashed_eval.endpoint_time - 6.0 * math.pi / math.sqrt(5.0)) < 0.05


def test_standard_terminal_chart_rejects_opposite_round_sign() -> None:
    """The positive round sign needs the outer automorphism, not the standard K- chart."""
    evaluation = scout.evaluate_a(scout.ROUND_A_CANONICAL)
    assert evaluation.classification != "recovered_round_s7"
    assert evaluation.endpoint_loss > 0.5


def test_degenerate_near_zero_seed_is_not_a_terminal_closure() -> None:
    """The initial left seed should not be mistaken for the far endpoint."""
    evaluation = scout.evaluate_a(0.2)
    assert evaluation.classification == "failed"
    assert evaluation.endpoint_time is None or evaluation.endpoint_time >= 0.1


def test_axis_and_local_minima_are_deterministic() -> None:
    """The default grid should include both direct homogeneous parameters."""
    axis = scout.axis_values(-60.0, 60.0, 0.2)
    assert len(axis) == 601
    assert any(abs(value - scout.ROUND_A_DIRECT) < 1e-12 for value in axis)
    assert any(abs(value - scout.SQUASHED_A_DIRECT) < 1e-12 for value in axis)
    evaluations = [
        scout.PodestaEvaluation(-1.0, "ok", "inconclusive", 2.0, None, None, None, 0),
        scout.PodestaEvaluation(0.0, "ok", "inconclusive", 1.0, None, None, None, 0),
        scout.PodestaEvaluation(1.0, "ok", "inconclusive", 3.0, None, None, None, 0),
    ]
    assert [item.a for item in scout.local_minima(evaluations)] == [0.0]


def test_recovery_smoke_and_top_level_shim() -> None:
    """The recovery smoke should be JSON-friendly and exposed by the top-level shim."""
    payload = scout.recovery_smoke()
    assert payload["classification_counts"] == {"recovered_round_s7": 1, "recovered_squashed_s7": 1}
    assert experiments.s7_su2_cubed_scout.main is scout.main
