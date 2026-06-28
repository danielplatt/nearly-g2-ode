"""Tests for Foscolo-Haskins S6 maximal-volume matching."""

from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.foscolo_haskins import s6_common as fh


def _round_derivative(t: float) -> fh.FHState:
    """Analytic derivative of FH's explicit round S6 solution."""
    c = math.cos(t)
    s = math.sin(t)
    return fh.FHState(
        -1.5 * s,
        -1.5 * c * (2.0 - 5.0 * c * c) - 15.0 * s * s * c,
        -3.0 * c * (1.0 - 2.0 * c * c) - 12.0 * s * s * c,
        -4.5 * c**3 + 9.0 * s * s * c,
        -4.5 * s * c * (4.0 - 5.0 * c * c) + 22.5 * s * c**3,
        18.0 * s * c * (c * c - s * s),
        -4.5 * s * c * (3.0 * c * c - 2.0) - 13.5 * s * c**3,
    )


def test_rhs_matches_explicit_round_solution() -> None:
    """The fundamental ODE should reproduce FH's explicit round S6 curve."""
    state = fh.round_state(0.37)
    rhs = fh.ode_rhs(state)
    expected = _round_derivative(0.37)
    for actual, target in zip(rhs.as_tuple(), expected.as_tuple()):
        assert abs(actual - target) < 1e-10


def test_constraints_vanish_on_round_solution() -> None:
    """The algebraic nearly-hypo constraints should hold on round S6."""
    state = fh.round_state(0.51)
    assert fh.constraint_norm(state) < 1e-12
    assert fh.branch_valid(state)


def test_endpoint_seeds_are_branch_valid() -> None:
    """Both regularized singular seeds should start on the FH branch."""
    s2 = fh.s2_seed(math.sqrt(3.0))
    s3 = fh.s3_seed(1.5)
    assert s2.diagnostics["branch_valid"]
    assert s3.diagnostics["branch_valid"]
    assert s2.diagnostics["constraint_norm"] < 1e-10
    assert s3.diagnostics["constraint_norm"] < 1e-8


def test_march_detects_round_maximal_volume_orbits() -> None:
    """Marching both round endpoint families should hit matching max-volume orbits."""
    s2_orbit = fh.march_to_max_volume("s2", math.sqrt(3.0))
    s3_orbit = fh.march_to_max_volume("s3", 1.5)
    assert s2_orbit.status == "ok"
    assert s3_orbit.status == "ok"
    assert abs(s2_orbit.max_volume_residual) < 1e-9
    assert abs(s3_orbit.max_volume_residual) < 1e-9
    residual = fh.reflected_residual(s2_orbit.w, s3_orbit.w, (1, -1))
    assert fh.residual_norm(residual) < 1e-6


def test_hyperboloid_projection_has_unit_minkowski_norm() -> None:
    """The projected maximal-volume orbit should lie on the unit hyperboloid."""
    orbit = fh.march_to_max_volume("s3", 1.5)
    assert orbit.status == "ok"
    assert abs(fh.hyperboloid_defect(orbit.w)) < 1e-9


def test_best_reflection_for_round_and_exotic_benchmarks() -> None:
    """Round and exotic S6 use the expected FH reflection symmetries."""
    round_eval = fh.evaluate_match(math.sqrt(3.0), 1.5)
    exotic_eval = fh.evaluate_match(0.5646, 0.5985)
    assert round_eval.reflection == (1, -1)
    assert round_eval.residual_norm < 1e-6
    assert exotic_eval.reflection == (-1, 1)
    assert exotic_eval.residual_norm < 1e-3


def test_recover_exotic_match_from_offset_guess() -> None:
    """Newton matching should refine the approximate FH exotic table values."""
    run = fh.recover_match("exotic", 0.55, 0.6)
    assert run.classification == "recovered_exotic_s6"
    assert abs(run.final.a - 0.564550017) < 1e-6
    assert abs(run.final.b - 0.599013546) < 1e-6
    assert run.final.residual_norm < 1e-7


def test_cli_evaluate_writes_summary(tmp_path: Path, monkeypatch) -> None:
    """The top-level CLI should write JSONL and summary artifacts."""
    monkeypatch.setattr(fh, "OUTPUT_DIR", tmp_path)
    fh.main(["--evaluate", "--a", str(math.sqrt(3.0)), "--b", "1.5"])
    summaries = sorted(tmp_path.glob("*-summary.json"))
    jsonls = sorted(tmp_path.glob("*.jsonl"))
    assert len(summaries) == 1
    assert len(jsonls) == 1
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert payload["classification"] == "recovered_round_s6"
    assert payload["final"]["reflection"] == [1, -1]
