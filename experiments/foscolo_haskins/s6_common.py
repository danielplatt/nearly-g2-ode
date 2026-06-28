"""Maximal-volume matching for the Foscolo-Haskins S6 nearly Kahler ODE.

The variables and normalisations follow Foscolo-Haskins, arXiv:1501.07838,
Sections 3, 4, and 9.  The state is

    (lambda, u0, u1, u2, v0, v1, v2)

with the Minkowski metric of signature (-,+,+) on u and v.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from ..shared.non_mirrored_common import RANDOM_SEED, _event, _write_jsonl_event, _write_summary


MATCH_VERSION = "fh-s6-max-volume-match-v1"
OUTPUT_DIR = Path("output/fh_s6_max_volume_matches")
STATE_FIELDS = ("lambda", "u0", "u1", "u2", "v0", "v1", "v2")
REFLECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
ROUND_TARGET = (math.sqrt(3.0), 1.5)
EXOTIC_TARGET = (0.5646, 0.5985)


@dataclass(frozen=True)
class FHState:
    """One state of the Foscolo-Haskins seven-variable ODE."""

    lambda_: float
    u0: float
    u1: float
    u2: float
    v0: float
    v1: float
    v2: float

    def as_tuple(self) -> tuple[float, float, float, float, float, float, float]:
        """Return the state as a tuple in ODE order."""
        return (self.lambda_, self.u0, self.u1, self.u2, self.v0, self.v1, self.v2)

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "FHState":
        """Build a state from seven numeric values."""
        return cls(*(float(value) for value in values))


@dataclass(frozen=True)
class FHEndpointSeed:
    """Regularized seed near one singular orbit."""

    family: str
    parameter: float
    epsilon: float
    state: FHState
    diagnostics: dict[str, float | str | bool]


@dataclass(frozen=True)
class FHMaxVolumeOrbit:
    """A solution marched from one singular orbit to its maximal-volume orbit."""

    family: str
    parameter: float
    seed_epsilon: float
    seed_state: FHState
    seed_diagnostics: dict[str, float | str | bool]
    elapsed: float
    state: FHState
    w: tuple[float, float, float]
    volume: float
    max_volume_residual: float
    diagnostics: dict[str, float | str | bool]
    status: str


@dataclass(frozen=True)
class FHMatchEvaluation:
    """One pair of maximal-volume orbits and their matching residual."""

    a: float
    b: float
    reflection: tuple[int, int]
    s2_orbit: FHMaxVolumeOrbit
    s3_orbit: FHMaxVolumeOrbit
    residual: tuple[float, float]
    residual_norm: float
    status: str


@dataclass(frozen=True)
class FHNewtonStep:
    """Diagnostic payload for one log-parameter Newton step."""

    index: int
    a: float
    b: float
    residual: tuple[float, float]
    residual_norm: float
    delta: tuple[float, float] | None
    damping: float | None
    trial_norms: tuple[tuple[float, float, float], ...]
    status: str


@dataclass(frozen=True)
class FHMatchRun:
    """Complete result of one FH S6 matching run."""

    target: str
    initial_a: float
    initial_b: float
    final: FHMatchEvaluation
    steps: tuple[FHNewtonStep, ...]
    classification: str


@dataclass(frozen=True)
class FHMarchSettings:
    """Numerical settings for fixed-step RK4 marching to maximal volume."""

    s2_epsilon: float = 0.01
    s3_epsilon: float = 0.01
    step_size: float = 0.001
    max_time: float = 8.0
    bisection_steps: int = 44
    constraint_tolerance: float = 1e-4


@dataclass(frozen=True)
class FHNewtonSettings:
    """Numerical settings for two-parameter max-volume matching."""

    max_steps: int = 8
    tolerance: float = 1e-8
    fd_step: float = 1e-4
    dampings: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)


def _now() -> datetime:
    """Return a filesystem-friendly UTC-ish timestamp source."""
    return datetime.now()


def _output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped JSONL and summary paths."""
    stamp = (now or _now()).strftime("%Y%m%d-%H%M%S-%f")
    base = f"{stamp}-seed{RANDOM_SEED}-{MATCH_VERSION}"
    return OUTPUT_DIR / f"{base}.jsonl", OUTPUT_DIR / f"{base}-summary.json"


def _float(value: float) -> float:
    """Return a JSON-friendly finite float when possible."""
    return float(value)


def _state_payload(state: FHState) -> dict[str, float]:
    """Return a JSON-ready state payload."""
    return {name: _float(value) for name, value in zip(STATE_FIELDS, state.as_tuple())}


def _orbit_payload(orbit: FHMaxVolumeOrbit) -> dict:
    """Return a JSON-ready max-volume orbit payload."""
    return {
        "family": orbit.family,
        "parameter": _float(orbit.parameter),
        "seed_epsilon": _float(orbit.seed_epsilon),
        "seed_state": _state_payload(orbit.seed_state),
        "seed_diagnostics": orbit.seed_diagnostics,
        "elapsed": _float(orbit.elapsed),
        "state": _state_payload(orbit.state),
        "w": [_float(value) for value in orbit.w],
        "volume": _float(orbit.volume),
        "max_volume_residual": _float(orbit.max_volume_residual),
        "diagnostics": orbit.diagnostics,
        "status": orbit.status,
    }


def _evaluation_payload(evaluation: FHMatchEvaluation) -> dict:
    """Return a JSON-ready match evaluation payload."""
    return {
        "a": _float(evaluation.a),
        "b": _float(evaluation.b),
        "reflection": list(evaluation.reflection),
        "s2_orbit": _orbit_payload(evaluation.s2_orbit),
        "s3_orbit": _orbit_payload(evaluation.s3_orbit),
        "residual": [_float(value) for value in evaluation.residual],
        "residual_norm": _float(evaluation.residual_norm),
        "status": evaluation.status,
    }


def _step_payload(step: FHNewtonStep) -> dict:
    """Return a JSON-ready Newton step payload."""
    return {
        "index": step.index,
        "a": _float(step.a),
        "b": _float(step.b),
        "residual": [_float(value) for value in step.residual],
        "residual_norm": _float(step.residual_norm),
        "delta": None if step.delta is None else [_float(value) for value in step.delta],
        "damping": None if step.damping is None else _float(step.damping),
        "trial_norms": [[_float(item) for item in trial] for trial in step.trial_norms],
        "status": step.status,
    }


def _endpoint_seed_payload(orbit: FHMaxVolumeOrbit) -> dict:
    """Return a JSON-ready endpoint seed payload from an orbit."""
    return {
        "family": orbit.family,
        "parameter": _float(orbit.parameter),
        "epsilon": _float(orbit.seed_epsilon),
        "state": _state_payload(orbit.seed_state),
        "diagnostics": orbit.seed_diagnostics,
    }


def _emit_evaluation_events(event_sink: Callable[[str, dict], None], evaluation: FHMatchEvaluation) -> None:
    """Emit endpoint, max-volume, and match events for one accepted evaluation."""
    for orbit in (evaluation.s2_orbit, evaluation.s3_orbit):
        event_sink("endpoint_seed", _endpoint_seed_payload(orbit))
        event_sink("max_volume_orbit", _orbit_payload(orbit))
    event_sink("match_evaluation", _evaluation_payload(evaluation))


def _run_payload(run: FHMatchRun, jsonl_path: Path, summary_path: Path, settings: FHMarchSettings, newton: FHNewtonSettings) -> dict:
    """Return a JSON-ready final run summary."""
    counts = Counter(step.status for step in run.steps)
    return {
        "match_version": MATCH_VERSION,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "target": run.target,
        "initial": {"a": _float(run.initial_a), "b": _float(run.initial_b)},
        "final": _evaluation_payload(run.final),
        "steps": [_step_payload(step) for step in run.steps],
        "step_status_counts": dict(counts),
        "classification": run.classification,
        "march_settings": settings.__dict__,
        "newton_settings": {
            "max_steps": newton.max_steps,
            "tolerance": newton.tolerance,
            "fd_step": newton.fd_step,
            "dampings": list(newton.dampings),
        },
    }


def minkowski_dot(x: tuple[float, float, float], y: tuple[float, float, float]) -> float:
    """Return the signature (-,+,+) inner product."""
    return -x[0] * y[0] + x[1] * y[1] + x[2] * y[2]


def mu_squared(state: FHState) -> float:
    """Return |u|^2 in signature (-,+,+)."""
    return -state.u0 * state.u0 + state.u1 * state.u1 + state.u2 * state.u2


def v_squared(state: FHState) -> float:
    """Return |v|^2 in signature (-,+,+)."""
    return -state.v0 * state.v0 + state.v1 * state.v1 + state.v2 * state.v2


def branch_sign(state: FHState) -> float:
    """Return the FH orientation sign u1*v2-u2*v1."""
    return state.u1 * state.v2 - state.u2 * state.v1


def orbital_volume(state: FHState) -> float:
    """Return the orbital volume V=lambda*mu^2."""
    return state.lambda_ * mu_squared(state)


def max_volume_function(state: FHState) -> float:
    """Return the algebraic maximal-volume event function."""
    return 2.0 * state.lambda_**4 * state.u1 - 3.0 * state.u2 * state.v2


def constraint_residuals(state: FHState) -> tuple[float, float, float, float]:
    """Return the four algebraic constraint residuals."""
    mu2 = mu_squared(state)
    uv = minkowski_dot((state.u0, state.u1, state.u2), (state.v0, state.v1, state.v2))
    return (
        uv,
        state.lambda_ * state.lambda_ * mu2 - state.u2 * state.u2,
        state.lambda_ * state.lambda_ * mu2 - v_squared(state),
        state.v1 - mu2,
    )


def constraint_norm(state: FHState) -> float:
    """Return an infinity norm of the algebraic constraint residuals."""
    return max(abs(value) for value in constraint_residuals(state))


def hyperboloid_projection(state: FHState) -> tuple[float, float, float]:
    """Return FH's unit hyperboloid coordinate w."""
    volume = orbital_volume(state)
    if volume == 0.0:
        raise ValueError("orbital volume is zero")
    return (
        (state.u1 * state.v2 - state.u2 * state.v1) / volume,
        (state.u0 * state.v2 - state.u2 * state.v0) / volume,
        (state.u1 * state.v0 - state.u0 * state.v1) / volume,
    )


def hyperboloid_defect(w: tuple[float, float, float]) -> float:
    """Return the defect from -w0^2+w1^2+w2^2=-1."""
    return -w[0] * w[0] + w[1] * w[1] + w[2] * w[2] + 1.0


def branch_valid(state: FHState) -> bool:
    """Return whether a state satisfies the open FH branch inequalities."""
    return state.lambda_ > 0.0 and state.u2 < 0.0 and mu_squared(state) > 0.0 and branch_sign(state) > 0.0


def ode_rhs(state: FHState) -> FHState:
    """Return the right-hand side of the FH fundamental ODE system."""
    lam = state.lambda_
    mu2 = mu_squared(state)
    if lam == 0.0 or mu2 == 0.0:
        raise ZeroDivisionError("singular FH state")
    lambda_dot = -(2.0 * lam**4 * state.u1 + 3.0 * state.u2 * state.v2) / (lam * lam * mu2)
    return FHState(
        lambda_dot,
        -3.0 * state.v0 / lam,
        (2.0 * lam * lam - 3.0 * state.v1) / lam,
        -3.0 * state.v2 / lam,
        4.0 * lam * state.u0,
        4.0 * lam * state.u1,
        (4.0 * lam * lam * state.u2 - 3.0 * state.u2) / lam,
    )


def _state_add(state: FHState, deriv: FHState, scale: float) -> FHState:
    """Return state + scale * deriv."""
    return FHState.from_iterable(a + scale * b for a, b in zip(state.as_tuple(), deriv.as_tuple()))


def rk4_step(state: FHState, step_size: float) -> FHState:
    """Advance one fixed RK4 step."""
    k1 = ode_rhs(state)
    k2 = ode_rhs(_state_add(state, k1, 0.5 * step_size))
    k3 = ode_rhs(_state_add(state, k2, 0.5 * step_size))
    k4 = ode_rhs(_state_add(state, k3, step_size))
    values = (
        y + step_size * (a + 2.0 * b + 2.0 * c + d) / 6.0
        for y, a, b, c, d in zip(state.as_tuple(), k1.as_tuple(), k2.as_tuple(), k3.as_tuple(), k4.as_tuple())
    )
    return FHState.from_iterable(values)


def s2_seed(a: float, epsilon: float = 0.01) -> FHEndpointSeed:
    """Return the S2-closing Taylor seed Psi_a(epsilon)."""
    if a <= 0.0:
        raise ValueError("a must be positive")
    t = epsilon
    sqrt3 = math.sqrt(3.0)
    a2 = a * a
    a4 = a2 * a2
    a6 = a4 * a2
    state = FHState(
        1.5 * t
        - (2.0 * a2 + 3.0) / (12.0 * a2) * t**3
        + (116.0 * a4 - 381.0 * a2 + 261.0) / (1440.0 * a4) * t**5
        + (5500.0 * a6 - 26523.0 * a4 + 34209.0 * a2 - 13149.0) / (90720.0 * a6) * t**7,
        a2 - 3.0 * a2 * t**2 + (52.0 * a2 - 3.0) / 24.0 * t**4 - (172.0 * a4 + 3.0 * a2 - 18.0) / (270.0 * a2) * t**6,
        a2
        - 1.5 * (2.0 * a2 - 1.0) * t**2
        + (52.0 * a4 - 32.0 * a2 - 3.0) / (24.0 * a2) * t**4
        - (2752.0 * a6 - 1688.0 * a4 + 93.0 * a2 - 261.0) / (4320.0 * a4) * t**6,
        -1.5 * sqrt3 * a * t**2
        + sqrt3 * (16.0 * a2 - 3.0) / (12.0 * a) * t**4
        + sqrt3 * (-3412.0 * a4 + 267.0 * a2 + 423.0) / (8640.0 * a**3) * t**6,
        3.0 * a2 * t**2 - (0.25 + 14.0 * a2 / 3.0) * t**4 + (5516.0 * a4 + 429.0 * a2 + 261.0) / (2160.0 * a2) * t**6,
        3.0 * a2 * t**2 + (2.0 - 14.0 * a2 / 3.0) * t**4 + (5516.0 * a4 - 2541.0 * a2 - 549.0) / (2160.0 * a2) * t**6,
        1.5 * sqrt3 * a * t**2
        - sqrt3 * (34.0 * a2 - 3.0) / (12.0 * a) * t**4
        + sqrt3 * (13492.0 * a4 + 273.0 * a2 - 423.0) / (8640.0 * a**3) * t**6,
    )
    return FHEndpointSeed("s2", a, epsilon, state, seed_diagnostics(state))


def s3_seed(b: float, epsilon: float = 0.01) -> FHEndpointSeed:
    """Return the S3-closing Taylor seed Psi_b from FH's scaled s-series."""
    if b <= 0.0:
        raise ValueError("b must be positive")
    s = epsilon
    b2 = b * b
    b3 = b2 * b
    b4 = b2 * b2
    lam2_tilde = 1.0 - 9.0 / 5.0 * (b2 - 1.0) * s**2 + 27.0 / 35.0 * (b2 - 1.0) * (2.0 * b2 - 1.0) * s**4
    if lam2_tilde <= 0.0:
        raise ValueError("S3 Taylor seed has non-positive lambda^2")
    lambda_tilde = math.sqrt(lam2_tilde)
    u0_tilde = 2.0 * b * s - 4.0 * b3 * s**3 + 6.0 / 25.0 * b3 * (19.0 * b2 - 9.0) * s**5
    u1_tilde = 2.0 * s - 2.0 / 5.0 * (13.0 * b2 - 3.0) * s**3 + 6.0 / 175.0 * (172.0 * b4 - 111.0 * b2 + 9.0) * s**5
    u2_tilde = -2.0 * b * s + b * (4.0 * b2 - 3.0) * s**3 - 3.0 / 100.0 * b * (152.0 * b4 - 192.0 * b2 + 45.0) * s**5
    v0_tilde = -2.0 / 3.0 + 4.0 * b2 * s**2 - 2.0 / 5.0 * b2 * (19.0 * b2 - 9.0) * s**4
    v1_tilde = 4.0 * b * s**2 - 4.0 / 5.0 * b * (11.0 * b2 - 6.0) * s**4
    v2_tilde = 2.0 / 3.0 - (4.0 * b2 - 3.0) * s**2 + 1.0 / 20.0 * (152.0 * b4 - 192.0 * b2 + 45.0) * s**4
    state = FHState(
        b * lambda_tilde,
        b2 * u0_tilde,
        b2 * u1_tilde,
        b2 * u2_tilde,
        b3 * v0_tilde,
        b3 * v1_tilde,
        b3 * v2_tilde,
    )
    return FHEndpointSeed("s3", b, epsilon, state, seed_diagnostics(state))


def seed_diagnostics(state: FHState) -> dict[str, float | str | bool]:
    """Return compact diagnostics for one regularized endpoint seed."""
    return {
        "constraint_norm": constraint_norm(state),
        "mu_squared": mu_squared(state),
        "branch_sign": branch_sign(state),
        "volume": orbital_volume(state),
        "max_volume_function": max_volume_function(state),
        "branch_valid": branch_valid(state),
    }


def _failure_orbit(family: str, parameter: float, seed: FHEndpointSeed, elapsed: float, state: FHState, status: str, message: str) -> FHMaxVolumeOrbit:
    """Return a failed max-volume orbit payload."""
    diagnostics = seed_diagnostics(state)
    diagnostics["message"] = message
    return FHMaxVolumeOrbit(
        family,
        parameter,
        seed.epsilon,
        seed.state,
        seed.diagnostics,
        elapsed,
        state,
        (math.nan, math.nan, math.nan),
        math.nan,
        max_volume_function(state) if all(math.isfinite(value) for value in state.as_tuple()) else math.nan,
        diagnostics,
        status,
    )


def march_to_max_volume(family: str, parameter: float, settings: FHMarchSettings = FHMarchSettings()) -> FHMaxVolumeOrbit:
    """March one endpoint family to its unique maximal-volume orbit."""
    seed = s2_seed(parameter, settings.s2_epsilon) if family == "s2" else s3_seed(parameter, settings.s3_epsilon)
    state = seed.state
    if not branch_valid(state):
        return _failure_orbit(family, parameter, seed, 0.0, state, "seed_invalid", "seed violates FH branch inequalities")
    if constraint_norm(state) > settings.constraint_tolerance:
        return _failure_orbit(family, parameter, seed, 0.0, state, "seed_invalid", "seed constraint drift exceeds tolerance")

    previous = state
    previous_f = max_volume_function(previous)
    elapsed = 0.0
    max_steps = max(1, int(math.ceil(settings.max_time / settings.step_size)))
    for _ in range(max_steps):
        try:
            current = rk4_step(previous, settings.step_size)
        except (ArithmeticError, ValueError, OverflowError) as exc:
            return _failure_orbit(family, parameter, seed, elapsed, previous, "integration_failure", str(exc))
        elapsed += settings.step_size
        current_f = max_volume_function(current)
        if not all(math.isfinite(value) for value in current.as_tuple()):
            return _failure_orbit(family, parameter, seed, elapsed, current, "integration_failure", "non-finite state")
        if not branch_valid(current):
            return _failure_orbit(family, parameter, seed, elapsed, current, "branch_exit", "state left FH branch before maximal volume")
        if previous_f == 0.0 or current_f == 0.0 or previous_f * current_f < 0.0:
            event_elapsed, event_state = _bisect_event(previous, previous_f, settings.step_size, settings.bisection_steps)
            event_total = elapsed - settings.step_size + event_elapsed
            return _successful_orbit(family, parameter, seed, event_total, event_state)
        previous = current
        previous_f = current_f
    return _failure_orbit(family, parameter, seed, elapsed, previous, "max_volume_not_found", "no event before max_time")


def _bisect_event(start: FHState, start_f: float, step_size: float, iterations: int) -> tuple[float, FHState]:
    """Locate a max-volume sign change inside one RK4 step."""
    lo_t = 0.0
    hi_t = step_size
    lo_f = start_f
    hi_state = rk4_step(start, step_size)
    for _ in range(iterations):
        mid_t = 0.5 * (lo_t + hi_t)
        mid_state = rk4_step(start, mid_t)
        mid_f = max_volume_function(mid_state)
        if lo_f == 0.0 or lo_f * mid_f <= 0.0:
            hi_t = mid_t
            hi_state = mid_state
        else:
            lo_t = mid_t
            lo_f = mid_f
    return hi_t, hi_state


def _successful_orbit(family: str, parameter: float, seed: FHEndpointSeed, elapsed: float, state: FHState) -> FHMaxVolumeOrbit:
    """Return a successful max-volume orbit payload."""
    w = hyperboloid_projection(state)
    diagnostics = seed_diagnostics(state)
    diagnostics["hyperboloid_defect"] = hyperboloid_defect(w)
    diagnostics["seed_constraint_norm"] = seed.diagnostics["constraint_norm"]
    diagnostics["seed_max_volume_function"] = seed.diagnostics["max_volume_function"]
    return FHMaxVolumeOrbit(
        family,
        parameter,
        seed.epsilon,
        seed.state,
        seed.diagnostics,
        elapsed,
        state,
        w,
        orbital_volume(state),
        max_volume_function(state),
        diagnostics,
        "ok",
    )


def reflected_residual(
    s2_w: tuple[float, float, float],
    s3_w: tuple[float, float, float],
    reflection: tuple[int, int],
) -> tuple[float, float]:
    """Return residual after reflecting the S2 point in the w1/w2 axes."""
    return (reflection[0] * s2_w[1] - s3_w[1], reflection[1] * s2_w[2] - s3_w[2])


def residual_norm(residual: tuple[float, float]) -> float:
    """Return Euclidean norm for a two-dimensional match residual."""
    return math.hypot(residual[0], residual[1])


def _parse_reflection(value: str) -> tuple[int, int] | None:
    """Parse a reflection string."""
    if value == "auto":
        return None
    aliases = {
        "++": (1, 1),
        "+-": (1, -1),
        "-+": (-1, 1),
        "--": (-1, -1),
        "1,1": (1, 1),
        "1,-1": (1, -1),
        "-1,1": (-1, 1),
        "-1,-1": (-1, -1),
    }
    if value not in aliases:
        raise argparse.ArgumentTypeError("reflection must be auto, ++, +-, -+, --, or comma signs")
    return aliases[value]


def best_reflection(s2_w: tuple[float, float, float], s3_w: tuple[float, float, float]) -> tuple[int, int]:
    """Return the reflection with the smallest w1/w2 residual."""
    return min(REFLECTIONS, key=lambda refl: residual_norm(reflected_residual(s2_w, s3_w, refl)))


def evaluate_match(a: float, b: float, reflection: tuple[int, int] | None = None, settings: FHMarchSettings = FHMarchSettings()) -> FHMatchEvaluation:
    """Evaluate the S2/S3 maximal-volume matching residual."""
    s2_orbit = march_to_max_volume("s2", a, settings)
    s3_orbit = march_to_max_volume("s3", b, settings)
    status = "ok" if s2_orbit.status == "ok" and s3_orbit.status == "ok" else "failed"
    actual_reflection = reflection
    if status == "ok" and actual_reflection is None:
        actual_reflection = best_reflection(s2_orbit.w, s3_orbit.w)
    if status == "ok" and actual_reflection is not None:
        residual = reflected_residual(s2_orbit.w, s3_orbit.w, actual_reflection)
        norm = residual_norm(residual)
    else:
        actual_reflection = reflection or (1, 1)
        residual = (math.inf, math.inf)
        norm = math.inf
    return FHMatchEvaluation(a, b, actual_reflection, s2_orbit, s3_orbit, residual, norm, status)


def _solve_2x2(matrix: tuple[tuple[float, float], tuple[float, float]], rhs: tuple[float, float]) -> tuple[float, float] | None:
    """Solve a 2x2 linear system."""
    (a, b), (c, d) = matrix
    det = a * d - b * c
    if abs(det) < 1e-18 or not math.isfinite(det):
        return None
    return ((rhs[0] * d - b * rhs[1]) / det, (a * rhs[1] - rhs[0] * c) / det)


def recover_match(
    target: str,
    initial_a: float,
    initial_b: float,
    reflection: tuple[int, int] | None = None,
    settings: FHMarchSettings = FHMarchSettings(),
    newton: FHNewtonSettings = FHNewtonSettings(),
    event_sink: Callable[[str, dict], None] | None = None,
) -> FHMatchRun:
    """Recover one S6 max-volume intersection by Newton in log(a), log(b)."""
    log_a = math.log(initial_a)
    log_b = math.log(initial_b)
    first = evaluate_match(math.exp(log_a), math.exp(log_b), reflection, settings)
    actual_reflection = first.reflection
    steps: list[FHNewtonStep] = []
    current = first
    if event_sink is not None:
        _emit_evaluation_events(event_sink, current)
    for index in range(newton.max_steps):
        if current.status != "ok" or current.residual_norm <= newton.tolerance:
            steps.append(FHNewtonStep(index, current.a, current.b, current.residual, current.residual_norm, None, None, (), "converged" if current.status == "ok" else "failed"))
            break
        base_residual = current.residual
        plus_a = evaluate_match(math.exp(log_a + newton.fd_step), math.exp(log_b), actual_reflection, settings)
        plus_b = evaluate_match(math.exp(log_a), math.exp(log_b + newton.fd_step), actual_reflection, settings)
        if plus_a.status != "ok" or plus_b.status != "ok":
            steps.append(FHNewtonStep(index, current.a, current.b, current.residual, current.residual_norm, None, None, (), "fd_failed"))
            break
        jacobian = (
            ((plus_a.residual[0] - base_residual[0]) / newton.fd_step, (plus_b.residual[0] - base_residual[0]) / newton.fd_step),
            ((plus_a.residual[1] - base_residual[1]) / newton.fd_step, (plus_b.residual[1] - base_residual[1]) / newton.fd_step),
        )
        delta = _solve_2x2(jacobian, (-base_residual[0], -base_residual[1]))
        if delta is None:
            steps.append(FHNewtonStep(index, current.a, current.b, current.residual, current.residual_norm, None, None, (), "singular_jacobian"))
            break
        trial_norms: list[tuple[float, float, float]] = []
        accepted: FHMatchEvaluation | None = None
        accepted_damping: float | None = None
        for damping in newton.dampings:
            trial_a = math.exp(log_a + damping * delta[0])
            trial_b = math.exp(log_b + damping * delta[1])
            trial = evaluate_match(trial_a, trial_b, actual_reflection, settings)
            trial_norms.append((damping, trial.a, trial.residual_norm))
            if trial.status == "ok" and trial.residual_norm < current.residual_norm:
                accepted = trial
                accepted_damping = damping
                break
        if accepted is None or accepted_damping is None:
            steps.append(FHNewtonStep(index, current.a, current.b, current.residual, current.residual_norm, delta, None, tuple(trial_norms), "no_improving_damping"))
            break
        steps.append(FHNewtonStep(index, current.a, current.b, current.residual, current.residual_norm, delta, accepted_damping, tuple(trial_norms), "accepted"))
        log_a = math.log(accepted.a)
        log_b = math.log(accepted.b)
        current = accepted
        if event_sink is not None:
            event_sink("newton_step", _step_payload(steps[-1]))
            _emit_evaluation_events(event_sink, current)
    classification = classify_match(target, current)
    return FHMatchRun(target, initial_a, initial_b, current, tuple(steps), classification)


def classify_match(target: str, evaluation: FHMatchEvaluation) -> str:
    """Classify a recovered FH max-volume match."""
    if evaluation.status != "ok":
        return "failed"
    if evaluation.residual_norm > 1e-6:
        return "inconclusive"
    if target == "round" or (abs(evaluation.a - ROUND_TARGET[0]) < 1e-3 and abs(evaluation.b - ROUND_TARGET[1]) < 1e-3):
        return "recovered_round_s6"
    if target == "exotic" or (abs(evaluation.a - EXOTIC_TARGET[0]) < 5e-3 and abs(evaluation.b - EXOTIC_TARGET[1]) < 5e-3):
        return "recovered_exotic_s6"
    return "possible_other_s6_match"


def round_state(t: float) -> FHState:
    """Return the explicit round S6 state from FH."""
    c = math.cos(t)
    s = math.sin(t)
    return FHState(
        1.5 * c,
        -1.5 * s * (2.0 - 5.0 * c * c),
        -3.0 * s * (1.0 - 2.0 * c * c),
        -4.5 * s * c * c,
        2.25 * c * c * (4.0 - 5.0 * c * c),
        9.0 * s * s * c * c,
        2.25 * c * c * (3.0 * c * c - 2.0),
    )


def round_max_volume_time() -> float:
    """Return the explicit round S6 maximal-volume time from the S3 end."""
    return math.atan(math.sqrt(2.0 / 3.0))


def _run_start_payload(
    jsonl_path: Path,
    summary_path: Path,
    mode: str,
    target: str,
    a: float,
    b: float,
    reflection: tuple[int, int] | None,
    march_settings: FHMarchSettings,
    newton_settings: FHNewtonSettings,
) -> dict:
    """Return JSON-ready run-start metadata."""
    return {
        "match_version": MATCH_VERSION,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "mode": mode,
        "target": target,
        "initial": {"a": a, "b": b},
        "reflection": "auto" if reflection is None else list(reflection),
        "march_settings": march_settings.__dict__,
        "newton_settings": {
            "max_steps": newton_settings.max_steps,
            "tolerance": newton_settings.tolerance,
            "fd_step": newton_settings.fd_step,
            "dampings": list(newton_settings.dampings),
        },
    }


def _target_initial(target: str) -> tuple[float, float]:
    """Return a good default initial guess for one target."""
    if target == "round":
        return (1.6, 1.4)
    if target == "exotic":
        return (0.55, 0.6)
    raise ValueError(f"unknown target {target!r}")


def _positive_float(value: str) -> float:
    """Parse a positive float CLI value."""
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for FH S6 maximal-volume matching."""
    parser = argparse.ArgumentParser(description="Recover Foscolo-Haskins S6 max-volume matches.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--recover-round", action="store_true", help="recover the round S6 max-volume match")
    mode.add_argument("--recover-exotic", action="store_true", help="recover the inhomogeneous FH S6 max-volume match")
    mode.add_argument("--evaluate", action="store_true", help="evaluate one a,b pair without Newton refinement")
    parser.add_argument("--a", type=_positive_float, default=None, help="S2-family parameter a")
    parser.add_argument("--b", type=_positive_float, default=None, help="S3-family parameter b")
    parser.add_argument("--reflection", type=_parse_reflection, default=None, help="auto, ++, +-, -+, --; default auto")
    parser.add_argument("--step-size", type=_positive_float, default=FHMarchSettings.step_size, help="RK4 step size")
    parser.add_argument("--s2-epsilon", type=_positive_float, default=FHMarchSettings.s2_epsilon, help="S2 Taylor seed epsilon")
    parser.add_argument("--s3-epsilon", type=_positive_float, default=FHMarchSettings.s3_epsilon, help="S3 Taylor seed epsilon")
    parser.add_argument("--max-time", type=_positive_float, default=FHMarchSettings.max_time, help="max march time before failure")
    parser.add_argument("--tolerance", type=_positive_float, default=FHNewtonSettings.tolerance, help="Newton residual tolerance")
    parser.add_argument("--max-steps", type=int, default=FHNewtonSettings.max_steps, help="maximum Newton steps")
    parser.add_argument("--fd-step", type=_positive_float, default=FHNewtonSettings.fd_step, help="log-parameter finite-difference step")
    parser.add_argument("--dry-run", action="store_true", help="print the planned run without writing output")
    args = parser.parse_args(argv)

    target = "custom"
    if args.recover_round:
        target = "round"
    elif args.recover_exotic:
        target = "exotic"
    if target in {"round", "exotic"}:
        default_a, default_b = _target_initial(target)
        initial_a = args.a if args.a is not None else default_a
        initial_b = args.b if args.b is not None else default_b
        mode_name = f"recover-{target}"
    else:
        initial_a = args.a if args.a is not None else ROUND_TARGET[0]
        initial_b = args.b if args.b is not None else ROUND_TARGET[1]
        mode_name = "evaluate"

    march_settings = FHMarchSettings(args.s2_epsilon, args.s3_epsilon, args.step_size, args.max_time)
    newton_settings = FHNewtonSettings(args.max_steps, args.tolerance, args.fd_step)
    if args.dry_run:
        print(f"mode: {mode_name}", flush=True)
        print(f"initial a,b: {initial_a:.12g}, {initial_b:.12g}", flush=True)
        print(f"reflection: {'auto' if args.reflection is None else args.reflection}", flush=True)
        print(f"march settings: {march_settings}", flush=True)
        return

    jsonl_path, summary_path = _output_paths()
    print(f"writing JSONL events to {jsonl_path}", flush=True)
    _write_jsonl_event(
        jsonl_path,
        _event("run_start", _run_start_payload(jsonl_path, summary_path, mode_name, target, initial_a, initial_b, args.reflection, march_settings, newton_settings)),
    )

    def sink(name: str, payload: dict) -> None:
        _write_jsonl_event(jsonl_path, _event(name, payload))

    do_evaluate = args.evaluate or not (args.recover_round or args.recover_exotic)
    if do_evaluate:
        evaluation = evaluate_match(initial_a, initial_b, args.reflection, march_settings)
        _emit_evaluation_events(sink, evaluation)
        classification = classify_match(target, evaluation)
        run = FHMatchRun(target, initial_a, initial_b, evaluation, (), classification)
    else:
        run = recover_match(target, initial_a, initial_b, args.reflection, march_settings, newton_settings, sink)
    summary = _run_payload(run, jsonl_path, summary_path, march_settings, newton_settings)
    _write_jsonl_event(jsonl_path, _event("solution_classification", {"classification": run.classification, "residual_norm": run.final.residual_norm}))
    _write_jsonl_event(jsonl_path, _event("run_summary", summary))
    _write_summary(summary_path, summary)
    print(f"classification: {run.classification}", flush=True)
    print(f"final a={run.final.a:.12g} b={run.final.b:.12g} residual={run.final.residual_norm:.6g} reflection={run.final.reflection}", flush=True)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
