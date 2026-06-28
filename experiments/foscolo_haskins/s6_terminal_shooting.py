"""Naive two-ended terminal shooting for Foscolo-Haskins S6.

This is intentionally less geometric than the maximal-volume matcher.  It
starts from the S3 and S2 singular Taylor seeds, marches both for the same
interior time, applies a fixed terminal symmetry to the S2 branch, and asks
Gauss-Newton to match the seven ODE variables directly.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from ..shared.non_mirrored_common import RANDOM_SEED, _event, _write_jsonl_event, _write_summary
from . import s6_common as fh


SHOOTING_VERSION = "fh-s6-terminal-shooting-v1"
OUTPUT_DIR = Path("output/fh_s6_terminal_shooting")


@dataclass(frozen=True)
class TerminalShootingSettings:
    """Numerical settings for naive terminal shooting."""

    s2_epsilon: float = 0.01
    s3_epsilon: float = 0.01
    step_size: float = 0.001
    max_steps: int = 8
    tolerance: float = 1e-7
    fd_step: float = 1e-5
    dampings: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)


@dataclass(frozen=True)
class TerminalShootingEvaluation:
    """One direct two-ended residual evaluation."""

    a: float
    b: float
    match_time: float
    transform: str
    left_state: fh.FHState
    right_state: fh.FHState
    transformed_right_state: fh.FHState
    residual: tuple[float, ...]
    residual_norm: float
    diagnostics: dict[str, float | str | bool]
    status: str


@dataclass(frozen=True)
class TerminalNewtonStep:
    """One Gauss-Newton step in log(a), log(b), log(match_time)."""

    index: int
    a: float
    b: float
    match_time: float
    residual_norm: float
    delta: tuple[float, float, float] | None
    damping: float | None
    trial_norms: tuple[tuple[float, float], ...]
    status: str


@dataclass(frozen=True)
class TerminalShootingRun:
    """Complete terminal shooting run result."""

    target: str
    initial_a: float
    initial_b: float
    initial_match_time: float
    transform: str
    final: TerminalShootingEvaluation
    steps: tuple[TerminalNewtonStep, ...]
    classification: str


def transform_identity(state: fh.FHState) -> fh.FHState:
    """Return the state unchanged."""
    return state


def transform_round_terminal(state: fh.FHState) -> fh.FHState:
    """Terminal S2 transform that recovers the round S6 match."""
    return fh.FHState(state.lambda_, -state.u0, -state.u1, state.u2, state.v0, state.v1, -state.v2)


def transform_exotic_terminal(state: fh.FHState) -> fh.FHState:
    """Terminal S2 transform that recovers the inhomogeneous FH S6 match."""
    return fh.FHState(state.lambda_, state.u0, -state.u1, state.u2, -state.v0, state.v1, -state.v2)


TRANSFORMS: dict[str, Callable[[fh.FHState], fh.FHState]] = {
    "identity": transform_identity,
    "round-terminal": transform_round_terminal,
    "exotic-terminal": transform_exotic_terminal,
}


def _output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """Return timestamped JSONL and summary paths."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S-%f")
    base = f"{stamp}-seed{RANDOM_SEED}-{SHOOTING_VERSION}"
    return OUTPUT_DIR / f"{base}.jsonl", OUTPUT_DIR / f"{base}-summary.json"


def _state_payload(state: fh.FHState) -> dict[str, float]:
    """Return a JSON-ready state payload."""
    return {name: float(value) for name, value in zip(fh.STATE_FIELDS, state.as_tuple())}


def _march_for_time(seed: fh.FHEndpointSeed, duration: float, step_size: float) -> fh.FHState:
    """March one seed for a fixed positive duration."""
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    state = seed.state
    full_steps = int(duration // step_size)
    remainder = duration - full_steps * step_size
    for _ in range(full_steps):
        state = fh.rk4_step(state, step_size)
    if remainder > 1e-15:
        state = fh.rk4_step(state, remainder)
    return state


def _diagnostics(left: fh.FHState, right: fh.FHState, transformed_right: fh.FHState) -> dict[str, float | str | bool]:
    """Return compact diagnostics for one direct terminal evaluation."""
    return {
        "left_constraint_norm": fh.constraint_norm(left),
        "right_constraint_norm": fh.constraint_norm(right),
        "transformed_right_constraint_norm": fh.constraint_norm(transformed_right),
        "left_branch_valid": fh.branch_valid(left),
        "right_branch_valid": fh.branch_valid(right),
        "left_volume": fh.orbital_volume(left),
        "right_volume": fh.orbital_volume(right),
    }


def evaluate_terminal_shooting(
    a: float,
    b: float,
    match_time: float,
    transform_name: str,
    settings: TerminalShootingSettings = TerminalShootingSettings(),
) -> TerminalShootingEvaluation:
    """Evaluate the naive terminal shooting residual."""
    if transform_name not in TRANSFORMS:
        raise ValueError(f"unknown terminal transform {transform_name!r}")
    try:
        left_seed = fh.s3_seed(b, settings.s3_epsilon)
        right_seed = fh.s2_seed(a, settings.s2_epsilon)
        left = _march_for_time(left_seed, match_time, settings.step_size)
        right = _march_for_time(right_seed, match_time, settings.step_size)
        transformed = TRANSFORMS[transform_name](right)
        residual = tuple(x - y for x, y in zip(left.as_tuple(), transformed.as_tuple()))
        norm = math.sqrt(sum(value * value for value in residual))
        status = "ok" if all(math.isfinite(value) for value in residual) else "failed"
        diagnostics = _diagnostics(left, right, transformed)
        if not fh.branch_valid(left) or not fh.branch_valid(right):
            status = "branch_exit"
    except (ArithmeticError, ValueError, OverflowError) as exc:
        nan_state = fh.FHState(*(math.nan for _ in range(7)))
        return TerminalShootingEvaluation(
            a,
            b,
            match_time,
            transform_name,
            nan_state,
            nan_state,
            nan_state,
            (math.inf,) * 7,
            math.inf,
            {"message": str(exc)},
            "failed",
        )
    return TerminalShootingEvaluation(a, b, match_time, transform_name, left, right, transformed, residual, norm, diagnostics, status)


def _solve_least_squares(jacobian: np.ndarray, residual: np.ndarray) -> tuple[float, float, float] | None:
    """Solve the linearized least-squares Newton equation."""
    try:
        delta, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        return None
    return (float(delta[0]), float(delta[1]), float(delta[2]))


def recover_terminal_shooting(
    target: str,
    initial_a: float,
    initial_b: float,
    initial_match_time: float,
    transform_name: str,
    settings: TerminalShootingSettings = TerminalShootingSettings(),
    event_sink: Callable[[str, dict], None] | None = None,
) -> TerminalShootingRun:
    """Recover a direct two-ended terminal match with damped Gauss-Newton."""
    x = np.log(np.array([initial_a, initial_b, initial_match_time], dtype=float))
    current = evaluate_terminal_shooting(math.exp(x[0]), math.exp(x[1]), math.exp(x[2]), transform_name, settings)
    steps: list[TerminalNewtonStep] = []
    if event_sink is not None:
        event_sink("shooting_evaluation", _evaluation_payload(current))
    for index in range(settings.max_steps):
        if current.status != "ok" or current.residual_norm <= settings.tolerance:
            steps.append(
                TerminalNewtonStep(
                    index,
                    current.a,
                    current.b,
                    current.match_time,
                    current.residual_norm,
                    None,
                    None,
                    (),
                    "converged" if current.status == "ok" else current.status,
                )
            )
            break
        base = np.array(current.residual, dtype=float)
        columns: list[np.ndarray] = []
        fd_failed = False
        for coordinate in range(3):
            trial_x = x.copy()
            trial_x[coordinate] += settings.fd_step
            trial = evaluate_terminal_shooting(math.exp(trial_x[0]), math.exp(trial_x[1]), math.exp(trial_x[2]), transform_name, settings)
            if trial.status != "ok":
                fd_failed = True
                break
            columns.append((np.array(trial.residual, dtype=float) - base) / settings.fd_step)
        if fd_failed:
            steps.append(TerminalNewtonStep(index, current.a, current.b, current.match_time, current.residual_norm, None, None, (), "fd_failed"))
            break
        jacobian = np.column_stack(columns)
        delta = _solve_least_squares(jacobian, base)
        if delta is None:
            steps.append(TerminalNewtonStep(index, current.a, current.b, current.match_time, current.residual_norm, None, None, (), "singular_jacobian"))
            break
        accepted: TerminalShootingEvaluation | None = None
        accepted_damping: float | None = None
        trial_norms: list[tuple[float, float]] = []
        for damping in settings.dampings:
            trial_x = x + damping * np.array(delta)
            trial = evaluate_terminal_shooting(math.exp(trial_x[0]), math.exp(trial_x[1]), math.exp(trial_x[2]), transform_name, settings)
            trial_norms.append((float(damping), float(trial.residual_norm)))
            if trial.status == "ok" and trial.residual_norm < current.residual_norm:
                accepted = trial
                accepted_damping = float(damping)
                break
        if accepted is None or accepted_damping is None:
            steps.append(
                TerminalNewtonStep(index, current.a, current.b, current.match_time, current.residual_norm, delta, None, tuple(trial_norms), "no_improving_damping")
            )
            break
        steps.append(TerminalNewtonStep(index, current.a, current.b, current.match_time, current.residual_norm, delta, accepted_damping, tuple(trial_norms), "accepted"))
        x = np.log(np.array([accepted.a, accepted.b, accepted.match_time], dtype=float))
        current = accepted
        if event_sink is not None:
            event_sink("newton_step", _step_payload(steps[-1]))
            event_sink("shooting_evaluation", _evaluation_payload(current))
    classification = classify_terminal_run(target, current)
    return TerminalShootingRun(target, initial_a, initial_b, initial_match_time, transform_name, current, tuple(steps), classification)


def classify_terminal_run(target: str, evaluation: TerminalShootingEvaluation) -> str:
    """Classify one terminal shooting result."""
    if evaluation.status != "ok":
        return "failed"
    if evaluation.residual_norm > 1e-6:
        return "inconclusive"
    if target == "round" or (abs(evaluation.a - fh.ROUND_TARGET[0]) < 1e-4 and abs(evaluation.b - fh.ROUND_TARGET[1]) < 1e-4):
        return "recovered_round_s6"
    if target == "exotic" or (abs(evaluation.a - fh.EXOTIC_TARGET[0]) < 5e-3 and abs(evaluation.b - fh.EXOTIC_TARGET[1]) < 5e-3):
        return "recovered_exotic_s6"
    return "possible_other_s6_terminal_match"


def _evaluation_payload(evaluation: TerminalShootingEvaluation) -> dict:
    """Return JSON-ready evaluation payload."""
    return {
        "a": float(evaluation.a),
        "b": float(evaluation.b),
        "match_time": float(evaluation.match_time),
        "transform": evaluation.transform,
        "left_state": _state_payload(evaluation.left_state),
        "right_state": _state_payload(evaluation.right_state),
        "transformed_right_state": _state_payload(evaluation.transformed_right_state),
        "residual": [float(value) for value in evaluation.residual],
        "residual_norm": float(evaluation.residual_norm),
        "diagnostics": evaluation.diagnostics,
        "status": evaluation.status,
    }


def _step_payload(step: TerminalNewtonStep) -> dict:
    """Return JSON-ready Newton step payload."""
    return {
        "index": step.index,
        "a": float(step.a),
        "b": float(step.b),
        "match_time": float(step.match_time),
        "residual_norm": float(step.residual_norm),
        "delta": None if step.delta is None else [float(value) for value in step.delta],
        "damping": step.damping,
        "trial_norms": [[float(value) for value in item] for item in step.trial_norms],
        "status": step.status,
    }


def _summary_payload(run: TerminalShootingRun, jsonl_path: Path, summary_path: Path, settings: TerminalShootingSettings) -> dict:
    """Return JSON-ready run summary."""
    return {
        "shooting_version": SHOOTING_VERSION,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "target": run.target,
        "initial": {"a": run.initial_a, "b": run.initial_b, "match_time": run.initial_match_time},
        "transform": run.transform,
        "final": _evaluation_payload(run.final),
        "steps": [_step_payload(step) for step in run.steps],
        "step_status_counts": dict(Counter(step.status for step in run.steps)),
        "classification": run.classification,
        "settings": {
            "s2_epsilon": settings.s2_epsilon,
            "s3_epsilon": settings.s3_epsilon,
            "step_size": settings.step_size,
            "max_steps": settings.max_steps,
            "tolerance": settings.tolerance,
            "fd_step": settings.fd_step,
            "dampings": list(settings.dampings),
        },
    }


def _target_defaults(target: str) -> tuple[float, float, float, str]:
    """Return a default starting guess and terminal transform."""
    if target == "round":
        return (1.6, 1.4, 0.75, "round-terminal")
    if target == "exotic":
        return (0.55, 0.6, 1.2, "exotic-terminal")
    raise ValueError(f"unknown target {target!r}")


def _positive_float(value: str) -> float:
    """Parse a positive CLI float."""
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _transform_name(value: str) -> str:
    """Parse one terminal transform name."""
    if value not in TRANSFORMS:
        raise argparse.ArgumentTypeError(f"transform must be one of {', '.join(sorted(TRANSFORMS))}")
    return value


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Naive FH S6 terminal singular shooting.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--recover-round", action="store_true", help="recover the round S6 terminal match")
    mode.add_argument("--recover-exotic", action="store_true", help="recover the inhomogeneous FH S6 terminal match")
    mode.add_argument("--evaluate", action="store_true", help="evaluate one terminal-shooting residual without Newton")
    parser.add_argument("--a", type=_positive_float, default=None, help="S2 endpoint parameter")
    parser.add_argument("--b", type=_positive_float, default=None, help="S3 endpoint parameter")
    parser.add_argument("--match-time", type=_positive_float, default=None, help="interior march time from both seeds")
    parser.add_argument("--transform", type=_transform_name, default=None, help="terminal transform")
    parser.add_argument("--step-size", type=_positive_float, default=TerminalShootingSettings.step_size, help="RK4 step size")
    parser.add_argument("--s2-epsilon", type=_positive_float, default=TerminalShootingSettings.s2_epsilon, help="S2 Taylor seed epsilon")
    parser.add_argument("--s3-epsilon", type=_positive_float, default=TerminalShootingSettings.s3_epsilon, help="S3 Taylor seed epsilon")
    parser.add_argument("--max-steps", type=int, default=TerminalShootingSettings.max_steps, help="maximum Newton steps")
    parser.add_argument("--tolerance", type=_positive_float, default=TerminalShootingSettings.tolerance, help="Newton residual tolerance")
    parser.add_argument("--fd-step", type=_positive_float, default=TerminalShootingSettings.fd_step, help="finite-difference step in log coordinates")
    parser.add_argument("--dry-run", action="store_true", help="print planned run without writing output")
    args = parser.parse_args(argv)

    target = "custom"
    if args.recover_round:
        target = "round"
    elif args.recover_exotic:
        target = "exotic"
    if target in {"round", "exotic"}:
        default_a, default_b, default_time, default_transform = _target_defaults(target)
    else:
        default_a, default_b, default_time, default_transform = (fh.ROUND_TARGET[0], fh.ROUND_TARGET[1], 0.77289845, "round-terminal")
    a = args.a if args.a is not None else default_a
    b = args.b if args.b is not None else default_b
    match_time = args.match_time if args.match_time is not None else default_time
    transform = args.transform if args.transform is not None else default_transform
    mode_name = "evaluate" if args.evaluate or target == "custom" else f"recover-{target}"
    settings = TerminalShootingSettings(args.s2_epsilon, args.s3_epsilon, args.step_size, args.max_steps, args.tolerance, args.fd_step)
    if args.dry_run:
        print(f"mode: {mode_name}", flush=True)
        print(f"initial a,b,match_time: {a:.12g}, {b:.12g}, {match_time:.12g}", flush=True)
        print(f"transform: {transform}", flush=True)
        print(f"settings: {settings}", flush=True)
        return

    jsonl_path, summary_path = _output_paths()
    print(f"writing JSONL events to {jsonl_path}", flush=True)
    _write_jsonl_event(
        jsonl_path,
        _event(
            "run_start",
            {
                "shooting_version": SHOOTING_VERSION,
                "mode": mode_name,
                "target": target,
                "initial": {"a": a, "b": b, "match_time": match_time},
                "transform": transform,
                "settings": settings.__dict__,
            },
        ),
    )

    def sink(name: str, payload: dict) -> None:
        _write_jsonl_event(jsonl_path, _event(name, payload))

    if args.evaluate or target == "custom":
        evaluation = evaluate_terminal_shooting(a, b, match_time, transform, settings)
        sink("shooting_evaluation", _evaluation_payload(evaluation))
        run = TerminalShootingRun(target, a, b, match_time, transform, evaluation, (), classify_terminal_run(target, evaluation))
    else:
        run = recover_terminal_shooting(target, a, b, match_time, transform, settings, sink)
    summary = _summary_payload(run, jsonl_path, summary_path, settings)
    _write_jsonl_event(jsonl_path, _event("solution_classification", {"classification": run.classification, "residual_norm": run.final.residual_norm}))
    _write_jsonl_event(jsonl_path, _event("run_summary", summary))
    _write_summary(summary_path, summary)
    print(f"classification: {run.classification}", flush=True)
    print(
        f"final a={run.final.a:.12g} b={run.final.b:.12g} match_time={run.final.match_time:.12g} residual={run.final.residual_norm:.6g} transform={run.transform}",
        flush=True,
    )
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()

