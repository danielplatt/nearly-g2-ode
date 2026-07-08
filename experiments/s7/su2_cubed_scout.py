"""One-parameter Podesta SU(2)^3 scout for compact S7 closure.

The equations are Podesta's five-function reduction for the Sp(1)^3 action on
S7.  We fix the nearly-parallel constant to lambda=1.  Smoothness at the left
singular S3 is encoded by the regular variables

    f0=t h0, f1=t^4 h1, f2=h2, f3=t^2 h3, f4=t^2 h4,

with h4=-h3-h0^2/6 and initial data

    h0(0)=a, h1(0)=27/4, h2(0)=-a^3/27, h3(0)=3a.

The compact S7 right endpoint is detected by the K- conditions in the inward
coordinate: f0, f2, f3, and f4 should vanish while f1 stays nonzero.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from ..shared.non_mirrored_common import RANDOM_SEED, _event, _output_paths, _write_jsonl_event, _write_summary


SCOUT_VERSION = "s7-su2-cubed-podesta-scout-v1"
OUTPUT_DIR = Path("output/s7_su2_cubed_scouts")
ROUND_A_DIRECT = -36.0
ROUND_A_CANONICAL = 36.0
SQUASHED_A_DIRECT = 108.0 / 5.0
DEFAULT_A_MIN = -60.0
DEFAULT_A_MAX = 60.0
DEFAULT_SPACING = 0.2


@dataclass(frozen=True)
class PodestaSettings:
    """Fixed numerical settings for one-parameter Podesta shooting."""

    lambda_: float = 1.0
    seed_epsilon: float = 1e-3
    step_size: float = 1e-3
    max_time: float = 12.0
    closure_threshold: float = 1e-3
    local_signal_threshold: float = 0.1
    min_endpoint_time: float = 0.1
    min_terminal_abs_f1: float = 1.0


@dataclass(frozen=True)
class PodestaEvaluation:
    """One evaluated left-end parameter."""

    a: float
    status: str
    classification: str
    endpoint_loss: float
    endpoint_time: float | None
    endpoint_f: tuple[float, float, float, float, float] | None
    f0_cross_time: float | None
    steps: int
    message: str | None = None


@dataclass(frozen=True)
class RefinedMinimum:
    """One scalar local minimization result."""

    bracket: tuple[float, float]
    evaluation: PodestaEvaluation
    iterations: int


def h4_from_h(h: tuple[float, float, float, float], lambda_: float = 1.0) -> float:
    """Return the algebraic h4 component."""
    h0, _h1, _h2, h3 = h
    return -h3 - lambda_ * h0 * h0 / 6.0


def initial_h(a: float, lambda_: float = 1.0) -> tuple[float, float, float, float]:
    """Return the left singular initial h-data."""
    if a == 0.0:
        raise ValueError("a must be nonzero")
    return (a, 27.0 * lambda_ / 4.0, -(a**3) / 27.0, 3.0 * a)


def f_from_h(t: float, h: tuple[float, float, float, float], lambda_: float = 1.0) -> tuple[float, float, float, float, float]:
    """Convert regular h variables to Podesta's f variables."""
    h0, h1, h2, h3 = h
    h4 = h4_from_h(h, lambda_)
    return (t * h0, t**4 * h1, h2, t * t * h3, t * t * h4)


def h_rhs(t: float, h: tuple[float, float, float, float], lambda_: float = 1.0) -> tuple[float, float, float, float]:
    """Return the regularized left-end h-equations."""
    h0, h1, h2, h3 = h
    h4 = h4_from_h(h, lambda_)
    h0_2 = h0 * h0
    h0_3 = h0_2 * h0
    h0_4 = h0_2 * h0_2
    t2 = t * t
    a0 = -h0 - 3.0 * h2 * h3 * h3 / h0_4
    a1 = -4.0 * h1 + lambda_ * h3**3 / h0_3
    a3 = -2.0 * h3 + 6.0 * h0
    b0 = -3.0 / (2.0 * h0_4) * (
        t * (h3 - h4) * (h1 * h2 + h3 * h4) - 2.0 * t**3 * h1 * h4 * h4
    )
    b1 = lambda_ * t / (2.0 * h0_3) * (h1 * h1 * h2 - 3.0 * h1 * h3 * h4)
    b2 = lambda_ * t / h0_3 * (
        h4 * (h2 * h3 - t2 * h4 * h4) - 0.5 * h2 * (h1 * h2 - h3 * h4)
    )
    b3 = lambda_ * t / (2.0 * h0_3) * (
        h1 * h2 * h3 + h3 * h3 * h4 - 2.0 * t2 * h1 * h4 * h4
    )
    return (a0 / t + b0, a1 / t + b1, b2, a3 / t + b3)


def rk4_step(t: float, h: tuple[float, float, float, float], step: float, lambda_: float = 1.0) -> tuple[float, float, float, float]:
    """Advance the h-system by one fixed RK4 step."""
    k1 = h_rhs(t, h, lambda_)
    h2 = tuple(value + 0.5 * step * slope for value, slope in zip(h, k1))
    k2 = h_rhs(t + 0.5 * step, h2, lambda_)
    h3 = tuple(value + 0.5 * step * slope for value, slope in zip(h, k2))
    k3 = h_rhs(t + 0.5 * step, h3, lambda_)
    h4 = tuple(value + step * slope for value, slope in zip(h, k3))
    k4 = h_rhs(t + step, h4, lambda_)
    return tuple(
        value + step * (s1 + 2.0 * s2 + 2.0 * s3 + s4) / 6.0
        for value, s1, s2, s3, s4 in zip(h, k1, k2, k3, k4)
    )


def right_endpoint_loss(f: tuple[float, float, float, float, float]) -> float:
    """Return a scale-normalized proxy for K- endpoint closure."""
    f0, f1, f2, f3, f4 = f
    scale_f = max(1.0, abs(f1))
    scale_f0 = max(1.0, abs(f1) ** (2.0 / 3.0))
    return math.sqrt((f0 / scale_f0) ** 2 + (f2 / scale_f) ** 2 + (f3 / scale_f) ** 2 + (f4 / scale_f) ** 2)


def endpoint_eligible(t: float, f: tuple[float, float, float, float, float], settings: PodestaSettings) -> bool:
    """Return whether a sample can represent the far singular endpoint."""
    return t >= settings.min_endpoint_time and abs(f[1]) >= settings.min_terminal_abs_f1


def classify_evaluation(evaluation: PodestaEvaluation, settings: PodestaSettings = PodestaSettings()) -> str:
    """Classify one endpoint-loss evaluation."""
    if evaluation.status not in {"crossed_f0", "approached_endpoint"}:
        return "failed"
    if evaluation.endpoint_loss > settings.closure_threshold:
        return "inconclusive"
    if abs(evaluation.a - ROUND_A_DIRECT) < settings.step_size * 200.0:
        return "recovered_round_s7"
    if abs(evaluation.a - SQUASHED_A_DIRECT) < settings.step_size * 200.0:
        return "recovered_squashed_s7"
    return "possible_compact_s7_signal"


def evaluate_a(a: float, settings: PodestaSettings = PodestaSettings()) -> PodestaEvaluation:
    """March one left-end parameter and evaluate the best right endpoint approach."""
    if not math.isfinite(a) or abs(a) < 1e-12:
        return PodestaEvaluation(a, "failed", "failed", math.inf, None, None, None, 0, "a must be finite and nonzero")
    try:
        h = initial_h(a, settings.lambda_)
        t = settings.seed_epsilon
        f = f_from_h(t, h, settings.lambda_)
        initial_sign = math.copysign(1.0, f[0])
        best_loss = right_endpoint_loss(f) if endpoint_eligible(t, f, settings) else math.inf
        best_t: float | None = t if math.isfinite(best_loss) else None
        best_f: tuple[float, float, float, float, float] | None = f if math.isfinite(best_loss) else None
        f0_cross_time: float | None = None
        steps = 0
        while t < settings.max_time:
            next_h = rk4_step(t, h, settings.step_size, settings.lambda_)
            next_t = t + settings.step_size
            next_f = f_from_h(next_t, next_h, settings.lambda_)
            if not all(math.isfinite(value) for value in (*next_h, *next_f)):
                break
            loss = right_endpoint_loss(next_f) if endpoint_eligible(next_t, next_f, settings) else math.inf
            if loss < best_loss:
                best_loss = loss
                best_t = next_t
                best_f = next_f
            if math.copysign(1.0, next_f[0]) != initial_sign:
                alpha = abs(f[0]) / (abs(f[0]) + abs(next_f[0]))
                cross_t = t + alpha * settings.step_size
                cross_h = tuple(h_value + alpha * (next_h_value - h_value) for h_value, next_h_value in zip(h, next_h))
                cross_f = f_from_h(cross_t, cross_h, settings.lambda_)
                cross_loss = right_endpoint_loss(cross_f) if endpoint_eligible(cross_t, cross_f, settings) else math.inf
                if cross_loss < best_loss:
                    best_loss = cross_loss
                    best_t = cross_t
                    best_f = cross_f
                f0_cross_time = cross_t
                status = "crossed_f0"
                candidate = PodestaEvaluation(a, status, "pending", best_loss, best_t, best_f, f0_cross_time, steps + 1)
                return candidate.__class__(
                    candidate.a,
                    candidate.status,
                    classify_evaluation(candidate, settings),
                    candidate.endpoint_loss,
                    candidate.endpoint_time,
                    candidate.endpoint_f,
                    candidate.f0_cross_time,
                    candidate.steps,
                    candidate.message,
                )
            t = next_t
            h = next_h
            f = next_f
            steps += 1
        status = "approached_endpoint" if best_loss < settings.local_signal_threshold else "no_endpoint"
        candidate = PodestaEvaluation(a, status, "pending", best_loss, best_t, best_f, f0_cross_time, steps)
        return candidate.__class__(
            candidate.a,
            candidate.status,
            classify_evaluation(candidate, settings),
            candidate.endpoint_loss,
            candidate.endpoint_time,
            candidate.endpoint_f,
            candidate.f0_cross_time,
            candidate.steps,
            candidate.message,
        )
    except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError) as exc:
        return PodestaEvaluation(a, "failed", "failed", math.inf, None, None, None, 0, str(exc))


def axis_values(a_min: float, a_max: float, spacing: float) -> tuple[float, ...]:
    """Return an inclusive deterministic one-dimensional axis."""
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    if a_max < a_min:
        raise ValueError("a_max must be at least a_min")
    intervals = max(1, math.ceil((a_max - a_min) / spacing))
    step = (a_max - a_min) / intervals
    return tuple(a_min + index * step for index in range(intervals + 1))


def local_minima(evaluations: Iterable[PodestaEvaluation]) -> list[PodestaEvaluation]:
    """Return one-dimensional nearest-neighbor local minima sorted by loss."""
    ordered = sorted(evaluations, key=lambda item: item.a)
    minima = []
    for index, evaluation in enumerate(ordered):
        left = ordered[index - 1].endpoint_loss if index > 0 else math.inf
        right = ordered[index + 1].endpoint_loss if index + 1 < len(ordered) else math.inf
        if evaluation.endpoint_loss <= left and evaluation.endpoint_loss <= right:
            minima.append(evaluation)
    return sorted(minima, key=lambda item: (item.endpoint_loss, item.a))


def refine_minimum(
    low: float,
    high: float,
    settings: PodestaSettings = PodestaSettings(),
    iterations: int = 24,
) -> RefinedMinimum:
    """Minimize the endpoint proxy on one bracket by golden-section search."""
    if high < low:
        low, high = high, low
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    left = high - phi * (high - low)
    right = low + phi * (high - low)
    left_eval = evaluate_a(left, settings)
    right_eval = evaluate_a(right, settings)
    best = left_eval if left_eval.endpoint_loss <= right_eval.endpoint_loss else right_eval
    for _ in range(iterations):
        if left_eval.endpoint_loss <= right_eval.endpoint_loss:
            high = right
            right = left
            right_eval = left_eval
            left = high - phi * (high - low)
            left_eval = evaluate_a(left, settings)
        else:
            low = left
            left = right
            left_eval = right_eval
            right = low + phi * (high - low)
            right_eval = evaluate_a(right, settings)
        best = min((best, left_eval, right_eval), key=lambda item: item.endpoint_loss)
    return RefinedMinimum((low, high), best, iterations)


def recovery_smoke(settings: PodestaSettings = PodestaSettings()) -> dict:
    """Evaluate the two known compact homogeneous parameters."""
    evaluations = [evaluate_a(ROUND_A_DIRECT, settings), evaluate_a(SQUASHED_A_DIRECT, settings)]
    return {
        "lambda": settings.lambda_,
        "known_parameters": {
            "round_direct": ROUND_A_DIRECT,
            "round_canonical_positive": ROUND_A_CANONICAL,
            "squashed_direct": SQUASHED_A_DIRECT,
        },
        "results": [evaluation_payload(evaluation) for evaluation in evaluations],
        "classification_counts": dict(Counter(evaluation.classification for evaluation in evaluations)),
    }


def evaluation_payload(evaluation: PodestaEvaluation) -> dict:
    """Return a JSON-ready evaluation payload."""
    return {
        "a": evaluation.a,
        "status": evaluation.status,
        "classification": evaluation.classification,
        "endpoint_loss": evaluation.endpoint_loss,
        "endpoint_time": evaluation.endpoint_time,
        "endpoint_f": None if evaluation.endpoint_f is None else list(evaluation.endpoint_f),
        "f0_cross_time": evaluation.f0_cross_time,
        "steps": evaluation.steps,
        "message": evaluation.message,
    }


def settings_payload(settings: PodestaSettings) -> dict:
    """Return JSON-ready settings."""
    return {
        "lambda": settings.lambda_,
        "seed_epsilon": settings.seed_epsilon,
        "step_size": settings.step_size,
        "max_time": settings.max_time,
        "closure_threshold": settings.closure_threshold,
        "local_signal_threshold": settings.local_signal_threshold,
        "min_endpoint_time": settings.min_endpoint_time,
        "min_terminal_abs_f1": settings.min_terminal_abs_f1,
    }


def scout_metadata(a_min: float, a_max: float, spacing: float, limit: int | None) -> dict:
    """Return deterministic scout-grid metadata."""
    axis = axis_values(a_min, a_max, spacing)
    seed_count = len(axis) if limit is None else min(limit, len(axis))
    return {
        "a_min": a_min,
        "a_max": a_max,
        "spacing": spacing,
        "axis_count": len(axis),
        "seed_count": seed_count,
        "limit": limit,
        "known_values": {"round_direct": ROUND_A_DIRECT, "squashed_direct": SQUASHED_A_DIRECT},
    }


def scout_evaluations(
    a_min: float = DEFAULT_A_MIN,
    a_max: float = DEFAULT_A_MAX,
    spacing: float = DEFAULT_SPACING,
    limit: int | None = None,
    settings: PodestaSettings = PodestaSettings(),
) -> Iterator[PodestaEvaluation]:
    """Yield scout evaluations over the one-dimensional axis."""
    for index, a in enumerate(axis_values(a_min, a_max, spacing)):
        if limit is not None and index >= limit:
            return
        yield evaluate_a(a, settings)


def run_scout(
    a_min: float = DEFAULT_A_MIN,
    a_max: float = DEFAULT_A_MAX,
    spacing: float = DEFAULT_SPACING,
    limit: int | None = None,
    refine_best: int = 8,
    settings: PodestaSettings = PodestaSettings(),
    jsonl_path: Path | None = None,
) -> dict:
    """Run the one-dimensional scout and optional local refinements."""
    metadata = scout_metadata(a_min, a_max, spacing, limit)
    start_payload = {
        "scout_version": SCOUT_VERSION,
        "random_seed": RANDOM_SEED,
        "grid": metadata,
        "settings": settings_payload(settings),
    }
    if jsonl_path is not None:
        _write_jsonl_event(jsonl_path, _event("run_start", start_payload))
    evaluations = []
    for seed_index, evaluation in enumerate(scout_evaluations(a_min, a_max, spacing, limit, settings)):
        evaluations.append(evaluation)
        if jsonl_path is not None:
            _write_jsonl_event(jsonl_path, _event("scout_result", {"seed_index": seed_index, **evaluation_payload(evaluation)}))
    minima = local_minima(evaluations)
    refined = []
    if refine_best > 0:
        for minimum in minima[:refine_best]:
            low = max(a_min, minimum.a - spacing)
            high = min(a_max, minimum.a + spacing)
            refined_minimum = refine_minimum(low, high, settings)
            refined.append(refined_minimum)
            if jsonl_path is not None:
                _write_jsonl_event(
                    jsonl_path,
                    _event(
                        "refined_minimum",
                        {
                            "bracket": list(refined_minimum.bracket),
                            "iterations": refined_minimum.iterations,
                            "evaluation": evaluation_payload(refined_minimum.evaluation),
                        },
                    ),
                )
    classification_counts = Counter(evaluation.classification for evaluation in evaluations)
    refined_counts = Counter(item.evaluation.classification for item in refined)
    summary = {
        **start_payload,
        "classification_counts": dict(classification_counts),
        "best_scouts": [evaluation_payload(item) for item in sorted(evaluations, key=lambda value: value.endpoint_loss)[:20]],
        "local_minima": [evaluation_payload(item) for item in minima[:20]],
        "refined_minima": [
            {
                "bracket": list(item.bracket),
                "iterations": item.iterations,
                "evaluation": evaluation_payload(item.evaluation),
            }
            for item in sorted(refined, key=lambda value: value.evaluation.endpoint_loss)
        ],
        "refined_classification_counts": dict(refined_counts),
    }
    if jsonl_path is not None:
        _write_jsonl_event(jsonl_path, _event("run_summary", summary))
    return summary


def _settings_from_args(args: argparse.Namespace) -> PodestaSettings:
    """Build numerical settings from CLI arguments."""
    return PodestaSettings(
        lambda_=args.lambda_,
        seed_epsilon=args.seed_epsilon,
        step_size=args.step_size,
        max_time=args.max_time,
        closure_threshold=args.closure_threshold,
        local_signal_threshold=args.local_signal_threshold,
        min_endpoint_time=args.min_endpoint_time,
        min_terminal_abs_f1=args.min_terminal_abs_f1,
    )


def _print_recovery(payload: dict) -> None:
    """Print known-parameter recovery smoke results."""
    print("S7 SU(2)^3 Podesta recovery smoke", flush=True)
    print(f"classifications: {payload['classification_counts']}", flush=True)
    for result in payload["results"]:
        print(
            f"  a={result['a']:.12g}  {result['classification']}  "
            f"loss={result['endpoint_loss']:.6g}  t={result['endpoint_time']}",
            flush=True,
        )


def _print_scout(summary: dict) -> None:
    """Print a compact scout summary."""
    print("S7 SU(2)^3 Podesta scout", flush=True)
    print(f"grid: {summary['grid']}", flush=True)
    print(f"classifications: {summary['classification_counts']}", flush=True)
    print("best scouts:", flush=True)
    for item in summary["best_scouts"][:8]:
        print(
            f"  a={item['a']:.12g}  {item['classification']}  "
            f"loss={item['endpoint_loss']:.6g}  t={item['endpoint_time']}",
            flush=True,
        )
    if summary["refined_minima"]:
        print("refined minima:", flush=True)
        for item in summary["refined_minima"][:8]:
            evaluation = item["evaluation"]
            print(
                f"  a={evaluation['a']:.12g}  {evaluation['classification']}  "
                f"loss={evaluation['endpoint_loss']:.6g}  t={evaluation['endpoint_time']}  "
                f"bracket={item['bracket']}",
                flush=True,
            )


def main(argv: list[str] | None = None) -> None:
    """Run the Podesta SU(2)^3 recovery smoke or one-parameter scout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-known", action="store_true", help="only evaluate the known round/squashed parameters")
    parser.add_argument("--dry-run", action="store_true", help="print grid metadata without evaluating")
    parser.add_argument("--a-min", type=float, default=DEFAULT_A_MIN)
    parser.add_argument("--a-max", type=float, default=DEFAULT_A_MAX)
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refine-best", type=int, default=8)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument("--seed-epsilon", type=float, default=1e-3)
    parser.add_argument("--step-size", type=float, default=1e-3)
    parser.add_argument("--max-time", type=float, default=12.0)
    parser.add_argument("--closure-threshold", type=float, default=1e-3)
    parser.add_argument("--local-signal-threshold", type=float, default=0.1)
    parser.add_argument("--min-endpoint-time", type=float, default=0.1)
    parser.add_argument("--min-terminal-abs-f1", type=float, default=1.0)
    parser.add_argument("--no-write", action="store_true", help="do not write output files")
    args = parser.parse_args(argv)
    settings = _settings_from_args(args)
    if args.recover_known:
        payload = recovery_smoke(settings)
        _print_recovery(payload)
        return
    metadata = scout_metadata(args.a_min, args.a_max, args.spacing, args.limit)
    if args.dry_run:
        print(json.dumps({"scout_version": SCOUT_VERSION, "grid": metadata, "settings": settings_payload(settings)}, indent=2), flush=True)
        return
    jsonl_path: Path | None = None
    summary_path: Path | None = None
    if not args.no_write:
        jsonl_path, summary_path = _output_paths(OUTPUT_DIR, SCOUT_VERSION)
    summary = run_scout(args.a_min, args.a_max, args.spacing, args.limit, args.refine_best, settings, jsonl_path)
    if summary_path is not None:
        summary = {**summary, "jsonl_path": str(jsonl_path), "summary_path": str(summary_path)}
        _write_summary(summary_path, summary)
    _print_scout(summary)
    if summary_path is not None:
        print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
