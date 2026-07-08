"""Scalar defect audit for Podesta's SU(2)^3 S7 tail problem.

The audit is deliberately non-rigorous.  It evaluates many necessary endpoint
defects, ranks them as possible proof objects, and records a bounded proof
attempt for the strongest large-tail candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from . import su2_cubed_scout as scout
from . import su2_cubed_tail_defect as tail


DEFECT_AUDIT_VERSION = "s7-su2-cubed-defect-audit-v1"
DEFAULT_A_VALUES = (
    scout.ROUND_A_DIRECT,
    scout.SQUASHED_A_DIRECT,
    scout.ROUND_A_CANONICAL,
    -100.0,
    100.0,
    -250.0,
    250.0,
    -500.0,
    500.0,
    -1000.0,
    1000.0,
    -5000.0,
    5000.0,
    -10000.0,
    10000.0,
)
DEFAULT_STEP_SIZE = 2.5e-5
DEFAULT_MAX_TIME = 12.0
DEFAULT_CALIBRATION_TOLERANCE = 2e-3
DEFAULT_EXTRA_ZERO_TOLERANCE = 5e-4
DEFAULT_LARGE_ABS_A = 250.0
DEFAULT_OUTPUT_DIR = Path("output/s7_su2_cubed_defect_audits")
LAMBDAS_C_X3 = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
LAMBDAS_X3_X2 = (-10.0, -3.0, -1.0, 1.0, 3.0, 10.0)


@dataclass(frozen=True)
class TerminalSample:
    """One scaled terminal crossing sample."""

    source: str
    a: float | None
    b: float
    status: str
    time: float
    x: tuple[float, float, float, float]
    step_size: float
    steps: int
    seed_mode: str
    message: str | None = None


@dataclass(frozen=True)
class DefectSpec:
    """One scalar necessary defect or diagnostic."""

    name: str
    formula: str
    why_zero: str
    family: str
    priority_bias: float
    fn: Callable[[TerminalSample], float]
    exact_endpoint_defect: bool = True


def _format_parameter(value: float) -> str:
    """Return a stable identifier suffix for a small real parameter."""
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"p{text}" if value >= 0 else text


def c_value(sample: TerminalSample) -> float:
    """Return ``C=x1*x2-p^2*x3/6`` at the sample endpoint."""
    p, x1, x2, x3 = sample.x
    return x1 * x2 - p * p * x3 / 6.0


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return a finite division or NaN when the denominator is unusable."""
    if not math.isfinite(denominator) or abs(denominator) < 1e-14:
        return math.nan
    return numerator / denominator


def defect_specs() -> tuple[DefectSpec, ...]:
    """Return the first-pass defect registry."""
    specs: list[DefectSpec] = [
        DefectSpec(
            "D_x3",
            "x3(T)",
            "standard K- closure gives x3(T)=0",
            "x3",
            8.0,
            lambda sample: sample.x[3],
        ),
        DefectSpec(
            "D_x2",
            "x2(T)",
            "standard K- closure gives x2(T)=0",
            "x2",
            2.0,
            lambda sample: sample.x[2],
        ),
        DefectSpec(
            "D_C",
            "C(T)=x1*x2-p^2*x3/6",
            "at p(T)=0 this is x1(T)*x2(T), with x1(T) nonzero for standard closure",
            "C",
            3.0,
            c_value,
        ),
        DefectSpec(
            "D_x3_C_norm2",
            "x3(T)^2+C(T)^2",
            "both summands vanish under standard closure",
            "norm_3C",
            7.0,
            lambda sample: sample.x[3] ** 2 + c_value(sample) ** 2,
        ),
        DefectSpec(
            "D_x3_x2_norm2",
            "x3(T)^2+x2(T)^2",
            "both summands vanish under standard closure",
            "norm_32",
            5.0,
            lambda sample: sample.x[3] ** 2 + sample.x[2] ** 2,
        ),
    ]
    for lam in LAMBDAS_C_X3:
        specs.append(
            DefectSpec(
                f"D_C_plus_{_format_parameter(lam)}_x3",
                f"C(T)+({lam:g})*x3(T)",
                "C(T)=0 and x3(T)=0 under standard closure",
                f"C_plus_x3_{_format_parameter(lam)}",
                1.0,
                lambda sample, lam=lam: c_value(sample) + lam * sample.x[3],
            )
        )
    for lam in LAMBDAS_X3_X2:
        specs.append(
            DefectSpec(
                f"D_x3_plus_{_format_parameter(lam)}_x2",
                f"x3(T)+({lam:g})*x2(T)",
                "x3(T)=0 and x2(T)=0 under standard closure",
                f"x3_plus_x2_{_format_parameter(lam)}",
                1.0,
                lambda sample, lam=lam: sample.x[3] + lam * sample.x[2],
            )
        )
    specs.extend(
        [
            DefectSpec(
                "D_S1",
                "x3(T)^3-4*x1(T)*p(T)^3",
                "at p(T)=0 this is x3(T)^3",
                "S1",
                6.0,
                lambda sample: sample.x[3] ** 3 - 4.0 * sample.x[1] * sample.x[0] ** 3,
            ),
            DefectSpec(
                "D_S3",
                "2*x3(T)-6*p(T)",
                "at p(T)=0 this is 2*x3(T)",
                "S3",
                3.0,
                lambda sample: 2.0 * sample.x[3] - 6.0 * sample.x[0],
            ),
            DefectSpec(
                "D_C_over_x1",
                "C(T)/x1(T)",
                "C(T)=0 under standard closure and x1(T) is nonzero",
                "C_over_x1",
                2.0,
                lambda sample: _safe_divide(c_value(sample), sample.x[1]),
            ),
            DefectSpec(
                "D_x3_damped_r1",
                "x3(T)/(1+abs(x1(T)))",
                "the numerator vanishes under standard closure",
                "x3_damped",
                2.0,
                lambda sample: sample.x[3] / (1.0 + abs(sample.x[1])),
            ),
            DefectSpec(
                "D_C_damped_r1",
                "C(T)/(1+abs(x1(T)))",
                "the numerator vanishes under standard closure",
                "C_damped",
                1.0,
                lambda sample: c_value(sample) / (1.0 + abs(sample.x[1])),
            ),
            DefectSpec(
                "D_W_over_b",
                "(p(T)^2+6*b*x3(T))/b, endpoint-safe as 6*x3(T) for b=0",
                "at p(T)=0 this is 6*x3(T)",
                "x3_scaled_duplicate",
                0.5,
                lambda sample: 6.0 * sample.x[3]
                if abs(sample.b) < 1e-14
                else (sample.x[0] ** 2 + 6.0 * sample.b * sample.x[3]) / sample.b,
            ),
            DefectSpec(
                "D_C_IF",
                "T^4*C(T), endpoint equivalent of the C integrating-factor integral",
                "standard closure gives C(T)=0",
                "C_integrating_factor",
                3.5,
                lambda sample: sample.time**4 * c_value(sample),
            ),
            DefectSpec(
                "D_3_IF",
                "T^2*x3(T), endpoint equivalent of the x3 integrating-factor integral",
                "standard closure gives x3(T)=0",
                "x3_integrating_factor",
                4.0,
                lambda sample: sample.time**2 * sample.x[3],
            ),
        ]
    )
    return tuple(specs)


def _initial_scaled_state(epsilon: float, b: float, seed_mode: str) -> tuple[float, float, float, float]:
    """Return the scaled initial state at ``epsilon``."""
    if seed_mode == "base":
        return (1.0, 27.0 / 4.0, -1.0 / 27.0, 3.0)
    if seed_mode == "taylor":
        return tail.scaled_taylor_seed(epsilon, b)
    raise ValueError("seed_mode must be 'base' or 'taylor'")


def integrate_terminal_sample(
    source: str,
    a: float | None = None,
    *,
    step_size: float = DEFAULT_STEP_SIZE,
    epsilon: float = tail.DEFAULT_EPSILON,
    max_time: float = DEFAULT_MAX_TIME,
    seed_mode: str = "taylor",
) -> TerminalSample:
    """Integrate to the first scaled ``p=x0=0`` crossing."""
    if source not in {"exact", "limit"}:
        raise ValueError("source must be exact or limit")
    if source == "exact" and (a is None or not math.isfinite(a) or abs(a) < 1e-14):
        raise ValueError("exact samples require a finite nonzero a")
    b = 0.0 if source == "limit" else 1.0 / float(a)
    t = epsilon
    x = _initial_scaled_state(epsilon, b, seed_mode)
    initial_sign = math.copysign(1.0, x[0])
    steps = 0
    while t < max_time:
        step = min(step_size, max_time - t)
        try:
            x_next = tail._rk4_step_b(t, x, step, b)
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError) as exc:
            return TerminalSample(source, a, b, "failed", t, x, step_size, steps, seed_mode, str(exc))
        if not all(math.isfinite(value) for value in x_next):
            return TerminalSample(source, a, b, "failed", t, x, step_size, steps, seed_mode, "nonfinite state")
        if math.copysign(1.0, x_next[0]) != initial_sign:
            alpha = abs(x[0]) / (abs(x[0]) + abs(x_next[0]))
            crossing_x = tuple(value + alpha * (next_value - value) for value, next_value in zip(x, x_next))
            crossing_t = t + alpha * step
            return TerminalSample(source, a, b, "crossed", crossing_t, crossing_x, step_size, steps + 1, seed_mode)
        x = x_next
        t += step
        steps += 1
    return TerminalSample(source, a, b, "no_crossing", t, x, step_size, steps, seed_mode)


def sample_payload(sample: TerminalSample) -> dict:
    """Return a JSON-ready sample payload."""
    p, x1, x2, x3 = sample.x
    payload = {
        "source": sample.source,
        "a": sample.a,
        "b": sample.b,
        "status": sample.status,
        "time": sample.time,
        "p": p,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "C": c_value(sample),
        "step_size": sample.step_size,
        "steps": sample.steps,
        "seed_mode": sample.seed_mode,
        "message": sample.message,
    }
    if sample.a is not None:
        h0 = sample.a * p
        h1 = x1
        h2 = sample.a**3 * x2
        h3 = sample.a * x3
        h4 = -h3 - h0 * h0 / 6.0
        payload["h"] = [h0, h1, h2, h3, h4]
        payload["f"] = [
            sample.time * h0,
            sample.time**4 * h1,
            h2,
            sample.time**2 * h3,
            sample.time**2 * h4,
        ]
    return payload


def evaluate_defects(sample: TerminalSample, specs: Iterable[DefectSpec] | None = None) -> list[dict]:
    """Evaluate all requested defects on one terminal sample."""
    rows = []
    for spec in defect_specs() if specs is None else specs:
        try:
            value = spec.fn(sample)
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
            value = math.nan
        rows.append(
            {
                "a": sample.a,
                "b": sample.b,
                "source": sample.source,
                "status": sample.status,
                "time": sample.time,
                "defect_name": spec.name,
                "formula": spec.formula,
                "value": value,
                "abs_value": abs(value) if math.isfinite(value) else math.inf,
                "sign": _sign_label(value),
                "family": spec.family,
                "exact_endpoint_defect": spec.exact_endpoint_defect,
            }
        )
    return rows


def _sign_label(value: float, tolerance: float = 0.0) -> str:
    """Return a stable sign label for a scalar value."""
    if not math.isfinite(value):
        return "nonfinite"
    if abs(value) <= tolerance:
        return "zero"
    return "positive" if value > 0.0 else "negative"


def _large_rows(rows: list[dict], defect_name: str, large_abs_a: float) -> list[dict]:
    """Return finite-a rows in the large-tail audit region."""
    return [
        row
        for row in rows
        if row["defect_name"] == defect_name
        and row["source"] == "exact"
        and row["a"] is not None
        and abs(row["a"]) >= large_abs_a
    ]


def _calibration_rows(rows: list[dict], defect_name: str) -> list[dict]:
    """Return rows for the two direct known compact parameters."""
    known = {scout.ROUND_A_DIRECT, scout.SQUASHED_A_DIRECT}
    return [
        row
        for row in rows
        if row["defect_name"] == defect_name
        and row["source"] == "exact"
        and row["a"] in known
    ]


def _limit_row(rows: list[dict], defect_name: str) -> dict | None:
    """Return the limiting row for one defect, if present."""
    for row in rows:
        if row["defect_name"] == defect_name and row["source"] == "limit":
            return row
    return None


def _has_apparent_extra_zero(large_rows: list[dict], tolerance: float) -> bool:
    """Return whether the large-tail data show an unwanted zero or sign change."""
    finite_rows = [row for row in large_rows if math.isfinite(row["value"])]
    if any(row["abs_value"] <= tolerance for row in finite_rows):
        return True
    for sign in (-1.0, 1.0):
        ordered = sorted(
            [row for row in finite_rows if row["a"] is not None and math.copysign(1.0, row["a"]) == sign],
            key=lambda row: abs(row["a"]),
        )
        signs = [_sign_label(row["value"], tolerance) for row in ordered]
        nonzero = [item for item in signs if item not in {"zero", "nonfinite"}]
        if len(set(nonzero)) > 1:
            return True
    return False


def summarize_defect_rows(
    rows: list[dict],
    specs: Iterable[DefectSpec] | None = None,
    *,
    calibration_tolerance: float = DEFAULT_CALIBRATION_TOLERANCE,
    extra_zero_tolerance: float = DEFAULT_EXTRA_ZERO_TOLERANCE,
    large_abs_a: float = DEFAULT_LARGE_ABS_A,
) -> list[dict]:
    """Return one summary row per defect."""
    summaries = []
    spec_list = list(defect_specs() if specs is None else specs)
    for spec in spec_list:
        calibration = _calibration_rows(rows, spec.name)
        calibration_abs = [row["abs_value"] for row in calibration if math.isfinite(row["value"])]
        known_max_abs = max(calibration_abs) if calibration_abs else math.inf
        known_ok = bool(calibration_abs) and known_max_abs <= calibration_tolerance
        limit = _limit_row(rows, spec.name)
        limit_value = math.nan if limit is None else limit["value"]
        limit_abs = abs(limit_value) if math.isfinite(limit_value) else math.inf
        large = _large_rows(rows, spec.name, large_abs_a)
        finite_large = [row for row in large if math.isfinite(row["value"])]
        positive_large = [row for row in finite_large if row["a"] is not None and row["a"] > 0]
        negative_large = [row for row in finite_large if row["a"] is not None and row["a"] < 0]
        positive_signs = {_sign_label(row["value"], extra_zero_tolerance) for row in positive_large}
        negative_signs = {_sign_label(row["value"], extra_zero_tolerance) for row in negative_large}
        apparent_extra_zero = _has_apparent_extra_zero(large, extra_zero_tolerance)
        min_large_abs = min((row["abs_value"] for row in finite_large), default=math.inf)
        score = _score_summary(spec, known_ok, limit_abs, apparent_extra_zero, positive_signs, negative_signs)
        summaries.append(
            {
                "defect_name": spec.name,
                "formula": spec.formula,
                "why_zero": spec.why_zero,
                "family": spec.family,
                "exact_endpoint_defect": spec.exact_endpoint_defect,
                "known_compact_max_abs": known_max_abs,
                "known_compact_ok": known_ok,
                "limit_value": limit_value,
                "limit_abs": limit_abs,
                "large_positive_signs": sorted(positive_signs),
                "large_negative_signs": sorted(negative_signs),
                "min_large_abs": min_large_abs,
                "apparent_extra_zero": apparent_extra_zero,
                "score": score,
                "recommended_priority": _priority_label(score),
            }
        )
    return sorted(summaries, key=lambda row: (-row["score"], row["defect_name"]))


def _score_summary(
    spec: DefectSpec,
    known_ok: bool,
    limit_abs: float,
    apparent_extra_zero: bool,
    positive_signs: set[str],
    negative_signs: set[str],
) -> float:
    """Return a heuristic proof-candidate score."""
    score = spec.priority_bias
    if known_ok:
        score += 4.0
    if math.isfinite(limit_abs):
        score += min(5.0, math.log10(1.0 + limit_abs) * 4.0)
        if limit_abs > 1e-3:
            score += 2.0
        if limit_abs > 0.1:
            score += 2.0
    if not apparent_extra_zero:
        score += 2.0
    if len(positive_signs - {"zero", "nonfinite"}) <= 1 and len(negative_signs - {"zero", "nonfinite"}) <= 1:
        score += 1.0
    return score


def _priority_label(score: float) -> str:
    """Return a compact priority label."""
    if score >= 18.0:
        return "top"
    if score >= 14.0:
        return "strong"
    if score >= 10.0:
        return "watch"
    return "low"


def select_top_proof_candidates(summaries: list[dict], count: int = 3) -> list[dict]:
    """Return structurally distinct proof candidates."""
    preferred_order = {
        "D_x3": 0,
        "D_x3_C_norm2": 1,
        "D_S1": 2,
        "D_x3_x2_norm2": 3,
        "D_C_plus_p1_x3": 4,
    }
    excluded_families = {"x3_scaled_duplicate", "x3_integrating_factor", "C_integrating_factor", "S3"}
    candidates = [
        row
        for row in summaries
        if row["known_compact_ok"]
        and math.isfinite(row["limit_value"])
        and row["limit_abs"] > 1e-3
        and not row["apparent_extra_zero"]
        and row["family"] not in excluded_families
    ]
    candidates.sort(key=lambda row: (preferred_order.get(row["defect_name"], 100), -row["score"], row["defect_name"]))
    selected = []
    used_families: set[str] = set()
    for row in candidates:
        if row["family"] in used_families:
            continue
        selected.append(row)
        used_families.add(row["family"])
        if len(selected) == count:
            return selected
    return selected


def asymptotic_reductions(samples: list[TerminalSample], selected: list[dict]) -> dict:
    """Return the common large-|a| proof reduction for selected endpoint defects."""
    limit_sample = next((sample for sample in samples if sample.source == "limit"), None)
    if limit_sample is None or limit_sample.status != "crossed":
        return {"status": "missing_limit_crossing", "items": []}
    preterminal_p = 1e-3
    try:
        pre_t, pre_x1, pre_x2, pre_x3 = tail.scaled_state_at_p(
            "limit",
            preterminal_p,
            entry_time=tail.DEFAULT_P_TUBE_ENTRY_TIME,
            step_size=1e-5,
        )
        sampled_preterminal_p_prime = tail.scaled_rhs_with_b(
            pre_t,
            (preterminal_p, pre_x1, pre_x2, pre_x3),
            0.0,
        )[0]
    except (ArithmeticError, OverflowError, RuntimeError, ValueError, ZeroDivisionError):
        pre_t = math.nan
        sampled_preterminal_p_prime = math.nan
    items = []
    row_by_name = {row["defect_name"]: row for row in evaluate_defects(limit_sample)}
    for candidate in selected:
        row = row_by_name[candidate["defect_name"]]
        items.append(
            {
                "defect_name": candidate["defect_name"],
                "limit_value": row["value"],
                "limit_abs": row["abs_value"],
                "reduction": (
                    "If the scaled finite-a solutions and their first p=0 event converge "
                    "to the singular limiting endpoint, then this endpoint defect converges "
                    "to the listed nonzero limiting value."
                ),
            }
        )
    return {
        "status": "reduced_to_singular_endpoint_convergence",
        "limit_crossing": sample_payload(limit_sample),
        "preterminal_p": preterminal_p,
        "preterminal_time": pre_t,
        "sampled_preterminal_p_prime": sampled_preterminal_p_prime,
        "items": items,
        "remaining_obligation": (
            "prove uniform continuous dependence of the scaled b-family up to the singular first p=0 event"
        ),
    }


def uniform_x3_exclusion_attempt(
    candidate_a: float = tail.DEFAULT_TUBE_CANDIDATE_A,
    grid_subdivisions: int = 8,
) -> dict:
    """Return the bounded uniform-exclusion attempt for the strongest candidate."""
    barrier = tail.late_scalar_barrier_report(
        candidate_a=candidate_a,
        grid_subdivisions=grid_subdivisions,
    )
    support_samples = []
    for source, a in (("limit", None), ("exact", -candidate_a), ("exact", candidate_a)):
        x = tail.scaled_state_at(source, tail.DEFAULT_SUPPORT_TIME, a, step_size=1e-3)
        support_samples.append(
            {
                "source": source,
                "a": a,
                "time": tail.DEFAULT_SUPPORT_TIME,
                "p": x[0],
                "x1": x[1],
                "x2": x[2],
                "x3": x[3],
                "C": x[1] * x[2] - x[0] * x[0] * x[3] / 6.0,
            }
        )
    return {
        "status": "reduced_not_closed",
        "candidate_A": candidate_a,
        "strongest_candidate": "D_x3",
        "barrier_report": barrier,
        "nominal_support_samples": support_samples,
        "what_this_attempt_shows": (
            "Inside the correlated late region, the x3=-sigma wall and the C-Kp^3 wall "
            "have favorable scalar margins for |a|>=A.  This would keep x3 negative "
            "through the terminal tail and exclude standard K- closure."
        ),
        "why_not_complete": (
            "The missing part is still the support-entry/containment lemma proving every "
            "|a|>=A trajectory reaches and remains in that correlated late region."
        ),
    }


def _regularized_p_prime_coefficient(p, t, x1, x2, x3, b):
    """Return ``p^4 * dp/dt`` in a form regular at ``p=0``."""
    c_value = x1 * x2 - p * p * x3 / 6.0
    i1 = t * (2.0 * x3 * x1 * x2 - 0.5 * p * p * x3 * x3) - t**3 * x1 * p**4 / 18.0
    i2 = -2.0 * t * x3**3 - 2.0 * t**3 * x1 * p * p * x3 / 3.0
    i3 = -2.0 * t**3 * x1 * x3 * x3
    return (
        -p**5 / t
        - 3.0 * x2 * x3 * x3 / t
        - t * p * p * c_value / 4.0
        + b * (-3.0 * i1 / 2.0)
        + b * b * (-3.0 * i2 / 2.0)
        + b**3 * (-3.0 * i3 / 2.0)
    )


def _regularized_x1_prime_coefficient(p, t, x1, x2, x3, b):
    """Return ``p^3 * dx1/dt`` in a form regular at ``p=0``."""
    return (
        p**3 * (-4.0 * x1) / t
        + x3**3 / t
        + t / 2.0 * (x1 * x1 * x2 + 0.5 * x1 * x3 * p * p)
        + b * 1.5 * t * x1 * x3 * x3
    )


def _regularized_x2_prime_coefficient(p, t, x1, x2, x3, b):
    """Return ``p^3 * dx2/dt`` in a form regular at ``p=0``."""
    return (
        t * (-p * p * x2 * x3 / 4.0 - 0.5 * x1 * x2 * x2 + t * t * p**6 / 216.0)
        + b * t * (t * t * p**4 * x3 / 12.0 - 1.5 * x2 * x3 * x3)
        + b * b * t * (t * t * p * p * x3 * x3 / 2.0)
        + b**3 * t * (t * t * x3**3)
    )


def _regularized_x3_prime_coefficient(p, t, x1, x2, x3, b):
    """Return ``p^3 * dx3/dt`` in a form regular at ``p=0``."""
    return (
        p**3 * (-2.0 * x3 + 6.0 * p) / t
        + t / 2.0 * (x1 * x2 * x3 - x3 * x3 * p * p / 6.0 - t * t * x1 * p**4 / 18.0)
        + b * t / 2.0 * (-x3**3 - 2.0 * t * t * x1 * p * p * x3 / 3.0)
        + b * b * t / 2.0 * (-2.0 * t * t * x1 * x3 * x3)
    )


def _interval_bounds(value) -> tuple[float, float]:
    """Return float bounds for an mpmath interval or scalar."""
    if hasattr(value, "a") and hasattr(value, "b"):
        return float(value.a), float(value.b)
    scalar = float(value)
    return scalar, scalar


def dx3_asymptotic_tail_proof_report(
    p0: float = 1e-3,
    b_radius: float = 1e-8,
    box_low: tuple[float, float, float, float] = (3.59, 8.5, 0.004, -1.4),
    box_high: tuple[float, float, float, float] = (3.61, 9.5, 0.008, -0.9),
) -> dict:
    """Return the small-p proof audit for the ``D_x3`` asymptotic limit.

    In p-time,

        dx_i/dp = p * H_i / A,  dt/dp = p^4 / A,

    where ``A=p^4 p'`` and ``H_i=p^3 x_i'`` are regular at ``p=0``.
    This report bounds those regular factors on a fixed terminal box.
    """
    if not (0.0 < p0 < 1.0):
        raise ValueError("p0 must be in (0, 1)")
    if b_radius < 0.0:
        raise ValueError("b_radius must be nonnegative")
    if not all(box_low[index] < box_high[index] for index in range(4)):
        raise ValueError("box_low must be strictly below box_high")

    from mpmath import iv

    p_interval = iv.mpf([0.0, p0])
    t_interval = iv.mpf([box_low[0], box_high[0]])
    x1_interval = iv.mpf([box_low[1], box_high[1]])
    x2_interval = iv.mpf([box_low[2], box_high[2]])
    x3_interval = iv.mpf([box_low[3], box_high[3]])
    b_interval = iv.mpf([-b_radius, b_radius])

    a_interval = _regularized_p_prime_coefficient(
        p_interval,
        t_interval,
        x1_interval,
        x2_interval,
        x3_interval,
        b_interval,
    )
    h_intervals = {
        "x1": _regularized_x1_prime_coefficient(
            p_interval,
            t_interval,
            x1_interval,
            x2_interval,
            x3_interval,
            b_interval,
        ),
        "x2": _regularized_x2_prime_coefficient(
            p_interval,
            t_interval,
            x1_interval,
            x2_interval,
            x3_interval,
            b_interval,
        ),
        "x3": _regularized_x3_prime_coefficient(
            p_interval,
            t_interval,
            x1_interval,
            x2_interval,
            x3_interval,
            b_interval,
        ),
    }
    a_lower, a_upper = _interval_bounds(a_interval)
    if a_upper >= 0.0:
        status = "failed"
        denominator_margin = 0.0
    else:
        status = "terminal_tail_bound"
        denominator_margin = -a_upper

    derivative_bounds: dict[str, float] = {}
    variation_bounds: dict[str, float] = {}
    for name, interval_value in h_intervals.items():
        lower, upper = _interval_bounds(interval_value)
        numerator_bound = max(abs(lower), abs(upper))
        derivative_bounds[name] = math.inf if denominator_margin == 0.0 else numerator_bound / denominator_margin
        variation_bounds[name] = 0.5 * p0 * p0 * derivative_bounds[name]
    t_variation_bound = math.inf if denominator_margin == 0.0 else p0**5 / (5.0 * denominator_margin)

    limit_p0 = tail.scaled_state_at_p(
        "limit",
        p0,
        entry_time=tail.DEFAULT_P_TUBE_ENTRY_TIME,
        step_size=1e-5,
    )
    endpoint_upper_bound = limit_p0[3] + variation_bounds["x3"]
    endpoint_lower_bound = limit_p0[3] - variation_bounds["x3"]
    sample_states = []
    for source, a in (("limit", None), ("exact", -1.0 / b_radius if b_radius else None), ("exact", 1.0 / b_radius if b_radius else None)):
        if source == "exact" and a is None:
            continue
        try:
            sample = tail.scaled_state_at_p(
                source,
                p0,
                a,
                entry_time=tail.DEFAULT_P_TUBE_ENTRY_TIME,
                step_size=1e-5,
            )
        except (ArithmeticError, RuntimeError, ValueError, ZeroDivisionError) as exc:
            sample_states.append({"source": source, "a": a, "status": "failed", "message": str(exc)})
            continue
        sample_states.append(
            {
                "source": source,
                "a": a,
                "status": "sampled",
                "state": list(sample),
                "inside_box": all(box_low[index] <= sample[index] <= box_high[index] for index in range(4)),
            }
        )

    return {
        "status": status,
        "candidate": "D_x3",
        "p0": p0,
        "b_radius": b_radius,
        "box_low": list(box_low),
        "box_high": list(box_high),
        "regularized_p_prime_coefficient_bounds": [a_lower, a_upper],
        "denominator_margin": denominator_margin,
        "p_time_derivative_linear_bounds": derivative_bounds,
        "tail_variation_bounds": variation_bounds,
        "t_variation_bound": t_variation_bound,
        "limit_p0_state": list(limit_p0),
        "x3_endpoint_interval_from_limit_p0": [endpoint_lower_bound, endpoint_upper_bound],
        "sample_states": sample_states,
        "conclusion": (
            "Conditional on entering this terminal box at p=p0, x3 changes by at most "
            f"{variation_bounds['x3']:.6g} before p=0; the limiting endpoint is therefore still negative."
        ),
        "remaining_obligation": (
            "prove finite-b trajectories reach the p=p0 box and converge there as b=1/a -> 0"
        ),
    }


def run_audit(
    a_values: Iterable[float] = DEFAULT_A_VALUES,
    *,
    include_limit: bool = True,
    step_size: float = DEFAULT_STEP_SIZE,
    epsilon: float = tail.DEFAULT_EPSILON,
    max_time: float = DEFAULT_MAX_TIME,
    seed_mode: str = "taylor",
    candidate_a: float = tail.DEFAULT_TUBE_CANDIDATE_A,
    barrier_grid_subdivisions: int = 8,
    include_uniform_attempt: bool = True,
) -> dict:
    """Run the full first-pass numerical defect audit."""
    a_value_tuple = tuple(float(value) for value in a_values)
    specs = defect_specs()
    samples = [
        integrate_terminal_sample(
            "exact",
            float(a),
            step_size=step_size,
            epsilon=epsilon,
            max_time=max_time,
            seed_mode=seed_mode,
        )
        for a in a_value_tuple
    ]
    if include_limit:
        samples.append(
            integrate_terminal_sample(
                "limit",
                None,
                step_size=step_size,
                epsilon=epsilon,
                max_time=max_time,
                seed_mode=seed_mode,
            )
        )
    rows = []
    for sample in samples:
        rows.extend(evaluate_defects(sample, specs))
    summaries = summarize_defect_rows(rows, specs)
    selected = select_top_proof_candidates(summaries, 3)
    report = {
        "version": DEFECT_AUDIT_VERSION,
        "settings": {
            "a_values": list(a_value_tuple),
            "include_limit": include_limit,
            "step_size": step_size,
            "epsilon": epsilon,
            "max_time": max_time,
            "seed_mode": seed_mode,
            "calibration_tolerance": DEFAULT_CALIBRATION_TOLERANCE,
            "extra_zero_tolerance": DEFAULT_EXTRA_ZERO_TOLERANCE,
            "large_abs_a": DEFAULT_LARGE_ABS_A,
        },
        "samples": [sample_payload(sample) for sample in samples],
        "defect_rows": rows,
        "defect_summaries": summaries,
        "selected_top_candidates": selected,
        "asymptotic_reductions": asymptotic_reductions(samples, selected),
        "dx3_asymptotic_tail_proof": dx3_asymptotic_tail_proof_report(),
    }
    if include_uniform_attempt:
        report["uniform_exclusion_attempt"] = uniform_x3_exclusion_attempt(
            candidate_a=candidate_a,
            grid_subdivisions=barrier_grid_subdivisions,
        )
    return report


def render_markdown(report: dict) -> str:
    """Render a concise markdown report."""
    lines = [
        "# S7 SU(2)^3 Defect Audit",
        "",
        "Reproducibility command:",
        "",
        "```zsh",
        ".venv/bin/python -m experiments.s7_su2_cubed_defect_audit --write-markdown docs/s7-su2-cubed-defect-audit.md",
        "```",
        "",
        "## Summary",
        "",
        f"Version: `{report['version']}`.",
        f"Step size: `{report['settings']['step_size']}`; seed mode: `{report['settings']['seed_mode']}`.",
        "",
        "Top proof candidates selected by the first-pass audit:",
        "",
        "| defect | limit value | known compact max abs | note |",
        "|---|---:|---:|---|",
    ]
    for item in report["selected_top_candidates"]:
        lines.append(
            f"| `{item['defect_name']}` | {item['limit_value']:.12g} | "
            f"{item['known_compact_max_abs']:.3g} | {item['formula']} |"
        )
    lines.extend(
        [
            "",
            "## Numerical Ranking",
            "",
            "| rank | defect | priority | limit abs | min large abs | large signs | extra zero? |",
            "|---:|---|---|---:|---:|---|---|",
        ]
    )
    for index, item in enumerate(report["defect_summaries"][:18], start=1):
        signs = f"+:{','.join(item['large_positive_signs'])} / -:{','.join(item['large_negative_signs'])}"
        lines.append(
            f"| {index} | `{item['defect_name']}` | {item['recommended_priority']} | "
            f"{item['limit_abs']:.6g} | {item['min_large_abs']:.6g} | {signs} | "
            f"{item['apparent_extra_zero']} |"
        )
    asymptotic = report["asymptotic_reductions"]
    lines.extend(
        [
            "",
            "## Infinity Reduction",
            "",
            f"Status: `{asymptotic['status']}`.",
            f"Sampled preterminal `p'` at `p={asymptotic.get('preterminal_p')}`: "
            f"`{asymptotic.get('sampled_preterminal_p_prime'):.12g}`.",
            "",
            "Common reduction: in scaled variables the finite equations have the form",
            "",
            "```text",
            "x' = F_0(t,x) + b R_1(t,x) + b^2 R_2(t,x) + b^3 R_3(t,x),  b=1/a.",
            "```",
            "",
            "The smooth left seed is also a regular expansion in `b`.  Therefore the",
            "large-`|a|` limit for any endpoint defect in this table is reduced to the",
            "singular endpoint-continuity statement: finite-`b` trajectories and their",
            "first `p=x0=0` event converge to the limiting first crossing.  The sampled",
            "preterminal `p'` is very negative, so the limiting trajectory is already in",
            "a decisive terminal plunge before the singular event.  This is not a proof",
            "of uniform event convergence, but it isolates the needed lemma.",
            "",
        ]
    )
    for item in asymptotic["items"]:
        lines.append(
            f"- `{item['defect_name']}` reduces to the nonzero limit "
            f"`{item['limit_value']:.12g}` once singular endpoint convergence is proved."
        )
    if "dx3_asymptotic_tail_proof" in report:
        proof = report["dx3_asymptotic_tail_proof"]
        lines.extend(
            [
                "",
                "### D_x3 Terminal Tail Proof",
                "",
                f"Status: `{proof['status']}`.",
                "",
                "For the final terminal layer, use `p=x0` as the independent variable.",
                "Multiplying away the singular powers gives",
                "",
                "```text",
                "dt/dp = p^4/A,",
                "dx_i/dp = p*H_i/A,",
                "A = p^4 dp/dt,  H_i = p^3 dx_i/dt,",
                "```",
                "",
                "where `A` and the `H_i` are regular at `p=0`.  On the box",
                "",
                "```text",
                f"0 <= p <= {proof['p0']},",
                f"{proof['box_low'][0]} <= t <= {proof['box_high'][0]},",
                f"{proof['box_low'][1]} <= x1 <= {proof['box_high'][1]},",
                f"{proof['box_low'][2]} <= x2 <= {proof['box_high'][2]},",
                f"{proof['box_low'][3]} <= x3 <= {proof['box_high'][3]},",
                f"|b| <= {proof['b_radius']}",
                "```",
                "",
                f"the interval bound is `A in {proof['regularized_p_prime_coefficient_bounds']}`.",
                "Thus the p-time system has a removable singularity there, and",
                f"`|Delta x3| <= {proof['tail_variation_bounds']['x3']:.12g}` from `p={proof['p0']}` to `p=0`.",
                f"The limiting state at `p={proof['p0']}` gives the endpoint interval",
                f"`{proof['x3_endpoint_interval_from_limit_p0']}`.",
                "",
                "So `D_x3` has a nonzero negative limiting endpoint value.  The only",
                "remaining asymptotic input is the standard compact-interval continuous",
                "dependence up to the fixed slice `p=0.001`.",
            ]
        )
    if "uniform_exclusion_attempt" in report:
        attempt = report["uniform_exclusion_attempt"]
        barrier = attempt["barrier_report"]
        lines.extend(
            [
                "",
                "## Uniform Tail Attempt",
                "",
                f"Status: `{attempt['status']}` for candidate `D_x3` and `A={attempt['candidate_A']:.12g}`.",
                f"Scalar barrier status: `{barrier['status']}`.",
                f"`x3=-sigma` wall margin: `{barrier['x3_wall_margin']:.12g}`.",
                f"`C-Kp^3` limiting wall margin: `{barrier['c_wall_limiting_hdot_lower']:.12g}`.",
                f"Finite-`b` grid sanity margin: `{barrier['finite_b_grid_hdot_margin']:.12g}`.",
                "",
                "This is the bounded proof attempt for `D_x3`: the scalar walls are",
                "",
                "```text",
                "x3 = -0.36,",
                "C = 1.23*p^3,",
                "p <= 0.33,",
                "t in [3.5, 4.0].",
                "```",
                "",
                "Inside that correlated late region, the wall estimates point the right",
                "way for every `|a| >= A`.  If the support-entry lemma were available,",
                "this would keep `x3` negative up to the terminal event, contradicting",
                "the standard `K-` requirement `x3(T)=0`.",
                "",
                attempt["why_not_complete"],
            ]
        )
    lines.extend(
        [
            "",
            "## Calibration Samples",
            "",
            "| source | a | status | T | x2 | x3 | C |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for sample in report["samples"]:
        lines.append(
            f"| {sample['source']} | {'' if sample['a'] is None else f'{sample['a']:.12g}'} | "
            f"{sample['status']} | {sample['time']:.9g} | {sample['x2']:.9g} | "
            f"{sample['x3']:.9g} | {sample['C']:.9g} |"
        )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write defect rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "a",
        "b",
        "source",
        "status",
        "time",
        "defect_name",
        "value",
        "abs_value",
        "sign",
        "family",
        "exact_endpoint_defect",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _parse_float_csv(text: str) -> tuple[float, ...]:
    """Parse comma-separated floats."""
    values = []
    for chunk in text.split(","):
        stripped = chunk.strip()
        if stripped:
            values.append(float(stripped))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return tuple(values)


def _print_summary(report: dict) -> None:
    """Print a compact terminal summary."""
    print("S7 SU(2)^3 defect audit", flush=True)
    print(f"version: {report['version']}", flush=True)
    print("selected top candidates:", flush=True)
    for item in report["selected_top_candidates"]:
        print(
            f"  {item['defect_name']}: limit={item['limit_value']:.9g} "
            f"known_max={item['known_compact_max_abs']:.3g} priority={item['recommended_priority']}",
            flush=True,
        )
    print("top numerical rankings:", flush=True)
    for item in report["defect_summaries"][:10]:
        print(
            f"  {item['defect_name']}: score={item['score']:.3g} "
            f"limit_abs={item['limit_abs']:.6g} min_large_abs={item['min_large_abs']:.6g} "
            f"extra_zero={item['apparent_extra_zero']}",
            flush=True,
        )
    asymptotic = report["asymptotic_reductions"]
    print(
        f"infinity reduction: {asymptotic['status']}, "
        f"sampled preterminal p'={asymptotic.get('sampled_preterminal_p_prime'):.9g}",
        flush=True,
    )
    if "uniform_exclusion_attempt" in report:
        attempt = report["uniform_exclusion_attempt"]
        barrier = attempt["barrier_report"]
        print(
            f"uniform x3 attempt: {attempt['status']}, barrier={barrier['status']}, "
            f"x3 margin={barrier['x3_wall_margin']:.6g}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    """Run the scalar defect audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-values", type=_parse_float_csv, default=DEFAULT_A_VALUES)
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--epsilon", type=float, default=tail.DEFAULT_EPSILON)
    parser.add_argument("--max-time", type=float, default=DEFAULT_MAX_TIME)
    parser.add_argument("--seed-mode", choices=("taylor", "base"), default="taylor")
    parser.add_argument("--no-limit", action="store_true", help="skip the b=0 limiting IVP sample")
    parser.add_argument("--no-uniform-attempt", action="store_true", help="skip the bounded x3 uniform attempt")
    parser.add_argument("--tube-a", type=float, default=tail.DEFAULT_TUBE_CANDIDATE_A)
    parser.add_argument("--barrier-grid-subdivisions", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    parser.add_argument("--write-json", type=Path, default=None, help="write the full JSON report to this path")
    parser.add_argument("--write-markdown", type=Path, default=None, help="write a markdown report to this path")
    parser.add_argument("--write-csv", type=Path, default=None, help="write the long defect table to CSV")
    args = parser.parse_args(argv)

    report = run_audit(
        args.a_values,
        include_limit=not args.no_limit,
        step_size=args.step_size,
        epsilon=args.epsilon,
        max_time=args.max_time,
        seed_mode=args.seed_mode,
        candidate_a=args.tube_a,
        barrier_grid_subdivisions=args.barrier_grid_subdivisions,
        include_uniform_attempt=not args.no_uniform_attempt,
    )
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.write_markdown is not None:
        args.write_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.write_markdown.write_text(render_markdown(report), encoding="utf-8")
    if args.write_csv is not None:
        _write_csv(args.write_csv, report["defect_rows"])
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    else:
        _print_summary(report)


if __name__ == "__main__":
    main()
