"""Next-sprint audits for the S7 SU(2)^3 large-|a| exclusion problem.

The goal of this module is diagnostic rather than proof-complete: it evaluates
regular p-sections, the normalized ``c=C/p^3`` cone, the ``D_C_IF`` integral
pieces, the structured scalar ``L``, and a first terminal-separator proxy.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import su2_cubed_scout as scout
from . import su2_cubed_tail_defect as tail


NEXT_SPRINT_AUDIT_VERSION = "s7-su2-cubed-next-sprint-audit-v1"
DEFAULT_OUTPUT_DIR = Path("experiments/s7/output")
DEFAULT_REPORT_PATH = Path("docs/s7-su2-cubed-next-sprint-report.md")
DEFAULT_P_SECTIONS = (0.45, 0.40, 0.35, 0.33, 0.30, 0.25)
DEFAULT_STABILITY_START_P = 0.65
DEFAULT_CIF_CHECKPOINTS = (0.60, 0.45, 0.40, 0.35, 0.33, 0.30, 0.25, 0.20, 0.10, 0.01, 0.001)
DEFAULT_B_VALUES = (-1e-8, -5e-9, 0.0, 5e-9, 1e-8)
DEFAULT_SIGMAS = (0.20, 0.25, 0.30, 0.36, 0.45, 0.60)
DEFAULT_K_VALUES = (0.50, 0.75, 1.00, 1.23, 1.50, 2.00)
DEFAULT_STEP_SIZE = 2.5e-4
DEFAULT_EPSILON = tail.DEFAULT_EPSILON
DEFAULT_MAX_TIME = 8.0
DEFAULT_TARGET_A = tail.DEFAULT_TUBE_CANDIDATE_A


@dataclass(frozen=True)
class SectionState:
    """One interpolated state on a fixed p-section."""

    b: float
    p: float
    t: float
    x1: float
    x2: float
    x3: float
    c_value: float
    c_normalized: float
    dpdt: float

    def as_dict(self) -> dict:
        return {
            "b": self.b,
            "p": self.p,
            "t": self.t,
            "x1": self.x1,
            "x2": self.x2,
            "x3": self.x3,
            "C": self.c_value,
            "c": self.c_normalized,
            "dpdt": self.dpdt,
        }


@dataclass(frozen=True)
class TraceResult:
    """One forward t-time trace with section and integral records."""

    b: float
    status: str
    crossing_time: float | None
    crossing_state: tuple[float, float, float, float] | None
    sections: dict[float, SectionState]
    cif_records: dict[float, dict]
    endpoint_cif: dict | None
    samples: tuple[tuple[float, tuple[float, float, float, float]], ...]
    message: str | None = None


def _state_c(p: float, x1: float, x2: float, x3: float) -> float:
    return x1 * x2 - p * p * x3 / 6.0


def _initial_state(epsilon: float, b: float) -> tuple[float, float, float, float]:
    return tail.scaled_taylor_seed(epsilon, b)


def _rhs_for_b(t: float, x: tuple[float, float, float, float], b: float) -> tuple[float, float, float, float]:
    return tail.scaled_rhs_with_b(t, x, b)


def _rk4_step_b(t: float, x: tuple[float, float, float, float], step: float, b: float) -> tuple[float, float, float, float]:
    return tail._rk4_step_b(t, x, step, b)


def _interpolate_state(
    t: float,
    x: tuple[float, float, float, float],
    next_t: float,
    x_next: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, tuple[float, float, float, float]]:
    return (
        t + alpha * (next_t - t),
        tuple(value + alpha * (next_value - value) for value, next_value in zip(x, x_next)),
    )


def _section_state(
    b: float,
    p: float,
    t: float,
    x: tuple[float, float, float, float],
) -> SectionState:
    _p_state, x1, x2, x3 = x
    c_value = _state_c(p, x1, x2, x3)
    dpdt = _rhs_for_b(t, (p, x1, x2, x3), b)[0]
    return SectionState(
        b=b,
        p=p,
        t=t,
        x1=x1,
        x2=x2,
        x3=x3,
        c_value=c_value,
        c_normalized=c_value / (p**3),
        dpdt=dpdt,
    )


def _cif_integrand(t: float, x: tuple[float, float, float, float]) -> tuple[float, float, float]:
    p, x1, x2, x3 = x
    if p <= 0.0:
        raise ValueError("CIF integrand requires p>0")
    return (
        2.0 * t**3 * x2 * x3**3 / p**3,
        -t**3 * p**3,
        x1 * t**7 * p**3 / 108.0,
    )


def _add3(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _scale3(value: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (value[0] * scale, value[1] * scale, value[2] * scale)


def _cif_record(
    b: float,
    p: float,
    t: float,
    x: tuple[float, float, float, float],
    pieces: tuple[float, float, float],
    method: str,
) -> dict:
    _p_state, x1, x2, x3 = x
    c_value = _state_c(p, x1, x2, x3)
    return {
        "b": b,
        "p": p,
        "t": t,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "C": c_value,
        "c": c_value / p**3 if p > 0.0 else None,
        "I1": pieces[0],
        "I2": pieces[1],
        "I3": pieces[2],
        "Itotal": sum(pieces),
        "endpoint_identity_t4C": t**4 * c_value,
        "method": method,
    }


def _p_time_cif_rhs(
    p: float,
    z: tuple[float, float, float, float, float, float, float],
    b: float,
) -> tuple[float, float, float, float, float, float, float]:
    t, x1, x2, x3, i1, i2, i3 = z
    y_p = tail.p_time_rhs(p, (t, x1, x2, x3), b)
    pieces_t = _cif_integrand(t, (p, x1, x2, x3))
    return (
        y_p[0],
        y_p[1],
        y_p[2],
        y_p[3],
        pieces_t[0] * y_p[0],
        pieces_t[1] * y_p[0],
        pieces_t[2] * y_p[0],
    )


def _rk4_step_p_cif(
    p: float,
    z: tuple[float, float, float, float, float, float, float],
    step: float,
    b: float,
) -> tuple[float, float, float, float, float, float, float]:
    k1 = _p_time_cif_rhs(p, z, b)
    z2 = tuple(value + 0.5 * step * slope for value, slope in zip(z, k1))
    k2 = _p_time_cif_rhs(p + 0.5 * step, z2, b)
    z3 = tuple(value + 0.5 * step * slope for value, slope in zip(z, k2))
    k3 = _p_time_cif_rhs(p + 0.5 * step, z3, b)
    z4 = tuple(value + step * slope for value, slope in zip(z, k3))
    k4 = _p_time_cif_rhs(p + step, z4, b)
    return tuple(
        value + step * (s1 + 2.0 * s2 + 2.0 * s3 + s4) / 6.0
        for value, s1, s2, s3, s4 in zip(z, k1, k2, k3, k4)
    )


def _extend_cif_tail_records_in_p(
    b: float,
    start_record: dict,
    *,
    targets: tuple[float, ...] = (0.001,),
    p_step: float = 1e-4,
    p_min: float = 1e-8,
) -> tuple[dict[float, dict], dict]:
    """Extend CIF pieces through the tiny tail using p-time quadrature."""
    p = float(start_record["p"])
    z = (
        float(start_record["t"]),
        float(start_record["x1"]),
        float(start_record["x2"]),
        float(start_record["x3"]),
        float(start_record["I1"]),
        float(start_record["I2"]),
        float(start_record["I3"]),
    )
    records: dict[float, dict] = {}
    for target in sorted((value for value in targets if value > 0.0 and value < p), reverse=True):
        while p > target:
            step = -min(p_step, p - target)
            try:
                z = _rk4_step_p_cif(p, z, step, b)
            except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError) as exc:
                failed = {
                    **start_record,
                    "p": p,
                    "method": "p_time_tail_failed",
                    "message": str(exc),
                }
                return records, failed
            p += step
        t, x1, x2, x3, i1, i2, i3 = z
        records[target] = _cif_record(
            b,
            target,
            t,
            (target, x1, x2, x3),
            (i1, i2, i3),
            "p_time_tail",
        )
    while p > p_min:
        step = -min(p_step, p - p_min)
        try:
            z = _rk4_step_p_cif(p, z, step, b)
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError) as exc:
            failed = {
                **start_record,
                "p": p,
                "method": "p_time_tail_failed",
                "message": str(exc),
            }
            return records, failed
        p += step
    t, x1, x2, x3, i1, i2, i3 = z
    endpoint_c = x1 * x2
    endpoint = {
        "b": b,
        "p": 0.0,
        "tail_p_min": p_min,
        "t": t,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "C": endpoint_c,
        "c": None,
        "I1": i1,
        "I2": i2,
        "I3": i3,
        "Itotal": i1 + i2 + i3,
        "endpoint_identity_t4C": t**4 * endpoint_c,
        "identity_minus_integral": t**4 * endpoint_c - (i1 + i2 + i3),
        "method": "p_time_tail_to_p_min_plus_endpoint_identity",
    }
    return records, endpoint


def trace_b_family(
    b: float,
    *,
    p_sections: Iterable[float] = DEFAULT_P_SECTIONS,
    cif_checkpoints: Iterable[float] = DEFAULT_CIF_CHECKPOINTS,
    step_size: float = DEFAULT_STEP_SIZE,
    epsilon: float = DEFAULT_EPSILON,
    max_time: float = DEFAULT_MAX_TIME,
    sample_stride: int = 20,
) -> TraceResult:
    """Trace one scaled b-family trajectory and record requested p-events."""
    requested_sections = sorted({float(value) for value in p_sections}, reverse=True)
    requested_cif = sorted({float(value) for value in cif_checkpoints}, reverse=True)
    all_targets = sorted(set(requested_sections) | set(requested_cif), reverse=True)
    target_index = 0
    t = epsilon
    x = _initial_state(epsilon, b)
    pieces = (0.0, 0.0, 0.0)
    sections: dict[float, SectionState] = {}
    cif_records: dict[float, dict] = {}
    samples: list[tuple[float, tuple[float, float, float, float]]] = [(t, x)]
    steps = 0

    while t < max_time:
        step = min(step_size, max_time - t)
        try:
            x_next = _rk4_step_b(t, x, step, b)
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError) as exc:
            return TraceResult(b, "failed", None, None, sections, cif_records, None, tuple(samples), str(exc))
        if not all(math.isfinite(value) for value in x_next):
            return TraceResult(b, "failed", None, None, sections, cif_records, None, tuple(samples), "nonfinite state")
        next_t = t + step

        if x[0] > 0.0 and x_next[0] > 0.0:
            f0 = _cif_integrand(t, x)
            f1 = _cif_integrand(next_t, x_next)
            full_increment = _scale3(_add3(f0, f1), 0.5 * step)
        else:
            f0 = (0.0, 0.0, 0.0)
            full_increment = (0.0, 0.0, 0.0)

        while target_index < len(all_targets) and x_next[0] <= all_targets[target_index] <= x[0]:
            target_p = all_targets[target_index]
            alpha = (x[0] - target_p) / (x[0] - x_next[0])
            target_t, target_x = _interpolate_state(t, x, next_t, x_next, alpha)
            if x[0] > 0.0 and target_p > 0.0:
                f_target = _cif_integrand(target_t, (target_p, target_x[1], target_x[2], target_x[3]))
                target_pieces = _add3(pieces, _scale3(_add3(f0, f_target), 0.5 * step * alpha))
            else:
                target_pieces = pieces
            if target_p in requested_sections:
                sections[target_p] = _section_state(
                    b,
                    target_p,
                    target_t,
                    (target_p, target_x[1], target_x[2], target_x[3]),
                )
            if target_p in requested_cif:
                cif_records[target_p] = _cif_record(
                    b,
                    target_p,
                    target_t,
                    (target_p, target_x[1], target_x[2], target_x[3]),
                    target_pieces,
                    "t_time_trapezoid",
                )
            target_index += 1

        if x_next[0] <= 0.0:
            alpha = abs(x[0]) / (abs(x[0]) + abs(x_next[0]))
            crossing_t, crossing_x = _interpolate_state(t, x, next_t, x_next, alpha)
            endpoint = _cif_record(
                b,
                0.0,
                crossing_t,
                (0.0, crossing_x[1], crossing_x[2], crossing_x[3]),
                pieces,
                "endpoint_identity_not_piece_decomposed",
            )
            if 0.1 in cif_records:
                refined_records, endpoint = _extend_cif_tail_records_in_p(b, cif_records[0.1], targets=(0.01, 0.001))
                cif_records.update(refined_records)
                endpoint["crossing_time_linear"] = crossing_t
                endpoint["crossing_x_linear"] = list(crossing_x)
            elif 0.01 in cif_records:
                refined_records, endpoint = _extend_cif_tail_records_in_p(b, cif_records[0.01], targets=(0.001,))
                cif_records.update(refined_records)
                endpoint["crossing_time_linear"] = crossing_t
                endpoint["crossing_x_linear"] = list(crossing_x)
            elif 0.001 in cif_records:
                _refined_records, endpoint = _extend_cif_tail_records_in_p(b, cif_records[0.001], targets=())
                endpoint["crossing_time_linear"] = crossing_t
                endpoint["crossing_x_linear"] = list(crossing_x)
            return TraceResult(
                b,
                "crossed",
                crossing_t,
                (0.0, crossing_x[1], crossing_x[2], crossing_x[3]),
                sections,
                cif_records,
                endpoint,
                tuple(samples),
            )

        pieces = _add3(pieces, full_increment)
        t = next_t
        x = x_next
        steps += 1
        if steps % sample_stride == 0:
            samples.append((t, x))

    return TraceResult(b, "no_crossing", None, None, sections, cif_records, None, tuple(samples))


def _best_threshold(values: Iterable[float], candidates: Iterable[float], *, side: str) -> tuple[float | None, float | None]:
    vals = list(values)
    if side == "upper_negative":
        max_value = max(vals)
        valid = [(candidate, -candidate - max_value) for candidate in candidates if max_value <= -candidate]
    elif side == "lower":
        min_value = min(vals)
        valid = [(candidate, min_value - candidate) for candidate in candidates if min_value >= candidate]
    else:
        raise ValueError("unknown threshold side")
    if not valid:
        return None, None
    return max(valid, key=lambda item: item[0])


def _max_deviation(reference: SectionState, samples: Iterable[SectionState]) -> float:
    ref_values = reference.as_dict()
    keys = ("t", "x1", "x2", "x3", "C", "c", "dpdt")
    return max(abs(sample.as_dict()[key] - ref_values[key]) for sample in samples for key in keys)


def _sampled_jacobian_norm(t: float, x: tuple[float, float, float, float], *, dx: float = 1e-5) -> float:
    base = _rhs_for_b(t, x, 0.0)
    max_row_sum = 0.0
    columns: list[tuple[float, float, float, float]] = []
    for index in range(4):
        xp = list(x)
        xm = list(x)
        xp[index] += dx
        xm[index] -= dx
        if index == 0 and xm[index] <= 0.0:
            xm[index] = x[index] * 0.5
        fp = _rhs_for_b(t, tuple(xp), 0.0)
        fm = _rhs_for_b(t, tuple(xm), 0.0)
        denominator = xp[index] - xm[index]
        columns.append(tuple((fp[row] - fm[row]) / denominator for row in range(4)))
    for row in range(4):
        max_row_sum = max(max_row_sum, sum(abs(columns[col][row]) for col in range(4)))
    if not all(math.isfinite(value) for value in base):
        return math.inf
    return max_row_sum


def event_stability_summary(
    limit_trace: TraceResult,
    chosen_p: float,
    *,
    start_p: float = DEFAULT_STABILITY_START_P,
    initial_delta: float | None = None,
    max_b: float = 1e-8,
) -> dict:
    """Return sampled regular-section event-map stability constants."""
    samples = [(t, x) for t, x in limit_trace.samples if chosen_p <= x[0] <= start_p]
    if not samples:
        return {"status": "inconclusive", "blocker": "no limiting samples before chosen p-section"}
    jac_norms = []
    partial_b_norms = []
    dpdt_abs = []
    p_values = []
    t_values = []
    x_columns = [[], [], [], []]
    for t, x in samples:
        p_values.append(x[0])
        t_values.append(t)
        for index, value in enumerate(x):
            x_columns[index].append(value)
        jac_norms.append(_sampled_jacobian_norm(t, x))
        r1, _r2, _r3 = tail.finite_a_error_coefficients(t, x)
        partial_b_norms.append(max(abs(value) for value in r1))
        dpdt_abs.append(abs(_rhs_for_b(t, x, 0.0)[0]))
    duration = max(t_values) - min(t_values)
    l_bound = max(jac_norms)
    m_bound = max(partial_b_norms)
    transversality = min(dpdt_abs)
    if initial_delta is None:
        initial_delta = max(
            abs(a - b)
            for a, b in zip(_initial_state(DEFAULT_EPSILON, max_b), _initial_state(DEFAULT_EPSILON, 0.0))
        )
    if math.isfinite(l_bound) and l_bound * duration < 50.0:
        gronwall_state_error = math.exp(l_bound * duration) * (initial_delta + max_b * m_bound * duration)
    else:
        gronwall_state_error = math.inf
    event_time_error = math.inf if transversality == 0.0 else gronwall_state_error / transversality
    box = {
        "t": [min(t_values), max(t_values)],
        "p": [min(p_values), max(p_values)],
        "x1": [min(x_columns[1]), max(x_columns[1])],
        "x2": [min(x_columns[2]), max(x_columns[2])],
        "x3": [min(x_columns[3]), max(x_columns[3])],
    }
    status = "promising" if math.isfinite(gronwall_state_error) and gronwall_state_error < 1e-3 else (
        "inconclusive, with exact blocker"
    )
    blocker = None if status == "promising" else "sampled row-sum Gronwall bound is too pessimistic for a proof certificate"
    return {
        "status": status,
        "chosen_p": chosen_p,
        "regular_start_p": start_p,
        "compact_box": box,
        "sampled_L_row_sum": l_bound,
        "sampled_partial_b_max": m_bound,
        "sampled_transversality_m": transversality,
        "duration": duration,
        "initial_delta_at_regular_start_for_abs_b_le_1e-8": initial_delta,
        "predicted_state_error_for_abs_b_le_1e-8": gronwall_state_error,
        "predicted_event_time_error": event_time_error,
        "blocker": blocker,
    }


def psection_audit(
    traces: Iterable[TraceResult],
    *,
    p_sections: Iterable[float] = DEFAULT_P_SECTIONS,
    sigmas: Iterable[float] = DEFAULT_SIGMAS,
    k_values: Iterable[float] = DEFAULT_K_VALUES,
) -> dict:
    """Audit regular p-sections and normalized c-cone margins."""
    trace_by_b = {trace.b: trace for trace in traces}
    if 0.0 not in trace_by_b:
        raise ValueError("psection audit requires a b=0 trace")
    rows = []
    for p in p_sections:
        states = [trace.sections[p] for trace in trace_by_b.values() if p in trace.sections]
        if len(states) != len(trace_by_b):
            rows.append({"p": p, "status": "inconclusive, with exact blocker", "blocker": "missing section record"})
            continue
        limit_state = trace_by_b[0.0].sections[p]
        best_sigma, sigma_margin = _best_threshold((state.x3 for state in states), sigmas, side="upper_negative")
        best_k, k_margin = _best_threshold((state.c_normalized for state in states), k_values, side="lower")
        finite_dev = _max_deviation(limit_state, states)
        x1_min = min(state.x1 for state in states)
        x2_min = min(state.x2 for state in states)
        dpdt_min = min(abs(state.dpdt) for state in states)
        max_x3 = max(state.x3 for state in states)
        min_c = min(state.c_normalized for state in states)
        proof_sigma = 0.36 if max_x3 <= -0.36 else best_sigma
        proof_k = 1.23 if min_c >= 1.23 else best_k
        proof_sigma_margin = (-proof_sigma - max_x3) if proof_sigma is not None else None
        proof_k_margin = (min_c - proof_k) if proof_k is not None else None
        cone_ready = (
            proof_sigma is not None
            and proof_k is not None
            and proof_sigma >= 0.36
            and proof_k >= 1.23
            and x1_min > 0.0
            and x2_min > 0.0
            and dpdt_min > 1e-3
        )
        status = "promising" if cone_ready else "not promising"
        if cone_ready and (sigma_margin is not None and k_margin is not None):
            if finite_dev > 0.5 * min(max(sigma_margin, 1e-30), max(k_margin, 1e-30)):
                status = "inconclusive, with exact blocker"
        row = {
            "p": p,
            "status": status,
            "limit": limit_state.as_dict(),
            "finite_b_samples": [state.as_dict() for state in sorted(states, key=lambda item: item.b)],
            "best_sigma": best_sigma,
            "x3_margin": sigma_margin,
            "best_K": best_k,
            "c_margin": k_margin,
            "proof_sigma": proof_sigma,
            "proof_x3_margin": proof_sigma_margin,
            "proof_K": proof_k,
            "proof_c_margin": proof_k_margin,
            "x1_min": x1_min,
            "x2_min": x2_min,
            "abs_dpdt_min": dpdt_min,
            "finite_b_max_deviation_from_limit": finite_dev,
            "recommended_cone": None,
            "blocker": None,
        }
        if cone_ready:
            row["recommended_cone"] = {
                "p_max": p,
                "x3_upper": -proof_sigma,
                "c_lower": proof_k,
                "x1_lower_observed": x1_min,
                "x2_lower_observed": x2_min,
            }
        else:
            row["blocker"] = "section does not meet x3<=-0.36 and c>=1.23 with positive observed margins"
        rows.append(row)

    promising = [row for row in rows if row.get("status") == "promising"]
    chosen = min(promising, key=lambda row: row["p"], default=None)
    if promising:
        chosen = sorted(promising, key=lambda row: (-row["p"], row["finite_b_max_deviation_from_limit"]))[0]
    limit_trace = trace_by_b[0.0]
    if chosen and all(DEFAULT_STABILITY_START_P in trace.sections for trace in trace_by_b.values()):
        start_states = [trace.sections[DEFAULT_STABILITY_START_P] for trace in trace_by_b.values()]
        initial_delta = _max_deviation(trace_by_b[0.0].sections[DEFAULT_STABILITY_START_P], start_states)
    else:
        initial_delta = None
    stability = event_stability_summary(limit_trace, chosen["p"], initial_delta=initial_delta) if chosen else {
        "status": "inconclusive, with exact blocker",
        "blocker": "no p-section met the coarse cone thresholds",
    }
    barrier = None
    if chosen:
        cone = chosen["recommended_cone"]
        barrier = tail.late_scalar_barrier_report(
            candidate_a=DEFAULT_TARGET_A,
            sigma=abs(float(cone["x3_upper"])),
            k_value=float(cone["c_lower"]),
            p_max=float(chosen["p"]),
            grid_subdivisions=8,
        )
    return {
        "version": NEXT_SPRINT_AUDIT_VERSION,
        "p_sections": rows,
        "chosen_section": chosen,
        "finite_b_event_map_stability": stability,
        "normalized_c_cone": {
            "status": "promising" if barrier and barrier.get("status") == "scalar_margins_positive" else (
                "inconclusive, with exact blocker"
            ),
            "barrier_report": barrier,
            "blocker": None if barrier and barrier.get("status") == "scalar_margins_positive" else (
                "no selected section could be checked by the scalar late-cone wall helper"
            ),
        },
    }


def cif_integral_audit(traces: Iterable[TraceResult]) -> dict:
    """Audit cumulative pieces of the D_C_IF integral identity."""
    rows_by_b = {}
    endpoint_rows = {}
    for trace in traces:
        records = [trace.cif_records[p] for p in sorted(trace.cif_records, reverse=True)]
        rows_by_b[str(trace.b)] = records
        if trace.endpoint_cif is not None:
            endpoint_rows[str(trace.b)] = trace.endpoint_cif
    limit_records = rows_by_b.get("0.0", [])
    limit_by_p = {record["p"]: record for record in limit_records}
    selected = {}
    for p in (0.45, 0.40, 0.35, 0.33, 0.30, 0.25, 0.20, 0.10, 0.01, 0.001):
        if p in limit_by_p:
            selected[p] = limit_by_p[p]
    endpoint = endpoint_rows.get("0.0")
    total_at_033 = selected.get(0.33, {}).get("Itotal")
    total_at_025 = selected.get(0.25, {}).get("Itotal")
    endpoint_total = endpoint.get("endpoint_identity_t4C") if endpoint else None
    finite_totals = [row.get("endpoint_identity_t4C") for row in endpoint_rows.values() if row.get("endpoint_identity_t4C") is not None]
    finite_sign_stable = bool(finite_totals) and min(finite_totals) > 0.0
    if total_at_033 is not None and total_at_033 > 1.0 and finite_sign_stable:
        status = "promising"
        blocker = None
    elif endpoint_total is not None and endpoint_total > 1.0 and finite_sign_stable:
        status = "inconclusive, with exact blocker"
        blocker = "positive endpoint identity is stable, but cumulative pieces do not yet give a compact dominance proof"
    else:
        status = "not promising"
        blocker = "CIF sign was not stable in the sampled traces"
    return {
        "version": NEXT_SPRINT_AUDIT_VERSION,
        "status": status,
        "blocker": blocker,
        "checkpoints_by_b": rows_by_b,
        "endpoint_by_b": endpoint_rows,
        "limit_summary": {
            "Itotal_at_p_0.33": total_at_033,
            "Itotal_at_p_0.25": total_at_025,
            "endpoint_identity_t4C": endpoint_total,
            "finite_endpoint_identity_min": min(finite_totals) if finite_totals else None,
            "finite_endpoint_identity_max": max(finite_totals) if finite_totals else None,
        },
    }


def l_scalar(t: float, p: float, x1: float, x2: float, x3: float) -> float:
    c_value = _state_c(p, x1, x2, x3)
    return x3**3 - 0.5 * t * t * x1 * c_value


def lprime_formula(t: float, p: float, x1: float, x2: float, x3: float) -> float:
    c_value = _state_c(p, x1, x2, x3)
    bracket = (
        54.0 * c_value**2 * t**4 * x1
        - 648.0 * c_value * p**3 * t**2 * x1
        + 36.0 * c_value * p**2 * t**4 * x1 * x3
        + p**6 * t**6 * x1**2
        - 108.0 * p**6 * t**2 * x1
        + 18.0 * p**4 * t**4 * x1 * x3**2
        - 3888.0 * p**4 * x3**2
        + 1296.0 * p**3 * x3**3
        + 36.0 * p**2 * t**2 * x3**4
    )
    return -bracket / (216.0 * p**3 * t)


def _lprime_chain_rule(t: float, p: float, x1: float, x2: float, x3: float) -> float:
    x = (p, x1, x2, x3)
    rhs = _rhs_for_b(t, x, 0.0)
    eps = 1e-6
    base = l_scalar(t, p, x1, x2, x3)
    dt_part = (l_scalar(t + eps, p, x1, x2, x3) - l_scalar(t - eps, p, x1, x2, x3)) / (2.0 * eps)
    grad = []
    values = [p, x1, x2, x3]
    for index in range(4):
        plus = values[:]
        minus = values[:]
        plus[index] += eps
        minus[index] -= eps
        if index == 0 and minus[index] <= 0.0:
            minus[index] = values[index] * 0.5
        grad.append(
            (l_scalar(t, plus[0], plus[1], plus[2], plus[3]) - l_scalar(t, minus[0], minus[1], minus[2], minus[3]))
            / (plus[index] - minus[index])
        )
    if not math.isfinite(base):
        return math.nan
    return dt_part + sum(grad[index] * rhs[index] for index in range(4))


def l_scalar_audit(traces: Iterable[TraceResult], psections: dict) -> dict:
    rows = []
    formula_errors = []
    for trace in traces:
        for p, state in sorted(trace.sections.items(), reverse=True):
            formula = lprime_formula(state.t, p, state.x1, state.x2, state.x3)
            chain = _lprime_chain_rule(state.t, p, state.x1, state.x2, state.x3)
            formula_errors.append(abs(formula - chain))
            rows.append(
                {
                    "source": "limit" if trace.b == 0.0 else "finite_b",
                    "b": trace.b,
                    "p": p,
                    "t": state.t,
                    "x1": state.x1,
                    "x2": state.x2,
                    "x3": state.x3,
                    "C": state.c_value,
                    "L": l_scalar(state.t, p, state.x1, state.x2, state.x3),
                    "Lprime_formula": formula,
                    "Lprime_chain_rule": chain,
                }
            )
    limit_rows = [row for row in rows if row["source"] == "limit"]
    stable_l_sign = all(row["L"] < 0.0 for row in limit_rows) or all(row["L"] > 0.0 for row in limit_rows)
    stable_lprime_sign = all(row["Lprime_formula"] < 0.0 for row in limit_rows) or all(
        row["Lprime_formula"] > 0.0 for row in limit_rows
    )
    if stable_l_sign and stable_lprime_sign:
        status = "promising"
        blocker = None
    else:
        status = "not promising"
        blocker = "L or Lprime does not show a single useful sign across the audited regular sections"
    return {
        "version": NEXT_SPRINT_AUDIT_VERSION,
        "status": status,
        "blocker": blocker,
        "max_formula_chain_rule_error": max(formula_errors) if formula_errors else None,
        "section_rows": rows,
        "known_compact_note": "Known compact endpoint values vanish by construction; regular-section L is not expected to vanish.",
        "psection_context_status": psections.get("chosen_section", {}).get("status"),
    }


def terminal_separator_audit(traces: Iterable[TraceResult]) -> dict:
    """Attempt a regular-section separator using known compact trajectories as proxies.

    This is deliberately labeled as a proxy: the current codebase does not
    contain a backward K_- terminal Taylor chart for the Podesta coordinates.
    The two known compact trajectories are terminal-admissible samples, but
    they are not an enclosure of the full terminal-admissible set.
    """
    trace_by_b = {trace.b: trace for trace in traces}
    left_trace = trace_by_b.get(0.0)
    if left_trace is None:
        return {"status": "inconclusive, with exact blocker", "blocker": "missing b=0 left-shot trace"}
    proxy_as = (scout.ROUND_A_DIRECT, scout.SQUASHED_A_DIRECT)
    proxy_traces = [
        trace_b_family(1.0 / a, p_sections=(0.33, 0.25), cif_checkpoints=(), step_size=DEFAULT_STEP_SIZE)
        for a in proxy_as
    ]
    attempts = []
    for p in (0.33, 0.25):
        left_states = [trace.sections[p] for trace in traces if p in trace.sections]
        proxy_states = [trace.sections[p] for trace in proxy_traces if p in trace.sections]
        feature_getters = {
            "t": lambda state: state.t,
            "x1": lambda state: state.x1,
            "x2": lambda state: state.x2,
            "x3": lambda state: state.x3,
            "C": lambda state: state.c_value,
            "c": lambda state: state.c_normalized,
        }
        best = None
        for name, getter in feature_getters.items():
            left_values = [getter(state) for state in left_states]
            proxy_values = [getter(state) for state in proxy_states]
            if max(left_values) < min(proxy_values):
                margin = min(proxy_values) - max(left_values)
                direction = "left_below_proxy"
            elif max(proxy_values) < min(left_values):
                margin = min(left_values) - max(proxy_values)
                direction = "left_above_proxy"
            else:
                margin = -min(max(left_values) - min(proxy_values), max(proxy_values) - min(left_values))
                direction = "overlap"
            candidate = {"feature": name, "margin": margin, "direction": direction}
            if best is None or candidate["margin"] > best["margin"]:
                best = candidate
        attempts.append(
            {
                "p": p,
                "separator_type": "coordinate_proxy",
                "best_proxy_separator": best,
                "left_shot_samples": [state.as_dict() for state in left_states],
                "terminal_proxy_samples": [state.as_dict() for state in proxy_states],
                "status": "inconclusive, with exact blocker",
                "blocker": "known compact proxy samples are not an enclosure of the backward K_- terminal-admissible set",
            }
        )
    return {
        "version": NEXT_SPRINT_AUDIT_VERSION,
        "status": "inconclusive, with exact blocker",
        "blocker": "no implemented backward K_- terminal Taylor chart in the current Podesta SU(2)^3 code",
        "attempts": attempts,
    }


def run_audit(
    *,
    step_size: float = DEFAULT_STEP_SIZE,
    b_values: Iterable[float] = DEFAULT_B_VALUES,
) -> dict:
    traces = [
        trace_b_family(
            float(b),
            step_size=step_size,
            p_sections=(DEFAULT_STABILITY_START_P, *DEFAULT_P_SECTIONS),
            cif_checkpoints=DEFAULT_CIF_CHECKPOINTS,
        )
        for b in b_values
    ]
    psections = psection_audit(traces)
    cif = cif_integral_audit(traces)
    l_audit = l_scalar_audit(traces, psections)
    separator = terminal_separator_audit(traces)
    recommendations = rank_routes(psections, cif, l_audit, separator)
    return {
        "version": NEXT_SPRINT_AUDIT_VERSION,
        "step_size": step_size,
        "b_values": list(b_values),
        "trace_statuses": {str(trace.b): trace.status for trace in traces},
        "psection_audit": psections,
        "CIF_integral_audit": cif,
        "L_scalar_audit": l_audit,
        "terminal_separator_audit": separator,
        "recommendations": recommendations,
    }


def _route_score(status: str) -> int:
    if status == "promising":
        return 0
    if status == "inconclusive, with exact blocker":
        return 1
    return 2


def rank_routes(psections: dict, cif: dict, l_audit: dict, separator: dict) -> list[dict]:
    psection_status = "promising"
    if not psections.get("chosen_section"):
        psection_status = "not promising"
    elif psections.get("normalized_c_cone", {}).get("status") != "promising":
        psection_status = "inconclusive, with exact blocker"
    elif psections.get("finite_b_event_map_stability", {}).get("status") != "promising":
        psection_status = "inconclusive, with exact blocker"
    routes = [
        {
            "route": "p-section cone route",
            "status": psection_status,
            "reason": _psection_recommendation_reason(psections),
        },
        {
            "route": "D_C_IF integral route",
            "status": cif["status"],
            "reason": cif.get("blocker") or "CIF cumulative integral has stable positive sign in the sampled large-tail family.",
        },
        {
            "route": "terminal-manifold separation route",
            "status": separator["status"],
            "reason": separator.get("blocker"),
        },
    ]
    return sorted(routes, key=lambda item: (_route_score(item["status"]), item["route"]))


def _psection_recommendation_reason(psections: dict) -> str:
    chosen = psections.get("chosen_section")
    stability = psections.get("finite_b_event_map_stability", {})
    cone = psections.get("normalized_c_cone", {})
    if not chosen:
        return "No audited p-section met the coarse x3/c cone thresholds."
    if cone.get("status") != "promising":
        return cone.get("blocker") or "The normalized c-cone wall check did not pass."
    if stability.get("status") != "promising":
        return (
            f"Section p={chosen['p']} enters the cone, but the simple Gronwall/event-map proof remains "
            f"inconclusive: {stability.get('blocker')}"
        )
    return f"Section p={chosen['p']} enters x3/c cone and sampled finite-b stability is small."


def _fmt(value: float | None, digits: int = 8) -> str:
    if value is None:
        return "n/a"
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return str(value)
    return f"{float(value):.{digits}g}"


def render_markdown(audit: dict) -> str:
    psections = audit["psection_audit"]
    cif = audit["CIF_integral_audit"]
    l_audit = audit["L_scalar_audit"]
    separator = audit["terminal_separator_audit"]
    lines = [
        "# S7 SU(2)^3 Next-Sprint Audit",
        "",
        "Reproducibility command:",
        "",
        "```zsh",
        ".venv/bin/python -m experiments.s7_su2_cubed_next_sprint_audit",
        "```",
        "",
        "## Executive summary",
        "",
    ]
    best = audit["recommendations"][0]
    lines.extend(
        [
            f"Top-ranked route: **{best['route']}**.",
            "",
            f"Status: `{best['status']}`.",
            "",
            best["reason"] or "",
            "",
            "The short conclusion is that the `D_C_IF` integral route is currently the cleanest new proof target.  The regular p-section cone route found a strong `p=0.33` cone entry and positive normalized-`c` wall margins, but the available one-number event-map stability bound is still too crude to certify finite `|b|<=1e-8`.  Terminal-manifold separation remains blocked by the missing backward `K_-` terminal chart.",
            "",
            "## What Was Preserved From The Previous D_x3 Proof",
            "",
            "The existing conditional downstream tail exclusion is preserved.  Once a trajectory is in the late correlated region, the previous `D_x3` terminal/tail mechanism still gives the contradiction to compact `K_-` closure.  This sprint only tries to replace the upstream tiny `t=3.5` support-entry step.",
            "",
            "## p-section audit results",
            "",
            "| p | status | t(limit) | x3(limit) | c(limit) | best sigma | x3 margin | best K | c margin | finite-b max dev |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in psections["p_sections"]:
        limit = row.get("limit", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("p")),
                    row.get("status", ""),
                    _fmt(limit.get("t")),
                    _fmt(limit.get("x3")),
                    _fmt(limit.get("c")),
                    _fmt(row.get("best_sigma")),
                    _fmt(row.get("x3_margin")),
                    _fmt(row.get("best_K")),
                    _fmt(row.get("c_margin")),
                    _fmt(row.get("finite_b_max_deviation_from_limit")),
                ]
            )
            + " |"
        )
    chosen = psections.get("chosen_section")
    if chosen:
        lines.extend(
            [
                "",
                f"Recommended section: `p={_fmt(chosen['p'])}` with cone `{chosen.get('recommended_cone')}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## finite-b event-map stability results",
            "",
        ]
    )
    stability = psections["finite_b_event_map_stability"]
    lines.extend(
        [
            f"Status: `{stability.get('status')}`.",
            "",
            f"Sampled `L`: `{_fmt(stability.get('sampled_L_row_sum'))}`.",
            f"Sampled `partial_b` max: `{_fmt(stability.get('sampled_partial_b_max'))}`.",
            f"Transversality `m`: `{_fmt(stability.get('sampled_transversality_m'))}`.",
            f"Regular segment: `p={_fmt(stability.get('regular_start_p'))}` to `p={_fmt(stability.get('chosen_p'))}`.",
            f"Predicted state error from crude Gronwall: `{_fmt(stability.get('predicted_state_error_for_abs_b_le_1e-8'))}`.",
            "",
            f"Blocker: {stability.get('blocker') or 'none'}",
            "",
            "## normalized c=C/p^3 cone results",
            "",
        ]
    )
    cone = psections["normalized_c_cone"]
    barrier = cone.get("barrier_report") or {}
    lines.extend(
        [
            f"Status: `{cone.get('status')}`.",
            "",
            f"`x3` wall margin: `{_fmt(barrier.get('x3_wall_margin'))}`.",
            f"`c` wall limiting hdot lower: `{_fmt(barrier.get('c_wall_limiting_hdot_lower'))}`.",
            f"finite-b grid hdot margin: `{_fmt(barrier.get('finite_b_grid_hdot_margin'))}`.",
            "",
            "## D_C_IF integral decomposition",
            "",
            f"Status: `{cif['status']}`.",
            "",
            f"Endpoint `T^4 C(T)` in limit: `{_fmt(cif['limit_summary'].get('endpoint_identity_t4C'))}`.",
            f"Integral total at `p=0.33`: `{_fmt(cif['limit_summary'].get('Itotal_at_p_0.33'))}`.",
            f"Integral total at `p=0.25`: `{_fmt(cif['limit_summary'].get('Itotal_at_p_0.25'))}`.",
            "",
            "| p | I1 | I2 | I3 | total | endpoint t^4 C |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in cif["checkpoints_by_b"].get("0.0", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(record.get("p")),
                    _fmt(record.get("I1")),
                    _fmt(record.get("I2")),
                    _fmt(record.get("I3")),
                    _fmt(record.get("Itotal")),
                    _fmt(record.get("endpoint_identity_t4C")),
                ]
            )
            + " |"
        )
    endpoint_record = cif.get("endpoint_by_b", {}).get("0.0")
    if endpoint_record:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(endpoint_record.get("p")),
                    _fmt(endpoint_record.get("I1")),
                    _fmt(endpoint_record.get("I2")),
                    _fmt(endpoint_record.get("I3")),
                    _fmt(endpoint_record.get("Itotal")),
                    _fmt(endpoint_record.get("endpoint_identity_t4C")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## L scalar audit",
            "",
            f"Status: `{l_audit['status']}`.",
            "",
            f"Formula/chain-rule max discrepancy on audited sections: `{_fmt(l_audit.get('max_formula_chain_rule_error'))}`.",
            "",
            f"Blocker: {l_audit.get('blocker') or 'none'}",
            "",
            "## terminal-manifold separation attempt",
            "",
            f"Status: `{separator['status']}`.",
            "",
            f"Blocker: {separator.get('blocker')}",
            "",
            "| p | separator type | best feature | margin | status |",
            "|---:|---|---|---:|---|",
        ]
    )
    for attempt in separator.get("attempts", []):
        best_sep = attempt.get("best_proxy_separator") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(attempt.get("p")),
                    attempt.get("separator_type", ""),
                    str(best_sep.get("feature")),
                    _fmt(best_sep.get("margin")),
                    attempt.get("status", ""),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## recommended proof route",
            "",
            "| rank | route | status | reason |",
            "|---:|---|---|---|",
        ]
    )
    for index, route in enumerate(audit["recommendations"], start=1):
        lines.append(f"| {index} | {route['route']} | `{route['status']}` | {route.get('reason') or ''} |")
    lines.extend(
        [
            "",
            "## remaining gaps",
            "",
            "- Upgrade the p-section finite-b event-map stability estimate from sampled diagnostics to a proof.  The current one-number Gronwall bound is intentionally crude and may be too pessimistic.",
            "- Turn the normalized `c=C/p^3` cone wall margins into a full entry-and-invariance lemma from the selected regular section.",
            "- If pursuing `D_C_IF`, prove an integral dominance estimate on a compact interval and a tail bound showing later pieces cannot cancel the accumulated sign.",
            "- Derive or implement the backward smooth `K_-` terminal chart before treating terminal-manifold separation as more than a proxy diagnostic.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(audit: dict, output_dir: Path = DEFAULT_OUTPUT_DIR, report_path: Path = DEFAULT_REPORT_PATH) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "psection_audit.json").write_text(
        json.dumps(audit["psection_audit"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "CIF_integral_audit.json").write_text(
        json.dumps(audit["CIF_integral_audit"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "terminal_separator_audit.json").write_text(
        json.dumps(audit["terminal_separator_audit"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "next_sprint_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(audit), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--no-write", action="store_true", help="Run the audit without writing JSON/Markdown outputs.")
    args = parser.parse_args(argv)
    audit = run_audit(step_size=args.step_size)
    if not args.no_write:
        write_outputs(audit, args.output_dir, args.report_path)
    print("S7 SU(2)^3 next-sprint audit", flush=True)
    print(f"version: {audit['version']}", flush=True)
    print(f"top route: {audit['recommendations'][0]['route']} ({audit['recommendations'][0]['status']})", flush=True)
    if not args.no_write:
        print(f"report: {args.report_path}", flush=True)
        print(f"json output dir: {args.output_dir}", flush=True)
    return audit


if __name__ == "__main__":
    main()
