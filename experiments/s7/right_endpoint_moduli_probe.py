"""Probe whether the fixed S7 right charts are ready for full moduli search."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from mpmath import mp

from problem import (
    FixedRightEndpointData,
    ProblemParameters,
    S7_P2_RIGHT_CHART,
    S7_P3_RIGHT_CHART,
    SolverConfig,
    State,
    initial_right_series,
    round_s7_candidate_parameters,
    squashed_s7_parameters,
    weighted_series_residual,
)


OUTPUT_DIR = Path("output/s7_endpoint_moduli_probe")
DEFAULT_SERIES_ORDER = 10
DEFAULT_WORKING_DPS = 80
DEFAULT_TARGET_DPS = 30
RESIDUAL_DEGREES_TO_CHECK = 6


@dataclass(frozen=True)
class TargetProbe:
    """One right endpoint target and its current fixed chart."""

    name: str
    params: ProblemParameters
    chart_name: str


@dataclass(frozen=True)
class TargetDiagnostics:
    """Probe results for one S7 right endpoint chart."""

    target: str
    right_chart: str
    fixed_right_label: str
    offset_ratios: tuple[mp.mpf, ...]
    collapse_defect: mp.mpf
    berger_form_defect: mp.mpf
    explicit_seed_residual_norm: mp.mpf
    recurrence_seed_residual_norm: mp.mpf
    global_solve_residual_norm: mp.mpf
    search_ready: bool


def _target_probe(name: str) -> TargetProbe:
    """Return one named S7 target probe."""
    if name == "round":
        return TargetProbe(name, round_s7_candidate_parameters(), S7_P3_RIGHT_CHART.name)
    if name == "squashed":
        return TargetProbe(name, squashed_s7_parameters(), S7_P2_RIGHT_CHART.name)
    raise ValueError(f"Unknown S7 target {name!r}; choose round, squashed, or both.")


def _chart_for_probe(probe: TargetProbe):
    """Return the weighted chart object for one target probe."""
    if probe.chart_name == S7_P3_RIGHT_CHART.name:
        return S7_P3_RIGHT_CHART
    if probe.chart_name == S7_P2_RIGHT_CHART.name:
        return S7_P2_RIGHT_CHART
    raise ValueError(f"Unknown S7 right chart {probe.chart_name!r}.")


def _config(params: ProblemParameters, order: int, dps: int, target_dps: int) -> SolverConfig:
    """Return a compact right-end Taylor validation config."""
    return SolverConfig(order, dps, target_dps, mp.mpf("0.5"), 0, params.interval_end / 2)


def _max_abs(values) -> mp.mpf:
    """Return the largest absolute value in an iterable."""
    return max(abs(value) for value in values)


def _series_residual_norm(probe: TargetProbe, config: SolverConfig) -> mp.mpf:
    """Return the residual norm for the currently trusted explicit S7 right seed."""
    chart = _chart_for_probe(probe)
    coeffs = initial_right_series(probe.params, config)
    residual = weighted_series_residual(chart, coeffs, mp.zero, probe.params)
    return _max_abs(value for component in residual for value in component[:RESIDUAL_DEGREES_TO_CHECK])


def _generic_probe(probe: TargetProbe) -> TargetProbe:
    """Return a probe with the explicit homogeneous-series shortcut disabled."""
    fixed = probe.params.fixed_right
    if fixed is None:
        raise ValueError("S7 probe requires fixed right endpoint data.")
    generic_fixed = FixedRightEndpointData(
        label=f"{fixed.label}_generic_probe",
        offset=fixed.offset,
        zero_jet=fixed.zero_jet,
        first_jet=fixed.first_jet,
    )
    generic_params = ProblemParameters(
        lam=probe.params.lam,
        interval_end=probe.params.interval_end,
        left=probe.params.left,
        right=probe.params.right,
        right_chart=probe.params.right_chart,
        fixed_right=generic_fixed,
    )
    return TargetProbe(probe.name, generic_params, probe.chart_name)


def _recurrence_seed_residual_norm(probe: TargetProbe, config: SolverConfig) -> mp.mpf:
    """Return the residual norm after disabling the explicit homogeneous-series shortcut.

    A small value here would indicate that the stored offset/zero/first jets are
    enough for the generic Taylor recurrence to generate the endpoint family.
    The current S7 p2/p3 endpoints are not in that state; their valid seeds are
    hardcoded from the explicit homogeneous q(t) formula.
    """
    return _series_residual_norm(_generic_probe(probe), config)


def _pack_coefficients(coeffs: State[list[mp.mpf]], *, fixed_degree: int, order: int) -> np.ndarray:
    """Pack non-fixed Taylor coefficient levels into one numpy vector."""
    return np.asarray(
        [float(component[degree]) for degree in range(fixed_degree + 1, order + 1) for component in coeffs],
        dtype=np.float64,
    )


def _unpack_coefficients(
    values: np.ndarray,
    template: State[list[mp.mpf]],
    *,
    fixed_degree: int,
    order: int,
) -> State[list[mp.mpf]]:
    """Unpack one numpy vector into a coefficient state with fixed low-order data."""
    components = [list(component) for component in template]
    index = 0
    for degree in range(fixed_degree + 1, order + 1):
        for component in components:
            component[degree] = mp.mpf(values[index])
            index += 1
    return State.from_iterable(components)


def _residual_vector(
    chart,
    params: ProblemParameters,
    coeffs: State[list[mp.mpf]],
    *,
    levels: int,
) -> np.ndarray:
    """Return selected weighted residual coefficients as a float vector."""
    residual = weighted_series_residual(chart, coeffs, mp.zero, params)
    return np.asarray([float(component[degree]) for degree in range(levels) for component in residual], dtype=np.float64)


def _least_squares_endpoint_series(
    probe: TargetProbe,
    config: SolverConfig,
    *,
    fixed_degree: int = 1,
    max_iterations: int = 4,
    tolerance: float = 1e-12,
) -> State[list[mp.mpf]]:
    """Solve endpoint Taylor coefficients globally with the low jet held fixed.

    The standard recurrence solves one coefficient layer at a time.  In the S7
    right p2/p3 charts, weighted divisions make low residual coefficients depend
    on later Taylor layers, so the one-layer recurrence leaves a real residual.
    This global solve keeps the stored offset/zero/first jet fixed and corrects
    all remaining coefficient layers simultaneously.

    The last coefficient layer is underdetermined by a finite truncation; the
    least-squares solution chooses one valid representative for the truncated
    endpoint germ.
    """
    generic = _generic_probe(probe)
    chart = _chart_for_probe(generic)
    order = config.series_order
    template = initial_right_series(generic.params, config)
    levels = order - fixed_degree
    values = _pack_coefficients(template, fixed_degree=fixed_degree, order=order)

    def unpack(current: np.ndarray) -> State[list[mp.mpf]]:
        return _unpack_coefficients(current, template, fixed_degree=fixed_degree, order=order)

    def residual(current: np.ndarray) -> np.ndarray:
        return _residual_vector(chart, generic.params, unpack(current), levels=levels)

    for _iteration in range(max_iterations):
        base = residual(values)
        base_norm = float(np.max(np.abs(base)))
        if base_norm < tolerance:
            break
        jacobian = np.empty((len(base), len(values)), dtype=np.float64)
        step = 1e-6
        for index in range(len(values)):
            trial = values.copy()
            trial[index] += step
            jacobian[:, index] = (residual(trial) - base) / step

        delta = np.linalg.lstsq(jacobian, -base, rcond=1e-12)[0]
        damping = 1.0
        while damping > 1e-4:
            trial = values + damping * delta
            if float(np.max(np.abs(residual(trial)))) < base_norm:
                values = trial
                break
            damping *= 0.5
        else:
            break

    return unpack(values)


def _global_solve_residual_norm(probe: TargetProbe, config: SolverConfig) -> mp.mpf:
    """Return the residual norm after the global S7 endpoint coefficient solve."""
    generic = _generic_probe(probe)
    coeffs = _least_squares_endpoint_series(probe, config)
    chart = _chart_for_probe(generic)
    levels = config.series_order - 1
    residual = weighted_series_residual(chart, coeffs, mp.zero, generic.params)
    return _max_abs(value for component in residual for value in component[:levels])


def _offset_scale(params: ProblemParameters) -> mp.mpf:
    """Return the homogeneous S7 offset scale used by the known targets."""
    return mp.sqrt(5) / 25


def _offset_ratios(params: ProblemParameters) -> tuple[mp.mpf, ...]:
    """Return the fixed right offset divided by sqrt(5)/25."""
    if params.fixed_right is None:
        raise ValueError("S7 probe requires fixed right endpoint data.")
    scale = _offset_scale(params)
    return tuple(value / scale for value in params.fixed_right.offset)


def _berger_form_defect(offset: State[mp.mpf]) -> mp.mpf:
    """Measure distance from the Berger terminal offset form."""
    return _max_abs(
        (
            offset.y1 - 3 * offset.y2,
            offset.y3,
            offset.y4,
            offset.y5,
            offset.y6,
            offset.y7 - 3 * offset.y8,
        )
    )


def _collapse_defect(probe: TargetProbe) -> mp.mpf:
    """Measure the expected p2 or p3 collapsing offset identities."""
    if probe.params.fixed_right is None:
        raise ValueError("S7 probe requires fixed right endpoint data.")
    q = probe.params.fixed_right.offset
    if probe.chart_name == S7_P3_RIGHT_CHART.name:
        return _max_abs((q.y3 + q.y6, q.y4 + q.y5))
    if probe.chart_name == S7_P2_RIGHT_CHART.name:
        return _max_abs((q.y2 + q.y7, q.y4 + q.y5))
    raise ValueError(f"Unknown S7 right chart {probe.chart_name!r}.")


def target_diagnostics(name: str, *, order: int = DEFAULT_SERIES_ORDER, dps: int = DEFAULT_WORKING_DPS) -> TargetDiagnostics:
    """Return the right-end moduli-readiness diagnostics for one S7 target."""
    probe = _target_probe(name)
    if probe.params.fixed_right is None:
        raise ValueError("S7 probe requires fixed right endpoint data.")
    config = _config(probe.params, order, dps, DEFAULT_TARGET_DPS)
    explicit_norm = _series_residual_norm(probe, config)
    recurrence_norm = _recurrence_seed_residual_norm(probe, config)
    global_norm = _global_solve_residual_norm(probe, config)
    return TargetDiagnostics(
        target=name,
        right_chart=probe.params.right_chart,
        fixed_right_label=probe.params.fixed_right.label,
        offset_ratios=_offset_ratios(probe.params),
        collapse_defect=_collapse_defect(probe),
        berger_form_defect=_berger_form_defect(probe.params.fixed_right.offset),
        explicit_seed_residual_norm=explicit_norm,
        recurrence_seed_residual_norm=recurrence_norm,
        global_solve_residual_norm=global_norm,
        search_ready=False,
    )


def _mp_string(value: mp.mpf) -> str:
    """Return a stable JSON/string representation of an mpmath scalar."""
    if value == mp.inf:
        return "inf"
    if value == -mp.inf:
        return "-inf"
    return mp.nstr(value, 50)


def _diagnostics_payload(item: TargetDiagnostics) -> dict:
    """Return JSON-ready diagnostics."""
    return {
        "target": item.target,
        "right_chart": item.right_chart,
        "fixed_right_label": item.fixed_right_label,
        "offset_ratios": [_mp_string(value) for value in item.offset_ratios],
        "collapse_defect": _mp_string(item.collapse_defect),
        "berger_form_defect": _mp_string(item.berger_form_defect),
        "explicit_seed_residual_norm": _mp_string(item.explicit_seed_residual_norm),
        "recurrence_seed_residual_norm": _mp_string(item.recurrence_seed_residual_norm),
        "global_solve_residual_norm": _mp_string(item.global_solve_residual_norm),
        "search_ready": item.search_ready,
    }


def _format_tuple(values: tuple[mp.mpf, ...], digits: int = 8) -> str:
    """Format one tuple of mpmath values compactly."""
    return "(" + ", ".join(mp.nstr(value, digits) for value in values) + ")"


def _print_report(items: list[TargetDiagnostics], *, order: int, dps: int) -> None:
    """Print a concise right-moduli readiness report."""
    print("S7 right-endpoint moduli probe", flush=True)
    print(f"series order: {order}", flush=True)
    print(f"working dps: {dps}", flush=True)
    print()
    for item in items:
        print(f"{item.target} ({item.right_chart}, fixed label {item.fixed_right_label})", flush=True)
        print(f"  offset / (sqrt(5)/25): {_format_tuple(item.offset_ratios)}", flush=True)
        print(f"  p-collapse offset defect: {mp.nstr(item.collapse_defect, 12)}", flush=True)
        print(f"  Berger terminal form defect: {mp.nstr(item.berger_form_defect, 12)}", flush=True)
        print(f"  explicit homogeneous right seed residual: {mp.nstr(item.explicit_seed_residual_norm, 12)}", flush=True)
        print(f"  one-layer endpoint recurrence residual: {mp.nstr(item.recurrence_seed_residual_norm, 12)}", flush=True)
        print(f"  global endpoint coefficient-solve residual: {mp.nstr(item.global_solve_residual_norm, 12)}", flush=True)
        print(f"  full right-moduli search ready: {item.search_ready}", flush=True)
        print()
    if not all(item.search_ready for item in items):
        print(
            "verdict: the Taylor-coefficient solve works, but this is still not an honest 7D S7 scout until the p2/p3 right coordinates are derived.",
            flush=True,
        )


def _write_json(items: list[TargetDiagnostics], *, order: int, dps: int) -> Path:
    """Write a JSON copy of the probe results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = OUTPUT_DIR / f"{stamp}-s7-right-moduli-probe.json"
    payload = {
        "event": "s7_right_endpoint_moduli_probe",
        "series_order": order,
        "working_dps": dps,
        "residual_degrees_checked": RESIDUAL_DEGREES_TO_CHECK,
        "targets": [_diagnostics_payload(item) for item in items],
        "search_ready": all(item.search_ready for item in items),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Probe whether S7 right charts have parameterized moduli data.")
    parser.add_argument("--target", choices=("round", "squashed", "both"), default="both")
    parser.add_argument("--order", type=int, default=DEFAULT_SERIES_ORDER)
    parser.add_argument("--dps", type=int, default=DEFAULT_WORKING_DPS)
    parser.add_argument("--write-json", action="store_true", help="write a JSON report under output/s7_endpoint_moduli_probe")
    return parser.parse_args()


def main() -> None:
    """Run the S7 right endpoint moduli probe."""
    args = _parse_args()
    mp.dps = args.dps
    names = ("round", "squashed") if args.target == "both" else (args.target,)
    items = [target_diagnostics(name, order=args.order, dps=args.dps) for name in names]
    _print_report(items, order=args.order, dps=args.dps)
    if args.write_json:
        path = _write_json(items, order=args.order, dps=args.dps)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
