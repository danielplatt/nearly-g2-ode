"""Direct homogeneous recovery check for the ``N_{1,1}`` action model.

The endpoint-reduced scout chart is still provisional.  This module instead
uses the exact homogeneous trajectory induced by the
``SO(3)_real x SO(3)_fiber`` action, then integrates from both ends to an
interior slice.  It is a calibration of the ODE/action model before returning
to blind endpoint scouting.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mpmath import mp
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from experiments.shared.non_mirrored_common import RANDOM_SEED, _event, _mp_string, _output_paths, _write_jsonl_event, _write_summary

from . import ansatz
from .evolution import (
    AWSettings,
    GAMMA_BASIS,
    OMEGA_BASIS,
    algebraic_residual,
    omega_form,
    omega_volume,
    rhs,
)


HOMOGENEOUS_RECOVERY_VERSION = "aloff-wallach-n11-homogeneous-recovery-v1"
OUTPUT_DIR = Path("output/aloff_wallach_homogeneous_recoveries")
OUTPUT_SUFFIX = HOMOGENEOUS_RECOVERY_VERSION
TARGET_NAMES = tuple(solution.label for solution in ansatz.n11_known_solutions())
BASE_STRUCTURE_SCALE = -1.0
FIBER_STRUCTURE_SCALE = -2.0


@dataclass(frozen=True)
class HomogeneousTrajectory:
    """Exact product-action trajectory data for one homogeneous solution."""

    target: str
    lambda_scale: float
    normal_speed: float
    interval_t: float
    phi: ansatz.Form
    action_basis: tuple[np.ndarray, ...]
    normal_matrix: np.ndarray
    metric_c: float


def _target_solution(target: str) -> ansatz.N11KnownSolution:
    for solution in ansatz.n11_known_solutions():
        if solution.label == target:
            return solution
    raise ValueError(f"unknown Aloff-Wallach target {target!r}")


def _basis_components(matrix: np.ndarray) -> np.ndarray:
    basis = ansatz.n11_su3_basis_matrices()[:7]
    return np.array([ansatz._inner(matrix, item) for item in basis], dtype=float)


def _evaluate_form(item: ansatz.Form, vectors: list[np.ndarray]) -> float:
    rows = np.array(vectors, dtype=float)
    total = 0.0
    for key, coefficient in item.items():
        total += float(coefficient) * float(np.linalg.det(rows[:, [index - 1 for index in key]]))
    return total


def homogeneous_trajectory(target: str, lam: float = 4.0) -> HomogeneousTrajectory:
    """Return exact action-basis trajectory data for a known homogeneous target."""
    solution = _target_solution(target)
    matrices = ansatz.n11_su3_basis_matrices()
    sqrt2 = np.sqrt(2.0)
    lambda_scale = float(solution.lambda_value / mp.mpf(str(lam)))
    A = float(solution.A) * lambda_scale
    B = float(solution.B) * lambda_scale
    C = float(solution.C) * lambda_scale
    D = float(solution.D) * lambda_scale
    phi = ansatz.aloff_wallach_phi(mp.mpf(A), mp.mpf(B), mp.mpf(C), mp.mpf(D))

    # Action basis:
    #   base_3 and fiber_3 are scaled to make base_3 + fiber_3 collapse at RP2.
    #   This gives Maurer-Cartan scales -1 on the base factor and -2 on fiber.
    base = (sqrt2 * matrices[1], sqrt2 * matrices[2], sqrt2 * matrices[0])
    fiber = (sqrt2 * matrices[3], sqrt2 * matrices[4], sqrt2 * matrices[0])
    action_basis = base + tuple(-item for item in fiber)
    normal_matrix = sqrt2 * matrices[6]
    metric_c = C
    return HomogeneousTrajectory(
        target=target,
        lambda_scale=lambda_scale,
        normal_speed=sqrt2 * metric_c,
        interval_t=float(np.pi / 4),
        phi=phi,
        action_basis=action_basis,
        normal_matrix=normal_matrix,
        metric_c=metric_c,
    )


def trajectory_state(trajectory: HomogeneousTrajectory, t: float, *, normal_sign: int = -1) -> np.ndarray:
    """Return the 19 ``omega,gamma`` coefficients at action-coordinate ``t``."""
    a = expm(t * trajectory.normal_matrix)
    tangent_vectors = [
        _basis_components(a.conj().T @ item @ a)
        for item in trajectory.action_basis[:3]
    ] + [_basis_components(item) for item in trajectory.action_basis[3:]]
    normal = normal_sign * _basis_components(trajectory.normal_matrix) / trajectory.normal_speed
    values = []
    for key in OMEGA_BASIS:
        values.append(_evaluate_form(trajectory.phi, [normal, tangent_vectors[key[0] - 1], tangent_vectors[key[1] - 1]]))
    for key in GAMMA_BASIS:
        values.append(_evaluate_form(trajectory.phi, [tangent_vectors[index - 1] for index in key]))
    return np.array(values, dtype=float)


def fiber_invariance_norm(target: str) -> float:
    """Return the largest infinitesimal right-fiber invariance defect."""
    solution = _target_solution(target)
    phi = ansatz.aloff_wallach_phi(solution.A, solution.B, solution.C, solution.D)
    matrices = ansatz.n11_su3_basis_matrices()
    vertical = (matrices[0], matrices[3], matrices[4])
    basis = matrices[:7]

    def ad_matrix(generator: np.ndarray) -> np.ndarray:
        columns = []
        for item in basis:
            columns.append(_basis_components(generator @ item - item @ generator))
        return np.stack(columns, axis=1)

    def form_array() -> np.ndarray:
        array = np.zeros((7, 7, 7), dtype=float)
        for key, coefficient in phi.items():
            for permuted in __import__("itertools").permutations(key):
                inversions = sum(1 for i, left in enumerate(permuted) for right in permuted[i + 1 :] if left > right)
                array[tuple(index - 1 for index in permuted)] = float(coefficient) * ((-1.0) ** inversions)
        return array

    coefficients = form_array()
    worst = 0.0
    for generator in vertical:
        action = ad_matrix(generator)
        derivative = np.zeros_like(coefficients)
        for a_index in range(7):
            for b_index in range(7):
                for c_index in range(7):
                    total = 0.0
                    for p_index in range(7):
                        total -= action[p_index, a_index] * coefficients[p_index, b_index, c_index]
                        total -= action[p_index, b_index] * coefficients[a_index, p_index, c_index]
                        total -= action[p_index, c_index] * coefficients[a_index, b_index, p_index]
                    derivative[a_index, b_index, c_index] = total
        worst = max(worst, float(np.max(np.abs(derivative))))
    return worst


def _rhs_forward(state: np.ndarray, settings: AWSettings) -> np.ndarray:
    return rhs(
        state,
        settings.lam,
        settings.structure_scale,
        base_structure_scale=settings.base_structure_scale,
        fiber_structure_scale=settings.fiber_structure_scale,
    )


def _integrate(state: np.ndarray, duration: float, sign: float, settings: AWSettings) -> tuple[np.ndarray, bool, str]:
    def ode(_tau: float, current: np.ndarray) -> np.ndarray:
        return sign * _rhs_forward(current, settings)

    result = solve_ivp(
        ode,
        (0.0, duration),
        state,
        method="RK45",
        rtol=settings.rtol,
        atol=settings.atol,
        max_step=settings.max_step,
    )
    return result.y[:, -1], bool(result.success), str(result.message)


def recover_target(target: str, settings: AWSettings, epsilon: float = 2e-2, match_fraction: float = 0.5) -> dict:
    """Run a two-sided exact-trajectory recovery for one target."""
    invariance = fiber_invariance_norm(target)
    if invariance > 1e-8:
        return {
            "target": target,
            "classification": "not_invariant_under_fiber_action",
            "fiber_invariance_norm": invariance,
        }
    trajectory = homogeneous_trajectory(target, settings.lam)
    match_t = trajectory.interval_t * match_fraction
    if not epsilon < match_t < trajectory.interval_t - epsilon:
        raise ValueError("epsilon and match_fraction do not leave room for two-sided marching")

    left_initial = trajectory_state(trajectory, epsilon)
    right_initial = trajectory_state(trajectory, trajectory.interval_t - epsilon)
    expected = trajectory_state(trajectory, match_t)
    left_duration = (match_t - epsilon) * trajectory.normal_speed
    right_duration = (trajectory.interval_t - epsilon - match_t) * trajectory.normal_speed

    # With the chosen smooth normal orientation, increasing action-coordinate t
    # follows -rhs; marching inward from the right follows +rhs.
    left_final, left_success, left_message = _integrate(left_initial, left_duration, -1.0, settings)
    right_final, right_success, right_message = _integrate(right_initial, right_duration, 1.0, settings)
    left_error = float(np.linalg.norm(left_final - expected, ord=np.inf))
    right_error = float(np.linalg.norm(right_final - expected, ord=np.inf))
    match_error = float(np.linalg.norm(left_final - right_final, ord=np.inf))
    algebraic = float(
        np.linalg.norm(
            algebraic_residual(
                expected,
                settings.lam,
                settings.structure_scale,
                base_structure_scale=settings.base_structure_scale,
                fiber_structure_scale=settings.fiber_structure_scale,
            ),
            ord=np.inf,
        )
    )
    classification = "recovered_homogeneous" if left_success and right_success and match_error < 1e-6 and algebraic < 1e-8 else "failed"
    return {
        "target": target,
        "classification": classification,
        "fiber_invariance_norm": invariance,
        "epsilon": epsilon,
        "match_fraction": match_fraction,
        "match_t": match_t,
        "normal_speed": trajectory.normal_speed,
        "left_duration": left_duration,
        "right_duration": right_duration,
        "left_success": left_success,
        "right_success": right_success,
        "left_message": left_message,
        "right_message": right_message,
        "left_error": left_error,
        "right_error": right_error,
        "match_error": match_error,
        "algebraic_residual": algebraic,
        "expected_volume": omega_volume(omega_form(expected)),
    }


def _settings_from_args(args: argparse.Namespace) -> AWSettings:
    return AWSettings(
        lam=float(args.lam),
        structure_scale=None,
        base_structure_scale=float(args.base_structure_scale),
        fiber_structure_scale=float(args.fiber_structure_scale),
        max_tau=float(args.max_tau),
        max_step=float(args.max_step),
        rtol=float(args.rtol),
        atol=float(args.atol),
    )


def _parse_targets(value: str) -> tuple[str, ...]:
    if value == "all":
        return TARGET_NAMES
    targets = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [target for target in targets if target not in TARGET_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown target(s): {', '.join(unknown)}")
    return targets


def run_recovery(targets: tuple[str, ...], settings: AWSettings, epsilon: float, match_fraction: float) -> dict:
    """Run homogeneous recovery targets and return a JSON-ready payload."""
    results = [recover_target(target, settings, epsilon, match_fraction) for target in targets]
    return {
        "homogeneous_recovery_version": HOMOGENEOUS_RECOVERY_VERSION,
        "random_seed": RANDOM_SEED,
        "targets": list(targets),
        "settings": {
            "lambda": settings.lam,
            "base_structure_scale": settings.base_structure_scale,
            "fiber_structure_scale": settings.fiber_structure_scale,
            "rtol": settings.rtol,
            "atol": settings.atol,
            "max_step": settings.max_step,
        },
        "classification_counts": dict(Counter(result["classification"] for result in results)),
        "results": results,
    }


def _print_summary(payload: dict) -> None:
    print(f"classifications: {payload['classification_counts']}", flush=True)
    for result in payload["results"]:
        print()
        print(result["target"], flush=True)
        print(f"  classification: {result['classification']}", flush=True)
        print(f"  fiber invariance norm: {result['fiber_invariance_norm']}", flush=True)
        if "match_error" in result:
            print(f"  match error: {result['match_error']}", flush=True)
            print(f"  left/right exact errors: {result['left_error']} / {result['right_error']}", flush=True)
            print(f"  algebraic residual: {result['algebraic_residual']}", flush=True)


def main(argv: list[str] | None = None) -> None:
    """Run the direct homogeneous recovery check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=_parse_targets, default=TARGET_NAMES, help="all or comma-separated target list")
    parser.add_argument("--epsilon", type=float, default=2e-2)
    parser.add_argument("--match-fraction", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=4.0)
    parser.add_argument("--base-structure-scale", type=float, default=BASE_STRUCTURE_SCALE)
    parser.add_argument("--fiber-structure-scale", type=float, default=FIBER_STRUCTURE_SCALE)
    parser.add_argument("--max-tau", type=float, default=2.0)
    parser.add_argument("--max-step", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--no-write", action="store_true", help="print only, without writing output")
    args = parser.parse_args(argv)

    settings = _settings_from_args(args)
    payload = run_recovery(args.targets, settings, args.epsilon, args.match_fraction)
    _print_summary(payload)
    if args.no_write:
        return
    jsonl_path, summary_path = _output_paths(OUTPUT_DIR, OUTPUT_SUFFIX)
    payload = {**payload, "jsonl_path": str(jsonl_path), "summary_path": str(summary_path)}
    _write_jsonl_event(jsonl_path, _event("run_summary", payload))
    _write_summary(summary_path, payload)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
