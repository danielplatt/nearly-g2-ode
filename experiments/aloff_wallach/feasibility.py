"""Feasibility checks for Aloff-Wallach nearly-G2 searches.

The current numerical pipeline is a cohomogeneity-one endpoint matcher.  Generic
Aloff-Wallach spaces N_{k,l}=SU(3)/S^1_{k,l} have known homogeneous nearly
parallel G2 structures, but a homogeneous solution alone does not supply a
one-dimensional endpoint problem.  This module records which Aloff-Wallach
cases are immediately ruled out for the current workflow and why the exceptional
space N_{1,1} remains a plausible next target after deriving a new invariant
ODE system.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


ALOFF_WALLACH_FEASIBILITY_VERSION = "aloff-wallach-feasibility-v1"
ALOFF_WALLACH_DIMENSION = 7
CURRENT_Q_SYSTEM_PRINCIPAL_ORBIT = "SO(4)/Z_2^2"
CURRENT_Q_SYSTEM_PRINCIPAL_ORBIT_DIMENSION = 6


@dataclass(frozen=True)
class ActionCandidate:
    """One possible symmetry action for an Aloff-Wallach search."""

    topology: str
    group: str
    dimension: int
    generic_stabilizer_dimension: int | None
    cohomogeneity: int | None
    preserves_known_calibration: str
    current_q_system_ready: bool
    verdict: str


def generic_aloff_wallach_action_candidates() -> tuple[ActionCandidate, ...]:
    """Return standard connected homogeneous-calibration candidates for N_{k,l}.

    For generic Aloff-Wallach spaces the connected automorphism group preserving
    the standard homogeneous data is SU(3).  Its standard proper connected
    subgroups are too small to carry six-dimensional principal orbits.
    """
    return (
        ActionCandidate(
            topology="generic N_{k,l}",
            group="SU(3)",
            dimension=8,
            generic_stabilizer_dimension=1,
            cohomogeneity=0,
            preserves_known_calibration="yes",
            current_q_system_ready=False,
            verdict="transitive; no endpoint problem",
        ),
        ActionCandidate(
            topology="generic N_{k,l}",
            group="S(U(2)U(1))",
            dimension=4,
            generic_stabilizer_dimension=None,
            cohomogeneity=None,
            preserves_known_calibration="yes",
            current_q_system_ready=False,
            verdict="dimension too small for six-dimensional principal orbits",
        ),
        ActionCandidate(
            topology="generic N_{k,l}",
            group="SO(3)",
            dimension=3,
            generic_stabilizer_dimension=None,
            cohomogeneity=None,
            preserves_known_calibration="yes",
            current_q_system_ready=False,
            verdict="dimension too small for six-dimensional principal orbits",
        ),
        ActionCandidate(
            topology="generic N_{k,l}",
            group="T^2",
            dimension=2,
            generic_stabilizer_dimension=None,
            cohomogeneity=None,
            preserves_known_calibration="yes",
            current_q_system_ready=False,
            verdict="dimension too small for six-dimensional principal orbits",
        ),
    )


def cp2_real_so3_generic_orbit_dimension() -> int:
    """Return the generic orbit dimension of real SO(3) acting on CP^2.

    Write a complex line as [x + i y] with x,y in R^3.  After changing phase one
    may assume x and y are orthogonal.  The residual invariant
    |x|^2 - |y|^2 gives a one-dimensional orbit space on CP^2, whose real
    dimension is four.
    """
    cp2_dimension = 4
    cohomogeneity = 1
    return cp2_dimension - cohomogeneity


def n11_fiber_group_dimension() -> int:
    """Return the dimension of the K/H fiber group in N_{1,1}->CP^2."""
    k_dimension = 4
    h_dimension = 1
    return k_dimension - h_dimension


def n11_product_action_generic_orbit_dimension() -> int:
    """Return the generic orbit dimension of SO(3)_real x SO(3)_fiber."""
    return cp2_real_so3_generic_orbit_dimension() + n11_fiber_group_dimension()


def n11_product_action_cohomogeneity() -> int:
    """Return the cohomogeneity of the N_{1,1} product action."""
    return ALOFF_WALLACH_DIMENSION - n11_product_action_generic_orbit_dimension()


def n11_action_candidates() -> tuple[ActionCandidate, ...]:
    """Return special-action candidates for N_{1,1}.

    N_{1,1}=SU(3)/S^1_{1,1} has S^1_{1,1} central in K=S(U(2)U(1)).  Hence
    there is a right K/H ~= SO(3) fiber action over SU(3)/K ~= CP^2.  Combining
    it with the real SO(3)<SU(3) action on the base gives six-dimensional
    generic orbits, so this is a genuine cohomogeneity-one candidate.
    """
    return (
        ActionCandidate(
            topology="N_{1,1}",
            group="SU(3)",
            dimension=8,
            generic_stabilizer_dimension=1,
            cohomogeneity=0,
            preserves_known_calibration="yes",
            current_q_system_ready=False,
            verdict="transitive; no endpoint problem",
        ),
        ActionCandidate(
            topology="N_{1,1}",
            group="SO(3)_real x SO(3)_fiber",
            dimension=6,
            generic_stabilizer_dimension=0,
            cohomogeneity=n11_product_action_cohomogeneity(),
            preserves_known_calibration=(
                "plausible for the 3-Sasakian/Sasaki-Einstein homogeneous "
                "structure; strict homogeneous NP structures need a separate "
                "invariance check"
            ),
            current_q_system_ready=False,
            verdict="viable-new-ode-candidate",
        ),
    )


def generic_spaces_have_endpoint_volume_candidate() -> bool:
    """Return whether generic N_{k,l} has a candidate for this workflow."""
    return any(
        candidate.cohomogeneity == 1 and candidate.dimension >= CURRENT_Q_SYSTEM_PRINCIPAL_ORBIT_DIMENSION
        for candidate in generic_aloff_wallach_action_candidates()
    )


def n11_has_endpoint_volume_candidate() -> bool:
    """Return whether N_{1,1} has a cohomogeneity-one candidate action."""
    return any(candidate.cohomogeneity == 1 for candidate in n11_action_candidates())


def _candidate_payload(candidate: ActionCandidate) -> dict:
    """Return one action candidate in JSON-ready form."""
    return {
        "topology": candidate.topology,
        "group": candidate.group,
        "dimension": candidate.dimension,
        "generic_stabilizer_dimension": candidate.generic_stabilizer_dimension,
        "cohomogeneity": candidate.cohomogeneity,
        "preserves_known_calibration": candidate.preserves_known_calibration,
        "current_q_system_ready": candidate.current_q_system_ready,
        "verdict": candidate.verdict,
    }


def build_summary() -> dict:
    """Return a JSON-ready Aloff-Wallach feasibility summary."""
    return {
        "version": ALOFF_WALLACH_FEASIBILITY_VERSION,
        "known_homogeneous_structures": {
            "generic_Nkl": (
                "homogeneous nearly-parallel G2 structures on "
                "N_{k,l}=SU(3)/S^1_{k,l}"
            ),
            "N11": (
                "special Aloff-Wallach space with 3-Sasakian/Sasaki-Einstein "
                "geometry and additional homogeneous nearly-parallel structures"
            ),
            "literature": [
                "Ball-Oliveira, Gauge theory on Aloff-Wallach spaces, arXiv:1610.04557",
                "Aleshin, The bar-nu invariant of G2-structures on Aloff-Wallach spaces, arXiv:2604.04605",
            ],
        },
        "generic_Nkl": {
            "endpoint_volume_candidate": generic_spaces_have_endpoint_volume_candidate(),
            "action_candidates": [
                _candidate_payload(candidate) for candidate in generic_aloff_wallach_action_candidates()
            ],
            "verdict": (
                "not-ready: the calibrated homogeneous SU(3) action is "
                "transitive, and standard proper connected subgroups are too "
                "small for a seven-dimensional cohomogeneity-one search"
            ),
        },
        "N11": {
            "fibration": "SO(3) -> N_{1,1}=SU(3)/S^1_{1,1} -> CP^2",
            "base_action": "SO(3)_real on CP^2 has cohomogeneity one",
            "fiber_group_dimension": n11_fiber_group_dimension(),
            "product_action_generic_orbit_dimension": n11_product_action_generic_orbit_dimension(),
            "product_action_cohomogeneity": n11_product_action_cohomogeneity(),
            "endpoint_volume_candidate": n11_has_endpoint_volume_candidate(),
            "action_candidates": [_candidate_payload(candidate) for candidate in n11_action_candidates()],
            "verdict": (
                "promising-but-not-current-q-system: N_{1,1} has a plausible "
                "SO(3)_real x SO(3)_fiber cohomogeneity-one action, but it "
                "requires a new invariant-form basis, endpoint charts, and ODE"
            ),
        },
        "current_q_system": {
            "principal_orbit": CURRENT_Q_SYSTEM_PRINCIPAL_ORBIT,
            "principal_orbit_dimension": CURRENT_Q_SYSTEM_PRINCIPAL_ORBIT_DIMENSION,
            "aloff_wallach_ready": False,
            "reason": (
                "the current q_i equations are specialized to SO(4)/Z_2^2 "
                "principal orbits, not the N_{1,1} product-action principal orbits"
            ),
        },
        "recommended_next_step": (
            "derive the SO(3)_real x SO(3)_fiber invariant forms and singular "
            "orbit endpoint conditions on N_{1,1}, then verify a known "
            "homogeneous structure before scouting"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Print the Aloff-Wallach feasibility summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)
    summary = build_summary()
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return
    print("Aloff-Wallach feasibility", flush=True)
    print(f"version: {summary['version']}", flush=True)
    print(f"generic N_{{k,l}} endpoint-volume candidate: {summary['generic_Nkl']['endpoint_volume_candidate']}", flush=True)
    print(f"N_{{1,1}} endpoint-volume candidate: {summary['N11']['endpoint_volume_candidate']}", flush=True)
    print(
        "N_{1,1} product-action cohomogeneity: "
        f"{summary['N11']['product_action_cohomogeneity']}",
        flush=True,
    )
    print(f"current q-system ready: {summary['current_q_system']['aloff_wallach_ready']}", flush=True)
    print(f"recommended next step: {summary['recommended_next_step']}", flush=True)


if __name__ == "__main__":
    main()

