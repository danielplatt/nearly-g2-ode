"""Feasibility checks for a Stiefel V_{5,2} nearly-G2 search.

The current numerical marcher is a cohomogeneity-one SO(4)-invariant q-system
with six-dimensional principal orbits.  The standard Stiefel manifold
V_{5,2}=SO(5)/SO(3) is homogeneous, but the natural SO(4) subgroup has generic
five-dimensional orbits on it.  This module records that obstruction and the
known homogeneous nearly-parallel G2 algebraic calibration data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from mpmath import mp


STIEFEL_FEASIBILITY_VERSION = "stiefel-feasibility-v1"


@dataclass(frozen=True)
class HomogeneousStiefelG2Parameters:
    """One SO(5)-invariant G2 parameter point in Moreno-Portilla notation."""

    a: mp.mpf
    b: mp.mpf
    x: mp.mpf
    y: mp.mpf
    z: mp.mpf


@dataclass(frozen=True)
class StiefelActionCandidate:
    """One possible symmetry group for a calibrated Stiefel search."""

    group: str
    dimension: int
    generic_stabilizer_dimension: int | None
    cohomogeneity: int | None
    verdict: str


def homogeneous_nearly_parallel_parameters(theta: mp.mpf = mp.zero) -> HomogeneousStiefelG2Parameters:
    """Return one known homogeneous nearly-parallel Stiefel G2 parameter point.

    Moreno-Portilla's invariant Stiefel family is nearly parallel for
    x=a, y=b, a^2+b^2=27/512, and z=-9/32.  The angle theta parametrizes the
    harmless S^1-family in the homogeneous invariant form.
    """
    radius = mp.sqrt(mp.mpf(27) / 512)
    a = radius * mp.cos(theta)
    b = radius * mp.sin(theta)
    return HomogeneousStiefelG2Parameters(a=a, b=b, x=a, y=b, z=-mp.mpf(9) / 32)


def homogeneous_np_defects(params: HomogeneousStiefelG2Parameters) -> dict[str, mp.mpf]:
    """Return exact-zero defects for the homogeneous Stiefel nearly-parallel conditions."""
    return {
        "x_minus_a": params.x - params.a,
        "y_minus_b": params.y - params.b,
        "circle": params.a**2 + params.b**2 - mp.mpf(27) / 512,
        "z_plus_9_over_32": params.z + mp.mpf(9) / 32,
    }


def generic_so4_stiefel_orbit_dimension() -> int:
    """Return the generic SO(4)-orbit dimension on V_{5,2}.

    View V_{5,2} as oriented orthonormal pairs (x,y) in R^5 and SO(4) as the
    subgroup fixing the fifth coordinate.  The two fifth-coordinate components
    (x_5,y_5) are independent invariants on the open disk x_5^2+y_5^2<1.
    A generic pair of R^4 projections spans a two-plane, so the stabilizer is
    the SO(2) rotating its orthogonal complement.
    """
    so4_dimension = 6
    generic_stabilizer_dimension = 1
    return so4_dimension - generic_stabilizer_dimension


def natural_so4_stiefel_cohomogeneity() -> int:
    """Return the cohomogeneity of the natural SO(4)-action on V_{5,2}."""
    stiefel_dimension = 7
    return stiefel_dimension - generic_so4_stiefel_orbit_dimension()


def current_q_system_principal_orbit_dimension() -> int:
    """Return the principal-orbit dimension assumed by the current q-system."""
    so4_dimension = 6
    finite_principal_isotropy_dimension = 0
    return so4_dimension - finite_principal_isotropy_dimension


def current_q_system_is_stiefel_ready() -> bool:
    """Return whether the current cohomogeneity-one q-system can scout Stiefel."""
    return natural_so4_stiefel_cohomogeneity() == 1


def action_candidates_preserving_homogeneous_stiefel_geometry() -> tuple[StiefelActionCandidate, ...]:
    """Return standard connected action candidates for homogeneous Stiefel calibration.

    The known homogeneous Stiefel nearly-parallel structure is SO(5)-invariant.
    A calibrated cohomogeneity-one search using that known target should use a
    subgroup of the homogeneous automorphism group.  Among the standard connected
    proper subgroups of SO(5), SO(4) is the only one large enough to have
    six-dimensional generic orbits; its natural action on V_{5,2} has an SO(2)
    stabilizer and hence cohomogeneity two.
    """
    return (
        StiefelActionCandidate("SO(5)", 10, 3, 0, "transitive; no endpoint problem"),
        StiefelActionCandidate(
            "SO(4) fixing a line",
            6,
            1,
            natural_so4_stiefel_cohomogeneity(),
            "cohomogeneity two; not compatible with one-dimensional matching",
        ),
        StiefelActionCandidate("U(2)", 4, None, None, "dimension too small for six-dimensional principal orbits"),
        StiefelActionCandidate(
            "irreducible SO(3)",
            3,
            None,
            None,
            "dimension too small for six-dimensional principal orbits",
        ),
    )


def has_known_cohomogeneity_one_calibration_action() -> bool:
    """Return whether the standard candidate list contains a calibrated coh1 action."""
    return any(candidate.cohomogeneity == 1 for candidate in action_candidates_preserving_homogeneous_stiefel_geometry())


def _mp_string(value: mp.mpf) -> str:
    """Return a stable string for one mp scalar."""
    return mp.nstr(value, 80)


def build_summary() -> dict:
    """Return a JSON-ready Stiefel feasibility summary."""
    with mp.workdps(80):
        homogeneous = homogeneous_nearly_parallel_parameters()
        defects = homogeneous_np_defects(homogeneous)
        max_defect = max(abs(value) for value in defects.values())
    return {
        "version": STIEFEL_FEASIBILITY_VERSION,
        "topology": "V_{5,2}=SO(5)/SO(3)_standard",
        "homogeneous_calibration": {
            "source_notation": "Moreno-Portilla (a,b,x,y,z)",
            "parameters": {
                "a": _mp_string(homogeneous.a),
                "b": _mp_string(homogeneous.b),
                "x": _mp_string(homogeneous.x),
                "y": _mp_string(homogeneous.y),
                "z": _mp_string(homogeneous.z),
            },
            "nearly_parallel_defects": {key: _mp_string(value) for key, value in defects.items()},
            "max_abs_defect": _mp_string(max_defect),
        },
        "natural_so4_action": {
            "generic_orbit_dimension": generic_so4_stiefel_orbit_dimension(),
            "cohomogeneity": natural_so4_stiefel_cohomogeneity(),
            "independent_invariants": ["x_5", "y_5"],
        },
        "candidate_calibration_actions": [
            {
                "group": candidate.group,
                "dimension": candidate.dimension,
                "generic_stabilizer_dimension": candidate.generic_stabilizer_dimension,
                "cohomogeneity": candidate.cohomogeneity,
                "verdict": candidate.verdict,
            }
            for candidate in action_candidates_preserving_homogeneous_stiefel_geometry()
        ],
        "has_known_cohomogeneity_one_calibration_action": has_known_cohomogeneity_one_calibration_action(),
        "current_q_system": {
            "principal_orbit": "SO(4)/Z_2^2",
            "principal_orbit_dimension": current_q_system_principal_orbit_dimension(),
            "cohomogeneity": 1,
            "stiefel_ready": current_q_system_is_stiefel_ready(),
        },
        "verdict": (
            "not-ready: standard Stiefel V_{5,2} is not a calibrated endpoint "
            "problem for the current SO(4)/Z_2^2 cohomogeneity-one q-system"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Print the Stiefel feasibility summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)
    summary = build_summary()
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return
    print("Stiefel V_{5,2} feasibility", flush=True)
    print(f"version: {summary['version']}", flush=True)
    print(f"homogeneous max defect: {summary['homogeneous_calibration']['max_abs_defect']}", flush=True)
    print(
        "natural SO(4) action cohomogeneity: "
        f"{summary['natural_so4_action']['cohomogeneity']}",
        flush=True,
    )
    print(
        "current q-system principal orbit dimension: "
        f"{summary['current_q_system']['principal_orbit_dimension']}",
        flush=True,
    )
    print(f"verdict: {summary['verdict']}", flush=True)


if __name__ == "__main__":
    main()
