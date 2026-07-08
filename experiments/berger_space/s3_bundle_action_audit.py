"""Audit the Grove-Ziller S3-bundle route to the Berger topology.

The point of this module is deliberately narrow. It checks the topological
identification of the Berger space with an S3-bundle over S4, then verifies
whether the Grove-Ziller bundle action supplies a new seven-dimensional
cohomogeneity-one endpoint problem. The answer recorded here is no: the
Grove-Ziller cohomogeneity-one action lives on the ten-dimensional principal
SO(4) bundle; the induced action on the associated S3-bundle is only an SO(3)
action with three-dimensional generic orbits.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


AUDIT_VERSION = "berger-s3-bundle-action-audit-v1"


@dataclass(frozen=True)
class CrowleyEscherBundle:
    """Crowley-Escher ``M_{m,n}`` sphere-bundle labels."""

    m: int
    n: int

    @property
    def euler_class(self) -> int:
        return self.n

    @property
    def tangent_p1(self) -> int:
        return 2 * (self.n + 2 * self.m)

    @property
    def tangent_p1_mod_euler(self) -> int | None:
        if self.n == 0:
            return None
        return self.tangent_p1 % abs(self.n)


@dataclass(frozen=True)
class GroveZillerBundle:
    """Grove-Ziller ``M_{k,l}`` sphere-bundle labels."""

    k: int
    l: int

    @property
    def euler_class(self) -> int:
        return self.k + self.l

    @property
    def tangent_p1_representative(self) -> int:
        return 4 * self.l

    @property
    def tangent_p1_mod_euler(self) -> int | None:
        euler = self.euler_class
        if euler == 0:
            return None
        return self.tangent_p1_representative % abs(euler)


@dataclass(frozen=True)
class GroveZillerSlopes:
    """Integer slopes for the Grove-Ziller principal S3 x S3 bundle diagram."""

    p_minus: int
    p_plus: int
    q_minus: int
    q_plus: int

    @property
    def k(self) -> int:
        return (self.p_minus**2 - self.p_plus**2) // 8

    @property
    def l(self) -> int:
        return -(self.q_minus**2 - self.q_plus**2) // 8

    def all_congruent_one_mod_four(self) -> bool:
        return all(value % 4 == 1 for value in (self.p_minus, self.p_plus, self.q_minus, self.q_plus))


def berger_crowley_escher_model() -> CrowleyEscherBundle:
    """Return the orientation convention used for the Berger S3-bundle."""
    return CrowleyEscherBundle(m=-1, n=10)


def crowley_escher_to_grove_ziller(bundle: CrowleyEscherBundle) -> GroveZillerBundle:
    """Return a Grove-Ziller label with matching Euler class and tangent p1.

    Grove-Ziller record ``e = k + l`` and ``p1(TM_{k,l}) = +/- 4l`` modulo
    the Euler class. For the Berger orientation used here, choosing the plus
    sign gives ``l = (n + 2m) / 2`` and ``k = n - l``.
    """
    numerator = bundle.n + 2 * bundle.m
    if numerator % 2:
        raise ValueError("this convention needs n + 2m even")
    l_value = numerator // 2
    return GroveZillerBundle(k=bundle.n - l_value, l=l_value)


def _slope_pair_for_class(target: int, search_radius: int = 80) -> tuple[int, int]:
    """Find small slopes a,b == 1 mod 4 with ``(a^2-b^2)/8 = target``."""
    if target % 2 == 0:
        return 2 * target + 1, -2 * target + 1
    if target % 4 == 1:
        return -target - 2, -target + 2
    if target % 4 == 3:
        return target + 2, target - 2
    candidates = [value for value in range(-search_radius, search_radius + 1) if value % 4 == 1]
    matches: list[tuple[int, int]] = []
    for left in candidates:
        for right in candidates:
            if left * left - right * right == 8 * target:
                matches.append((left, right))
    if not matches:
        raise ValueError(f"no slope pair found for {target} within +/-{search_radius}")
    return min(matches, key=lambda pair: (max(abs(pair[0]), abs(pair[1])), abs(pair[0]) + abs(pair[1]), pair))


def slopes_for_grove_ziller_bundle(bundle: GroveZillerBundle) -> GroveZillerSlopes:
    """Return deterministic small slopes for the Grove-Ziller diagram."""
    p_minus, p_plus = _slope_pair_for_class(bundle.k)
    q_minus, q_plus = _slope_pair_for_class(-bundle.l)
    return GroveZillerSlopes(p_minus=p_minus, p_plus=p_plus, q_minus=q_minus, q_plus=q_plus)


def induced_so3_orbit_types(slopes: GroveZillerSlopes) -> tuple[str, ...]:
    """Return the finite orbit types from Grove-Ziller Theorem 4.1."""

    def label(value: int) -> str:
        if value == 0:
            return "D0(SO2/O2)"
        if value == 1:
            return "D1=Z2"
        return f"D{value}"

    values = (
        abs(slopes.p_minus + slopes.q_minus) // 2,
        abs(slopes.p_minus - slopes.q_minus) // 2,
        abs(slopes.p_plus + slopes.q_plus) // 2,
        abs(slopes.p_plus - slopes.q_plus) // 2,
    )
    return ("1", "Z2", "D2") + tuple(label(value) for value in values)


def build_summary() -> dict:
    """Return a JSON-ready audit summary."""
    ce_model = berger_crowley_escher_model()
    gz_model = crowley_escher_to_grove_ziller(ce_model)
    slopes = slopes_for_grove_ziller_bundle(gz_model)
    topology_matches = (
        ce_model.euler_class == gz_model.euler_class == 10
        and ce_model.tangent_p1_mod_euler == gz_model.tangent_p1_mod_euler == 6
        and slopes.k == gz_model.k
        and slopes.l == gz_model.l
        and slopes.all_congruent_one_mod_four()
    )
    return {
        "version": AUDIT_VERSION,
        "topology": {
            "crowley_escher_model": {
                "label": f"M_{{{ce_model.m},{ce_model.n}}}",
                "euler_class": ce_model.euler_class,
                "tangent_p1": ce_model.tangent_p1,
                "tangent_p1_mod_euler": ce_model.tangent_p1_mod_euler,
            },
            "grove_ziller_model": {
                "label": f"M_{{{gz_model.k},{gz_model.l}}}",
                "euler_class": gz_model.euler_class,
                "tangent_p1_representative": gz_model.tangent_p1_representative,
                "tangent_p1_mod_euler": gz_model.tangent_p1_mod_euler,
            },
            "topology_matches_berger": topology_matches,
            "note": "matches H4=Z_10 and p1(TM)=6 mod 10 in the Berger orientation convention",
        },
        "grove_ziller_slopes": {
            "p_minus": slopes.p_minus,
            "p_plus": slopes.p_plus,
            "q_minus": slopes.q_minus,
            "q_plus": slopes.q_plus,
            "k": slopes.k,
            "l": slopes.l,
            "all_congruent_one_mod_four": slopes.all_congruent_one_mod_four(),
        },
        "principal_so4_bundle_action": {
            "space": "principal SO(4) bundle P_{6,4}",
            "dimension": 10,
            "effective_group": "SO(4) x SO(3)",
            "effective_group_dimension": 9,
            "principal_orbit_dimension": 9,
            "cohomogeneity": 1,
            "singular_orbit_codimension": 2,
            "g2_target": False,
            "reason": "correct cohomogeneity but wrong total dimension for a 7D G2 ansatz",
        },
        "associated_s3_bundle_action": {
            "space": "associated S3-bundle M_{6,4}, diffeomorphic to Berger",
            "dimension": 7,
            "induced_group": "SO(3)",
            "induced_group_dimension": 3,
            "generic_stabilizer_dimension": 0,
            "principal_orbit_dimension": 3,
            "cohomogeneity": 4,
            "orbit_types": induced_so3_orbit_types(slopes),
            "g2_endpoint_problem": False,
            "reason": "the induced action is almost free on the 7D sphere bundle, not cohomogeneity one",
        },
        "known_berger_action": {
            "space": "SO(5)/SO(3)_irr",
            "group": "SO(4) < SO(5)",
            "principal_orbit": "SO(4)/Z_2^2",
            "dimension": 7,
            "principal_orbit_dimension": 6,
            "cohomogeneity": 1,
            "singular_orbit_codimension": 2,
            "status": "the already-scouted Berger action",
        },
        "endpoint_smoothness": {
            "status": "not_applicable_to_new_7d_action",
            "principal_so4_bundle": "codimension-2 endpoint charts exist in dimension 10, not for G2",
            "associated_s3_bundle": "principal orbits are 3D, so there are no 6D hypersurface endpoints",
        },
        "literature": [
            {
                "source": "Goette-Kitchloo-Shankar, Diffeomorphism type of the Berger space",
                "url": "https://arxiv.org/abs/math/0204352",
                "finding": "Berger is diffeomorphic to an S3-bundle over S4 with Euler class 10 and p1 representative 16.",
            },
            {
                "source": "Crowley-Escher, A classification of S3-bundles over S4",
                "url": "https://arxiv.org/abs/math/0004147",
                "finding": "Their M_{m,n} notation gives H4=Z_n and p1=2(n+2m); M_{-1,10} has p1=16 = 6 mod 10.",
            },
            {
                "source": "Grove-Ziller, Curvature and symmetry of Milnor spheres",
                "url": "https://arxiv.org/abs/math/0007198",
                "finding": "Principal SO(4) bundles over S4 have cohomogeneity-one SO(4)xSO(3) structures; associated S3-bundles inherit SO(3) actions and nonnegative-curvature metrics.",
            },
        ],
        "literature_g2_status": (
            "no calibration found in this audit: the cited bundle literature gives nonnegative "
            "curvature and positive-Ricci/connection-metric context, but no nearly, closed, "
            "or coclosed G2 solution attached to this Grove-Ziller associated-bundle action"
        ),
        "verdict": "topology-matches-but-no-new-7d-cohomogeneity-one-action-from-this-route",
    }


def main(argv: list[str] | None = None) -> None:
    """Run the Berger S3-bundle action audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)
    summary = build_summary()
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return
    topology = summary["topology"]
    associated = summary["associated_s3_bundle_action"]
    principal = summary["principal_so4_bundle_action"]
    print("Berger S3-bundle action audit", flush=True)
    print(f"version: {summary['version']}", flush=True)
    print(
        "topology: "
        f"{topology['crowley_escher_model']['label']} -> {topology['grove_ziller_model']['label']}",
        flush=True,
    )
    print(f"topology matches Berger invariants: {topology['topology_matches_berger']}", flush=True)
    print(
        "Grove-Ziller slopes: "
        f"p=({summary['grove_ziller_slopes']['p_minus']}, {summary['grove_ziller_slopes']['p_plus']}), "
        f"q=({summary['grove_ziller_slopes']['q_minus']}, {summary['grove_ziller_slopes']['q_plus']})",
        flush=True,
    )
    print(
        "principal SO4-bundle action: "
        f"dimension {principal['dimension']}, cohomogeneity {principal['cohomogeneity']}",
        flush=True,
    )
    print(
        "associated Berger-sized action: "
        f"dimension {associated['dimension']}, principal orbit dimension "
        f"{associated['principal_orbit_dimension']}, cohomogeneity {associated['cohomogeneity']}",
        flush=True,
    )
    print(f"endpoint smoothness status: {summary['endpoint_smoothness']['status']}", flush=True)
    print(f"verdict: {summary['verdict']}", flush=True)


if __name__ == "__main__":
    main()
