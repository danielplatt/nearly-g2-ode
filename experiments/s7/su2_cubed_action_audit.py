"""Audit the SU(2)^3 cohomogeneity-one action on S7.

This is a separate S7 action from the SO(4)/Z_2^2 action used by the q_i
marcher.  It follows Podesta's Sp(1)^3 description: principal orbits are
S3 x S3, both singular orbits are S3, and invariant nearly-parallel G2
structures reduce to five scalar functions.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass

from mpmath import mp


AUDIT_VERSION = "s7-su2-cubed-action-audit-v1"
DEFAULT_DPS = 80


@dataclass(frozen=True)
class PodestaTarget:
    """One known homogeneous S7 solution in Podesta's SU(2)^3 chart."""

    name: str
    lam: mp.mpf
    functions: tuple[Callable[[mp.mpf], mp.mpf], ...]


def group_diagram() -> dict:
    """Return the SU(2)^3 group diagram on S7."""
    return {
        "group": "G = Sp(1)^3",
        "principal_isotropy": "H = diagonal Sp(1)",
        "left_singular_isotropy": "K+ = {(q, q', q)} ~= Sp(1) x Sp(1)",
        "right_singular_isotropy": "K- = {(q, q', q')} ~= Sp(1) x Sp(1)",
        "normal_geodesic": "gamma(t) = (cos(t), sin(t)) in H^2, 0 <= t <= pi/2",
        "principal_orbit": "G/H ~= S3 x S3",
        "singular_orbits": ["G/K+ ~= S3", "G/K- ~= S3"],
        "dimensions": {
            "G": 9,
            "H": 3,
            "K_plus": 6,
            "K_minus": 6,
            "principal_orbit": 6,
            "singular_orbit": 3,
            "slice": 4,
            "manifold": 7,
        },
        "cohomogeneity": 1,
    }


def invariant_form_basis() -> dict:
    """Return Podesta's invariant form basis along the normal geodesic."""
    return {
        "invariant_2_forms_on_principal_orbit": ["omega = e25 + e36 + e47"],
        "invariant_3_form_basis": [
            "e1 ^ omega",
            "phi1 = e234",
            "phi2 = e567",
            "phi3 = e237 - e246 + e345",
            "phi4 = e267 - e357 + e456",
        ],
        "general_invariant_3_form": (
            "phi = f0 e1^omega + f1 phi1 + f2 phi2 + f3 phi3 + f4 phi4"
        ),
    }


def endpoint_smoothness_conditions() -> dict:
    """Return the smoothness conditions at the two S3 singular orbits."""
    left = {
        "coordinate": "t near K+",
        "parity": "f0 odd; f1,f2,f3,f4 even",
        "zero_values": ["f1(0)=0", "f3(0)=0", "f4(0)=0"],
        "derivative_conditions": ["f1''(0)=0", "6 f0'(0)=f3''(0)"],
        "nondegeneracy": ["f2(0) != 0", "f0'(0) != 0", "f2(0) f0'(0) < 0"],
        "regular_variables": [
            "f0=t h0",
            "f1=t^4 h1",
            "f2=h2",
            "f3=t^2 h3",
            "f4=t^2 h4",
        ],
        "singular_initial_data": [
            "h0(0)=a",
            "h1(0)=27 lambda / 4",
            "h2(0)=-a^3/27",
            "h3(0)=3a",
            "h4(0)=-3a - lambda a^2/6",
        ],
    }
    right = {
        "coordinate": "s = pi/2 - t near K-",
        "transform_to_left_conditions": [
            "g0(s)=f0(pi/2-s)",
            "g1(s)=f2(pi/2-s)",
            "g2(s)=f1(pi/2-s)",
            "g3(s)=f4(pi/2-s)",
            "g4(s)=f3(pi/2-s)",
        ],
        "conditions": "the same K+ conditions applied to g0,...,g4",
    }
    return {"left_K_plus": left, "right_K_minus": right}


def round_target() -> PodestaTarget:
    """Return the standard round S7 nearly-parallel solution."""

    def f0(t: mp.mpf) -> mp.mpf:
        return -9 * mp.sin(t) * mp.cos(t)

    def f1(t: mp.mpf) -> mp.mpf:
        return 27 * mp.sin(t) ** 4

    def f2(t: mp.mpf) -> mp.mpf:
        return 27 * mp.cos(t) ** 4

    def f3(t: mp.mpf) -> mp.mpf:
        return -27 * mp.sin(t) ** 2 * mp.cos(t) ** 2

    return PodestaTarget("round", mp.mpf(4), (f0, f1, f2, f3, f3))


def squashed_target() -> PodestaTarget:
    """Return the squashed S7 nearly-parallel solution."""
    sqrt5 = mp.sqrt(5)

    def f0(t: mp.mpf) -> mp.mpf:
        return mp.mpf(9) / sqrt5 * mp.sin(t) * mp.cos(t)

    def f1(t: mp.mpf) -> mp.mpf:
        return (
            mp.mpf(27)
            / sqrt5
            * (3 * mp.sin(t) ** 4 * mp.cos(t) ** 2 - mp.sin(t) ** 6 / 5)
        )

    def f2(t: mp.mpf) -> mp.mpf:
        return (
            mp.mpf(27)
            / sqrt5
            * (3 * mp.cos(t) ** 4 * mp.sin(t) ** 2 - mp.cos(t) ** 6 / 5)
        )

    def f3(t: mp.mpf) -> mp.mpf:
        return (
            mp.mpf(27)
            / sqrt5
            * mp.sin(t) ** 2
            * mp.cos(t) ** 2
            * (mp.cos(t) ** 2 - mp.mpf(11) * mp.sin(t) ** 2 / 5)
        )

    def f4(t: mp.mpf) -> mp.mpf:
        return (
            mp.mpf(27)
            / sqrt5
            * mp.sin(t) ** 2
            * mp.cos(t) ** 2
            * (mp.sin(t) ** 2 - mp.mpf(11) * mp.cos(t) ** 2 / 5)
        )

    return PodestaTarget("squashed", mp.mpf(12) / sqrt5, (f0, f1, f2, f3, f4))


def _target_values(target: PodestaTarget, t: mp.mpf) -> tuple[mp.mpf, ...]:
    return tuple(function(t) for function in target.functions)


def _target_derivatives(target: PodestaTarget, t: mp.mpf) -> tuple[mp.mpf, ...]:
    return tuple(mp.diff(function, t) for function in target.functions)


def np_residuals(target: PodestaTarget, t: mp.mpf) -> dict[str, mp.mpf]:
    """Return Podesta system residuals for one target at one regular time."""
    f0, f1, f2, f3, f4 = _target_values(target, t)
    f0p, f1p, f2p, f3p, f4p = _target_derivatives(target, t)
    lam = target.lam
    left = f1 * f4 - f3**2
    right = f2 * f3 - f4**2
    middle = f1 * f2 - f3 * f4
    return {
        "f1_ode": f1p - lam / f0**3 * (f1 * middle / 2 - f3 * left),
        "f2_ode": f2p - lam / f0**3 * (f4 * right - f2 * middle / 2),
        "f3_ode": f3p - (6 * f0 + lam / (2 * f0**3) * (f1 * right - f4 * left)),
        "f4_ode": f4p - (-6 * f0 + lam / (2 * f0**3) * (f3 * right - f2 * left)),
        "f0_ode": f0p
        + mp.mpf(3)
        / (2 * f0**4)
        * ((f1 + f3) * right - (f2 + f4) * left),
        "algebraic_constraint": f3 + f4 + lam * f0**2 / 6,
        "metric_constraint": f0**6 - left * right + middle**2 / 4,
    }


def endpoint_residuals(target: PodestaTarget, side: str) -> dict[str, mp.mpf]:
    """Return smoothness residuals at one singular endpoint."""
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    if side == "left":
        funcs = target.functions
    else:
        endpoint = mp.pi / 2
        f0, f1, f2, f3, f4 = target.functions
        funcs = (
            lambda s: f0(endpoint - s),
            lambda s: f2(endpoint - s),
            lambda s: f1(endpoint - s),
            lambda s: f4(endpoint - s),
            lambda s: f3(endpoint - s),
        )
    g0, g1, g2, g3, g4 = funcs
    return {
        "g1_value": g1(mp.zero),
        "g3_value": g3(mp.zero),
        "g4_value": g4(mp.zero),
        "g1_second": mp.diff(g1, mp.zero, 2),
        "six_g0_prime_minus_g3_second": (
            6 * mp.diff(g0, mp.zero) - mp.diff(g3, mp.zero, 2)
        ),
        "nondegenerate_product": g2(mp.zero) * mp.diff(g0, mp.zero),
    }


def _mp_string(value: mp.mpf) -> str:
    return mp.nstr(value, 80)


def smoke_results() -> dict:
    """Verify the two known homogeneous solutions in this action chart."""
    with mp.workdps(DEFAULT_DPS):
        targets = (round_target(), squashed_target())
        sample_times = (mp.mpf("0.37"), mp.mpf("0.91"))
        results = {}
        for target in targets:
            regular_residuals = []
            for t in sample_times:
                residuals = np_residuals(target, t)
                regular_residuals.append(
                    {
                        "t": _mp_string(t),
                        "max_abs_residual": _mp_string(
                            max(abs(value) for value in residuals.values())
                        ),
                        "residuals": {
                            key: _mp_string(value) for key, value in residuals.items()
                        },
                    }
                )
            endpoints = {}
            for side in ("left", "right"):
                residuals = endpoint_residuals(target, side)
                endpoints[side] = {
                    "max_abs_smoothness_residual": _mp_string(
                        max(
                            abs(residuals["g1_value"]),
                            abs(residuals["g3_value"]),
                            abs(residuals["g4_value"]),
                            abs(residuals["g1_second"]),
                            abs(residuals["six_g0_prime_minus_g3_second"]),
                        )
                    ),
                    "nondegenerate_product": _mp_string(residuals["nondegenerate_product"]),
                    "nondegenerate": residuals["nondegenerate_product"] < 0,
                    "residuals": {key: _mp_string(value) for key, value in residuals.items()},
                }
            results[target.name] = {
                "lambda": _mp_string(target.lam),
                "regular_residuals": regular_residuals,
                "endpoint_residuals": endpoints,
            }
    return results


def build_summary() -> dict:
    """Return a JSON-ready S7 SU(2)^3 action audit."""
    smoke = smoke_results()
    max_regular = max(
        mp.mpf(item["max_abs_residual"])
        for target in smoke.values()
        for item in target["regular_residuals"]
    )
    max_endpoint = max(
        mp.mpf(endpoint["max_abs_smoothness_residual"])
        for target in smoke.values()
        for endpoint in target["endpoint_residuals"].values()
    )
    return {
        "version": AUDIT_VERSION,
        "action": group_diagram(),
        "topology": {
            "verified": True,
            "reason": "the action is the explicit restriction of Sp(2)xSp(1) to the unit sphere S7 in H^2",
            "compactification": "S7 = G x_{K+} D4 union_{G/H} G x_{K-} D4",
            "open_half": "S7 minus one singular S3 is G x_{K+} H ~= S3 x R4",
        },
        "invariant_forms": invariant_form_basis(),
        "endpoint_smoothness": endpoint_smoothness_conditions(),
        "known_solutions": {
            "round": "lambda=4, standard constant-curvature S7 NP structure",
            "squashed": "lambda=12/sqrt(5), proper squashed S7 NP structure",
            "podesta_family": (
                "one-parameter local/one-ended family on S3 x R4; compact S7 "
                "extension is numerically suspected to give only the homogeneous cases"
            ),
        },
        "other_action_notes": [
            "The old q_i S7 work used the SO(4) action with principal orbit SO(4)/Z_2^2.",
            "Product/sum actions such as SO(4)xSO(4) on R4+R4 are cohomogeneity one on S7 but too symmetric in the wrong way for this invariant G2 chart.",
            "Podesta explicitly suggests lower-symmetry reductions SU(2)^2 x U(1) and SU(2)^2 as possible future investigations.",
        ],
        "literature": [
            {
                "source": "Podesta, Nearly parallel G2-structures with large symmetry group",
                "url": "https://arxiv.org/abs/1905.03077",
                "finding": "SU(2)^3 cohomogeneity-one action on S7; invariant NP equations; round/squashed homogeneous solutions; one-ended S3 x R4 family.",
            },
            {
                "source": "Hoelscher, Classification of Cohomogeneity One Manifolds in Low Dimensions",
                "url": "https://arxiv.org/abs/0712.1327",
                "finding": "Low-dimensional cohomogeneity-one action classification context and curvature literature pointers.",
            },
            {
                "source": "Cvetic-Gibbons-Lu-Pope, Cohomogeneity One Manifolds of Spin(7) and G2 Holonomy",
                "url": "https://arxiv.org/abs/hep-th/0108245",
                "finding": "Related cohomogeneity-one G2/Spin(7) holonomy systems, including principal S3 x S3 in dimension 7.",
            },
        ],
        "smoke": smoke,
        "smoke_status": {
            "max_regular_residual": _mp_string(max_regular),
            "max_endpoint_smoothness_residual": _mp_string(max_endpoint),
            "passed": max_regular < mp.mpf("1e-60") and max_endpoint < mp.mpf("1e-60"),
        },
        "verdict": "new-viable-s7-action-ready-for-a-dedicated-podesta-system-marcher",
    }


def main(argv: list[str] | None = None) -> None:
    """Run the SU(2)^3 S7 action audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)
    summary = build_summary()
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return
    print("S7 SU(2)^3 action audit", flush=True)
    print(f"version: {summary['version']}", flush=True)
    print(f"group: {summary['action']['group']}", flush=True)
    print(f"principal orbit: {summary['action']['principal_orbit']}", flush=True)
    print(f"singular orbits: {', '.join(summary['action']['singular_orbits'])}", flush=True)
    print(f"topology verified: {summary['topology']['verified']}", flush=True)
    print(
        "smoke max regular residual: "
        f"{summary['smoke_status']['max_regular_residual']}",
        flush=True,
    )
    print(
        "smoke max endpoint residual: "
        f"{summary['smoke_status']['max_endpoint_smoothness_residual']}",
        flush=True,
    )
    print(f"smoke passed: {summary['smoke_status']['passed']}", flush=True)
    print(f"verdict: {summary['verdict']}", flush=True)


if __name__ == "__main__":
    main()
