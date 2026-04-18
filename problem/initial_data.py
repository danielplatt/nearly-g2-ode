"""User-specified parameters, configs, and verified singular-end data."""

from __future__ import annotations

from dataclasses import dataclass

from mpmath import mp

from .types import State


@dataclass(frozen=True)
class ProblemParameters:
    """Problem parameters for one exploration point."""

    a: mp.mpf
    c: mp.mpf
    alpha: mp.mpf
    lam: mp.mpf


@dataclass(frozen=True)
class SolverConfig:
    """Numerical settings for one Taylor-marching run."""

    series_order: int
    working_dps: int
    target_dps: int
    step_safety: mp.mpf
    sample_points: int
    target_t: mp.mpf


def _known_alpha(a: mp.mpf, c: mp.mpf, lam: mp.mpf) -> mp.mpf:
    """Return the branch-selected y2'(0) formula from the source note."""
    sign_arg = lam ** mp.mpf("1.5") * mp.sqrt(-a * c) - mp.sqrt(3 * a - c)
    numer = lam**2 * a * c + mp.sqrt(-lam * a * c * (3 * a - c))
    sign = mp.sign(sign_arg) or mp.one
    return sign * numer / (2 * (3 * a - c))


def _default_params() -> ProblemParameters:
    """Build the verified base parameter point."""
    with mp.workdps(80):
        a = mp.sqrt(5) / 20
        c = -3 * mp.sqrt(5) / 100
        lam = 6 / mp.sqrt(5)
        alpha = _known_alpha(a, c, lam)
    return ProblemParameters(a=a, c=c, alpha=alpha, lam=lam)


def _default_config(order: int, dps: int, target_dps: int) -> SolverConfig:
    """Create one solver configuration with the common midpoint target."""
    with mp.workdps(80):
        return SolverConfig(
            series_order=order,
            working_dps=dps,
            target_dps=target_dps,
            step_safety=mp.mpf("0.5"),
            sample_points=5,
            target_t=mp.pi / 6,
        )


DEFAULT_PARAMS = _default_params()
DEFAULT_CONFIG = _default_config(24, 120, 50)
REFINED_CONFIG = _default_config(32, 180, 80)
RHO = State(mp.zero, mp.one, -mp.one, mp.zero, mp.zero, mp.mpf("7"), mp.mpf("-7"), mp.zero)


def _matches_base_branch(params: ProblemParameters) -> bool:
    """Check whether the current parameters match the verified base branch."""
    tolerance = mp.mpf("1e-40")
    return (
        abs(params.a - DEFAULT_PARAMS.a) < tolerance
        and abs(params.c - DEFAULT_PARAMS.c) < tolerance
        and abs(params.lam - DEFAULT_PARAMS.lam) < tolerance
    )


def y_zero_jet(params: ProblemParameters) -> State[mp.mpf]:
    """Return the verified y(0) branch; alpha is accepted but unused here."""
    if not _matches_base_branch(params):
        raise NotImplementedError("V1 only stores the verified zero jet for the default (a, c, lam) branch.")
    with mp.workdps(80):
        sqrt5 = mp.sqrt(5)
        sqrt15 = mp.sqrt(15)
        return State(
            9 * sqrt5 / 100,
            sqrt15 / 25,
            sqrt15 / 25,
            sqrt5 / 100,
            23 * sqrt5 / 100,
            2 * sqrt15 / 25,
            2 * sqrt15 / 25,
            -9 * sqrt5 / 100,
        )


def y_first_jet(params: ProblemParameters) -> State[mp.mpf]:
    """Return the forced first-jet direction determined by alpha."""
    return params.alpha * RHO
