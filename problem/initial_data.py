"""User-specified endpoint parameters, configs, and Berger validation data."""

from __future__ import annotations

from dataclasses import dataclass

from mpmath import mp

from .types import State


@dataclass(frozen=True)
class LeftEndpointParameters:
    """Parameters attached to the left singular orbit at t = 0."""

    a: mp.mpf
    c: mp.mpf
    alpha: mp.mpf


@dataclass(frozen=True)
class RightEndpointParameters:
    """Parameters attached to the right singular orbit at t = pi/3."""

    d: mp.mpf
    f: mp.mpf
    omega: mp.mpf


@dataclass(frozen=True)
class ProblemParameters:
    """Two-ended parameters for one weighted matching problem."""

    lam: mp.mpf
    interval_end: mp.mpf
    left: LeftEndpointParameters
    right: RightEndpointParameters


@dataclass(frozen=True)
class SolverConfig:
    """Numerical settings for one two-sided Taylor-marching run."""

    series_order: int
    working_dps: int
    target_dps: int
    step_safety: mp.mpf
    sample_points: int
    match_t: mp.mpf


LEFT_RHO = State(mp.zero, mp.one, -mp.one, mp.zero, mp.zero, mp.mpf("7"), mp.mpf("-7"), mp.zero)
RIGHT_RHO = State(mp.zero, mp.zero, mp.mpf("-7"), mp.one, mp.mpf("-7"), mp.one, mp.zero, mp.zero)


def _known_alpha(a: mp.mpf, c: mp.mpf, lam: mp.mpf) -> mp.mpf:
    """Return the branch-selected left odd coefficient from the source note."""
    sign_arg = lam ** mp.mpf("1.5") * mp.sqrt(-a * c) - mp.sqrt(3 * a - c)
    numer = lam**2 * a * c + mp.sqrt(-lam * a * c * (3 * a - c))
    sign = mp.sign(sign_arg) or mp.one
    return sign * numer / (2 * (3 * a - c))


def _default_params() -> ProblemParameters:
    """Build the stored Berger validation parameter point."""
    with mp.workdps(80):
        sqrt5 = mp.sqrt(5)
        a = sqrt5 / 20
        c = -3 * sqrt5 / 100
        alpha = _known_alpha(a, c, 6 / sqrt5)
        right = RightEndpointParameters(
            d=-a,
            f=-c,
            omega=-alpha,
        )
        left = LeftEndpointParameters(a=a, c=c, alpha=alpha)
        return ProblemParameters(
            lam=6 / sqrt5,
            interval_end=mp.pi / 3,
            left=left,
            right=right,
        )


def _default_config(order: int, dps: int, target_dps: int) -> SolverConfig:
    """Create one solver configuration with the common Berger match point."""
    with mp.workdps(80):
        return SolverConfig(
            series_order=order,
            working_dps=dps,
            target_dps=target_dps,
            step_safety=mp.mpf("0.5"),
            sample_points=5,
            match_t=mp.pi / 6,
        )


DEFAULT_PARAMS = _default_params()
DEFAULT_CONFIG = _default_config(24, 120, 50)
REFINED_CONFIG = _default_config(32, 180, 80)


def _matches_berger_branch(params: ProblemParameters) -> bool:
    """Check whether the stored Berger endpoint constants match the defaults."""
    tolerance = mp.mpf("1e-40")
    return (
        abs(params.lam - DEFAULT_PARAMS.lam) < tolerance
        and abs(params.interval_end - DEFAULT_PARAMS.interval_end) < tolerance
        and abs(params.left.a - DEFAULT_PARAMS.left.a) < tolerance
        and abs(params.left.c - DEFAULT_PARAMS.left.c) < tolerance
        and abs(params.right.d - DEFAULT_PARAMS.right.d) < tolerance
        and abs(params.right.f - DEFAULT_PARAMS.right.f) < tolerance
    )


def left_zero_jet(params: ProblemParameters) -> State[mp.mpf]:
    """Return the stored Berger left zero jet; alpha is accepted but unused here."""
    if not _matches_berger_branch(params):
        raise NotImplementedError("V2 only stores the Berger left zero jet for the default endpoint constants.")
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


def left_first_jet(params: ProblemParameters) -> State[mp.mpf]:
    """Return the forced Berger left first jet determined by alpha."""
    return params.left.alpha * LEFT_RHO


def right_zero_jet(params: ProblemParameters) -> State[mp.mpf]:
    """Return the stored Berger right zero jet; omega is accepted but unused here."""
    if not _matches_berger_branch(params):
        raise NotImplementedError("V2 only stores the Berger right zero jet for the default endpoint constants.")
    with mp.workdps(80):
        sqrt5 = mp.sqrt(5)
        sqrt15 = mp.sqrt(15)
        return State(
            -9 * sqrt5 / 100,
            -sqrt5 / 100,
            2 * sqrt15 / 25,
            -sqrt15 / 25,
            -2 * sqrt15 / 25,
            sqrt15 / 25,
            -23 * sqrt5 / 100,
            9 * sqrt5 / 100,
        )


def right_first_jet(params: ProblemParameters) -> State[mp.mpf]:
    """Return the forced Berger right first jet determined by omega."""
    return params.right.omega * RIGHT_RHO


def endpoint_zero_jet(side: str, params: ProblemParameters) -> State[mp.mpf]:
    """Return the stored Berger zero jet for one endpoint side."""
    if side == "left":
        return left_zero_jet(params)
    if side == "right":
        return right_zero_jet(params)
    raise ValueError(f"Unknown endpoint side {side!r}.")


def endpoint_first_jet(side: str, params: ProblemParameters) -> State[mp.mpf]:
    """Return the stored Berger first jet for one endpoint side."""
    if side == "left":
        return left_first_jet(params)
    if side == "right":
        return right_first_jet(params)
    raise ValueError(f"Unknown endpoint side {side!r}.")
