"""User-specified endpoint parameters, configs, and endpoint Taylor data."""

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
class FixedRightEndpointData:
    """Fixed weighted data for a right endpoint chart not covered by Berger parameters."""

    offset: State[mp.mpf]
    zero_jet: State[mp.mpf]
    first_jet: State[mp.mpf]
    label: str = "fixed"
    series_coefficients: State[tuple[mp.mpf, ...]] | None = None


@dataclass(frozen=True)
class ProblemParameters:
    """Two-ended parameters for one weighted matching problem."""

    lam: mp.mpf
    interval_end: mp.mpf
    left: LeftEndpointParameters
    right: RightEndpointParameters
    right_chart: str = "berger"
    fixed_right: FixedRightEndpointData | None = None
    left_mu: int = -1
    right_mu: int = -1
    p_signs: tuple[int, int, int] = (-1, 1, 1)
    right_p_signs: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class LeftEndpointPreset:
    """Named left-end data before the matching right endpoint is known."""

    lam: mp.mpf
    left: LeftEndpointParameters


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


def source_alpha(a: mp.mpf, c: mp.mpf, lam: mp.mpf) -> mp.mpf:
    """Return the branch-selected left odd coefficient from the source note."""
    sign_arg = lam ** mp.mpf("1.5") * mp.sqrt(-a * c) - mp.sqrt(3 * a - c)
    numer = lam**2 * a * c + mp.sqrt(-lam * a * c * (3 * a - c))
    sign = mp.sign(sign_arg) or mp.one
    return sign * numer / (2 * (3 * a - c))


def _endpoint_mu(mu: int) -> int:
    """Normalize one source-formula mu branch sign."""
    parsed = int(mu)
    if parsed not in {-1, 1}:
        raise ValueError("Endpoint mu branch must be either -1 or 1.")
    return parsed


def endpoint_p_signs(signs: tuple[int, int, int]) -> tuple[int, int, int]:
    """Normalize the three square-root signs used to reconstruct p-values."""
    if len(signs) != 3:
        raise ValueError("Endpoint p-sign branch must contain exactly three signs.")
    parsed = tuple(int(sign) for sign in signs)
    if any(sign not in {-1, 1} for sign in parsed):
        raise ValueError("Endpoint p-sign branch entries must be either -1 or 1.")
    return parsed


def _left_y6_ratio(a: mp.mpf, c: mp.mpf, lam: mp.mpf, mu: int = -1) -> mp.mpf:
    """Return the source relation y6'(0) / y2'(0) for the left chart."""
    branch = _left_formula_branch(a, c, lam)
    root = mp.sqrt(3 * a - c)
    if branch == "ac_negative":
        mu = _endpoint_mu(mu)
        scaled_root = lam ** mp.mpf("1.5") * mp.sqrt(-a * c)
        return (root - 3 * mu * scaled_root) / (root + mu * scaled_root)
    scaled_root = lam ** mp.mpf("1.5") * mp.sqrt(a * c)
    return (root - 3 * scaled_root) / (root + scaled_root)


def mirrored_problem_parameters(
    a: mp.mpf,
    c: mp.mpf,
    alpha: mp.mpf,
    lam: mp.mpf,
    interval_end: mp.mpf,
    *,
    left_mu: int = -1,
    right_mu: int | None = None,
    p_signs: tuple[int, int, int] = (-1, 1, 1),
    right_p_signs: tuple[int, int, int] | None = None,
) -> ProblemParameters:
    """Build the two-ended parameter package using the confirmed mirror map."""
    left = LeftEndpointParameters(a=a, c=c, alpha=alpha)
    right = RightEndpointParameters(d=-a, f=-c, omega=-alpha)
    if right_mu is None:
        right_mu = left_mu
    return ProblemParameters(
        lam=lam,
        interval_end=interval_end,
        left=left,
        right=right,
        left_mu=_endpoint_mu(left_mu),
        right_mu=_endpoint_mu(right_mu),
        p_signs=endpoint_p_signs(p_signs),
        right_p_signs=None if right_p_signs is None else endpoint_p_signs(right_p_signs),
    )


def _default_params() -> ProblemParameters:
    """Build the stored Berger validation parameter point."""
    with mp.workdps(80):
        sqrt5 = mp.sqrt(5)
        lam = 6 / sqrt5
        a = sqrt5 / 20
        c = -3 * sqrt5 / 100
        alpha = source_alpha(a, c, lam)
        return mirrored_problem_parameters(a, c, alpha, lam, mp.pi / 3)


def berger_parameters() -> ProblemParameters:
    """Return the validated homogeneous Berger two-ended parameter package."""
    return DEFAULT_PARAMS


def round_s7_left_parameters() -> LeftEndpointPreset:
    """Return the derived left-end data for the round S7 homogeneous solution."""
    with mp.workdps(80):
        sqrt5 = mp.sqrt(5)
        return LeftEndpointPreset(
            lam=6 / sqrt5,
            left=LeftEndpointParameters(a=sqrt5 / 25, c=-3 * sqrt5 / 5, alpha=sqrt5 / 50),
        )


def round_s7_candidate_parameters() -> ProblemParameters:
    """Return the derived round-S7 two-ended parameter package."""
    with mp.workdps(80):
        sqrt5 = mp.sqrt(5)
        sqrt15 = mp.sqrt(15)
        left = LeftEndpointParameters(a=sqrt5 / 25, c=-3 * sqrt5 / 5, alpha=sqrt5 / 50)
        offset_scale = sqrt5 / 25
        fixed_right = FixedRightEndpointData(
            label="round_s7",
            offset=State.from_iterable(
                offset_scale * value for value in (1, -1, -2, 2, -2, 2, 19, -19)
            ),
            zero_jet=State(
                mp.zero,
                sqrt5 / 25,
                sqrt15 / 25,
                -11 * sqrt15 / 25,
                -sqrt15 / 25,
                11 * sqrt15 / 25,
                -sqrt5,
                144 * sqrt5 / 25,
            ),
            first_jet=State(
                mp.zero,
                mp.zero,
                sqrt5 / 50,
                -sqrt5 / 2,
                sqrt5 / 50,
                -sqrt5 / 2,
                mp.zero,
                mp.zero,
            ),
        )
        return ProblemParameters(
            lam=6 / sqrt5,
            interval_end=mp.pi / 3,
            left=left,
            right=DEFAULT_PARAMS.right,
            right_chart="s7_p3",
            fixed_right=fixed_right,
        )


def squashed_s7_parameters() -> ProblemParameters:
    """Return the derived squashed-S7 two-ended parameter package.

    The right endpoint is a p2-collapse chart.  The data are derived from the
    3-Sasakian frame formula after translating it into this repository's q-basis
    and validating the resulting q(t) against the q-system with lambda > 0.
    """
    with mp.workdps(80):
        sqrt5 = mp.sqrt(5)
        sqrt15 = mp.sqrt(15)
        left = LeftEndpointParameters(a=sqrt5 / 25, c=-3 * sqrt5 / 5, alpha=-sqrt5 / 50)
        offset_scale = sqrt5 / 25
        fixed_right = FixedRightEndpointData(
            label="squashed_s7",
            offset=State.from_iterable(
                offset_scale * value for value in (1, -2, -1, 2, -2, 19, 2, -19)
            ),
            zero_jet=State(
                mp.zero,
                sqrt15 / 25,
                sqrt5 / 25,
                -11 * sqrt15 / 25,
                -sqrt15 / 25,
                -sqrt5,
                11 * sqrt15 / 25,
                144 * sqrt5 / 25,
            ),
            first_jet=State(
                mp.zero,
                sqrt5 / 50,
                mp.zero,
                -sqrt5 / 2,
                sqrt5 / 50,
                mp.zero,
                -sqrt5 / 2,
                mp.zero,
            ),
        )
        return ProblemParameters(
            lam=6 / sqrt5,
            interval_end=mp.pi / 3,
            left=left,
            right=DEFAULT_PARAMS.right,
            right_chart="s7_p2",
            fixed_right=fixed_right,
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


def _require_left_formula_branch(a: mp.mpf, c: mp.mpf, lam: mp.mpf) -> None:
    """Require one implemented real endpoint formula branch."""
    _left_formula_branch(a, c, lam)


def _left_formula_branch(a: mp.mpf, c: mp.mpf, lam: mp.mpf) -> str:
    """Return the implemented real endpoint branch for one left-type orbit."""
    if lam <= 0 or a == 0 or c == 0 or 3 * a - c <= 0:
        raise NotImplementedError("Only branches with lambda > 0, ac != 0, and 3a-c > 0 are implemented.")
    if a * c < 0:
        return "ac_negative"
    return "ac_positive"


def _mirror_state(state: State[mp.mpf]) -> State[mp.mpf]:
    """Mirror one weighted left endpoint state into right endpoint coordinates."""
    return State(state.y8, -state.y4, state.y6, -state.y2, -state.y7, state.y3, -state.y5, state.y1)


def _left_zero_jet_from_values(a: mp.mpf, c: mp.mpf, lam: mp.mpf, mu: int = -1) -> State[mp.mpf]:
    """Return the source formula zero jet for one left-type endpoint."""
    branch = _left_formula_branch(a, c, lam)
    base = (3 * a + c) / (4 * (3 * a - c))
    if branch == "ac_negative":
        mu = mp.mpf(_endpoint_mu(mu))
        root = mp.sqrt(-lam * a * c * (3 * a - c))
        b2 = mu * lam * mp.sqrt(-a * c) + mp.sqrt((3 * a - c) / lam)
        b6 = -3 * lam * mu * mp.sqrt(-a * c) - mp.sqrt((3 * a - c) / lam)
        b1 = -lam**2 * a * (base + 1) / 2 + mu * root / (2 * c)
        correction = mp.sqrt(-lam * a * c / (3 * a - c))
        b4 = lam**2 * c * (base - 1) / 2 + mu * root / (2 * a) - mu * correction
        b5 = 3 * lam**2 * a * (base + 1) / 2 - mu * root / (2 * c) + mu * correction
        b8 = -3 * lam**2 * c * (base - 1) / 2 - mu * root / (2 * a)
        return State(b1, b2, b2, b4, b5, b6, b6, b8)

    root = mp.sqrt(lam * a * c * (3 * a - c))
    correction = mp.sqrt(lam * a * c / (3 * a - c))
    b1 = -lam**2 * a * (base - 1) / 2 + root / (2 * c)
    b2 = lam * mp.sqrt(a * c) + mp.sqrt((3 * a - c) / lam)
    b4 = lam**2 * c * (base + 1) / 2 + root / (2 * a) + correction
    b5 = 3 * lam**2 * a * (base - 1) / 2 - root / (2 * c) - correction
    b6 = -3 * lam * mp.sqrt(a * c) - mp.sqrt((3 * a - c) / lam)
    b8 = -3 * lam**2 * c * (base + 1) / 2 - root / (2 * a)
    return State(b1, b2, b2, b4, b5, b6, b6, b8)


def _left_first_jet_from_values(a: mp.mpf, c: mp.mpf, alpha: mp.mpf, lam: mp.mpf, mu: int = -1) -> State[mp.mpf]:
    """Return the source-determined first jet for one left-type endpoint."""
    _require_left_formula_branch(a, c, lam)
    eta = _left_y6_ratio(a, c, lam, mu) * alpha
    return State(mp.zero, alpha, -alpha, mp.zero, mp.zero, eta, -eta, mp.zero)


def left_zero_jet_from_values(a: mp.mpf, c: mp.mpf, lam: mp.mpf, mu: int = -1) -> State[mp.mpf]:
    """Return the source-formula left zero jet for explicit endpoint values."""
    return _left_zero_jet_from_values(a, c, lam, mu)


def left_first_jet_from_values(a: mp.mpf, c: mp.mpf, alpha: mp.mpf, lam: mp.mpf, mu: int = -1) -> State[mp.mpf]:
    """Return the source-formula left first jet for explicit endpoint values."""
    return _left_first_jet_from_values(a, c, alpha, lam, mu)


def left_zero_jet(params: ProblemParameters) -> State[mp.mpf]:
    """Return the source formula left zero jet; alpha is accepted but unused."""
    return _left_zero_jet_from_values(params.left.a, params.left.c, params.lam, params.left_mu)


def left_first_jet(params: ProblemParameters) -> State[mp.mpf]:
    """Return the source-determined left first jet from alpha."""
    return _left_first_jet_from_values(params.left.a, params.left.c, params.left.alpha, params.lam, params.left_mu)


def right_zero_jet(params: ProblemParameters) -> State[mp.mpf]:
    """Return the right zero jet from independent right endpoint data."""
    return _mirror_state(_left_zero_jet_from_values(-params.right.d, -params.right.f, params.lam, params.right_mu))


def right_first_jet(params: ProblemParameters) -> State[mp.mpf]:
    """Return the right first jet from independent right endpoint data."""
    return _mirror_state(
        _left_first_jet_from_values(-params.right.d, -params.right.f, -params.right.omega, params.lam, params.right_mu)
    )


def endpoint_zero_jet(side: str, params: ProblemParameters) -> State[mp.mpf]:
    """Return the formula-derived zero jet for one endpoint side."""
    if side == "left":
        return left_zero_jet(params)
    if side == "right":
        return right_zero_jet(params)
    if side == "s7_p2_right":
        if params.fixed_right is None:
            raise ValueError("S7 p2 right chart requires fixed_right endpoint data.")
        return params.fixed_right.zero_jet
    if side == "s7_p3_right":
        if params.fixed_right is None:
            raise ValueError("S7 p3 right chart requires fixed_right endpoint data.")
        return params.fixed_right.zero_jet
    raise ValueError(f"Unknown endpoint side {side!r}.")


def endpoint_first_jet(side: str, params: ProblemParameters) -> State[mp.mpf]:
    """Return the formula-derived first jet for one endpoint side."""
    if side == "left":
        return left_first_jet(params)
    if side == "right":
        return right_first_jet(params)
    if side == "s7_p2_right":
        if params.fixed_right is None:
            raise ValueError("S7 p2 right chart requires fixed_right endpoint data.")
        return params.fixed_right.first_jet
    if side == "s7_p3_right":
        if params.fixed_right is None:
            raise ValueError("S7 p3 right chart requires fixed_right endpoint data.")
        return params.fixed_right.first_jet
    raise ValueError(f"Unknown endpoint side {side!r}.")
