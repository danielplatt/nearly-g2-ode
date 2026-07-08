"""Tail-defect diagnostics for the Podesta SU(2)^3 S7 scout.

This module tests shooting defects that could support a tail exclusion for the
one-parameter Podesta family.  The main candidate is the scaled right-endpoint
coordinate

    X2(a) = h2(T_a) / a^3,

where T_a is the first zero of h0, equivalently f0.  A standard K- closure
requires f2(T_a)=0, hence X2(a)=0.  After scaling

    h0=a x0, h1=x1, h2=a^3 x2, h3=a x3,

the exact ODE has a large-|a| limit.  Numerically, the limiting first crossing
has x2 about 0.006 and x3 about -1.1, so X2 is a plausible tail witness.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from mpmath import mp

from . import su2_cubed_scout as scout


TAIL_DEFECT_VERSION = "s7-su2-cubed-tail-defect-v1"
DEFAULT_STEP = 2.5e-4
DEFAULT_EPSILON = 1e-3
DEFAULT_SUPPORT_TIME = 3.5
DEFAULT_TERMINAL_BARRIER_TIME = 3.58
DEFAULT_CANDIDATE_A = 10000.0
DEFAULT_TUBE_CANDIDATE_A = 100000000.0
DEFAULT_P_TUBE_ENTRY_TIME = 3.527
DEFAULT_P_TUBE_START = 0.305
DEFAULT_P_TUBE_END = 0.25
DEFAULT_P_TUBE_STEP = 5e-5
DEFAULT_P_CORRIDOR_START = 0.25
DEFAULT_P_CORRIDOR_END = 0.2
DEFAULT_P_CORRIDOR_STEP = 5e-4
DEFAULT_P_CORRIDOR_LOWER_START = (3.54, 6.7, 0.0055, -0.75)
DEFAULT_P_CORRIDOR_UPPER_START = (3.585, 8.0, 0.012, -0.6)
DEFAULT_P_CORRIDOR_LOWER_SLOPE = (0.0, 30.0, 0.025075, 12.0)
DEFAULT_P_CORRIDOR_UPPER_SLOPE = (-12.0, -150.0, -0.12, 0.0)
DEFAULT_P_CORRIDOR_TUNE_X2_SLOPES = (0.02505, 0.025075, 0.0251, 0.02515)
DEFAULT_P_CORRIDOR_TUNE_X1_UPPER_SLOPES = (-150.0, -151.0, -152.0, -155.0)
DEFAULT_TERMINAL_TAKEOVER_P_MIN = 1e-3
DEFAULT_TERMINAL_TAKEOVER_P_STEP = 1e-2
DEFAULT_TERMINAL_TAKEOVER_BOX_LOW = (3.5, 2.0, 0.001, -4.0)
DEFAULT_TERMINAL_TAKEOVER_BOX_HIGH = (4.0, 30.0, 0.03, -0.5)
DEFAULT_TERMINAL_TAKEOVER_X3_WALL = -0.5
DEFAULT_LATE_TAIL_TAKEOVER_START = 0.20995
DEFAULT_LATE_TAIL_TAKEOVER_BOX_LOW = (3.5, 0.8, 0.001, -10.0)
DEFAULT_LATE_TAIL_TAKEOVER_BOX_HIGH = (8.0, 100.0, 0.1, -0.6)
DEFAULT_LATE_TAIL_TAKEOVER_X3_WALL = -0.6
DEFAULT_FRONTIER_CONTINUATION_END = 0.23595
DEFAULT_FRONTIER_CONTINUATION_SUBDIVISIONS = (2, 1, 2, 1)
DEFAULT_FRONTIER_CONTINUATION_P_SUBDIVISIONS = 2
DEFAULT_P_TUBE_FRONTIER_LOW = (
    3.555323784527346,
    6.798075137396692,
    0.005626009445354444,
    -0.7280060017773002,
)
DEFAULT_P_TUBE_FRONTIER_HIGH = (
    3.584573784527305,
    7.82007513739715,
    0.011541013428022802,
    -0.6139850702696757,
)
DEFAULT_P_TUBE_CONTINUED_FRONTIER_LOW = (
    3.543705770716578,
    6.542864320170368,
    0.0027721071923574466,
    -0.840813636651889,
)
DEFAULT_P_TUBE_CONTINUED_FRONTIER_HIGH = (
    3.6019357707165045,
    8.389364320170866,
    0.013204610753265497,
    -0.6225926814317722,
)
DEFAULT_PIECEWISE_CORRIDOR_KNOTS = (
    (
        DEFAULT_FRONTIER_CONTINUATION_END,
        DEFAULT_P_TUBE_CONTINUED_FRONTIER_LOW,
        DEFAULT_P_TUBE_CONTINUED_FRONTIER_HIGH,
    ),
    (
        0.22995,
        (3.54, 5.9, 0.00245, -1.05),
        (3.65, 9.5, 0.0145, -0.6),
    ),
    (
        0.21995,
        (3.54, 4.0, 0.0022, -1.35),
        (3.7, 13.0, 0.016, -0.6),
    ),
    (
        0.20995,
        (3.53, 1.45, 0.0020, -1.7),
        (3.9, 18.0, 0.022, -0.6),
    ),
)
DEFAULT_SEGMENTED_TUBE_PROFILES = (
    ((0.005, 0.05, 0.0005, 0.005), (0.005, 0.05, 0.0005, 0.005)),
    ((0.005, 0.05, 0.0005, 0.0051), (0.005, 0.05, 0.0005, 0.0052)),
    ((0.01, 0.1, 0.001, 0.01), (0.01, 0.1, 0.001, 0.01)),
    ((0.02, 0.2, 0.002, 0.02), (0.02, 0.2, 0.002, 0.02)),
    ((0.05, 0.5, 0.005, 0.05), (0.05, 0.5, 0.005, 0.05)),
)
DEFAULT_SUPPORT_TUBE_RADIUS = (1e-7, 1e-6, 1e-8, 1e-7)
DEFAULT_SEGMENTED_P_TUBE_PROFILES = (
    (0.05, 1.0, 0.01, 0.1),
    (0.1, 3.0, 0.02, 0.3),
    (0.2, 8.0, 0.05, 0.9),
    (0.5, 20.0, 0.1, 2.0),
    (0.5, 20.0, 0.1, 2.5),
    (0.8, 25.0, 0.15, 3.0),
    (1.2, 30.0, 0.2, 4.0),
)
DEFAULT_ASYMMETRIC_P_TUBE_PROFILES = (
    ((0.05, 1.0, 0.01, 0.1), (0.05, 1.0, 0.01, 0.1)),
    ((0.1, 3.0, 0.02, 0.3), (0.1, 3.0, 0.02, 0.3)),
    ((0.2, 8.0, 0.05, 0.9), (0.2, 8.0, 0.05, 0.9)),
    ((0.5, 20.0, 0.1, 2.0), (0.5, 20.0, 0.1, 2.0)),
    ((0.5, 20.0, 0.1, 2.5), (0.5, 20.0, 0.1, 2.5)),
    ((0.8, 25.0, 0.15, 3.0), (0.8, 25.0, 0.15, 3.0)),
    ((1.2, 30.0, 0.2, 4.0), (1.2, 30.0, 0.2, 4.0)),
    ((0.5, 20.0, 0.1, 2.5), (0.5, 20.0, 0.1, 2.0)),
    ((1.0, 25.0, 0.15, 4.0), (0.5, 24.0, 0.1, 2.5)),
    ((1.5, 30.0, 0.2, 5.0), (0.8, 32.0, 0.15, 3.0)),
    ((2.5, 45.0, 0.3, 7.0), (1.2, 45.0, 0.2, 4.0)),
)
DEFAULT_HYBRID_HANDOFF_START_P = 0.325
DEFAULT_HYBRID_HANDOFF_ENTRY_TIME = 3.5
DEFAULT_HYBRID_HANDOFF_BRIDGE_AFTER_TIME = 3.5056
DEFAULT_HYBRID_HANDOFF_TUBE_END_P = 0.272
DEFAULT_HYBRID_HANDOFF_FRONTIER_P = 0.25
DEFAULT_HYBRID_HANDOFF_FRONTIER_LOW = (3.545, 6.35, 0.0052, -0.78)
DEFAULT_HYBRID_HANDOFF_FRONTIER_HIGH = (3.60, 8.3, 0.012, -0.58)
DEFAULT_HYBRID_HANDOFF_P_TUBE_RADIUS0 = (8e-5, 1e-3, 1e-5, 2.5e-4)
DEFAULT_HYBRID_HANDOFF_CORRIDOR_SUBDIVISIONS = (2, 2, 2, 1)
DEFAULT_BROAD_TAIL_AUTOMATIC_END_P = 0.212
DEFAULT_BROAD_TAIL_AUTOMATIC_SAFETY = 1e-4
DEFAULT_BROAD_TAIL_AUTOMATIC_SUBDIVISIONS = (2, 2, 2, 1)
DEFAULT_BROAD_TAIL_AUTOMATIC_P_SUBDIVISIONS = 1
DEFAULT_SAMPLED_CARRIED_C_P_TUBE_START = 0.65
DEFAULT_SAMPLED_CARRIED_C_P_TUBE_END = DEFAULT_HYBRID_HANDOFF_START_P
DEFAULT_SAMPLED_CARRIED_C_P_TUBE_ENTRY_TIME = 2.0
DEFAULT_SAMPLED_CARRIED_C_P_TUBE_STEP = 5e-4
DEFAULT_SAMPLED_CARRIED_C_P_TUBE_RADIUS = (1e-5, 1e-4, 1e-6, 1e-5, 1e-5)
DEFAULT_TAYLOR_P_SLICE_CAUCHY_RADII = (3.2, 3.4, 3.5, 3.55, 3.58)
DEFAULT_SAMPLED_CARRIED_C_P_TUBE_SUBDIVISIONS = (1, 1, 1, 2, 1)
DEFAULT_SAMPLED_CARRIED_C_P_TUBE_PROFILES = (
    ((0.01, 0.2, 0.002, 0.02, 0.02), (2.0, 20.0, 1.0, 10.0, 2.0)),
    ((0.02, 0.5, 0.005, 0.05, 0.05), (5.0, 50.0, 2.0, 20.0, 5.0)),
    ((0.05, 1.0, 0.01, 0.1, 0.1), (10.0, 100.0, 5.0, 50.0, 10.0)),
)
TIGHT_SAMPLED_CARRIED_C_P_TUBE_PROFILES = (
    ((0.002, 0.05, 0.0005, 0.005, 0.005), (0.5, 5.0, 0.2, 2.0, 0.5)),
    ((0.005, 0.1, 0.001, 0.01, 0.01), (1.0, 10.0, 0.5, 5.0, 1.0)),
    ((0.01, 0.2, 0.002, 0.02, 0.02), (2.0, 20.0, 1.0, 10.0, 2.0)),
    ((0.02, 0.5, 0.005, 0.05, 0.05), (5.0, 50.0, 2.0, 20.0, 5.0)),
)
SAMPLED_CARRIED_C_P_TUBE_PROFILE_SETS = {
    "robust": DEFAULT_SAMPLED_CARRIED_C_P_TUBE_PROFILES,
    "tight": TIGHT_SAMPLED_CARRIED_C_P_TUBE_PROFILES,
}
DEFAULT_CARRIED_C_P_CORRIDOR_START = DEFAULT_HYBRID_HANDOFF_START_P
DEFAULT_CARRIED_C_P_CORRIDOR_END = 0.25
DEFAULT_CARRIED_C_P_CORRIDOR_STEP = 5e-4
DEFAULT_CARRIED_C_P_CORRIDOR_SAFETY = (2e-3, 5e-3, 5e-5, 5e-4, 4.0)
DEFAULT_CARRIED_C_P_CORRIDOR_SUBDIVISIONS = (2, 2, 2, 2, 1)
DEFAULT_CARRIED_C_P_CORRIDOR_P_SUBDIVISIONS = 1
DEFAULT_CARRIED_C_P_CORRIDOR_SOURCE_LOW = (
    3.403622084208176,
    3.5351608186538455,
    0.006607311202170192,
    -1.0376220176296234,
    0.023046158239418968,
)
DEFAULT_CARRIED_C_P_CORRIDOR_SOURCE_HIGH = (
    3.6541885741258002,
    10.872623537272094,
    0.013096277692288452,
    0.017708843880467624,
    0.14768464899477785,
)
DEFAULT_CARRIED_C_P_WALL_START = 0.29
DEFAULT_CARRIED_C_P_WALL_END = DEFAULT_TERMINAL_TAKEOVER_P_MIN
DEFAULT_CARRIED_C_P_WALL_STEP = 0.005
DEFAULT_CARRIED_C_P_WALL_BOX_LOW = (
    3.440353980346136,
    1.6214293214935245,
    0.0,
    -1.5162298825611453,
)
DEFAULT_CARRIED_C_P_WALL_BOX_HIGH = (
    3.7311582192101933,
    14.194506388360724,
    0.014427895811878644,
    0.0,
)
DEFAULT_CARRIED_C_P_WALL_SUBDIVISIONS = (2, 2, 2, 2, 1)
DEFAULT_CARRIED_C_P_WALL_P_SUBDIVISIONS = 4
DEFAULT_X2_ZERO_FACTOR_P_RANGE = (3.5e-4, 0.29)
DEFAULT_X2_ZERO_FACTOR_TIME_RANGE = (
    DEFAULT_CARRIED_C_P_WALL_BOX_LOW[0],
    DEFAULT_CARRIED_C_P_WALL_BOX_HIGH[0],
)
DEFAULT_X2_ZERO_FACTOR_X3_RANGE = (
    DEFAULT_CARRIED_C_P_WALL_BOX_LOW[3],
    DEFAULT_CARRIED_C_P_WALL_BOX_HIGH[3],
)
DEFAULT_REGULAR_TIME_AUTOMATIC_START = 0.5
DEFAULT_REGULAR_TIME_AUTOMATIC_END = DEFAULT_SUPPORT_TIME
DEFAULT_REGULAR_TIME_AUTOMATIC_STEP = 1e-3
DEFAULT_REGULAR_TIME_AUTOMATIC_SAFETY = (5e-3, 5e-2, 5e-4, 5e-3)
DEFAULT_REGULAR_TIME_AUTOMATIC_RADIUS0 = (1e-8, 1e-7, 1e-9, 1e-8)
DEFAULT_REGULAR_TIME_AUTOMATIC_SUBDIVISIONS = (1, 1, 1, 1)
DEFAULT_REGULAR_TIME_AUTOMATIC_TIME_SUBDIVISIONS = 1
DEFAULT_TAYLOR_START_TIME = DEFAULT_EPSILON
DEFAULT_TAYLOR_START_STEP = 5e-5
DEFAULT_TAYLOR_START_RADIUS = (1e-8, 1e-8, 1e-8, 1e-8)
DEFAULT_TAYLOR_START_SAFETY = (1e-3, 5e-2, 5e-4, 5e-3)
DEFAULT_TAYLOR_BRIDGE_END = 2.0
DEFAULT_TAYLOR_BRIDGE_MAX_ATTEMPTS = 180
DEFAULT_TAYLOR_BRIDGE_STAGES = (
    (0.002, 5e-5, 1, (5e-5, 0.0025, 2.5e-5, 0.00025), (0.0005, 0.025, 0.0025, 0.0025)),
    (0.01, 1e-4, 1, (5e-5, 0.0025, 2.5e-5, 0.00025), (0.001, 0.05, 0.005, 0.005)),
    (0.05, 5e-4, 1, (5e-5, 0.0025, 2.5e-5, 0.00025), (0.0025, 0.1, 0.005, 0.01)),
    (0.1, 1e-3, 1, (5e-5, 0.0025, 2.5e-5, 0.00025), (0.0025, 0.1, 0.005, 0.01)),
    (0.5, 1e-3, 1, (0.0001, 0.004, 4e-5, 0.0004), (0.005, 0.2, 0.01, 0.02)),
    (1.0, 1e-3, 1, (0.0002, 0.006, 6e-5, 0.0006), (0.01, 0.4, 0.02, 0.04)),
    (1.5, 1e-3, 1, (0.0003, 0.009, 9e-5, 0.0009), (0.025, 0.75, 0.04, 0.075)),
    (2.0, 1e-3, 1, (0.001, 0.03, 0.0003, 0.003), (0.1, 3.0, 0.2, 0.3)),
)
DEFAULT_TAYLOR_FRONTIER_END = 2.6
DEFAULT_TAYLOR_FRONTIER_STEP = 1e-3
DEFAULT_TAYLOR_FRONTIER_INITIAL_GROWTH = (0.01, 0.3, 0.003, 0.03)
DEFAULT_TAYLOR_FRONTIER_MAX_GROWTH = (20.0, 200.0, 50.0, 10.0)
DEFAULT_TAYLOR_FRONTIER_MAX_ATTEMPTS = 450
DEFAULT_TAYLOR_FRONTIER_RETRY_SUBDIVISIONS = (
    (2, 1, 2, 2),
    (3, 2, 2, 3),
    (4, 4, 3, 4),
)
DEFAULT_TAYLOR_RESTART_CHAIN_END = 3.021
DEFAULT_TAYLOR_RESTART_CHAIN_MAX_GROWTH = (20.0, 200.0, 50.0, 10.0)
DEFAULT_TAYLOR_RESTART_CHAIN_MAX_ATTEMPTS = 450
DEFAULT_TUNED_TUBE_INITIAL_GROWTH = (0.005, 0.05, 0.0005, 0.005)
DEFAULT_TUNED_TUBE_MAX_GROWTH = (0.5, 5.0, 0.05, 0.5)
DEFAULT_TUNED_TUBE_GROWTH_FACTOR = 1.6
DEFAULT_TUNED_TUBE_MAX_ATTEMPTS = 24
DEFAULT_TUNED_P_TUBE_INITIAL_GROWTH = (0.05, 1.0, 0.01, 0.1)
DEFAULT_TUNED_P_TUBE_MAX_GROWTH = (5.0, 100.0, 1.0, 20.0)
DEFAULT_TUNED_P_TUBE_MAX_ATTEMPTS = 80
DEFAULT_STAGED_UNION_P_TUBE_START = 0.423
DEFAULT_STAGED_UNION_P_TUBE_SOURCE_LOW = (
    3.2025835417813724,
    3.014649320179461,
    0.008307880677192934,
    -0.6410085153670267,
)
DEFAULT_STAGED_UNION_P_TUBE_SOURCE_HIGH = (
    3.603053031976674,
    9.469593608149035,
    0.016164378948973348,
    0.4076732771398493,
)
DEFAULT_STAGED_UNION_P_TUBE_STAGES = (
    (0.4, (2, 2, 1, 4)),
)
DEFAULT_ADAPTIVE_UNION_P_TUBE_SOURCE_JSON = "output/s7_tail_proof/staged_union_p_tube_0.423_to_0.4.json"
DEFAULT_ADAPTIVE_UNION_P_TUBE_START = 0.4
DEFAULT_ADAPTIVE_UNION_P_TUBE_END = 0.37
DEFAULT_ADAPTIVE_UNION_MAX_DEPTH = 4
DEFAULT_ADAPTIVE_UNION_MAX_LEAF_BOXES = 4096
DEFAULT_ADAPTIVE_UNION_MAX_PROCESSED_BOXES: int | None = None
DEFAULT_X3_ZERO_WALL_TIME_RANGE = (3.02, 3.5)
DEFAULT_X3_ZERO_WALL_X0_RANGE = (0.30, 0.56)
DEFAULT_X3_ZERO_WALL_X1_RANGE = (5.0, 7.0)
DEFAULT_X3_ZERO_WALL_X2_RANGE = (0.005, 0.02)
DEFAULT_X3_ZERO_WALL_SUBDIVISIONS = (8, 8, 2, 1)
DEFAULT_X3_ZERO_WALL_TIME_SUBDIVISIONS = 4
DEFAULT_LATE_X3_DESCENT_START = 3.021
DEFAULT_LATE_X3_DESCENT_END = 3.45
DEFAULT_LATE_X3_DESCENT_STEP = 1e-3
DEFAULT_LATE_X3_DESCENT_RADIUS0 = (1e-4, 1e-3, 1e-5, 1e-4)
DEFAULT_LATE_X3_DESCENT_SAFETY = (1e-2, 1e-1, 1e-3, 1e-2)
DEFAULT_LATE_X3_DESCENT_X0_TARGET = 0.4
DEFAULT_LATE_X3_DESCENT_WALL_TIME_RANGE = (3.45, 3.5)
DEFAULT_LATE_X3_DESCENT_WALL_X0_RANGE = (0.3, 0.4)
DEFAULT_LATE_X3_DESCENT_WALL_X1_RANGE = (4.8, 8.1)
DEFAULT_LATE_X3_DESCENT_WALL_X2_RANGE = (0.006, 0.015)
DEFAULT_LATE_SCALAR_BARRIER_SIGMA = 0.36
DEFAULT_LATE_SCALAR_BARRIER_K = 1.23
DEFAULT_LATE_SCALAR_BARRIER_P_MAX = 0.33
DEFAULT_LATE_SCALAR_BARRIER_TIME_RANGE = (3.5, 4.0)
DEFAULT_LATE_SCALAR_BARRIER_X1_RANGE = (1.0, 30.0)


@dataclass(frozen=True)
class ScaledCrossing:
    """First crossing of x0=0 in scaled variables."""

    source: str
    a: float | None
    time: float
    x: tuple[float, float, float, float]
    step_size: float
    status: str


def limiting_scaled_rhs(t: float, x: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Return the large-|a| limiting scaled ODE."""
    x0, x1, x2, x3 = x
    x0_2 = x0 * x0
    x0_3 = x0_2 * x0
    x0_4 = x0_2 * x0_2
    t2 = t * t
    return (
        (-x0 - 3.0 * x2 * x3 * x3 / x0_4) / t
        - t / (4.0 * x0_2) * (x1 * x2 - x3 * x0_2 / 6.0),
        (-4.0 * x1 + x3**3 / x0_3) / t
        + t / (2.0 * x0_3) * (x1 * x1 * x2 + 0.5 * x1 * x3 * x0_2),
        t
        / x0_3
        * (-x0_2 * x2 * x3 / 4.0 - 0.5 * x1 * x2 * x2 + t2 * x0_3 * x0_3 / 216.0),
        (-2.0 * x3 + 6.0 * x0) / t
        + t
        / (2.0 * x0_3)
        * (x1 * x2 * x3 - x3 * x3 * x0_2 / 6.0 - t2 * x1 * x0_2 * x0_2 / 18.0),
    )


def exact_scaled_rhs(t: float, x: tuple[float, float, float, float], a: float) -> tuple[float, float, float, float]:
    """Return the exact scaled ODE for finite nonzero a."""
    h = (a * x[0], x[1], a**3 * x[2], a * x[3])
    dh = scout.h_rhs(t, h, 1.0)
    return (dh[0] / a, dh[1], dh[2] / a**3, dh[3] / a)


def finite_a_error_coefficients(t: float, x: tuple[float, float, float, float]) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    """Return R1,R2,R3 with exact_rhs = limit_rhs + b R1 + b^2 R2 + b^3 R3."""
    p, q, r, s = x
    p2 = p * p
    p3 = p2 * p
    p4 = p2 * p2
    t2 = t * t
    i1 = t * (2.0 * s * q * r - 0.5 * p2 * s * s) - t**3 * q * p4 / 18.0
    i2 = -2.0 * t * s**3 - 2.0 * t**3 * q * p2 * s / 3.0
    i3 = -2.0 * t**3 * q * s * s
    r1 = (
        -3.0 * i1 / (2.0 * p4),
        3.0 * t * q * s * s / (2.0 * p3),
        t / p3 * (t2 * p4 * s / 12.0 - 1.5 * r * s * s),
        t / (2.0 * p3) * (-s**3 - 2.0 * t2 * q * p2 * s / 3.0),
    )
    r2 = (
        -3.0 * i2 / (2.0 * p4),
        0.0,
        t / p3 * (t2 * p2 * s * s / 2.0),
        t / (2.0 * p3) * (-2.0 * t2 * q * s * s),
    )
    r3 = (
        -3.0 * i3 / (2.0 * p4),
        0.0,
        t / p3 * (t2 * s**3),
        0.0,
    )
    return r1, r2, r3


def perturbation_rhs_from_coefficients(
    t: float,
    x: tuple[float, float, float, float],
    a: float,
) -> tuple[float, float, float, float]:
    """Reconstruct the finite-a scaled RHS from the perturbation polynomial."""
    b = 1.0 / a
    limit = limiting_scaled_rhs(t, x)
    r1, r2, r3 = finite_a_error_coefficients(t, x)
    return tuple(limit[i] + b * r1[i] + b * b * r2[i] + b**3 * r3[i] for i in range(4))


def scaled_rhs_with_b(
    t: float,
    x: tuple[float, float, float, float],
    b: float,
) -> tuple[float, float, float, float]:
    """Return the scaled RHS using b=1/a, allowing b=0 for the limit."""
    limit = limiting_scaled_rhs(t, x)
    if b == 0.0:
        return limit
    r1, r2, r3 = finite_a_error_coefficients(t, x)
    return tuple(limit[i] + b * r1[i] + b * b * r2[i] + b**3 * r3[i] for i in range(4))


def scaled_taylor_c2(b: float) -> tuple[float, float, float, float]:
    """Return the exact t^2 coefficient of the smooth scaled singular-end IVP.

    The smooth scaled solution has the even expansion

        x(t,b) = x_* + c2(b) t^2 + O(t^4),

    where ``x_*=(1,27/4,-1/27,3)`` and ``b=1/a``.  The coefficients are
    obtained by matching the t-linear terms in the regular-singular scaled
    equations.
    """
    return (
        -5.0 / 96.0 + 27.0 * b * b / 2.0,
        -27.0 / 128.0 - 729.0 * b * b / 8.0,
        5.0 / 432.0 + b / 4.0,
        -23.0 / 64.0 - 27.0 * b / 8.0 + 81.0 * b * b / 4.0,
    )


def scaled_taylor_seed(time: float, b: float) -> tuple[float, float, float, float]:
    """Return the second-order Taylor seed for the smooth scaled IVP."""
    base = (1.0, 27.0 / 4.0, -1.0 / 27.0, 3.0)
    c2 = scaled_taylor_c2(b)
    time2 = time * time
    return tuple(base[index] + c2[index] * time2 for index in range(4))


def _poly_zero(order: int) -> list[mp.mpf]:
    return [mp.zero for _ in range(order + 1)]


def _poly_constant(value: mp.mpf | float, order: int) -> list[mp.mpf]:
    coeffs = _poly_zero(order)
    coeffs[0] = mp.mpf(value)
    return coeffs


def _poly_variable(order: int) -> list[mp.mpf]:
    coeffs = _poly_zero(order)
    if order >= 1:
        coeffs[1] = mp.one
    return coeffs


def _is_interval_value(value) -> bool:
    """Return whether a scalar is an mpmath interval value."""
    return hasattr(value, "a") and hasattr(value, "b")


def _poly_has_interval(poly: list) -> bool:
    """Return whether any coefficient is an interval value."""
    return any(_is_interval_value(value) for value in poly)


def _as_interval_value(value):
    """Return a point or interval scalar as an mpmath interval."""
    if _is_interval_value(value):
        return value
    from mpmath import iv

    return iv.mpf([value, value])


def _add_value(left, right):
    if _is_interval_value(left) or _is_interval_value(right):
        return _as_interval_value(left) + _as_interval_value(right)
    return left + right


def _sub_value(left, right):
    if _is_interval_value(left) or _is_interval_value(right):
        return _as_interval_value(left) - _as_interval_value(right)
    return left - right


def _mul_value(left, right):
    if _is_interval_value(left) or _is_interval_value(right):
        return _as_interval_value(left) * _as_interval_value(right)
    return left * right


def _div_value(left, right):
    if _is_interval_value(left) or _is_interval_value(right):
        return _as_interval_value(left) / _as_interval_value(right)
    return left / right


def _poly_add(left: list[mp.mpf], right: list[mp.mpf], order: int) -> list[mp.mpf]:
    if not _poly_has_interval(left) and not _poly_has_interval(right):
        return [left[index] + right[index] for index in range(order + 1)]
    return [_add_value(left[index], right[index]) for index in range(order + 1)]


def _poly_sub(left: list[mp.mpf], right: list[mp.mpf], order: int) -> list[mp.mpf]:
    if not _poly_has_interval(left) and not _poly_has_interval(right):
        return [left[index] - right[index] for index in range(order + 1)]
    return [_sub_value(left[index], right[index]) for index in range(order + 1)]


def _poly_neg(poly: list[mp.mpf], order: int) -> list[mp.mpf]:
    return [-poly[index] for index in range(order + 1)]


def _poly_scale(poly: list[mp.mpf], scalar: mp.mpf | float, order: int) -> list[mp.mpf]:
    try:
        value = mp.mpf(scalar)
    except (TypeError, ValueError):
        # mpmath cannot coerce a non-degenerate interval or genuinely complex
        # scalar with mp.mpf; keep it for interval/complex coefficient audits.
        value = scalar
    if not _is_interval_value(value) and not _poly_has_interval(poly):
        return [value * poly[index] for index in range(order + 1)]
    return [_mul_value(poly[index], value) for index in range(order + 1)]


def _poly_mul(left: list[mp.mpf], right: list[mp.mpf], order: int) -> list[mp.mpf]:
    if not _poly_has_interval(left) and not _poly_has_interval(right):
        return [
            sum(left[k] * right[degree - k] for k in range(degree + 1))
            for degree in range(order + 1)
        ]
    result = []
    for degree in range(order + 1):
        total = None
        for k in range(degree + 1):
            term = _mul_value(left[k], right[degree - k])
            total = term if total is None else _add_value(total, term)
        result.append(mp.zero if total is None else total)
    return result


def _poly_inverse(poly: list[mp.mpf], order: int) -> list[mp.mpf]:
    if (_is_interval_value(poly[0]) and _interval_contains_zero(poly[0])) or (not _is_interval_value(poly[0]) and poly[0] == 0):
        raise ZeroDivisionError("Taylor polynomial denominator has zero constant term")
    if not _poly_has_interval(poly):
        quotient = _poly_zero(order)
        quotient[0] = 1 / poly[0]
        for degree in range(1, order + 1):
            total = sum(poly[k] * quotient[degree - k] for k in range(1, degree + 1))
            quotient[degree] = -total / poly[0]
        return quotient
    quotient = _poly_zero(order)
    quotient[0] = _div_value(1, poly[0])
    for degree in range(1, order + 1):
        total = None
        for k in range(1, degree + 1):
            term = _mul_value(poly[k], quotient[degree - k])
            total = term if total is None else _add_value(total, term)
        total = mp.zero if total is None else total
        quotient[degree] = _div_value(-total, poly[0])
    return quotient


def _poly_div(left: list[mp.mpf], right: list[mp.mpf], order: int) -> list[mp.mpf]:
    return _poly_mul(left, _poly_inverse(right, order), order)


def _poly_pow(poly: list[mp.mpf], power: int, order: int) -> list[mp.mpf]:
    result = _poly_constant(1, order)
    for _ in range(power):
        result = _poly_mul(result, poly, order)
    return result


def _scaled_taylor_g_series(
    t: list[mp.mpf],
    p: list[mp.mpf],
    x1: list[mp.mpf],
    x2: list[mp.mpf],
    x3: list[mp.mpf],
    b: mp.mpf,
    order: int,
) -> tuple[list[mp.mpf], list[mp.mpf], list[mp.mpf], list[mp.mpf]]:
    """Return the series for ``t*x'`` in the scaled b-family."""
    p2 = _poly_mul(p, p, order)
    p3 = _poly_mul(p2, p, order)
    p4 = _poly_mul(p2, p2, order)
    t2 = _poly_mul(t, t, order)
    t3 = _poly_mul(t2, t, order)
    x3_2 = _poly_mul(x3, x3, order)
    x3_3 = _poly_mul(x3_2, x3, order)
    x2_x3_2 = _poly_mul(x2, x3_2, order)

    limit0 = _poly_sub(
        _poly_sub(_poly_neg(p, order), _poly_scale(_poly_div(x2_x3_2, p4, order), 3, order), order),
        _poly_div(
            _poly_mul(
                t2,
                _poly_sub(_poly_mul(x1, x2, order), _poly_scale(_poly_mul(x3, p2, order), mp.mpf(1) / 6, order), order),
                order,
            ),
            _poly_scale(p2, 4, order),
            order,
        ),
        order,
    )
    limit1 = _poly_add(
        _poly_add(_poly_scale(x1, -4, order), _poly_div(x3_3, p3, order), order),
        _poly_div(
            _poly_mul(
                t2,
                _poly_add(
                    _poly_mul(_poly_mul(x1, x1, order), x2, order),
                    _poly_scale(_poly_mul(_poly_mul(x1, x3, order), p2, order), mp.mpf("0.5"), order),
                    order,
                ),
                order,
            ),
            _poly_scale(p3, 2, order),
            order,
        ),
        order,
    )
    limit2 = _poly_div(
        _poly_mul(
            t2,
            _poly_add(
                _poly_add(
                    _poly_scale(_poly_mul(_poly_mul(p2, x2, order), x3, order), -mp.mpf(1) / 4, order),
                    _poly_scale(_poly_mul(_poly_mul(x1, x2, order), x2, order), -mp.mpf("0.5"), order),
                    order,
                ),
                _poly_scale(_poly_mul(t2, _poly_mul(p3, p3, order), order), mp.mpf(1) / 216, order),
                order,
            ),
            order,
        ),
        p3,
        order,
    )
    limit3 = _poly_add(
        _poly_add(_poly_scale(x3, -2, order), _poly_scale(p, 6, order), order),
        _poly_div(
            _poly_mul(
                t2,
                _poly_sub(
                    _poly_sub(
                        _poly_mul(_poly_mul(x1, x2, order), x3, order),
                        _poly_scale(_poly_mul(x3_2, p2, order), mp.mpf(1) / 6, order),
                        order,
                    ),
                    _poly_scale(_poly_mul(_poly_mul(t2, x1, order), p4, order), mp.mpf(1) / 18, order),
                    order,
                ),
                order,
            ),
            _poly_scale(p3, 2, order),
            order,
        ),
        order,
    )

    i1 = _poly_sub(
        _poly_mul(
            t,
            _poly_sub(
                _poly_scale(_poly_mul(_poly_mul(x3, x1, order), x2, order), 2, order),
                _poly_scale(_poly_mul(p2, x3_2, order), mp.mpf("0.5"), order),
                order,
            ),
            order,
        ),
        _poly_scale(_poly_mul(_poly_mul(t3, x1, order), p4, order), mp.mpf(1) / 18, order),
        order,
    )
    i2 = _poly_add(
        _poly_scale(_poly_mul(t, x3_3, order), -2, order),
        _poly_scale(_poly_mul(_poly_mul(_poly_mul(t3, x1, order), p2, order), x3, order), -mp.mpf(2) / 3, order),
        order,
    )
    i3 = _poly_scale(_poly_mul(_poly_mul(t3, x1, order), x3_2, order), -2, order)

    r1 = (
        _poly_scale(_poly_div(i1, p4, order), -mp.mpf(3) / 2, order),
        _poly_scale(_poly_div(_poly_mul(_poly_mul(t, x1, order), x3_2, order), p3, order), mp.mpf(3) / 2, order),
        _poly_div(
            _poly_mul(
                t,
                _poly_sub(
                    _poly_scale(_poly_mul(_poly_mul(t2, p4, order), x3, order), mp.mpf(1) / 12, order),
                    _poly_scale(_poly_mul(x2, x3_2, order), mp.mpf("1.5"), order),
                    order,
                ),
                order,
            ),
            p3,
            order,
        ),
        _poly_div(
            _poly_mul(
                t,
                _poly_sub(
                    _poly_neg(x3_3, order),
                    _poly_scale(_poly_mul(_poly_mul(_poly_mul(t2, x1, order), p2, order), x3, order), mp.mpf(2) / 3, order),
                    order,
                ),
                order,
            ),
            _poly_scale(p3, 2, order),
            order,
        ),
    )
    r2 = (
        _poly_scale(_poly_div(i2, p4, order), -mp.mpf(3) / 2, order),
        _poly_zero(order),
        _poly_div(_poly_mul(t, _poly_scale(_poly_mul(_poly_mul(t2, p2, order), x3_2, order), mp.mpf("0.5"), order), order), p3, order),
        _poly_div(
            _poly_mul(t, _poly_scale(_poly_mul(_poly_mul(t2, x1, order), x3_2, order), -2, order), order),
            _poly_scale(p3, 2, order),
            order,
        ),
    )
    r3 = (
        _poly_scale(_poly_div(i3, p4, order), -mp.mpf(3) / 2, order),
        _poly_zero(order),
        _poly_div(_poly_mul(t, _poly_mul(t2, x3_3, order), order), p3, order),
        _poly_zero(order),
    )
    return tuple(
        _poly_add(
            limit,
            _poly_mul(t, _poly_add(_poly_add(_poly_scale(r1[index], b, order), _poly_scale(r2[index], b * b, order), order), _poly_scale(r3[index], b**3, order), order), order),
            order,
        )
        for index, limit in enumerate((limit0, limit1, limit2, limit3))
    )


def _scaled_taylor_coefficients_for_b_value(order: int, b_value) -> tuple[tuple, tuple, tuple, tuple]:
    """Return Taylor coefficients for a prepared real or complex ``b`` value."""
    coefficients = [_poly_zero(order) for _ in range(4)]
    for index, value in enumerate((1, mp.mpf(27) / 4, -mp.mpf(1) / 27, 3)):
        coefficients[index][0] = mp.mpf(value)
    t_series = _poly_variable(order)

    def state_with_trial(degree: int, trial: tuple) -> tuple[list, ...]:
        state = []
        for component in range(4):
            row = list(coefficients[component])
            row[degree] = trial[component]
            state.append(row)
        return tuple(state)

    for degree in range(1, order + 1):
        zero_trial = (mp.zero, mp.zero, mp.zero, mp.zero)
        g_zero = _scaled_taylor_g_series(t_series, *state_with_trial(degree, zero_trial), b_value, order)
        base_residual = [degree * zero_trial[index] - g_zero[index][degree] for index in range(4)]
        matrix = mp.matrix(4, 4)
        for trial_index in range(4):
            trial = [mp.zero, mp.zero, mp.zero, mp.zero]
            trial[trial_index] = mp.one
            g_trial = _scaled_taylor_g_series(t_series, *state_with_trial(degree, tuple(trial)), b_value, order)
            residual = [degree * trial[index] - g_trial[index][degree] for index in range(4)]
            for row_index in range(4):
                matrix[row_index, trial_index] = residual[row_index] - base_residual[row_index]
        solution = mp.lu_solve(matrix, mp.matrix([-value for value in base_residual]))
        for component in range(4):
            coefficients[component][degree] = +solution[component]
    return tuple(tuple(row) for row in coefficients)


def scaled_taylor_coefficients(
    order: int,
    b: float = 0.0,
    working_dps: int = 80,
) -> tuple[tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...]]:
    """Return high-order Taylor coefficients for the smooth scaled IVP.

    The coefficients solve ``t*x' = G(t,x,b)`` recursively.  This is intended
    for proof diagnostics near the singular left endpoint; it avoids the raw
    zero-jet start used by older lightweight tail probes.
    """
    if order < 0:
        raise ValueError("order must be nonnegative")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    with mp.workdps(working_dps):
        return _scaled_taylor_coefficients_for_b_value(order, mp.mpf(b))  # type: ignore[return-value]


def complex_scaled_taylor_coefficients(
    order: int,
    b,
    working_dps: int = 80,
) -> tuple[tuple, tuple, tuple, tuple]:
    """Return Taylor coefficients for a complex ``b`` diagnostic sample."""
    if order < 0:
        raise ValueError("order must be nonnegative")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    with mp.workdps(working_dps):
        return _scaled_taylor_coefficients_for_b_value(order, mp.mpc(b))


def _interval_contains_zero(value) -> bool:
    """Return whether an mpmath interval contains zero."""
    return float(value.a) <= 0.0 <= float(value.b)


def _interval_abs_lower(value) -> float:
    """Return a lower bound for the absolute value over an interval."""
    lower = float(value.a)
    upper = float(value.b)
    if lower <= 0.0 <= upper:
        return 0.0
    return min(abs(lower), abs(upper))


def _interval_midpoint(value):
    """Return the midpoint of a point or interval scalar."""
    if _is_interval_value(value):
        return mp.mpf((float(value.a) + float(value.b)) / 2)
    return mp.mpf(value)


def _interval_radius(value):
    """Return the radius of a point or interval scalar."""
    if _is_interval_value(value):
        return mp.mpf((float(value.b) - float(value.a)) / 2)
    return mp.zero


def _preconditioned_interval_linear_solve(matrix: list[list], rhs: list) -> list:
    """Enclose an interval solve using the midpoint inverse as preconditioner."""
    size = len(rhs)
    from mpmath import iv

    midpoint_matrix = mp.matrix(size, size)
    rhs_midpoint = mp.matrix(size, 1)
    matrix_radius = [[mp.zero for _ in range(size)] for _ in range(size)]
    rhs_radius = [mp.zero for _ in range(size)]
    for row in range(size):
        rhs_midpoint[row] = _interval_midpoint(rhs[row])
        rhs_radius[row] = _interval_radius(rhs[row])
        for col in range(size):
            midpoint_matrix[row, col] = _interval_midpoint(matrix[row][col])
            matrix_radius[row][col] = _interval_radius(matrix[row][col])
    midpoint_inverse = mp.inverse(midpoint_matrix)
    point_solution = midpoint_inverse * rhs_midpoint

    abs_inverse = [[abs(midpoint_inverse[row, col]) for col in range(size)] for row in range(size)]
    abs_solution = [abs(point_solution[row]) for row in range(size)]
    residual_radius = []
    for row in range(size):
        total = mp.zero
        for col in range(size):
            total += matrix_radius[row][col] * abs_solution[col]
        residual_radius.append(rhs_radius[row] + total)

    base_radius = []
    for row in range(size):
        total = mp.zero
        for col in range(size):
            total += abs_inverse[row][col] * residual_radius[col]
        base_radius.append(total)

    contraction = mp.matrix(size, size)
    for row in range(size):
        for col in range(size):
            total = mp.zero
            for k in range(size):
                total += abs_inverse[row][k] * matrix_radius[k][col]
            contraction[row, col] = total
    identity_minus = mp.eye(size) - contraction
    row_sums = [sum(contraction[row, col] for col in range(size)) for row in range(size)]
    if any(row_sum >= 1 for row_sum in row_sums):
        raise ZeroDivisionError("interval Taylor midpoint preconditioner is not contractive")
    error_radius = mp.lu_solve(identity_minus, mp.matrix(base_radius))
    return [
        iv.mpf([point_solution[index] - error_radius[index], point_solution[index] + error_radius[index]])
        for index in range(size)
    ]


def _interval_linear_solve(matrix: list[list], rhs: list) -> list:
    """Solve a small interval linear system by interval Gaussian elimination."""
    size = len(rhs)
    original_matrix = [[matrix[row][col] for col in range(size)] for row in range(size)]
    original_rhs = [rhs[row] for row in range(size)]
    rows = [[matrix[row][col] for col in range(size)] for row in range(size)]
    values = [rhs[row] for row in range(size)]
    try:
        for col in range(size):
            pivot_row = None
            pivot_abs = -math.inf
            for row in range(col, size):
                if not _interval_contains_zero(rows[row][col]):
                    abs_lower = _interval_abs_lower(rows[row][col])
                    if abs_lower > pivot_abs:
                        pivot_abs = abs_lower
                        pivot_row = row
            if pivot_row is None:
                raise ZeroDivisionError("interval Taylor solve has no nonzero pivot")
            if pivot_row != col:
                rows[col], rows[pivot_row] = rows[pivot_row], rows[col]
                values[col], values[pivot_row] = values[pivot_row], values[col]
            pivot = rows[col][col]
            rows[col] = [entry / pivot for entry in rows[col]]
            values[col] = values[col] / pivot
            for row in range(size):
                if row == col:
                    continue
                factor = rows[row][col]
                rows[row] = [
                    rows[row][entry_col] - factor * rows[col][entry_col]
                    for entry_col in range(size)
                ]
                values[row] = values[row] - factor * values[col]
        return values
    except ZeroDivisionError:
        return _preconditioned_interval_linear_solve(original_matrix, original_rhs)


def interval_scaled_taylor_coefficients(
    order: int,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    working_dps: int = 80,
    b_range: tuple[float, float] | None = None,
) -> tuple[tuple, tuple, tuple, tuple]:
    """Return interval Taylor coefficients for all ``|b| <= 1/A``.

    This is an inclusion audit for finite Taylor coefficients.  It is not by
    itself a proof of the infinite tail bound, but it lets us replace sampled
    b-values by an interval b-family for finite same-parity ratio checks.
    """
    if order < 0:
        raise ValueError("order must be nonnegative")
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    beta = 1.0 / candidate_a
    if b_range is None:
        b_range = (-beta, beta)
    if b_range[0] > b_range[1]:
        raise ValueError("b_range must be ordered")
    if b_range[0] < -beta or b_range[1] > beta:
        raise ValueError("b_range must stay inside [-1/A,1/A]")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    from mpmath import iv

    with mp.workdps(working_dps):
        b_value = iv.mpf([b_range[0], b_range[1]])
        zero = iv.mpf([0.0, 0.0])
        one = iv.mpf([1.0, 1.0])
        coefficients = [[zero for _ in range(order + 1)] for _ in range(4)]
        for index, value in enumerate((1.0, 27.0 / 4.0, -1.0 / 27.0, 3.0)):
            coefficients[index][0] = iv.mpf([value, value])
        t_series = [zero for _ in range(order + 1)]
        if order >= 1:
            t_series[1] = one

        def state_with_trial(degree: int, trial: tuple) -> tuple[list, ...]:
            state = []
            for component in range(4):
                row = list(coefficients[component])
                row[degree] = trial[component]
                state.append(row)
            return tuple(state)

        for degree in range(1, order + 1):
            zero_trial = (zero, zero, zero, zero)
            g_zero = _scaled_taylor_g_series(t_series, *state_with_trial(degree, zero_trial), b_value, order)
            base_residual = [degree * zero_trial[index] - g_zero[index][degree] for index in range(4)]
            matrix = [[zero for _ in range(4)] for _ in range(4)]
            for trial_index in range(4):
                trial = [zero, zero, zero, zero]
                trial[trial_index] = one
                g_trial = _scaled_taylor_g_series(t_series, *state_with_trial(degree, tuple(trial)), b_value, order)
                residual = [degree * trial[index] - g_trial[index][degree] for index in range(4)]
                for row_index in range(4):
                    matrix[row_index][trial_index] = residual[row_index] - base_residual[row_index]
            solution = _interval_linear_solve(matrix, [-value for value in base_residual])
            for component in range(4):
                coefficients[component][degree] = solution[component]
        return tuple(tuple(row) for row in coefficients)  # type: ignore[return-value]


def evaluate_scaled_taylor_coefficients(
    coefficients: tuple[tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...]],
    time: float,
) -> tuple[float, float, float, float]:
    """Evaluate one scaled Taylor coefficient block at ``time``."""
    time_value = mp.mpf(time)
    values = []
    for row in coefficients:
        value = mp.zero
        for coeff in reversed(row):
            value = value * time_value + coeff
        values.append(float(value))
    return tuple(values)  # type: ignore[return-value]


def high_order_scaled_taylor_seed(
    time: float,
    b: float,
    order: int = 20,
    working_dps: int = 80,
) -> tuple[float, float, float, float]:
    """Return a high-order Taylor seed for the smooth scaled IVP."""
    return evaluate_scaled_taylor_coefficients(scaled_taylor_coefficients(order, b, working_dps), time)


def high_order_scaled_taylor_state_at_p(
    target_p: float,
    b: float,
    order: int = 40,
    working_dps: int = 100,
    time_low: float = 0.0,
    time_high: float = 3.0,
    iterations: int = 80,
) -> tuple[float, float, float, float]:
    """Return ``(t,x1,x2,x3)`` on a high-order Taylor ``p=target_p`` slice."""
    coefficients = scaled_taylor_coefficients(order, b, working_dps)
    return high_order_scaled_taylor_state_at_p_from_coefficients(
        target_p,
        coefficients,
        time_low=time_low,
        time_high=time_high,
        iterations=iterations,
    )


def high_order_scaled_taylor_state_at_p_from_coefficients(
    target_p: float,
    coefficients: tuple[tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...]],
    time_low: float = 0.0,
    time_high: float = 3.0,
    iterations: int = 80,
) -> tuple[float, float, float, float]:
    """Return ``(t,x1,x2,x3)`` on a Taylor ``p=target_p`` slice."""
    if not (0.0 < target_p < 1.0):
        raise ValueError("target_p must lie between 0 and 1")
    if not (0.0 <= time_low < time_high):
        raise ValueError("time bracket must be ordered and nonnegative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    def p_at(time: float) -> float:
        return evaluate_scaled_taylor_coefficients(coefficients, time)[0]

    low_value = p_at(time_low)
    high_value = p_at(time_high)
    if low_value < target_p or high_value > target_p:
        raise ValueError("time bracket does not contain the requested p slice")
    low = time_low
    high = time_high
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if p_at(mid) >= target_p:
            low = mid
        else:
            high = mid
    time = 0.5 * (low + high)
    _p, x1, x2, x3 = evaluate_scaled_taylor_coefficients(coefficients, time)
    return (time, x1, x2, x3)


def _complex_taylor_state_at_p_from_coefficients(
    target_p: float,
    coefficients: tuple[tuple, tuple, tuple, tuple],
    initial_time: float,
    iterations: int = 30,
    tolerance: float = 1e-40,
) -> tuple[mp.mpc, mp.mpc, mp.mpc, mp.mpc]:
    """Return the complex Taylor p-slice state by Newton continuation."""
    if not (0.0 < target_p < 1.0):
        raise ValueError("target_p must lie between 0 and 1")
    if initial_time <= 0.0:
        raise ValueError("initial_time must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    time = mp.mpc(initial_time)
    target = mp.mpf(target_p)
    for _ in range(iterations):
        value = _evaluate_polynomial_complex(coefficients[0], time) - target
        derivative = _evaluate_polynomial_derivative_complex(coefficients[0], time)
        if derivative == 0:
            raise ZeroDivisionError("complex p-slice Newton derivative vanished")
        time -= value / derivative
        if abs(value) <= tolerance:
            break
    residual = _evaluate_polynomial_complex(coefficients[0], time) - target
    if abs(residual) > tolerance:
        raise ArithmeticError(f"complex p-slice Newton did not converge; residual={residual}")
    values = tuple(_evaluate_polynomial_complex(component, time) for component in coefficients)
    return (time, values[1], values[2], values[3])


def _taylor_term_magnitudes(
    coefficients: tuple[tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...]],
    time: float,
) -> list[list[float]]:
    """Return absolute Taylor term magnitudes ``|c_n t^n|`` by component."""
    time_value = mp.mpf(time)
    return [
        [abs(float(coefficient * (time_value ** degree))) for degree, coefficient in enumerate(row)]
        for row in coefficients
    ]


def _geometric_tail_estimate(
    terms: list[float],
    tail_start: int,
    ratio_start: int,
    ratio_bound: float | None = None,
) -> tuple[float, float, float, bool, tuple[int, int, float] | None]:
    """Return a parity-aware formal geometric tail estimate and max ratio."""
    if tail_start + 1 >= len(terms):
        raise ValueError("tail_start must leave at least one observed tail term")
    if ratio_bound is not None and not (0.0 <= ratio_bound < 1.0):
        raise ValueError("ratio_bound must lie in [0,1)")
    total_tail = 0.0
    max_ratio = 0.0
    max_ratio_witness = None
    ratios_inside_bound = True
    for parity in (0, 1):
        first_tail = tail_start + 1
        if first_tail % 2 != parity:
            first_tail += 1
        if first_tail >= len(terms):
            continue
        ratio = 0.0
        first_ratio = max(0, ratio_start)
        if first_ratio % 2 != parity:
            first_ratio += 1
        for degree in range(first_ratio, len(terms) - 2, 2):
            if terms[degree] > 0.0:
                observed = terms[degree + 2] / terms[degree]
                ratio = max(ratio, observed)
                if observed > max_ratio:
                    max_ratio_witness = (degree, degree + 2, observed)
        max_ratio = max(max_ratio, ratio)
        if ratio_bound is not None:
            ratios_inside_bound = ratios_inside_bound and ratio <= ratio_bound
        tail_ratio = ratio if ratio_bound is None else ratio_bound
        if tail_ratio >= 1.0:
            total_tail = math.inf
        elif math.isfinite(total_tail):
            total_tail += terms[first_tail] / (1.0 - tail_ratio)
    return (
        total_tail,
        max_ratio,
        (max_ratio if ratio_bound is None else ratio_bound),
        ratios_inside_bound,
        max_ratio_witness,
    )


def _c_tail_bound_at_p(
    target_p: float,
    state: tuple[float, float, float, float],
    tail_bounds_4d: tuple[float, float, float, float],
) -> float:
    """Return a first-order-plus-product bound for the induced C tail."""
    _time, x1, x2, _x3 = state
    _dt, dx1, dx2, dx3 = tail_bounds_4d
    return abs(x2) * dx1 + abs(x1) * dx2 + dx1 * dx2 + target_p * target_p * dx3 / 6.0


def _symmetric_b_grid(candidate_a: float, sample_count: int = 3) -> tuple[float, ...]:
    """Return an odd symmetric grid in ``b`` containing endpoints and zero."""
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if sample_count < 3 or sample_count % 2 == 0:
        raise ValueError("sample_count must be an odd integer at least 3")
    beta = 1.0 / candidate_a
    return tuple(-beta + 2 * beta * index / (sample_count - 1) for index in range(sample_count))


def taylor_p_slice_convergence_audit(
    target_p: float = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_START,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    low_order: int = 30,
    high_order: int = 40,
    working_dps: int = 80,
    radius0: tuple[float, float, float, float, float] = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_RADIUS,
) -> dict:
    """Compare two high-order Taylor p-slices against the p-tube start radius.

    This is a proof-building audit, not a Taylor remainder theorem.  Passing
    means that the observed change from ``low_order`` to ``high_order`` at the
    requested p-slice is smaller than the start radius used by the sampled
    carried-C p-tube.  The remaining mathematical obligation is to replace this
    observed convergence check by a rigorous tail/remainder estimate.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if not (0.0 < target_p < 1.0):
        raise ValueError("target_p must lie between 0 and 1")
    if low_order < 0 or high_order <= low_order:
        raise ValueError("orders must satisfy 0 <= low_order < high_order")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    if len(radius0) != 5 or any(value <= 0.0 for value in radius0):
        raise ValueError("radius0 must contain five positive values")

    beta = 1.0 / candidate_a
    rows = []
    max_order_difference = [0.0 for _ in range(5)]
    high_samples = []
    for b in (-beta, 0.0, beta):
        low_state = high_order_scaled_taylor_state_at_p(
            target_p,
            b,
            order=low_order,
            working_dps=working_dps,
        )
        high_state = high_order_scaled_taylor_state_at_p(
            target_p,
            b,
            order=high_order,
            working_dps=working_dps,
        )
        low_state_5d = (*low_state, cancellation_c_value(target_p, low_state))
        high_state_5d = (*high_state, cancellation_c_value(target_p, high_state))
        order_difference = tuple(
            abs(high_state_5d[index] - low_state_5d[index])
            for index in range(5)
        )
        max_order_difference = [
            max(max_order_difference[index], order_difference[index])
            for index in range(5)
        ]
        high_samples.append(high_state_5d)
        rows.append(
            {
                "b": b,
                "low_order_state_5d": list(low_state_5d),
                "high_order_state_5d": list(high_state_5d),
                "order_difference_5d": list(order_difference),
            }
        )

    sample_low = [
        min(sample[index] for sample in high_samples)
        for index in range(5)
    ]
    sample_high = [
        max(sample[index] for sample in high_samples)
        for index in range(5)
    ]
    start_box_low = [
        sample_low[index] - radius0[index]
        for index in range(5)
    ]
    start_box_high = [
        sample_high[index] + radius0[index]
        for index in range(5)
    ]
    radius_ratio = [
        max_order_difference[index] / radius0[index]
        for index in range(5)
    ]
    observed_inside_radius = all(
        max_order_difference[index] < radius0[index]
        for index in range(5)
    )
    return {
        "status": "observed_convergence_inside_start_radius"
        if observed_inside_radius
        else "observed_convergence_exceeds_start_radius",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "target_p": target_p,
        "low_order": low_order,
        "high_order": high_order,
        "working_dps": working_dps,
        "radius0": list(radius0),
        "rows": rows,
        "high_order_sample_hull_5d": {"low": sample_low, "high": sample_high},
        "sampled_carried_c_start_box_5d": {"low": start_box_low, "high": start_box_high},
        "max_order_difference_5d": max_order_difference,
        "max_order_difference_over_radius": radius_ratio,
        "remaining_obligation": (
            "replace the observed low/high Taylor-order difference by a rigorous "
            "Taylor remainder bound at the p-slice"
        ),
    }


def taylor_p_slice_tail_ratio_audit(
    target_p: float = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_START,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    order: int = 60,
    tail_start: int = 50,
    ratio_start: int = 45,
    ratio_bound: float | None = None,
    b_sample_count: int = 3,
    working_dps: int = 90,
    radius0: tuple[float, float, float, float, float] = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_RADIUS,
) -> dict:
    """Estimate whether a geometric Taylor tail bound would fit the p-tube.

    This is not a rigorous Taylor remainder proof.  It computes the observed
    late-term ratios at the fixed Taylor p-slice and asks whether the resulting
    formal geometric tail estimate would fit inside the existing carried-C
    start radius.  A proof still has to replace the observed ratio by a genuine
    majorant/Cauchy estimate.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if not (0.0 < target_p < 1.0):
        raise ValueError("target_p must lie between 0 and 1")
    if order <= 0:
        raise ValueError("order must be positive")
    if not (0 <= ratio_start <= tail_start < order):
        raise ValueError("orders must satisfy 0 <= ratio_start <= tail_start < order")
    if ratio_bound is not None and not (0.0 <= ratio_bound < 1.0):
        raise ValueError("ratio_bound must lie in [0,1)")
    b_values = _symmetric_b_grid(candidate_a, b_sample_count)
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    if len(radius0) != 5 or any(value <= 0.0 for value in radius0):
        raise ValueError("radius0 must contain five positive values")

    beta = 1.0 / candidate_a
    rows = []
    max_tail_4d = [0.0 for _ in range(4)]
    max_ratio_4d = [0.0 for _ in range(4)]
    max_ratio_witness_4d = [None for _ in range(4)]
    max_used_ratio_4d = [0.0 for _ in range(4)]
    observed_ratios_inside_bound = True
    max_c_tail = 0.0
    max_time_shift_bound = 0.0
    min_abs_p_prime = math.inf
    for b in b_values:
        coefficients = scaled_taylor_coefficients(order, b, working_dps)
        state = high_order_scaled_taylor_state_at_p_from_coefficients(target_p, coefficients)
        terms = _taylor_term_magnitudes(coefficients, state[0])
        component_tails = []
        component_ratios = []
        component_ratio_witnesses = []
        component_used_ratios = []
        component_inside_bound = []
        for component in range(4):
            tail_estimate, ratio, used_ratio, inside_bound, witness = _geometric_tail_estimate(
                terms[component],
                tail_start,
                ratio_start,
                ratio_bound=ratio_bound,
            )
            component_tails.append(tail_estimate)
            component_ratios.append(ratio)
            component_ratio_witnesses.append(
                None
                if witness is None
                else {"degree": witness[0], "next_degree": witness[1], "ratio": witness[2]}
            )
            component_used_ratios.append(used_ratio)
            component_inside_bound.append(inside_bound)
            max_tail_4d[component] = max(max_tail_4d[component], tail_estimate)
            if ratio > max_ratio_4d[component]:
                max_ratio_4d[component] = ratio
                max_ratio_witness_4d[component] = (
                    None
                    if witness is None
                    else {"b": b, "degree": witness[0], "next_degree": witness[1], "ratio": witness[2]}
                )
            max_used_ratio_4d[component] = max(max_used_ratio_4d[component], used_ratio)
            observed_ratios_inside_bound = observed_ratios_inside_bound and inside_bound
        c_tail = _c_tail_bound_at_p(target_p, state, tuple(component_tails))
        max_c_tail = max(max_c_tail, c_tail)
        rhs = scaled_rhs_with_b(state[0], (target_p, state[1], state[2], state[3]), b)
        abs_p_prime = abs(rhs[0])
        min_abs_p_prime = min(min_abs_p_prime, abs_p_prime)
        time_shift_bound = math.inf if abs_p_prime == 0.0 else component_tails[0] / abs_p_prime
        max_time_shift_bound = max(max_time_shift_bound, time_shift_bound)
        rows.append(
            {
                "b": b,
                "state_4d": list(state),
                "p_prime": rhs[0],
                "component_tail_estimate_4d": component_tails,
                "component_observed_ratio_4d": component_ratios,
                "component_observed_ratio_witness_4d": component_ratio_witnesses,
                "component_used_ratio_4d": component_used_ratios,
                "component_observed_ratio_inside_bound_4d": component_inside_bound,
                "c_tail_estimate": c_tail,
                "time_shift_bound_from_p_tail": time_shift_bound,
                "tail_start_terms_4d": [terms[component][tail_start + 1] for component in range(4)],
            }
        )

    max_tail_5d = [*max_tail_4d, max_c_tail]
    ratio_to_radius = [
        max_tail_5d[index] / radius0[index]
        for index in range(5)
    ]
    inside_radius = all(value < radius0[index] for index, value in enumerate(max_tail_5d))
    return {
        "status": "formal_geometric_tail_inside_start_radius"
        if inside_radius
        else "formal_geometric_tail_exceeds_start_radius",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "target_p": target_p,
        "order": order,
        "tail_start": tail_start,
        "ratio_start": ratio_start,
        "ratio_bound": ratio_bound,
        "b_sample_count": b_sample_count,
        "b_samples": list(b_values),
        "working_dps": working_dps,
        "radius0": list(radius0),
        "rows": rows,
        "max_tail_estimate_5d": max_tail_5d,
        "max_observed_ratio_4d": max_ratio_4d,
        "max_observed_ratio_witness_4d": max_ratio_witness_4d,
        "max_used_ratio_4d": max_used_ratio_4d,
        "observed_ratios_inside_bound": observed_ratios_inside_bound,
        "max_tail_estimate_over_radius": ratio_to_radius,
        "min_abs_p_prime": min_abs_p_prime,
        "max_time_shift_bound_from_p_tail": max_time_shift_bound,
        "remaining_obligation": (
            "replace the observed late-term ratio by a rigorous geometric "
            "majorant or Cauchy tail estimate, and include the resulting p-slice "
            "event-time shift in the start box"
        ),
    }


def _cauchy_tail_from_bound(cauchy_bound: float, time: float, analytic_radius: float, tail_start: int) -> float:
    """Return the ordinary Cauchy tail after ``tail_start`` at ``time``."""
    if analytic_radius <= time:
        return math.inf
    ratio = time / analytic_radius
    return cauchy_bound * (ratio ** (tail_start + 1)) / (1.0 - ratio)


def _observed_cauchy_floor(coefficients: tuple[mp.mpf, ...], analytic_radius: float) -> float:
    """Return the least Cauchy circle bound forced by observed coefficients."""
    radius = mp.mpf(analytic_radius)
    return float(max(abs(coefficient) * (radius ** degree) for degree, coefficient in enumerate(coefficients)))


def _evaluate_polynomial_complex(coefficients: tuple[mp.mpf, ...], point) -> mp.mpc:
    """Evaluate a coefficient row at a real or complex point."""
    value = mp.zero
    for coefficient in reversed(coefficients):
        value = value * point + coefficient
    return value


def _evaluate_polynomial_derivative_complex(coefficients: tuple[mp.mpf, ...], point) -> mp.mpc:
    """Evaluate the derivative of one coefficient row at a real or complex point."""
    value = mp.zero
    for degree in range(len(coefficients) - 1, 0, -1):
        value = value * point + degree * coefficients[degree]
    return value


def _circle_abs_range_for_polynomial(
    coefficients: tuple[mp.mpf, ...],
    radius: float,
    sample_count: int,
) -> dict:
    """Return sampled and derivative-certified absolute bounds on a circle."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    radius_value = mp.mpf(radius)
    min_abs = mp.inf
    max_abs = mp.zero
    for index in range(sample_count):
        theta = 2 * mp.pi * index / sample_count
        point = mp.mpc(radius_value * mp.cos(theta), radius_value * mp.sin(theta))
        value = mp.zero
        power = mp.one
        for coefficient in coefficients:
            value += coefficient * power
            power *= point
        magnitude = abs(value)
        min_abs = min(min_abs, magnitude)
        max_abs = max(max_abs, magnitude)
    angular_derivative_bound = sum(
        degree * abs(coefficient) * (radius_value ** degree)
        for degree, coefficient in enumerate(coefficients)
    )
    half_spacing = mp.pi / sample_count
    sampling_loss = angular_derivative_bound * half_spacing
    return {
        "sample_min_abs": float(min_abs),
        "sample_max_abs": float(max_abs),
        "angular_derivative_bound": float(angular_derivative_bound),
        "sample_spacing_loss": float(sampling_loss),
        "certified_min_abs_lower": float(min_abs - sampling_loss),
    }


def _empty_circle_abs_certificate() -> dict:
    """Return a neutral accumulator for circle absolute-value certificates."""
    return {
        "sample_min_abs": math.inf,
        "sample_max_abs": 0.0,
        "angular_derivative_bound": 0.0,
        "sample_spacing_loss": 0.0,
        "certified_min_abs_lower": math.inf,
    }


def _merge_circle_abs_certificate(left: dict, right: dict) -> dict:
    """Merge sampled circle certificates over several parameter samples."""
    return {
        "sample_min_abs": min(left["sample_min_abs"], right["sample_min_abs"]),
        "sample_max_abs": max(left["sample_max_abs"], right["sample_max_abs"]),
        "angular_derivative_bound": max(left["angular_derivative_bound"], right["angular_derivative_bound"]),
        "sample_spacing_loss": max(left["sample_spacing_loss"], right["sample_spacing_loss"]),
        "certified_min_abs_lower": min(left["certified_min_abs_lower"], right["certified_min_abs_lower"]),
    }


def taylor_p_slice_cauchy_budget_audit(
    target_p: float = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_START,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    order: int = 70,
    tail_start: int = 50,
    b_sample_count: int = 3,
    working_dps: int = 90,
    analytic_radii: tuple[float, ...] = DEFAULT_TAYLOR_P_SLICE_CAUCHY_RADII,
    circle_sample_count: int = 360,
    circle_tail_ratio_bound: float | None = 0.95,
    radius0: tuple[float, float, float, float, float] = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_RADIUS,
) -> dict:
    """Report the Cauchy-bound budget needed to close the p-slice tail.

    This is a feasibility diagnostic, not a proof.  For each proposed analytic
    radius R it computes the smallest circle bound already forced by the known
    Taylor coefficients, then asks whether a Cauchy tail estimate with a bound
    of that size would fit into the existing carried-C p-tube start radius.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if not (0.0 < target_p < 1.0):
        raise ValueError("target_p must lie between 0 and 1")
    if order <= 0:
        raise ValueError("order must be positive")
    if not (0 <= tail_start < order):
        raise ValueError("orders must satisfy 0 <= tail_start < order")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    if len(radius0) != 5 or any(value <= 0.0 for value in radius0):
        raise ValueError("radius0 must contain five positive values")
    if not analytic_radii or any(value <= 0.0 for value in analytic_radii):
        raise ValueError("analytic_radii must contain positive values")
    if circle_sample_count <= 0:
        raise ValueError("circle_sample_count must be positive")
    if circle_tail_ratio_bound is not None and not (0.0 <= circle_tail_ratio_bound < 1.0):
        raise ValueError("circle_tail_ratio_bound must lie in [0,1)")

    beta = 1.0 / candidate_a
    limiting_crossing = first_scaled_crossing("limit", step_size=DEFAULT_STEP)
    terminal_time = limiting_crossing.time if limiting_crossing.status == "crossed" else math.inf
    b_values = _symmetric_b_grid(candidate_a, b_sample_count)
    coefficient_samples = []
    for b in b_values:
        coefficients = scaled_taylor_coefficients(order, b, working_dps)
        state = high_order_scaled_taylor_state_at_p_from_coefficients(target_p, coefficients)
        coefficient_samples.append((b, coefficients, state))

    radius_rows = []
    for analytic_radius in analytic_radii:
        rows = []
        below_real_terminal = analytic_radius < terminal_time
        max_floor_tail_4d = [0.0 for _ in range(4)]
        max_observed_cauchy_floor_4d = [0.0 for _ in range(4)]
        min_allowed_cauchy_bound_4d = [math.inf for _ in range(4)]
        min_cauchy_headroom_4d = [math.inf for _ in range(4)]
        p_circle_abs_partial = _empty_circle_abs_certificate()
        max_p_circle_tail_estimate = 0.0
        max_p_circle_observed_tail_ratio = 0.0
        p_circle_ratio_witness = None
        p_circle_tail_inside_ratio_bound = True
        max_c_tail = 0.0
        valid_radius = True
        for b, coefficients, state in coefficient_samples:
            time = state[0]
            if analytic_radius <= time:
                valid_radius = False
                rows.append(
                    {
                        "b": b,
                        "state_4d": list(state),
                        "status": "radius_not_larger_than_slice_time",
                    }
                )
                continue
            ratio = time / analytic_radius
            component_cauchy_floor = [
                _observed_cauchy_floor(coefficients[component], analytic_radius)
                for component in range(4)
            ]
            p_circle = _circle_abs_range_for_polynomial(
                coefficients[0],
                analytic_radius,
                circle_sample_count,
            )
            p_circle_terms = [
                abs(float(coefficient * (mp.mpf(analytic_radius) ** degree)))
                for degree, coefficient in enumerate(coefficients[0])
            ]
            (
                p_circle_tail_estimate,
                p_circle_ratio,
                _p_circle_used_ratio,
                p_circle_inside_bound,
                p_circle_witness,
            ) = _geometric_tail_estimate(
                p_circle_terms,
                tail_start,
                tail_start,
                ratio_bound=circle_tail_ratio_bound,
            )
            p_circle_abs_partial = _merge_circle_abs_certificate(p_circle_abs_partial, p_circle)
            max_p_circle_tail_estimate = max(max_p_circle_tail_estimate, p_circle_tail_estimate)
            if p_circle_ratio > max_p_circle_observed_tail_ratio:
                max_p_circle_observed_tail_ratio = p_circle_ratio
                p_circle_ratio_witness = (
                    None
                    if p_circle_witness is None
                    else {"b": b, "degree": p_circle_witness[0], "next_degree": p_circle_witness[1], "ratio": p_circle_witness[2]}
                )
            p_circle_tail_inside_ratio_bound = p_circle_tail_inside_ratio_bound and p_circle_inside_bound
            component_tail_floor = [
                _cauchy_tail_from_bound(component_cauchy_floor[component], time, analytic_radius, tail_start)
                for component in range(4)
            ]
            component_allowed_bound = [
                radius0[component] * (1.0 - ratio) / (ratio ** (tail_start + 1))
                for component in range(4)
            ]
            component_headroom = [
                math.inf
                if component_cauchy_floor[component] == 0.0
                else component_allowed_bound[component] / component_cauchy_floor[component]
                for component in range(4)
            ]
            c_tail = _c_tail_bound_at_p(target_p, state, tuple(component_tail_floor))
            max_c_tail = max(max_c_tail, c_tail)
            for component in range(4):
                max_floor_tail_4d[component] = max(max_floor_tail_4d[component], component_tail_floor[component])
                max_observed_cauchy_floor_4d[component] = max(
                    max_observed_cauchy_floor_4d[component],
                    component_cauchy_floor[component],
                )
                min_allowed_cauchy_bound_4d[component] = min(
                    min_allowed_cauchy_bound_4d[component],
                    component_allowed_bound[component],
                )
                min_cauchy_headroom_4d[component] = min(
                    min_cauchy_headroom_4d[component],
                    component_headroom[component],
                )
            rows.append(
                {
                    "b": b,
                    "state_4d": list(state),
                    "slice_time_over_radius": ratio,
                    "component_observed_cauchy_floor_4d": component_cauchy_floor,
                    "p_circle_abs_partial": p_circle,
                    "p_circle_tail_estimate": p_circle_tail_estimate,
                    "p_circle_observed_tail_ratio": p_circle_ratio,
                    "p_circle_tail_inside_ratio_bound": p_circle_inside_bound,
                    "p_circle_ratio_witness": (
                        None
                        if p_circle_witness is None
                        else {"degree": p_circle_witness[0], "next_degree": p_circle_witness[1], "ratio": p_circle_witness[2]}
                    ),
                    "component_allowed_cauchy_bound_4d": component_allowed_bound,
                    "component_cauchy_headroom_4d": component_headroom,
                    "component_tail_floor_4d": component_tail_floor,
                    "component_tail_floor_over_radius_4d": [
                        component_tail_floor[component] / radius0[component]
                        for component in range(4)
                    ],
                    "c_tail_floor": c_tail,
                    "c_tail_floor_over_radius": c_tail / radius0[4],
                    "status": "observed_cauchy_floor_inside_start_radius"
                    if all(component_tail_floor[component] < radius0[component] for component in range(4))
                    and c_tail < radius0[4]
                    else "observed_cauchy_floor_exceeds_start_radius",
                }
            )
        max_tail_5d = [*max_floor_tail_4d, max_c_tail]
        max_tail_over_radius = [
            max_tail_5d[index] / radius0[index]
            for index in range(5)
        ]
        if not valid_radius:
            status = "cauchy_radius_not_larger_than_slice_time"
        elif all(value < 1.0 for value in max_tail_over_radius):
            status = "observed_cauchy_floor_inside_start_radius"
        else:
            status = "observed_cauchy_floor_exceeds_start_radius"
        proof_relevant_status = (
            "below_real_terminal"
            if below_real_terminal
            else "at_or_beyond_real_terminal"
        )
        radius_rows.append(
            {
                "analytic_radius": analytic_radius,
                "status": status,
                "proof_relevant_status": proof_relevant_status,
                "rows": rows,
                "max_observed_cauchy_floor_4d": max_observed_cauchy_floor_4d,
                "p_circle_abs_partial": p_circle_abs_partial,
                "p_circle_tail_ratio_bound": circle_tail_ratio_bound,
                "max_p_circle_tail_estimate": max_p_circle_tail_estimate,
                "max_p_circle_observed_tail_ratio": max_p_circle_observed_tail_ratio,
                "p_circle_ratio_witness": p_circle_ratio_witness,
                "p_circle_tail_inside_ratio_bound": p_circle_tail_inside_ratio_bound,
                "p_circle_rouche_margin": p_circle_abs_partial["certified_min_abs_lower"] - max_p_circle_tail_estimate,
                "min_allowed_cauchy_bound_4d": min_allowed_cauchy_bound_4d,
                "min_cauchy_headroom_4d": min_cauchy_headroom_4d,
                "max_tail_floor_5d": max_tail_5d,
                "max_tail_floor_over_radius": max_tail_over_radius,
            }
        )

    viable_rows = [
        row
        for row in radius_rows
        if row["status"] == "observed_cauchy_floor_inside_start_radius"
    ]
    proof_relevant_viable_rows = [
        row
        for row in viable_rows
        if row["proof_relevant_status"] == "below_real_terminal"
    ]
    best_row = min(
        radius_rows,
        key=lambda row: max(row["max_tail_floor_over_radius"])
        if row["status"] != "cauchy_radius_not_larger_than_slice_time"
        else math.inf,
    )
    return {
        "status": "observed_cauchy_budget_has_proof_relevant_viable_radius"
        if proof_relevant_viable_rows
        else (
            "observed_cauchy_budget_only_viable_beyond_real_terminal"
            if viable_rows
            else "observed_cauchy_budget_has_no_viable_radius"
        ),
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "limiting_crossing_time_reference": terminal_time,
        "limiting_crossing_status": limiting_crossing.status,
        "target_p": target_p,
        "order": order,
        "tail_start": tail_start,
        "b_sample_count": b_sample_count,
        "b_samples": list(b_values),
        "working_dps": working_dps,
        "circle_sample_count": circle_sample_count,
        "circle_tail_ratio_bound": circle_tail_ratio_bound,
        "radius0": list(radius0),
        "analytic_radii": list(analytic_radii),
        "radius_rows": radius_rows,
        "viable_analytic_radii": [row["analytic_radius"] for row in viable_rows],
        "proof_relevant_viable_analytic_radii": [
            row["analytic_radius"]
            for row in proof_relevant_viable_rows
        ],
        "best_radius_by_observed_floor": best_row["analytic_radius"],
        "best_max_tail_floor_over_radius": max(best_row["max_tail_floor_over_radius"]),
        "best_radius_p_circle_abs_partial": best_row["p_circle_abs_partial"],
        "best_radius_min_p_circle_abs_partial": best_row["p_circle_abs_partial"]["sample_min_abs"],
        "best_radius_certified_min_p_circle_abs_partial": best_row["p_circle_abs_partial"]["certified_min_abs_lower"],
        "best_radius_max_p_circle_abs_partial": best_row["p_circle_abs_partial"]["sample_max_abs"],
        "best_radius_p_circle_tail_estimate": best_row["max_p_circle_tail_estimate"],
        "best_radius_p_circle_rouche_margin": best_row["p_circle_rouche_margin"],
        "best_radius_p_circle_observed_tail_ratio": best_row["max_p_circle_observed_tail_ratio"],
        "best_radius_p_circle_tail_inside_ratio_bound": best_row["p_circle_tail_inside_ratio_bound"],
        "remaining_obligation": (
            "prove a uniform analytic disk bound at one viable radius; the "
            "observed Cauchy floor is only a lower bound on the required circle "
            "bound, not a proof of the bound"
        ),
    }


def _ratio_profile_for_terms(
    terms: list[float],
    ratio_start: int,
    ratio_bound: float | None,
) -> dict:
    """Return parity-aware ratio diagnostics for a finite term list."""
    if ratio_start < 0:
        raise ValueError("ratio_start must be nonnegative")
    if ratio_bound is not None and not (0.0 <= ratio_bound < 1.0):
        raise ValueError("ratio_bound must lie in [0,1)")
    ratios = []
    max_ratio = 0.0
    max_witness = None
    inside_bound = True
    for parity in (0, 1):
        first_degree = ratio_start if ratio_start % 2 == parity else ratio_start + 1
        for degree in range(first_degree, len(terms) - 2, 2):
            if terms[degree] <= 0.0:
                continue
            ratio = terms[degree + 2] / terms[degree]
            item = {
                "degree": degree,
                "next_degree": degree + 2,
                "ratio": ratio,
                "parity": parity,
            }
            ratios.append(item)
            if ratio > max_ratio:
                max_ratio = ratio
                max_witness = item
            if ratio_bound is not None and ratio > ratio_bound:
                inside_bound = False
    tail = ratios[-6:] if len(ratios) > 6 else list(ratios)
    return {
        "ratio_start": ratio_start,
        "ratio_bound": ratio_bound,
        "max_ratio": max_ratio,
        "max_witness": max_witness,
        "inside_bound": inside_bound,
        "tail_ratios": tail,
        "ratio_count": len(ratios),
    }


def _geometric_envelope_profile_for_terms(
    terms: list[float],
    tail_start: int,
    ratio_bound: float,
) -> dict:
    """Return whether terms after ``tail_start`` fit a same-parity envelope."""
    if tail_start < 0:
        raise ValueError("tail_start must be nonnegative")
    if not (0.0 <= ratio_bound < 1.0):
        raise ValueError("ratio_bound must lie in [0,1)")
    rows = []
    max_usage = 0.0
    max_strict_usage = 0.0
    worst = None
    worst_strict = None
    envelope_tail_sum = 0.0
    observed_tail_sum = 0.0
    inside = True
    for parity in (0, 1):
        first_tail = tail_start + 1
        if first_tail % 2 != parity:
            first_tail += 1
        if first_tail >= len(terms):
            continue
        anchor = terms[first_tail]
        envelope_tail_sum += math.inf if ratio_bound >= 1.0 else anchor / (1.0 - ratio_bound)
        for degree in range(first_tail, len(terms), 2):
            observed_tail_sum += terms[degree]
            exponent = (degree - first_tail) // 2
            envelope = anchor * (ratio_bound ** exponent)
            if envelope == 0.0:
                usage = math.inf if terms[degree] > 0.0 else 0.0
            else:
                usage = terms[degree] / envelope
            row = {
                "parity": parity,
                "anchor_degree": first_tail,
                "degree": degree,
                "term": terms[degree],
                "envelope": envelope,
                "usage": usage,
            }
            rows.append(row)
            if usage > max_usage:
                max_usage = usage
                worst = row
            if degree > first_tail and usage > max_strict_usage:
                max_strict_usage = usage
                worst_strict = row
            if usage > 1.0:
                inside = False
    return {
        "tail_start": tail_start,
        "ratio_bound": ratio_bound,
        "inside_envelope": inside,
        "max_usage": max_usage,
        "max_strict_post_anchor_usage": max_strict_usage,
        "worst": worst,
        "worst_strict_post_anchor": worst_strict,
        "observed_tail_sum": observed_tail_sum,
        "envelope_tail_sum": envelope_tail_sum,
        "tail_sum_usage": (
            math.inf
            if envelope_tail_sum == 0.0 and observed_tail_sum > 0.0
            else (0.0 if envelope_tail_sum == 0.0 else observed_tail_sum / envelope_tail_sum)
        ),
        "rows": rows,
    }


def _s_series_terms_from_coefficients(
    coefficients: tuple[tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...]],
    t_radius: float,
) -> list[list[float]]:
    """Return term magnitudes for the even ``s=t^2`` Taylor series."""
    radius_s = mp.mpf(t_radius) ** 2
    return [
        [
            abs(float(row[2 * index] * (radius_s ** index)))
            for index in range((len(row) + 1) // 2)
        ]
        for row in coefficients
    ]


def _s_index_after_t_degree(tail_start_degree: int) -> int:
    """Return the first omitted s-index after a t-degree cutoff."""
    return tail_start_degree // 2 + 1


def _ordinary_ratio_profile_for_terms(
    terms: list[float],
    ratio_start: int,
    ratio_bound: float | None,
) -> dict:
    """Return consecutive ratio diagnostics for an ordinary one-variable series."""
    if ratio_start < 0:
        raise ValueError("ratio_start must be nonnegative")
    if ratio_bound is not None and not (0.0 <= ratio_bound < 1.0):
        raise ValueError("ratio_bound must lie in [0,1)")
    ratios = []
    max_ratio = 0.0
    max_witness = None
    inside_bound = True
    for index in range(ratio_start, len(terms) - 1):
        if terms[index] <= 0.0:
            continue
        ratio = terms[index + 1] / terms[index]
        item = {"index": index, "next_index": index + 1, "ratio": ratio}
        ratios.append(item)
        if ratio > max_ratio:
            max_ratio = ratio
            max_witness = item
        if ratio_bound is not None and ratio > ratio_bound:
            inside_bound = False
    return {
        "ratio_start": ratio_start,
        "ratio_bound": ratio_bound,
        "max_ratio": max_ratio,
        "max_witness": max_witness,
        "inside_bound": inside_bound,
        "tail_ratios": ratios[-6:] if len(ratios) > 6 else list(ratios),
        "ratio_count": len(ratios),
    }


def _ordinary_geometric_envelope_profile_for_terms(
    terms: list[float],
    tail_start: int,
    ratio_bound: float,
) -> dict:
    """Return whether ordinary terms after ``tail_start`` fit a geometric envelope."""
    if tail_start < 0:
        raise ValueError("tail_start must be nonnegative")
    if not (0.0 <= ratio_bound < 1.0):
        raise ValueError("ratio_bound must lie in [0,1)")
    first_tail = tail_start + 1
    rows = []
    max_usage = 0.0
    max_strict_usage = 0.0
    worst = None
    worst_strict = None
    observed_tail_sum = 0.0
    if first_tail >= len(terms):
        envelope_tail_sum = 0.0
        inside = True
    else:
        anchor = terms[first_tail]
        envelope_tail_sum = anchor / (1.0 - ratio_bound)
        inside = True
        for index in range(first_tail, len(terms)):
            observed_tail_sum += terms[index]
            exponent = index - first_tail
            envelope = anchor * (ratio_bound ** exponent)
            usage = math.inf if envelope == 0.0 and terms[index] > 0.0 else (0.0 if envelope == 0.0 else terms[index] / envelope)
            row = {
                "anchor_index": first_tail,
                "index": index,
                "term": terms[index],
                "envelope": envelope,
                "usage": usage,
            }
            rows.append(row)
            if usage > max_usage:
                max_usage = usage
                worst = row
            if index > first_tail and usage > max_strict_usage:
                max_strict_usage = usage
                worst_strict = row
            if usage > 1.0:
                inside = False
    return {
        "tail_start": tail_start,
        "ratio_bound": ratio_bound,
        "inside_envelope": inside,
        "max_usage": max_usage,
        "max_strict_post_anchor_usage": max_strict_usage,
        "worst": worst,
        "worst_strict_post_anchor": worst_strict,
        "observed_tail_sum": observed_tail_sum,
        "envelope_tail_sum": envelope_tail_sum,
        "tail_sum_usage": (
            math.inf
            if envelope_tail_sum == 0.0 and observed_tail_sum > 0.0
            else (0.0 if envelope_tail_sum == 0.0 else observed_tail_sum / envelope_tail_sum)
        ),
        "rows": rows,
    }


def taylor_ratio_profile_audit(
    target_p: float = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_START,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    order: int = 80,
    ratio_start: int = 70,
    b_sample_count: int = 3,
    b_mode: str = "grid",
    working_dps: int = 90,
    circle_radius: float = 3.5,
    circle_ratio_bound: float | None = 0.95,
    p_slice_ratio_bound: float | None = 0.53,
) -> dict:
    """Report finite same-parity ratio profiles for the current Taylor target.

    This is a proof-target diagnostic.  It does not prove the infinite tail
    estimate; it records whether the finite coefficient window is compatible
    with proposed geometric ratio bounds and where the worst ratios occur.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if not (0.0 < target_p < 1.0):
        raise ValueError("target_p must lie between 0 and 1")
    if order <= 0:
        raise ValueError("order must be positive")
    if not (0 <= ratio_start < order):
        raise ValueError("orders must satisfy 0 <= ratio_start < order")
    if b_mode not in {"grid", "limit"}:
        raise ValueError("b_mode must be grid or limit")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    if circle_radius <= 0.0:
        raise ValueError("circle_radius must be positive")
    if circle_ratio_bound is not None and not (0.0 <= circle_ratio_bound < 1.0):
        raise ValueError("circle_ratio_bound must lie in [0,1)")
    if p_slice_ratio_bound is not None and not (0.0 <= p_slice_ratio_bound < 1.0):
        raise ValueError("p_slice_ratio_bound must lie in [0,1)")

    beta = 1.0 / candidate_a
    b_values = (0.0,) if b_mode == "limit" else _symmetric_b_grid(candidate_a, b_sample_count)
    rows = []
    max_circle_ratio_4d = [0.0 for _ in range(4)]
    max_circle_witness_4d = [None for _ in range(4)]
    max_p_slice_ratio_4d = [0.0 for _ in range(4)]
    max_p_slice_witness_4d = [None for _ in range(4)]
    circle_inside_bound = True
    p_slice_inside_bound = True
    for b in b_values:
        coefficients = scaled_taylor_coefficients(order, b, working_dps)
        state = high_order_scaled_taylor_state_at_p_from_coefficients(target_p, coefficients)
        p_slice_terms = _taylor_term_magnitudes(coefficients, state[0])
        circle_terms = [
            [
                abs(float(coefficient * (mp.mpf(circle_radius) ** degree)))
                for degree, coefficient in enumerate(component)
            ]
            for component in coefficients
        ]
        circle_profiles = []
        p_slice_profiles = []
        for component in range(4):
            circle_profile = _ratio_profile_for_terms(
                circle_terms[component],
                ratio_start,
                circle_ratio_bound,
            )
            p_slice_profile = _ratio_profile_for_terms(
                p_slice_terms[component],
                ratio_start,
                p_slice_ratio_bound,
            )
            circle_profiles.append(circle_profile)
            p_slice_profiles.append(p_slice_profile)
            if circle_profile["max_ratio"] > max_circle_ratio_4d[component]:
                max_circle_ratio_4d[component] = circle_profile["max_ratio"]
                witness = circle_profile["max_witness"]
                max_circle_witness_4d[component] = (
                    None
                    if witness is None
                    else {"b": b, **witness}
                )
            if p_slice_profile["max_ratio"] > max_p_slice_ratio_4d[component]:
                max_p_slice_ratio_4d[component] = p_slice_profile["max_ratio"]
                witness = p_slice_profile["max_witness"]
                max_p_slice_witness_4d[component] = (
                    None
                    if witness is None
                    else {"b": b, **witness}
                )
            circle_inside_bound = circle_inside_bound and circle_profile["inside_bound"]
            p_slice_inside_bound = p_slice_inside_bound and p_slice_profile["inside_bound"]
        rows.append(
            {
                "b": b,
                "p_slice_state_4d": list(state),
                "circle_profiles_4d": circle_profiles,
                "p_slice_profiles_4d": p_slice_profiles,
            }
        )

    return {
        "status": "observed_ratios_inside_bounds"
        if circle_inside_bound and p_slice_inside_bound
        else "observed_ratios_exceed_bounds",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "b_sample_count": len(b_values),
        "requested_b_sample_count": b_sample_count,
        "b_mode": b_mode,
        "b_samples": list(b_values),
        "target_p": target_p,
        "order": order,
        "ratio_start": ratio_start,
        "working_dps": working_dps,
        "circle_radius": circle_radius,
        "circle_ratio_bound": circle_ratio_bound,
        "p_slice_ratio_bound": p_slice_ratio_bound,
        "circle_inside_bound": circle_inside_bound,
        "p_slice_inside_bound": p_slice_inside_bound,
        "max_circle_ratio_4d": max_circle_ratio_4d,
        "max_circle_witness_4d": max_circle_witness_4d,
        "max_p_slice_ratio_4d": max_p_slice_ratio_4d,
        "max_p_slice_witness_4d": max_p_slice_witness_4d,
        "rows": rows,
        "remaining_obligation": (
            "replace finite ratio-profile evidence by a recurrence or majorant "
            "argument controlling all later coefficients"
        ),
    }


def taylor_geometric_envelope_audit(
    target_p: float = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_START,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    order: int = 120,
    tail_start: int = 50,
    b_sample_count: int = 3,
    b_mode: str = "grid",
    working_dps: int = 110,
    circle_radius: float = 3.5,
    circle_ratio_bound: float = 0.95,
    p_slice_ratio_bound: float = 0.6,
) -> dict:
    """Check a concrete same-parity geometric tail-envelope ansatz.

    This is still finite-window evidence, not the induction proof.  It records
    whether every computed term after ``tail_start`` is dominated by the first
    omitted same-parity term times the proposed geometric ratio.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if not (0.0 < target_p < 1.0):
        raise ValueError("target_p must lie between 0 and 1")
    if order <= 0:
        raise ValueError("order must be positive")
    if not (0 <= tail_start < order):
        raise ValueError("orders must satisfy 0 <= tail_start < order")
    if b_mode not in {"grid", "limit"}:
        raise ValueError("b_mode must be grid or limit")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    if circle_radius <= 0.0:
        raise ValueError("circle_radius must be positive")
    if not (0.0 <= circle_ratio_bound < 1.0):
        raise ValueError("circle_ratio_bound must lie in [0,1)")
    if not (0.0 <= p_slice_ratio_bound < 1.0):
        raise ValueError("p_slice_ratio_bound must lie in [0,1)")

    beta = 1.0 / candidate_a
    b_values = (0.0,) if b_mode == "limit" else _symmetric_b_grid(candidate_a, b_sample_count)
    rows = []
    circle_inside = True
    p_slice_inside = True
    max_circle_usage_4d = [0.0 for _ in range(4)]
    max_p_slice_usage_4d = [0.0 for _ in range(4)]
    max_circle_strict_usage_4d = [0.0 for _ in range(4)]
    max_p_slice_strict_usage_4d = [0.0 for _ in range(4)]
    max_circle_tail_sum_usage_4d = [0.0 for _ in range(4)]
    max_p_slice_tail_sum_usage_4d = [0.0 for _ in range(4)]
    circle_worst_4d = [None for _ in range(4)]
    p_slice_worst_4d = [None for _ in range(4)]
    circle_worst_strict_4d = [None for _ in range(4)]
    p_slice_worst_strict_4d = [None for _ in range(4)]
    for b in b_values:
        coefficients = scaled_taylor_coefficients(order, b, working_dps)
        state = high_order_scaled_taylor_state_at_p_from_coefficients(target_p, coefficients)
        p_slice_terms = _taylor_term_magnitudes(coefficients, state[0])
        circle_terms = [
            [
                abs(float(coefficient * (mp.mpf(circle_radius) ** degree)))
                for degree, coefficient in enumerate(component)
            ]
            for component in coefficients
        ]
        circle_profiles = []
        p_slice_profiles = []
        for component in range(4):
            circle_profile = _geometric_envelope_profile_for_terms(
                circle_terms[component],
                tail_start,
                circle_ratio_bound,
            )
            p_slice_profile = _geometric_envelope_profile_for_terms(
                p_slice_terms[component],
                tail_start,
                p_slice_ratio_bound,
            )
            circle_profiles.append(circle_profile)
            p_slice_profiles.append(p_slice_profile)
            circle_inside = circle_inside and circle_profile["inside_envelope"]
            p_slice_inside = p_slice_inside and p_slice_profile["inside_envelope"]
            if circle_profile["max_usage"] > max_circle_usage_4d[component]:
                max_circle_usage_4d[component] = circle_profile["max_usage"]
                worst = circle_profile["worst"]
                circle_worst_4d[component] = None if worst is None else {"b": b, **worst}
            if p_slice_profile["max_usage"] > max_p_slice_usage_4d[component]:
                max_p_slice_usage_4d[component] = p_slice_profile["max_usage"]
                worst = p_slice_profile["worst"]
                p_slice_worst_4d[component] = None if worst is None else {"b": b, **worst}
            if circle_profile["max_strict_post_anchor_usage"] > max_circle_strict_usage_4d[component]:
                max_circle_strict_usage_4d[component] = circle_profile["max_strict_post_anchor_usage"]
                worst = circle_profile["worst_strict_post_anchor"]
                circle_worst_strict_4d[component] = None if worst is None else {"b": b, **worst}
            if p_slice_profile["max_strict_post_anchor_usage"] > max_p_slice_strict_usage_4d[component]:
                max_p_slice_strict_usage_4d[component] = p_slice_profile["max_strict_post_anchor_usage"]
                worst = p_slice_profile["worst_strict_post_anchor"]
                p_slice_worst_strict_4d[component] = None if worst is None else {"b": b, **worst}
            max_circle_tail_sum_usage_4d[component] = max(
                max_circle_tail_sum_usage_4d[component],
                circle_profile["tail_sum_usage"],
            )
            max_p_slice_tail_sum_usage_4d[component] = max(
                max_p_slice_tail_sum_usage_4d[component],
                p_slice_profile["tail_sum_usage"],
            )
        rows.append(
            {
                "b": b,
                "p_slice_state_4d": list(state),
                "circle_profiles_4d": circle_profiles,
                "p_slice_profiles_4d": p_slice_profiles,
            }
        )

    return {
        "status": "observed_terms_inside_geometric_envelopes"
        if circle_inside and p_slice_inside
        else "observed_terms_exceed_geometric_envelopes",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "b_mode": b_mode,
        "b_sample_count": len(b_values),
        "requested_b_sample_count": b_sample_count,
        "b_samples": list(b_values),
        "target_p": target_p,
        "order": order,
        "tail_start": tail_start,
        "working_dps": working_dps,
        "circle_radius": circle_radius,
        "circle_ratio_bound": circle_ratio_bound,
        "p_slice_ratio_bound": p_slice_ratio_bound,
        "circle_inside_envelope": circle_inside,
        "p_slice_inside_envelope": p_slice_inside,
        "max_circle_envelope_usage_4d": max_circle_usage_4d,
        "max_p_slice_envelope_usage_4d": max_p_slice_usage_4d,
        "max_circle_strict_post_anchor_usage_4d": max_circle_strict_usage_4d,
        "max_p_slice_strict_post_anchor_usage_4d": max_p_slice_strict_usage_4d,
        "max_circle_tail_sum_usage_4d": max_circle_tail_sum_usage_4d,
        "max_p_slice_tail_sum_usage_4d": max_p_slice_tail_sum_usage_4d,
        "max_circle_worst_4d": circle_worst_4d,
        "max_p_slice_worst_4d": p_slice_worst_4d,
        "max_circle_worst_strict_4d": circle_worst_strict_4d,
        "max_p_slice_worst_strict_4d": p_slice_worst_strict_4d,
        "rows": rows,
        "remaining_obligation": (
            "turn the finite-window envelope check into an induction or "
            "coefficient-majorant proof for all later same-parity coefficients"
        ),
    }


def taylor_even_s_series_audit(
    target_p: float = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_START,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    order: int = 120,
    tail_start: int = 50,
    b_sample_count: int = 3,
    b_mode: str = "grid",
    working_dps: int = 110,
    circle_radius: float = 3.5,
    circle_ratio_bound: float = 0.95,
    p_slice_ratio_bound: float = 0.6,
) -> dict:
    """Report the tail target in the even variable ``s=t^2``.

    Because all odd Taylor coefficients vanish, this is the natural ordinary
    one-variable series behind the same-parity diagnostics.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if not (0.0 < target_p < 1.0):
        raise ValueError("target_p must lie between 0 and 1")
    if order <= 0:
        raise ValueError("order must be positive")
    if not (0 <= tail_start < order):
        raise ValueError("orders must satisfy 0 <= tail_start < order")
    if b_mode not in {"grid", "limit"}:
        raise ValueError("b_mode must be grid or limit")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    if circle_radius <= 0.0:
        raise ValueError("circle_radius must be positive")
    if not (0.0 <= circle_ratio_bound < 1.0):
        raise ValueError("circle_ratio_bound must lie in [0,1)")
    if not (0.0 <= p_slice_ratio_bound < 1.0):
        raise ValueError("p_slice_ratio_bound must lie in [0,1)")

    beta = 1.0 / candidate_a
    b_values = (0.0,) if b_mode == "limit" else _symmetric_b_grid(candidate_a, b_sample_count)
    tail_start_s = _s_index_after_t_degree(tail_start) - 1
    ratio_start_s = tail_start_s
    circle_radius_s = circle_radius * circle_radius
    limiting_crossing = first_scaled_crossing("limit", step_size=DEFAULT_STEP)
    limiting_crossing_s = limiting_crossing.time * limiting_crossing.time if limiting_crossing.status == "crossed" else math.inf

    rows = []
    max_circle_ratio_4d = [0.0 for _ in range(4)]
    max_p_slice_ratio_4d = [0.0 for _ in range(4)]
    max_circle_strict_usage_4d = [0.0 for _ in range(4)]
    max_p_slice_strict_usage_4d = [0.0 for _ in range(4)]
    max_circle_tail_sum_usage_4d = [0.0 for _ in range(4)]
    max_p_slice_tail_sum_usage_4d = [0.0 for _ in range(4)]
    min_inferred_circle_radius_s_4d = [math.inf for _ in range(4)]
    min_inferred_p_slice_radius_s_4d = [math.inf for _ in range(4)]
    circle_inside = True
    p_slice_inside = True
    for b in b_values:
        coefficients = scaled_taylor_coefficients(order, b, working_dps)
        state = high_order_scaled_taylor_state_at_p_from_coefficients(target_p, coefficients)
        p_slice_radius = state[0]
        p_slice_radius_s = p_slice_radius * p_slice_radius
        circle_terms = _s_series_terms_from_coefficients(coefficients, circle_radius)
        p_slice_terms = _s_series_terms_from_coefficients(coefficients, p_slice_radius)
        circle_profiles = []
        p_slice_profiles = []
        for component in range(4):
            circle_ratio_profile = _ordinary_ratio_profile_for_terms(
                circle_terms[component],
                ratio_start_s,
                circle_ratio_bound,
            )
            p_slice_ratio_profile = _ordinary_ratio_profile_for_terms(
                p_slice_terms[component],
                ratio_start_s,
                p_slice_ratio_bound,
            )
            circle_envelope = _ordinary_geometric_envelope_profile_for_terms(
                circle_terms[component],
                tail_start_s,
                circle_ratio_bound,
            )
            p_slice_envelope = _ordinary_geometric_envelope_profile_for_terms(
                p_slice_terms[component],
                tail_start_s,
                p_slice_ratio_bound,
            )
            circle_profiles.append({"ratio": circle_ratio_profile, "envelope": circle_envelope})
            p_slice_profiles.append({"ratio": p_slice_ratio_profile, "envelope": p_slice_envelope})
            max_circle_ratio_4d[component] = max(max_circle_ratio_4d[component], circle_ratio_profile["max_ratio"])
            max_p_slice_ratio_4d[component] = max(max_p_slice_ratio_4d[component], p_slice_ratio_profile["max_ratio"])
            max_circle_strict_usage_4d[component] = max(
                max_circle_strict_usage_4d[component],
                circle_envelope["max_strict_post_anchor_usage"],
            )
            max_p_slice_strict_usage_4d[component] = max(
                max_p_slice_strict_usage_4d[component],
                p_slice_envelope["max_strict_post_anchor_usage"],
            )
            max_circle_tail_sum_usage_4d[component] = max(
                max_circle_tail_sum_usage_4d[component],
                circle_envelope["tail_sum_usage"],
            )
            max_p_slice_tail_sum_usage_4d[component] = max(
                max_p_slice_tail_sum_usage_4d[component],
                p_slice_envelope["tail_sum_usage"],
            )
            if circle_ratio_profile["max_ratio"] > 0.0:
                min_inferred_circle_radius_s_4d[component] = min(
                    min_inferred_circle_radius_s_4d[component],
                    circle_radius_s / circle_ratio_profile["max_ratio"],
                )
            if p_slice_ratio_profile["max_ratio"] > 0.0:
                min_inferred_p_slice_radius_s_4d[component] = min(
                    min_inferred_p_slice_radius_s_4d[component],
                    p_slice_radius_s / p_slice_ratio_profile["max_ratio"],
                )
            circle_inside = circle_inside and circle_ratio_profile["inside_bound"] and circle_envelope["inside_envelope"]
            p_slice_inside = p_slice_inside and p_slice_ratio_profile["inside_bound"] and p_slice_envelope["inside_envelope"]
        rows.append(
            {
                "b": b,
                "p_slice_state_4d": list(state),
                "p_slice_radius_s": p_slice_radius_s,
                "circle_profiles_4d": circle_profiles,
                "p_slice_profiles_4d": p_slice_profiles,
            }
        )

    return {
        "status": "observed_s_series_inside_targets"
        if circle_inside and p_slice_inside
        else "observed_s_series_exceeds_targets",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "b_mode": b_mode,
        "b_sample_count": len(b_values),
        "requested_b_sample_count": b_sample_count,
        "b_samples": list(b_values),
        "target_p": target_p,
        "order": order,
        "tail_start_t_degree": tail_start,
        "tail_start_s_index": tail_start_s,
        "working_dps": working_dps,
        "circle_radius_t": circle_radius,
        "circle_radius_s": circle_radius_s,
        "circle_ratio_bound": circle_ratio_bound,
        "p_slice_ratio_bound": p_slice_ratio_bound,
        "limiting_crossing_time_reference": limiting_crossing.time,
        "limiting_crossing_s_reference": limiting_crossing_s,
        "max_circle_ratio_4d": max_circle_ratio_4d,
        "max_p_slice_ratio_4d": max_p_slice_ratio_4d,
        "max_circle_strict_usage_4d": max_circle_strict_usage_4d,
        "max_p_slice_strict_usage_4d": max_p_slice_strict_usage_4d,
        "max_circle_tail_sum_usage_4d": max_circle_tail_sum_usage_4d,
        "max_p_slice_tail_sum_usage_4d": max_p_slice_tail_sum_usage_4d,
        "min_inferred_circle_radius_s_4d": min_inferred_circle_radius_s_4d,
        "min_inferred_p_slice_radius_s_4d": min_inferred_p_slice_radius_s_4d,
        "circle_inside_target": circle_inside,
        "p_slice_inside_target": p_slice_inside,
        "rows": rows,
        "remaining_obligation": (
            "prove the ordinary s-series geometric envelope from the recurrence "
            "2s dX/ds = G(s,X,b)"
        ),
    }


def _recurrence_matrix_apply(degree: int, vector: tuple) -> tuple:
    """Apply the explicit degree-``degree`` Taylor recurrence matrix."""
    if degree <= 0:
        raise ValueError("degree must be positive")
    p_coeff, x1_coeff, x2_coeff, x3_coeff = vector
    d = mp.mpf(degree)
    return (
        (d + 5) * p_coeff + 27 * x2_coeff - mp.mpf(2) * x3_coeff / 3,
        81 * p_coeff + (d + 4) * x1_coeff - 27 * x3_coeff,
        d * x2_coeff,
        -6 * p_coeff + (d + 2) * x3_coeff,
    )


def _recurrence_inverse_apply(degree: int, forcing: tuple) -> tuple:
    """Apply the closed-form inverse of the Taylor recurrence matrix."""
    if degree <= 0:
        raise ValueError("degree must be positive")
    d = mp.mpf(degree)
    r0, r1, r2, r3 = forcing
    u0 = r0 - 27 * r2 / d
    denominator = (d + 1) * (d + 6)
    p_coeff = ((d + 2) * u0 + mp.mpf(2) * r3 / 3) / denominator
    x3_coeff = (6 * u0 + (d + 5) * r3) / denominator
    x2_coeff = r2 / d
    x1_coeff = r1 / (d + 4) + (-81 * d * u0 + 27 * (d + 3) * r3) / (denominator * (d + 4))
    return (p_coeff, x1_coeff, x2_coeff, x3_coeff)


def _recurrence_inverse_abs_bound(degree: int, forcing_abs: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Return a componentwise absolute inverse bound for one forcing vector."""
    if degree <= 0:
        raise ValueError("degree must be positive")
    d = float(degree)
    r0, r1, r2, r3 = forcing_abs
    u0 = r0 + 27.0 * r2 / d
    denominator = (d + 1.0) * (d + 6.0)
    p_bound = ((d + 2.0) * u0 + (2.0 / 3.0) * r3) / denominator
    x3_bound = (6.0 * u0 + (d + 5.0) * r3) / denominator
    x2_bound = r2 / d
    x1_bound = r1 / (d + 4.0) + (81.0 * d * u0 + 27.0 * (d + 3.0) * r3) / (denominator * (d + 4.0))
    return (p_bound, x1_bound, x2_bound, x3_bound)


def taylor_recurrence_forcing_audit(
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    order: int = 60,
    tail_start: int = 50,
    b_sample_count: int = 3,
    b_mode: str = "grid",
    working_dps: int = 90,
    circle_radius: float = 3.5,
    circle_ratio_bound: float = 0.95,
    reconstruction_tolerance: float = 1e-60,
) -> dict:
    """Audit the explicit inverse and lower-order forcing in the s-series recurrence.

    At t-degree ``d`` the new coefficient vector solves ``M_d y_d = R_d`` with
    an explicit degree-linear matrix.  This diagnostic checks the closed-form
    inverse against computed coefficients and records geometric-ratio profiles
    for both the solution coefficients and the forcing terms on the proof
    circle ``|s|=circle_radius^2``.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if order <= 2:
        raise ValueError("order must be greater than 2")
    if not (0 <= tail_start < order):
        raise ValueError("orders must satisfy 0 <= tail_start < order")
    if b_sample_count < 3 or b_sample_count % 2 == 0:
        raise ValueError("b_sample_count must be an odd integer at least 3")
    if b_mode not in {"grid", "limit"}:
        raise ValueError("b_mode must be grid or limit")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    if circle_radius <= 0.0:
        raise ValueError("circle_radius must be positive")
    if not (0.0 <= circle_ratio_bound < 1.0):
        raise ValueError("circle_ratio_bound must lie in [0,1)")
    if reconstruction_tolerance < 0.0:
        raise ValueError("reconstruction_tolerance must be nonnegative")

    beta = 1.0 / candidate_a
    b_values = (0.0,) if b_mode == "limit" else _symmetric_b_grid(candidate_a, b_sample_count)
    circle_radius_s = circle_radius * circle_radius
    tail_start_s = _s_index_after_t_degree(tail_start) - 1
    max_reconstruction_error_4d = [0.0 for _ in range(4)]
    max_inverse_bound_usage_4d = [0.0 for _ in range(4)]
    max_solution_ratio_4d = [0.0 for _ in range(4)]
    max_forcing_ratio_4d = [0.0 for _ in range(4)]
    worst_inverse_bound_usage_4d = [None for _ in range(4)]
    rows = []
    solution_inside = True
    forcing_inside = True

    for b in b_values:
        with mp.workdps(working_dps):
            coefficients = scaled_taylor_coefficients(order, b, working_dps)
            s_order = order // 2
            solution_terms = [[0.0 for _ in range(s_order + 1)] for _ in range(4)]
            forcing_terms = [[0.0 for _ in range(s_order + 1)] for _ in range(4)]
            row_max_reconstruction = [0.0 for _ in range(4)]
            row_max_inverse_usage = [0.0 for _ in range(4)]
            for degree in range(2, order + 1, 2):
                s_index = degree // 2
                vector = tuple(coefficients[component][degree] for component in range(4))
                forcing = _recurrence_matrix_apply(degree, vector)
                reconstructed = _recurrence_inverse_apply(degree, forcing)
                forcing_abs = tuple(float(abs(value)) for value in forcing)
                inverse_bound = _recurrence_inverse_abs_bound(degree, forcing_abs)  # type: ignore[arg-type]
                scale = float(mp.mpf(circle_radius_s) ** s_index)
                for component in range(4):
                    solution_abs = float(abs(vector[component]))
                    error = float(abs(reconstructed[component] - vector[component]))
                    row_max_reconstruction[component] = max(row_max_reconstruction[component], error)
                    max_reconstruction_error_4d[component] = max(max_reconstruction_error_4d[component], error)
                    solution_terms[component][s_index] = solution_abs * scale
                    forcing_terms[component][s_index] = forcing_abs[component] * scale
                    if inverse_bound[component] == 0.0:
                        usage = math.inf if solution_abs > 0.0 else 0.0
                    else:
                        usage = solution_abs / inverse_bound[component]
                    row_max_inverse_usage[component] = max(row_max_inverse_usage[component], usage)
                    if usage > max_inverse_bound_usage_4d[component]:
                        max_inverse_bound_usage_4d[component] = usage
                        worst_inverse_bound_usage_4d[component] = {
                            "b": b,
                            "degree": degree,
                            "s_index": s_index,
                            "usage": usage,
                            "solution_abs": solution_abs,
                            "inverse_bound": inverse_bound[component],
                            "forcing_abs_4d": list(forcing_abs),
                        }

            solution_profiles = []
            forcing_profiles = []
            for component in range(4):
                solution_profile = _ordinary_ratio_profile_for_terms(
                    solution_terms[component],
                    tail_start_s,
                    circle_ratio_bound,
                )
                forcing_profile = _ordinary_ratio_profile_for_terms(
                    forcing_terms[component],
                    tail_start_s,
                    circle_ratio_bound,
                )
                solution_profiles.append(solution_profile)
                forcing_profiles.append(forcing_profile)
                max_solution_ratio_4d[component] = max(max_solution_ratio_4d[component], solution_profile["max_ratio"])
                max_forcing_ratio_4d[component] = max(max_forcing_ratio_4d[component], forcing_profile["max_ratio"])
                solution_inside = solution_inside and solution_profile["inside_bound"]
                forcing_inside = forcing_inside and forcing_profile["inside_bound"]

            rows.append(
                {
                    "b": b,
                    "max_reconstruction_error_4d": row_max_reconstruction,
                    "max_inverse_bound_usage_4d": row_max_inverse_usage,
                    "solution_profiles_4d": solution_profiles,
                    "forcing_profiles_4d": forcing_profiles,
                }
            )

    reconstruction_ok = max(max_reconstruction_error_4d) <= reconstruction_tolerance
    inverse_bound_ok = max(max_inverse_bound_usage_4d) <= 1.0 + 1e-12
    return {
        "status": "observed_recurrence_forcing_inside_targets"
        if reconstruction_ok and inverse_bound_ok and solution_inside and forcing_inside
        else "observed_recurrence_forcing_exceeds_targets",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "b_mode": b_mode,
        "b_sample_count": len(b_values),
        "requested_b_sample_count": b_sample_count,
        "b_samples": list(b_values),
        "order": order,
        "tail_start_t_degree": tail_start,
        "tail_start_s_index": tail_start_s,
        "working_dps": working_dps,
        "circle_radius_t": circle_radius,
        "circle_radius_s": circle_radius_s,
        "circle_ratio_bound": circle_ratio_bound,
        "reconstruction_tolerance": reconstruction_tolerance,
        "max_reconstruction_error_4d": max_reconstruction_error_4d,
        "max_inverse_bound_usage_4d": max_inverse_bound_usage_4d,
        "worst_inverse_bound_usage_4d": worst_inverse_bound_usage_4d,
        "max_solution_ratio_4d": max_solution_ratio_4d,
        "max_forcing_ratio_4d": max_forcing_ratio_4d,
        "solution_inside_bound": solution_inside,
        "forcing_inside_bound": forcing_inside,
        "inverse_bound_ok": inverse_bound_ok,
        "reconstruction_ok": reconstruction_ok,
        "matrix_determinant_formula": "d*(d+1)*(d+4)*(d+6)",
        "rows": rows,
        "remaining_obligation": (
            "turn the observed forcing ratio profile into a symbolic convolution "
            "majorant for R_d under the ordinary s-series envelope"
        ),
    }


def taylor_even_parity_audit(
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    order: int = 40,
    b_sample_count: int = 3,
    working_dps: int = 80,
    complex_b_radius: float | None = None,
    complex_b_sample_count: int = 8,
    tolerance: float = 1e-60,
) -> dict:
    """Check the even-in-t parity of the Taylor recurrence.

    The scaled equation has an even Taylor germ in the singular time variable.
    This audit keeps that fact reproducible for real finite-b samples and,
    optionally, for a complex-b circle used by the event-map diagnostics.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if order < 0:
        raise ValueError("order must be nonnegative")
    if b_sample_count < 3 or b_sample_count % 2 == 0:
        raise ValueError("b_sample_count must be an odd integer at least 3")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    if complex_b_radius is not None and complex_b_radius <= 0.0:
        raise ValueError("complex_b_radius must be positive")
    if complex_b_sample_count <= 0:
        raise ValueError("complex_b_sample_count must be positive")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")

    beta = 1.0 / candidate_a
    rows = []
    max_odd_abs_4d = [0.0 for _ in range(4)]
    max_even_abs_4d = [0.0 for _ in range(4)]

    def add_row(label: str, b_value, coefficients) -> None:
        odd_abs = []
        even_abs = []
        first_nonzero_odd = []
        for component, row in enumerate(coefficients):
            component_odd_abs = [float(abs(row[degree])) for degree in range(1, len(row), 2)]
            component_even_abs = [float(abs(row[degree])) for degree in range(0, len(row), 2)]
            odd_max = max(component_odd_abs, default=0.0)
            even_max = max(component_even_abs, default=0.0)
            odd_abs.append(odd_max)
            even_abs.append(even_max)
            max_odd_abs_4d[component] = max(max_odd_abs_4d[component], odd_max)
            max_even_abs_4d[component] = max(max_even_abs_4d[component], even_max)
            first_nonzero_odd.append(
                next(
                    (
                        degree
                        for degree in range(1, len(row), 2)
                        if float(abs(row[degree])) > tolerance
                    ),
                    None,
                )
            )
        rows.append(
            {
                "label": label,
                "b_real": float(mp.re(b_value)),
                "b_imag": float(mp.im(b_value)),
                "max_odd_abs_4d": odd_abs,
                "max_even_abs_4d": even_abs,
                "first_odd_degree_above_tolerance_4d": first_nonzero_odd,
            }
        )

    for b in _symmetric_b_grid(candidate_a, b_sample_count):
        add_row("real_grid", mp.mpf(b), scaled_taylor_coefficients(order, b, working_dps))
    if complex_b_radius is not None:
        for sample_index in range(complex_b_sample_count):
            theta = 2 * mp.pi * sample_index / complex_b_sample_count
            b_value = mp.mpc(complex_b_radius * mp.cos(theta), complex_b_radius * mp.sin(theta))
            add_row("complex_circle", b_value, complex_scaled_taylor_coefficients(order, b_value, working_dps))

    return {
        "status": "observed_odd_coefficients_zero"
        if max(max_odd_abs_4d, default=0.0) <= tolerance
        else "observed_odd_coefficients_nonzero",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "order": order,
        "b_sample_count": b_sample_count,
        "working_dps": working_dps,
        "complex_b_radius": complex_b_radius,
        "complex_b_sample_count": complex_b_sample_count if complex_b_radius is not None else 0,
        "tolerance": tolerance,
        "max_odd_abs_4d": max_odd_abs_4d,
        "max_even_abs_4d": max_even_abs_4d,
        "rows": rows,
        "remaining_obligation": (
            "The numerical check supports the exact parity lemma; the proof is "
            "by invariance of the recurrence under t -> -t."
        ),
    }


def _p_slice_state_5d(
    target_p: float,
    state_4d: tuple[float, float, float, float],
) -> tuple[float, float, float, float, float]:
    """Return the carried-C augmented p-slice state."""
    return (*state_4d, cancellation_c_value(target_p, state_4d))


def _circle_term_magnitudes(
    coefficients: tuple[tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...]],
    radius: float,
) -> list[list[float]]:
    """Return absolute Taylor term magnitudes ``|c_n R^n|`` on a circle."""
    radius_value = mp.mpf(radius)
    return [
        [abs(float(coefficient * (radius_value ** degree))) for degree, coefficient in enumerate(row)]
        for row in coefficients
    ]


def _coefficient_delta_term_magnitudes(
    coefficients: tuple[tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...]],
    baseline_coefficients: tuple[tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...], tuple[mp.mpf, ...]],
    radius: float,
) -> list[list[float]]:
    """Return ``|(c_n(b)-c_n(0)) R^n|`` by component."""
    radius_value = mp.mpf(radius)
    return [
        [
            abs(float((coefficient - baseline_coefficients[component][degree]) * (radius_value ** degree)))
            for degree, coefficient in enumerate(row)
        ]
        for component, row in enumerate(coefficients)
    ]


def taylor_b_sensitivity_audit(
    target_p: float = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_START,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    order: int = 80,
    ratio_start: int = 70,
    b_sample_count: int = 3,
    working_dps: int = 90,
    circle_radius: float = 3.5,
    radius0: tuple[float, float, float, float, float] = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_RADIUS,
) -> dict:
    """Measure sampled finite-``b`` sensitivity of the Taylor handoff.

    This is a proof-target diagnostic.  It quantifies how much the high-order
    Taylor p-slice and proof-circle coefficient magnitudes move over
    ``|b| <= 1/A`` relative to the limiting ``b=0`` germ.  A small result
    supports reducing the uniform finite-``A`` entry problem to a limiting
    coefficient/tail majorant plus a perturbation estimate.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if not (0.0 < target_p < 1.0):
        raise ValueError("target_p must lie between 0 and 1")
    if order <= 0:
        raise ValueError("order must be positive")
    if not (0 <= ratio_start < order):
        raise ValueError("orders must satisfy 0 <= ratio_start < order")
    if b_sample_count < 3 or b_sample_count % 2 == 0:
        raise ValueError("b_sample_count must be an odd integer at least 3")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    if circle_radius <= 0.0:
        raise ValueError("circle_radius must be positive")
    if len(radius0) != 5 or any(value <= 0.0 for value in radius0):
        raise ValueError("radius0 must contain five positive values")

    beta = 1.0 / candidate_a
    b_values = _symmetric_b_grid(candidate_a, b_sample_count)
    baseline_coefficients = scaled_taylor_coefficients(order, 0.0, working_dps)
    baseline_state_4d = high_order_scaled_taylor_state_at_p_from_coefficients(target_p, baseline_coefficients)
    baseline_state_5d = _p_slice_state_5d(target_p, baseline_state_4d)
    baseline_circle_terms = _circle_term_magnitudes(baseline_coefficients, circle_radius)
    baseline_circle_l1_4d = [sum(terms) for terms in baseline_circle_terms]
    baseline_circle_tail_l1_4d = [
        sum(terms[ratio_start:])
        for terms in baseline_circle_terms
    ]

    rows = []
    max_state_delta_5d = [0.0 for _ in range(5)]
    max_circle_delta_l1_4d = [0.0 for _ in range(4)]
    max_circle_tail_delta_l1_4d = [0.0 for _ in range(4)]
    max_circle_delta_component_l1_relative_4d = [0.0 for _ in range(4)]
    max_circle_tail_delta_l1_relative_4d = [0.0 for _ in range(4)]
    max_circle_term_delta_4d = [0.0 for _ in range(4)]
    max_circle_term_delta_relative_4d = [0.0 for _ in range(4)]
    max_p_slice_term_delta_l1_4d = [0.0 for _ in range(4)]
    max_p_slice_tail_delta_l1_4d = [0.0 for _ in range(4)]

    for b in b_values:
        coefficients = scaled_taylor_coefficients(order, b, working_dps)
        state_4d = high_order_scaled_taylor_state_at_p_from_coefficients(target_p, coefficients)
        state_5d = _p_slice_state_5d(target_p, state_4d)
        state_delta_5d = [
            abs(state_5d[index] - baseline_state_5d[index])
            for index in range(5)
        ]
        for index, value in enumerate(state_delta_5d):
            max_state_delta_5d[index] = max(max_state_delta_5d[index], value)

        circle_delta_terms = _coefficient_delta_term_magnitudes(
            coefficients,
            baseline_coefficients,
            circle_radius,
        )
        p_slice_delta_terms = _coefficient_delta_term_magnitudes(
            coefficients,
            baseline_coefficients,
            state_4d[0],
        )
        circle_delta_l1_4d = [sum(terms) for terms in circle_delta_terms]
        circle_tail_delta_l1_4d = [sum(terms[ratio_start:]) for terms in circle_delta_terms]
        p_slice_delta_l1_4d = [sum(terms) for terms in p_slice_delta_terms]
        p_slice_tail_delta_l1_4d = [sum(terms[ratio_start:]) for terms in p_slice_delta_terms]
        circle_term_delta_4d = [max(terms, default=0.0) for terms in circle_delta_terms]
        circle_term_delta_relative_4d = []
        for component in range(4):
            relative_terms = [
                circle_delta_terms[component][degree] / baseline_circle_terms[component][degree]
                for degree in range(len(circle_delta_terms[component]))
                if baseline_circle_terms[component][degree] > 0.0
            ]
            circle_term_delta_relative_4d.append(max(relative_terms, default=0.0))
            max_circle_delta_l1_4d[component] = max(
                max_circle_delta_l1_4d[component],
                circle_delta_l1_4d[component],
            )
            max_circle_tail_delta_l1_4d[component] = max(
                max_circle_tail_delta_l1_4d[component],
                circle_tail_delta_l1_4d[component],
            )
            max_circle_delta_component_l1_relative_4d[component] = max(
                max_circle_delta_component_l1_relative_4d[component],
                0.0
                if baseline_circle_l1_4d[component] == 0.0
                else circle_delta_l1_4d[component] / baseline_circle_l1_4d[component],
            )
            max_circle_tail_delta_l1_relative_4d[component] = max(
                max_circle_tail_delta_l1_relative_4d[component],
                0.0
                if baseline_circle_tail_l1_4d[component] == 0.0
                else circle_tail_delta_l1_4d[component] / baseline_circle_tail_l1_4d[component],
            )
            max_circle_term_delta_4d[component] = max(
                max_circle_term_delta_4d[component],
                circle_term_delta_4d[component],
            )
            max_circle_term_delta_relative_4d[component] = max(
                max_circle_term_delta_relative_4d[component],
                circle_term_delta_relative_4d[component],
            )
            max_p_slice_term_delta_l1_4d[component] = max(
                max_p_slice_term_delta_l1_4d[component],
                p_slice_delta_l1_4d[component],
            )
            max_p_slice_tail_delta_l1_4d[component] = max(
                max_p_slice_tail_delta_l1_4d[component],
                p_slice_tail_delta_l1_4d[component],
            )

        rows.append(
            {
                "b": b,
                "state_5d": list(state_5d),
                "state_delta_from_limit_5d": state_delta_5d,
                "state_delta_over_radius_5d": [
                    state_delta_5d[index] / radius0[index]
                    for index in range(5)
                ],
                "circle_delta_l1_4d": circle_delta_l1_4d,
                "circle_tail_delta_l1_4d": circle_tail_delta_l1_4d,
                "circle_delta_l1_relative_to_limit_4d": [
                    0.0
                    if baseline_circle_l1_4d[component] == 0.0
                    else circle_delta_l1_4d[component] / baseline_circle_l1_4d[component]
                    for component in range(4)
                ],
                "circle_tail_delta_l1_relative_to_limit_4d": [
                    0.0
                    if baseline_circle_tail_l1_4d[component] == 0.0
                    else circle_tail_delta_l1_4d[component] / baseline_circle_tail_l1_4d[component]
                    for component in range(4)
                ],
                "max_circle_term_delta_4d": circle_term_delta_4d,
                "max_circle_term_delta_relative_to_limit_4d": circle_term_delta_relative_4d,
                "p_slice_delta_l1_4d": p_slice_delta_l1_4d,
                "p_slice_tail_delta_l1_4d": p_slice_tail_delta_l1_4d,
            }
        )

    max_state_delta_over_radius = [
        max_state_delta_5d[index] / radius0[index]
        for index in range(5)
    ]
    return {
        "status": "finite_b_state_delta_inside_start_radius"
        if all(value < 1.0 for value in max_state_delta_over_radius)
        else "finite_b_state_delta_exceeds_start_radius",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "b_sample_count": b_sample_count,
        "b_samples": list(b_values),
        "target_p": target_p,
        "order": order,
        "ratio_start": ratio_start,
        "working_dps": working_dps,
        "circle_radius": circle_radius,
        "radius0": list(radius0),
        "limit_state_5d": list(baseline_state_5d),
        "baseline_circle_l1_4d": baseline_circle_l1_4d,
        "baseline_circle_tail_l1_4d": baseline_circle_tail_l1_4d,
        "rows": rows,
        "max_state_delta_5d": max_state_delta_5d,
        "max_state_delta_over_radius": max_state_delta_over_radius,
        "max_circle_delta_l1_4d": max_circle_delta_l1_4d,
        "max_circle_tail_delta_l1_4d": max_circle_tail_delta_l1_4d,
        "max_circle_delta_l1_relative_to_limit_4d": max_circle_delta_component_l1_relative_4d,
        "max_circle_tail_delta_l1_relative_to_limit_4d": max_circle_tail_delta_l1_relative_4d,
        "max_circle_term_delta_4d": max_circle_term_delta_4d,
        "max_circle_term_delta_relative_to_limit_4d": max_circle_term_delta_relative_4d,
        "max_p_slice_delta_l1_4d": max_p_slice_term_delta_l1_4d,
        "max_p_slice_tail_delta_l1_4d": max_p_slice_tail_delta_l1_4d,
        "remaining_obligation": (
            "replace sampled endpoint b-sensitivity by an interval or analytic "
            "coefficient perturbation bound uniform over |b|<=1/A"
        ),
    }


def taylor_p_slice_b_cauchy_event_audit(
    target_p: float = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_START,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    order: int = 80,
    working_dps: int = 90,
    b_cauchy_radius: float = 1e-7,
    b_circle_sample_count: int = 8,
    b_outer_cauchy_radius: float | None = None,
    b_outer_circle_sample_count: int | None = None,
    b_enclosure_cauchy_radius: float | None = None,
    b_enclosure_circle_sample_count: int | None = None,
    radius0: tuple[float, float, float, float, float] = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_RADIUS,
    include_direct_endpoints: bool = True,
) -> dict:
    """Sample a complex-b Cauchy bound for the p-slice event map.

    This is the p-slice analogue of ``taylor_b_cauchy_coefficient_audit``.
    For the finite Taylor polynomial, the implicit p-slice map is analytic in
    ``b`` as long as the p-polynomial has a simple target crossing.  Sampling
    a complex circle ``|b|=B`` gives the proof target

        |Y(b)-Y(0)| <= (|b|/B) max_{|z|=B} |Y(z)-Y(0)|,

    where ``Y=(t,x1,x2,x3,C)`` at ``p=target_p``.  The sampled circle maximum
    still has to be replaced by a certified maximum for a proof, but this
    directly measures the finite-A perturbation of the event data that feeds
    the carried-C p-tube.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if not (0.0 < target_p < 1.0):
        raise ValueError("target_p must lie between 0 and 1")
    if order <= 0:
        raise ValueError("order must be positive")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    if b_cauchy_radius <= 0.0:
        raise ValueError("b_cauchy_radius must be positive")
    if b_circle_sample_count <= 0:
        raise ValueError("b_circle_sample_count must be positive")
    if b_outer_cauchy_radius is not None and b_outer_cauchy_radius <= b_cauchy_radius:
        raise ValueError("b_outer_cauchy_radius must be larger than b_cauchy_radius")
    if b_outer_circle_sample_count is not None and b_outer_circle_sample_count <= 0:
        raise ValueError("b_outer_circle_sample_count must be positive")
    if b_enclosure_cauchy_radius is not None and b_outer_cauchy_radius is None:
        raise ValueError("b_enclosure_cauchy_radius requires b_outer_cauchy_radius")
    if (
        b_enclosure_cauchy_radius is not None
        and b_outer_cauchy_radius is not None
        and b_enclosure_cauchy_radius <= b_outer_cauchy_radius
    ):
        raise ValueError("b_enclosure_cauchy_radius must be larger than b_outer_cauchy_radius")
    if b_enclosure_circle_sample_count is not None and b_enclosure_circle_sample_count <= 0:
        raise ValueError("b_enclosure_circle_sample_count must be positive")
    if len(radius0) != 5 or any(value <= 0.0 for value in radius0):
        raise ValueError("radius0 must contain five positive values")

    beta = 1.0 / candidate_a
    if b_cauchy_radius <= beta:
        raise ValueError("b_cauchy_radius must be larger than 1/candidate_a")

    with mp.workdps(working_dps):
        baseline_coefficients = scaled_taylor_coefficients(order, 0.0, working_dps)
        baseline_state_4d = high_order_scaled_taylor_state_at_p_from_coefficients(target_p, baseline_coefficients)
        baseline_state_5d = _p_slice_state_5d(target_p, baseline_state_4d)
        baseline_state_complex = tuple(mp.mpc(value) for value in baseline_state_5d)
        shrink_factor = beta / b_cauchy_radius

        def event_state_for_b(b_value):
            coefficients = complex_scaled_taylor_coefficients(order, b_value, working_dps)
            state_4d = _complex_taylor_state_at_p_from_coefficients(
                target_p,
                coefficients,
                initial_time=baseline_state_4d[0],
                iterations=40,
                tolerance=mp.mpf("1e-45"),
            )
            p_residual = _evaluate_polynomial_complex(coefficients[0], state_4d[0]) - target_p
            p_derivative = _evaluate_polynomial_derivative_complex(coefficients[0], state_4d[0])
            state_5d = (
                state_4d[0],
                state_4d[1],
                state_4d[2],
                state_4d[3],
                state_4d[1] * state_4d[2] - mp.mpf(target_p) ** 2 * state_4d[3] / 6,
            )
            delta = [float(abs(state_5d[index] - baseline_state_complex[index])) for index in range(5)]
            return state_4d, state_5d, p_residual, delta, float(abs(p_derivative))

        sample_rows = []
        max_circle_delta_5d = [0.0 for _ in range(5)]
        max_circle_delta_witness_5d = [None for _ in range(5)]
        event_values_5d = []
        max_residual_abs = 0.0
        min_event_p_derivative_abs = math.inf
        min_event_p_derivative_witness = None
        for sample_index in range(b_circle_sample_count):
            theta = 2 * mp.pi * sample_index / b_circle_sample_count
            b_value = mp.mpc(b_cauchy_radius * mp.cos(theta), b_cauchy_radius * mp.sin(theta))
            state_4d, state_5d, p_residual, delta, p_derivative_abs = event_state_for_b(b_value)
            for index, value in enumerate(delta):
                if value > max_circle_delta_5d[index]:
                    max_circle_delta_5d[index] = value
                    max_circle_delta_witness_5d[index] = {
                        "sample_index": sample_index,
                        "b_real": float(mp.re(b_value)),
                        "b_imag": float(mp.im(b_value)),
                            "delta": value,
                    }
            max_residual_abs = max(max_residual_abs, float(abs(p_residual)))
            if p_derivative_abs < min_event_p_derivative_abs:
                min_event_p_derivative_abs = p_derivative_abs
                min_event_p_derivative_witness = {
                    "circle": "inner",
                    "sample_index": sample_index,
                    "b_real": float(mp.re(b_value)),
                    "b_imag": float(mp.im(b_value)),
                    "event_p_derivative_abs": p_derivative_abs,
                }
            event_values_5d.append(state_5d)
            sample_rows.append(
                {
                    "sample_index": sample_index,
                    "b_real": float(mp.re(b_value)),
                    "b_imag": float(mp.im(b_value)),
                    "event_time_real": float(mp.re(state_4d[0])),
                    "event_time_imag": float(mp.im(state_4d[0])),
                    "p_residual_abs": float(abs(p_residual)),
                    "event_p_derivative_abs": p_derivative_abs,
                    "circle_delta_5d": delta,
                }
            )

        max_adjacent_angular_slope_5d = [0.0 for _ in range(5)]
        if len(event_values_5d) > 1:
            angle_step = 2.0 * math.pi / len(event_values_5d)
            for sample_index, state in enumerate(event_values_5d):
                next_state = event_values_5d[(sample_index + 1) % len(event_values_5d)]
                for component in range(5):
                    slope = float(abs(next_state[component] - state[component])) / angle_step
                    max_adjacent_angular_slope_5d[component] = max(max_adjacent_angular_slope_5d[component], slope)
        empirical_half_step_variation_5d = [
            value * math.pi / b_circle_sample_count
            for value in max_adjacent_angular_slope_5d
        ]
        empirical_circle_delta_bound_5d = [
            max_circle_delta_5d[index] + empirical_half_step_variation_5d[index]
            for index in range(5)
        ]
        cauchy_delta_bound_5d = [shrink_factor * value for value in max_circle_delta_5d]
        cauchy_delta_over_radius = [
            cauchy_delta_bound_5d[index] / radius0[index]
            for index in range(5)
        ]
        empirical_cauchy_delta_bound_5d = [
            shrink_factor * value
            for value in empirical_circle_delta_bound_5d
        ]
        empirical_cauchy_delta_over_radius = [
            empirical_cauchy_delta_bound_5d[index] / radius0[index]
            for index in range(5)
        ]

        outer_sample_rows = []
        outer_max_circle_delta_5d = None
        outer_max_circle_delta_witness_5d = None
        outer_cauchy_angular_derivative_bound_5d = None
        outer_cauchy_half_step_variation_5d = None
        outer_cauchy_circle_delta_bound_5d = None
        outer_cauchy_delta_bound_5d = None
        outer_cauchy_delta_over_radius = None
        enclosure_sample_rows = []
        enclosure_max_circle_delta_5d = None
        enclosure_max_circle_delta_witness_5d = None
        enclosure_cauchy_outer_angular_derivative_bound_5d = None
        enclosure_cauchy_outer_half_step_variation_5d = None
        enclosure_cauchy_outer_circle_delta_bound_5d = None
        if b_outer_cauchy_radius is not None:
            outer_count = b_outer_circle_sample_count or b_circle_sample_count
            outer_max_circle_delta_5d = [0.0 for _ in range(5)]
            outer_max_circle_delta_witness_5d = [None for _ in range(5)]
            for sample_index in range(outer_count):
                theta = 2 * mp.pi * sample_index / outer_count
                b_value = mp.mpc(b_outer_cauchy_radius * mp.cos(theta), b_outer_cauchy_radius * mp.sin(theta))
                state_4d, _state_5d, p_residual, delta, p_derivative_abs = event_state_for_b(b_value)
                for component, value in enumerate(delta):
                    if value > outer_max_circle_delta_5d[component]:
                        outer_max_circle_delta_5d[component] = value
                        outer_max_circle_delta_witness_5d[component] = {
                            "sample_index": sample_index,
                            "b_real": float(mp.re(b_value)),
                            "b_imag": float(mp.im(b_value)),
                            "delta": value,
                        }
                max_residual_abs = max(max_residual_abs, float(abs(p_residual)))
                if p_derivative_abs < min_event_p_derivative_abs:
                    min_event_p_derivative_abs = p_derivative_abs
                    min_event_p_derivative_witness = {
                        "circle": "outer",
                        "sample_index": sample_index,
                        "b_real": float(mp.re(b_value)),
                        "b_imag": float(mp.im(b_value)),
                        "event_p_derivative_abs": p_derivative_abs,
                    }
                outer_sample_rows.append(
                    {
                        "sample_index": sample_index,
                        "b_real": float(mp.re(b_value)),
                        "b_imag": float(mp.im(b_value)),
                        "event_time_real": float(mp.re(state_4d[0])),
                        "event_time_imag": float(mp.im(state_4d[0])),
                        "p_residual_abs": float(abs(p_residual)),
                        "event_p_derivative_abs": p_derivative_abs,
                        "circle_delta_5d": delta,
                    }
                )
            outer_circle_bound_for_inner_5d = list(outer_max_circle_delta_5d)
            if b_enclosure_cauchy_radius is not None:
                enclosure_count = b_enclosure_circle_sample_count or outer_count
                enclosure_max_circle_delta_5d = [0.0 for _ in range(5)]
                enclosure_max_circle_delta_witness_5d = [None for _ in range(5)]
                for sample_index in range(enclosure_count):
                    theta = 2 * mp.pi * sample_index / enclosure_count
                    b_value = mp.mpc(
                        b_enclosure_cauchy_radius * mp.cos(theta),
                        b_enclosure_cauchy_radius * mp.sin(theta),
                    )
                    state_4d, _state_5d, p_residual, delta, p_derivative_abs = event_state_for_b(b_value)
                    for component, value in enumerate(delta):
                        if value > enclosure_max_circle_delta_5d[component]:
                            enclosure_max_circle_delta_5d[component] = value
                            enclosure_max_circle_delta_witness_5d[component] = {
                                "sample_index": sample_index,
                                "b_real": float(mp.re(b_value)),
                                "b_imag": float(mp.im(b_value)),
                                "delta": value,
                            }
                    max_residual_abs = max(max_residual_abs, float(abs(p_residual)))
                    if p_derivative_abs < min_event_p_derivative_abs:
                        min_event_p_derivative_abs = p_derivative_abs
                        min_event_p_derivative_witness = {
                            "circle": "enclosure",
                            "sample_index": sample_index,
                            "b_real": float(mp.re(b_value)),
                            "b_imag": float(mp.im(b_value)),
                            "event_p_derivative_abs": p_derivative_abs,
                        }
                    enclosure_sample_rows.append(
                        {
                            "sample_index": sample_index,
                            "b_real": float(mp.re(b_value)),
                            "b_imag": float(mp.im(b_value)),
                            "event_time_real": float(mp.re(state_4d[0])),
                            "event_time_imag": float(mp.im(state_4d[0])),
                            "p_residual_abs": float(abs(p_residual)),
                            "event_p_derivative_abs": p_derivative_abs,
                            "circle_delta_5d": delta,
                        }
                    )
                enclosure_gap = b_enclosure_cauchy_radius - b_outer_cauchy_radius
                enclosure_cauchy_outer_angular_derivative_bound_5d = [
                    b_outer_cauchy_radius * value / enclosure_gap
                    for value in enclosure_max_circle_delta_5d
                ]
                enclosure_cauchy_outer_half_step_variation_5d = [
                    value * math.pi / outer_count
                    for value in enclosure_cauchy_outer_angular_derivative_bound_5d
                ]
                enclosure_cauchy_outer_circle_delta_bound_5d = [
                    outer_max_circle_delta_5d[index] + enclosure_cauchy_outer_half_step_variation_5d[index]
                    for index in range(5)
                ]
                outer_circle_bound_for_inner_5d = enclosure_cauchy_outer_circle_delta_bound_5d
            outer_gap = b_outer_cauchy_radius - b_cauchy_radius
            outer_cauchy_angular_derivative_bound_5d = [
                b_cauchy_radius * value / outer_gap
                for value in outer_circle_bound_for_inner_5d
            ]
            outer_cauchy_half_step_variation_5d = [
                value * math.pi / b_circle_sample_count
                for value in outer_cauchy_angular_derivative_bound_5d
            ]
            outer_cauchy_circle_delta_bound_5d = [
                max_circle_delta_5d[index] + outer_cauchy_half_step_variation_5d[index]
                for index in range(5)
            ]
            outer_cauchy_delta_bound_5d = [
                shrink_factor * value
                for value in outer_cauchy_circle_delta_bound_5d
            ]
            outer_cauchy_delta_over_radius = [
                outer_cauchy_delta_bound_5d[index] / radius0[index]
                for index in range(5)
            ]

        proof_cauchy_delta_bound_5d = (
            outer_cauchy_delta_bound_5d
            if outer_cauchy_delta_bound_5d is not None
            else cauchy_delta_bound_5d
        )
        proof_cauchy_delta_over_radius = (
            outer_cauchy_delta_over_radius
            if outer_cauchy_delta_over_radius is not None
            else cauchy_delta_over_radius
        )
        proof_cauchy_source = (
            "sampled_enclosure_circle_nested_cauchy"
            if enclosure_cauchy_outer_circle_delta_bound_5d is not None
            else (
                "sampled_outer_circle_cauchy_angular_bound"
                if outer_cauchy_delta_bound_5d is not None
                else "sampled_inner_circle_max"
            )
        )

        direct_rows = []
        max_direct_delta_5d = None
        if include_direct_endpoints:
            max_direct_delta_5d = [0.0 for _ in range(5)]
            for b in (-beta, beta):
                state_4d = high_order_scaled_taylor_state_at_p(target_p, b, order=order, working_dps=working_dps)
                state_5d = _p_slice_state_5d(target_p, state_4d)
                delta = [abs(state_5d[index] - baseline_state_5d[index]) for index in range(5)]
                for index, value in enumerate(delta):
                    max_direct_delta_5d[index] = max(max_direct_delta_5d[index], value)
                direct_rows.append(
                    {
                        "b": b,
                        "state_5d": list(state_5d),
                        "delta_from_limit_5d": delta,
                    }
                )

    return {
        "status": "sampled_b_cauchy_event_delta_inside_start_radius"
        if all(value < 1.0 for value in proof_cauchy_delta_over_radius)
        else "sampled_b_cauchy_event_delta_exceeds_start_radius",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "b_cauchy_radius": b_cauchy_radius,
        "b_shrink_factor": shrink_factor,
        "b_circle_sample_count": b_circle_sample_count,
        "b_outer_cauchy_radius": b_outer_cauchy_radius,
        "b_outer_circle_sample_count": b_outer_circle_sample_count,
        "b_enclosure_cauchy_radius": b_enclosure_cauchy_radius,
        "b_enclosure_circle_sample_count": b_enclosure_circle_sample_count,
        "target_p": target_p,
        "order": order,
        "working_dps": working_dps,
        "radius0": list(radius0),
        "limit_state_5d": list(baseline_state_5d),
        "sample_rows": sample_rows,
        "max_circle_delta_5d": max_circle_delta_5d,
        "max_circle_delta_witness_5d": max_circle_delta_witness_5d,
        "max_adjacent_angular_slope_5d": max_adjacent_angular_slope_5d,
        "empirical_half_step_variation_5d": empirical_half_step_variation_5d,
        "empirical_circle_delta_bound_5d": empirical_circle_delta_bound_5d,
        "cauchy_delta_bound_5d": cauchy_delta_bound_5d,
        "cauchy_delta_bound_over_radius": cauchy_delta_over_radius,
        "empirical_cauchy_delta_bound_5d": empirical_cauchy_delta_bound_5d,
        "empirical_cauchy_delta_bound_over_radius": empirical_cauchy_delta_over_radius,
        "outer_sample_rows": outer_sample_rows,
        "outer_max_circle_delta_5d": outer_max_circle_delta_5d,
        "outer_max_circle_delta_witness_5d": outer_max_circle_delta_witness_5d,
        "outer_cauchy_angular_derivative_bound_5d": outer_cauchy_angular_derivative_bound_5d,
        "outer_cauchy_half_step_variation_5d": outer_cauchy_half_step_variation_5d,
        "outer_cauchy_circle_delta_bound_5d": outer_cauchy_circle_delta_bound_5d,
        "outer_cauchy_delta_bound_5d": outer_cauchy_delta_bound_5d,
        "outer_cauchy_delta_bound_over_radius": outer_cauchy_delta_over_radius,
        "enclosure_sample_rows": enclosure_sample_rows,
        "enclosure_max_circle_delta_5d": enclosure_max_circle_delta_5d,
        "enclosure_max_circle_delta_witness_5d": enclosure_max_circle_delta_witness_5d,
        "enclosure_cauchy_outer_angular_derivative_bound_5d": enclosure_cauchy_outer_angular_derivative_bound_5d,
        "enclosure_cauchy_outer_half_step_variation_5d": enclosure_cauchy_outer_half_step_variation_5d,
        "enclosure_cauchy_outer_circle_delta_bound_5d": enclosure_cauchy_outer_circle_delta_bound_5d,
        "proof_cauchy_delta_bound_5d": proof_cauchy_delta_bound_5d,
        "proof_cauchy_delta_bound_over_radius": proof_cauchy_delta_over_radius,
        "proof_cauchy_source": proof_cauchy_source,
        "max_p_residual_abs": max_residual_abs,
        "min_event_p_derivative_abs": min_event_p_derivative_abs,
        "min_event_p_derivative_witness": min_event_p_derivative_witness,
        "include_direct_endpoints": include_direct_endpoints,
        "direct_endpoint_rows": direct_rows,
        "max_direct_delta_5d": max_direct_delta_5d,
        "max_direct_delta_over_radius": None
        if max_direct_delta_5d is None
        else [
            max_direct_delta_5d[index] / radius0[index]
            for index in range(5)
        ],
        "remaining_obligation": (
            "replace sampled complex-b event-map maxima by certified maxima; "
            "this controls only the finite Taylor polynomial, so it must still "
            "be combined with the Taylor tail/event-time majorant"
        ),
    }


def taylor_p_slice_entry_budget_audit(
    target_p: float = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_START,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    order: int = 60,
    tail_start: int = 50,
    ratio_start: int = 45,
    ratio_bound: float | None = 0.53,
    b_sample_count: int = 3,
    working_dps: int = 90,
    b_cauchy_radius: float = 1e-7,
    b_circle_sample_count: int = 4,
    b_outer_cauchy_radius: float | None = None,
    b_outer_circle_sample_count: int | None = None,
    b_enclosure_cauchy_radius: float | None = None,
    b_enclosure_circle_sample_count: int | None = None,
    radius0: tuple[float, float, float, float, float] = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_RADIUS,
    include_direct_endpoints: bool = True,
) -> dict:
    """Combine finite-b event and Taylor-tail budgets for the p-slice handoff.

    The start box for the carried-C p-tube is expressed in
    ``(t,x1,x2,x3,C)``.  The formal Taylor tail audit estimates the residual
    in the Taylor coordinates, including a p-coordinate tail.  This combined
    diagnostic uses the induced p-event time-shift bound for the ``t`` budget,
    then uses the x/C tail estimates for the remaining coordinates.  It is
    still a proof-target diagnostic: both the complex-b event maximum and the
    late Taylor ratio/majorant must be certified before this becomes a proof.
    """
    if len(radius0) != 5 or any(value <= 0.0 for value in radius0):
        raise ValueError("radius0 must contain five positive values")

    tail_audit = taylor_p_slice_tail_ratio_audit(
        target_p=target_p,
        candidate_a=candidate_a,
        order=order,
        tail_start=tail_start,
        ratio_start=ratio_start,
        ratio_bound=ratio_bound,
        b_sample_count=b_sample_count,
        working_dps=working_dps,
        radius0=radius0,
    )
    event_audit = taylor_p_slice_b_cauchy_event_audit(
        target_p=target_p,
        candidate_a=candidate_a,
        order=order,
        working_dps=working_dps,
        b_cauchy_radius=b_cauchy_radius,
        b_circle_sample_count=b_circle_sample_count,
        b_outer_cauchy_radius=b_outer_cauchy_radius,
        b_outer_circle_sample_count=b_outer_circle_sample_count,
        b_enclosure_cauchy_radius=b_enclosure_cauchy_radius,
        b_enclosure_circle_sample_count=b_enclosure_circle_sample_count,
        radius0=radius0,
        include_direct_endpoints=include_direct_endpoints,
    )

    tail_raw = tail_audit["max_tail_estimate_5d"]
    tail_budget_5d = [
        tail_audit["max_time_shift_bound_from_p_tail"],
        tail_raw[1],
        tail_raw[2],
        tail_raw[3],
        tail_raw[4],
    ]
    finite_b_budget_5d = event_audit["proof_cauchy_delta_bound_5d"]
    combined_budget_5d = [
        finite_b_budget_5d[index] + tail_budget_5d[index]
        for index in range(5)
    ]
    tail_budget_over_radius = [
        tail_budget_5d[index] / radius0[index]
        for index in range(5)
    ]
    finite_b_budget_over_radius = [
        finite_b_budget_5d[index] / radius0[index]
        for index in range(5)
    ]
    combined_budget_over_radius = [
        combined_budget_5d[index] / radius0[index]
        for index in range(5)
    ]
    max_combined_ratio = max(combined_budget_over_radius)
    return {
        "status": "formal_entry_budget_inside_start_radius"
        if max_combined_ratio < 1.0
        else "formal_entry_budget_exceeds_start_radius",
        "candidate_A": candidate_a,
        "target_p": target_p,
        "order": order,
        "tail_start": tail_start,
        "ratio_start": ratio_start,
        "ratio_bound": ratio_bound,
        "b_sample_count": b_sample_count,
        "working_dps": working_dps,
        "b_cauchy_radius": b_cauchy_radius,
        "b_circle_sample_count": b_circle_sample_count,
        "b_outer_cauchy_radius": b_outer_cauchy_radius,
        "b_outer_circle_sample_count": b_outer_circle_sample_count,
        "b_enclosure_cauchy_radius": b_enclosure_cauchy_radius,
        "b_enclosure_circle_sample_count": b_enclosure_circle_sample_count,
        "radius0": list(radius0),
        "tail_budget_5d": tail_budget_5d,
        "tail_budget_over_radius": tail_budget_over_radius,
        "finite_b_budget_5d": finite_b_budget_5d,
        "finite_b_budget_over_radius": finite_b_budget_over_radius,
        "combined_budget_5d": combined_budget_5d,
        "combined_budget_over_radius": combined_budget_over_radius,
        "max_combined_budget_over_radius": max_combined_ratio,
        "observed_ratios_inside_bound": tail_audit["observed_ratios_inside_bound"],
        "event_cauchy_status": event_audit["status"],
        "event_cauchy_source": event_audit["proof_cauchy_source"],
        "tail_status": tail_audit["status"],
        "tail_audit": tail_audit,
        "event_audit": event_audit,
        "remaining_obligation": (
            "certify the complex-b event-map maximum and replace the formal "
            "Taylor ratio tail by a rigorous same-parity majorant; if those "
            "two inputs hold, the p-slice entry lies inside the carried-C "
            "start box with the reported radius budget"
        ),
    }


def taylor_p_slice_required_a_audit(
    target_p: float = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_START,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    order: int = 60,
    tail_start: int = 50,
    ratio_start: int = 45,
    ratio_bound: float | None = 0.6,
    b_sample_count: int = 3,
    working_dps: int = 90,
    b_cauchy_radius: float = 1e-7,
    b_circle_sample_count: int = 4,
    b_outer_cauchy_radius: float | None = None,
    b_outer_circle_sample_count: int | None = None,
    b_enclosure_cauchy_radius: float | None = None,
    b_enclosure_circle_sample_count: int | None = None,
    radius0: tuple[float, float, float, float, float] = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_RADIUS,
    include_direct_endpoints: bool = True,
) -> dict:
    """Return the explicit ``A`` threshold implied by the p-slice budget.

    The finite-``b`` event-map budget from ``taylor_p_slice_entry_budget_audit``
    scales linearly with ``beta = 1/A`` once the complex-``b`` proof circle is
    fixed.  This audit records the minimum ``A`` that would make the combined
    p-slice handoff fit, conditional on the supplied Taylor-tail and event-map
    majorants being certified.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    entry = taylor_p_slice_entry_budget_audit(
        target_p=target_p,
        candidate_a=candidate_a,
        order=order,
        tail_start=tail_start,
        ratio_start=ratio_start,
        ratio_bound=ratio_bound,
        b_sample_count=b_sample_count,
        working_dps=working_dps,
        b_cauchy_radius=b_cauchy_radius,
        b_circle_sample_count=b_circle_sample_count,
        b_outer_cauchy_radius=b_outer_cauchy_radius,
        b_outer_circle_sample_count=b_outer_circle_sample_count,
        b_enclosure_cauchy_radius=b_enclosure_cauchy_radius,
        b_enclosure_circle_sample_count=b_enclosure_circle_sample_count,
        radius0=radius0,
        include_direct_endpoints=include_direct_endpoints,
    )

    tail_ratios = entry["tail_budget_over_radius"]
    finite_ratios = entry["finite_b_budget_over_radius"]
    component_rows = []
    required_a = 0.0
    finite_impossible = False
    for index, (tail_ratio, finite_ratio) in enumerate(zip(tail_ratios, finite_ratios)):
        remaining_radius_ratio = 1.0 - tail_ratio
        if remaining_radius_ratio <= 0.0:
            component_required = math.inf
            finite_impossible = True
        elif finite_ratio <= 0.0:
            component_required = 0.0
        else:
            component_required = candidate_a * finite_ratio / remaining_radius_ratio
        required_a = max(required_a, component_required)
        component_rows.append(
            {
                "component": ("t", "x1", "x2", "x3", "C")[index],
                "tail_budget_over_radius": tail_ratio,
                "finite_b_budget_over_radius_at_candidate_A": finite_ratio,
                "combined_budget_over_radius_at_candidate_A": entry["combined_budget_over_radius"][index],
                "remaining_radius_ratio_after_tail": remaining_radius_ratio,
                "required_A_from_component": component_required,
            }
        )

    if finite_impossible:
        status = "tail_budget_exceeds_start_radius"
        headroom = 0.0
    elif candidate_a > required_a:
        status = "candidate_A_fits_conditional_entry_budget"
        headroom = math.inf if required_a == 0.0 else candidate_a / required_a
    else:
        status = "candidate_A_below_conditional_entry_threshold"
        headroom = 0.0 if required_a == 0.0 else candidate_a / required_a

    return {
        "status": status,
        "candidate_A": candidate_a,
        "minimum_A_for_conditional_entry_budget": required_a,
        "candidate_A_headroom_factor": headroom,
        "max_tail_budget_over_radius": max(tail_ratios),
        "max_finite_b_budget_over_radius_at_candidate_A": max(finite_ratios),
        "max_combined_budget_over_radius_at_candidate_A": entry["max_combined_budget_over_radius"],
        "event_cauchy_source": entry["event_cauchy_source"],
        "event_cauchy_status": entry["event_cauchy_status"],
        "tail_status": entry["tail_status"],
        "component_rows": component_rows,
        "entry_budget_audit": entry,
        "remaining_obligation": (
            "The threshold is conditional: it becomes a proof-level explicit A "
            "only after the Taylor-tail majorant and complex-b event-map maximum "
            "used by the entry budget are certified."
        ),
    }


def taylor_support_time_convergence_audit(
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    support_time: float = DEFAULT_SUPPORT_TIME,
    low_order: int = 60,
    high_order: int = 80,
    b_sample_count: int = 3,
    working_dps: int = 90,
    support_radius0: tuple[float, float, float, float] = DEFAULT_SUPPORT_TUBE_RADIUS,
) -> dict:
    """Compare Taylor orders at the support time against a support box.

    This is the support-time analogue of the p-slice convergence audit.  It is
    a proof-target diagnostic: passing means the observed change between two
    Taylor truncation orders at ``t=support_time`` fits inside the proposed
    support radius for every sampled ``b`` value.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if support_time <= 0.0:
        raise ValueError("support_time must be positive")
    if low_order < 0 or high_order <= low_order:
        raise ValueError("orders must satisfy 0 <= low_order < high_order")
    if b_sample_count < 3 or b_sample_count % 2 == 0:
        raise ValueError("b_sample_count must be an odd integer at least 3")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    if len(support_radius0) != 4 or any(value <= 0.0 for value in support_radius0):
        raise ValueError("support_radius0 must contain four positive values")

    beta = 1.0 / candidate_a
    b_values = _symmetric_b_grid(candidate_a, b_sample_count)
    rows = []
    high_samples = []
    max_order_difference = [0.0 for _ in range(4)]
    for b in b_values:
        low_coefficients = scaled_taylor_coefficients(low_order, b, working_dps)
        high_coefficients = scaled_taylor_coefficients(high_order, b, working_dps)
        low_state = evaluate_scaled_taylor_coefficients(low_coefficients, support_time)
        high_state = evaluate_scaled_taylor_coefficients(high_coefficients, support_time)
        difference = [abs(high_state[index] - low_state[index]) for index in range(4)]
        for index in range(4):
            max_order_difference[index] = max(max_order_difference[index], difference[index])
        high_samples.append(high_state)
        rows.append(
            {
                "b": b,
                "low_order_state_4d": list(low_state),
                "high_order_state_4d": list(high_state),
                "order_difference_4d": difference,
                "order_difference_over_support_radius": [
                    difference[index] / support_radius0[index]
                    for index in range(4)
                ],
            }
        )

    high_sample_low = [
        min(sample[index] for sample in high_samples)
        for index in range(4)
    ]
    high_sample_high = [
        max(sample[index] for sample in high_samples)
        for index in range(4)
    ]
    support_box_low = [
        high_sample_low[index] - support_radius0[index]
        for index in range(4)
    ]
    support_box_high = [
        high_sample_high[index] + support_radius0[index]
        for index in range(4)
    ]
    max_difference_over_radius = [
        max_order_difference[index] / support_radius0[index]
        for index in range(4)
    ]
    return {
        "status": "observed_support_time_convergence_inside_radius"
        if all(value < 1.0 for value in max_difference_over_radius)
        else "observed_support_time_convergence_exceeds_radius",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "b_sample_count": b_sample_count,
        "b_samples": list(b_values),
        "support_time": support_time,
        "low_order": low_order,
        "high_order": high_order,
        "working_dps": working_dps,
        "support_radius0": list(support_radius0),
        "rows": rows,
        "high_order_sample_hull_4d": {"low": high_sample_low, "high": high_sample_high},
        "sampled_support_box_4d": {"low": support_box_low, "high": support_box_high},
        "max_order_difference_4d": max_order_difference,
        "max_order_difference_over_support_radius": max_difference_over_radius,
        "remaining_obligation": (
            "replace observed support-time order convergence by a rigorous "
            "Taylor remainder or coefficient-majorant bound at t=3.5"
        ),
    }


def taylor_b_cauchy_coefficient_audit(
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    order: int = 40,
    working_dps: int = 90,
    time_radius: float = DEFAULT_SUPPORT_TIME,
    b_cauchy_radius: float = 1e-7,
    b_circle_sample_count: int = 8,
    support_radius0: tuple[float, float, float, float] = DEFAULT_SUPPORT_TUBE_RADIUS,
    include_direct_endpoints: bool = True,
) -> dict:
    """Use a sampled complex-b Cauchy estimate for finite coefficient motion.

    For each Taylor coefficient ``c_n(b)`` this samples
    ``g_n(b)=c_n(b)-c_n(0)`` on ``|b|=B``.  If the sampled circle maximum were
    replaced by a certified circle maximum, the maximum-modulus principle for
    ``g_n(b)/b`` would give

        |g_n(b)| <= (|b|/B) max_{|z|=B} |g_n(z)|

    for all ``|b| <= 1/A``.  This is therefore a proof-target diagnostic for
    avoiding the raw interval-coefficient wrapping.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if order <= 0:
        raise ValueError("order must be positive")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    if time_radius <= 0.0:
        raise ValueError("time_radius must be positive")
    if b_cauchy_radius <= 0.0:
        raise ValueError("b_cauchy_radius must be positive")
    if b_circle_sample_count <= 0:
        raise ValueError("b_circle_sample_count must be positive")
    if len(support_radius0) != 4 or any(value <= 0.0 for value in support_radius0):
        raise ValueError("support_radius0 must contain four positive values")

    beta = 1.0 / candidate_a
    if b_cauchy_radius <= beta:
        raise ValueError("b_cauchy_radius must be larger than 1/candidate_a")

    baseline_coefficients = scaled_taylor_coefficients(order, 0.0, working_dps)
    baseline_state = evaluate_scaled_taylor_coefficients(baseline_coefficients, time_radius)
    time_value = mp.mpf(time_radius)
    shrink_factor = beta / b_cauchy_radius
    max_circle_delta_terms_4d = [[0.0 for _ in range(order + 1)] for _ in range(4)]
    max_circle_delta_l1_4d = [0.0 for _ in range(4)]
    sample_rows = []
    for sample_index in range(b_circle_sample_count):
        theta = 2 * mp.pi * sample_index / b_circle_sample_count
        b_value = mp.mpc(b_cauchy_radius * mp.cos(theta), b_cauchy_radius * mp.sin(theta))
        coefficients = complex_scaled_taylor_coefficients(order, b_value, working_dps)
        row_delta_l1 = [0.0 for _ in range(4)]
        row_max_term = [0.0 for _ in range(4)]
        for component in range(4):
            for degree in range(order + 1):
                delta = abs(
                    (coefficients[component][degree] - baseline_coefficients[component][degree])
                    * (time_value ** degree)
                )
                delta_float = float(delta)
                max_circle_delta_terms_4d[component][degree] = max(
                    max_circle_delta_terms_4d[component][degree],
                    delta_float,
                )
                row_delta_l1[component] += delta_float
                row_max_term[component] = max(row_max_term[component], delta_float)
        for component in range(4):
            max_circle_delta_l1_4d[component] = max(max_circle_delta_l1_4d[component], row_delta_l1[component])
        sample_rows.append(
            {
                "sample_index": sample_index,
                "b_real": float(mp.re(b_value)),
                "b_imag": float(mp.im(b_value)),
                "circle_delta_l1_4d": row_delta_l1,
                "circle_max_term_delta_4d": row_max_term,
            }
        )

    cauchy_term_bounds_4d = [
        [shrink_factor * value for value in terms]
        for terms in max_circle_delta_terms_4d
    ]
    cauchy_delta_bound_4d = [sum(terms) for terms in cauchy_term_bounds_4d]
    cauchy_delta_bound_over_support_radius = [
        cauchy_delta_bound_4d[component] / support_radius0[component]
        for component in range(4)
    ]

    direct_rows = []
    max_direct_delta_4d = None
    if include_direct_endpoints:
        max_direct_delta_4d = [0.0 for _ in range(4)]
        for b in (-beta, beta):
            coefficients = scaled_taylor_coefficients(order, b, working_dps)
            state = evaluate_scaled_taylor_coefficients(coefficients, time_radius)
            delta = [abs(state[component] - baseline_state[component]) for component in range(4)]
            for component in range(4):
                max_direct_delta_4d[component] = max(max_direct_delta_4d[component], delta[component])
            direct_rows.append(
                {
                    "b": b,
                    "state_4d": list(state),
                    "delta_from_limit_4d": delta,
                }
            )

    return {
        "status": "sampled_b_cauchy_delta_inside_support_radius"
        if all(value < 1.0 for value in cauchy_delta_bound_over_support_radius)
        else "sampled_b_cauchy_delta_exceeds_support_radius",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "b_cauchy_radius": b_cauchy_radius,
        "b_shrink_factor": shrink_factor,
        "b_circle_sample_count": b_circle_sample_count,
        "order": order,
        "working_dps": working_dps,
        "time_radius": time_radius,
        "support_radius0": list(support_radius0),
        "limit_state_4d": list(baseline_state),
        "sample_rows": sample_rows,
        "include_direct_endpoints": include_direct_endpoints,
        "direct_endpoint_rows": direct_rows,
        "max_direct_delta_4d": max_direct_delta_4d,
        "max_direct_delta_over_support_radius": None
        if max_direct_delta_4d is None
        else [
            max_direct_delta_4d[component] / support_radius0[component]
            for component in range(4)
        ],
        "max_circle_delta_l1_4d": max_circle_delta_l1_4d,
        "cauchy_delta_bound_4d": cauchy_delta_bound_4d,
        "cauchy_delta_bound_over_support_radius": cauchy_delta_bound_over_support_radius,
        "max_cauchy_term_bound_4d": [max(terms, default=0.0) for terms in cauchy_term_bounds_4d],
        "remaining_obligation": (
            "replace sampled complex-b circle maxima by certified maxima and "
            "combine this finite-degree b-perturbation bound with a Taylor tail "
            "majorant in the t variable"
        ),
    }


def taylor_circle_residual_audit(
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    order: int = 80,
    b_sample_count: int = 3,
    working_dps: int = 90,
    circle_radius: float = 3.5,
    circle_sample_count: int = 120,
) -> dict:
    """Sample the equation residual of the Taylor polynomial on a circle.

    For a degree-N Taylor polynomial P this reports

        t * P'(t) - t * f(t, P(t), b)

    on ``|t| = circle_radius``.  It is a diagnostic for a later
    a-posteriori analytic validation; it is not yet an interval proof.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if order <= 0:
        raise ValueError("order must be positive")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    if circle_radius <= 0.0:
        raise ValueError("circle_radius must be positive")
    if circle_sample_count <= 0:
        raise ValueError("circle_sample_count must be positive")

    beta = 1.0 / candidate_a
    b_values = _symmetric_b_grid(candidate_a, b_sample_count)
    rows = []
    max_residual_4d = [0.0 for _ in range(4)]
    max_state_abs_4d = [0.0 for _ in range(4)]
    min_p_abs = math.inf
    for b in b_values:
        coefficients = scaled_taylor_coefficients(order, b, working_dps)
        row_max_residual = [0.0 for _ in range(4)]
        row_max_state_abs = [0.0 for _ in range(4)]
        row_min_p_abs = math.inf
        worst_residual = None
        for sample_index in range(circle_sample_count):
            theta = 2 * mp.pi * sample_index / circle_sample_count
            point = mp.mpc(circle_radius * mp.cos(theta), circle_radius * mp.sin(theta))
            state = tuple(_evaluate_polynomial_complex(component, point) for component in coefficients)
            derivative = tuple(_evaluate_polynomial_derivative_complex(component, point) for component in coefficients)
            rhs = scaled_rhs_with_b(point, state, b)
            residual = tuple(point * (derivative[index] - rhs[index]) for index in range(4))
            residual_abs = [float(abs(value)) for value in residual]
            state_abs = [float(abs(value)) for value in state]
            for component in range(4):
                row_max_residual[component] = max(row_max_residual[component], residual_abs[component])
                row_max_state_abs[component] = max(row_max_state_abs[component], state_abs[component])
                max_residual_4d[component] = max(max_residual_4d[component], residual_abs[component])
                max_state_abs_4d[component] = max(max_state_abs_4d[component], state_abs[component])
            p_abs = float(abs(state[0]))
            row_min_p_abs = min(row_min_p_abs, p_abs)
            min_p_abs = min(min_p_abs, p_abs)
            sample_residual = max(residual_abs)
            if worst_residual is None or sample_residual > worst_residual["max_residual"]:
                worst_residual = {
                    "sample_index": sample_index,
                    "theta": float(theta),
                    "max_residual": sample_residual,
                    "residual_4d": residual_abs,
                    "state_abs_4d": state_abs,
                }
        rows.append(
            {
                "b": b,
                "max_residual_4d": row_max_residual,
                "max_state_abs_4d": row_max_state_abs,
                "min_p_abs": row_min_p_abs,
                "worst_residual_sample": worst_residual,
            }
        )

    max_residual = max(max_residual_4d)
    return {
        "status": "sampled_residual_small"
        if max_residual < 1e-6
        else "sampled_residual_not_small",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "b_sample_count": b_sample_count,
        "b_samples": list(b_values),
        "order": order,
        "working_dps": working_dps,
        "circle_radius": circle_radius,
        "circle_sample_count": circle_sample_count,
        "max_residual_4d": max_residual_4d,
        "max_residual": max_residual,
        "max_state_abs_4d": max_state_abs_4d,
        "min_p_abs": min_p_abs,
        "rows": rows,
        "remaining_obligation": (
            "replace sampled residuals by interval circle bounds and combine "
            "with an invertibility/Lipschitz estimate for an analytic "
            "a-posteriori theorem"
        ),
    }


def interval_taylor_finite_ratio_audit(
    target_p: float = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_START,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    order: int = 70,
    ratio_start: int = 45,
    ratio_bound: float = 0.53,
    b_subdivisions: int = 1,
    working_dps: int = 80,
    time_padding: float = 1e-8,
) -> dict:
    """Check finite same-parity Taylor ratios using interval coefficients.

    This strengthens the finite-window part of the tail-majorant target from
    three sampled b-values to interval coefficients for every ``|b| <= 1/A``.
    It is still conditional on the supplied time hull containing the true
    p-slice event times and does not prove the infinite tail after ``order``.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if not (0.0 < target_p < 1.0):
        raise ValueError("target_p must lie between 0 and 1")
    if order <= 2:
        raise ValueError("order must be greater than 2")
    if not (0 <= ratio_start < order - 1):
        raise ValueError("ratio_start must leave at least one finite ratio")
    if not (0.0 <= ratio_bound < 1.0):
        raise ValueError("ratio_bound must lie in [0,1)")
    if b_subdivisions <= 0:
        raise ValueError("b_subdivisions must be positive")
    if working_dps <= 0:
        raise ValueError("working_dps must be positive")
    if time_padding < 0.0:
        raise ValueError("time_padding must be nonnegative")

    beta = 1.0 / candidate_a
    sample_states = [
        high_order_scaled_taylor_state_at_p(target_p, b, order=min(order, 60), working_dps=working_dps)
        for b in (-beta, 0.0, beta)
    ]
    time_low = min(state[0] for state in sample_states) - time_padding
    time_high = max(state[0] for state in sample_states) + time_padding
    if time_low <= 0.0:
        raise ValueError("time padding makes the p-slice time hull nonpositive")

    component_max_ratios = []
    component_worst = []
    skipped_zero_denominator = []
    component_max_ratios = [0.0 for _ in range(4)]
    component_worst = [None for _ in range(4)]
    skipped_zero_denominator = [0 for _ in range(4)]
    subintervals = []
    failed_subinterval = None
    for split_index in range(b_subdivisions):
        b_low = -beta + (2 * beta) * split_index / b_subdivisions
        b_high = -beta + (2 * beta) * (split_index + 1) / b_subdivisions
        try:
            coefficients = interval_scaled_taylor_coefficients(
                order,
                candidate_a=candidate_a,
                working_dps=working_dps,
                b_range=(b_low, b_high),
            )
        except (ArithmeticError, ZeroDivisionError, ValueError) as exc:
            failed_subinterval = {
                "index": split_index,
                "b_range": [b_low, b_high],
                "failure": str(exc),
            }
            break
        sub_max = [0.0 for _ in range(4)]
        for component in range(4):
            for parity in (0, 1):
                first_degree = ratio_start
                if first_degree % 2 != parity:
                    first_degree += 1
                for degree in range(first_degree, order - 1, 2):
                    denominator = _interval_abs_lower(coefficients[component][degree]) * (time_low ** degree)
                    numerator = _interval_abs_upper(coefficients[component][degree + 2]) * (time_high ** (degree + 2))
                    if denominator == 0.0:
                        skipped_zero_denominator[component] += 1
                        continue
                    ratio = numerator / denominator
                    if ratio > sub_max[component]:
                        sub_max[component] = ratio
                    if ratio > component_max_ratios[component]:
                        component_max_ratios[component] = ratio
                        component_worst[component] = {
                            "subinterval_index": split_index,
                            "b_range": [b_low, b_high],
                            "degree": degree,
                            "next_degree": degree + 2,
                            "ratio_upper": ratio,
                            "numerator_upper": numerator,
                            "denominator_lower": denominator,
                            "coefficient_interval": [
                                float(coefficients[component][degree].a),
                                float(coefficients[component][degree].b),
                            ],
                            "next_coefficient_interval": [
                                float(coefficients[component][degree + 2].a),
                                float(coefficients[component][degree + 2].b),
                            ],
                        }
        subintervals.append({"index": split_index, "b_range": [b_low, b_high], "component_max_ratio_upper": sub_max})

    max_ratio = max(component_max_ratios)
    if failed_subinterval is not None:
        return {
            "status": "interval_finite_ratios_failed",
            "candidate_A": candidate_a,
            "b_interval": [-beta, beta],
            "target_p": target_p,
            "order": order,
            "ratio_start": ratio_start,
            "ratio_bound": ratio_bound,
            "b_subdivisions": b_subdivisions,
            "working_dps": working_dps,
            "time_padding": time_padding,
            "sample_time_hull": [min(state[0] for state in sample_states), max(state[0] for state in sample_states)],
            "checked_time_hull": [time_low, time_high],
            "failed_subinterval": failed_subinterval,
            "subintervals_completed": subintervals,
            "remaining_obligation": (
                "interval Taylor coefficient recurrence became overconservative; "
                "increase b_subdivisions or use a sharper coefficient enclosure"
            ),
        }
    return {
        "status": "interval_finite_ratios_inside_bound"
        if max_ratio <= ratio_bound
        else "interval_finite_ratios_exceed_bound",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "target_p": target_p,
        "order": order,
        "ratio_start": ratio_start,
        "ratio_bound": ratio_bound,
        "b_subdivisions": b_subdivisions,
        "working_dps": working_dps,
        "time_padding": time_padding,
        "sample_time_hull": [min(state[0] for state in sample_states), max(state[0] for state in sample_states)],
        "checked_time_hull": [time_low, time_high],
        "component_max_ratio_upper": component_max_ratios,
        "component_worst_ratio": component_worst,
        "max_ratio_upper": max_ratio,
        "skipped_zero_denominator_count": skipped_zero_denominator,
        "subintervals": subintervals,
        "remaining_obligation": (
            "prove the p-slice time hull and extend the finite interval ratio "
            "check to a genuine infinite same-parity tail majorant"
        ),
    }


def _rk4_step(
    rhs,
    t: float,
    x: tuple[float, float, float, float],
    step_size: float,
    *args,
) -> tuple[float, float, float, float]:
    """Advance one four-variable ODE by a fixed RK4 step."""
    k1 = rhs(t, x, *args)
    x2 = tuple(value + 0.5 * step_size * slope for value, slope in zip(x, k1))
    k2 = rhs(t + 0.5 * step_size, x2, *args)
    x3 = tuple(value + 0.5 * step_size * slope for value, slope in zip(x, k2))
    k3 = rhs(t + 0.5 * step_size, x3, *args)
    x4 = tuple(value + step_size * slope for value, slope in zip(x, k3))
    k4 = rhs(t + step_size, x4, *args)
    return tuple(
        value + step_size * (s1 + 2.0 * s2 + 2.0 * s3 + s4) / 6.0
        for value, s1, s2, s3, s4 in zip(x, k1, k2, k3, k4)
    )


def _rk4_step_b(
    t: float,
    x: tuple[float, float, float, float],
    step_size: float,
    b: float,
) -> tuple[float, float, float, float]:
    """Advance the scaled b-family by one fixed RK4 step."""
    if b == 0.0:
        return _rk4_step(limiting_scaled_rhs, t, x, step_size)
    return _rk4_step(perturbation_rhs_from_coefficients, t, x, step_size, 1.0 / b)


def p_time_rhs(
    p: float,
    y: tuple[float, float, float, float],
    b: float,
) -> tuple[float, float, float, float]:
    """Return d(t,x1,x2,x3)/dp for the scaled b-family.

    The late terminal segment is singular in t-time because p'=x0' becomes
    large and negative.  Using p=x0 as the independent variable divides the
    other equations by p' and gives a much better conditioned tail diagnostic.
    """
    t, x1, x2, x3 = y
    rhs = scaled_rhs_with_b(t, (p, x1, x2, x3), b)
    if rhs[0] == 0.0:
        raise ZeroDivisionError("p-time RHS requires nonzero x0 derivative")
    return (1.0 / rhs[0], rhs[1] / rhs[0], rhs[2] / rhs[0], rhs[3] / rhs[0])


def cancellation_c_value(p: float, y: tuple[float, float, float, float]) -> float:
    """Return ``C=x1*x2-p^2*x3/6`` for a p-slice state."""
    _t, x1, x2, x3 = y
    return x1 * x2 - p * p * x3 / 6.0


def p_time_rhs_carried_c(
    p: float,
    z: tuple[float, float, float, float, float],
    b: float,
) -> tuple[float, float, float, float, float]:
    """Return the p-time RHS in augmented variables ``(t,x1,x2,x3,C)``."""
    t, x1, x2, x3, c_value = z
    rhs = scaled_rhs_with_b(t, (p, x1, x2, x3), b)
    r1, r2, r3 = finite_a_error_coefficients(t, (p, x1, x2, x3))
    p_prime = (
        (-p - 3.0 * x2 * x3 * x3 / p**4) / t
        - t * c_value / (4.0 * p * p)
        + b * r1[0]
        + b * b * r2[0]
        + b**3 * r3[0]
    )
    if p_prime == 0.0:
        raise ZeroDivisionError("carried-C p-time RHS requires nonzero x0 derivative")
    t_p = 1.0 / p_prime
    x1_p = rhs[1] / p_prime
    x2_p = rhs[2] / p_prime
    x3_p = rhs[3] / p_prime
    c_p = x2 * x1_p + x1 * x2_p - p * x3 / 3.0 - p * p * x3_p / 6.0
    return (t_p, x1_p, x2_p, x3_p, c_p)


def _rk4_step_p(
    p: float,
    y: tuple[float, float, float, float],
    step_size: float,
    b: float,
) -> tuple[float, float, float, float]:
    """Advance the p-time system by one fixed RK4 step."""
    k1 = p_time_rhs(p, y, b)
    y2 = tuple(value + 0.5 * step_size * slope for value, slope in zip(y, k1))
    k2 = p_time_rhs(p + 0.5 * step_size, y2, b)
    y3 = tuple(value + 0.5 * step_size * slope for value, slope in zip(y, k2))
    k3 = p_time_rhs(p + 0.5 * step_size, y3, b)
    y4 = tuple(value + step_size * slope for value, slope in zip(y, k3))
    k4 = p_time_rhs(p + step_size, y4, b)
    return tuple(
        value + step_size * (s1 + 2.0 * s2 + 2.0 * s3 + s4) / 6.0
        for value, s1, s2, s3, s4 in zip(y, k1, k2, k3, k4)
    )


def _rk4_step_carried_c_p(
    p: float,
    z: tuple[float, float, float, float, float],
    step_size: float,
    b: float,
) -> tuple[float, float, float, float, float]:
    """Advance the augmented p-time system by one fixed RK4 step."""
    k1 = p_time_rhs_carried_c(p, z, b)
    z2 = tuple(value + 0.5 * step_size * slope for value, slope in zip(z, k1))
    k2 = p_time_rhs_carried_c(p + 0.5 * step_size, z2, b)
    z3 = tuple(value + 0.5 * step_size * slope for value, slope in zip(z, k2))
    k3 = p_time_rhs_carried_c(p + 0.5 * step_size, z3, b)
    z4 = tuple(value + step_size * slope for value, slope in zip(z, k3))
    k4 = p_time_rhs_carried_c(p + step_size, z4, b)
    return tuple(
        value + step_size * (s1 + 2.0 * s2 + 2.0 * s3 + s4) / 6.0
        for value, s1, s2, s3, s4 in zip(z, k1, k2, k3, k4)
    )


def first_scaled_crossing(
    source: str,
    a: float | None = None,
    step_size: float = DEFAULT_STEP,
    epsilon: float = DEFAULT_EPSILON,
    max_time: float = 8.0,
) -> ScaledCrossing:
    """Return the first x0=0 crossing for the exact or limiting scaled ODE."""
    if source not in {"exact", "limit"}:
        raise ValueError("source must be exact or limit")
    if source == "exact" and (a is None or a == 0.0):
        raise ValueError("exact scaled crossing requires nonzero a")
    x = (1.0, 27.0 / 4.0, -1.0 / 27.0, 3.0)
    t = epsilon
    initial_sign = math.copysign(1.0, x[0])
    while t < max_time:
        try:
            if source == "limit":
                x_next = _rk4_step(limiting_scaled_rhs, t, x, step_size)
            else:
                assert a is not None
                x_next = _rk4_step(perturbation_rhs_from_coefficients, t, x, step_size, a)
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
            return ScaledCrossing(source, a, t, x, step_size, "failed")
        if not all(math.isfinite(value) for value in x_next):
            return ScaledCrossing(source, a, t, x, step_size, "failed")
        if math.copysign(1.0, x_next[0]) != initial_sign:
            alpha = abs(x[0]) / (abs(x[0]) + abs(x_next[0]))
            crossing_time = t + alpha * step_size
            crossing_x = tuple(value + alpha * (next_value - value) for value, next_value in zip(x, x_next))
            return ScaledCrossing(source, a, crossing_time, crossing_x, step_size, "crossed")
        t += step_size
        x = x_next
    return ScaledCrossing(source, a, t, x, step_size, "no_crossing")


def scaled_state_at(
    source: str,
    target_time: float,
    a: float | None = None,
    step_size: float = DEFAULT_STEP,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[float, float, float, float]:
    """Return the scaled IVP state at one regular time before crossing."""
    if source not in {"exact", "limit"}:
        raise ValueError("source must be exact or limit")
    if source == "exact" and (a is None or a == 0.0):
        raise ValueError("exact scaled state requires nonzero a")
    if target_time < epsilon:
        raise ValueError("target_time must be at least epsilon")
    t = epsilon
    x = (1.0, 27.0 / 4.0, -1.0 / 27.0, 3.0)
    while t < target_time:
        step = min(step_size, target_time - t)
        if source == "limit":
            x = _rk4_step(limiting_scaled_rhs, t, x, step)
        else:
            assert a is not None
            x = _rk4_step(perturbation_rhs_from_coefficients, t, x, step, a)
        t += step
    return x


def scaled_state_at_p(
    source: str,
    target_p: float,
    a: float | None = None,
    entry_time: float = DEFAULT_P_TUBE_ENTRY_TIME,
    step_size: float = 1e-5,
) -> tuple[float, float, float, float]:
    """Return (t,x1,x2,x3) at the first post-entry slice x0=target_p."""
    if source not in {"exact", "limit"}:
        raise ValueError("source must be exact or limit")
    if source == "exact" and (a is None or a == 0.0):
        raise ValueError("exact scaled state requires nonzero a")
    x = scaled_state_at(source, entry_time, a, step_size=step_size)
    if x[0] < target_p:
        raise ValueError("entry state is already past target_p")
    t = entry_time
    while t < 8.0:
        if source == "limit":
            x_next = _rk4_step_b(t, x, step_size, 0.0)
        else:
            assert a is not None
            x_next = _rk4_step_b(t, x, step_size, 1.0 / a)
        next_t = t + step_size
        if x_next[0] <= target_p:
            alpha = (x[0] - target_p) / (x[0] - x_next[0])
            interpolated = tuple(value + alpha * (next_value - value) for value, next_value in zip(x, x_next))
            return (t + alpha * step_size, interpolated[1], interpolated[2], interpolated[3])
        t = next_t
        x = x_next
    raise RuntimeError("target_p was not reached before max_time")


def limiting_state_at(
    target_time: float,
    step_size: float = DEFAULT_STEP,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[float, float, float, float]:
    """Return the limiting IVP state at one regular time before crossing."""
    return scaled_state_at("limit", target_time, None, step_size, epsilon)


def x2_boundary_derivative(t: float, x0: float) -> float:
    """Return x2' on the limiting boundary x2=0."""
    return t**3 * x0**3 / 216.0


def x2_zero_boundary_factor_certificate(
    p_range: tuple[float, float] = DEFAULT_X2_ZERO_FACTOR_P_RANGE,
    time_range: tuple[float, float] = DEFAULT_X2_ZERO_FACTOR_TIME_RANGE,
    x3_range: tuple[float, float] = DEFAULT_X2_ZERO_FACTOR_X3_RANGE,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
) -> dict:
    """Certify the exact finite-A sign of ``x2'`` on the wall ``x2=0``.

    On ``x2=0`` the exact scaled equation factors as

        x2' = t^3 p^3 / 216 * (1 + 6 b x3 / p^2)^3,

    where ``p=x0`` and ``b=1/a``.  Thus the lower wall is inward whenever the
    factor is nonnegative over the requested box.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    p_low, p_high = p_range
    t_low, t_high = time_range
    x3_low, x3_high = x3_range
    if not (0.0 < p_low <= p_high):
        raise ValueError("p_range must be positive and ordered")
    if not (0.0 < t_low <= t_high):
        raise ValueError("time_range must be positive and ordered")
    if x3_low > x3_high:
        raise ValueError("x3_range must be ordered")

    beta = 1.0 / candidate_a
    endpoint_products = [
        b * s
        for b in (-beta, beta)
        for s in (x3_low, x3_high)
    ]
    product_min = min(endpoint_products)
    product_max = max(endpoint_products)
    factor_lower = 1.0 + 6.0 * product_min / (p_low * p_low)
    factor_upper = 1.0 + 6.0 * product_max / (p_low * p_low)
    derivative_lower = (
        t_low**3
        * p_low**3
        / 216.0
        * factor_lower**3
        if factor_lower >= 0.0
        else t_low**3 * p_low**3 / 216.0 * factor_lower**3
    )
    status = "certified" if factor_lower >= 0.0 else "failed"
    return {
        "status": status,
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "p_range": list(p_range),
        "time_range": list(time_range),
        "x3_range": list(x3_range),
        "factor_lower_bound": factor_lower,
        "factor_upper_bound": factor_upper,
        "min_b_x3_product": product_min,
        "max_b_x3_product": product_max,
        "x2_prime_lower_bound_on_wall": derivative_lower,
        "inward_margin": derivative_lower,
        "condition": "1 + 6*b*x3/p^2 >= 0 on x2=0",
        "conclusion": "x2=0 is an inward lower wall in ordinary time over this box"
        if status == "certified"
        else "the exact factor changes sign somewhere in the requested box",
    }


def x3_boundary_derivative(t: float, x0: float, x1: float, x2: float, boundary: float = -0.3) -> float:
    """Return x3' on a proposed x3=boundary barrier."""
    x3 = boundary
    return (
        (-2.0 * x3 + 6.0 * x0) / t
        + t
        / (2.0 * x0**3)
        * (x1 * x2 * x3 - x3 * x3 * x0 * x0 / 6.0 - t * t * x1 * x0**4 / 18.0)
    )


def x3_zero_boundary_derivative(t: float, x0: float, x1: float) -> float:
    """Return x3' on x3=0 for either the limiting or finite scaled equation.

    All finite-a perturbation terms vanish on this boundary.  The sign therefore
    gives an exact terminal-barrier diagnostic for the scaled family.
    """
    return x0 * (6.0 / t - t**3 * x1 / 36.0)


def x3_zero_threshold(t: float) -> float:
    """Return the x1 threshold above which x3=0 is an inward barrier."""
    return 216.0 / t**4


def _c_wall_negative_polynomial_bound(k_value: float) -> float:
    """Return ``M`` with ``(K-u/6)u^2(9K-2u) >= -M`` on ``0<=u<=6K``."""
    if k_value <= 0.0:
        raise ValueError("k_value must be positive")
    root = (63.0 + math.sqrt(513.0)) / 16.0
    normalized = 9.0 * root * root - 3.5 * root**3 + root**4 / 3.0
    return -normalized * k_value**4


def _late_c_wall_hdot(
    t: float,
    p: float,
    x1: float,
    u: float,
    b: float,
    k_value: float,
) -> float:
    """Return ``d(C-Kp^3)/dt`` on ``C=Kp^3`` and ``x3=-u*p``."""
    x3 = -u * p
    x2 = p**3 * (k_value - u / 6.0) / x1
    rhs = scaled_rhs_with_b(t, (p, x1, x2, x3), b)
    c_dot = rhs[1] * x2 + x1 * rhs[2] - (2.0 * p * rhs[0] * x3 + p * p * rhs[3]) / 6.0
    return c_dot - 3.0 * k_value * p * p * rhs[0]


def late_scalar_barrier_report(
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    sigma: float = DEFAULT_LATE_SCALAR_BARRIER_SIGMA,
    k_value: float = DEFAULT_LATE_SCALAR_BARRIER_K,
    p_max: float = DEFAULT_LATE_SCALAR_BARRIER_P_MAX,
    time_range: tuple[float, float] = DEFAULT_LATE_SCALAR_BARRIER_TIME_RANGE,
    x1_range: tuple[float, float] = DEFAULT_LATE_SCALAR_BARRIER_X1_RANGE,
    grid_subdivisions: int = 20,
) -> dict:
    """Return scalar late-region barrier margins for the proof reduction.

    This is a small proof-audit helper, not the complete large-tail theorem.
    It checks the hand-derived ``x3=-sigma`` wall and the limiting lower bound
    for the ``C-Kp^3`` wall.  A deterministic finite-``b`` grid sanity check is
    also reported separately.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if sigma <= 0.0 or k_value <= 0.0:
        raise ValueError("sigma and k_value must be positive")
    if p_max <= 0.0:
        raise ValueError("p_max must be positive")
    t_low, t_high = time_range
    x1_low, x1_high = x1_range
    if not (0.0 < t_low <= t_high):
        raise ValueError("time_range must be positive and ordered")
    if not (0.0 < x1_low <= x1_high):
        raise ValueError("x1_range must be positive and ordered")
    if grid_subdivisions <= 0:
        raise ValueError("grid_subdivisions must be positive")

    beta = 1.0 / candidate_a
    p_auto = sigma / (6.0 * k_value)
    if p_auto >= p_max:
        raise ValueError("sigma/(6K) must be below p_max")

    x3_regular_upper = (2.0 * sigma + 6.0 * p_max) / t_low
    x3_negative_term = t_low * sigma * k_value / 2.0
    x3_finite_error = (
        beta
        * t_high
        / (2.0 * p_max**3)
        * (sigma**3 + 2.0 * t_high**2 * x1_high * p_max * p_max * sigma / 3.0)
    )
    x3_upper_bound = x3_regular_upper - x3_negative_term + x3_finite_error

    c_negative_bound = _c_wall_negative_polynomial_bound(k_value)
    limiting_bracket_lower = (
        (-k_value - 1.0) / t_low
        + x1_low * t_low**3 / 108.0
        + 3.0 * k_value * k_value * t_low / 4.0
        - c_negative_bound / (x1_low * t_low)
    )
    limiting_hdot_lower = p_auto**3 * limiting_bracket_lower

    grid_margin = math.inf
    grid_witness: dict | None = None
    for b in (-beta, 0.0, beta):
        for t_index in range(grid_subdivisions + 1):
            t = t_low + (t_high - t_low) * t_index / grid_subdivisions
            for p_index in range(grid_subdivisions + 1):
                p = p_auto + (p_max - p_auto) * p_index / grid_subdivisions
                u_low = sigma / p
                u_high = 6.0 * k_value
                for x1 in (x1_low, x1_high):
                    for u_index in range(grid_subdivisions + 1):
                        u = u_low + (u_high - u_low) * u_index / grid_subdivisions
                        hdot = _late_c_wall_hdot(t, p, x1, u, b, k_value)
                        if hdot < grid_margin:
                            grid_margin = hdot
                            grid_witness = {"b": b, "t": t, "p": p, "x1": x1, "u": u}

    return {
        "status": "scalar_margins_positive"
        if x3_upper_bound < 0.0 and limiting_hdot_lower > 0.0
        else "failed",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "sigma": sigma,
        "K": k_value,
        "p_max": p_max,
        "p_auto": p_auto,
        "time_range": list(time_range),
        "x1_range": list(x1_range),
        "x3_wall_upper_bound": x3_upper_bound,
        "x3_wall_margin": -x3_upper_bound,
        "c_wall_negative_polynomial_bound": c_negative_bound,
        "c_wall_limiting_bracket_lower": limiting_bracket_lower,
        "c_wall_limiting_hdot_lower": limiting_hdot_lower,
        "finite_b_grid_hdot_margin": grid_margin,
        "finite_b_grid_witness": grid_witness,
        "grid_subdivisions": grid_subdivisions,
        "remaining_obligation": (
            "prove the trajectory stays in the correlated late region; "
            "these scalar margins only certify the proposed walls"
        ),
    }


def x3_zero_wall_certificate(
    time_range: tuple[float, float] = DEFAULT_X3_ZERO_WALL_TIME_RANGE,
    x0_range: tuple[float, float] = DEFAULT_X3_ZERO_WALL_X0_RANGE,
    x1_range: tuple[float, float] = DEFAULT_X3_ZERO_WALL_X1_RANGE,
    x2_range: tuple[float, float] = DEFAULT_X3_ZERO_WALL_X2_RANGE,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    subdivisions: tuple[int, int, int, int] = DEFAULT_X3_ZERO_WALL_SUBDIVISIONS,
    time_subdivisions: int = DEFAULT_X3_ZERO_WALL_TIME_SUBDIVISIONS,
) -> dict:
    """Certify that the wall ``x3=0`` points into the negative side.

    On this wall all finite-``a`` correction terms vanish, so the analytic
    condition is simply ``x1 > 216/t^4`` while ``x0 > 0``.  The interval check
    below independently evaluates the full scaled finite-``a`` vector field on
    the same box for ``b=1/a`` in ``[-1/A,1/A]``.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    t_low, t_high = time_range
    if not (0.0 < t_low <= t_high):
        raise ValueError("time_range must be positive and ordered")
    if not (0.0 < x0_range[0] <= x0_range[1]):
        raise ValueError("x0_range must be positive and ordered")
    if not (x1_range[0] <= x1_range[1] and x2_range[0] <= x2_range[1]):
        raise ValueError("x1_range and x2_range must be ordered")
    if any(value <= 0 for value in subdivisions):
        raise ValueError("subdivisions must be positive")
    if time_subdivisions <= 0:
        raise ValueError("time_subdivisions must be positive")

    from mpmath import iv

    beta = 1.0 / candidate_a
    t_interval = iv.mpf([t_low, t_high])
    box = (
        iv.mpf([x0_range[0], x0_range[1]]),
        iv.mpf([x1_range[0], x1_range[1]]),
        iv.mpf([x2_range[0], x2_range[1]]),
        iv.mpf([0.0, 0.0]),
    )
    rhs_lower, rhs_upper = _subdivided_interval_rhs_component(
        t_interval,
        box,
        iv.mpf([-beta, beta]),
        3,
        subdivisions,
        time_subdivisions,
    )
    threshold_margin = x1_range[0] - x3_zero_threshold(t_low)
    interval_inward_margin = -rhs_upper
    analytic_upper = x3_zero_boundary_derivative(t_low, x0_range[1], x1_range[0])
    analytic_inward_margin = -analytic_upper
    status = "certified" if threshold_margin > 0.0 and analytic_upper < 0.0 else "failed"
    return {
        "status": status,
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "time_range": list(time_range),
        "x0_range": list(x0_range),
        "x1_range": list(x1_range),
        "x2_range": list(x2_range),
        "x3_wall": 0.0,
        "threshold_at_t_low": x3_zero_threshold(t_low),
        "threshold_margin": threshold_margin,
        "analytic_rhs_x3_upper": analytic_upper,
        "analytic_inward_margin": analytic_inward_margin,
        "rhs_x3_lower": rhs_lower,
        "rhs_x3_upper": rhs_upper,
        "interval_inward_margin": interval_inward_margin,
        "interval_status": "certified" if rhs_upper < 0.0 else "overconservative_or_failed",
        "inward_margin": analytic_inward_margin,
        "subdivisions": list(subdivisions),
        "time_subdivisions": time_subdivisions,
        "conclusion": "x3=0 is a one-way wall into x3<0 while x0>0 and x1 stays above the threshold",
    }


def riccati_integral_to_crossing(
    support_time: float = DEFAULT_SUPPORT_TIME,
    step_size: float = 1e-5,
) -> dict:
    """Estimate the Riccati integral controlling positive x2 loss."""
    t = DEFAULT_EPSILON
    x = (1.0, 27.0 / 4.0, -1.0 / 27.0, 3.0)
    support_state: tuple[float, float, float, float] | None = None
    integral = 0.0
    previous_t: float | None = None
    previous_c: float | None = None
    while t < 8.0:
        x_next = _rk4_step(limiting_scaled_rhs, t, x, step_size)
        next_t = t + step_size
        if support_state is None and t <= support_time <= next_t:
            alpha = (support_time - t) / step_size
            support_state = tuple(value + alpha * (next_value - value) for value, next_value in zip(x, x_next))
            previous_t = support_time
            previous_c = support_time * support_state[1] / (2.0 * support_state[0] ** 3)
        if support_state is not None:
            c_value = t * x[1] / (2.0 * x[0] ** 3)
            if previous_t is not None and previous_c is not None and t >= support_time:
                integral += 0.5 * (previous_c + c_value) * (t - previous_t)
            previous_t = t
            previous_c = c_value
        if math.copysign(1.0, x_next[0]) != math.copysign(1.0, x[0]):
            alpha = abs(x[0]) / (abs(x[0]) + abs(x_next[0]))
            crossing_time = t + alpha * step_size
            crossing_x = tuple(value + alpha * (next_value - value) for value, next_value in zip(x, x_next))
            if support_state is None:
                raise RuntimeError("support time was not reached before crossing")
            lower_bound = 1.0 / (1.0 / support_state[2] + integral)
            return {
                "support_time": support_time,
                "support_state": list(support_state),
                "crossing_time": crossing_time,
                "crossing_state": list(crossing_x),
                "riccati_integral_numeric": integral,
                "riccati_lower_bound_numeric": lower_bound,
                "step_size": step_size,
            }
        t = next_t
        x = x_next
    raise RuntimeError("limiting IVP did not cross x0=0 before max_time")


def terminal_barrier_report(
    support_time: float = DEFAULT_TERMINAL_BARRIER_TIME,
    candidate_a: float = DEFAULT_CANDIDATE_A,
    step_size: float = DEFAULT_STEP,
) -> dict:
    """Return the late-time x3=0 terminal-barrier diagnostics.

    Right K- closure requires f3=f4=0.  At an f0=0 endpoint this is equivalent
    to x3=0 in the scaled variables.  On x3=0 the exact scaled equation gives

        x3' = x0 (6/t - t^3 x1/36),

    so x1 > 216/t^4 makes x3=0 an inward barrier while x0>0.
    """
    threshold = x3_zero_threshold(support_time)

    def payload(source: str, a: float | None, state: tuple[float, float, float, float]) -> dict:
        x0, x1, x2, x3 = state
        return {
            "source": source,
            "a": a,
            "time": support_time,
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x1_threshold_for_x3_zero_barrier": threshold,
            "x1_margin": x1 - threshold,
            "x3_zero_boundary_derivative": x3_zero_boundary_derivative(support_time, x0, x1),
        }

    finite_states = [
        scaled_state_at("exact", support_time, -candidate_a, step_size=step_size),
        scaled_state_at("exact", support_time, candidate_a, step_size=step_size),
    ]
    return {
        "support_time": support_time,
        "candidate_A": candidate_a,
        "step_size": step_size,
        "criterion": "x3=0 is inward when x0>0 and x1>216/t^4; K- closure requires x3=0",
        "limit": payload("limit", None, scaled_state_at("limit", support_time, step_size=step_size)),
        "finite_candidate_A": [
            payload("exact", -candidate_a, finite_states[0]),
            payload("exact", candidate_a, finite_states[1]),
        ],
    }


def _interval_lower(value) -> float:
    """Return the lower endpoint of an mpmath interval scalar."""
    return float(value.a)


def _interval_upper(value) -> float:
    """Return the upper endpoint of an mpmath interval scalar."""
    return float(value.b)


def _interval_intersection(current, candidate):
    """Return the nonempty interval intersection, or ``current`` if disjoint."""
    lower = max(_interval_lower(current), _interval_lower(candidate))
    upper = min(_interval_upper(current), _interval_upper(candidate))
    if lower <= upper:
        from mpmath import iv

        return iv.mpf([lower, upper])
    return current


def _interval_scaled_rhs(t_interval, x_interval: tuple, b_interval) -> tuple:
    """Return an interval enclosure of the scaled RHS over one box."""
    limit = limiting_scaled_rhs(t_interval, x_interval)
    r1, r2, r3 = finite_a_error_coefficients(t_interval, x_interval)
    return tuple(
        limit[index]
        + b_interval * r1[index]
        + b_interval * b_interval * r2[index]
        + b_interval**3 * r3[index]
        for index in range(4)
    )


def _interval_abs_upper(value) -> float:
    """Return an upper bound for the absolute value of an interval scalar."""
    return max(abs(_interval_lower(value)), abs(_interval_upper(value)))


def _interval_p_prime_cancellation_upper(
    p_interval,
    y_interval: tuple,
    b_interval,
) -> float | None:
    """Return a rigorous upper bound for p' using the C cancellation term.

    The limiting p' equation contains

        -t * (x1*x2 - p^2*x3/6) / (4 p^2).

    On boxes where ``C=x1*x2-p^2*x3/6`` has a positive interval lower bound,
    this gives a much sharper negative upper bound for p' than direct interval
    evaluation.  Finite-|a| terms are added back with absolute-value bounds, so
    the result is still an enclosure for every b in the requested interval.
    """
    t_interval, x1_interval, x2_interval, x3_interval = y_interval
    if _interval_lower(p_interval) <= 0.0 or _interval_lower(t_interval) <= 0.0:
        return None
    # The remaining limiting denominator term is
    # -3*x2*x3^2/(t*p^4).  We may drop it from an upper bound only on boxes
    # where x2 is nonnegative.
    if _interval_lower(x2_interval) < 0.0:
        return None

    c_interval = x1_interval * x2_interval - p_interval * p_interval * x3_interval / 6.0
    c_lower = _interval_lower(c_interval)
    if c_lower <= 0.0:
        return None

    p_low = _interval_lower(p_interval)
    p_high = _interval_upper(p_interval)
    t_low = _interval_lower(t_interval)
    t_high = _interval_upper(t_interval)
    limit_upper = -p_low / t_high - t_low * c_lower / (4.0 * p_high * p_high)

    r1, r2, r3 = finite_a_error_coefficients(
        t_interval,
        (p_interval, x1_interval, x2_interval, x3_interval),
    )
    beta = max(abs(_interval_lower(b_interval)), abs(_interval_upper(b_interval)))
    correction_upper = (
        beta * _interval_abs_upper(r1[0])
        + beta * beta * _interval_abs_upper(r2[0])
        + beta**3 * _interval_abs_upper(r3[0])
    )
    return limit_upper + correction_upper


def _interval_p_time_rhs(
    p_interval,
    y_interval: tuple,
    b_interval,
    use_cancellation_p_prime: bool = False,
) -> tuple:
    """Return an interval enclosure of the p-time RHS over one box."""
    t_interval, x1_interval, x2_interval, x3_interval = y_interval
    rhs = _interval_scaled_rhs(
        t_interval,
        (p_interval, x1_interval, x2_interval, x3_interval),
        b_interval,
    )
    if use_cancellation_p_prime:
        refined_upper = _interval_p_prime_cancellation_upper(
            p_interval,
            y_interval,
            b_interval,
        )
        if refined_upper is not None and refined_upper < _interval_upper(rhs[0]):
            from mpmath import iv

            rhs = (
                iv.mpf([_interval_lower(rhs[0]), refined_upper]),
                rhs[1],
                rhs[2],
                rhs[3],
            )
    if _interval_lower(rhs[0]) <= 0.0 <= _interval_upper(rhs[0]):
        raise ZeroDivisionError("p-time interval contains x0'=0")
    return (1.0 / rhs[0], rhs[1] / rhs[0], rhs[2] / rhs[0], rhs[3] / rhs[0])


def _interval_square(value):
    """Return a sharp interval enclosure for ``value**2``."""
    from mpmath import iv

    lower = _interval_lower(value)
    upper = _interval_upper(value)
    if lower <= 0.0 <= upper:
        return iv.mpf([0.0, max(lower * lower, upper * upper)])
    return iv.mpf([min(lower * lower, upper * upper), max(lower * lower, upper * upper)])


def _sharpen_carried_c_graph_intervals(
    p_interval,
    z_interval: tuple,
    rounds: int = 3,
) -> tuple:
    """Narrow ``(t,x1,x2,x3,C)`` by the graph identity defining ``C``.

    The true carried-C trajectory lies on

        C = x1*x2 - p^2*x3/6.

    Intersecting component intervals with interval consequences of this
    identity is rigorous and removes fake combinations introduced by
    axis-aligned boxes.
    """
    t_interval, x1_interval, x2_interval, x3_interval, c_interval = z_interval
    p2 = p_interval * p_interval
    for _ in range(rounds):
        c_interval = _interval_intersection(
            c_interval,
            x1_interval * x2_interval - p2 * x3_interval / 6.0,
        )
        x3_interval = _interval_intersection(
            x3_interval,
            6.0 * (x1_interval * x2_interval - c_interval) / p2,
        )
        if _interval_lower(x1_interval) > 0.0 or _interval_upper(x1_interval) < 0.0:
            x2_interval = _interval_intersection(
                x2_interval,
                (c_interval + p2 * x3_interval / 6.0) / x1_interval,
            )
        if _interval_lower(x2_interval) > 0.0 or _interval_upper(x2_interval) < 0.0:
            x1_interval = _interval_intersection(
                x1_interval,
                (c_interval + p2 * x3_interval / 6.0) / x2_interval,
            )
    return (t_interval, x1_interval, x2_interval, x3_interval, c_interval)


def _interval_scaled_p_prime_with_carried_c(
    p_interval,
    z_interval: tuple,
    b_interval,
):
    """Return an interval enclosure for p'=x0' using carried C."""
    t_interval, x1_interval, x2_interval, x3_interval, c_interval = _sharpen_carried_c_graph_intervals(
        p_interval,
        z_interval,
    )
    p2 = p_interval * p_interval
    p4 = p2 * p2
    r1, r2, r3 = finite_a_error_coefficients(
        t_interval,
        (p_interval, x1_interval, x2_interval, x3_interval),
    )
    carried_prime = (
        (-p_interval - 3.0 * x2_interval * _interval_square(x3_interval) / p4) / t_interval
        - t_interval * c_interval / (4.0 * p2)
        + b_interval * r1[0]
        + b_interval * b_interval * r2[0]
        + b_interval**3 * r3[0]
    )
    expanded_prime = _interval_scaled_rhs(
        t_interval,
        (p_interval, x1_interval, x2_interval, x3_interval),
        b_interval,
    )[0]
    return _interval_intersection(carried_prime, expanded_prime)


def _interval_carried_c_p_time_rhs(
    p_interval,
    z_interval: tuple,
    b_interval,
) -> tuple:
    """Return an interval enclosure of augmented p-time RHS over one box."""
    t_interval, x1_interval, x2_interval, x3_interval, _c_interval = z_interval
    if _interval_lower(p_interval) <= 0.0 or _interval_lower(t_interval) <= 0.0:
        raise ZeroDivisionError("carried-C p-time interval needs positive p and t")
    rhs = _interval_scaled_rhs(
        t_interval,
        (p_interval, x1_interval, x2_interval, x3_interval),
        b_interval,
    )
    p_prime = _interval_scaled_p_prime_with_carried_c(p_interval, z_interval, b_interval)
    if _interval_lower(p_prime) <= 0.0 <= _interval_upper(p_prime):
        raise ZeroDivisionError("carried-C p-time interval contains x0'=0")
    t_p = 1.0 / p_prime
    x1_p = rhs[1] / p_prime
    x2_p = rhs[2] / p_prime
    x3_p = rhs[3] / p_prime
    c_p = x2_interval * x1_p + x1_interval * x2_p - p_interval * x3_interval / 3.0 - (
        p_interval * p_interval * x3_p / 6.0
    )
    return (t_p, x1_p, x2_p, x3_p, c_p)


def _split_interval(interval, parts: int) -> list:
    """Split an mpmath interval into equal subintervals."""
    if parts <= 1:
        return [interval]
    from mpmath import iv

    lower = _interval_lower(interval)
    upper = _interval_upper(interval)
    width = (upper - lower) / parts
    return [
        iv.mpf([lower + index * width, lower + (index + 1) * width])
        for index in range(parts)
    ]


def _subdivided_interval_rhs_component(
    t_interval,
    x_interval: tuple,
    b_interval,
    component: int,
    subdivisions: tuple[int, int, int, int] = (1, 1, 1, 1),
    time_subdivisions: int = 1,
) -> tuple[float, float]:
    """Return rigorous component bounds from subdivided interval evaluation."""
    lower = math.inf
    upper = -math.inf
    t_pieces = _split_interval(t_interval, time_subdivisions)
    pieces = [_split_interval(x_interval[index], subdivisions[index]) for index in range(4)]
    for t_piece in t_pieces:
        for x_piece in itertools.product(*pieces):
            value = _interval_scaled_rhs(t_piece, tuple(x_piece), b_interval)[component]
            lower = min(lower, _interval_lower(value))
            upper = max(upper, _interval_upper(value))
    return lower, upper


def _subdivided_interval_p_time_rhs_component(
    p_interval,
    y_interval: tuple,
    b_interval,
    component: int,
    subdivisions: tuple[int, int, int, int] = (1, 1, 1, 1),
    p_subdivisions: int = 1,
    use_cancellation_p_prime: bool = False,
) -> tuple[float, float]:
    """Return rigorous p-time component bounds from subdivided intervals."""
    lower = math.inf
    upper = -math.inf
    p_pieces = _split_interval(p_interval, p_subdivisions)
    pieces = [_split_interval(y_interval[index], subdivisions[index]) for index in range(4)]
    for p_piece in p_pieces:
        for y_piece in itertools.product(*pieces):
            value = _interval_p_time_rhs(
                p_piece,
                tuple(y_piece),
                b_interval,
                use_cancellation_p_prime=use_cancellation_p_prime,
            )[component]
            lower = min(lower, _interval_lower(value))
            upper = max(upper, _interval_upper(value))
    return lower, upper


def _subdivided_interval_carried_c_p_time_rhs_component(
    p_interval,
    z_interval: tuple,
    b_interval,
    component: int,
    subdivisions: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1),
    p_subdivisions: int = 1,
) -> tuple[float, float]:
    """Return rigorous augmented p-time component bounds from intervals."""
    lower = math.inf
    upper = -math.inf
    p_pieces = _split_interval(p_interval, p_subdivisions)
    pieces = [_split_interval(z_interval[index], subdivisions[index]) for index in range(5)]
    for p_piece in p_pieces:
        for z_piece in itertools.product(*pieces):
            value = _interval_carried_c_p_time_rhs(
                p_piece,
                tuple(z_piece),
                b_interval,
            )[component]
            lower = min(lower, _interval_lower(value))
            upper = max(upper, _interval_upper(value))
    return lower, upper


def _normalize_p_tube_profile(profile) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Return lower and upper p-time radius growth vectors."""
    if len(profile) == 2 and all(hasattr(item, "__len__") and len(item) == 4 for item in profile):
        return tuple(float(value) for value in profile[0]), tuple(float(value) for value in profile[1])
    if len(profile) == 4:
        growth = tuple(float(value) for value in profile)
        return growth, growth
    raise ValueError("p-tube profiles must be 4-tuples or pairs of 4-tuples")


def _component_safety_tuple(safety) -> tuple[float, float, float, float]:
    """Return a nonnegative four-component safety tuple."""
    if isinstance(safety, (int, float)):
        value = float(safety)
        if value < 0.0:
            raise ValueError("safety must be nonnegative")
        return (value, value, value, value)
    if len(safety) != 4:
        raise ValueError("component safety must have four entries")
    values = tuple(float(value) for value in safety)
    if any(value < 0.0 for value in values):
        raise ValueError("safety must be nonnegative")
    return values


def _component_safety_tuple_n(safety, expected: int, name: str = "safety") -> tuple[float, ...]:
    """Return a nonnegative component safety tuple of a requested length."""
    if isinstance(safety, (int, float)):
        value = float(safety)
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative")
        return tuple(value for _ in range(expected))
    if len(safety) != expected:
        raise ValueError(f"{name} must have {expected} entries")
    values = tuple(float(value) for value in safety)
    if any(value < 0.0 for value in values):
        raise ValueError(f"{name} must be nonnegative")
    return values


def _affine_barrier_value(
    start_value: tuple[float, float, float, float],
    slope: tuple[float, float, float, float],
    start_p: float,
    p: float,
) -> tuple[float, float, float, float]:
    """Return one affine p-barrier value vector."""
    return tuple(start_value[index] + slope[index] * (p - start_p) for index in range(4))


def _affine_barrier_value_nd(
    start_value: tuple[float, ...],
    slope: tuple[float, ...],
    start_p: float,
    p: float,
) -> tuple[float, ...]:
    """Return one affine p-barrier value vector in arbitrary dimension."""
    return tuple(start_value[index] + slope[index] * (p - start_p) for index in range(len(start_value)))


def _corridor_contains_box(
    lower_start: tuple[float, float, float, float],
    upper_start: tuple[float, float, float, float],
    box_low: tuple[float, float, float, float],
    box_high: tuple[float, float, float, float],
) -> bool:
    """Return whether one start box is contained in the start corridor."""
    return all(lower_start[index] <= box_low[index] and box_high[index] <= upper_start[index] for index in range(4))


def _corridor_contains_box_nd(
    lower_start: tuple[float, ...],
    upper_start: tuple[float, ...],
    box_low: tuple[float, ...],
    box_high: tuple[float, ...],
) -> bool:
    """Return whether one start box is contained in the start corridor."""
    return all(
        lower_start[index] <= box_low[index] and box_high[index] <= upper_start[index]
        for index in range(len(lower_start))
    )


def _affine_time_barrier_value(
    start_value: tuple[float, float, float, float],
    slope: tuple[float, float, float, float],
    start_time: float,
    time: float,
) -> tuple[float, float, float, float]:
    """Return one affine t-barrier value vector."""
    return tuple(start_value[index] + slope[index] * (time - start_time) for index in range(4))


def _sample_scaled_grid(
    start_time: float,
    end_time: float,
    step_size: float,
    b: float,
    seed_step_size: float,
) -> list[tuple[float, float, float, float]]:
    """Return nominal samples for one b value on an evenly spaced grid."""
    steps = round((end_time - start_time) / step_size)
    if abs(start_time + steps * step_size - end_time) > 1e-12:
        raise ValueError("tube interval length must be an integer multiple of step_size")
    source = "limit" if b == 0.0 else "exact"
    a = None if b == 0.0 else 1.0 / b
    x = scaled_state_at(source, start_time, a, step_size=seed_step_size)
    values = [x]
    t = start_time
    for _ in range(steps):
        x = _rk4_step_b(t, x, step_size, b)
        t += step_size
        values.append(x)
    return values


def _nominal_scaled_samples_at_time(
    time: float,
    candidate_a: float,
    seed_step_size: float = 1e-5,
) -> tuple[tuple[float, float, float, float], ...]:
    """Return nominal b=-beta,0,+beta samples at a fixed ordinary time."""
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    beta = 1.0 / candidate_a
    return tuple(
        scaled_state_at(
            "limit" if b == 0.0 else "exact",
            time,
            None if b == 0.0 else 1.0 / b,
            step_size=seed_step_size,
        )
        for b in (-beta, 0.0, beta)
    )


def centered_restart_box(
    time: float,
    source_low: tuple[float, float, float, float],
    source_high: tuple[float, float, float, float],
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    padding: float = 1e-10,
    seed_step_size: float = 1e-5,
) -> dict:
    """Return a fresh centered nominal box containing a carried source box."""
    samples = _nominal_scaled_samples_at_time(time, candidate_a, seed_step_size)
    return centered_restart_box_from_samples(
        time,
        samples,
        source_low,
        source_high,
        candidate_a=candidate_a,
        padding=padding,
        sample_source="recomputed",
    )


def centered_restart_box_from_samples(
    time: float,
    samples: tuple[tuple[float, float, float, float], ...],
    source_low: tuple[float, float, float, float],
    source_high: tuple[float, float, float, float],
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    padding: float = 1e-10,
    sample_source: str = "supplied",
) -> dict:
    """Return a centered restart box around supplied nominal samples.

    The barrier certificate only requires the new box to contain the carried
    source box.  Reusing the nominal samples propagated by the previous
    certified block is therefore enough; recomputing them from the singular
    initial time is a performance choice, not a proof obligation.
    """
    if len(samples) == 0:
        raise ValueError("at least one nominal sample is required")
    nominal_low = tuple(min(sample[index] for sample in samples) for index in range(4))
    nominal_high = tuple(max(sample[index] for sample in samples) for index in range(4))
    radius = tuple(
        max(nominal_low[index] - source_low[index], source_high[index] - nominal_high[index], 0.0)
        + padding
        for index in range(4)
    )
    low = tuple(nominal_low[index] - radius[index] for index in range(4))
    high = tuple(nominal_high[index] + radius[index] for index in range(4))
    contained = _corridor_contains_box(low, high, source_low, source_high)
    return {
        "time": time,
        "candidate_A": candidate_a,
        "padding": padding,
        "sample_source": sample_source,
        "samples": [list(sample) for sample in samples],
        "nominal_low": list(nominal_low),
        "nominal_high": list(nominal_high),
        "source_box": {"low": list(source_low), "high": list(source_high)},
        "box": {"low": list(low), "high": list(high)},
        "radius": list(radius),
        "source_box_contained": contained,
    }


def moving_tube_certificate(
    start_time: float = DEFAULT_SUPPORT_TIME,
    end_time: float = DEFAULT_TERMINAL_BARRIER_TIME,
    step_size: float = 1e-4,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius0: tuple[float, float, float, float] = (1e-7, 1e-6, 1e-8, 1e-7),
    radius_growth: tuple[float, float, float, float] = (0.005, 0.05, 0.0005, 0.005),
    lower_radius0: tuple[float, float, float, float] | None = None,
    upper_radius0: tuple[float, float, float, float] | None = None,
    lower_radius_growth: tuple[float, float, float, float] | None = None,
    upper_radius_growth: tuple[float, float, float, float] | None = None,
    subdivisions: tuple[int, int, int, int] = (1, 1, 1, 1),
    time_subdivisions: int = 1,
    seed_step_size: float = 1e-5,
) -> dict:
    """Conditionally verify a moving interval tube for the b-family.

    This checks inward-pointing face inequalities on a linearly moving
    rectangle.  A successful result certifies the implication "start box
    contains the true state => end box contains the true state" for every
    b=1/a in [-1/candidate_a, 1/candidate_a].  It does not by itself certify
    the start box.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")

    from mpmath import iv

    beta = 1.0 / candidate_a
    steps = round((end_time - start_time) / step_size)
    if abs(start_time + steps * step_size - end_time) > 1e-12:
        raise ValueError("tube interval length must be an integer multiple of step_size")
    b_samples = (-beta, 0.0, beta)
    grids = [
        _sample_scaled_grid(start_time, end_time, step_size, b, seed_step_size)
        for b in b_samples
    ]
    lower_radius0 = radius0 if lower_radius0 is None else lower_radius0
    upper_radius0 = radius0 if upper_radius0 is None else upper_radius0
    lower_radius_growth = radius_growth if lower_radius_growth is None else lower_radius_growth
    upper_radius_growth = radius_growth if upper_radius_growth is None else upper_radius_growth
    lows: list[list[float]] = []
    highs: list[list[float]] = []
    for step_index in range(steps + 1):
        t = start_time + step_index * step_size
        lower_radius = [
            lower_radius0[index] + lower_radius_growth[index] * (t - start_time)
            for index in range(4)
        ]
        upper_radius = [
            upper_radius0[index] + upper_radius_growth[index] * (t - start_time)
            for index in range(4)
        ]
        samples = [grid[step_index] for grid in grids]
        lows.append(
            [
                min(sample[index] for sample in samples) - lower_radius[index]
                for index in range(4)
            ]
        )
        highs.append(
            [
                max(sample[index] for sample in samples) + upper_radius[index]
                for index in range(4)
            ]
        )

    b_interval = iv.mpf([-beta, beta])
    worst_margin = math.inf
    worst_face: dict | None = None
    for step_index in range(steps):
        t0 = start_time + step_index * step_size
        t1 = t0 + step_size
        t_interval = iv.mpf([t0, t1])
        union_box = [
            iv.mpf(
                [
                    min(lows[step_index][index], lows[step_index + 1][index]),
                    max(highs[step_index][index], highs[step_index + 1][index]),
                ]
            )
            for index in range(4)
        ]
        for index in range(4):
            lower_box = list(union_box)
            lower_box[index] = iv.mpf(
                [
                    min(lows[step_index][index], lows[step_index + 1][index]),
                    max(lows[step_index][index], lows[step_index + 1][index]),
                ]
            )
            lower_rhs_low, lower_rhs_high = _subdivided_interval_rhs_component(
                t_interval,
                tuple(lower_box),
                b_interval,
                index,
                subdivisions,
                time_subdivisions,
            )
            lower_slope = (lows[step_index + 1][index] - lows[step_index][index]) / step_size
            margin = lower_rhs_low - lower_slope
            if margin < worst_margin:
                worst_margin = margin
                worst_face = {
                    "side": "lower",
                    "step_index": step_index,
                    "component": index,
                    "rhs_lower": lower_rhs_low,
                    "rhs_upper": lower_rhs_high,
                    "face_slope": lower_slope,
                    "time_interval": [t0, t1],
                }
            if margin < 0.0:
                return {
                    "status": "failed",
                    "candidate_A": candidate_a,
                    "b_interval": [-beta, beta],
                    "start_time": start_time,
                    "end_time": end_time,
                    "step_size": step_size,
                    "steps": steps,
                    "lower_radius0": list(lower_radius0),
                    "upper_radius0": list(upper_radius0),
                    "lower_radius_growth": list(lower_radius_growth),
                    "upper_radius_growth": list(upper_radius_growth),
                    "subdivisions": list(subdivisions),
                    "time_subdivisions": time_subdivisions,
                    "start_box": {"low": lows[0], "high": highs[0]},
                    "end_box": {"low": lows[-1], "high": highs[-1]},
                    "worst_margin": worst_margin,
                    "failing_face": worst_face,
                    "conditional": "start_box_contains_true_state",
                }

            upper_box = list(union_box)
            upper_box[index] = iv.mpf(
                [
                    min(highs[step_index][index], highs[step_index + 1][index]),
                    max(highs[step_index][index], highs[step_index + 1][index]),
                ]
            )
            upper_rhs_low, upper_rhs_high = _subdivided_interval_rhs_component(
                t_interval,
                tuple(upper_box),
                b_interval,
                index,
                subdivisions,
                time_subdivisions,
            )
            upper_slope = (highs[step_index + 1][index] - highs[step_index][index]) / step_size
            margin = upper_slope - upper_rhs_high
            if margin < worst_margin:
                worst_margin = margin
                worst_face = {
                    "side": "upper",
                    "step_index": step_index,
                    "component": index,
                    "rhs_lower": upper_rhs_low,
                    "rhs_upper": upper_rhs_high,
                    "face_slope": upper_slope,
                    "time_interval": [t0, t1],
                }
            if margin < 0.0:
                return {
                    "status": "failed",
                    "candidate_A": candidate_a,
                    "b_interval": [-beta, beta],
                    "start_time": start_time,
                    "end_time": end_time,
                    "step_size": step_size,
                    "steps": steps,
                    "lower_radius0": list(lower_radius0),
                    "upper_radius0": list(upper_radius0),
                    "lower_radius_growth": list(lower_radius_growth),
                    "upper_radius_growth": list(upper_radius_growth),
                    "subdivisions": list(subdivisions),
                    "time_subdivisions": time_subdivisions,
                    "start_box": {"low": lows[0], "high": highs[0]},
                    "end_box": {"low": lows[-1], "high": highs[-1]},
                    "worst_margin": worst_margin,
                    "failing_face": worst_face,
                    "conditional": "start_box_contains_true_state",
                }

    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "start_time": start_time,
        "end_time": end_time,
        "step_size": step_size,
        "steps": steps,
        "lower_radius0": list(lower_radius0),
        "upper_radius0": list(upper_radius0),
        "lower_radius_growth": list(lower_radius_growth),
        "upper_radius_growth": list(upper_radius_growth),
        "subdivisions": list(subdivisions),
        "time_subdivisions": time_subdivisions,
        "start_box": {"low": lows[0], "high": highs[0]},
        "end_box": {"low": lows[-1], "high": highs[-1]},
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "conditional": "start_box_contains_true_state",
    }


def _tube_block_certificate(
    start_time: float,
    step_size: float,
    block_steps: int,
    candidate_a: float,
    start_samples: tuple[tuple[float, float, float, float], ...],
    start_low: tuple[float, float, float, float],
    start_high: tuple[float, float, float, float],
    lower_radius_growth: tuple[float, float, float, float],
    upper_radius_growth: tuple[float, float, float, float],
    subdivisions: tuple[int, int, int, int],
    time_subdivisions: int,
) -> dict:
    """Certify one segmented moving-tube block from a supplied start box."""
    from mpmath import iv

    beta = 1.0 / candidate_a
    b_values = (-beta, 0.0, beta)
    samples = [[start_samples[index]] for index in range(3)]
    for step_index in range(block_steps):
        t = start_time + step_index * step_size
        for index, b in enumerate(b_values):
            samples[index].append(_rk4_step_b(t, samples[index][-1], step_size, b))

    lower_radius0 = []
    upper_radius0 = []
    for index in range(4):
        nominal_low = min(sample[0][index] for sample in samples)
        nominal_high = max(sample[0][index] for sample in samples)
        lower_radius0.append(max(0.0, nominal_low - start_low[index]))
        upper_radius0.append(max(0.0, start_high[index] - nominal_high))

    lows: list[list[float]] = []
    highs: list[list[float]] = []
    for step_index in range(block_steps + 1):
        t = start_time + step_index * step_size
        lower_radius = [
            lower_radius0[index] + lower_radius_growth[index] * (t - start_time)
            for index in range(4)
        ]
        upper_radius = [
            upper_radius0[index] + upper_radius_growth[index] * (t - start_time)
            for index in range(4)
        ]
        lows.append(
            [
                min(sample[step_index][index] for sample in samples) - lower_radius[index]
                for index in range(4)
            ]
        )
        highs.append(
            [
                max(sample[step_index][index] for sample in samples) + upper_radius[index]
                for index in range(4)
            ]
        )

    b_interval = iv.mpf([-beta, beta])
    worst_margin = math.inf
    worst_face: dict | None = None
    for step_index in range(block_steps):
        t0 = start_time + step_index * step_size
        t1 = t0 + step_size
        t_interval = iv.mpf([t0, t1])
        union_box = [
            iv.mpf(
                [
                    min(lows[step_index][index], lows[step_index + 1][index]),
                    max(highs[step_index][index], highs[step_index + 1][index]),
                ]
            )
            for index in range(4)
        ]
        for index in range(4):
            lower_box = list(union_box)
            lower_box[index] = iv.mpf(
                [
                    min(lows[step_index][index], lows[step_index + 1][index]),
                    max(lows[step_index][index], lows[step_index + 1][index]),
                ]
            )
            lower_rhs_low, lower_rhs_high = _subdivided_interval_rhs_component(
                t_interval,
                tuple(lower_box),
                b_interval,
                index,
                subdivisions,
                time_subdivisions,
            )
            lower_slope = (lows[step_index + 1][index] - lows[step_index][index]) / step_size
            margin = lower_rhs_low - lower_slope
            if margin < worst_margin:
                worst_margin = margin
                worst_face = {
                    "side": "lower",
                    "step_index": step_index,
                    "component": index,
                    "rhs_lower": lower_rhs_low,
                    "rhs_upper": lower_rhs_high,
                    "face_slope": lower_slope,
                    "time_interval": [t0, t1],
                }
            if margin < 0.0:
                return {
                    "status": "failed",
                    "end_samples": [sample[-1] for sample in samples],
                    "start_box": {"low": list(start_low), "high": list(start_high)},
                    "end_box": {"low": lows[-1], "high": highs[-1]},
                    "worst_margin": worst_margin,
                    "failing_face": worst_face,
                    "lower_radius0": lower_radius0,
                    "upper_radius0": upper_radius0,
                    "lower_radius_growth": list(lower_radius_growth),
                    "upper_radius_growth": list(upper_radius_growth),
                }

            upper_box = list(union_box)
            upper_box[index] = iv.mpf(
                [
                    min(highs[step_index][index], highs[step_index + 1][index]),
                    max(highs[step_index][index], highs[step_index + 1][index]),
                ]
            )
            upper_rhs_low, upper_rhs_high = _subdivided_interval_rhs_component(
                t_interval,
                tuple(upper_box),
                b_interval,
                index,
                subdivisions,
                time_subdivisions,
            )
            upper_slope = (highs[step_index + 1][index] - highs[step_index][index]) / step_size
            margin = upper_slope - upper_rhs_high
            if margin < worst_margin:
                worst_margin = margin
                worst_face = {
                    "side": "upper",
                    "step_index": step_index,
                    "component": index,
                    "rhs_lower": upper_rhs_low,
                    "rhs_upper": upper_rhs_high,
                    "face_slope": upper_slope,
                    "time_interval": [t0, t1],
                }
            if margin < 0.0:
                return {
                    "status": "failed",
                    "end_samples": [sample[-1] for sample in samples],
                    "start_box": {"low": list(start_low), "high": list(start_high)},
                    "end_box": {"low": lows[-1], "high": highs[-1]},
                    "worst_margin": worst_margin,
                    "failing_face": worst_face,
                    "lower_radius0": lower_radius0,
                    "upper_radius0": upper_radius0,
                    "lower_radius_growth": list(lower_radius_growth),
                    "upper_radius_growth": list(upper_radius_growth),
                }

    return {
        "status": "certified",
        "end_samples": [sample[-1] for sample in samples],
        "start_box": {"low": list(start_low), "high": list(start_high)},
        "end_box": {"low": lows[-1], "high": highs[-1]},
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "lower_radius0": lower_radius0,
        "upper_radius0": upper_radius0,
        "lower_radius_growth": list(lower_radius_growth),
        "upper_radius_growth": list(upper_radius_growth),
    }


def segmented_moving_tube_certificate(
    start_time: float = DEFAULT_SUPPORT_TIME,
    end_time: float = DEFAULT_TERMINAL_BARRIER_TIME,
    step_size: float = 1e-4,
    block_steps: int = 10,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius0: tuple[float, float, float, float] = DEFAULT_SUPPORT_TUBE_RADIUS,
    profiles: tuple[
        tuple[tuple[float, float, float, float], tuple[float, float, float, float]],
        ...
    ] = DEFAULT_SEGMENTED_TUBE_PROFILES,
    subdivisions: tuple[int, int, int, int] = (2, 2, 1, 2),
    time_subdivisions: int = 1,
    seed_step_size: float = 1e-5,
) -> dict:
    """Certify a chain of local moving tubes, carrying boxes block by block."""
    if block_steps <= 0:
        raise ValueError("block_steps must be positive")
    beta = 1.0 / candidate_a
    total_steps = round((end_time - start_time) / step_size)
    if abs(start_time + total_steps * step_size - end_time) > 1e-12:
        raise ValueError("segmented tube interval length must be an integer multiple of step_size")
    start_samples = tuple(
        scaled_state_at(
            "limit" if b == 0.0 else "exact",
            start_time,
            None if b == 0.0 else 1.0 / b,
            step_size=seed_step_size,
        )
        for b in (-beta, 0.0, beta)
    )
    low = tuple(min(sample[index] for sample in start_samples) - radius0[index] for index in range(4))
    high = tuple(max(sample[index] for sample in start_samples) + radius0[index] for index in range(4))
    samples = start_samples
    certified_blocks = 0
    certified_until = start_time
    last_block: dict | None = None

    while certified_blocks * block_steps < total_steps:
        remaining_steps = total_steps - certified_blocks * block_steps
        current_block_steps = min(block_steps, remaining_steps)
        block_start = start_time + certified_blocks * block_steps * step_size
        candidates = []
        failures = []
        for lower_growth, upper_growth in profiles:
            block = _tube_block_certificate(
                block_start,
                step_size,
                current_block_steps,
                candidate_a,
                samples,
                low,
                high,
                lower_growth,
                upper_growth,
                subdivisions,
                time_subdivisions,
            )
            if block["status"] == "certified":
                end_low = block["end_box"]["low"]
                end_high = block["end_box"]["high"]
                width_sum = sum(end_high[index] - end_low[index] for index in range(4))
                candidates.append((width_sum, block))
            else:
                failures.append(block)
        if not candidates:
            best_failure = max(failures, key=lambda item: item["worst_margin"]) if failures else None
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "b_interval": [-beta, beta],
                "start_time": start_time,
                "end_time": end_time,
                "step_size": step_size,
                "block_steps": block_steps,
                "subdivisions": list(subdivisions),
                "time_subdivisions": time_subdivisions,
                "certified_until": certified_until,
                "blocks_certified": certified_blocks,
                "current_start_box": {"low": list(low), "high": list(high)},
                "failing_block": best_failure,
                "last_certified_block": last_block,
                "conditional": "initial_start_box_contains_true_state",
            }
        _width, block = min(candidates, key=lambda item: item[0])
        last_block = {key: value for key, value in block.items() if key != "end_samples"}
        samples = tuple(tuple(sample) for sample in block["end_samples"])
        low = tuple(block["end_box"]["low"])
        high = tuple(block["end_box"]["high"])
        certified_blocks += 1
        certified_until = start_time + min(certified_blocks * block_steps, total_steps) * step_size

    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "start_time": start_time,
        "end_time": end_time,
        "step_size": step_size,
        "block_steps": block_steps,
        "subdivisions": list(subdivisions),
        "time_subdivisions": time_subdivisions,
        "certified_until": certified_until,
        "blocks_certified": certified_blocks,
        "end_box": {"low": list(low), "high": list(high)},
        "last_certified_block": last_block,
        "conditional": "initial_start_box_contains_true_state",
    }


def _grow_profile_component(
    value: float,
    component: int,
    factor: float,
    max_growth: tuple[float, float, float, float],
) -> float:
    """Return a larger local tube growth for one failed component."""
    floor = max_growth[component] * 1e-6
    candidate = max(value * factor, value + floor)
    return min(candidate, max_growth[component])


def tuned_tube_block_certificate(
    start_time: float,
    step_size: float,
    block_steps: int,
    candidate_a: float,
    start_samples: tuple[tuple[float, float, float, float], ...],
    start_low: tuple[float, float, float, float],
    start_high: tuple[float, float, float, float],
    initial_growth: tuple[float, float, float, float] = DEFAULT_TUNED_TUBE_INITIAL_GROWTH,
    max_growth: tuple[float, float, float, float] = DEFAULT_TUNED_TUBE_MAX_GROWTH,
    growth_factor: float = DEFAULT_TUNED_TUBE_GROWTH_FACTOR,
    max_attempts: int = DEFAULT_TUNED_TUBE_MAX_ATTEMPTS,
    subdivisions: tuple[int, int, int, int] = (1, 1, 1, 1),
    time_subdivisions: int = 1,
) -> dict:
    """Tune one centered t-time tube block from failed face diagnostics."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if growth_factor <= 1.0:
        raise ValueError("growth_factor must be greater than 1")
    lower_growth = list(initial_growth)
    upper_growth = list(initial_growth)
    attempts = []
    best_failure: dict | None = None

    for attempt_index in range(max_attempts):
        block = _tube_block_certificate(
            start_time,
            step_size,
            block_steps,
            candidate_a,
            start_samples,
            start_low,
            start_high,
            tuple(lower_growth),
            tuple(upper_growth),
            subdivisions,
            time_subdivisions,
        )
        face = block.get("failing_face") or block.get("worst_face")
        attempts.append(
            {
                "attempt": attempt_index,
                "status": block["status"],
                "worst_margin": block["worst_margin"],
                "face": face,
                "lower_growth": list(lower_growth),
                "upper_growth": list(upper_growth),
            }
        )
        if block["status"] == "certified":
            block["tuning_attempts"] = attempts
            return block
        if best_failure is None or block["worst_margin"] > best_failure["worst_margin"]:
            best_failure = block
        if face is None:
            break
        component = int(face["component"])
        if face["side"] == "lower":
            new_value = _grow_profile_component(
                lower_growth[component],
                component,
                growth_factor,
                max_growth,
            )
            if new_value <= lower_growth[component]:
                break
            lower_growth[component] = new_value
        else:
            new_value = _grow_profile_component(
                upper_growth[component],
                component,
                growth_factor,
                max_growth,
            )
            if new_value <= upper_growth[component]:
                break
            upper_growth[component] = new_value

    failure = best_failure if best_failure is not None else block
    return {
        "status": "failed",
        "end_samples": failure.get("end_samples"),
        "start_box": {"low": list(start_low), "high": list(start_high)},
        "end_box": failure.get("end_box"),
        "worst_margin": failure.get("worst_margin", -math.inf),
        "failing_face": failure.get("failing_face"),
        "lower_radius_growth": failure.get("lower_radius_growth", list(lower_growth)),
        "upper_radius_growth": failure.get("upper_radius_growth", list(upper_growth)),
        "tuning_attempts": attempts,
    }


def tuned_segmented_moving_tube_certificate(
    start_time: float = DEFAULT_REGULAR_TIME_AUTOMATIC_START,
    end_time: float = DEFAULT_SUPPORT_TIME,
    step_size: float = 1e-3,
    block_steps: int = 1,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius0: tuple[float, float, float, float] = (1e-7, 1e-6, 1e-8, 1e-7),
    initial_growth: tuple[float, float, float, float] = DEFAULT_TUNED_TUBE_INITIAL_GROWTH,
    max_growth: tuple[float, float, float, float] = DEFAULT_TUNED_TUBE_MAX_GROWTH,
    growth_factor: float = DEFAULT_TUNED_TUBE_GROWTH_FACTOR,
    max_attempts: int = DEFAULT_TUNED_TUBE_MAX_ATTEMPTS,
    subdivisions: tuple[int, int, int, int] = (1, 1, 1, 1),
    time_subdivisions: int = 1,
    seed_step_size: float = 1e-5,
) -> dict:
    """Certify a centered t-time tube with local automatic profile tuning."""
    if block_steps <= 0:
        raise ValueError("block_steps must be positive")
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    total_steps = round((end_time - start_time) / step_size)
    if abs(start_time + total_steps * step_size - end_time) > 1e-12:
        raise ValueError("tuned tube interval length must be an integer multiple of step_size")
    beta = 1.0 / candidate_a
    start_samples = tuple(
        scaled_state_at(
            "limit" if b == 0.0 else "exact",
            start_time,
            None if b == 0.0 else 1.0 / b,
            step_size=seed_step_size,
        )
        for b in (-beta, 0.0, beta)
    )
    low = tuple(min(sample[index] for sample in start_samples) - radius0[index] for index in range(4))
    high = tuple(max(sample[index] for sample in start_samples) + radius0[index] for index in range(4))
    samples = start_samples
    certified_blocks = 0
    certified_until = start_time
    last_block: dict | None = None
    worst_margin = math.inf
    worst_face: dict | None = None
    tuning_attempt_count = 0

    while certified_blocks * block_steps < total_steps:
        remaining_steps = total_steps - certified_blocks * block_steps
        current_block_steps = min(block_steps, remaining_steps)
        block_start = start_time + certified_blocks * block_steps * step_size
        block = tuned_tube_block_certificate(
            block_start,
            step_size,
            current_block_steps,
            candidate_a,
            samples,
            low,
            high,
            initial_growth=initial_growth,
            max_growth=max_growth,
            growth_factor=growth_factor,
            max_attempts=max_attempts,
            subdivisions=subdivisions,
            time_subdivisions=time_subdivisions,
        )
        tuning_attempt_count += len(block.get("tuning_attempts", ()))
        if block.get("worst_margin", math.inf) < worst_margin:
            worst_margin = block["worst_margin"]
            worst_face = block.get("failing_face") or block.get("worst_face")
        if block["status"] != "certified":
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "b_interval": [-beta, beta],
                "start_time": start_time,
                "end_time": end_time,
                "step_size": step_size,
                "block_steps": block_steps,
                "subdivisions": list(subdivisions),
                "time_subdivisions": time_subdivisions,
                "initial_growth": list(initial_growth),
                "max_growth": list(max_growth),
                "growth_factor": growth_factor,
                "max_attempts": max_attempts,
                "certified_until": certified_until,
                "blocks_certified": certified_blocks,
                "tuning_attempt_count": tuning_attempt_count,
                "current_start_box": {"low": list(low), "high": list(high)},
                "failing_block": block,
                "last_certified_block": last_block,
                "worst_margin": worst_margin,
                "worst_face": worst_face,
                "conditional": "initial_start_box_contains_true_state",
            }
        last_block = {key: value for key, value in block.items() if key != "end_samples"}
        samples = tuple(tuple(sample) for sample in block["end_samples"])
        low = tuple(block["end_box"]["low"])
        high = tuple(block["end_box"]["high"])
        certified_blocks += 1
        certified_until = start_time + min(certified_blocks * block_steps, total_steps) * step_size

    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "start_time": start_time,
        "end_time": end_time,
        "step_size": step_size,
        "block_steps": block_steps,
        "subdivisions": list(subdivisions),
        "time_subdivisions": time_subdivisions,
        "initial_growth": list(initial_growth),
        "max_growth": list(max_growth),
        "growth_factor": growth_factor,
        "max_attempts": max_attempts,
        "certified_until": certified_until,
        "blocks_certified": certified_blocks,
        "tuning_attempt_count": tuning_attempt_count,
        "end_box": {"low": list(low), "high": list(high)},
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "last_certified_block": last_block,
        "conditional": "initial_start_box_contains_true_state",
    }


def restart_tuned_time_chain_certificate(
    start_time: float = 2.0,
    end_time: float = DEFAULT_SUPPORT_TIME,
    restart_interval: float = 0.05,
    step_size: float = 1e-3,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius0: tuple[float, float, float, float] = (1e-7, 1e-6, 1e-8, 1e-7),
    initial_growth: tuple[float, float, float, float] = DEFAULT_TUNED_TUBE_INITIAL_GROWTH,
    max_growth: tuple[float, float, float, float] = (2.0, 20.0, 20.0, 2.0),
    growth_factor: float = DEFAULT_TUNED_TUBE_GROWTH_FACTOR,
    max_attempts: int = 200,
    subdivisions: tuple[int, int, int, int] = (1, 1, 1, 1),
    time_subdivisions: int = 1,
    seed_step_size: float = 1e-5,
    restart_padding: float = 1e-10,
) -> dict:
    """Compose tuned centered tubes with explicit centered restart boxes."""
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    if restart_interval <= 0.0:
        raise ValueError("restart_interval must be positive")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")

    beta = 1.0 / candidate_a
    samples = _nominal_scaled_samples_at_time(start_time, candidate_a, seed_step_size)
    low = tuple(min(sample[index] for sample in samples) - radius0[index] for index in range(4))
    high = tuple(max(sample[index] for sample in samples) + radius0[index] for index in range(4))
    time = start_time
    time_tolerance = max(1e-12, step_size * 1e-9)
    segments = []
    restarts = []
    total_attempts = 0
    worst_margin = math.inf
    worst_face: dict | None = None

    while time < end_time - time_tolerance:
        segment_end = min(time + restart_interval, end_time)
        segment_steps = round((segment_end - time) / step_size)
        if segment_steps <= 0:
            time = segment_end
            break
        if abs(time + segment_steps * step_size - segment_end) > 1e-12:
            raise ValueError("each restart segment length must be an integer multiple of step_size")
        segment_start = time
        blocks_certified = 0
        segment_attempts = 0
        segment_worst = math.inf
        segment_worst_face: dict | None = None

        for _step_index in range(segment_steps):
            block = tuned_tube_block_certificate(
                time,
                step_size,
                1,
                candidate_a,
                samples,
                low,
                high,
                initial_growth=initial_growth,
                max_growth=max_growth,
                growth_factor=growth_factor,
                max_attempts=max_attempts,
                subdivisions=subdivisions,
                time_subdivisions=time_subdivisions,
            )
            attempts = len(block.get("tuning_attempts", ()))
            total_attempts += attempts
            segment_attempts += attempts
            if block.get("worst_margin", math.inf) < worst_margin:
                worst_margin = block["worst_margin"]
                worst_face = block.get("failing_face") or block.get("worst_face")
            if block.get("worst_margin", math.inf) < segment_worst:
                segment_worst = block["worst_margin"]
                segment_worst_face = block.get("failing_face") or block.get("worst_face")
            if block["status"] != "certified":
                segments.append(
                    {
                        "start_time": segment_start,
                        "target_time": segment_end,
                        "certified_until": time,
                        "blocks_certified": blocks_certified,
                        "tuning_attempts": segment_attempts,
                        "worst_margin": segment_worst,
                        "worst_face": segment_worst_face,
                        "status": "failed",
                    }
                )
                return {
                    "status": "failed",
                    "candidate_A": candidate_a,
                    "b_interval": [-beta, beta],
                    "start_time": start_time,
                    "end_time": end_time,
                    "restart_interval": restart_interval,
                    "step_size": step_size,
                    "subdivisions": list(subdivisions),
                    "time_subdivisions": time_subdivisions,
                    "initial_growth": list(initial_growth),
                    "max_growth": list(max_growth),
                    "growth_factor": growth_factor,
                    "max_attempts": max_attempts,
                    "restart_padding": restart_padding,
                    "certified_until": time,
                    "segments_certified": len(segments) - 1,
                    "blocks_certified": sum(segment["blocks_certified"] for segment in segments),
                    "tuning_attempt_count": total_attempts,
                    "segments": segments,
                    "restarts": restarts,
                    "current_start_box": {"low": list(low), "high": list(high)},
                    "failing_block": block,
                    "worst_margin": worst_margin,
                    "worst_face": worst_face,
                    "conditional": "initial_start_box_contains_true_state",
                }
            samples = tuple(tuple(sample) for sample in block["end_samples"])
            low = tuple(block["end_box"]["low"])
            high = tuple(block["end_box"]["high"])
            time += step_size
            blocks_certified += 1

        segments.append(
            {
                "start_time": segment_start,
                "target_time": segment_end,
                "certified_until": time,
                "blocks_certified": blocks_certified,
                "tuning_attempts": segment_attempts,
                "worst_margin": segment_worst,
                "worst_face": segment_worst_face,
                "status": "certified",
            }
        )
        if time < end_time - time_tolerance:
            restart = centered_restart_box_from_samples(
                time,
                samples,
                low,
                high,
                candidate_a=candidate_a,
                padding=restart_padding,
                sample_source="propagated",
            )
            if not restart["source_box_contained"]:
                return {
                    "status": "failed",
                    "candidate_A": candidate_a,
                    "stage": "restart_containment",
                    "certified_until": time,
                    "restart": restart,
                    "segments": segments,
                    "restarts": restarts,
                    "worst_margin": worst_margin,
                    "worst_face": worst_face,
                    "conditional": "initial_start_box_contains_true_state",
                }
            restarts.append(restart)
            samples = tuple(tuple(sample) for sample in restart["samples"])
            low = tuple(restart["box"]["low"])
            high = tuple(restart["box"]["high"])

    if abs(time - end_time) <= time_tolerance:
        time = end_time

    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "start_time": start_time,
        "end_time": end_time,
        "restart_interval": restart_interval,
        "step_size": step_size,
        "subdivisions": list(subdivisions),
        "time_subdivisions": time_subdivisions,
        "initial_growth": list(initial_growth),
        "max_growth": list(max_growth),
        "growth_factor": growth_factor,
        "max_attempts": max_attempts,
        "restart_padding": restart_padding,
        "certified_until": time,
        "segments_certified": len(segments),
        "blocks_certified": sum(segment["blocks_certified"] for segment in segments),
        "tuning_attempt_count": total_attempts,
        "end_box": {"low": list(low), "high": list(high)},
        "segments": segments,
        "restarts": restarts,
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "conditional": "initial_start_box_contains_true_state",
    }


def affine_time_corridor_certificate(
    start_time: float,
    end_time: float,
    candidate_a: float,
    lower_start: tuple[float, float, float, float],
    upper_start: tuple[float, float, float, float],
    lower_slope: tuple[float, float, float, float],
    upper_slope: tuple[float, float, float, float],
    subdivisions: tuple[int, int, int, int] = DEFAULT_REGULAR_TIME_AUTOMATIC_SUBDIVISIONS,
    time_subdivisions: int = DEFAULT_REGULAR_TIME_AUTOMATIC_TIME_SUBDIVISIONS,
    source_box_low: tuple[float, float, float, float] | None = None,
    source_box_high: tuple[float, float, float, float] | None = None,
) -> dict:
    """Check one affine barrier corridor in ordinary increasing time.

    On a lower face the forward-time inward condition is ``L' <= F_j``; on an
    upper face it is ``F_j <= U'``.  The check evaluates the finite-|a| scaled
    vector field on interval slabs for every ``b=1/a`` in the requested range.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    if not all(lower_start[index] <= upper_start[index] for index in range(4)):
        raise ValueError("lower_start must be componentwise <= upper_start")

    from mpmath import iv

    beta = 1.0 / candidate_a
    b_interval = iv.mpf([-beta, beta])
    t_interval = iv.mpf([start_time, end_time])
    lower0 = lower_start
    lower1 = _affine_time_barrier_value(lower_start, lower_slope, start_time, end_time)
    upper0 = upper_start
    upper1 = _affine_time_barrier_value(upper_start, upper_slope, start_time, end_time)
    slab_low = [min(lower0[index], lower1[index]) for index in range(4)]
    slab_high = [max(upper0[index], upper1[index]) for index in range(4)]
    if any(slab_low[index] > slab_high[index] for index in range(4)):
        return {
            "status": "failed",
            "failure": "empty_corridor_slab",
            "candidate_A": candidate_a,
            "start_time": start_time,
            "end_time": end_time,
            "slab_low": slab_low,
            "slab_high": slab_high,
        }

    union_box = [iv.mpf([slab_low[index], slab_high[index]]) for index in range(4)]
    worst_margin = math.inf
    worst_face: dict | None = None
    for index in range(4):
        lower_box = list(union_box)
        lower_face_values = [lower0[index], lower1[index]]
        lower_box[index] = iv.mpf([min(lower_face_values), max(lower_face_values)])
        lower_rhs_low, lower_rhs_high = _subdivided_interval_rhs_component(
            t_interval,
            tuple(lower_box),
            b_interval,
            index,
            subdivisions,
            time_subdivisions,
        )
        margin = lower_rhs_low - lower_slope[index]
        if margin < worst_margin:
            worst_margin = margin
            worst_face = {
                "side": "lower",
                "component": index,
                "time_interval": [start_time, end_time],
                "rhs_lower": lower_rhs_low,
                "rhs_upper": lower_rhs_high,
                "face_slope": lower_slope[index],
                "slab_low": slab_low,
                "slab_high": slab_high,
            }
        if margin < 0.0:
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "b_interval": [-beta, beta],
                "start_time": start_time,
                "end_time": end_time,
                "lower_start": list(lower_start),
                "upper_start": list(upper_start),
                "lower_slope": list(lower_slope),
                "upper_slope": list(upper_slope),
                "subdivisions": list(subdivisions),
                "time_subdivisions": time_subdivisions,
                "source_box_contained": True
                if source_box_low is None or source_box_high is None
                else _corridor_contains_box(lower_start, upper_start, source_box_low, source_box_high),
                "source_box": None
                if source_box_low is None or source_box_high is None
                else {"low": list(source_box_low), "high": list(source_box_high)},
                "worst_margin": worst_margin,
                "failing_face": worst_face,
                "conditional": "source_start_box_contains_true_state",
            }

        upper_box = list(union_box)
        upper_face_values = [upper0[index], upper1[index]]
        upper_box[index] = iv.mpf([min(upper_face_values), max(upper_face_values)])
        upper_rhs_low, upper_rhs_high = _subdivided_interval_rhs_component(
            t_interval,
            tuple(upper_box),
            b_interval,
            index,
            subdivisions,
            time_subdivisions,
        )
        margin = upper_slope[index] - upper_rhs_high
        if margin < worst_margin:
            worst_margin = margin
            worst_face = {
                "side": "upper",
                "component": index,
                "time_interval": [start_time, end_time],
                "rhs_lower": upper_rhs_low,
                "rhs_upper": upper_rhs_high,
                "face_slope": upper_slope[index],
                "slab_low": slab_low,
                "slab_high": slab_high,
            }
        if margin < 0.0:
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "b_interval": [-beta, beta],
                "start_time": start_time,
                "end_time": end_time,
                "lower_start": list(lower_start),
                "upper_start": list(upper_start),
                "lower_slope": list(lower_slope),
                "upper_slope": list(upper_slope),
                "subdivisions": list(subdivisions),
                "time_subdivisions": time_subdivisions,
                "source_box_contained": True
                if source_box_low is None or source_box_high is None
                else _corridor_contains_box(lower_start, upper_start, source_box_low, source_box_high),
                "source_box": None
                if source_box_low is None or source_box_high is None
                else {"low": list(source_box_low), "high": list(source_box_high)},
                "worst_margin": worst_margin,
                "failing_face": worst_face,
                "conditional": "source_start_box_contains_true_state",
            }

    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "start_time": start_time,
        "end_time": end_time,
        "lower_start": list(lower_start),
        "upper_start": list(upper_start),
        "lower_slope": list(lower_slope),
        "upper_slope": list(upper_slope),
        "end_box": {"low": list(lower1), "high": list(upper1)},
        "subdivisions": list(subdivisions),
        "time_subdivisions": time_subdivisions,
        "source_box_contained": True
        if source_box_low is None or source_box_high is None
        else _corridor_contains_box(lower_start, upper_start, source_box_low, source_box_high),
        "source_box": None
        if source_box_low is None or source_box_high is None
        else {"low": list(source_box_low), "high": list(source_box_high)},
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "conditional": "source_start_box_contains_true_state",
    }


def automatic_time_barrier_corridor_certificate(
    start_time: float = DEFAULT_REGULAR_TIME_AUTOMATIC_START,
    end_time: float = DEFAULT_REGULAR_TIME_AUTOMATIC_END,
    step_size: float = DEFAULT_REGULAR_TIME_AUTOMATIC_STEP,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius0: tuple[float, float, float, float] = DEFAULT_REGULAR_TIME_AUTOMATIC_RADIUS0,
    safety=DEFAULT_REGULAR_TIME_AUTOMATIC_SAFETY,
    subdivisions: tuple[int, int, int, int] = DEFAULT_REGULAR_TIME_AUTOMATIC_SUBDIVISIONS,
    time_subdivisions: int = DEFAULT_REGULAR_TIME_AUTOMATIC_TIME_SUBDIVISIONS,
    seed_step_size: float = 1e-5,
) -> dict:
    """Greedily certify an ordinary-time barrier corridor.

    This is the forward-time analogue of ``automatic_p_barrier_corridor``.
    It is conditional on the start-time box containing the true scaled IVP.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    safety_tuple = _component_safety_tuple(safety)
    total_steps = round((end_time - start_time) / step_size)
    if abs(start_time + total_steps * step_size - end_time) > 1e-12:
        raise ValueError("automatic time corridor length must be an integer multiple of step_size")

    from mpmath import iv

    beta = 1.0 / candidate_a
    b_values = (-beta, 0.0, beta)
    samples = tuple(
        scaled_state_at(
            "limit" if b == 0.0 else "exact",
            start_time,
            None if b == 0.0 else 1.0 / b,
            step_size=seed_step_size,
        )
        for b in b_values
    )
    low = tuple(min(sample[index] for sample in samples) - radius0[index] for index in range(4))
    high = tuple(max(sample[index] for sample in samples) + radius0[index] for index in range(4))
    b_interval = iv.mpf([-beta, beta])
    time = start_time
    worst_margin = math.inf
    worst_face: dict | None = None
    last_step: dict | None = None

    for step_index in range(total_steps):
        t_interval = iv.mpf([time, time])
        current_box = tuple(iv.mpf([low[index], high[index]]) for index in range(4))
        rhs_bounds = tuple(
            _subdivided_interval_rhs_component(
                t_interval,
                current_box,
                b_interval,
                component,
                subdivisions,
                time_subdivisions,
            )
            for component in range(4)
        )
        lower_slope = tuple(bound[0] - safety_tuple[index] for index, bound in enumerate(rhs_bounds))
        upper_slope = tuple(bound[1] + safety_tuple[index] for index, bound in enumerate(rhs_bounds))
        next_time = time + step_size
        next_low = tuple(low[index] + step_size * lower_slope[index] for index in range(4))
        next_high = tuple(high[index] + step_size * upper_slope[index] for index in range(4))
        if any(next_low[index] > next_high[index] for index in range(4)):
            return {
                "status": "failed",
                "failure": "empty_next_box",
                "candidate_A": candidate_a,
                "start_time": start_time,
                "end_time": end_time,
                "step_size": step_size,
                "safety": list(safety_tuple),
                "subdivisions": list(subdivisions),
                "time_subdivisions": time_subdivisions,
                "certified_until": time,
                "steps_certified": step_index,
                "current_box": {"low": list(low), "high": list(high)},
                "proposed_next_box": {"low": list(next_low), "high": list(next_high)},
                "rhs_bounds": [list(bound) for bound in rhs_bounds],
                "conditional": "start_box_contains_true_state",
            }

        step_certificate = affine_time_corridor_certificate(
            start_time=time,
            end_time=next_time,
            candidate_a=candidate_a,
            lower_start=low,
            upper_start=high,
            lower_slope=lower_slope,
            upper_slope=upper_slope,
            subdivisions=subdivisions,
            time_subdivisions=time_subdivisions,
            source_box_low=low,
            source_box_high=high,
        )
        if step_certificate["status"] != "certified":
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "start_time": start_time,
                "end_time": end_time,
                "step_size": step_size,
                "safety": list(safety_tuple),
                "subdivisions": list(subdivisions),
                "time_subdivisions": time_subdivisions,
                "certified_until": time,
                "steps_certified": step_index,
                "current_box": {"low": list(low), "high": list(high)},
                "proposed_next_box": {"low": list(next_low), "high": list(next_high)},
                "rhs_bounds": [list(bound) for bound in rhs_bounds],
                "failing_step": step_certificate,
                "last_certified_step": last_step,
                "conditional": "start_box_contains_true_state",
            }
        if step_certificate["worst_margin"] < worst_margin:
            worst_margin = step_certificate["worst_margin"]
            worst_face = step_certificate.get("worst_face")
        last_step = {
            "start_time": time,
            "end_time": next_time,
            "worst_margin": step_certificate["worst_margin"],
            "worst_face": step_certificate.get("worst_face"),
            "end_box": {"low": list(next_low), "high": list(next_high)},
        }
        time = next_time
        low = next_low
        high = next_high

    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "start_time": start_time,
        "end_time": end_time,
        "step_size": step_size,
        "steps": total_steps,
        "steps_certified": total_steps,
        "safety": list(safety_tuple),
        "radius0": list(radius0),
        "subdivisions": list(subdivisions),
        "time_subdivisions": time_subdivisions,
        "start_box": {
            "low": [
                min(sample[index] for sample in samples) - radius0[index]
                for index in range(4)
            ],
            "high": [
                max(sample[index] for sample in samples) + radius0[index]
                for index in range(4)
            ],
        },
        "end_box": {"low": list(low), "high": list(high)},
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "last_certified_step": last_step,
        "conditional": "start_box_contains_true_state",
    }


def late_x3_descent_certificate(
    start_time: float = DEFAULT_LATE_X3_DESCENT_START,
    end_time: float = DEFAULT_LATE_X3_DESCENT_END,
    step_size: float = DEFAULT_LATE_X3_DESCENT_STEP,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius0: tuple[float, float, float, float] = DEFAULT_LATE_X3_DESCENT_RADIUS0,
    safety=DEFAULT_LATE_X3_DESCENT_SAFETY,
    x0_target: float = DEFAULT_LATE_X3_DESCENT_X0_TARGET,
    subdivisions: tuple[int, int, int, int] = (1, 1, 1, 1),
    time_subdivisions: int = 1,
    seed_step_size: float = 2e-5,
) -> dict:
    """Certify late descent into the negative ``x3`` side.

    This bridge is conditional on the start box at ``start_time`` containing
    the true scaled trajectories.  It proves that the certified end box has
    ``x3 < 0`` and ``x0 < x0_target``; it does not by itself provide the later
    p-slice handoff.
    """
    corridor = automatic_time_barrier_corridor_certificate(
        start_time=start_time,
        end_time=end_time,
        step_size=step_size,
        candidate_a=candidate_a,
        radius0=radius0,
        safety=safety,
        subdivisions=subdivisions,
        time_subdivisions=time_subdivisions,
        seed_step_size=seed_step_size,
    )
    if corridor["status"] != "certified":
        return {
            "status": "failed",
            "candidate_A": candidate_a,
            "stage": "time_corridor",
            "time_corridor": corridor,
            "conditional": "late_descent_start_box_contains_true_state",
        }

    end_box = corridor["end_box"]
    end_low = end_box["low"]
    end_high = end_box["high"]
    wall = x3_zero_wall_certificate(
        time_range=DEFAULT_LATE_X3_DESCENT_WALL_TIME_RANGE,
        x0_range=DEFAULT_LATE_X3_DESCENT_WALL_X0_RANGE,
        x1_range=DEFAULT_LATE_X3_DESCENT_WALL_X1_RANGE,
        x2_range=DEFAULT_LATE_X3_DESCENT_WALL_X2_RANGE,
        candidate_a=candidate_a,
    )
    x3_negative = end_high[3] < 0.0
    x0_below_target = end_high[0] < x0_target
    positive_x0_floor = end_low[0] > 0.0
    positive_x2_floor = end_low[2] > 0.0
    end_box_contained_in_wall_box = (
        DEFAULT_LATE_X3_DESCENT_WALL_X0_RANGE[0]
        <= end_low[0]
        <= end_high[0]
        <= DEFAULT_LATE_X3_DESCENT_WALL_X0_RANGE[1]
        and DEFAULT_LATE_X3_DESCENT_WALL_X1_RANGE[0]
        <= end_low[1]
        <= end_high[1]
        <= DEFAULT_LATE_X3_DESCENT_WALL_X1_RANGE[1]
        and DEFAULT_LATE_X3_DESCENT_WALL_X2_RANGE[0]
        <= end_low[2]
        <= end_high[2]
        <= DEFAULT_LATE_X3_DESCENT_WALL_X2_RANGE[1]
    )
    status = "certified_conditional"
    if (
        not x3_negative
        or not x0_below_target
        or not positive_x0_floor
        or not positive_x2_floor
        or not end_box_contained_in_wall_box
        or wall["status"] != "certified"
    ):
        status = "failed"
    return {
        "status": status,
        "candidate_A": candidate_a,
        "start_time": start_time,
        "end_time": end_time,
        "step_size": step_size,
        "x0_target": x0_target,
        "time_corridor": corridor,
        "end_box": end_box,
        "x3_negative": x3_negative,
        "x0_below_target": x0_below_target,
        "positive_x0_floor": positive_x0_floor,
        "positive_x2_floor": positive_x2_floor,
        "end_box_contained_in_wall_box": end_box_contained_in_wall_box,
        "x3_zero_wall": wall,
        "conditional": "late_descent_start_box_contains_true_state",
        "conclusion": "from the certified start box, x3 is negative and x0 is below target at end_time",
    }


def taylor_start_box(
    start_time: float = DEFAULT_TAYLOR_START_TIME,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius: tuple[float, float, float, float] = DEFAULT_TAYLOR_START_RADIUS,
) -> dict:
    """Return the c2 Taylor start box for the smooth scaled singular-end IVP."""
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if start_time <= 0.0:
        raise ValueError("start_time must be positive")
    beta = 1.0 / candidate_a
    samples = tuple(scaled_taylor_seed(start_time, b) for b in (-beta, 0.0, beta))
    low = tuple(min(sample[index] for sample in samples) - radius[index] for index in range(4))
    high = tuple(max(sample[index] for sample in samples) + radius[index] for index in range(4))
    return {
        "start_time": start_time,
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "radius": list(radius),
        "c2_at_b_minus_beta": list(scaled_taylor_c2(-beta)),
        "c2_at_b_0": list(scaled_taylor_c2(0.0)),
        "c2_at_b_plus_beta": list(scaled_taylor_c2(beta)),
        "samples": [list(sample) for sample in samples],
        "box": {"low": list(low), "high": list(high)},
        "conditional": "taylor_remainder_is_inside_radius",
    }


def taylor_start_block_certificate(
    start_time: float = DEFAULT_TAYLOR_START_TIME,
    step_size: float = DEFAULT_TAYLOR_START_STEP,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius: tuple[float, float, float, float] = DEFAULT_TAYLOR_START_RADIUS,
    safety=DEFAULT_TAYLOR_START_SAFETY,
    subdivisions: tuple[int, int, int, int] = DEFAULT_REGULAR_TIME_AUTOMATIC_SUBDIVISIONS,
    time_subdivisions: int = DEFAULT_REGULAR_TIME_AUTOMATIC_TIME_SUBDIVISIONS,
) -> dict:
    """Certify the first ordinary-time slab from the c2 Taylor start box.

    The result is conditional on the true smooth IVP value at ``start_time``
    lying in the Taylor start box.  It is designed to make the singular-end
    proof obligation explicit and reproducible.
    """
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    start = taylor_start_box(start_time=start_time, candidate_a=candidate_a, radius=radius)
    low = tuple(start["box"]["low"])
    high = tuple(start["box"]["high"])
    safety_tuple = _component_safety_tuple(safety)

    from mpmath import iv

    beta = 1.0 / candidate_a
    b_interval = iv.mpf([-beta, beta])
    t_interval = iv.mpf([start_time, start_time])
    current_box = tuple(iv.mpf([low[index], high[index]]) for index in range(4))
    rhs_bounds = tuple(
        _subdivided_interval_rhs_component(
            t_interval,
            current_box,
            b_interval,
            component,
            subdivisions,
            time_subdivisions,
        )
        for component in range(4)
    )
    lower_slope = tuple(bound[0] - safety_tuple[index] for index, bound in enumerate(rhs_bounds))
    upper_slope = tuple(bound[1] + safety_tuple[index] for index, bound in enumerate(rhs_bounds))
    step = affine_time_corridor_certificate(
        start_time=start_time,
        end_time=start_time + step_size,
        candidate_a=candidate_a,
        lower_start=low,
        upper_start=high,
        lower_slope=lower_slope,
        upper_slope=upper_slope,
        subdivisions=subdivisions,
        time_subdivisions=time_subdivisions,
        source_box_low=low,
        source_box_high=high,
    )
    status = "certified_conditional" if step["status"] == "certified" else "failed"
    return {
        "status": status,
        "candidate_A": candidate_a,
        "start_time": start_time,
        "end_time": start_time + step_size,
        "step_size": step_size,
        "taylor_start_box": start,
        "rhs_bounds_at_start": [list(bound) for bound in rhs_bounds],
        "lower_slope": list(lower_slope),
        "upper_slope": list(upper_slope),
        "safety": list(safety_tuple),
        "step_certificate": step,
        "worst_margin": step.get("worst_margin"),
        "conditional": "taylor_remainder_is_inside_radius",
    }


def taylor_time_bridge_certificate(
    end_time: float = DEFAULT_TAYLOR_BRIDGE_END,
    start_time: float = DEFAULT_TAYLOR_START_TIME,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius: tuple[float, float, float, float] = DEFAULT_TAYLOR_START_RADIUS,
    stages: tuple = DEFAULT_TAYLOR_BRIDGE_STAGES,
    max_attempts: int = 180,
    subdivisions: tuple[int, int, int, int] = DEFAULT_REGULAR_TIME_AUTOMATIC_SUBDIVISIONS,
    time_subdivisions: int = DEFAULT_REGULAR_TIME_AUTOMATIC_TIME_SUBDIVISIONS,
    progress_callback=None,
    progress_every_blocks: int = 0,
) -> dict:
    """Certify a staged ordinary-time bridge from the singular Taylor box.

    This is a proof-building certificate for the compact regular interval.  It
    deliberately starts from ``taylor_start_box`` rather than from the
    uncorrected numerical epsilon state used by lightweight diagnostics.  The
    current axis-aligned boxes certify only an initial compact bridge and become
    too wide for the late support handoff; the returned widths make that
    obstruction explicit and reproducible.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    start = taylor_start_box(start_time=start_time, candidate_a=candidate_a, radius=radius)
    time = start_time
    samples = tuple(tuple(sample) for sample in start["samples"])
    low = tuple(start["box"]["low"])
    high = tuple(start["box"]["high"])
    segments = []
    total_attempts = 0
    total_blocks = 0
    worst_margin = math.inf
    worst_face: dict | None = None

    for stage_index, stage in enumerate(stages):
        stage_end, step_size, block_steps, initial_growth, max_growth = stage
        if stage_end <= time + 1e-15:
            continue
        segment_end = min(float(stage_end), end_time)
        if segment_end <= time + 1e-15:
            continue
        if step_size <= 0.0:
            raise ValueError("stage step_size must be positive")
        if block_steps <= 0:
            raise ValueError("stage block_steps must be positive")
        total_steps = round((segment_end - time) / step_size)
        if abs(time + total_steps * step_size - segment_end) > 1e-12:
            raise ValueError("stage endpoint must align with its step size")

        segment_start = time
        segment_blocks = 0
        segment_attempts = 0
        segment_worst = math.inf
        segment_worst_face: dict | None = None

        while time < segment_end - 1e-15:
            remaining_steps = round((segment_end - time) / step_size)
            if remaining_steps <= 0:
                time = segment_end
                break
            current_block_steps = min(int(block_steps), remaining_steps)
            block = tuned_tube_block_certificate(
                time,
                step_size,
                current_block_steps,
                candidate_a,
                samples,
                low,
                high,
                initial_growth=tuple(initial_growth),
                max_growth=tuple(max_growth),
                max_attempts=max_attempts,
                subdivisions=subdivisions,
                time_subdivisions=time_subdivisions,
            )
            attempts = len(block.get("tuning_attempts", ()))
            total_attempts += attempts
            segment_attempts += attempts
            face = block.get("failing_face") or block.get("worst_face")
            if block.get("worst_margin", math.inf) < worst_margin:
                worst_margin = block["worst_margin"]
                worst_face = face
            if block.get("worst_margin", math.inf) < segment_worst:
                segment_worst = block["worst_margin"]
                segment_worst_face = face
            if block["status"] != "certified":
                segments.append(
                    {
                        "stage_index": stage_index,
                        "start_time": segment_start,
                        "target_time": segment_end,
                        "certified_until": time,
                        "step_size": step_size,
                        "block_steps": block_steps,
                        "blocks_certified": segment_blocks,
                        "tuning_attempts": segment_attempts,
                        "worst_margin": segment_worst,
                        "worst_face": segment_worst_face,
                        "status": "failed",
                    }
                )
                return {
                    "status": "failed",
                    "candidate_A": candidate_a,
                    "start_time": start_time,
                    "end_time": end_time,
                    "certified_until": time,
                    "taylor_start_box": start,
                    "segments": segments,
                    "blocks_certified": total_blocks,
                    "tuning_attempt_count": total_attempts,
                    "current_box": {"low": list(low), "high": list(high)},
                    "current_samples": [list(sample) for sample in samples],
                    "current_width": [high[index] - low[index] for index in range(4)],
                    "failing_block": block,
                    "worst_margin": worst_margin,
                    "worst_face": worst_face,
                    "conditional": "taylor_remainder_is_inside_radius",
                }

            samples = tuple(tuple(sample) for sample in block["end_samples"])
            low = tuple(block["end_box"]["low"])
            high = tuple(block["end_box"]["high"])
            time += current_block_steps * step_size
            segment_blocks += 1
            total_blocks += 1
            if (
                progress_callback is not None
                and progress_every_blocks > 0
                and (total_blocks % progress_every_blocks == 0 or time >= segment_end - 1e-15)
            ):
                progress_callback(
                    {
                        "event": "taylor_time_bridge_progress",
                        "stage_index": stage_index,
                        "certified_until": time,
                        "stage_target": segment_end,
                        "end_time": end_time,
                        "blocks_certified": total_blocks,
                        "stage_blocks_certified": segment_blocks,
                        "tuning_attempt_count": total_attempts,
                        "worst_margin": worst_margin,
                        "current_width": [high[index] - low[index] for index in range(4)],
                    }
                )

        segments.append(
            {
                "stage_index": stage_index,
                "start_time": segment_start,
                "target_time": segment_end,
                "certified_until": time,
                "step_size": step_size,
                "block_steps": block_steps,
                "blocks_certified": segment_blocks,
                "tuning_attempts": segment_attempts,
                "worst_margin": segment_worst,
                "worst_face": segment_worst_face,
                "status": "certified",
                "end_box": {"low": list(low), "high": list(high)},
                "end_width": [high[index] - low[index] for index in range(4)],
            }
        )
        if time >= end_time - 1e-15:
            break

    if time < end_time - 1e-15:
        return {
            "status": "failed",
            "failure": "no_stage_reaches_requested_end_time",
            "candidate_A": candidate_a,
            "start_time": start_time,
            "end_time": end_time,
            "certified_until": time,
            "taylor_start_box": start,
            "segments": segments,
            "current_box": {"low": list(low), "high": list(high)},
            "current_samples": [list(sample) for sample in samples],
            "current_width": [high[index] - low[index] for index in range(4)],
            "conditional": "taylor_remainder_is_inside_radius",
        }

    return {
        "status": "certified_conditional",
        "candidate_A": candidate_a,
        "start_time": start_time,
        "end_time": end_time,
        "certified_until": time,
        "taylor_start_box": start,
        "segments": segments,
        "blocks_certified": total_blocks,
        "tuning_attempt_count": total_attempts,
        "end_box": {"low": list(low), "high": list(high)},
        "end_samples": [list(sample) for sample in samples],
        "end_width": [high[index] - low[index] for index in range(4)],
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "conditional": "taylor_remainder_is_inside_radius",
        "conclusion": "the true scaled trajectory is inside the end box if the Taylor remainder is inside the start radius",
    }


def taylor_frontier_continuation_certificate(
    bridge_end_time: float = DEFAULT_TAYLOR_BRIDGE_END,
    end_time: float = DEFAULT_TAYLOR_FRONTIER_END,
    step_size: float = DEFAULT_TAYLOR_FRONTIER_STEP,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius: tuple[float, float, float, float] = DEFAULT_TAYLOR_START_RADIUS,
    bridge_max_attempts: int = DEFAULT_TAYLOR_BRIDGE_MAX_ATTEMPTS,
    initial_growth: tuple[float, float, float, float] = DEFAULT_TAYLOR_FRONTIER_INITIAL_GROWTH,
    max_growth: tuple[float, float, float, float] = DEFAULT_TAYLOR_FRONTIER_MAX_GROWTH,
    growth_factor: float = DEFAULT_TUNED_TUBE_GROWTH_FACTOR,
    max_attempts: int = DEFAULT_TAYLOR_FRONTIER_MAX_ATTEMPTS,
    subdivisions: tuple[int, int, int, int] = DEFAULT_REGULAR_TIME_AUTOMATIC_SUBDIVISIONS,
    retry_subdivisions: tuple[tuple[int, int, int, int], ...] = DEFAULT_TAYLOR_FRONTIER_RETRY_SUBDIVISIONS,
    time_subdivisions: int = DEFAULT_REGULAR_TIME_AUTOMATIC_TIME_SUBDIVISIONS,
) -> dict:
    """Continue the Taylor bridge with direct tuned t-time blocks.

    This composes the certified Taylor-start bridge with a direct continuation
    from its endpoint box.  It is still an axis-aligned certificate, so it is a
    frontier diagnostic rather than the final compact-interval proof, but it
    gives a reproducible lower bound on how far the current rectangular method
    genuinely reaches.
    """
    if end_time <= bridge_end_time:
        raise ValueError("end_time must be greater than bridge_end_time")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")

    bridge = taylor_time_bridge_certificate(
        end_time=bridge_end_time,
        candidate_a=candidate_a,
        radius=radius,
        max_attempts=bridge_max_attempts,
        subdivisions=subdivisions,
        time_subdivisions=time_subdivisions,
    )
    if bridge["status"] != "certified_conditional":
        return {
            "status": "failed",
            "candidate_A": candidate_a,
            "stage": "taylor_bridge",
            "taylor_bridge": bridge,
            "conditional": "taylor_remainder_is_inside_radius",
        }

    time = bridge_end_time
    time_tolerance = max(1e-12, step_size * 1e-9)
    samples = tuple(tuple(sample) for sample in bridge["end_samples"])
    low = tuple(bridge["end_box"]["low"])
    high = tuple(bridge["end_box"]["high"])
    total_steps = round((end_time - bridge_end_time) / step_size)
    if abs(bridge_end_time + total_steps * step_size - end_time) > 1e-12:
        raise ValueError("frontier continuation length must be an integer multiple of step_size")

    blocks_certified = 0
    total_attempts = 0
    retry_count = 0
    retry_log = []
    worst_margin = math.inf
    worst_face: dict | None = None
    last_block: dict | None = None
    for _step_index in range(total_steps):
        active_subdivisions = subdivisions
        tried_blocks = []
        block = None
        for attempt_subdivisions in (subdivisions, *retry_subdivisions):
            block = tuned_tube_block_certificate(
                time,
                step_size,
                1,
                candidate_a,
                samples,
                low,
                high,
                initial_growth=initial_growth,
                max_growth=max_growth,
                growth_factor=growth_factor,
                max_attempts=max_attempts,
                subdivisions=attempt_subdivisions,
                time_subdivisions=time_subdivisions,
            )
            tried_blocks.append(block)
            active_subdivisions = attempt_subdivisions
            if block["status"] == "certified":
                if attempt_subdivisions != subdivisions:
                    retry_count += 1
                    retry_log.append(
                        {
                            "time": time,
                            "subdivisions": list(attempt_subdivisions),
                            "worst_margin": block.get("worst_margin"),
                            "worst_face": block.get("worst_face"),
                        }
                    )
                break
        assert block is not None
        total_attempts += sum(len(item.get("tuning_attempts", ())) for item in tried_blocks)
        face = block.get("failing_face") or block.get("worst_face")
        if block.get("worst_margin", math.inf) < worst_margin:
            worst_margin = block["worst_margin"]
            worst_face = face
        if block["status"] != "certified":
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "stage": "frontier_continuation",
                "bridge_end_time": bridge_end_time,
                "end_time": end_time,
                "step_size": step_size,
                "certified_until": time,
                "steps_certified": blocks_certified,
                "taylor_bridge": bridge,
                "current_box": {"low": list(low), "high": list(high)},
                "current_samples": [list(sample) for sample in samples],
                "current_width": [high[index] - low[index] for index in range(4)],
                "failing_block": block,
                "tried_blocks": [
                    {
                        "subdivisions": list((subdivisions, *retry_subdivisions)[index]),
                        "status": item["status"],
                        "worst_margin": item.get("worst_margin"),
                        "face": item.get("failing_face") or item.get("worst_face"),
                    }
                    for index, item in enumerate(tried_blocks)
                ],
                "last_certified_block": last_block,
                "tuning_attempt_count": total_attempts,
                "retry_count": retry_count,
                "retry_log": retry_log,
                "worst_margin": worst_margin,
                "worst_face": worst_face,
                "conditional": "taylor_remainder_is_inside_radius",
            }
        samples = tuple(tuple(sample) for sample in block["end_samples"])
        low = tuple(block["end_box"]["low"])
        high = tuple(block["end_box"]["high"])
        time += step_size
        blocks_certified += 1
        last_block = {key: value for key, value in block.items() if key != "end_samples"}
        last_block["subdivisions"] = list(active_subdivisions)

    return {
        "status": "certified_conditional",
        "candidate_A": candidate_a,
        "bridge_end_time": bridge_end_time,
        "end_time": end_time,
        "step_size": step_size,
        "certified_until": time,
        "steps_certified": blocks_certified,
        "taylor_bridge": bridge,
        "end_box": {"low": list(low), "high": list(high)},
        "end_samples": [list(sample) for sample in samples],
        "end_width": [high[index] - low[index] for index in range(4)],
        "last_certified_block": last_block,
        "tuning_attempt_count": total_attempts,
        "retry_count": retry_count,
        "retry_log": retry_log,
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "conditional": "taylor_remainder_is_inside_radius",
        "conclusion": "the current axis-aligned proof reaches this frontier from the Taylor start box",
    }


def taylor_restart_chain_certificate(
    bridge_end_time: float = DEFAULT_TAYLOR_BRIDGE_END,
    end_time: float = DEFAULT_TAYLOR_RESTART_CHAIN_END,
    restart_interval: float = 0.05,
    step_size: float = DEFAULT_TAYLOR_FRONTIER_STEP,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius: tuple[float, float, float, float] = DEFAULT_TAYLOR_START_RADIUS,
    bridge_max_attempts: int = DEFAULT_TAYLOR_BRIDGE_MAX_ATTEMPTS,
    initial_growth: tuple[float, float, float, float] = DEFAULT_TAYLOR_FRONTIER_INITIAL_GROWTH,
    max_growth: tuple[float, float, float, float] = DEFAULT_TAYLOR_RESTART_CHAIN_MAX_GROWTH,
    growth_factor: float = DEFAULT_TUNED_TUBE_GROWTH_FACTOR,
    max_attempts: int = DEFAULT_TAYLOR_RESTART_CHAIN_MAX_ATTEMPTS,
    subdivisions: tuple[int, int, int, int] = DEFAULT_REGULAR_TIME_AUTOMATIC_SUBDIVISIONS,
    retry_subdivisions: tuple[tuple[int, int, int, int], ...] = DEFAULT_TAYLOR_FRONTIER_RETRY_SUBDIVISIONS,
    time_subdivisions: int = DEFAULT_REGULAR_TIME_AUTOMATIC_TIME_SUBDIVISIONS,
    restart_padding: float = 1e-10,
    bridge_certificate: dict | None = None,
    bridge_progress_callback=None,
    bridge_progress_every_blocks: int = 0,
    progress_callback=None,
    progress_every_segments: int = 0,
) -> dict:
    """Compose the Taylor bridge with centered restart containment boxes.

    ``restart_tuned_time_chain_certificate`` is useful diagnostically, but it
    starts from a fresh nominal box.  This certificate starts instead from the
    verified endpoint of ``taylor_time_bridge_certificate`` and therefore
    preserves the same proof conditional all the way from the singular Taylor
    start.
    """
    if end_time <= bridge_end_time:
        raise ValueError("end_time must be greater than bridge_end_time")
    if restart_interval <= 0.0:
        raise ValueError("restart_interval must be positive")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    bridge = (
        taylor_time_bridge_certificate(
            end_time=bridge_end_time,
            candidate_a=candidate_a,
            radius=radius,
            max_attempts=bridge_max_attempts,
            subdivisions=subdivisions,
            time_subdivisions=time_subdivisions,
            progress_callback=bridge_progress_callback,
            progress_every_blocks=bridge_progress_every_blocks,
        )
        if bridge_certificate is None
        else bridge_certificate
    )
    if bridge["status"] != "certified_conditional":
        return {
            "status": "failed",
            "candidate_A": candidate_a,
            "stage": "taylor_bridge",
            "taylor_bridge": bridge,
            "conditional": "taylor_remainder_is_inside_radius",
        }
    if abs(float(bridge["certified_until"]) - bridge_end_time) > 1e-12:
        return {
            "status": "failed",
            "candidate_A": candidate_a,
            "stage": "bridge_endpoint_mismatch",
            "bridge_end_time": bridge_end_time,
            "bridge_certified_until": bridge.get("certified_until"),
            "taylor_bridge": bridge,
            "conditional": "taylor_remainder_is_inside_radius",
        }

    beta = 1.0 / candidate_a
    time = bridge_end_time
    time_tolerance = max(1e-12, step_size * 1e-9)
    samples = tuple(tuple(sample) for sample in bridge["end_samples"])
    low = tuple(bridge["end_box"]["low"])
    high = tuple(bridge["end_box"]["high"])
    segments = []
    restarts = []
    total_attempts = 0
    retry_count = 0
    retry_log = []
    worst_margin = math.inf
    worst_face: dict | None = None
    last_block: dict | None = None

    while time < end_time - time_tolerance:
        segment_end = min(time + restart_interval, end_time)
        segment_steps = round((segment_end - time) / step_size)
        if segment_steps <= 0:
            time = segment_end
            break
        if abs(time + segment_steps * step_size - segment_end) > 1e-12:
            raise ValueError("each restart segment length must be an integer multiple of step_size")
        segment_start = time
        blocks_certified = 0
        segment_attempts = 0
        segment_worst = math.inf
        segment_worst_face: dict | None = None

        for _step_index in range(segment_steps):
            active_subdivisions = subdivisions
            tried_blocks = []
            block = None
            for attempt_subdivisions in (subdivisions, *retry_subdivisions):
                block = tuned_tube_block_certificate(
                    time,
                    step_size,
                    1,
                    candidate_a,
                    samples,
                    low,
                    high,
                    initial_growth=initial_growth,
                    max_growth=max_growth,
                    growth_factor=growth_factor,
                    max_attempts=max_attempts,
                    subdivisions=attempt_subdivisions,
                    time_subdivisions=time_subdivisions,
                )
                tried_blocks.append(block)
                active_subdivisions = attempt_subdivisions
                if block["status"] == "certified":
                    if attempt_subdivisions != subdivisions:
                        retry_count += 1
                        retry_log.append(
                            {
                                "time": time,
                                "subdivisions": list(attempt_subdivisions),
                                "worst_margin": block.get("worst_margin"),
                                "worst_face": block.get("worst_face"),
                            }
                        )
                    break
            assert block is not None
            attempts = sum(len(item.get("tuning_attempts", ())) for item in tried_blocks)
            total_attempts += attempts
            segment_attempts += attempts
            face = block.get("failing_face") or block.get("worst_face")
            if block.get("worst_margin", math.inf) < worst_margin:
                worst_margin = block["worst_margin"]
                worst_face = face
            if block.get("worst_margin", math.inf) < segment_worst:
                segment_worst = block["worst_margin"]
                segment_worst_face = face
            if block["status"] != "certified":
                segments.append(
                    {
                        "start_time": segment_start,
                        "target_time": segment_end,
                        "certified_until": time,
                        "blocks_certified": blocks_certified,
                        "tuning_attempts": segment_attempts,
                        "worst_margin": segment_worst,
                        "worst_face": segment_worst_face,
                        "status": "failed",
                    }
                )
                return {
                    "status": "failed",
                    "candidate_A": candidate_a,
                    "b_interval": [-beta, beta],
                    "bridge_end_time": bridge_end_time,
                    "end_time": end_time,
                    "restart_interval": restart_interval,
                    "step_size": step_size,
                    "subdivisions": list(subdivisions),
                    "time_subdivisions": time_subdivisions,
                    "initial_growth": list(initial_growth),
                    "max_growth": list(max_growth),
                    "growth_factor": growth_factor,
                    "max_attempts": max_attempts,
                    "restart_padding": restart_padding,
                    "certified_until": time,
                    "segments_certified": len(segments) - 1,
                    "blocks_certified": sum(segment["blocks_certified"] for segment in segments),
                    "tuning_attempt_count": total_attempts,
                    "retry_count": retry_count,
                    "retry_log": retry_log,
                    "taylor_bridge": bridge,
                    "segments": segments,
                    "restarts": restarts,
                    "current_start_box": {"low": list(low), "high": list(high)},
                    "current_samples": [list(sample) for sample in samples],
                    "current_width": [high[index] - low[index] for index in range(4)],
                    "failing_block": block,
                    "tried_blocks": [
                        {
                            "subdivisions": list((subdivisions, *retry_subdivisions)[index]),
                            "status": item["status"],
                            "worst_margin": item.get("worst_margin"),
                            "face": item.get("failing_face") or item.get("worst_face"),
                        }
                        for index, item in enumerate(tried_blocks)
                    ],
                    "last_certified_block": last_block,
                    "worst_margin": worst_margin,
                    "worst_face": worst_face,
                    "conditional": "taylor_remainder_is_inside_radius",
                }

            samples = tuple(tuple(sample) for sample in block["end_samples"])
            low = tuple(block["end_box"]["low"])
            high = tuple(block["end_box"]["high"])
            time += step_size
            blocks_certified += 1
            last_block = {key: value for key, value in block.items() if key != "end_samples"}
            last_block["subdivisions"] = list(active_subdivisions)

        segments.append(
            {
                "start_time": segment_start,
                "target_time": segment_end,
                "certified_until": time,
                "blocks_certified": blocks_certified,
                "tuning_attempts": segment_attempts,
                "worst_margin": segment_worst,
                "worst_face": segment_worst_face,
                "status": "certified",
            }
        )
        if (
            progress_callback is not None
            and progress_every_segments > 0
                    and (len(segments) % progress_every_segments == 0 or time >= end_time - time_tolerance)
        ):
            progress_callback(
                {
                    "event": "taylor_restart_chain_progress",
                    "certified_until": time,
                    "end_time": end_time,
                    "segments_certified": len(segments),
                    "blocks_certified": sum(segment["blocks_certified"] for segment in segments),
                    "tuning_attempt_count": total_attempts,
                    "retry_count": retry_count,
                    "worst_margin": worst_margin,
                    "current_width": [high[index] - low[index] for index in range(4)],
                }
            )
        if time < end_time - time_tolerance:
            restart = centered_restart_box_from_samples(
                time,
                samples,
                low,
                high,
                candidate_a=candidate_a,
                padding=restart_padding,
                sample_source="propagated_from_taylor_bridge",
            )
            if not restart["source_box_contained"]:
                return {
                    "status": "failed",
                    "candidate_A": candidate_a,
                    "stage": "restart_containment",
                    "certified_until": time,
                    "taylor_bridge": bridge,
                    "restart": restart,
                    "segments": segments,
                    "restarts": restarts,
                    "worst_margin": worst_margin,
                    "worst_face": worst_face,
                    "conditional": "taylor_remainder_is_inside_radius",
                }
            restarts.append(restart)
            samples = tuple(tuple(sample) for sample in restart["samples"])
            low = tuple(restart["box"]["low"])
            high = tuple(restart["box"]["high"])

    if abs(time - end_time) <= time_tolerance:
        time = end_time

    return {
        "status": "certified_conditional",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "bridge_end_time": bridge_end_time,
        "end_time": end_time,
        "restart_interval": restart_interval,
        "step_size": step_size,
        "subdivisions": list(subdivisions),
        "time_subdivisions": time_subdivisions,
        "initial_growth": list(initial_growth),
        "max_growth": list(max_growth),
        "growth_factor": growth_factor,
        "max_attempts": max_attempts,
        "restart_padding": restart_padding,
        "certified_until": time,
        "segments_certified": len(segments),
        "blocks_certified": sum(segment["blocks_certified"] for segment in segments),
        "tuning_attempt_count": total_attempts,
        "retry_count": retry_count,
        "retry_log": retry_log,
        "taylor_bridge": bridge,
        "end_box": {"low": list(low), "high": list(high)},
        "end_samples": [list(sample) for sample in samples],
        "end_width": [high[index] - low[index] for index in range(4)],
        "segments": segments,
        "restarts": restarts,
        "last_certified_block": last_block,
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "conditional": "taylor_remainder_is_inside_radius",
        "conclusion": "the true scaled trajectory is inside the end box if the Taylor remainder is inside the start radius",
    }


def _p_tube_block_certificate(
    start_p: float,
    step_size: float,
    block_steps: int,
    candidate_a: float,
    start_samples: tuple[tuple[float, float, float, float], ...],
    start_low: tuple[float, float, float, float],
    start_high: tuple[float, float, float, float],
    lower_radius_growth: tuple[float, float, float, float],
    upper_radius_growth: tuple[float, float, float, float],
    subdivisions: tuple[int, int, int, int],
    p_subdivisions: int,
    use_cancellation_p_prime: bool = False,
) -> dict:
    """Certify one p-time block from a supplied start box."""
    from mpmath import iv

    beta = 1.0 / candidate_a
    b_values = (-beta, 0.0, beta)
    samples = [[start_samples[index]] for index in range(3)]
    for step_index in range(block_steps):
        p = start_p + step_index * step_size
        for index, b in enumerate(b_values):
            samples[index].append(_rk4_step_p(p, samples[index][-1], step_size, b))

    radius0_low = []
    radius0_high = []
    for index in range(4):
        nominal_low = min(sample[0][index] for sample in samples)
        nominal_high = max(sample[0][index] for sample in samples)
        radius0_low.append(max(0.0, nominal_low - start_low[index]))
        radius0_high.append(max(0.0, start_high[index] - nominal_high))

    lows: list[list[float]] = []
    highs: list[list[float]] = []
    for step_index in range(block_steps + 1):
        distance = abs(step_index * step_size)
        step_samples = [sample[step_index] for sample in samples]
        lows.append(
            [
                min(sample[index] for sample in step_samples)
                - radius0_low[index]
                - lower_radius_growth[index] * distance
                for index in range(4)
            ]
        )
        highs.append(
            [
                max(sample[index] for sample in step_samples)
                + radius0_high[index]
                + upper_radius_growth[index] * distance
                for index in range(4)
            ]
        )

    b_interval = iv.mpf([-beta, beta])
    worst_margin = math.inf
    worst_face: dict | None = None
    for step_index in range(block_steps):
        p0 = start_p + step_index * step_size
        p1 = p0 + step_size
        p_interval = iv.mpf([min(p0, p1), max(p0, p1)])
        union_box = [
            iv.mpf(
                [
                    min(lows[step_index][index], lows[step_index + 1][index]),
                    max(highs[step_index][index], highs[step_index + 1][index]),
                ]
            )
            for index in range(4)
        ]
        for index in range(4):
            lower_box = list(union_box)
            lower_box[index] = iv.mpf(
                [
                    min(lows[step_index][index], lows[step_index + 1][index]),
                    max(lows[step_index][index], lows[step_index + 1][index]),
                ]
            )
            try:
                lower_rhs_low, lower_rhs_high = _subdivided_interval_p_time_rhs_component(
                    p_interval,
                    tuple(lower_box),
                    b_interval,
                    index,
                    subdivisions,
                    p_subdivisions,
                    use_cancellation_p_prime=use_cancellation_p_prime,
                )
            except ZeroDivisionError as exc:
                return {
                    "status": "failed",
                    "failure": str(exc),
                    "end_samples": [sample[-1] for sample in samples],
                    "start_box": {"low": list(start_low), "high": list(start_high)},
                    "end_box": {"low": lows[-1], "high": highs[-1]},
                    "worst_margin": -math.inf,
                    "failing_face": {
                        "side": "lower",
                        "step_index": step_index,
                        "component": index,
                        "p_interval": [p0, p1],
                    },
                    "lower_radius_growth": list(lower_radius_growth),
                    "upper_radius_growth": list(upper_radius_growth),
                }
            lower_slope = (lows[step_index + 1][index] - lows[step_index][index]) / step_size
            # The independent variable p decreases.  On a lower face the tube
            # is inward if G <= L', hence the upper RHS bound is used.
            margin = lower_slope - lower_rhs_high
            if margin < worst_margin:
                worst_margin = margin
                worst_face = {
                    "side": "lower",
                    "step_index": step_index,
                    "component": index,
                    "rhs_lower": lower_rhs_low,
                    "rhs_upper": lower_rhs_high,
                    "face_slope": lower_slope,
                    "p_interval": [p0, p1],
                }
            if margin < 0.0:
                return {
                    "status": "failed",
                    "end_samples": [sample[-1] for sample in samples],
                    "start_box": {"low": list(start_low), "high": list(start_high)},
                    "end_box": {"low": lows[-1], "high": highs[-1]},
                    "worst_margin": worst_margin,
                    "failing_face": worst_face,
                    "lower_radius_growth": list(lower_radius_growth),
                    "upper_radius_growth": list(upper_radius_growth),
                }

            upper_box = list(union_box)
            upper_box[index] = iv.mpf(
                [
                    min(highs[step_index][index], highs[step_index + 1][index]),
                    max(highs[step_index][index], highs[step_index + 1][index]),
                ]
            )
            try:
                upper_rhs_low, upper_rhs_high = _subdivided_interval_p_time_rhs_component(
                    p_interval,
                    tuple(upper_box),
                    b_interval,
                    index,
                    subdivisions,
                    p_subdivisions,
                    use_cancellation_p_prime=use_cancellation_p_prime,
                )
            except ZeroDivisionError as exc:
                return {
                    "status": "failed",
                    "failure": str(exc),
                    "end_samples": [sample[-1] for sample in samples],
                    "start_box": {"low": list(start_low), "high": list(start_high)},
                    "end_box": {"low": lows[-1], "high": highs[-1]},
                    "worst_margin": -math.inf,
                    "failing_face": {
                        "side": "upper",
                        "step_index": step_index,
                        "component": index,
                        "p_interval": [p0, p1],
                    },
                    "lower_radius_growth": list(lower_radius_growth),
                    "upper_radius_growth": list(upper_radius_growth),
                }
            upper_slope = (highs[step_index + 1][index] - highs[step_index][index]) / step_size
            # On an upper face with decreasing p the tube is inward if U' <= G.
            margin = upper_rhs_low - upper_slope
            if margin < worst_margin:
                worst_margin = margin
                worst_face = {
                    "side": "upper",
                    "step_index": step_index,
                    "component": index,
                    "rhs_lower": upper_rhs_low,
                    "rhs_upper": upper_rhs_high,
                    "face_slope": upper_slope,
                    "p_interval": [p0, p1],
                }
            if margin < 0.0:
                return {
                    "status": "failed",
                    "end_samples": [sample[-1] for sample in samples],
                    "start_box": {"low": list(start_low), "high": list(start_high)},
                    "end_box": {"low": lows[-1], "high": highs[-1]},
                    "worst_margin": worst_margin,
                    "failing_face": worst_face,
                    "lower_radius_growth": list(lower_radius_growth),
                    "upper_radius_growth": list(upper_radius_growth),
                }

    return {
        "status": "certified",
        "end_samples": [sample[-1] for sample in samples],
        "start_box": {"low": list(start_low), "high": list(start_high)},
        "end_box": {"low": lows[-1], "high": highs[-1]},
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "lower_radius_growth": list(lower_radius_growth),
        "upper_radius_growth": list(upper_radius_growth),
    }


def segmented_p_tube_certificate(
    start_p: float = DEFAULT_P_TUBE_START,
    end_p: float = DEFAULT_P_TUBE_END,
    entry_time: float = DEFAULT_P_TUBE_ENTRY_TIME,
    step_size: float = DEFAULT_P_TUBE_STEP,
    block_steps: int = 1,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius0: tuple[float, float, float, float] = (1e-5, 1e-4, 1e-6, 1e-5),
    profiles: tuple = DEFAULT_SEGMENTED_P_TUBE_PROFILES,
    subdivisions: tuple[int, int, int, int] = (1, 1, 1, 1),
    p_subdivisions: int = 1,
    seed_step_size: float = 1e-5,
) -> dict:
    """Certify a segmented p=x0 time tube for the late terminal tail.

    This is conditional on the start p-slice box containing the true state for
    every b=1/a in the requested interval.  It is intended to be composed with
    an earlier t-time tube certificate.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if start_p <= end_p:
        raise ValueError("start_p must be greater than end_p")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if block_steps <= 0:
        raise ValueError("block_steps must be positive")
    total_steps = round((start_p - end_p) / step_size)
    if abs(start_p - total_steps * step_size - end_p) > 1e-12:
        raise ValueError("p-tube length must be an integer multiple of step_size")

    beta = 1.0 / candidate_a
    p_step = -step_size
    start_samples = tuple(
        scaled_state_at_p(
            "limit" if b == 0.0 else "exact",
            start_p,
            None if b == 0.0 else 1.0 / b,
            entry_time=entry_time,
            step_size=seed_step_size,
        )
        for b in (-beta, 0.0, beta)
    )
    low = tuple(min(sample[index] for sample in start_samples) - radius0[index] for index in range(4))
    high = tuple(max(sample[index] for sample in start_samples) + radius0[index] for index in range(4))
    samples = start_samples
    certified_blocks = 0
    certified_to_p = start_p
    last_block: dict | None = None
    worst_margin = math.inf
    worst_block_face: dict | None = None

    while certified_blocks * block_steps < total_steps:
        remaining_steps = total_steps - certified_blocks * block_steps
        current_block_steps = min(block_steps, remaining_steps)
        block_start = start_p - certified_blocks * block_steps * step_size
        candidates = []
        failures = []
        for profile in profiles:
            lower_growth, upper_growth = _normalize_p_tube_profile(profile)
            block = _p_tube_block_certificate(
                block_start,
                p_step,
                current_block_steps,
                candidate_a,
                samples,
                low,
                high,
                lower_growth,
                upper_growth,
                subdivisions,
                p_subdivisions,
            )
            if block["status"] == "certified":
                end_low = block["end_box"]["low"]
                end_high = block["end_box"]["high"]
                width_sum = sum(end_high[index] - end_low[index] for index in range(4))
                candidates.append((width_sum, block))
            else:
                failures.append(block)
        if not candidates:
            best_failure = max(failures, key=lambda item: item["worst_margin"]) if failures else None
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "b_interval": [-beta, beta],
                "entry_time": entry_time,
                "start_p": start_p,
                "end_p": end_p,
                "step_size": step_size,
                "block_steps": block_steps,
                "subdivisions": list(subdivisions),
                "p_subdivisions": p_subdivisions,
                "certified_to_p": certified_to_p,
                "blocks_certified": certified_blocks,
                "current_start_box": {"low": list(low), "high": list(high)},
                "failing_block": best_failure,
                "last_certified_block": last_block,
                "conditional": "start_p_slice_box_contains_true_state",
            }
        _width, block = min(candidates, key=lambda item: item[0])
        if block["worst_margin"] < worst_margin:
            worst_margin = block["worst_margin"]
            worst_block_face = block.get("worst_face")
        last_block = {key: value for key, value in block.items() if key != "end_samples"}
        samples = tuple(tuple(sample) for sample in block["end_samples"])
        low = tuple(block["end_box"]["low"])
        high = tuple(block["end_box"]["high"])
        certified_blocks += 1
        certified_to_p = start_p - min(certified_blocks * block_steps, total_steps) * step_size

    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "entry_time": entry_time,
        "start_p": start_p,
        "end_p": end_p,
        "step_size": step_size,
        "block_steps": block_steps,
        "subdivisions": list(subdivisions),
        "p_subdivisions": p_subdivisions,
        "certified_to_p": certified_to_p,
        "blocks_certified": certified_blocks,
        "end_box": {"low": list(low), "high": list(high)},
        "worst_margin": worst_margin,
        "worst_face": worst_block_face,
        "last_certified_block": last_block,
        "conditional": "start_p_slice_box_contains_true_state",
    }


def tuned_p_tube_block_certificate(
    start_p: float,
    step_size: float,
    block_steps: int,
    candidate_a: float,
    start_samples: tuple[tuple[float, float, float, float], ...],
    start_low: tuple[float, float, float, float],
    start_high: tuple[float, float, float, float],
    initial_growth: tuple[float, float, float, float] = DEFAULT_TUNED_P_TUBE_INITIAL_GROWTH,
    max_growth: tuple[float, float, float, float] = DEFAULT_TUNED_P_TUBE_MAX_GROWTH,
    growth_factor: float = DEFAULT_TUNED_TUBE_GROWTH_FACTOR,
    max_attempts: int = DEFAULT_TUNED_P_TUBE_MAX_ATTEMPTS,
    subdivisions: tuple[int, int, int, int] = (1, 1, 1, 1),
    p_subdivisions: int = 1,
    use_cancellation_p_prime: bool = False,
) -> dict:
    """Tune one p-time tube block from failed face diagnostics."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if growth_factor <= 1.0:
        raise ValueError("growth_factor must be greater than 1")
    lower_growth = list(initial_growth)
    upper_growth = list(initial_growth)
    attempts = []
    best_failure: dict | None = None

    for attempt_index in range(max_attempts):
        block = _p_tube_block_certificate(
            start_p,
            step_size,
            block_steps,
            candidate_a,
            start_samples,
            start_low,
            start_high,
            tuple(lower_growth),
            tuple(upper_growth),
            subdivisions,
            p_subdivisions,
            use_cancellation_p_prime=use_cancellation_p_prime,
        )
        face = block.get("failing_face") or block.get("worst_face")
        attempts.append(
            {
                "attempt": attempt_index,
                "status": block["status"],
                "worst_margin": block["worst_margin"],
                "face": face,
                "lower_growth": list(lower_growth),
                "upper_growth": list(upper_growth),
            }
        )
        if block["status"] == "certified":
            block["tuning_attempts"] = attempts
            return block
        if best_failure is None or block["worst_margin"] > best_failure["worst_margin"]:
            best_failure = block
        if face is None:
            break
        component = int(face["component"])
        if face["side"] == "lower":
            new_value = _grow_profile_component(
                lower_growth[component],
                component,
                growth_factor,
                max_growth,
            )
            if new_value <= lower_growth[component]:
                break
            lower_growth[component] = new_value
        else:
            new_value = _grow_profile_component(
                upper_growth[component],
                component,
                growth_factor,
                max_growth,
            )
            if new_value <= upper_growth[component]:
                break
            upper_growth[component] = new_value

    failure = best_failure if best_failure is not None else block
    return {
        "status": "failed",
        "end_samples": failure.get("end_samples"),
        "start_box": {"low": list(start_low), "high": list(start_high)},
        "end_box": failure.get("end_box"),
        "worst_margin": failure.get("worst_margin", -math.inf),
        "failing_face": failure.get("failing_face"),
        "lower_radius_growth": failure.get("lower_radius_growth", list(lower_growth)),
        "upper_radius_growth": failure.get("upper_radius_growth", list(upper_growth)),
        "tuning_attempts": attempts,
    }


def tuned_segmented_p_tube_certificate(
    start_p: float = DEFAULT_P_TUBE_START,
    end_p: float = DEFAULT_P_TUBE_END,
    entry_time: float = DEFAULT_P_TUBE_ENTRY_TIME,
    step_size: float = DEFAULT_P_TUBE_STEP,
    block_steps: int = 1,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius0: tuple[float, float, float, float] = (1e-5, 1e-4, 1e-6, 1e-5),
    initial_growth: tuple[float, float, float, float] = DEFAULT_TUNED_P_TUBE_INITIAL_GROWTH,
    max_growth: tuple[float, float, float, float] = DEFAULT_TUNED_P_TUBE_MAX_GROWTH,
    growth_factor: float = DEFAULT_TUNED_TUBE_GROWTH_FACTOR,
    max_attempts: int = DEFAULT_TUNED_P_TUBE_MAX_ATTEMPTS,
    subdivisions: tuple[int, int, int, int] = (1, 1, 1, 1),
    p_subdivisions: int = 1,
    seed_step_size: float = 1e-5,
    use_cancellation_p_prime: bool = False,
) -> dict:
    """Certify a p=x0 tube with local automatic profile tuning."""
    if block_steps <= 0:
        raise ValueError("block_steps must be positive")
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if start_p <= end_p:
        raise ValueError("start_p must be greater than end_p")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    total_steps = round((start_p - end_p) / step_size)
    if abs(start_p - total_steps * step_size - end_p) > 1e-12:
        raise ValueError("p-tube length must be an integer multiple of step_size")

    beta = 1.0 / candidate_a
    p_step = -step_size
    start_samples = tuple(
        scaled_state_at_p(
            "limit" if b == 0.0 else "exact",
            start_p,
            None if b == 0.0 else 1.0 / b,
            entry_time=entry_time,
            step_size=seed_step_size,
        )
        for b in (-beta, 0.0, beta)
    )
    low = tuple(min(sample[index] for sample in start_samples) - radius0[index] for index in range(4))
    high = tuple(max(sample[index] for sample in start_samples) + radius0[index] for index in range(4))
    samples = start_samples
    certified_blocks = 0
    certified_to_p = start_p
    last_block: dict | None = None
    worst_margin = math.inf
    worst_face: dict | None = None
    tuning_attempt_count = 0

    while certified_blocks * block_steps < total_steps:
        remaining_steps = total_steps - certified_blocks * block_steps
        current_block_steps = min(block_steps, remaining_steps)
        block_start = start_p - certified_blocks * block_steps * step_size
        block = tuned_p_tube_block_certificate(
            block_start,
            p_step,
            current_block_steps,
            candidate_a,
            samples,
            low,
            high,
            initial_growth=initial_growth,
            max_growth=max_growth,
            growth_factor=growth_factor,
            max_attempts=max_attempts,
            subdivisions=subdivisions,
            p_subdivisions=p_subdivisions,
            use_cancellation_p_prime=use_cancellation_p_prime,
        )
        tuning_attempt_count += len(block.get("tuning_attempts", ()))
        if block.get("worst_margin", math.inf) < worst_margin:
            worst_margin = block["worst_margin"]
            worst_face = block.get("failing_face") or block.get("worst_face")
        if block["status"] != "certified":
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "b_interval": [-beta, beta],
                "entry_time": entry_time,
                "start_p": start_p,
                "end_p": end_p,
                "step_size": step_size,
                "block_steps": block_steps,
                "subdivisions": list(subdivisions),
                "p_subdivisions": p_subdivisions,
                "initial_growth": list(initial_growth),
                "max_growth": list(max_growth),
                "growth_factor": growth_factor,
                "max_attempts": max_attempts,
                "certified_to_p": certified_to_p,
                "blocks_certified": certified_blocks,
                "tuning_attempt_count": tuning_attempt_count,
                "current_start_box": {"low": list(low), "high": list(high)},
                "failing_block": block,
                "last_certified_block": last_block,
                "worst_margin": worst_margin,
                "worst_face": worst_face,
                "conditional": "start_p_slice_box_contains_true_state",
            }
        last_block = {key: value for key, value in block.items() if key != "end_samples"}
        samples = tuple(tuple(sample) for sample in block["end_samples"])
        low = tuple(block["end_box"]["low"])
        high = tuple(block["end_box"]["high"])
        certified_blocks += 1
        certified_to_p = start_p - min(certified_blocks * block_steps, total_steps) * step_size

    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "entry_time": entry_time,
        "start_p": start_p,
        "end_p": end_p,
        "step_size": step_size,
        "block_steps": block_steps,
        "subdivisions": list(subdivisions),
        "p_subdivisions": p_subdivisions,
        "initial_growth": list(initial_growth),
        "max_growth": list(max_growth),
        "growth_factor": growth_factor,
        "max_attempts": max_attempts,
        "certified_to_p": certified_to_p,
        "blocks_certified": certified_blocks,
        "tuning_attempt_count": tuning_attempt_count,
        "end_box": {"low": list(low), "high": list(high)},
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "last_certified_block": last_block,
        "conditional": "start_p_slice_box_contains_true_state",
    }


def _partition_box(
    box_low: tuple[float, float, float, float],
    box_high: tuple[float, float, float, float],
    splits: tuple[int, int, int, int],
) -> list[tuple[tuple[float, float, float, float], tuple[float, float, float, float]]]:
    """Return a finite componentwise partition of one rectangular box."""
    if any(value <= 0 for value in splits):
        raise ValueError("all split counts must be positive")
    if not all(box_low[index] <= box_high[index] for index in range(4)):
        raise ValueError("box_low must be componentwise <= box_high")
    intervals = []
    for lower, upper, count in zip(box_low, box_high, splits):
        width = (upper - lower) / count
        intervals.append(tuple((lower + index * width, lower + (index + 1) * width) for index in range(count)))
    boxes = []
    for indices in itertools.product(*(range(count) for count in splits)):
        low = tuple(intervals[component][index][0] for component, index in enumerate(indices))
        high = tuple(intervals[component][index][1] for component, index in enumerate(indices))
        boxes.append((low, high))
    return boxes


def _partition_box_nd(
    box_low: tuple[float, ...],
    box_high: tuple[float, ...],
    splits: tuple[int, ...],
) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    """Return a finite componentwise partition of an n-dimensional box."""
    if not (len(box_low) == len(box_high) == len(splits)):
        raise ValueError("box bounds and splits must have the same dimension")
    if any(value <= 0 for value in splits):
        raise ValueError("all split counts must be positive")
    if not all(box_low[index] <= box_high[index] for index in range(len(box_low))):
        raise ValueError("box_low must be componentwise <= box_high")
    intervals = []
    for lower, upper, count in zip(box_low, box_high, splits):
        width = (upper - lower) / count
        intervals.append(tuple((lower + index * width, lower + (index + 1) * width) for index in range(count)))
    boxes = []
    for indices in itertools.product(*(range(count) for count in splits)):
        low = tuple(intervals[component][index][0] for component, index in enumerate(indices))
        high = tuple(intervals[component][index][1] for component, index in enumerate(indices))
        boxes.append((low, high))
    return boxes


def tuned_p_tube_from_box_certificate(
    start_p: float,
    end_p: float,
    start_low: tuple[float, float, float, float],
    start_high: tuple[float, float, float, float],
    step_size: float = DEFAULT_P_TUBE_STEP,
    block_steps: int = 1,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    initial_growth: tuple[float, float, float, float] = DEFAULT_TUNED_P_TUBE_INITIAL_GROWTH,
    max_growth: tuple[float, float, float, float] = DEFAULT_TUNED_P_TUBE_MAX_GROWTH,
    growth_factor: float = DEFAULT_TUNED_TUBE_GROWTH_FACTOR,
    max_attempts: int = DEFAULT_TUNED_P_TUBE_MAX_ATTEMPTS,
    subdivisions: tuple[int, int, int, int] = (1, 1, 1, 1),
    p_subdivisions: int = 1,
    use_cancellation_p_prime: bool = False,
) -> dict:
    """Certify a p-time tube from an arbitrary start box.

    The centre curves start from the box midpoint.  They are not assumed to be
    the actual parameter-family trajectories; the proof obligation is instead
    that the supplied start box lies inside the tube and that all interval face
    inequalities hold for every ``b`` in ``[-1/A,1/A]``.
    """
    if start_p <= end_p:
        raise ValueError("start_p must be greater than end_p")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if block_steps <= 0:
        raise ValueError("block_steps must be positive")
    if not all(start_low[index] <= start_high[index] for index in range(4)):
        raise ValueError("start_low must be componentwise <= start_high")
    total_steps = round((start_p - end_p) / step_size)
    if abs(start_p - total_steps * step_size - end_p) > 1e-12:
        raise ValueError("p-tube length must be an integer multiple of step_size")

    beta = 1.0 / candidate_a
    p_step = -step_size
    midpoint = tuple((start_low[index] + start_high[index]) * 0.5 for index in range(4))
    samples = (midpoint, midpoint, midpoint)
    low = tuple(start_low)
    high = tuple(start_high)
    certified_blocks = 0
    certified_to_p = start_p
    last_block: dict | None = None
    worst_margin = math.inf
    worst_face: dict | None = None
    tuning_attempt_count = 0

    while certified_blocks * block_steps < total_steps:
        remaining_steps = total_steps - certified_blocks * block_steps
        current_block_steps = min(block_steps, remaining_steps)
        block_start = start_p - certified_blocks * block_steps * step_size
        block = tuned_p_tube_block_certificate(
            block_start,
            p_step,
            current_block_steps,
            candidate_a,
            samples,
            low,
            high,
            initial_growth=initial_growth,
            max_growth=max_growth,
            growth_factor=growth_factor,
            max_attempts=max_attempts,
            subdivisions=subdivisions,
            p_subdivisions=p_subdivisions,
            use_cancellation_p_prime=use_cancellation_p_prime,
        )
        tuning_attempt_count += len(block.get("tuning_attempts", ()))
        if block.get("worst_margin", math.inf) < worst_margin:
            worst_margin = block["worst_margin"]
            worst_face = block.get("failing_face") or block.get("worst_face")
        if block["status"] != "certified":
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "b_interval": [-beta, beta],
                "start_p": start_p,
                "end_p": end_p,
                "step_size": step_size,
                "block_steps": block_steps,
                "certified_to_p": certified_to_p,
                "blocks_certified": certified_blocks,
                "tuning_attempt_count": tuning_attempt_count,
                "use_cancellation_p_prime": use_cancellation_p_prime,
                "start_box": {"low": list(start_low), "high": list(start_high)},
                "current_start_box": {"low": list(low), "high": list(high)},
                "failing_block": block,
                "last_certified_block": last_block,
                "worst_margin": worst_margin,
                "worst_face": worst_face,
                "center_sample_source": "start_box_midpoint",
                "conditional": "start_box_contains_true_state",
            }
        last_block = {key: value for key, value in block.items() if key != "end_samples"}
        samples = tuple(tuple(sample) for sample in block["end_samples"])
        low = tuple(block["end_box"]["low"])
        high = tuple(block["end_box"]["high"])
        certified_blocks += 1
        certified_to_p = start_p - min(certified_blocks * block_steps, total_steps) * step_size

    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "start_p": start_p,
        "end_p": end_p,
        "step_size": step_size,
        "block_steps": block_steps,
        "certified_to_p": certified_to_p,
        "blocks_certified": certified_blocks,
        "tuning_attempt_count": tuning_attempt_count,
        "use_cancellation_p_prime": use_cancellation_p_prime,
        "start_box": {"low": list(start_low), "high": list(start_high)},
        "end_box": {"low": list(low), "high": list(high)},
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "last_certified_block": last_block,
        "center_sample_source": "start_box_midpoint",
        "conditional": "start_box_contains_true_state",
    }


def cancellation_c_bounds_for_p_slice_box(
    p: float,
    box_low: tuple[float, float, float, float],
    box_high: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Return interval bounds for ``C=x1*x2-p^2*x3/6`` on a p-slice box."""
    if not all(box_low[index] <= box_high[index] for index in range(4)):
        raise ValueError("box_low must be componentwise <= box_high")
    products = [
        x1 * x2
        for x1 in (box_low[1], box_high[1])
        for x2 in (box_low[2], box_high[2])
    ]
    correction = [p * p * x3 / 6.0 for x3 in (box_low[3], box_high[3])]
    return min(products) - max(correction), max(products) - min(correction)


def augment_p_slice_box_with_c(
    p: float,
    box_low: tuple[float, float, float, float],
    box_high: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float, float], tuple[float, float, float, float, float]]:
    """Add a rigorous algebraic C interval to a four-coordinate p-slice box."""
    c_low, c_high = cancellation_c_bounds_for_p_slice_box(p, box_low, box_high)
    return (*box_low, c_low), (*box_high, c_high)


def sharpen_carried_c_p_slice_box(
    p: float,
    box_low: tuple[float, float, float, float, float],
    box_high: tuple[float, float, float, float, float],
) -> tuple[tuple[float, float, float, float, float], tuple[float, float, float, float, float], bool]:
    """Intersect a carried C interval with the algebraic end-slice C interval."""
    algebraic_low, algebraic_high = cancellation_c_bounds_for_p_slice_box(p, box_low[:4], box_high[:4])
    sharpened_low = max(box_low[4], algebraic_low)
    sharpened_high = min(box_high[4], algebraic_high)
    if sharpened_low <= sharpened_high:
        return (*box_low[:4], sharpened_low), (*box_high[:4], sharpened_high), True
    return box_low, box_high, False


def _carried_c_p_tube_block_certificate(
    start_p: float,
    step_size: float,
    block_steps: int,
    candidate_a: float,
    start_samples: tuple[tuple[float, float, float, float, float], ...],
    start_low: tuple[float, float, float, float, float],
    start_high: tuple[float, float, float, float, float],
    lower_radius_growth: tuple[float, float, float, float, float],
    upper_radius_growth: tuple[float, float, float, float, float],
    subdivisions: tuple[int, int, int, int, int],
    p_subdivisions: int,
) -> dict:
    """Certify one augmented p-time block carrying ``C`` explicitly."""
    from mpmath import iv

    beta = 1.0 / candidate_a
    b_values = (-beta, 0.0, beta)
    samples = [[start_samples[index]] for index in range(3)]
    for step_index in range(block_steps):
        p = start_p + step_index * step_size
        for index, b in enumerate(b_values):
            samples[index].append(_rk4_step_carried_c_p(p, samples[index][-1], step_size, b))

    radius0_low = []
    radius0_high = []
    for index in range(5):
        nominal_low = min(sample[0][index] for sample in samples)
        nominal_high = max(sample[0][index] for sample in samples)
        radius0_low.append(max(0.0, nominal_low - start_low[index]))
        radius0_high.append(max(0.0, start_high[index] - nominal_high))

    lows: list[list[float]] = []
    highs: list[list[float]] = []
    for step_index in range(block_steps + 1):
        distance = abs(step_index * step_size)
        step_samples = [sample[step_index] for sample in samples]
        lows.append(
            [
                min(sample[index] for sample in step_samples)
                - radius0_low[index]
                - lower_radius_growth[index] * distance
                for index in range(5)
            ]
        )
        highs.append(
            [
                max(sample[index] for sample in step_samples)
                + radius0_high[index]
                + upper_radius_growth[index] * distance
                for index in range(5)
            ]
        )

    b_interval = iv.mpf([-beta, beta])
    worst_margin = math.inf
    worst_face: dict | None = None
    for step_index in range(block_steps):
        p0 = start_p + step_index * step_size
        p1 = p0 + step_size
        p_interval = iv.mpf([min(p0, p1), max(p0, p1)])
        union_box = [
            iv.mpf(
                [
                    min(lows[step_index][index], lows[step_index + 1][index]),
                    max(highs[step_index][index], highs[step_index + 1][index]),
                ]
            )
            for index in range(5)
        ]
        for index in range(5):
            lower_box = list(union_box)
            lower_box[index] = iv.mpf(
                [
                    min(lows[step_index][index], lows[step_index + 1][index]),
                    max(lows[step_index][index], lows[step_index + 1][index]),
                ]
            )
            try:
                lower_rhs_low, lower_rhs_high = _subdivided_interval_carried_c_p_time_rhs_component(
                    p_interval,
                    tuple(lower_box),
                    b_interval,
                    index,
                    subdivisions,
                    p_subdivisions,
                )
            except ZeroDivisionError as exc:
                return {
                    "status": "failed",
                    "failure": str(exc),
                    "end_samples": [sample[-1] for sample in samples],
                    "start_box_5d": {"low": list(start_low), "high": list(start_high)},
                    "end_box_5d": {"low": lows[-1], "high": highs[-1]},
                    "worst_margin": -math.inf,
                    "failing_face": {
                        "side": "lower",
                        "step_index": step_index,
                        "component": index,
                        "p_interval": [p0, p1],
                    },
                    "lower_radius_growth": list(lower_radius_growth),
                    "upper_radius_growth": list(upper_radius_growth),
                }
            lower_slope = (lows[step_index + 1][index] - lows[step_index][index]) / step_size
            margin = lower_slope - lower_rhs_high
            if margin < worst_margin:
                worst_margin = margin
                worst_face = {
                    "side": "lower",
                    "step_index": step_index,
                    "component": index,
                    "rhs_lower": lower_rhs_low,
                    "rhs_upper": lower_rhs_high,
                    "face_slope": lower_slope,
                    "p_interval": [p0, p1],
                }
            if margin < 0.0:
                return {
                    "status": "failed",
                    "end_samples": [sample[-1] for sample in samples],
                    "start_box_5d": {"low": list(start_low), "high": list(start_high)},
                    "end_box_5d": {"low": lows[-1], "high": highs[-1]},
                    "worst_margin": worst_margin,
                    "failing_face": worst_face,
                    "lower_radius_growth": list(lower_radius_growth),
                    "upper_radius_growth": list(upper_radius_growth),
                }

            upper_box = list(union_box)
            upper_box[index] = iv.mpf(
                [
                    min(highs[step_index][index], highs[step_index + 1][index]),
                    max(highs[step_index][index], highs[step_index + 1][index]),
                ]
            )
            try:
                upper_rhs_low, upper_rhs_high = _subdivided_interval_carried_c_p_time_rhs_component(
                    p_interval,
                    tuple(upper_box),
                    b_interval,
                    index,
                    subdivisions,
                    p_subdivisions,
                )
            except ZeroDivisionError as exc:
                return {
                    "status": "failed",
                    "failure": str(exc),
                    "end_samples": [sample[-1] for sample in samples],
                    "start_box_5d": {"low": list(start_low), "high": list(start_high)},
                    "end_box_5d": {"low": lows[-1], "high": highs[-1]},
                    "worst_margin": -math.inf,
                    "failing_face": {
                        "side": "upper",
                        "step_index": step_index,
                        "component": index,
                        "p_interval": [p0, p1],
                    },
                    "lower_radius_growth": list(lower_radius_growth),
                    "upper_radius_growth": list(upper_radius_growth),
                }
            upper_slope = (highs[step_index + 1][index] - highs[step_index][index]) / step_size
            margin = upper_rhs_low - upper_slope
            if margin < worst_margin:
                worst_margin = margin
                worst_face = {
                    "side": "upper",
                    "step_index": step_index,
                    "component": index,
                    "rhs_lower": upper_rhs_low,
                    "rhs_upper": upper_rhs_high,
                    "face_slope": upper_slope,
                    "p_interval": [p0, p1],
                }
            if margin < 0.0:
                return {
                    "status": "failed",
                    "end_samples": [sample[-1] for sample in samples],
                    "start_box_5d": {"low": list(start_low), "high": list(start_high)},
                    "end_box_5d": {"low": lows[-1], "high": highs[-1]},
                    "worst_margin": worst_margin,
                    "failing_face": worst_face,
                    "lower_radius_growth": list(lower_radius_growth),
                    "upper_radius_growth": list(upper_radius_growth),
                }

    end_p = start_p + block_steps * step_size
    end_low, end_high, c_handoff_sharpened = sharpen_carried_c_p_slice_box(
        end_p,
        tuple(lows[-1]),
        tuple(highs[-1]),
    )
    return {
        "status": "certified",
        "end_samples": [sample[-1] for sample in samples],
        "start_box_5d": {"low": list(start_low), "high": list(start_high)},
        "end_box_5d": {"low": list(end_low), "high": list(end_high)},
        "raw_end_box_5d": {"low": lows[-1], "high": highs[-1]},
        "c_handoff_sharpened": c_handoff_sharpened,
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "lower_radius_growth": list(lower_radius_growth),
        "upper_radius_growth": list(upper_radius_growth),
    }


def tuned_carried_c_p_tube_block_certificate(
    start_p: float,
    step_size: float,
    block_steps: int,
    candidate_a: float,
    start_samples: tuple[tuple[float, float, float, float, float], ...],
    start_low: tuple[float, float, float, float, float],
    start_high: tuple[float, float, float, float, float],
    initial_growth: tuple[float, float, float, float, float],
    max_growth: tuple[float, float, float, float, float],
    growth_factor: float = DEFAULT_TUNED_TUBE_GROWTH_FACTOR,
    max_attempts: int = DEFAULT_TUNED_P_TUBE_MAX_ATTEMPTS,
    subdivisions: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1),
    p_subdivisions: int = 1,
) -> dict:
    """Tune one augmented p-time block from failed face diagnostics."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if growth_factor <= 1.0:
        raise ValueError("growth_factor must be greater than 1")
    lower_growth = list(initial_growth)
    upper_growth = list(initial_growth)
    attempts = []
    best_failure: dict | None = None

    for attempt_index in range(max_attempts):
        block = _carried_c_p_tube_block_certificate(
            start_p,
            step_size,
            block_steps,
            candidate_a,
            start_samples,
            start_low,
            start_high,
            tuple(lower_growth),
            tuple(upper_growth),
            subdivisions,
            p_subdivisions,
        )
        face = block.get("failing_face") or block.get("worst_face")
        attempts.append(
            {
                "attempt": attempt_index,
                "status": block["status"],
                "worst_margin": block["worst_margin"],
                "face": face,
                "lower_growth": list(lower_growth),
                "upper_growth": list(upper_growth),
            }
        )
        if block["status"] == "certified":
            block["tuning_attempts"] = attempts
            return block
        if best_failure is None or block["worst_margin"] > best_failure["worst_margin"]:
            best_failure = block
        if face is None:
            break
        component = int(face["component"])
        if face["side"] == "lower":
            new_value = _grow_profile_component(lower_growth[component], component, growth_factor, max_growth)
            if new_value <= lower_growth[component]:
                break
            lower_growth[component] = new_value
        else:
            new_value = _grow_profile_component(upper_growth[component], component, growth_factor, max_growth)
            if new_value <= upper_growth[component]:
                break
            upper_growth[component] = new_value

    failure = best_failure if best_failure is not None else block
    return {
        "status": "failed",
        "end_samples": failure.get("end_samples"),
        "start_box_5d": {"low": list(start_low), "high": list(start_high)},
        "end_box_5d": failure.get("end_box_5d"),
        "worst_margin": failure.get("worst_margin", -math.inf),
        "failing_face": failure.get("failing_face"),
        "lower_radius_growth": failure.get("lower_radius_growth", list(lower_growth)),
        "upper_radius_growth": failure.get("upper_radius_growth", list(upper_growth)),
        "tuning_attempts": attempts,
    }


def tuned_carried_c_p_tube_from_box_certificate(
    start_p: float,
    end_p: float,
    start_low: tuple[float, ...],
    start_high: tuple[float, ...],
    step_size: float = DEFAULT_P_TUBE_STEP,
    block_steps: int = 1,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    initial_growth: tuple[float, float, float, float, float] = (0.05, 1.0, 0.01, 0.1, 0.1),
    max_growth: tuple[float, float, float, float, float] = (20.0, 200.0, 2.0, 50.0, 10.0),
    growth_factor: float = DEFAULT_TUNED_TUBE_GROWTH_FACTOR,
    max_attempts: int = DEFAULT_TUNED_P_TUBE_MAX_ATTEMPTS,
    subdivisions: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1),
    p_subdivisions: int = 1,
) -> dict:
    """Certify an augmented p-time tube from a four- or five-dimensional box."""
    if start_p <= end_p:
        raise ValueError("start_p must be greater than end_p")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if block_steps <= 0:
        raise ValueError("block_steps must be positive")
    if len(start_low) == 4 and len(start_high) == 4:
        low, high = augment_p_slice_box_with_c(start_p, tuple(start_low), tuple(start_high))
        c_source = "algebraic_start_box"
    elif len(start_low) == 5 and len(start_high) == 5:
        low = tuple(float(value) for value in start_low)
        high = tuple(float(value) for value in start_high)
        c_source = "carried_start_box"
    else:
        raise ValueError("start boxes must have either four or five coordinates")
    if not all(low[index] <= high[index] for index in range(5)):
        raise ValueError("start_low must be componentwise <= start_high")
    total_steps = round((start_p - end_p) / step_size)
    if abs(start_p - total_steps * step_size - end_p) > 1e-12:
        raise ValueError("p-tube length must be an integer multiple of step_size")

    beta = 1.0 / candidate_a
    p_step = -step_size
    midpoint4 = tuple((low[index] + high[index]) * 0.5 for index in range(4))
    midpoint = (*midpoint4, cancellation_c_value(start_p, midpoint4))
    samples = (midpoint, midpoint, midpoint)
    certified_blocks = 0
    certified_to_p = start_p
    last_block: dict | None = None
    worst_margin = math.inf
    worst_face: dict | None = None
    tuning_attempt_count = 0

    while certified_blocks * block_steps < total_steps:
        remaining_steps = total_steps - certified_blocks * block_steps
        current_block_steps = min(block_steps, remaining_steps)
        block_start = start_p - certified_blocks * block_steps * step_size
        block = tuned_carried_c_p_tube_block_certificate(
            block_start,
            p_step,
            current_block_steps,
            candidate_a,
            samples,
            low,
            high,
            initial_growth=initial_growth,
            max_growth=max_growth,
            growth_factor=growth_factor,
            max_attempts=max_attempts,
            subdivisions=subdivisions,
            p_subdivisions=p_subdivisions,
        )
        tuning_attempt_count += len(block.get("tuning_attempts", ()))
        if block.get("worst_margin", math.inf) < worst_margin:
            worst_margin = block["worst_margin"]
            worst_face = block.get("failing_face") or block.get("worst_face")
        if block["status"] != "certified":
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "b_interval": [-beta, beta],
                "start_p": start_p,
                "end_p": end_p,
                "step_size": step_size,
                "block_steps": block_steps,
                "certified_to_p": certified_to_p,
                "blocks_certified": certified_blocks,
                "tuning_attempt_count": tuning_attempt_count,
                "start_box_5d": {"low": list(low), "high": list(high)},
                "current_start_box_5d": {"low": list(low), "high": list(high)},
                "failing_block": block,
                "last_certified_block": last_block,
                "worst_margin": worst_margin,
                "worst_face": worst_face,
                "center_sample_source": "start_box_midpoint",
                "c_source": c_source,
                "conditional": "start_box_contains_true_state_and_C",
            }
        last_block = {key: value for key, value in block.items() if key != "end_samples"}
        samples = tuple(tuple(sample) for sample in block["end_samples"])
        low = tuple(block["end_box_5d"]["low"])
        high = tuple(block["end_box_5d"]["high"])
        certified_blocks += 1
        certified_to_p = start_p - min(certified_blocks * block_steps, total_steps) * step_size

    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "start_p": start_p,
        "end_p": end_p,
        "step_size": step_size,
        "block_steps": block_steps,
        "certified_to_p": certified_to_p,
        "blocks_certified": certified_blocks,
        "tuning_attempt_count": tuning_attempt_count,
        "start_box_5d": {"low": list(start_low), "high": list(start_high)},
        "end_box_5d": {"low": list(low), "high": list(high)},
        "end_box": {"low": list(low[:4]), "high": list(high[:4])},
        "c_interval": [low[4], high[4]],
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "last_certified_block": last_block,
        "center_sample_source": "start_box_midpoint",
        "c_source": c_source,
        "conditional": "start_box_contains_true_state_and_C",
    }


def sampled_carried_c_p_tube_certificate(
    start_p: float = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_START,
    end_p: float = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_END,
    entry_time: float = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_ENTRY_TIME,
    step_size: float = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_STEP,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius0: tuple[float, float, float, float, float] = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_RADIUS,
    profiles: tuple = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_PROFILES,
    max_attempts: int = 120,
    subdivisions: tuple[int, int, int, int, int] = DEFAULT_SAMPLED_CARRIED_C_P_TUBE_SUBDIVISIONS,
    p_subdivisions: int = 1,
    seed_step_size: float = 5e-5,
    progress_callback=None,
    progress_every: int = 0,
) -> dict:
    """Certify a narrow carried-C p-time tube around the sampled b-family.

    This is a proof-building diagnostic for the large-|a| exclusion.  It is
    conditional on the explicit start-p sample box containing the true scaled
    finite-|a| trajectories.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if start_p <= end_p:
        raise ValueError("start_p must be greater than end_p")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if len(radius0) != 5 or any(value < 0.0 for value in radius0):
        raise ValueError("radius0 must contain five nonnegative values")
    if not profiles:
        raise ValueError("at least one profile is required")
    total_steps = round((start_p - end_p) / step_size)
    if abs(start_p - total_steps * step_size - end_p) > 1e-12:
        raise ValueError("sampled carried-C p-tube length must be an integer multiple of step_size")

    beta = 1.0 / candidate_a
    samples4 = tuple(
        scaled_state_at_p(
            "limit" if b == 0.0 else "exact",
            start_p,
            None if b == 0.0 else 1.0 / b,
            entry_time=entry_time,
            step_size=seed_step_size,
        )
        for b in (-beta, 0.0, beta)
    )
    samples = tuple(
        (*sample, cancellation_c_value(start_p, sample))
        for sample in samples4
    )
    start_samples = samples
    low = tuple(min(sample[index] for sample in samples) - radius0[index] for index in range(5))
    high = tuple(max(sample[index] for sample in samples) + radius0[index] for index in range(5))
    p_value = start_p
    certified_blocks = 0
    total_attempts = 0
    worst_margin = math.inf
    worst_face: dict | None = None
    last_block: dict | None = None
    checkpoints = []

    while certified_blocks < total_steps:
        failures = []
        certified_block = None
        for initial_growth, max_growth in profiles:
            block = tuned_carried_c_p_tube_block_certificate(
                p_value,
                -step_size,
                1,
                candidate_a,
                samples,
                low,
                high,
                initial_growth=tuple(initial_growth),
                max_growth=tuple(max_growth),
                max_attempts=max_attempts,
                subdivisions=subdivisions,
                p_subdivisions=p_subdivisions,
            )
            if block["status"] == "certified":
                certified_block = block
                break
            failures.append(block)
        if certified_block is None:
            best_failure = max(failures, key=lambda item: item.get("worst_margin", -math.inf))
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "b_interval": [-beta, beta],
                "start_p": start_p,
                "end_p": end_p,
                "entry_time": entry_time,
                "step_size": step_size,
                "certified_to_p": p_value,
                "blocks_certified": certified_blocks,
                "tuning_attempt_count": total_attempts
                + sum(len(item.get("tuning_attempts", ())) for item in failures),
                "radius0": list(radius0),
                "profiles": [
                    {"initial_growth": list(initial), "max_growth": list(maximum)}
                    for initial, maximum in profiles
                ],
                "subdivisions": list(subdivisions),
                "p_subdivisions": p_subdivisions,
                "seed_step_size": seed_step_size,
                "start_samples_5d": [list(sample) for sample in start_samples],
                "current_box_5d": {"low": list(low), "high": list(high)},
                "failing_block": best_failure,
                "last_certified_block": last_block,
                "checkpoints": checkpoints,
                "conditional": "sampled_start_box_contains_true_state_and_C",
            }

        total_attempts += len(certified_block.get("tuning_attempts", ()))
        if certified_block.get("worst_margin", math.inf) < worst_margin:
            worst_margin = certified_block["worst_margin"]
            worst_face = certified_block.get("worst_face")
        samples = tuple(tuple(sample) for sample in certified_block["end_samples"])
        low = tuple(certified_block["end_box_5d"]["low"])
        high = tuple(certified_block["end_box_5d"]["high"])
        last_block = {key: value for key, value in certified_block.items() if key != "end_samples"}
        certified_blocks += 1
        p_value = start_p - certified_blocks * step_size
        if progress_every > 0 and certified_blocks % progress_every == 0:
            checkpoint = {
                "blocks_certified": certified_blocks,
                "certified_to_p": p_value,
                "worst_margin": worst_margin,
                "width_5d": [high[index] - low[index] for index in range(5)],
            }
            checkpoints.append(checkpoint)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "sampled_carried_c_p_tube_progress",
                        **checkpoint,
                    }
                )

    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "start_p": start_p,
        "end_p": end_p,
        "entry_time": entry_time,
        "step_size": step_size,
        "certified_to_p": end_p,
        "blocks_certified": certified_blocks,
        "tuning_attempt_count": total_attempts,
        "radius0": list(radius0),
        "profiles": [
            {"initial_growth": list(initial), "max_growth": list(maximum)}
            for initial, maximum in profiles
        ],
        "subdivisions": list(subdivisions),
        "p_subdivisions": p_subdivisions,
        "seed_step_size": seed_step_size,
        "start_samples_5d": [list(sample) for sample in start_samples],
        "end_samples_5d": [list(sample) for sample in samples],
        "end_box_5d": {"low": list(low), "high": list(high)},
        "end_box": {"low": list(low[:4]), "high": list(high[:4])},
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "last_certified_block": last_block,
        "checkpoints": checkpoints,
        "conditional": "sampled_start_box_contains_true_state_and_C",
    }


def staged_union_p_tube_certificate(
    start_p: float = DEFAULT_STAGED_UNION_P_TUBE_START,
    source_box_low: tuple[float, float, float, float] = DEFAULT_STAGED_UNION_P_TUBE_SOURCE_LOW,
    source_box_high: tuple[float, float, float, float] = DEFAULT_STAGED_UNION_P_TUBE_SOURCE_HIGH,
    stages: tuple[tuple[float, tuple[int, int, int, int]], ...] = DEFAULT_STAGED_UNION_P_TUBE_STAGES,
    step_size: float = 5e-4,
    block_steps: int = 1,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    initial_growth: tuple[float, float, float, float] = DEFAULT_TUNED_P_TUBE_INITIAL_GROWTH,
    max_growth: tuple[float, float, float, float] = (20.0, 200.0, 2.0, 50.0),
    growth_factor: float = DEFAULT_TUNED_TUBE_GROWTH_FACTOR,
    max_attempts: int = 120,
    subdivisions: tuple[int, int, int, int] = (1, 1, 1, 1),
    p_subdivisions: int = 1,
    keep_end_boxes: bool = True,
    use_cancellation_p_prime: bool = False,
) -> dict:
    """Certify a finite-union p-time continuation with staged box splits."""
    if not stages:
        raise ValueError("at least one staged union target is required")
    current_p = start_p
    current_boxes = [(tuple(source_box_low), tuple(source_box_high))]
    stage_reports = []
    total_attempts = 0
    total_blocks = 0
    total_leaf_boxes = 1
    worst_margin = math.inf
    worst_face: dict | None = None

    for stage_index, (target_p, splits) in enumerate(stages):
        if current_p <= target_p:
            raise ValueError("stage target p values must strictly decrease")
        next_boxes = []
        stage_attempts = 0
        stage_blocks = 0
        stage_worst = math.inf
        stage_worst_face: dict | None = None
        partition_count = 0
        for parent_index, (parent_low, parent_high) in enumerate(current_boxes):
            children = _partition_box(parent_low, parent_high, splits)
            partition_count += len(children)
            for child_index, (child_low, child_high) in enumerate(children):
                certificate = tuned_p_tube_from_box_certificate(
                    current_p,
                    target_p,
                    child_low,
                    child_high,
                    step_size=step_size,
                    block_steps=block_steps,
                    candidate_a=candidate_a,
                    initial_growth=initial_growth,
                    max_growth=max_growth,
                    growth_factor=growth_factor,
                    max_attempts=max_attempts,
                    subdivisions=subdivisions,
                    p_subdivisions=p_subdivisions,
                    use_cancellation_p_prime=use_cancellation_p_prime,
                )
                total_attempts += certificate.get("tuning_attempt_count", 0)
                total_blocks += certificate.get("blocks_certified", 0)
                stage_attempts += certificate.get("tuning_attempt_count", 0)
                stage_blocks += certificate.get("blocks_certified", 0)
                if certificate.get("worst_margin", math.inf) < worst_margin:
                    worst_margin = certificate["worst_margin"]
                    worst_face = certificate.get("worst_face")
                if certificate.get("worst_margin", math.inf) < stage_worst:
                    stage_worst = certificate["worst_margin"]
                    stage_worst_face = certificate.get("worst_face")
                if certificate["status"] != "certified":
                    return {
                        "status": "failed",
                        "candidate_A": candidate_a,
                        "start_p": start_p,
                        "certified_to_p": current_p,
                        "failed_target_p": target_p,
                        "stage_index": stage_index,
                        "parent_index": parent_index,
                        "child_index": child_index,
                        "splits": list(splits),
                        "source_box": {"low": list(source_box_low), "high": list(source_box_high)},
                        "stage_reports": stage_reports,
                        "failing_certificate": certificate,
                        "tuning_attempt_count": total_attempts,
                        "use_cancellation_p_prime": use_cancellation_p_prime,
                        "blocks_certified": total_blocks,
                        "leaf_boxes_certified": total_leaf_boxes,
                        "worst_margin": worst_margin,
                        "worst_face": worst_face,
                        "conditional": "source_box_contains_true_state",
                    }
                end_box = certificate["end_box"]
                next_boxes.append((tuple(end_box["low"]), tuple(end_box["high"])))
        total_leaf_boxes = len(next_boxes)
        stage_report = {
            "stage_index": stage_index,
            "start_p": current_p,
            "target_p": target_p,
            "splits": list(splits),
            "input_boxes": len(current_boxes),
            "output_boxes": len(next_boxes),
            "partition_count": partition_count,
            "blocks_certified": stage_blocks,
            "tuning_attempt_count": stage_attempts,
            "worst_margin": stage_worst,
            "worst_face": stage_worst_face,
            "status": "certified",
        }
        if keep_end_boxes:
            stage_report["end_boxes"] = [
                {"low": list(low), "high": list(high)}
                for low, high in next_boxes
            ]
        stage_reports.append(stage_report)
        current_p = target_p
        current_boxes = next_boxes

    end_low = tuple(min(box_low[index] for box_low, _box_high in current_boxes) for index in range(4))
    end_high = tuple(max(box_high[index] for _box_low, box_high in current_boxes) for index in range(4))
    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "start_p": start_p,
        "certified_to_p": current_p,
        "source_box": {"low": list(source_box_low), "high": list(source_box_high)},
        "end_hull": {"low": list(end_low), "high": list(end_high)},
        "stage_reports": stage_reports,
        "leaf_boxes": [
            {"low": list(low), "high": list(high)}
            for low, high in current_boxes
        ]
        if keep_end_boxes
        else None,
        "leaf_box_count": len(current_boxes),
        "blocks_certified": total_blocks,
        "tuning_attempt_count": total_attempts,
        "use_cancellation_p_prime": use_cancellation_p_prime,
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "conditional": "source_box_contains_true_state",
        "conclusion": "the source box is covered by a finite union of certified p-time boxes through the final stage",
    }


def _certificate_failing_face(certificate: dict) -> dict | None:
    """Return the most specific failing/worst face recorded in a certificate."""
    face = certificate.get("failing_face") or certificate.get("worst_face")
    if face is not None:
        return face
    block = certificate.get("failing_block") or certificate.get("last_certified_block")
    if isinstance(block, dict):
        face = block.get("failing_face") or block.get("worst_face")
        if face is not None:
            return face
    nested = certificate.get("failing_certificate")
    if isinstance(nested, dict):
        return _certificate_failing_face(nested)
    return None


def _load_union_leaf_boxes(path: Path) -> list[tuple[tuple[float, float, float, float], tuple[float, float, float, float]]]:
    """Load finite-union leaf boxes from a staged/adaptive certificate JSON."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "staged_union_p_tube_certificate" in payload:
        certificate = payload["staged_union_p_tube_certificate"]
    elif "adaptive_union_p_tube_certificate" in payload:
        certificate = payload["adaptive_union_p_tube_certificate"]
    else:
        certificate = payload
    if not isinstance(certificate, dict):
        raise ValueError("union source JSON must contain an object certificate")

    raw_boxes = certificate.get("leaf_boxes")
    if raw_boxes is None:
        stage_reports = certificate.get("stage_reports") or ()
        if stage_reports:
            raw_boxes = stage_reports[-1].get("end_boxes")
    if not raw_boxes:
        raise ValueError("union source JSON does not contain leaf_boxes; rerun with end boxes enabled")

    boxes = []
    for raw_box in raw_boxes:
        low = tuple(float(value) for value in raw_box["low"])
        high = tuple(float(value) for value in raw_box["high"])
        if len(low) != 4 or len(high) != 4:
            raise ValueError("each loaded leaf box must have four low/high coordinates")
        if not all(low[index] <= high[index] for index in range(4)):
            raise ValueError("loaded leaf box has low > high")
        boxes.append((low, high))
    return boxes


def _load_union_leaf_boxes_for_carried_c(
    path: Path,
    start_p: float,
) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    """Load carried-C leaf boxes, augmenting four-dimensional boxes if needed."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "adaptive_carried_c_union_p_tube_certificate" in payload:
        certificate = payload["adaptive_carried_c_union_p_tube_certificate"]
    elif "adaptive_union_p_tube_certificate" in payload:
        certificate = payload["adaptive_union_p_tube_certificate"]
    elif "staged_union_p_tube_certificate" in payload:
        certificate = payload["staged_union_p_tube_certificate"]
    else:
        certificate = payload
    if not isinstance(certificate, dict):
        raise ValueError("union source JSON must contain an object certificate")

    raw_boxes = certificate.get("leaf_boxes_5d")
    if raw_boxes is not None:
        boxes = []
        for raw_box in raw_boxes:
            low = tuple(float(value) for value in raw_box["low"])
            high = tuple(float(value) for value in raw_box["high"])
            if len(low) != 5 or len(high) != 5:
                raise ValueError("leaf_boxes_5d entries must have five low/high coordinates")
            if not all(low[index] <= high[index] for index in range(5)):
                raise ValueError("loaded carried-C leaf box has low > high")
            boxes.append((low, high))
        return boxes

    return [
        augment_p_slice_box_with_c(start_p, low, high)
        for low, high in _load_union_leaf_boxes(path)
    ]


def _box_hull(
    boxes: list[tuple[tuple[float, ...], tuple[float, ...]]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the componentwise hull of nonempty low/high boxes."""
    if not boxes:
        raise ValueError("cannot build a hull from no boxes")
    dimension = len(boxes[0][0])
    low = tuple(min(box[0][index] for box in boxes) for index in range(dimension))
    high = tuple(max(box[1][index] for box in boxes) for index in range(dimension))
    return low, high


def _load_carried_c_corridor_source_box(path: Path) -> tuple[tuple[float, ...], tuple[float, ...], float | None, str]:
    """Load one carried-C p-corridor source box from a saved certificate JSON."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    recognized_keys = (
        "sampled_carried_c_p_tube_certificate",
        "automatic_carried_c_p_corridor_certificate",
        "adaptive_carried_c_union_p_tube_certificate",
        "adaptive_union_p_tube_certificate",
        "staged_union_p_tube_certificate",
    )
    source_key = "bare"
    certificate = payload
    if isinstance(payload, dict):
        for key in recognized_keys:
            if key in payload:
                source_key = key
                certificate = payload[key]
                break
    if not isinstance(certificate, dict):
        raise ValueError("carried-C corridor source JSON must contain an object certificate")

    source_p = certificate.get("certified_to_p", certificate.get("end_p"))
    source_p = None if source_p is None else float(source_p)

    for box_key in ("end_box_5d", "end_hull_5d"):
        raw_box = certificate.get(box_key)
        if raw_box is not None:
            low = tuple(float(value) for value in raw_box["low"])
            high = tuple(float(value) for value in raw_box["high"])
            if len(low) != 5 or len(high) != 5:
                raise ValueError(f"{box_key} must have five low/high coordinates")
            if not all(low[index] <= high[index] for index in range(5)):
                raise ValueError(f"{box_key} has low > high")
            return low, high, source_p, f"{source_key}.{box_key}"

    raw_boxes = certificate.get("leaf_boxes_5d")
    if raw_boxes:
        boxes = []
        for raw_box in raw_boxes:
            low = tuple(float(value) for value in raw_box["low"])
            high = tuple(float(value) for value in raw_box["high"])
            if len(low) != 5 or len(high) != 5:
                raise ValueError("leaf_boxes_5d entries must have five low/high coordinates")
            if not all(low[index] <= high[index] for index in range(5)):
                raise ValueError("leaf_boxes_5d entry has low > high")
            boxes.append((low, high))
        low, high = _box_hull(boxes)
        return low, high, source_p, f"{source_key}.leaf_boxes_5d_hull"

    raw_box = certificate.get("end_box") or certificate.get("end_hull")
    if raw_box is not None:
        if source_p is None:
            raise ValueError("four-dimensional source boxes require certified_to_p or end_p")
        low4 = tuple(float(value) for value in raw_box["low"])
        high4 = tuple(float(value) for value in raw_box["high"])
        if len(low4) != 4 or len(high4) != 4:
            raise ValueError("four-dimensional source box must have four low/high coordinates")
        low, high = augment_p_slice_box_with_c(source_p, low4, high4)
        return low, high, source_p, f"{source_key}.augmented_4d_box"

    raise ValueError("source JSON does not contain end_box_5d, end_hull_5d, leaf_boxes_5d, or end_box")


def adaptive_union_p_tube_certificate(
    start_p: float = DEFAULT_ADAPTIVE_UNION_P_TUBE_START,
    end_p: float = DEFAULT_ADAPTIVE_UNION_P_TUBE_END,
    source_boxes: tuple[
        tuple[tuple[float, float, float, float], tuple[float, float, float, float]],
        ...
    ] = ((DEFAULT_STAGED_UNION_P_TUBE_SOURCE_LOW, DEFAULT_STAGED_UNION_P_TUBE_SOURCE_HIGH),),
    step_size: float = 5e-4,
    block_steps: int = 1,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    initial_growth: tuple[float, float, float, float] = DEFAULT_TUNED_P_TUBE_INITIAL_GROWTH,
    max_growth: tuple[float, float, float, float] = (20.0, 200.0, 2.0, 50.0),
    growth_factor: float = DEFAULT_TUNED_TUBE_GROWTH_FACTOR,
    max_attempts: int = 120,
    subdivisions: tuple[int, int, int, int] = (1, 1, 1, 1),
    p_subdivisions: int = 1,
    max_depth: int = DEFAULT_ADAPTIVE_UNION_MAX_DEPTH,
    max_leaf_boxes: int = DEFAULT_ADAPTIVE_UNION_MAX_LEAF_BOXES,
    max_processed_boxes: int | None = DEFAULT_ADAPTIVE_UNION_MAX_PROCESSED_BOXES,
    use_cancellation_p_prime: bool = False,
    progress_callback=None,
    progress_every: int = 0,
) -> dict:
    """Certify a finite-union p-time continuation with adaptive splitting.

    Each source box is tried as a p-time tube from ``start_p`` to ``end_p``.
    A failed box is split in the failed face component and retried, up to the
    requested depth/budget.  Component ``t`` failures also split ``x1`` and
    ``x2`` because the p-time denominator contains the cancellation term
    ``x1*x2-p^2*x3/6``.  Component ``x2`` failures split ``x1`` as well, for
    the same dependency reason.
    """
    if not source_boxes:
        raise ValueError("at least one source box is required")
    if start_p <= end_p:
        raise ValueError("start_p must be greater than end_p")
    if max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    if max_leaf_boxes <= 0:
        raise ValueError("max_leaf_boxes must be positive")
    if max_processed_boxes is not None and max_processed_boxes <= 0:
        raise ValueError("max_processed_boxes must be positive when supplied")

    queue = [
        {
            "source_index": source_index,
            "depth": 0,
            "low": tuple(low),
            "high": tuple(high),
            "split_history": [],
        }
        for source_index, (low, high) in enumerate(source_boxes)
    ]
    certified = []
    failed_leaves = []
    processed = 0
    total_attempts = 0
    total_blocks = 0
    split_count = 0
    worst_certified_margin = math.inf
    worst_certified_face: dict | None = None
    worst_failed_attempt_margin = math.inf
    worst_failed_attempt_face: dict | None = None
    stopped_reason = None

    while queue:
        if max_processed_boxes is not None and processed >= max_processed_boxes:
            stopped_reason = "max_processed_boxes_exceeded"
            break
        item = queue.pop(0)
        processed += 1
        low = item["low"]
        high = item["high"]
        certificate = tuned_p_tube_from_box_certificate(
            start_p,
            end_p,
            low,
            high,
            step_size=step_size,
            block_steps=block_steps,
            candidate_a=candidate_a,
            initial_growth=initial_growth,
            max_growth=max_growth,
            growth_factor=growth_factor,
            max_attempts=max_attempts,
            subdivisions=subdivisions,
            p_subdivisions=p_subdivisions,
            use_cancellation_p_prime=use_cancellation_p_prime,
        )
        total_attempts += certificate.get("tuning_attempt_count", 0)
        total_blocks += certificate.get("blocks_certified", 0)
        face = _certificate_failing_face(certificate)

        if certificate["status"] == "certified":
            if certificate.get("worst_margin", math.inf) < worst_certified_margin:
                worst_certified_margin = certificate["worst_margin"]
                worst_certified_face = certificate.get("worst_face") or face
            end_box = certificate["end_box"]
            certified.append(
                {
                    "source_index": item["source_index"],
                    "depth": item["depth"],
                    "split_history": item["split_history"],
                    "start_box": {"low": list(low), "high": list(high)},
                    "end_box": end_box,
                    "blocks_certified": certificate["blocks_certified"],
                    "tuning_attempt_count": certificate["tuning_attempt_count"],
                    "worst_margin": certificate["worst_margin"],
                    "worst_face": certificate.get("worst_face"),
                }
            )
            if (
                progress_callback is not None
                and progress_every > 0
                and processed % progress_every == 0
            ):
                progress_callback(
                    {
                        "event": "adaptive_union_p_tube_progress",
                        "processed": processed,
                        "certified_leaf_boxes": len(certified),
                        "queued_boxes": len(queue),
                        "split_count": split_count,
                        "tuning_attempt_count": total_attempts,
                        "worst_margin": worst_certified_margin,
                    }
                )
            continue

        if certificate.get("worst_margin", math.inf) < worst_failed_attempt_margin:
            worst_failed_attempt_margin = certificate["worst_margin"]
            worst_failed_attempt_face = face

        if item["depth"] >= max_depth:
            failed_leaves.append(
                {
                    "source_index": item["source_index"],
                    "depth": item["depth"],
                    "split_history": item["split_history"],
                    "start_box": {"low": list(low), "high": list(high)},
                    "certified_to_p": certificate.get("certified_to_p"),
                    "worst_margin": certificate.get("worst_margin"),
                    "failing_face": face,
                    "failing_certificate": certificate,
                }
            )
            continue

        component = 0 if face is None else int(face.get("component", 0))
        if component == 0:
            split_components = (0, 1, 2)
        elif component == 2:
            split_components = (1, 2)
        else:
            split_components = (component,)
        splits = tuple(2 if index in split_components else 1 for index in range(4))
        children = _partition_box(low, high, splits)
        if len(queue) + len(children) + len(certified) + len(failed_leaves) > max_leaf_boxes:
            failed_leaves.append(
                {
                    "source_index": item["source_index"],
                    "depth": item["depth"],
                    "split_history": item["split_history"],
                    "start_box": {"low": list(low), "high": list(high)},
                    "certified_to_p": certificate.get("certified_to_p"),
                    "worst_margin": certificate.get("worst_margin"),
                    "failing_face": face,
                    "failure": "max_leaf_boxes_exceeded",
                    "failing_certificate": certificate,
                }
            )
            continue
        split_count += 1
        for child_index, (child_low, child_high) in enumerate(children):
            queue.append(
                {
                    "source_index": item["source_index"],
                    "depth": item["depth"] + 1,
                    "low": child_low,
                    "high": child_high,
                    "split_history": [
                        *item["split_history"],
                        {
                            "component": component,
                            "split_components": list(split_components),
                            "child_index": child_index,
                            "parent_face": face,
                        },
                    ],
                }
            )
        if progress_callback is not None and progress_every > 0 and processed % progress_every == 0:
            progress_callback(
                {
                    "event": "adaptive_union_p_tube_progress",
                    "processed": processed,
                    "certified_leaf_boxes": len(certified),
                    "queued_boxes": len(queue),
                    "split_count": split_count,
                    "tuning_attempt_count": total_attempts,
                    "worst_margin": worst_certified_margin,
                }
            )

    status = "certified" if not failed_leaves and not queue and stopped_reason is None else "failed"
    end_boxes = [item["end_box"] for item in certified]
    end_hull = None
    if end_boxes:
        end_hull = {
            "low": [
                min(box["low"][index] for box in end_boxes)
                for index in range(4)
            ],
            "high": [
                max(box["high"][index] for box in end_boxes)
                for index in range(4)
            ],
        }
    if status == "certified":
        worst_margin = worst_certified_margin
        worst_face = worst_certified_face
    elif worst_certified_margin <= worst_failed_attempt_margin:
        worst_margin = worst_certified_margin
        worst_face = worst_certified_face
    else:
        worst_margin = worst_failed_attempt_margin
        worst_face = worst_failed_attempt_face
    return {
        "status": status,
        "candidate_A": candidate_a,
        "start_p": start_p,
        "end_p": end_p,
        "certified_to_p": end_p if status == "certified" else None,
        "step_size": step_size,
        "block_steps": block_steps,
        "subdivisions": list(subdivisions),
        "p_subdivisions": p_subdivisions,
        "initial_growth": list(initial_growth),
        "max_growth": list(max_growth),
        "growth_factor": growth_factor,
        "max_attempts": max_attempts,
        "max_depth": max_depth,
        "max_leaf_boxes": max_leaf_boxes,
        "max_processed_boxes": max_processed_boxes,
        "use_cancellation_p_prime": use_cancellation_p_prime,
        "source_box_count": len(source_boxes),
        "processed_boxes": processed,
        "remaining_queue_count": len(queue),
        "split_count": split_count,
        "certified_leaf_box_count": len(certified),
        "failed_leaf_box_count": len(failed_leaves) + len(queue),
        "blocks_certified": total_blocks,
        "tuning_attempt_count": total_attempts,
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "worst_failed_attempt_margin": worst_failed_attempt_margin,
        "worst_failed_attempt_face": worst_failed_attempt_face,
        "end_hull": end_hull,
        "leaf_boxes": end_boxes if status == "certified" else None,
        "certified_leaves": certified,
        "failed_leaves": failed_leaves,
        "remaining_queue_preview": [
            {
                "source_index": item["source_index"],
                "depth": item["depth"],
                "split_history": item["split_history"],
                "start_box": {"low": list(item["low"]), "high": list(item["high"])},
            }
            for item in queue[:8]
        ],
        "stopped_reason": stopped_reason,
        "conditional": "source_boxes_contain_true_state",
        "conclusion": "the source boxes are covered by a finite union of certified p-time boxes through end_p"
        if status == "certified"
        else "the requested finite split budget did not certify every source box",
    }


def adaptive_carried_c_union_p_tube_certificate(
    start_p: float,
    end_p: float,
    source_boxes: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...],
    step_size: float = 5e-4,
    block_steps: int = 1,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    initial_growth: tuple[float, float, float, float, float] = (0.05, 1.0, 0.01, 0.1, 0.1),
    max_growth: tuple[float, float, float, float, float] = (20.0, 200.0, 2.0, 50.0, 10.0),
    growth_factor: float = DEFAULT_TUNED_TUBE_GROWTH_FACTOR,
    max_attempts: int = 120,
    subdivisions: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1),
    p_subdivisions: int = 1,
    max_depth: int = DEFAULT_ADAPTIVE_UNION_MAX_DEPTH,
    max_leaf_boxes: int = DEFAULT_ADAPTIVE_UNION_MAX_LEAF_BOXES,
    max_processed_boxes: int | None = DEFAULT_ADAPTIVE_UNION_MAX_PROCESSED_BOXES,
    split_x3_on_x0_failure: bool = False,
    split_x3_on_x2_failure: bool = False,
    progress_callback=None,
    progress_every: int = 0,
) -> dict:
    """Certify a finite-union p-time continuation carrying C explicitly."""
    if not source_boxes:
        raise ValueError("at least one source box is required")
    if start_p <= end_p:
        raise ValueError("start_p must be greater than end_p")
    if max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    if max_leaf_boxes <= 0:
        raise ValueError("max_leaf_boxes must be positive")
    if max_processed_boxes is not None and max_processed_boxes <= 0:
        raise ValueError("max_processed_boxes must be positive when supplied")

    queue = []
    for source_index, (low, high) in enumerate(source_boxes):
        if len(low) == 4 and len(high) == 4:
            low, high = augment_p_slice_box_with_c(start_p, tuple(low), tuple(high))
        elif len(low) == 5 and len(high) == 5:
            low = tuple(float(value) for value in low)
            high = tuple(float(value) for value in high)
        else:
            raise ValueError("source boxes must have either four or five coordinates")
        queue.append(
            {
                "source_index": source_index,
                "depth": 0,
                "low": low,
                "high": high,
                "split_history": [],
            }
        )

    certified = []
    failed_leaves = []
    processed = 0
    total_attempts = 0
    total_blocks = 0
    split_count = 0
    worst_certified_margin = math.inf
    worst_certified_face: dict | None = None
    worst_failed_attempt_margin = math.inf
    worst_failed_attempt_face: dict | None = None
    stopped_reason = None

    while queue:
        if max_processed_boxes is not None and processed >= max_processed_boxes:
            stopped_reason = "max_processed_boxes_exceeded"
            break
        item = queue.pop(0)
        processed += 1
        low = item["low"]
        high = item["high"]
        certificate = tuned_carried_c_p_tube_from_box_certificate(
            start_p,
            end_p,
            low,
            high,
            step_size=step_size,
            block_steps=block_steps,
            candidate_a=candidate_a,
            initial_growth=initial_growth,
            max_growth=max_growth,
            growth_factor=growth_factor,
            max_attempts=max_attempts,
            subdivisions=subdivisions,
            p_subdivisions=p_subdivisions,
        )
        total_attempts += certificate.get("tuning_attempt_count", 0)
        total_blocks += certificate.get("blocks_certified", 0)
        face = _certificate_failing_face(certificate)

        if certificate["status"] == "certified":
            if certificate.get("worst_margin", math.inf) < worst_certified_margin:
                worst_certified_margin = certificate["worst_margin"]
                worst_certified_face = certificate.get("worst_face") or face
            end_box = certificate["end_box_5d"]
            certified.append(
                {
                    "source_index": item["source_index"],
                    "depth": item["depth"],
                    "split_history": item["split_history"],
                    "start_box_5d": {"low": list(low), "high": list(high)},
                    "end_box_5d": end_box,
                    "blocks_certified": certificate["blocks_certified"],
                    "tuning_attempt_count": certificate["tuning_attempt_count"],
                    "worst_margin": certificate["worst_margin"],
                    "worst_face": certificate.get("worst_face"),
                }
            )
            if progress_callback is not None and progress_every > 0 and processed % progress_every == 0:
                progress_callback(
                    {
                        "event": "adaptive_carried_c_union_p_tube_progress",
                        "processed": processed,
                        "certified_leaf_boxes": len(certified),
                        "queued_boxes": len(queue),
                        "split_count": split_count,
                        "tuning_attempt_count": total_attempts,
                        "worst_margin": worst_certified_margin,
                    }
                )
            continue

        if certificate.get("worst_margin", math.inf) < worst_failed_attempt_margin:
            worst_failed_attempt_margin = certificate["worst_margin"]
            worst_failed_attempt_face = face

        if item["depth"] >= max_depth:
            failed_leaves.append(
                {
                    "source_index": item["source_index"],
                    "depth": item["depth"],
                    "split_history": item["split_history"],
                    "start_box_5d": {"low": list(low), "high": list(high)},
                    "certified_to_p": certificate.get("certified_to_p"),
                    "worst_margin": certificate.get("worst_margin"),
                    "failing_face": face,
                    "failing_certificate": certificate,
                }
            )
            continue

        component = 0 if face is None else int(face.get("component", 0))
        if component == 0:
            split_components = (0, 1, 2, 3, 4) if split_x3_on_x0_failure else (0, 1, 2, 4)
        elif component == 2:
            split_components = (1, 2, 3, 4) if split_x3_on_x2_failure else (1, 2, 4)
        elif component == 4:
            split_components = (1, 2, 3, 4)
        else:
            split_components = (component,)
        splits = tuple(2 if index in split_components else 1 for index in range(5))
        children = _partition_box_nd(low, high, splits)
        if len(queue) + len(children) + len(certified) + len(failed_leaves) > max_leaf_boxes:
            failed_leaves.append(
                {
                    "source_index": item["source_index"],
                    "depth": item["depth"],
                    "split_history": item["split_history"],
                    "start_box_5d": {"low": list(low), "high": list(high)},
                    "certified_to_p": certificate.get("certified_to_p"),
                    "worst_margin": certificate.get("worst_margin"),
                    "failing_face": face,
                    "failure": "max_leaf_boxes_exceeded",
                    "failing_certificate": certificate,
                }
            )
            continue
        split_count += 1
        for child_index, (child_low, child_high) in enumerate(children):
            queue.append(
                {
                    "source_index": item["source_index"],
                    "depth": item["depth"] + 1,
                    "low": child_low,
                    "high": child_high,
                    "split_history": [
                        *item["split_history"],
                        {
                            "component": component,
                            "split_components": list(split_components),
                            "child_index": child_index,
                            "parent_face": face,
                        },
                    ],
                }
            )
        if progress_callback is not None and progress_every > 0 and processed % progress_every == 0:
            progress_callback(
                {
                    "event": "adaptive_carried_c_union_p_tube_progress",
                    "processed": processed,
                    "certified_leaf_boxes": len(certified),
                    "queued_boxes": len(queue),
                    "split_count": split_count,
                    "tuning_attempt_count": total_attempts,
                    "worst_margin": worst_certified_margin,
                }
            )

    status = "certified" if not failed_leaves and not queue and stopped_reason is None else "failed"
    end_boxes = [item["end_box_5d"] for item in certified]
    end_hull = None
    if end_boxes:
        end_hull = {
            "low": [min(box["low"][index] for box in end_boxes) for index in range(5)],
            "high": [max(box["high"][index] for box in end_boxes) for index in range(5)],
        }
    if status == "certified":
        worst_margin = worst_certified_margin
        worst_face = worst_certified_face
    elif worst_certified_margin <= worst_failed_attempt_margin:
        worst_margin = worst_certified_margin
        worst_face = worst_certified_face
    else:
        worst_margin = worst_failed_attempt_margin
        worst_face = worst_failed_attempt_face
    return {
        "status": status,
        "candidate_A": candidate_a,
        "start_p": start_p,
        "end_p": end_p,
        "certified_to_p": end_p if status == "certified" else None,
        "step_size": step_size,
        "block_steps": block_steps,
        "subdivisions": list(subdivisions),
        "p_subdivisions": p_subdivisions,
        "initial_growth": list(initial_growth),
        "max_growth": list(max_growth),
        "growth_factor": growth_factor,
        "max_attempts": max_attempts,
        "max_depth": max_depth,
        "max_leaf_boxes": max_leaf_boxes,
        "max_processed_boxes": max_processed_boxes,
        "split_x3_on_x0_failure": split_x3_on_x0_failure,
        "split_x3_on_x2_failure": split_x3_on_x2_failure,
        "source_box_count": len(source_boxes),
        "processed_boxes": processed,
        "remaining_queue_count": len(queue),
        "split_count": split_count,
        "certified_leaf_box_count": len(certified),
        "failed_leaf_box_count": len(failed_leaves) + len(queue),
        "blocks_certified": total_blocks,
        "tuning_attempt_count": total_attempts,
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "worst_failed_attempt_margin": worst_failed_attempt_margin,
        "worst_failed_attempt_face": worst_failed_attempt_face,
        "end_hull_5d": end_hull,
        "end_hull": {
            "low": end_hull["low"][:4],
            "high": end_hull["high"][:4],
        }
        if end_hull is not None
        else None,
        "leaf_boxes_5d": end_boxes if status == "certified" else None,
        "leaf_boxes": [
            {"low": box["low"][:4], "high": box["high"][:4]}
            for box in end_boxes
        ]
        if status == "certified"
        else None,
        "certified_leaves": certified,
        "failed_leaves": failed_leaves,
        "remaining_queue_preview": [
            {
                "source_index": item["source_index"],
                "depth": item["depth"],
                "split_history": item["split_history"],
                "start_box_5d": {"low": list(item["low"]), "high": list(item["high"])},
            }
            for item in queue[:8]
        ],
        "stopped_reason": stopped_reason,
        "conditional": "source_boxes_contain_true_state_and_C",
        "conclusion": "the source boxes are covered by a finite union of certified carried-C p-time boxes through end_p"
        if status == "certified"
        else "the requested finite split budget did not certify every source box",
    }


def affine_p_corridor_certificate(
    start_p: float = DEFAULT_P_CORRIDOR_START,
    end_p: float = DEFAULT_P_CORRIDOR_END,
    step_size: float = DEFAULT_P_CORRIDOR_STEP,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    lower_start: tuple[float, float, float, float] = DEFAULT_P_CORRIDOR_LOWER_START,
    upper_start: tuple[float, float, float, float] = DEFAULT_P_CORRIDOR_UPPER_START,
    lower_slope: tuple[float, float, float, float] = DEFAULT_P_CORRIDOR_LOWER_SLOPE,
    upper_slope: tuple[float, float, float, float] = DEFAULT_P_CORRIDOR_UPPER_SLOPE,
    subdivisions: tuple[int, int, int, int] = (2, 2, 2, 2),
    p_subdivisions: int = 2,
    source_box_low: tuple[float, float, float, float] = DEFAULT_P_TUBE_FRONTIER_LOW,
    source_box_high: tuple[float, float, float, float] = DEFAULT_P_TUBE_FRONTIER_HIGH,
) -> dict:
    """Check an affine barrier corridor in p=x0 time.

    The corridor is conditional on a previously certified start-p box.  Since p
    decreases, a lower face is inward when G_j <= L_j', while an upper face is
    inward when U_j' <= G_j.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if start_p <= end_p:
        raise ValueError("start_p must be greater than end_p")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    steps = round((start_p - end_p) / step_size)
    if abs(start_p - steps * step_size - end_p) > 1e-12:
        raise ValueError("corridor length must be an integer multiple of step_size")
    if not all(lower_start[index] <= upper_start[index] for index in range(4)):
        raise ValueError("lower_start must be componentwise <= upper_start")

    from mpmath import iv

    beta = 1.0 / candidate_a
    b_interval = iv.mpf([-beta, beta])
    worst_margin = math.inf
    worst_face: dict | None = None
    for step_index in range(steps):
        p0 = start_p - step_index * step_size
        p1 = p0 - step_size
        p_interval = iv.mpf([p1, p0])
        lower0 = _affine_barrier_value(lower_start, lower_slope, start_p, p0)
        lower1 = _affine_barrier_value(lower_start, lower_slope, start_p, p1)
        upper0 = _affine_barrier_value(upper_start, upper_slope, start_p, p0)
        upper1 = _affine_barrier_value(upper_start, upper_slope, start_p, p1)
        slab_low = [min(lower0[index], lower1[index]) for index in range(4)]
        slab_high = [max(upper0[index], upper1[index]) for index in range(4)]
        if any(slab_low[index] > slab_high[index] for index in range(4)):
            return {
                "status": "failed",
                "failure": "empty_corridor_slab",
                "candidate_A": candidate_a,
                "start_p": start_p,
                "end_p": end_p,
                "certified_to_p": p0,
                "steps_certified": step_index,
                "slab_index": step_index,
                "slab_low": slab_low,
                "slab_high": slab_high,
            }
        union_box = [iv.mpf([slab_low[index], slab_high[index]]) for index in range(4)]
        for index in range(4):
            lower_box = list(union_box)
            lower_face_values = [lower0[index], lower1[index]]
            lower_box[index] = iv.mpf([min(lower_face_values), max(lower_face_values)])
            try:
                lower_rhs_low, lower_rhs_high = _subdivided_interval_p_time_rhs_component(
                    p_interval,
                    tuple(lower_box),
                    b_interval,
                    index,
                    subdivisions,
                    p_subdivisions,
                )
            except ZeroDivisionError as exc:
                return {
                    "status": "failed",
                    "failure": str(exc),
                    "candidate_A": candidate_a,
                    "start_p": start_p,
                    "end_p": end_p,
                    "certified_to_p": p0,
                    "steps_certified": step_index,
                    "failing_face": {
                        "side": "lower",
                        "component": index,
                        "p_interval": [p1, p0],
                    },
                }
            margin = lower_slope[index] - lower_rhs_high
            if margin < worst_margin:
                worst_margin = margin
                worst_face = {
                    "side": "lower",
                    "component": index,
                    "p_interval": [p1, p0],
                    "rhs_lower": lower_rhs_low,
                    "rhs_upper": lower_rhs_high,
                    "face_slope": lower_slope[index],
                    "slab_low": slab_low,
                    "slab_high": slab_high,
                }
            if margin < 0.0:
                return {
                    "status": "failed",
                    "candidate_A": candidate_a,
                    "b_interval": [-beta, beta],
                    "start_p": start_p,
                    "end_p": end_p,
                    "step_size": step_size,
                    "steps": steps,
                    "certified_to_p": p0,
                    "steps_certified": step_index,
                    "lower_start": list(lower_start),
                    "upper_start": list(upper_start),
                    "lower_slope": list(lower_slope),
                    "upper_slope": list(upper_slope),
                    "subdivisions": list(subdivisions),
                    "p_subdivisions": p_subdivisions,
                    "source_box_contained": _corridor_contains_box(
                        lower_start,
                        upper_start,
                        source_box_low,
                        source_box_high,
                    ),
                    "source_box": {"low": list(source_box_low), "high": list(source_box_high)},
                    "worst_margin": worst_margin,
                    "failing_face": worst_face,
                    "conditional": "source_start_box_contains_true_state",
                }

            upper_box = list(union_box)
            upper_face_values = [upper0[index], upper1[index]]
            upper_box[index] = iv.mpf([min(upper_face_values), max(upper_face_values)])
            try:
                upper_rhs_low, upper_rhs_high = _subdivided_interval_p_time_rhs_component(
                    p_interval,
                    tuple(upper_box),
                    b_interval,
                    index,
                    subdivisions,
                    p_subdivisions,
                )
            except ZeroDivisionError as exc:
                return {
                    "status": "failed",
                    "failure": str(exc),
                    "candidate_A": candidate_a,
                    "start_p": start_p,
                    "end_p": end_p,
                    "certified_to_p": p0,
                    "steps_certified": step_index,
                    "failing_face": {
                        "side": "upper",
                        "component": index,
                        "p_interval": [p1, p0],
                    },
                }
            margin = upper_rhs_low - upper_slope[index]
            if margin < worst_margin:
                worst_margin = margin
                worst_face = {
                    "side": "upper",
                    "component": index,
                    "p_interval": [p1, p0],
                    "rhs_lower": upper_rhs_low,
                    "rhs_upper": upper_rhs_high,
                    "face_slope": upper_slope[index],
                    "slab_low": slab_low,
                    "slab_high": slab_high,
                }
            if margin < 0.0:
                return {
                    "status": "failed",
                    "candidate_A": candidate_a,
                    "b_interval": [-beta, beta],
                    "start_p": start_p,
                    "end_p": end_p,
                    "step_size": step_size,
                    "steps": steps,
                    "certified_to_p": p0,
                    "steps_certified": step_index,
                    "lower_start": list(lower_start),
                    "upper_start": list(upper_start),
                    "lower_slope": list(lower_slope),
                    "upper_slope": list(upper_slope),
                    "subdivisions": list(subdivisions),
                    "p_subdivisions": p_subdivisions,
                    "source_box_contained": _corridor_contains_box(
                        lower_start,
                        upper_start,
                        source_box_low,
                        source_box_high,
                    ),
                    "source_box": {"low": list(source_box_low), "high": list(source_box_high)},
                    "worst_margin": worst_margin,
                    "failing_face": worst_face,
                    "conditional": "source_start_box_contains_true_state",
                }

    end_lower = _affine_barrier_value(lower_start, lower_slope, start_p, end_p)
    end_upper = _affine_barrier_value(upper_start, upper_slope, start_p, end_p)
    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "start_p": start_p,
        "end_p": end_p,
        "step_size": step_size,
        "steps": steps,
        "certified_to_p": end_p,
        "steps_certified": steps,
        "lower_start": list(lower_start),
        "upper_start": list(upper_start),
        "lower_slope": list(lower_slope),
        "upper_slope": list(upper_slope),
        "subdivisions": list(subdivisions),
        "p_subdivisions": p_subdivisions,
        "source_box_contained": _corridor_contains_box(
            lower_start,
            upper_start,
            source_box_low,
            source_box_high,
        ),
        "source_box": {"low": list(source_box_low), "high": list(source_box_high)},
        "end_box": {"low": list(end_lower), "high": list(end_upper)},
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "conditional": "source_start_box_contains_true_state",
    }


def affine_carried_c_p_corridor_certificate(
    start_p: float,
    end_p: float,
    lower_start: tuple[float, ...],
    upper_start: tuple[float, ...],
    lower_slope: tuple[float, ...],
    upper_slope: tuple[float, ...],
    step_size: float = DEFAULT_CARRIED_C_P_CORRIDOR_STEP,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    subdivisions: tuple[int, int, int, int, int] = DEFAULT_CARRIED_C_P_CORRIDOR_SUBDIVISIONS,
    p_subdivisions: int = DEFAULT_CARRIED_C_P_CORRIDOR_P_SUBDIVISIONS,
    source_box_low: tuple[float, ...] | None = None,
    source_box_high: tuple[float, ...] | None = None,
) -> dict:
    """Check one affine p-time barrier corridor carrying C as a fifth state."""
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if start_p <= end_p:
        raise ValueError("start_p must be greater than end_p")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if len(lower_start) != 5 or len(upper_start) != 5:
        raise ValueError("carried-C corridor start bounds must have five entries")
    if len(lower_slope) != 5 or len(upper_slope) != 5:
        raise ValueError("carried-C corridor slopes must have five entries")
    if len(subdivisions) != 5:
        raise ValueError("carried-C corridor subdivisions must have five entries")
    steps = round((start_p - end_p) / step_size)
    if abs(start_p - steps * step_size - end_p) > 1e-12:
        raise ValueError("carried-C corridor length must be an integer multiple of step_size")
    lower_start = tuple(float(value) for value in lower_start)
    upper_start = tuple(float(value) for value in upper_start)
    lower_slope = tuple(float(value) for value in lower_slope)
    upper_slope = tuple(float(value) for value in upper_slope)
    if not all(lower_start[index] <= upper_start[index] for index in range(5)):
        raise ValueError("lower_start must be componentwise <= upper_start")
    if source_box_low is None:
        source_box_low = lower_start
    if source_box_high is None:
        source_box_high = upper_start
    source_box_low = tuple(float(value) for value in source_box_low)
    source_box_high = tuple(float(value) for value in source_box_high)
    if len(source_box_low) != 5 or len(source_box_high) != 5:
        raise ValueError("source box must have five entries")

    from mpmath import iv

    beta = 1.0 / candidate_a
    b_interval = iv.mpf([-beta, beta])
    worst_margin = math.inf
    worst_face: dict | None = None
    for step_index in range(steps):
        p0 = start_p - step_index * step_size
        p1 = p0 - step_size
        p_interval = iv.mpf([p1, p0])
        lower0 = _affine_barrier_value_nd(lower_start, lower_slope, start_p, p0)
        lower1 = _affine_barrier_value_nd(lower_start, lower_slope, start_p, p1)
        upper0 = _affine_barrier_value_nd(upper_start, upper_slope, start_p, p0)
        upper1 = _affine_barrier_value_nd(upper_start, upper_slope, start_p, p1)
        slab_low = [min(lower0[index], lower1[index]) for index in range(5)]
        slab_high = [max(upper0[index], upper1[index]) for index in range(5)]
        if any(slab_low[index] > slab_high[index] for index in range(5)):
            return {
                "status": "failed",
                "failure": "empty_corridor_slab",
                "candidate_A": candidate_a,
                "start_p": start_p,
                "end_p": end_p,
                "certified_to_p": p0,
                "steps_certified": step_index,
                "slab_index": step_index,
                "slab_low": slab_low,
                "slab_high": slab_high,
                "conditional": "source_start_box_contains_true_state_and_C",
            }

        union_box = [iv.mpf([slab_low[index], slab_high[index]]) for index in range(5)]
        for index in range(5):
            lower_box = list(union_box)
            lower_face_values = [lower0[index], lower1[index]]
            lower_box[index] = iv.mpf([min(lower_face_values), max(lower_face_values)])
            try:
                lower_rhs_low, lower_rhs_high = _subdivided_interval_carried_c_p_time_rhs_component(
                    p_interval,
                    tuple(lower_box),
                    b_interval,
                    index,
                    subdivisions,
                    p_subdivisions,
                )
            except ZeroDivisionError as exc:
                return {
                    "status": "failed",
                    "failure": str(exc),
                    "candidate_A": candidate_a,
                    "start_p": start_p,
                    "end_p": end_p,
                    "certified_to_p": p0,
                    "steps_certified": step_index,
                    "failing_face": {
                        "side": "lower",
                        "component": index,
                        "p_interval": [p1, p0],
                    },
                    "conditional": "source_start_box_contains_true_state_and_C",
                }
            margin = lower_slope[index] - lower_rhs_high
            if margin < worst_margin:
                worst_margin = margin
                worst_face = {
                    "side": "lower",
                    "component": index,
                    "p_interval": [p1, p0],
                    "rhs_lower": lower_rhs_low,
                    "rhs_upper": lower_rhs_high,
                    "face_slope": lower_slope[index],
                    "slab_low": slab_low,
                    "slab_high": slab_high,
                }
            if margin < 0.0:
                return {
                    "status": "failed",
                    "candidate_A": candidate_a,
                    "b_interval": [-beta, beta],
                    "start_p": start_p,
                    "end_p": end_p,
                    "step_size": step_size,
                    "steps": steps,
                    "certified_to_p": p0,
                    "steps_certified": step_index,
                    "lower_start": list(lower_start),
                    "upper_start": list(upper_start),
                    "lower_slope": list(lower_slope),
                    "upper_slope": list(upper_slope),
                    "subdivisions": list(subdivisions),
                    "p_subdivisions": p_subdivisions,
                    "source_box_contained": _corridor_contains_box_nd(
                        lower_start,
                        upper_start,
                        source_box_low,
                        source_box_high,
                    ),
                    "source_box_5d": {"low": list(source_box_low), "high": list(source_box_high)},
                    "worst_margin": worst_margin,
                    "failing_face": worst_face,
                    "conditional": "source_start_box_contains_true_state_and_C",
                }

            upper_box = list(union_box)
            upper_face_values = [upper0[index], upper1[index]]
            upper_box[index] = iv.mpf([min(upper_face_values), max(upper_face_values)])
            try:
                upper_rhs_low, upper_rhs_high = _subdivided_interval_carried_c_p_time_rhs_component(
                    p_interval,
                    tuple(upper_box),
                    b_interval,
                    index,
                    subdivisions,
                    p_subdivisions,
                )
            except ZeroDivisionError as exc:
                return {
                    "status": "failed",
                    "failure": str(exc),
                    "candidate_A": candidate_a,
                    "start_p": start_p,
                    "end_p": end_p,
                    "certified_to_p": p0,
                    "steps_certified": step_index,
                    "failing_face": {
                        "side": "upper",
                        "component": index,
                        "p_interval": [p1, p0],
                    },
                    "conditional": "source_start_box_contains_true_state_and_C",
                }
            margin = upper_rhs_low - upper_slope[index]
            if margin < worst_margin:
                worst_margin = margin
                worst_face = {
                    "side": "upper",
                    "component": index,
                    "p_interval": [p1, p0],
                    "rhs_lower": upper_rhs_low,
                    "rhs_upper": upper_rhs_high,
                    "face_slope": upper_slope[index],
                    "slab_low": slab_low,
                    "slab_high": slab_high,
                }
            if margin < 0.0:
                return {
                    "status": "failed",
                    "candidate_A": candidate_a,
                    "b_interval": [-beta, beta],
                    "start_p": start_p,
                    "end_p": end_p,
                    "step_size": step_size,
                    "steps": steps,
                    "certified_to_p": p0,
                    "steps_certified": step_index,
                    "lower_start": list(lower_start),
                    "upper_start": list(upper_start),
                    "lower_slope": list(lower_slope),
                    "upper_slope": list(upper_slope),
                    "subdivisions": list(subdivisions),
                    "p_subdivisions": p_subdivisions,
                    "source_box_contained": _corridor_contains_box_nd(
                        lower_start,
                        upper_start,
                        source_box_low,
                        source_box_high,
                    ),
                    "source_box_5d": {"low": list(source_box_low), "high": list(source_box_high)},
                    "worst_margin": worst_margin,
                    "failing_face": worst_face,
                    "conditional": "source_start_box_contains_true_state_and_C",
                }

    end_lower = _affine_barrier_value_nd(lower_start, lower_slope, start_p, end_p)
    end_upper = _affine_barrier_value_nd(upper_start, upper_slope, start_p, end_p)
    end_lower, end_upper, c_handoff_sharpened = sharpen_carried_c_p_slice_box(
        end_p,
        end_lower,
        end_upper,
    )
    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "start_p": start_p,
        "end_p": end_p,
        "step_size": step_size,
        "steps": steps,
        "certified_to_p": end_p,
        "steps_certified": steps,
        "lower_start": list(lower_start),
        "upper_start": list(upper_start),
        "lower_slope": list(lower_slope),
        "upper_slope": list(upper_slope),
        "subdivisions": list(subdivisions),
        "p_subdivisions": p_subdivisions,
        "source_box_contained": _corridor_contains_box_nd(
            lower_start,
            upper_start,
            source_box_low,
            source_box_high,
        ),
        "source_box_5d": {"low": list(source_box_low), "high": list(source_box_high)},
        "end_box_5d": {"low": list(end_lower), "high": list(end_upper)},
        "end_box": {"low": list(end_lower[:4]), "high": list(end_upper[:4])},
        "c_interval": [end_lower[4], end_upper[4]],
        "c_handoff_sharpened": c_handoff_sharpened,
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "conditional": "source_start_box_contains_true_state_and_C",
    }


def automatic_carried_c_p_barrier_corridor_certificate(
    start_p: float,
    end_p: float,
    source_box_low: tuple[float, ...],
    source_box_high: tuple[float, ...],
    step_size: float = DEFAULT_CARRIED_C_P_CORRIDOR_STEP,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    safety=DEFAULT_CARRIED_C_P_CORRIDOR_SAFETY,
    subdivisions: tuple[int, int, int, int, int] = DEFAULT_CARRIED_C_P_CORRIDOR_SUBDIVISIONS,
    p_subdivisions: int = DEFAULT_CARRIED_C_P_CORRIDOR_P_SUBDIVISIONS,
) -> dict:
    """Greedily certify a carried-C p-time barrier corridor."""
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if start_p <= end_p:
        raise ValueError("start_p must be greater than end_p")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if len(subdivisions) != 5:
        raise ValueError("carried-C corridor subdivisions must have five entries")
    safety_tuple = _component_safety_tuple_n(safety, 5, "carried-C corridor safety")
    total_steps = round((start_p - end_p) / step_size)
    if abs(start_p - total_steps * step_size - end_p) > 1e-12:
        raise ValueError("automatic carried-C corridor length must be an integer multiple of step_size")

    if len(source_box_low) == 4 and len(source_box_high) == 4:
        low, high = augment_p_slice_box_with_c(start_p, tuple(source_box_low), tuple(source_box_high))
        c_source = "algebraic_start_box"
    elif len(source_box_low) == 5 and len(source_box_high) == 5:
        low = tuple(float(value) for value in source_box_low)
        high = tuple(float(value) for value in source_box_high)
        c_source = "carried_start_box"
    else:
        raise ValueError("source boxes must have either four or five entries")
    if not all(low[index] <= high[index] for index in range(5)):
        raise ValueError("source box must be componentwise ordered")
    sharpened_low, sharpened_high, start_c_sharpened = sharpen_carried_c_p_slice_box(start_p, low, high)
    low = sharpened_low
    high = sharpened_high

    from mpmath import iv

    beta = 1.0 / candidate_a
    b_interval = iv.mpf([-beta, beta])
    p_value = start_p
    worst_margin = math.inf
    worst_face: dict | None = None
    last_step: dict | None = None

    for step_index in range(total_steps):
        p_interval = iv.mpf([p_value, p_value])
        current_box = tuple(iv.mpf([low[index], high[index]]) for index in range(5))
        try:
            rhs_bounds = tuple(
                _subdivided_interval_carried_c_p_time_rhs_component(
                    p_interval,
                    current_box,
                    b_interval,
                    component,
                    subdivisions,
                    p_subdivisions,
                )
                for component in range(5)
            )
        except ZeroDivisionError as exc:
            return {
                "status": "failed",
                "failure": str(exc),
                "candidate_A": candidate_a,
                "start_p": start_p,
                "end_p": end_p,
                "step_size": step_size,
                "safety": list(safety_tuple),
                "subdivisions": list(subdivisions),
                "p_subdivisions": p_subdivisions,
                "certified_to_p": p_value,
                "steps_certified": step_index,
                "current_box_5d": {"low": list(low), "high": list(high)},
                "last_certified_step": last_step,
                "conditional": "source_start_box_contains_true_state_and_C",
            }
        lower_slope = tuple(bound[1] + safety_tuple[index] for index, bound in enumerate(rhs_bounds))
        upper_slope = tuple(bound[0] - safety_tuple[index] for index, bound in enumerate(rhs_bounds))
        next_p = p_value - step_size
        next_low = tuple(low[index] - step_size * lower_slope[index] for index in range(5))
        next_high = tuple(high[index] - step_size * upper_slope[index] for index in range(5))
        if any(next_low[index] > next_high[index] for index in range(5)):
            return {
                "status": "failed",
                "failure": "empty_next_box",
                "candidate_A": candidate_a,
                "start_p": start_p,
                "end_p": end_p,
                "step_size": step_size,
                "safety": list(safety_tuple),
                "subdivisions": list(subdivisions),
                "p_subdivisions": p_subdivisions,
                "certified_to_p": p_value,
                "steps_certified": step_index,
                "current_box_5d": {"low": list(low), "high": list(high)},
                "proposed_next_box_5d": {"low": list(next_low), "high": list(next_high)},
                "rhs_bounds": [list(bound) for bound in rhs_bounds],
                "conditional": "source_start_box_contains_true_state_and_C",
            }

        step_certificate = affine_carried_c_p_corridor_certificate(
            start_p=p_value,
            end_p=next_p,
            step_size=step_size,
            candidate_a=candidate_a,
            lower_start=low,
            upper_start=high,
            lower_slope=lower_slope,
            upper_slope=upper_slope,
            subdivisions=subdivisions,
            p_subdivisions=p_subdivisions,
            source_box_low=low,
            source_box_high=high,
        )
        if step_certificate["status"] != "certified":
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "start_p": start_p,
                "end_p": end_p,
                "step_size": step_size,
                "safety": list(safety_tuple),
                "subdivisions": list(subdivisions),
                "p_subdivisions": p_subdivisions,
                "certified_to_p": p_value,
                "steps_certified": step_index,
                "current_box_5d": {"low": list(low), "high": list(high)},
                "proposed_next_box_5d": {"low": list(next_low), "high": list(next_high)},
                "rhs_bounds": [list(bound) for bound in rhs_bounds],
                "failing_step": step_certificate,
                "last_certified_step": last_step,
                "conditional": "source_start_box_contains_true_state_and_C",
            }
        if step_certificate["worst_margin"] < worst_margin:
            worst_margin = step_certificate["worst_margin"]
            worst_face = step_certificate.get("worst_face")
        low = tuple(step_certificate["end_box_5d"]["low"])
        high = tuple(step_certificate["end_box_5d"]["high"])
        last_step = {
            "start_p": p_value,
            "end_p": next_p,
            "worst_margin": step_certificate["worst_margin"],
            "worst_face": step_certificate.get("worst_face"),
            "end_box_5d": {"low": list(low), "high": list(high)},
            "c_handoff_sharpened": step_certificate.get("c_handoff_sharpened"),
        }
        p_value = next_p

    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "start_p": start_p,
        "end_p": end_p,
        "step_size": step_size,
        "steps": total_steps,
        "steps_certified": total_steps,
        "safety": list(safety_tuple),
        "subdivisions": list(subdivisions),
        "p_subdivisions": p_subdivisions,
        "certified_to_p": end_p,
        "source_box": {"low": list(source_box_low), "high": list(source_box_high)},
        "start_box_5d": {"low": list(sharpened_low), "high": list(sharpened_high)},
        "start_c_sharpened": start_c_sharpened,
        "c_source": c_source,
        "end_box_5d": {"low": list(low), "high": list(high)},
        "end_box": {"low": list(low[:4]), "high": list(high[:4])},
        "c_interval": [low[4], high[4]],
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "last_certified_step": last_step,
        "conditional": "source_start_box_contains_true_state_and_C",
    }


def tune_affine_p_corridor(
    x2_lower_slopes: tuple[float, ...] = DEFAULT_P_CORRIDOR_TUNE_X2_SLOPES,
    x1_upper_slopes: tuple[float, ...] = DEFAULT_P_CORRIDOR_TUNE_X1_UPPER_SLOPES,
    start_p: float = DEFAULT_P_CORRIDOR_START,
    end_p: float = DEFAULT_P_CORRIDOR_END,
    step_size: float = DEFAULT_P_CORRIDOR_STEP,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    lower_start: tuple[float, float, float, float] = DEFAULT_P_CORRIDOR_LOWER_START,
    upper_start: tuple[float, float, float, float] = DEFAULT_P_CORRIDOR_UPPER_START,
    lower_slope: tuple[float, float, float, float] = DEFAULT_P_CORRIDOR_LOWER_SLOPE,
    upper_slope: tuple[float, float, float, float] = DEFAULT_P_CORRIDOR_UPPER_SLOPE,
    subdivisions: tuple[int, int, int, int] = (2, 2, 2, 2),
    p_subdivisions: int = 2,
    max_runs: int | None = None,
) -> dict:
    """Tune the two active affine-corridor slopes near the current bottleneck."""
    if max_runs is not None and max_runs <= 0:
        raise ValueError("max_runs must be positive when supplied")
    results = []
    run_count = 0
    for x2_slope in x2_lower_slopes:
        for x1_slope in x1_upper_slopes:
            tuned_lower_slope = tuple(
                x2_slope if index == 2 else lower_slope[index]
                for index in range(4)
            )
            tuned_upper_slope = tuple(
                x1_slope if index == 1 else upper_slope[index]
                for index in range(4)
            )
            certificate = affine_p_corridor_certificate(
                start_p=start_p,
                end_p=end_p,
                step_size=step_size,
                candidate_a=candidate_a,
                lower_start=lower_start,
                upper_start=upper_start,
                lower_slope=tuned_lower_slope,
                upper_slope=tuned_upper_slope,
                subdivisions=subdivisions,
                p_subdivisions=p_subdivisions,
            )
            results.append(
                {
                    "x2_lower_slope": x2_slope,
                    "x1_upper_slope": x1_slope,
                    "status": certificate["status"],
                    "certified_to_p": certificate["certified_to_p"],
                    "steps_certified": certificate["steps_certified"],
                    "worst_margin": certificate["worst_margin"],
                    "face": certificate.get("failing_face") or certificate.get("worst_face"),
                }
            )
            run_count += 1
            if max_runs is not None and run_count >= max_runs:
                break
        if max_runs is not None and run_count >= max_runs:
            break

    def rank(item: dict) -> tuple:
        certified_distance = start_p - item["certified_to_p"]
        is_certified = 1 if item["status"] == "certified" else 0
        return (is_certified, certified_distance, item["worst_margin"])

    best = max(results, key=rank) if results else None
    return {
        "status": "no_runs" if best is None else "completed",
        "start_p": start_p,
        "end_p": end_p,
        "step_size": step_size,
        "candidate_A": candidate_a,
        "x2_lower_slopes": list(x2_lower_slopes),
        "x1_upper_slopes": list(x1_upper_slopes),
        "runs": run_count,
        "best": best,
        "results": sorted(results, key=rank, reverse=True),
    }


def p_tube_frontier_continuation_certificate(
    start_p: float = DEFAULT_P_TUBE_END,
    end_p: float = DEFAULT_FRONTIER_CONTINUATION_END,
    step_size: float = DEFAULT_P_TUBE_STEP,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    source_box_low: tuple[float, float, float, float] = DEFAULT_P_TUBE_FRONTIER_LOW,
    source_box_high: tuple[float, float, float, float] = DEFAULT_P_TUBE_FRONTIER_HIGH,
    profiles: tuple = DEFAULT_ASYMMETRIC_P_TUBE_PROFILES,
    subdivisions: tuple[int, int, int, int] = DEFAULT_FRONTIER_CONTINUATION_SUBDIVISIONS,
    p_subdivisions: int = DEFAULT_FRONTIER_CONTINUATION_P_SUBDIVISIONS,
    seed_step_size: float = 1e-5,
) -> dict:
    """Continue the certified p-tube frontier from a supplied start box."""
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if start_p <= end_p:
        raise ValueError("start_p must be greater than end_p")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    total_steps = round((start_p - end_p) / step_size)
    if abs(start_p - total_steps * step_size - end_p) > 1e-12:
        raise ValueError("frontier continuation length must be an integer multiple of step_size")

    beta = 1.0 / candidate_a
    samples = tuple(
        scaled_state_at_p(
            "limit" if b == 0.0 else "exact",
            start_p,
            None if b == 0.0 else 1.0 / b,
            step_size=seed_step_size,
        )
        for b in (-beta, 0.0, beta)
    )
    low = tuple(source_box_low)
    high = tuple(source_box_high)
    p_value = start_p
    certified_blocks = 0
    worst_margin = math.inf
    worst_face: dict | None = None
    last_block: dict | None = None

    while certified_blocks < total_steps:
        candidates = []
        failures = []
        for profile in profiles:
            lower_growth, upper_growth = _normalize_p_tube_profile(profile)
            block = _p_tube_block_certificate(
                p_value,
                -step_size,
                1,
                candidate_a,
                samples,
                low,
                high,
                lower_growth,
                upper_growth,
                subdivisions,
                p_subdivisions,
            )
            if block["status"] == "certified":
                end_low = block["end_box"]["low"]
                end_high = block["end_box"]["high"]
                width_sum = sum(end_high[index] - end_low[index] for index in range(4))
                candidates.append((width_sum, block))
            else:
                failures.append(block)
        if not candidates:
            best_failure = max(failures, key=lambda item: item["worst_margin"]) if failures else None
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "b_interval": [-beta, beta],
                "start_p": start_p,
                "end_p": end_p,
                "step_size": step_size,
                "subdivisions": list(subdivisions),
                "p_subdivisions": p_subdivisions,
                "certified_to_p": p_value,
                "blocks_certified": certified_blocks,
                "current_box": {"low": list(low), "high": list(high)},
                "failing_block": best_failure,
                "last_certified_block": last_block,
                "conditional": "source_start_box_contains_true_state",
            }
        _width, block = min(candidates, key=lambda item: item[0])
        if block["worst_margin"] < worst_margin:
            worst_margin = block["worst_margin"]
            worst_face = block.get("worst_face")
        last_block = {key: value for key, value in block.items() if key != "end_samples"}
        samples = tuple(tuple(sample) for sample in block["end_samples"])
        low = tuple(block["end_box"]["low"])
        high = tuple(block["end_box"]["high"])
        certified_blocks += 1
        p_value = start_p - certified_blocks * step_size

    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "start_p": start_p,
        "end_p": end_p,
        "step_size": step_size,
        "subdivisions": list(subdivisions),
        "p_subdivisions": p_subdivisions,
        "certified_to_p": p_value,
        "blocks_certified": certified_blocks,
        "end_box": {"low": list(low), "high": list(high)},
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "last_certified_block": last_block,
        "conditional": "source_start_box_contains_true_state",
    }


def piecewise_affine_p_corridor_certificate(
    knots: tuple = DEFAULT_PIECEWISE_CORRIDOR_KNOTS,
    step_size: float = 1e-3,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    subdivisions: tuple[int, int, int, int] = (4, 4, 4, 2),
    p_subdivisions: int = 1,
) -> dict:
    """Check a chain of affine p-time corridors between supplied knots."""
    if len(knots) < 2:
        raise ValueError("piecewise corridor needs at least two knots")
    segments = []
    worst_margin = math.inf
    worst_face: dict | None = None
    for index in range(len(knots) - 1):
        start_p, lower_start, upper_start = knots[index]
        end_p, lower_end, upper_end = knots[index + 1]
        if start_p <= end_p:
            raise ValueError("piecewise corridor knots must have decreasing p values")
        lower_slope = tuple(
            (lower_end[component] - lower_start[component]) / (end_p - start_p)
            for component in range(4)
        )
        upper_slope = tuple(
            (upper_end[component] - upper_start[component]) / (end_p - start_p)
            for component in range(4)
        )
        segment = affine_p_corridor_certificate(
            start_p=start_p,
            end_p=end_p,
            step_size=step_size,
            candidate_a=candidate_a,
            lower_start=lower_start,
            upper_start=upper_start,
            lower_slope=lower_slope,
            upper_slope=upper_slope,
            subdivisions=subdivisions,
            p_subdivisions=p_subdivisions,
            source_box_low=lower_start,
            source_box_high=upper_start,
        )
        segments.append(segment)
        if segment.get("worst_margin", math.inf) < worst_margin:
            worst_margin = segment["worst_margin"]
            worst_face = segment.get("failing_face") or segment.get("worst_face")
        if segment["status"] != "certified":
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "step_size": step_size,
                "subdivisions": list(subdivisions),
                "p_subdivisions": p_subdivisions,
                "certified_to_p": segment["certified_to_p"],
                "segments_certified": index,
                "failing_segment_index": index,
                "failing_segment": segment,
                "segments": segments,
                "worst_margin": worst_margin,
                "worst_face": worst_face,
                "conditional": "first_knot_box_contains_true_state",
            }

    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "step_size": step_size,
        "subdivisions": list(subdivisions),
        "p_subdivisions": p_subdivisions,
        "certified_to_p": knots[-1][0],
        "segments_certified": len(knots) - 1,
        "segments": segments,
        "end_box": {"low": list(knots[-1][1]), "high": list(knots[-1][2])},
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "conditional": "first_knot_box_contains_true_state",
    }


def p_start_slice_box(
    start_p: float = DEFAULT_HYBRID_HANDOFF_START_P,
    entry_time: float = DEFAULT_HYBRID_HANDOFF_ENTRY_TIME,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius0: tuple[float, float, float, float] = DEFAULT_HYBRID_HANDOFF_P_TUBE_RADIUS0,
    seed_step_size: float = 1e-5,
) -> dict:
    """Return the nominal p-start slice box used by the hybrid handoff."""
    beta = 1.0 / candidate_a
    samples = tuple(
        scaled_state_at_p(
            "limit" if b == 0.0 else "exact",
            start_p,
            None if b == 0.0 else 1.0 / b,
            entry_time=entry_time,
            step_size=seed_step_size,
        )
        for b in (-beta, 0.0, beta)
    )
    low = tuple(min(sample[index] for sample in samples) - radius0[index] for index in range(4))
    high = tuple(max(sample[index] for sample in samples) + radius0[index] for index in range(4))
    return {
        "start_p": start_p,
        "entry_time": entry_time,
        "candidate_A": candidate_a,
        "radius0": list(radius0),
        "samples": [list(sample) for sample in samples],
        "box": {"low": list(low), "high": list(high)},
    }


def p_start_slice_from_support_certificate(
    target_p: float = DEFAULT_HYBRID_HANDOFF_START_P,
    support_time: float = DEFAULT_SUPPORT_TIME,
    after_time: float = DEFAULT_HYBRID_HANDOFF_BRIDGE_AFTER_TIME,
    step_size: float = 1e-4,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    support_radius0: tuple[float, float, float, float] = DEFAULT_SUPPORT_TUBE_RADIUS,
    radius0: tuple[float, float, float, float] = DEFAULT_HYBRID_HANDOFF_P_TUBE_RADIUS0,
    subdivisions: tuple[int, int, int, int] = (2, 2, 1, 2),
    time_subdivisions: int = 1,
    seed_step_size: float = 1e-5,
) -> dict:
    """Certify that the support tube crosses into the hybrid p-start box.

    This is conditional on the ordinary segmented t-time tube start box at
    ``support_time`` containing the true finite-|a| scaled state.
    """
    if after_time <= support_time:
        raise ValueError("after_time must be greater than support_time")
    steps = round((after_time - support_time) / step_size)
    if abs(support_time + steps * step_size - after_time) > 1e-12:
        raise ValueError("bridge interval length must be an integer multiple of step_size")

    tube = segmented_moving_tube_certificate(
        start_time=support_time,
        end_time=after_time,
        step_size=step_size,
        block_steps=1,
        candidate_a=candidate_a,
        radius0=support_radius0,
        subdivisions=subdivisions,
        time_subdivisions=time_subdivisions,
        seed_step_size=seed_step_size,
    )
    if tube["status"] != "certified":
        return {
            "status": "failed",
            "candidate_A": candidate_a,
            "stage": "support_tube",
            "support_tube": tube,
            "conditional": "support_start_box_contains_true_state",
        }

    last_block = tube["last_certified_block"]
    before_box = last_block["start_box"]
    after_box = last_block["end_box"]
    before_time = after_time - step_size
    before_above = before_box["low"][0] > target_p
    after_below = after_box["high"][0] < target_p
    crossing_low = [
        before_time,
        min(before_box["low"][1], after_box["low"][1]),
        min(before_box["low"][2], after_box["low"][2]),
        min(before_box["low"][3], after_box["low"][3]),
    ]
    crossing_high = [
        after_time,
        max(before_box["high"][1], after_box["high"][1]),
        max(before_box["high"][2], after_box["high"][2]),
        max(before_box["high"][3], after_box["high"][3]),
    ]
    start_slice = p_start_slice_box(
        start_p=target_p,
        entry_time=support_time,
        candidate_a=candidate_a,
        radius0=radius0,
        seed_step_size=seed_step_size,
    )
    start_low = start_slice["box"]["low"]
    start_high = start_slice["box"]["high"]
    crossing_contained = all(
        start_low[index] <= crossing_low[index]
        and crossing_high[index] <= start_high[index]
        for index in range(4)
    )
    status = "certified_conditional"
    if not before_above or not after_below or not crossing_contained:
        status = "failed"
    return {
        "status": status,
        "candidate_A": candidate_a,
        "target_p": target_p,
        "support_time": support_time,
        "before_time": before_time,
        "after_time": after_time,
        "step_size": step_size,
        "support_radius0": list(support_radius0),
        "support_tube": tube,
        "before_above_target": before_above,
        "after_below_target": after_below,
        "crossing_slab": {"low": crossing_low, "high": crossing_high},
        "start_slice": start_slice,
        "crossing_slab_contained_in_start_slice": crossing_contained,
        "conditional": "support_start_box_contains_true_state",
    }


def hybrid_p_frontier_handoff_certificate(
    start_p: float = DEFAULT_HYBRID_HANDOFF_START_P,
    tube_end_p: float = DEFAULT_HYBRID_HANDOFF_TUBE_END_P,
    frontier_p: float = DEFAULT_HYBRID_HANDOFF_FRONTIER_P,
    entry_time: float = DEFAULT_HYBRID_HANDOFF_ENTRY_TIME,
    step_size: float = DEFAULT_P_TUBE_STEP,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    radius0: tuple[float, float, float, float] = DEFAULT_HYBRID_HANDOFF_P_TUBE_RADIUS0,
    frontier_low: tuple[float, float, float, float] = DEFAULT_HYBRID_HANDOFF_FRONTIER_LOW,
    frontier_high: tuple[float, float, float, float] = DEFAULT_HYBRID_HANDOFF_FRONTIER_HIGH,
    p_tube_profiles: tuple = DEFAULT_SEGMENTED_P_TUBE_PROFILES,
    p_tube_subdivisions: tuple[int, int, int, int] = (1, 1, 1, 1),
    p_tube_p_subdivisions: int = 1,
    corridor_subdivisions: tuple[int, int, int, int] = DEFAULT_HYBRID_HANDOFF_CORRIDOR_SUBDIVISIONS,
    corridor_p_subdivisions: int = 1,
    seed_step_size: float = 1e-5,
) -> dict:
    """Certify the current hybrid handoff into a broad p=0.25 frontier.

    The first stage is a narrow p-time tube from ``start_p`` to
    ``tube_end_p``.  The second stage abandons the narrow tube before it wraps
    too much and uses an affine barrier corridor into the supplied frontier
    box.  The result is still conditional on the true finite-|a| trajectories
    lying in the initial ``start_p`` slice box.
    """
    if start_p <= tube_end_p:
        raise ValueError("start_p must be greater than tube_end_p")
    if tube_end_p <= frontier_p:
        raise ValueError("tube_end_p must be greater than frontier_p")

    tube = segmented_p_tube_certificate(
        start_p=start_p,
        end_p=tube_end_p,
        entry_time=entry_time,
        step_size=step_size,
        candidate_a=candidate_a,
        radius0=radius0,
        profiles=p_tube_profiles,
        subdivisions=p_tube_subdivisions,
        p_subdivisions=p_tube_p_subdivisions,
        seed_step_size=seed_step_size,
    )
    if tube["status"] != "certified":
        return {
            "status": "failed",
            "candidate_A": candidate_a,
            "stage": "p_tube",
            "p_tube": tube,
            "conditional": "p_start_slice_box_contains_true_state",
        }

    source_low = tuple(tube["end_box"]["low"])
    source_high = tuple(tube["end_box"]["high"])
    lower_slope = tuple(
        (frontier_low[index] - source_low[index]) / (frontier_p - tube_end_p)
        for index in range(4)
    )
    upper_slope = tuple(
        (frontier_high[index] - source_high[index]) / (frontier_p - tube_end_p)
        for index in range(4)
    )
    corridor = affine_p_corridor_certificate(
        start_p=tube_end_p,
        end_p=frontier_p,
        step_size=step_size,
        candidate_a=candidate_a,
        lower_start=source_low,
        upper_start=source_high,
        lower_slope=lower_slope,
        upper_slope=upper_slope,
        subdivisions=corridor_subdivisions,
        p_subdivisions=corridor_p_subdivisions,
        source_box_low=source_low,
        source_box_high=source_high,
    )
    if corridor["status"] != "certified":
        return {
            "status": "failed",
            "candidate_A": candidate_a,
            "stage": "affine_corridor",
            "p_tube": tube,
            "affine_corridor": corridor,
            "frontier_box": {"low": list(frontier_low), "high": list(frontier_high)},
            "conditional": "p_start_slice_box_contains_true_state",
        }

    return {
        "status": "certified_conditional",
        "candidate_A": candidate_a,
        "start_p": start_p,
        "tube_end_p": tube_end_p,
        "frontier_p": frontier_p,
        "entry_time": entry_time,
        "step_size": step_size,
        "p_tube": tube,
        "affine_corridor": corridor,
        "frontier_box": {"low": list(frontier_low), "high": list(frontier_high)},
        "certified_from_p": start_p,
        "certified_to_p": frontier_p,
        "conditional": "p_start_slice_box_contains_true_state",
    }


def automatic_p_barrier_corridor_certificate(
    start_p: float,
    end_p: float,
    source_box_low: tuple[float, float, float, float],
    source_box_high: tuple[float, float, float, float],
    step_size: float = DEFAULT_P_TUBE_STEP,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    safety=DEFAULT_BROAD_TAIL_AUTOMATIC_SAFETY,
    subdivisions: tuple[int, int, int, int] = DEFAULT_BROAD_TAIL_AUTOMATIC_SUBDIVISIONS,
    p_subdivisions: int = DEFAULT_BROAD_TAIL_AUTOMATIC_P_SUBDIVISIONS,
) -> dict:
    """Greedily certify a p-time barrier corridor from interval RHS bounds.

    At each p-step the current box is used only to propose face slopes.  The
    step is then verified by the same affine face checker as hand-written
    corridors, so success is a genuine interval certificate and not an Euler
    approximation.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if start_p <= end_p:
        raise ValueError("start_p must be greater than end_p")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    safety_tuple = _component_safety_tuple(safety)
    total_steps = round((start_p - end_p) / step_size)
    if abs(start_p - total_steps * step_size - end_p) > 1e-12:
        raise ValueError("automatic corridor length must be an integer multiple of step_size")
    if not all(source_box_low[index] <= source_box_high[index] for index in range(4)):
        raise ValueError("source box must be componentwise ordered")

    from mpmath import iv

    beta = 1.0 / candidate_a
    b_interval = iv.mpf([-beta, beta])
    p_value = start_p
    low = tuple(source_box_low)
    high = tuple(source_box_high)
    worst_margin = math.inf
    worst_face: dict | None = None
    last_step: dict | None = None

    for step_index in range(total_steps):
        p_interval = iv.mpf([p_value, p_value])
        current_box = tuple(iv.mpf([low[index], high[index]]) for index in range(4))
        rhs_bounds = tuple(
            _subdivided_interval_p_time_rhs_component(
                p_interval,
                current_box,
                b_interval,
                component,
                subdivisions,
                p_subdivisions,
            )
            for component in range(4)
        )
        # p decreases.  A lower face is inward if L' >= G_high, and an
        # upper face is inward if U' <= G_low.
        lower_slope = tuple(bound[1] + safety_tuple[index] for index, bound in enumerate(rhs_bounds))
        upper_slope = tuple(bound[0] - safety_tuple[index] for index, bound in enumerate(rhs_bounds))
        next_p = p_value - step_size
        next_low = tuple(low[index] - step_size * lower_slope[index] for index in range(4))
        next_high = tuple(high[index] - step_size * upper_slope[index] for index in range(4))
        if any(next_low[index] > next_high[index] for index in range(4)):
            return {
                "status": "failed",
                "failure": "empty_next_box",
                "candidate_A": candidate_a,
                "start_p": start_p,
                "end_p": end_p,
                "step_size": step_size,
                "safety": list(safety_tuple),
                "subdivisions": list(subdivisions),
                "p_subdivisions": p_subdivisions,
                "certified_to_p": p_value,
                "steps_certified": step_index,
                "current_box": {"low": list(low), "high": list(high)},
                "proposed_next_box": {"low": list(next_low), "high": list(next_high)},
                "rhs_bounds": [list(bound) for bound in rhs_bounds],
                "conditional": "source_start_box_contains_true_state",
            }

        step_certificate = affine_p_corridor_certificate(
            start_p=p_value,
            end_p=next_p,
            step_size=step_size,
            candidate_a=candidate_a,
            lower_start=low,
            upper_start=high,
            lower_slope=lower_slope,
            upper_slope=upper_slope,
            subdivisions=subdivisions,
            p_subdivisions=p_subdivisions,
            source_box_low=low,
            source_box_high=high,
        )
        if step_certificate["status"] != "certified":
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "start_p": start_p,
                "end_p": end_p,
                "step_size": step_size,
                "safety": list(safety_tuple),
                "subdivisions": list(subdivisions),
                "p_subdivisions": p_subdivisions,
                "certified_to_p": p_value,
                "steps_certified": step_index,
                "current_box": {"low": list(low), "high": list(high)},
                "proposed_next_box": {"low": list(next_low), "high": list(next_high)},
                "rhs_bounds": [list(bound) for bound in rhs_bounds],
                "failing_step": step_certificate,
                "last_certified_step": last_step,
                "conditional": "source_start_box_contains_true_state",
            }
        if step_certificate["worst_margin"] < worst_margin:
            worst_margin = step_certificate["worst_margin"]
            worst_face = step_certificate.get("worst_face")
        last_step = {
            "start_p": p_value,
            "end_p": next_p,
            "worst_margin": step_certificate["worst_margin"],
            "worst_face": step_certificate.get("worst_face"),
            "end_box": {"low": list(next_low), "high": list(next_high)},
        }
        p_value = next_p
        low = next_low
        high = next_high

    return {
        "status": "certified",
        "candidate_A": candidate_a,
        "start_p": start_p,
        "end_p": end_p,
        "step_size": step_size,
        "steps": total_steps,
        "steps_certified": total_steps,
        "safety": list(safety_tuple),
        "subdivisions": list(subdivisions),
        "p_subdivisions": p_subdivisions,
        "certified_to_p": end_p,
        "source_box": {"low": list(source_box_low), "high": list(source_box_high)},
        "end_box": {"low": list(low), "high": list(high)},
        "worst_margin": worst_margin,
        "worst_face": worst_face,
        "last_certified_step": last_step,
        "conditional": "source_start_box_contains_true_state",
    }


def carried_c_p_wall_certificate(
    start_p: float = DEFAULT_CARRIED_C_P_WALL_START,
    end_p: float = DEFAULT_CARRIED_C_P_WALL_END,
    p_step: float = DEFAULT_CARRIED_C_P_WALL_STEP,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    box_low: tuple[float, ...] = DEFAULT_CARRIED_C_P_WALL_BOX_LOW,
    box_high: tuple[float, ...] = DEFAULT_CARRIED_C_P_WALL_BOX_HIGH,
    component: int = 2,
    side: str = "lower",
    wall_value: float = 0.0,
    source_box_low: tuple[float, ...] | None = None,
    source_box_high: tuple[float, ...] | None = None,
    subdivisions: tuple[int, int, int, int, int] = DEFAULT_CARRIED_C_P_WALL_SUBDIVISIONS,
    p_subdivisions: int = DEFAULT_CARRIED_C_P_WALL_P_SUBDIVISIONS,
) -> dict:
    """Check a constant p-time wall using the carried-C denominator."""
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if start_p <= end_p:
        raise ValueError("start_p must be greater than end_p")
    if p_step <= 0.0:
        raise ValueError("p_step must be positive")
    if component < 0 or component > 4:
        raise ValueError("wall component must be between 0 and 4")
    if side not in {"lower", "upper"}:
        raise ValueError("wall side must be 'lower' or 'upper'")
    if len(box_low) not in {4, 5} or len(box_high) != len(box_low):
        raise ValueError("wall box bounds must have four or five coordinates")
    if component == 4 and len(box_low) != 5:
        raise ValueError("C-wall checks require five-coordinate carried-C bounds")
    if len(subdivisions) != 5:
        raise ValueError("wall subdivisions must have five coordinates")
    dimension = len(box_low)
    if not all(box_low[index] <= box_high[index] for index in range(dimension)):
        raise ValueError("box_low must be componentwise <= box_high")
    if not (box_low[component] <= wall_value <= box_high[component]):
        raise ValueError("wall_value must lie inside the requested wall component range")
    if source_box_low is None:
        source_box_low = box_low
    if source_box_high is None:
        source_box_high = box_high
    if len(source_box_low) != dimension or len(source_box_high) != dimension:
        raise ValueError("source and wall boxes must have the same dimension")

    from mpmath import iv

    beta = 1.0 / candidate_a
    b_interval = iv.mpf([-beta, beta])
    source_contained = _corridor_contains_box_nd(
        tuple(box_low),
        tuple(box_high),
        tuple(source_box_low),
        tuple(source_box_high),
    )
    p_hi = start_p
    steps = 0
    worst_margin = math.inf
    worst_slice: dict | None = None
    while p_hi > end_p + 1e-15:
        p_lo = max(end_p, p_hi - p_step)
        p_interval = iv.mpf([p_lo, p_hi])
        face = [
            iv.mpf([box_low[index], box_high[index]])
            for index in range(dimension)
        ]
        face[component] = iv.mpf([wall_value, wall_value])
        if dimension == 4:
            c_interval = face[1] * face[2] - p_interval * p_interval * face[3] / 6.0
            z_interval = (face[0], face[1], face[2], face[3], c_interval)
        else:
            z_interval = tuple(face)
        try:
            rhs_low, rhs_high = _subdivided_interval_carried_c_p_time_rhs_component(
                p_interval,
                z_interval,
                b_interval,
                component,
                subdivisions,
                p_subdivisions,
            )
        except ZeroDivisionError as exc:
            return {
                "status": "failed",
                "failure": str(exc),
                "candidate_A": candidate_a,
                "b_interval": [-beta, beta],
                "start_p": start_p,
                "end_p": end_p,
                "p_step": p_step,
                "certified_to_p": p_hi,
                "steps_certified": steps,
                "box_low": list(box_low),
                "box_high": list(box_high),
                "component": component,
                "side": side,
                "wall_value": wall_value,
                "source_box": {"low": list(source_box_low), "high": list(source_box_high)},
                "source_box_contained": source_contained,
                "subdivisions": list(subdivisions),
                "p_subdivisions": p_subdivisions,
                "failing_slice": {"p_interval": [p_lo, p_hi]},
                "conditional": "trajectory_remains_in_wall_box",
            }
        margin = -rhs_high if side == "lower" else rhs_low
        if margin < worst_margin:
            worst_margin = margin
            worst_slice = {
                "p_interval": [p_lo, p_hi],
                "rhs_lower": rhs_low,
                "rhs_upper": rhs_high,
                "margin": margin,
            }
        if margin < 0.0:
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "b_interval": [-beta, beta],
                "start_p": start_p,
                "end_p": end_p,
                "p_step": p_step,
                "certified_to_p": p_hi,
                "steps_certified": steps,
                "box_low": list(box_low),
                "box_high": list(box_high),
                "component": component,
                "side": side,
                "wall_value": wall_value,
                "source_box": {"low": list(source_box_low), "high": list(source_box_high)},
                "source_box_contained": source_contained,
                "subdivisions": list(subdivisions),
                "p_subdivisions": p_subdivisions,
                "worst_margin": worst_margin,
                "failing_slice": worst_slice,
                "conditional": "trajectory_remains_in_wall_box",
            }
        steps += 1
        p_hi = p_lo

    return {
        "status": "certified_conditional",
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "start_p": start_p,
        "end_p": end_p,
        "p_step": p_step,
        "certified_to_p": end_p,
        "steps_certified": steps,
        "box_low": list(box_low),
        "box_high": list(box_high),
        "component": component,
        "side": side,
        "wall_value": wall_value,
        "source_box": {"low": list(source_box_low), "high": list(source_box_high)},
        "source_box_contained": source_contained,
        "subdivisions": list(subdivisions),
        "p_subdivisions": p_subdivisions,
        "worst_margin": worst_margin,
        "worst_slice": worst_slice,
        "conditional": "trajectory_remains_in_wall_box",
    }


def terminal_barrier_takeover_certificate(
    p_start: float = DEFAULT_P_TUBE_END,
    p_min: float = DEFAULT_TERMINAL_TAKEOVER_P_MIN,
    p_step: float = DEFAULT_TERMINAL_TAKEOVER_P_STEP,
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    box_low: tuple[float, float, float, float] = DEFAULT_TERMINAL_TAKEOVER_BOX_LOW,
    box_high: tuple[float, float, float, float] = DEFAULT_TERMINAL_TAKEOVER_BOX_HIGH,
    x3_wall: float = DEFAULT_TERMINAL_TAKEOVER_X3_WALL,
    source_box_low: tuple[float, float, float, float] = DEFAULT_P_TUBE_FRONTIER_LOW,
    source_box_high: tuple[float, float, float, float] = DEFAULT_P_TUBE_FRONTIER_HIGH,
    subdivisions: tuple[int, int, int, int] = (2, 2, 2, 1),
    p_subdivisions: int = 4,
) -> dict:
    """Check the conditional terminal x3-wall takeover.

    This is not the full tail proof.  It verifies the terminal face mechanism
    that would finish the proof once the late trajectory is kept inside the
    supplied coarse box with the supplied positive x2 floor.
    """
    if candidate_a <= 0.0:
        raise ValueError("candidate_a must be positive")
    if p_start <= p_min:
        raise ValueError("p_start must be greater than p_min")
    if p_step <= 0.0:
        raise ValueError("p_step must be positive")
    if not all(box_low[index] <= box_high[index] for index in range(4)):
        raise ValueError("box_low must be componentwise <= box_high")
    if not (box_low[3] <= x3_wall <= box_high[3]):
        raise ValueError("x3_wall must lie inside the x3 box")

    from mpmath import iv

    beta = 1.0 / candidate_a
    b_interval = iv.mpf([-beta, beta])
    source_contained = _corridor_contains_box(box_low, box_high, source_box_low, source_box_high)
    source_below_wall = source_box_high[3] <= x3_wall
    source_x2_floor = source_box_low[2] >= box_low[2]
    x3_zero_margin = box_low[1] - x3_zero_threshold(box_low[0])
    sigma = -x3_wall
    if sigma <= 0.0:
        raise ValueError("x3_wall must be negative")
    t_low, q_low, r_low, _s_low = box_low
    t_high, q_high, r_high, _s_high = box_high
    if t_low <= 0.0 or q_low <= 0.0 or r_low <= 0.0:
        raise ValueError("terminal takeover box needs positive t, x1, and x2 lower bounds")

    # For 0 < p <= p_min on the x3=x3_wall face, the singular terms dominate.
    # The p' upper bound is -margin/p^4.  The x3' upper bound is checked at
    # p_min; it only improves as p decreases.
    p_prime_negative_coefficient = 3.0 * r_low * sigma**2 / t_high
    p_prime_r1_coefficient = 1.5 * (
        t_high * (2.0 * sigma * q_high * r_high + 0.5 * p_min**2 * sigma**2)
        + t_high**3 * q_high * p_min**4 / 18.0
    )
    p_prime_r3_coefficient = 3.0 * t_high**3 * q_high * sigma**2
    p_prime_small_p_margin = (
        p_prime_negative_coefficient
        - beta * p_prime_r1_coefficient
        - beta**3 * p_prime_r3_coefficient
    )
    x3_prime_regular_upper = (2.0 * sigma + 6.0 * p_min) / t_low
    x3_prime_negative_coefficient = t_low * sigma * q_low * r_low / 2.0
    x3_prime_r1_coefficient = (
        t_high
        / 2.0
        * (sigma**3 + 2.0 * t_high**2 * q_high * p_min**2 * sigma / 3.0)
    )
    x3_prime_singular_margin = x3_prime_negative_coefficient - beta * x3_prime_r1_coefficient
    x3_prime_upper_at_p_min = x3_prime_regular_upper - x3_prime_singular_margin / p_min**3
    small_p_tail = {
        "p_range": [0.0, p_min],
        "p_prime_upper_bound": f"-{p_prime_small_p_margin}/p^4",
        "p_prime_negative_coefficient_margin": p_prime_small_p_margin,
        "x3_prime_upper_at_p_min": x3_prime_upper_at_p_min,
        "x3_prime_negative_margin_at_p_min": -x3_prime_upper_at_p_min,
        "x3_prime_singular_coefficient_margin": x3_prime_singular_margin,
    }

    worst_margin = math.inf
    worst_slice: dict | None = None
    p_hi = p_start
    while p_hi > p_min + 1e-15:
        p_lo = max(p_min, p_hi - p_step)
        p_interval = iv.mpf([p_lo, p_hi])
        face_box = [
            iv.mpf([box_low[index], box_high[index]])
            for index in range(4)
        ]
        face_box[3] = iv.mpf([x3_wall, x3_wall])
        try:
            rhs_low, rhs_high = _subdivided_interval_p_time_rhs_component(
                p_interval,
                tuple(face_box),
                b_interval,
                3,
                subdivisions,
                p_subdivisions,
            )
        except ZeroDivisionError as exc:
            return {
                "status": "failed",
                "failure": str(exc),
                "candidate_A": candidate_a,
                "p_start": p_start,
                "p_min": p_min,
                "p_step": p_step,
                "box_low": list(box_low),
                "box_high": list(box_high),
                "x3_wall": x3_wall,
                "source_box_contained": source_contained,
                "source_below_wall": source_below_wall,
                "source_x2_floor": source_x2_floor,
                "x3_zero_margin": x3_zero_margin,
                "small_p_tail": small_p_tail,
                "failing_slice": {"p_interval": [p_lo, p_hi]},
                "conditional": "late_box_and_x2_floor_are_preserved",
            }
        # For decreasing p, an upper wall U=x3_wall is inward if 0 <= G_3.
        margin = rhs_low
        if margin < worst_margin:
            worst_margin = margin
            worst_slice = {
                "p_interval": [p_lo, p_hi],
                "rhs_lower": rhs_low,
                "rhs_upper": rhs_high,
            }
        if margin < 0.0:
            return {
                "status": "failed",
                "candidate_A": candidate_a,
                "b_interval": [-beta, beta],
                "p_start": p_start,
                "p_min": p_min,
                "p_step": p_step,
                "box_low": list(box_low),
                "box_high": list(box_high),
                "x3_wall": x3_wall,
                "source_box": {"low": list(source_box_low), "high": list(source_box_high)},
                "source_box_contained": source_contained,
                "source_below_wall": source_below_wall,
                "source_x2_floor": source_x2_floor,
                "x3_zero_margin": x3_zero_margin,
                "small_p_tail": small_p_tail,
                "worst_margin": worst_margin,
                "failing_slice": worst_slice,
                "conditional": "late_box_and_x2_floor_are_preserved",
            }
        p_hi = p_lo

    status = "certified_conditional"
    if (
        not source_contained
        or not source_below_wall
        or not source_x2_floor
        or p_prime_small_p_margin <= 0.0
        or x3_prime_upper_at_p_min >= 0.0
    ):
        status = "failed"
    return {
        "status": status,
        "candidate_A": candidate_a,
        "b_interval": [-beta, beta],
        "p_start": p_start,
        "p_min": p_min,
        "p_step": p_step,
        "box_low": list(box_low),
        "box_high": list(box_high),
        "x3_wall": x3_wall,
        "source_box": {"low": list(source_box_low), "high": list(source_box_high)},
        "source_box_contained": source_contained,
        "source_below_wall": source_below_wall,
        "source_x2_floor": source_x2_floor,
        "x3_zero_margin": x3_zero_margin,
        "small_p_tail": small_p_tail,
        "worst_margin": worst_margin,
        "worst_slice": worst_slice,
        "subdivisions": list(subdivisions),
        "p_subdivisions": p_subdivisions,
        "conditional": "late_box_and_x2_floor_are_preserved",
    }


def late_tail_closure_certificate(
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
) -> dict:
    """Compose the current conditional late-tail exclusion certificate.

    A successful result certifies that every trajectory in the p=0.25 frontier
    box stays away from standard K- terminal closure through the remaining tail.
    It is still conditional on the p=0.25 frontier box containing the true
    scaled trajectories for every |a| >= candidate_a.
    """
    continuation = p_tube_frontier_continuation_certificate(candidate_a=candidate_a)
    if continuation["status"] != "certified":
        return {
            "status": "failed",
            "candidate_A": candidate_a,
            "stage": "frontier_continuation",
            "frontier_continuation": continuation,
            "conditional": "p_0_25_frontier_box_contains_true_state",
        }

    piecewise = piecewise_affine_p_corridor_certificate(candidate_a=candidate_a)
    if piecewise["status"] != "certified":
        return {
            "status": "failed",
            "candidate_A": candidate_a,
            "stage": "piecewise_corridor",
            "frontier_continuation": continuation,
            "piecewise_corridor": piecewise,
            "conditional": "p_0_25_frontier_box_contains_true_state",
        }

    terminal = terminal_barrier_takeover_certificate(
        p_start=DEFAULT_LATE_TAIL_TAKEOVER_START,
        p_min=DEFAULT_TERMINAL_TAKEOVER_P_MIN,
        p_step=DEFAULT_TERMINAL_TAKEOVER_P_STEP,
        candidate_a=candidate_a,
        box_low=DEFAULT_LATE_TAIL_TAKEOVER_BOX_LOW,
        box_high=DEFAULT_LATE_TAIL_TAKEOVER_BOX_HIGH,
        x3_wall=DEFAULT_LATE_TAIL_TAKEOVER_X3_WALL,
        source_box_low=DEFAULT_PIECEWISE_CORRIDOR_KNOTS[-1][1],
        source_box_high=DEFAULT_PIECEWISE_CORRIDOR_KNOTS[-1][2],
    )
    if terminal["status"] != "certified_conditional":
        return {
            "status": "failed",
            "candidate_A": candidate_a,
            "stage": "terminal_takeover",
            "frontier_continuation": continuation,
            "piecewise_corridor": piecewise,
            "terminal_takeover": terminal,
            "conditional": "p_0_25_frontier_box_contains_true_state",
        }

    return {
        "status": "certified_conditional",
        "candidate_A": candidate_a,
        "frontier_continuation": continuation,
        "piecewise_corridor": piecewise,
        "terminal_takeover": terminal,
        "certified_from_p": DEFAULT_P_TUBE_END,
        "certified_to_p": 0.0,
        "x3_wall": DEFAULT_LATE_TAIL_TAKEOVER_X3_WALL,
        "conditional": "p_0_25_frontier_box_contains_true_state",
    }


def broad_tail_closure_certificate(
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
) -> dict:
    """Compose the current broad-frontier tail certificate.

    This closes the tail from the hybrid ``p=0.325`` start slice to terminal:
    hybrid handoff, automatic p-time barrier corridor, then the terminal
    ``x3=-0.6`` wall.  It remains conditional on validating the initial
    ``p=0.325`` slice from the original scaled IVP.
    """
    handoff = hybrid_p_frontier_handoff_certificate(candidate_a=candidate_a)
    if handoff["status"] != "certified_conditional":
        return {
            "status": "failed",
            "candidate_A": candidate_a,
            "stage": "hybrid_handoff",
            "hybrid_handoff": handoff,
            "conditional": "p_0_325_start_slice_box_contains_true_state",
        }

    frontier_box = handoff["frontier_box"]
    automatic = automatic_p_barrier_corridor_certificate(
        start_p=handoff["frontier_p"],
        end_p=DEFAULT_BROAD_TAIL_AUTOMATIC_END_P,
        source_box_low=tuple(frontier_box["low"]),
        source_box_high=tuple(frontier_box["high"]),
        candidate_a=candidate_a,
    )
    if automatic["status"] != "certified":
        return {
            "status": "failed",
            "candidate_A": candidate_a,
            "stage": "automatic_corridor",
            "hybrid_handoff": handoff,
            "automatic_corridor": automatic,
            "conditional": "p_0_325_start_slice_box_contains_true_state",
        }

    terminal = terminal_barrier_takeover_certificate(
        p_start=DEFAULT_BROAD_TAIL_AUTOMATIC_END_P,
        p_min=DEFAULT_TERMINAL_TAKEOVER_P_MIN,
        p_step=DEFAULT_TERMINAL_TAKEOVER_P_STEP,
        candidate_a=candidate_a,
        box_low=DEFAULT_LATE_TAIL_TAKEOVER_BOX_LOW,
        box_high=DEFAULT_LATE_TAIL_TAKEOVER_BOX_HIGH,
        x3_wall=DEFAULT_LATE_TAIL_TAKEOVER_X3_WALL,
        source_box_low=tuple(automatic["end_box"]["low"]),
        source_box_high=tuple(automatic["end_box"]["high"]),
    )
    if terminal["status"] != "certified_conditional":
        return {
            "status": "failed",
            "candidate_A": candidate_a,
            "stage": "terminal_takeover",
            "hybrid_handoff": handoff,
            "automatic_corridor": automatic,
            "terminal_takeover": terminal,
            "conditional": "p_0_325_start_slice_box_contains_true_state",
        }

    return {
        "status": "certified_conditional",
        "candidate_A": candidate_a,
        "hybrid_handoff": handoff,
        "automatic_corridor": automatic,
        "terminal_takeover": terminal,
        "certified_from_p": DEFAULT_HYBRID_HANDOFF_START_P,
        "certified_to_p": 0.0,
        "x3_wall": DEFAULT_LATE_TAIL_TAKEOVER_X3_WALL,
        "conditional": "p_0_325_start_slice_box_contains_true_state",
    }


def support_tail_closure_certificate(
    candidate_a: float = DEFAULT_TUBE_CANDIDATE_A,
    support_radius0: tuple[float, float, float, float] = DEFAULT_SUPPORT_TUBE_RADIUS,
    bridge_after_time: float = DEFAULT_HYBRID_HANDOFF_BRIDGE_AFTER_TIME,
    bridge_step_size: float = 1e-4,
) -> dict:
    """Compose the support-time-to-terminal conditional certificate."""
    start_slice = p_start_slice_from_support_certificate(
        candidate_a=candidate_a,
        support_radius0=support_radius0,
        after_time=bridge_after_time,
        step_size=bridge_step_size,
    )
    if start_slice["status"] != "certified_conditional":
        return {
            "status": "failed",
            "candidate_A": candidate_a,
            "stage": "p_start_slice_bridge",
            "p_start_slice_bridge": start_slice,
            "conditional": "support_start_box_contains_true_state",
        }

    broad_tail = broad_tail_closure_certificate(candidate_a=candidate_a)
    if broad_tail["status"] != "certified_conditional":
        return {
            "status": "failed",
            "candidate_A": candidate_a,
            "stage": "broad_tail",
            "p_start_slice_bridge": start_slice,
            "broad_tail": broad_tail,
            "conditional": "support_start_box_contains_true_state",
        }

    return {
        "status": "certified_conditional",
        "candidate_A": candidate_a,
        "p_start_slice_bridge": start_slice,
        "broad_tail": broad_tail,
        "certified_from_time": DEFAULT_SUPPORT_TIME,
        "certified_to_p": 0.0,
        "conditional": "support_start_box_contains_true_state",
    }


def perturbation_sample_report(
    candidate_a: float = DEFAULT_CANDIDATE_A,
    sample_times: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.3, 3.5),
) -> dict:
    """Sample finite-a perturbation coefficients along the limiting trajectory."""
    maxima = [[0.0, 0.0, 0.0, 0.0] for _ in range(3)]
    samples = []
    for time in sample_times:
        state = limiting_state_at(time)
        coeffs = finite_a_error_coefficients(time, state)
        for order, values in enumerate(coeffs):
            for index, value in enumerate(values):
                maxima[order][index] = max(maxima[order][index], abs(value))
        samples.append(
            {
                "time": time,
                "state": list(state),
                "error_coefficients": [list(values) for values in coeffs],
            }
        )
    candidate_bound = [
        maxima[0][index] / candidate_a
        + maxima[1][index] / candidate_a**2
        + maxima[2][index] / candidate_a**3
        for index in range(4)
    ]
    return {
        "candidate_A": candidate_a,
        "sample_times": list(sample_times),
        "max_abs_R1_R2_R3_by_component": maxima,
        "sampled_rhs_error_bound_at_candidate_A": candidate_bound,
        "samples": samples,
    }


def crossing_payload(crossing: ScaledCrossing) -> dict:
    """Return a JSON-ready crossing payload."""
    x0, x1, x2, x3 = crossing.x
    payload = {
        "source": crossing.source,
        "a": crossing.a,
        "time": crossing.time,
        "x0": x0,
        "x1": x1,
        "x2_tail_defect": x2,
        "x3_auxiliary_defect": x3,
        "step_size": crossing.step_size,
        "status": crossing.status,
    }
    if crossing.a is not None and crossing.status == "crossed":
        a = crossing.a
        f1 = crossing.time**4 * x1
        payload["u_f2_over_f1"] = a**3 * x2 / f1
        payload["v_f3_over_f1"] = a * crossing.time**2 * x3 / f1
    return payload


def build_report(
    a_values: tuple[float, ...] = (-500.0, -250.0, 250.0, 500.0),
    step_size: float = DEFAULT_STEP,
) -> dict:
    """Return a report for the limiting and selected exact scaled crossings."""
    limit = first_scaled_crossing("limit", step_size=step_size)
    exact = [first_scaled_crossing("exact", a, step_size=step_size) for a in a_values]
    support = limiting_state_at(DEFAULT_SUPPORT_TIME)
    riccati = riccati_integral_to_crossing()
    terminal_barrier = terminal_barrier_report(step_size=step_size)
    perturbation = perturbation_sample_report()
    return {
        "version": TAIL_DEFECT_VERSION,
        "defect": "X2(a)=h2(T_a)/a^3 at the first x0=0 crossing; closure requires X2(a)=0",
        "limiting_crossing": crossing_payload(limit),
        "exact_crossings": [crossing_payload(item) for item in exact],
        "sign_support": {
            "support_time": DEFAULT_SUPPORT_TIME,
            "support_state": list(support),
            "x2_boundary_derivative_at_support": x2_boundary_derivative(DEFAULT_SUPPORT_TIME, support[0]),
            "x3_minus_0_3_boundary_derivative_at_support": x3_boundary_derivative(
                DEFAULT_SUPPORT_TIME,
                support[0],
                support[1],
                support[2],
            ),
            "riccati": riccati,
        },
        "terminal_barrier": terminal_barrier,
        "finite_a_perturbation": perturbation,
        "proof_strategy": [
            "Rewrite the exact scaled ODE as x'=F_infinity(t,x)+O(1/a) on compact subintervals before x0=0.",
            "Prove the limiting IVP has first crossing T_infinity near 3.598 with x2(T_infinity)>0.",
            "Use the exact x3=0 inward-barrier condition x1>216/t^4 as an auxiliary terminal obstruction.",
            "Use p=x0 as the late-tail independent variable to avoid the terminal t-time blow-up.",
            "Use continuous dependence and transversality of z=x0^5 at T_infinity to get nonzero terminal defects for |a|>=A.",
            "Since standard K- closure needs f2(T_a)=0 and x3(T_a)=0, this excludes all sufficiently large |a|.",
        ],
    }


def _load_taylor_bridge_certificate(path: Path) -> dict:
    """Load a Taylor bridge certificate from a full report or bare JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    bridge = payload.get("taylor_time_bridge_certificate", payload)
    if not isinstance(bridge, dict):
        raise ValueError("Taylor bridge JSON must contain an object payload")
    return bridge


def _print_taylor_progress(event: dict) -> None:
    """Print one Taylor bridge progress line to stderr."""
    print(
        "Taylor bridge progress: "
        f"stage={event['stage_index']} "
        f"t={event['certified_until']:.9g}/{event['end_time']:.9g} "
        f"blocks={event['blocks_certified']} "
        f"attempts={event['tuning_attempt_count']} "
        f"worst_margin={event['worst_margin']:.9g} "
        f"width={event['current_width']}",
        file=sys.stderr,
        flush=True,
    )


def _print_taylor_restart_progress(event: dict) -> None:
    """Print one Taylor restart-chain progress line to stderr."""
    print(
        "Taylor restart progress: "
        f"t={event['certified_until']:.9g}/{event['end_time']:.9g} "
        f"segments={event['segments_certified']} "
        f"blocks={event['blocks_certified']} "
        f"attempts={event['tuning_attempt_count']} "
        f"worst_margin={event['worst_margin']:.9g} "
        f"width={event['current_width']}",
        file=sys.stderr,
        flush=True,
    )


def _print_adaptive_union_progress(event: dict) -> None:
    """Print one adaptive p-union progress line to stderr."""
    print(
        "Adaptive p-union progress: "
        f"processed={event['processed']} "
        f"certified={event['certified_leaf_boxes']} "
        f"queued={event['queued_boxes']} "
        f"splits={event['split_count']} "
        f"attempts={event['tuning_attempt_count']} "
        f"worst_margin={event['worst_margin']:.9g}",
        file=sys.stderr,
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    """Print tail-defect diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-values", default="-500,-250,250,500")
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP)
    parser.add_argument("--tube-check", action="store_true", help="run the conditional interval moving-tube verifier")
    parser.add_argument("--tube-a", type=float, default=DEFAULT_TUBE_CANDIDATE_A)
    parser.add_argument("--tube-start", type=float, default=DEFAULT_SUPPORT_TIME)
    parser.add_argument("--tube-end", type=float, default=DEFAULT_TERMINAL_BARRIER_TIME)
    parser.add_argument("--tube-step", type=float, default=1e-4)
    parser.add_argument("--tube-subdivisions", default="1,1,1,1")
    parser.add_argument("--tube-time-subdivisions", type=int, default=1)
    parser.add_argument("--segmented-tube-check", action="store_true", help="run the segmented conditional tube verifier")
    parser.add_argument(
        "--tuned-tube-check",
        action="store_true",
        help="run the centered t-time tube verifier with automatic local profile tuning",
    )
    parser.add_argument(
        "--restart-tuned-chain-check",
        action="store_true",
        help="compose tuned t-time tubes with centered restart containment boxes",
    )
    parser.add_argument("--restart-interval", type=float, default=0.05)
    parser.add_argument("--tube-block-steps", type=int, default=10)
    parser.add_argument(
        "--tuned-tube-initial-growth",
        default=",".join(str(value) for value in DEFAULT_TUNED_TUBE_INITIAL_GROWTH),
    )
    parser.add_argument(
        "--tuned-tube-max-growth",
        default=",".join(str(value) for value in DEFAULT_TUNED_TUBE_MAX_GROWTH),
    )
    parser.add_argument("--tuned-tube-growth-factor", type=float, default=DEFAULT_TUNED_TUBE_GROWTH_FACTOR)
    parser.add_argument("--tuned-tube-max-attempts", type=int, default=DEFAULT_TUNED_TUBE_MAX_ATTEMPTS)
    parser.add_argument("--p-tube-check", action="store_true", help="run the segmented p=x0 conditional tube verifier")
    parser.add_argument(
        "--tuned-p-tube-check",
        action="store_true",
        help="run the p=x0 tube verifier with automatic local profile tuning",
    )
    parser.add_argument(
        "--staged-union-p-tube-check",
        action="store_true",
        help="run the staged finite-union p=x0 tube verifier",
    )
    parser.add_argument(
        "--adaptive-union-p-tube-check",
        action="store_true",
        help="continue a finite-union p=x0 tube by adaptively splitting failed boxes",
    )
    parser.add_argument(
        "--adaptive-carried-c-union-p-tube-check",
        action="store_true",
        help="continue a finite-union p=x0 tube while carrying C=x1*x2-p^2*x3/6 as a fifth variable",
    )
    parser.add_argument(
        "--sampled-carried-c-p-tube-check",
        action="store_true",
        help="certify a narrow sample-centered carried-C p=x0 tube for the b in [-1/A,1/A] family",
    )
    parser.add_argument(
        "--carried-c-p-tube-from-box-check",
        action="store_true",
        help="continue a tuned carried-C p=x0 tube from a saved five-dimensional source box",
    )
    parser.add_argument(
        "--automatic-carried-c-p-corridor-check",
        action="store_true",
        help="run the automatic affine p=x0 corridor while carrying C=x1*x2-p^2*x3/6",
    )
    parser.add_argument(
        "--carried-c-p-wall-check",
        action="store_true",
        help="check a constant p=x0 wall using the carried-C p-time denominator",
    )
    parser.add_argument("--p-tube-start", type=float, default=DEFAULT_P_TUBE_START)
    parser.add_argument("--p-tube-end", type=float, default=DEFAULT_P_TUBE_END)
    parser.add_argument("--p-tube-entry-time", type=float, default=DEFAULT_P_TUBE_ENTRY_TIME)
    parser.add_argument("--p-tube-step", type=float, default=DEFAULT_P_TUBE_STEP)
    parser.add_argument("--p-tube-block-steps", type=int, default=1)
    parser.add_argument("--p-tube-subdivisions", default="1,1,1,1")
    parser.add_argument("--carried-c-p-tube-subdivisions", default="1,1,1,1,1")
    parser.add_argument(
        "--carried-c-split-x3-on-x0-failure",
        action="store_true",
        help="also split x3 when adaptive carried-C certification fails on a p=x0 face",
    )
    parser.add_argument(
        "--carried-c-split-x3-on-x2-failure",
        action="store_true",
        help="also split x3 when adaptive carried-C certification fails on an x2 face",
    )
    parser.add_argument("--p-tube-p-subdivisions", type=int, default=1)
    parser.add_argument("--p-tube-asymmetric-profiles", action="store_true")
    parser.add_argument(
        "--p-tube-cancellation-prime",
        action="store_true",
        help="sharpen p-time denominators with C=x1*x2-p^2*x3/6 lower bounds",
    )
    parser.add_argument(
        "--tuned-p-tube-initial-growth",
        default=",".join(str(value) for value in DEFAULT_TUNED_P_TUBE_INITIAL_GROWTH),
    )
    parser.add_argument(
        "--tuned-p-tube-max-growth",
        default=",".join(str(value) for value in DEFAULT_TUNED_P_TUBE_MAX_GROWTH),
    )
    parser.add_argument("--carried-c-p-tube-initial-growth", default="0.05,1.0,0.01,0.1,0.1")
    parser.add_argument("--carried-c-p-tube-max-growth", default="20,200,2,50,10")
    parser.add_argument("--carried-c-p-tube-start", type=float, default=DEFAULT_CARRIED_C_P_CORRIDOR_START)
    parser.add_argument("--carried-c-p-tube-end", type=float, default=DEFAULT_CARRIED_C_P_CORRIDOR_END)
    parser.add_argument("--carried-c-p-tube-step", type=float, default=DEFAULT_CARRIED_C_P_CORRIDOR_STEP)
    parser.add_argument(
        "--carried-c-p-tube-source-json",
        default=None,
        help="JSON certificate with end_box_5d/end_hull_5d/leaf_boxes_5d to seed a tuned carried-C p-tube",
    )
    parser.add_argument("--sampled-carried-c-p-tube-start", type=float, default=DEFAULT_SAMPLED_CARRIED_C_P_TUBE_START)
    parser.add_argument("--sampled-carried-c-p-tube-end", type=float, default=DEFAULT_SAMPLED_CARRIED_C_P_TUBE_END)
    parser.add_argument("--sampled-carried-c-p-tube-entry-time", type=float, default=DEFAULT_SAMPLED_CARRIED_C_P_TUBE_ENTRY_TIME)
    parser.add_argument("--sampled-carried-c-p-tube-step", type=float, default=DEFAULT_SAMPLED_CARRIED_C_P_TUBE_STEP)
    parser.add_argument(
        "--sampled-carried-c-p-tube-radius",
        default=",".join(str(value) for value in DEFAULT_SAMPLED_CARRIED_C_P_TUBE_RADIUS),
    )
    parser.add_argument(
        "--sampled-carried-c-p-tube-subdivisions",
        default=",".join(str(value) for value in DEFAULT_SAMPLED_CARRIED_C_P_TUBE_SUBDIVISIONS),
    )
    parser.add_argument(
        "--sampled-carried-c-p-tube-profile-set",
        choices=tuple(SAMPLED_CARRIED_C_P_TUBE_PROFILE_SETS),
        default="robust",
        help="named carried-C sampled tube profile list; tight gives the sharper p=0.3255 diagnostic box",
    )
    parser.add_argument("--sampled-carried-c-p-tube-progress-every", type=int, default=0)
    parser.add_argument("--sampled-carried-c-p-tube-seed-step", type=float, default=5e-5)
    parser.add_argument("--carried-c-p-corridor-start", type=float, default=DEFAULT_CARRIED_C_P_CORRIDOR_START)
    parser.add_argument("--carried-c-p-corridor-end", type=float, default=DEFAULT_CARRIED_C_P_CORRIDOR_END)
    parser.add_argument("--carried-c-p-corridor-step", type=float, default=DEFAULT_CARRIED_C_P_CORRIDOR_STEP)
    parser.add_argument(
        "--carried-c-p-corridor-safety",
        default=",".join(str(value) for value in DEFAULT_CARRIED_C_P_CORRIDOR_SAFETY),
    )
    parser.add_argument(
        "--carried-c-p-corridor-subdivisions",
        default=",".join(str(value) for value in DEFAULT_CARRIED_C_P_CORRIDOR_SUBDIVISIONS),
    )
    parser.add_argument(
        "--carried-c-p-corridor-p-subdivisions",
        type=int,
        default=DEFAULT_CARRIED_C_P_CORRIDOR_P_SUBDIVISIONS,
    )
    parser.add_argument(
        "--carried-c-p-corridor-source-low",
        default=",".join(str(value) for value in DEFAULT_CARRIED_C_P_CORRIDOR_SOURCE_LOW),
    )
    parser.add_argument(
        "--carried-c-p-corridor-source-high",
        default=",".join(str(value) for value in DEFAULT_CARRIED_C_P_CORRIDOR_SOURCE_HIGH),
    )
    parser.add_argument(
        "--carried-c-p-corridor-source-json",
        default=None,
        help="JSON certificate with end_box_5d/end_hull_5d/leaf_boxes_5d to seed the carried-C corridor",
    )
    parser.add_argument("--carried-c-p-wall-start", type=float, default=DEFAULT_CARRIED_C_P_WALL_START)
    parser.add_argument("--carried-c-p-wall-end", type=float, default=DEFAULT_CARRIED_C_P_WALL_END)
    parser.add_argument("--carried-c-p-wall-step", type=float, default=DEFAULT_CARRIED_C_P_WALL_STEP)
    parser.add_argument(
        "--carried-c-p-wall-box-low",
        default=",".join(str(value) for value in DEFAULT_CARRIED_C_P_WALL_BOX_LOW),
    )
    parser.add_argument(
        "--carried-c-p-wall-box-high",
        default=",".join(str(value) for value in DEFAULT_CARRIED_C_P_WALL_BOX_HIGH),
    )
    parser.add_argument("--carried-c-p-wall-component", type=int, default=2)
    parser.add_argument("--carried-c-p-wall-side", choices=("lower", "upper"), default="lower")
    parser.add_argument("--carried-c-p-wall-value", type=float, default=0.0)
    parser.add_argument(
        "--carried-c-p-wall-subdivisions",
        default=",".join(str(value) for value in DEFAULT_CARRIED_C_P_WALL_SUBDIVISIONS),
    )
    parser.add_argument(
        "--carried-c-p-wall-p-subdivisions",
        type=int,
        default=DEFAULT_CARRIED_C_P_WALL_P_SUBDIVISIONS,
    )
    parser.add_argument(
        "--carried-c-p-wall-source-json",
        default=None,
        help="optional JSON certificate used only to report source-box containment for the wall check",
    )
    parser.add_argument("--tuned-p-tube-max-attempts", type=int, default=DEFAULT_TUNED_P_TUBE_MAX_ATTEMPTS)
    parser.add_argument(
        "--staged-union-p-tube-source-low",
        default=",".join(str(value) for value in DEFAULT_STAGED_UNION_P_TUBE_SOURCE_LOW),
    )
    parser.add_argument(
        "--staged-union-p-tube-source-high",
        default=",".join(str(value) for value in DEFAULT_STAGED_UNION_P_TUBE_SOURCE_HIGH),
    )
    parser.add_argument(
        "--staged-union-p-tube-stages",
        default=";".join(
            f"{target}:{','.join(str(value) for value in splits)}"
            for target, splits in DEFAULT_STAGED_UNION_P_TUBE_STAGES
        ),
        help="semicolon-separated p_target:s0,s1,s2,s3 staged split specs",
    )
    parser.add_argument(
        "--adaptive-union-p-tube-source-json",
        default=DEFAULT_ADAPTIVE_UNION_P_TUBE_SOURCE_JSON,
        help="JSON certificate containing leaf_boxes for adaptive continuation",
    )
    parser.add_argument("--adaptive-union-max-depth", type=int, default=DEFAULT_ADAPTIVE_UNION_MAX_DEPTH)
    parser.add_argument(
        "--adaptive-union-max-leaf-boxes",
        type=int,
        default=DEFAULT_ADAPTIVE_UNION_MAX_LEAF_BOXES,
    )
    parser.add_argument(
        "--adaptive-union-max-processed-boxes",
        type=int,
        default=DEFAULT_ADAPTIVE_UNION_MAX_PROCESSED_BOXES,
    )
    parser.add_argument(
        "--adaptive-union-progress-every",
        type=int,
        default=0,
        help="print adaptive union progress to stderr every N processed boxes",
    )
    parser.add_argument("--p-corridor-check", action="store_true", help="run the affine p-time barrier corridor verifier")
    parser.add_argument("--p-corridor-start", type=float, default=DEFAULT_P_CORRIDOR_START)
    parser.add_argument("--p-corridor-end", type=float, default=DEFAULT_P_CORRIDOR_END)
    parser.add_argument("--p-corridor-step", type=float, default=DEFAULT_P_CORRIDOR_STEP)
    parser.add_argument(
        "--p-corridor-lower-start",
        default=",".join(str(value) for value in DEFAULT_P_CORRIDOR_LOWER_START),
    )
    parser.add_argument(
        "--p-corridor-upper-start",
        default=",".join(str(value) for value in DEFAULT_P_CORRIDOR_UPPER_START),
    )
    parser.add_argument(
        "--p-corridor-lower-slope",
        default=",".join(str(value) for value in DEFAULT_P_CORRIDOR_LOWER_SLOPE),
    )
    parser.add_argument(
        "--p-corridor-upper-slope",
        default=",".join(str(value) for value in DEFAULT_P_CORRIDOR_UPPER_SLOPE),
    )
    parser.add_argument("--p-corridor-subdivisions", default="2,2,2,2")
    parser.add_argument("--p-corridor-p-subdivisions", type=int, default=2)
    parser.add_argument("--p-corridor-tune", action="store_true", help="scan the active affine corridor slope pair")
    parser.add_argument(
        "--p-corridor-tune-x2-slopes",
        default=",".join(str(value) for value in DEFAULT_P_CORRIDOR_TUNE_X2_SLOPES),
    )
    parser.add_argument(
        "--p-corridor-tune-x1-upper-slopes",
        default=",".join(str(value) for value in DEFAULT_P_CORRIDOR_TUNE_X1_UPPER_SLOPES),
    )
    parser.add_argument("--p-corridor-tune-max-runs", type=int, default=None)
    parser.add_argument(
        "--terminal-takeover-check",
        action="store_true",
        help="check the conditional late x3-wall terminal barrier takeover",
    )
    parser.add_argument(
        "--x3-zero-wall-check",
        action="store_true",
        help="certify the ordinary-time x3=0 one-way wall on a coarse late box",
    )
    parser.add_argument(
        "--x2-zero-factor-check",
        action="store_true",
        help="certify the exact finite-A factorization of x2' on the x2=0 wall",
    )
    parser.add_argument(
        "--late-x3-descent-check",
        action="store_true",
        help="certify the late ordinary-time bridge into x3<0 and x0<0.4",
    )
    parser.add_argument(
        "--x3-zero-wall-time-range",
        default=",".join(str(value) for value in DEFAULT_X3_ZERO_WALL_TIME_RANGE),
    )
    parser.add_argument(
        "--x3-zero-wall-x0-range",
        default=",".join(str(value) for value in DEFAULT_X3_ZERO_WALL_X0_RANGE),
    )
    parser.add_argument(
        "--x3-zero-wall-x1-range",
        default=",".join(str(value) for value in DEFAULT_X3_ZERO_WALL_X1_RANGE),
    )
    parser.add_argument(
        "--x3-zero-wall-x2-range",
        default=",".join(str(value) for value in DEFAULT_X3_ZERO_WALL_X2_RANGE),
    )
    parser.add_argument(
        "--x3-zero-wall-subdivisions",
        default=",".join(str(value) for value in DEFAULT_X3_ZERO_WALL_SUBDIVISIONS),
    )
    parser.add_argument(
        "--x3-zero-wall-time-subdivisions",
        type=int,
        default=DEFAULT_X3_ZERO_WALL_TIME_SUBDIVISIONS,
    )
    parser.add_argument(
        "--x2-zero-factor-p-range",
        default=",".join(str(value) for value in DEFAULT_X2_ZERO_FACTOR_P_RANGE),
    )
    parser.add_argument(
        "--x2-zero-factor-time-range",
        default=",".join(str(value) for value in DEFAULT_X2_ZERO_FACTOR_TIME_RANGE),
    )
    parser.add_argument(
        "--x2-zero-factor-x3-range",
        default=",".join(str(value) for value in DEFAULT_X2_ZERO_FACTOR_X3_RANGE),
    )
    parser.add_argument("--late-x3-descent-start", type=float, default=DEFAULT_LATE_X3_DESCENT_START)
    parser.add_argument("--late-x3-descent-end", type=float, default=DEFAULT_LATE_X3_DESCENT_END)
    parser.add_argument("--late-x3-descent-step", type=float, default=DEFAULT_LATE_X3_DESCENT_STEP)
    parser.add_argument(
        "--late-x3-descent-radius0",
        default=",".join(str(value) for value in DEFAULT_LATE_X3_DESCENT_RADIUS0),
    )
    parser.add_argument(
        "--late-x3-descent-safety",
        default=",".join(str(value) for value in DEFAULT_LATE_X3_DESCENT_SAFETY),
    )
    parser.add_argument("--late-x3-descent-x0-target", type=float, default=DEFAULT_LATE_X3_DESCENT_X0_TARGET)
    parser.add_argument(
        "--frontier-continuation-check",
        action="store_true",
        help="continue the certified p=0.25 frontier with subdivided p-tubes",
    )
    parser.add_argument(
        "--hybrid-handoff-check",
        action="store_true",
        help="run the hybrid p-tube/affine handoff into the broad p=0.25 frontier",
    )
    parser.add_argument(
        "--p-start-slice-bridge-check",
        action="store_true",
        help="certify the t-time support tube crossing into the p=0.325 start slice",
    )
    parser.add_argument(
        "--regular-time-corridor-check",
        action="store_true",
        help="run the automatic ordinary-time affine corridor verifier",
    )
    parser.add_argument(
        "--taylor-start-block-check",
        action="store_true",
        help="certify the first ordinary-time slab from the c2 Taylor start box",
    )
    parser.add_argument(
        "--taylor-time-bridge-check",
        action="store_true",
        help="certify the staged compact t-time bridge from the c2 Taylor start box",
    )
    parser.add_argument(
        "--taylor-frontier-continuation-check",
        action="store_true",
        help="continue the Taylor bridge with direct tuned t-time blocks",
    )
    parser.add_argument(
        "--taylor-restart-chain-check",
        action="store_true",
        help="compose the Taylor bridge with centered restart t-time tubes",
    )
    parser.add_argument(
        "--taylor-p-slice-audit",
        action="store_true",
        help="compare two high-order Taylor p-slices with the sampled p-tube start radius",
    )
    parser.add_argument(
        "--taylor-p-slice-tail-audit",
        action="store_true",
        help="estimate a formal geometric Taylor tail at the p-slice",
    )
    parser.add_argument(
        "--taylor-p-slice-interval-ratio-audit",
        action="store_true",
        help="check finite same-parity Taylor ratios using interval b-coefficients",
    )
    parser.add_argument(
        "--taylor-p-slice-cauchy-budget-audit",
        action="store_true",
        help="report Cauchy disk-bound budgets for the p-slice Taylor tail",
    )
    parser.add_argument(
        "--taylor-ratio-profile-audit",
        action="store_true",
        help="report finite same-parity ratio profiles on the p-slice and proof circle",
    )
    parser.add_argument(
        "--taylor-geometric-envelope-audit",
        action="store_true",
        help="check finite terms against a proposed same-parity geometric tail envelope",
    )
    parser.add_argument(
        "--taylor-even-parity-audit",
        action="store_true",
        help="check that odd Taylor coefficients vanish for sampled real/complex b values",
    )
    parser.add_argument(
        "--taylor-even-s-series-audit",
        action="store_true",
        help="report the even Taylor tail target as an ordinary s=t^2 series",
    )
    parser.add_argument(
        "--taylor-recurrence-forcing-audit",
        action="store_true",
        help="audit the explicit Taylor recurrence inverse and forcing ratios",
    )
    parser.add_argument(
        "--taylor-b-sensitivity-audit",
        action="store_true",
        help="measure sampled finite-b sensitivity of the Taylor handoff",
    )
    parser.add_argument(
        "--taylor-p-slice-entry-budget-audit",
        action="store_true",
        help="combine p-slice Taylor tail and finite-b event budgets against the carried-C start box",
    )
    parser.add_argument(
        "--taylor-p-slice-required-a-audit",
        action="store_true",
        help="compute the explicit A threshold implied by the conditional p-slice entry budget",
    )
    parser.add_argument(
        "--taylor-p-slice-b-cauchy-event-audit",
        action="store_true",
        help="sample complex-b Cauchy bounds for finite p-slice event perturbations",
    )
    parser.add_argument(
        "--taylor-b-cauchy-coefficient-audit",
        action="store_true",
        help="sample complex-b Cauchy bounds for finite Taylor coefficient perturbations",
    )
    parser.add_argument(
        "--taylor-support-time-audit",
        action="store_true",
        help="compare Taylor orders at the t=3.5 support time",
    )
    parser.add_argument(
        "--taylor-circle-residual-audit",
        action="store_true",
        help="sample the Taylor polynomial equation residual on the proof circle",
    )
    parser.add_argument("--taylor-start-time", type=float, default=DEFAULT_TAYLOR_START_TIME)
    parser.add_argument("--taylor-start-step", type=float, default=DEFAULT_TAYLOR_START_STEP)
    parser.add_argument("--taylor-bridge-end", type=float, default=DEFAULT_TAYLOR_BRIDGE_END)
    parser.add_argument("--taylor-frontier-end", type=float, default=DEFAULT_TAYLOR_FRONTIER_END)
    parser.add_argument("--taylor-restart-end", type=float, default=DEFAULT_TAYLOR_RESTART_CHAIN_END)
    parser.add_argument("--taylor-restart-interval", type=float, default=0.05)
    parser.add_argument("--taylor-bridge-max-attempts", type=int, default=DEFAULT_TAYLOR_BRIDGE_MAX_ATTEMPTS)
    parser.add_argument("--taylor-frontier-max-attempts", type=int, default=DEFAULT_TAYLOR_FRONTIER_MAX_ATTEMPTS)
    parser.add_argument("--taylor-restart-max-attempts", type=int, default=DEFAULT_TAYLOR_RESTART_CHAIN_MAX_ATTEMPTS)
    parser.add_argument(
        "--taylor-progress-every-blocks",
        type=int,
        default=0,
        help="print Taylor bridge progress to stderr every N certified blocks",
    )
    parser.add_argument(
        "--taylor-restart-progress-every-segments",
        type=int,
        default=0,
        help="print Taylor restart-chain progress to stderr every N certified segments",
    )
    parser.add_argument(
        "--taylor-restart-bridge-json",
        default=None,
        help="reuse a saved Taylor time-bridge JSON payload instead of recomputing the bridge",
    )
    parser.add_argument(
        "--taylor-restart-max-growth",
        default=",".join(str(value) for value in DEFAULT_TAYLOR_RESTART_CHAIN_MAX_GROWTH),
    )
    parser.add_argument(
        "--taylor-start-radius",
        default=",".join(str(value) for value in DEFAULT_TAYLOR_START_RADIUS),
    )
    parser.add_argument(
        "--taylor-start-safety",
        default=",".join(str(value) for value in DEFAULT_TAYLOR_START_SAFETY),
    )
    parser.add_argument("--taylor-p-slice-target", type=float, default=DEFAULT_SAMPLED_CARRIED_C_P_TUBE_START)
    parser.add_argument("--taylor-p-slice-low-order", type=int, default=30)
    parser.add_argument("--taylor-p-slice-high-order", type=int, default=40)
    parser.add_argument("--taylor-p-slice-tail-order", type=int, default=60)
    parser.add_argument("--taylor-p-slice-interval-order", type=int, default=70)
    parser.add_argument("--taylor-p-slice-b-samples", type=int, default=3)
    parser.add_argument("--taylor-p-slice-b-subdivisions", type=int, default=1)
    parser.add_argument("--taylor-p-slice-tail-start", type=int, default=50)
    parser.add_argument("--taylor-p-slice-ratio-start", type=int, default=45)
    parser.add_argument(
        "--taylor-p-slice-ratio-bound",
        type=float,
        default=None,
        help="optional proposed same-parity geometric ratio bound for the p-slice tail audit",
    )
    parser.add_argument("--taylor-p-slice-working-dps", type=int, default=80)
    parser.add_argument("--taylor-p-slice-tail-working-dps", type=int, default=90)
    parser.add_argument("--taylor-p-slice-interval-working-dps", type=int, default=80)
    parser.add_argument("--taylor-p-slice-time-padding", type=float, default=1e-8)
    parser.add_argument(
        "--taylor-p-slice-cauchy-radii",
        default=",".join(str(value) for value in DEFAULT_TAYLOR_P_SLICE_CAUCHY_RADII),
    )
    parser.add_argument("--taylor-p-slice-cauchy-circle-samples", type=int, default=360)
    parser.add_argument("--taylor-p-slice-cauchy-circle-tail-ratio-bound", type=float, default=0.95)
    parser.add_argument("--taylor-ratio-profile-circle-radius", type=float, default=3.5)
    parser.add_argument("--taylor-ratio-profile-circle-ratio-bound", type=float, default=0.95)
    parser.add_argument("--taylor-ratio-profile-p-slice-ratio-bound", type=float, default=0.53)
    parser.add_argument(
        "--taylor-ratio-profile-b-mode",
        choices=("grid", "limit"),
        default="grid",
        help="use the full finite-b sample grid or only the limiting b=0 germ",
    )
    parser.add_argument("--taylor-b-sensitivity-circle-radius", type=float, default=3.5)
    parser.add_argument("--taylor-b-cauchy-radius", type=float, default=1e-7)
    parser.add_argument("--taylor-b-cauchy-samples", type=int, default=8)
    parser.add_argument("--taylor-b-cauchy-outer-radius", type=float, default=None)
    parser.add_argument("--taylor-b-cauchy-outer-samples", type=int, default=None)
    parser.add_argument("--taylor-b-cauchy-enclosure-radius", type=float, default=None)
    parser.add_argument("--taylor-b-cauchy-enclosure-samples", type=int, default=None)
    parser.add_argument("--taylor-b-cauchy-time-radius", type=float, default=DEFAULT_SUPPORT_TIME)
    parser.add_argument(
        "--taylor-b-cauchy-skip-direct",
        action="store_true",
        help="skip real endpoint coefficient recomputation in the b-Cauchy audit",
    )
    parser.add_argument("--taylor-support-time", type=float, default=DEFAULT_SUPPORT_TIME)
    parser.add_argument("--taylor-support-low-order", type=int, default=60)
    parser.add_argument("--taylor-support-high-order", type=int, default=80)
    parser.add_argument("--taylor-circle-residual-radius", type=float, default=3.5)
    parser.add_argument("--taylor-circle-residual-samples", type=int, default=120)
    parser.add_argument(
        "--taylor-p-slice-radius",
        default=",".join(str(value) for value in DEFAULT_SAMPLED_CARRIED_C_P_TUBE_RADIUS),
    )
    parser.add_argument("--regular-time-start", type=float, default=DEFAULT_REGULAR_TIME_AUTOMATIC_START)
    parser.add_argument("--regular-time-end", type=float, default=DEFAULT_REGULAR_TIME_AUTOMATIC_END)
    parser.add_argument("--regular-time-step", type=float, default=DEFAULT_REGULAR_TIME_AUTOMATIC_STEP)
    parser.add_argument(
        "--regular-time-radius0",
        default=",".join(str(value) for value in DEFAULT_REGULAR_TIME_AUTOMATIC_RADIUS0),
    )
    parser.add_argument(
        "--regular-time-safety",
        default=",".join(str(value) for value in DEFAULT_REGULAR_TIME_AUTOMATIC_SAFETY),
    )
    parser.add_argument(
        "--regular-time-subdivisions",
        default=",".join(str(value) for value in DEFAULT_REGULAR_TIME_AUTOMATIC_SUBDIVISIONS),
    )
    parser.add_argument(
        "--regular-time-time-subdivisions",
        type=int,
        default=DEFAULT_REGULAR_TIME_AUTOMATIC_TIME_SUBDIVISIONS,
    )
    parser.add_argument(
        "--piecewise-corridor-check",
        action="store_true",
        help="check the draft piecewise affine p-corridor from the continued frontier",
    )
    parser.add_argument(
        "--late-tail-closure-check",
        action="store_true",
        help="compose the current p=0.25-to-terminal conditional exclusion certificate",
    )
    parser.add_argument(
        "--broad-tail-closure-check",
        action="store_true",
        help="compose the broad hybrid-frontier tail certificate down to terminal",
    )
    parser.add_argument(
        "--support-tail-closure-check",
        action="store_true",
        help="compose the support-time bridge and broad-tail closure certificate",
    )
    parser.add_argument(
        "--support-tail-support-radius",
        default=",".join(str(value) for value in DEFAULT_SUPPORT_TUBE_RADIUS),
        help="comma-separated t=3.5 support-box radii for support-tail closure",
    )
    parser.add_argument(
        "--support-tail-bridge-after-time",
        type=float,
        default=DEFAULT_HYBRID_HANDOFF_BRIDGE_AFTER_TIME,
        help="end time for the short support-to-p-start bridge",
    )
    parser.add_argument(
        "--support-tail-bridge-step",
        type=float,
        default=1e-4,
        help="step size for the short support-to-p-start bridge",
    )
    parser.add_argument("--terminal-takeover-p-min", type=float, default=DEFAULT_TERMINAL_TAKEOVER_P_MIN)
    parser.add_argument("--terminal-takeover-p-step", type=float, default=DEFAULT_TERMINAL_TAKEOVER_P_STEP)
    parser.add_argument(
        "--terminal-takeover-box-low",
        default=",".join(str(value) for value in DEFAULT_TERMINAL_TAKEOVER_BOX_LOW),
    )
    parser.add_argument(
        "--terminal-takeover-box-high",
        default=",".join(str(value) for value in DEFAULT_TERMINAL_TAKEOVER_BOX_HIGH),
    )
    parser.add_argument("--terminal-takeover-x3-wall", type=float, default=DEFAULT_TERMINAL_TAKEOVER_X3_WALL)
    parser.add_argument("--terminal-takeover-subdivisions", default="2,2,2,1")
    parser.add_argument("--terminal-takeover-p-subdivisions", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    def float_tuple(text: str, expected: int, name: str) -> tuple[float, ...]:
        values = tuple(float(item) for item in text.split(",") if item.strip())
        if len(values) != expected:
            raise ValueError(f"{name} must contain {expected} comma-separated values")
        return values

    def float_list(text: str, name: str) -> tuple[float, ...]:
        values = tuple(float(item) for item in text.split(",") if item.strip())
        if not values:
            raise ValueError(f"{name} must contain at least one value")
        return values

    def staged_union_specs(text: str) -> tuple[tuple[float, tuple[int, int, int, int]], ...]:
        stages = []
        for raw_stage in text.split(";"):
            raw_stage = raw_stage.strip()
            if not raw_stage:
                continue
            if ":" not in raw_stage:
                raise ValueError("--staged-union-p-tube-stages entries must have form target:s0,s1,s2,s3")
            target_text, split_text = raw_stage.split(":", 1)
            splits = tuple(int(item) for item in split_text.split(",") if item.strip())
            if len(splits) != 4:
                raise ValueError("--staged-union-p-tube-stages split specs must contain four integers")
            stages.append((float(target_text), splits))
        if not stages:
            raise ValueError("--staged-union-p-tube-stages must contain at least one stage")
        return tuple(stages)

    a_values = tuple(float(item) for item in args.a_values.split(",") if item.strip())
    report = build_report(a_values, args.step_size)
    if args.tube_check:
        subdivisions = tuple(int(item) for item in args.tube_subdivisions.split(",") if item.strip())
        if len(subdivisions) != 4:
            raise ValueError("--tube-subdivisions must contain four comma-separated integers")
        report["moving_tube_certificate"] = moving_tube_certificate(
            start_time=args.tube_start,
            end_time=args.tube_end,
            step_size=args.tube_step,
            candidate_a=args.tube_a,
            subdivisions=subdivisions,
            time_subdivisions=args.tube_time_subdivisions,
        )
    if args.segmented_tube_check:
        subdivisions = tuple(int(item) for item in args.tube_subdivisions.split(",") if item.strip())
        if len(subdivisions) != 4:
            raise ValueError("--tube-subdivisions must contain four comma-separated integers")
        report["segmented_moving_tube_certificate"] = segmented_moving_tube_certificate(
            start_time=args.tube_start,
            end_time=args.tube_end,
            step_size=args.tube_step,
            block_steps=args.tube_block_steps,
            candidate_a=args.tube_a,
            subdivisions=subdivisions,
            time_subdivisions=args.tube_time_subdivisions,
        )
    if args.tuned_tube_check:
        subdivisions = tuple(int(item) for item in args.tube_subdivisions.split(",") if item.strip())
        if len(subdivisions) != 4:
            raise ValueError("--tube-subdivisions must contain four comma-separated integers")
        report["tuned_segmented_moving_tube_certificate"] = tuned_segmented_moving_tube_certificate(
            start_time=args.tube_start,
            end_time=args.tube_end,
            step_size=args.tube_step,
            block_steps=args.tube_block_steps,
            candidate_a=args.tube_a,
            initial_growth=float_tuple(args.tuned_tube_initial_growth, 4, "--tuned-tube-initial-growth"),
            max_growth=float_tuple(args.tuned_tube_max_growth, 4, "--tuned-tube-max-growth"),
            growth_factor=args.tuned_tube_growth_factor,
            max_attempts=args.tuned_tube_max_attempts,
            subdivisions=subdivisions,
            time_subdivisions=args.tube_time_subdivisions,
        )
    if args.restart_tuned_chain_check:
        subdivisions = tuple(int(item) for item in args.tube_subdivisions.split(",") if item.strip())
        if len(subdivisions) != 4:
            raise ValueError("--tube-subdivisions must contain four comma-separated integers")
        report["restart_tuned_time_chain_certificate"] = restart_tuned_time_chain_certificate(
            start_time=args.tube_start,
            end_time=args.tube_end,
            restart_interval=args.restart_interval,
            step_size=args.tube_step,
            candidate_a=args.tube_a,
            initial_growth=float_tuple(args.tuned_tube_initial_growth, 4, "--tuned-tube-initial-growth"),
            max_growth=float_tuple(args.tuned_tube_max_growth, 4, "--tuned-tube-max-growth"),
            growth_factor=args.tuned_tube_growth_factor,
            max_attempts=args.tuned_tube_max_attempts,
            subdivisions=subdivisions,
            time_subdivisions=args.tube_time_subdivisions,
        )
    if args.p_tube_check:
        subdivisions = tuple(int(item) for item in args.p_tube_subdivisions.split(",") if item.strip())
        if len(subdivisions) != 4:
            raise ValueError("--p-tube-subdivisions must contain four comma-separated integers")
        report["segmented_p_tube_certificate"] = segmented_p_tube_certificate(
            start_p=args.p_tube_start,
            end_p=args.p_tube_end,
            entry_time=args.p_tube_entry_time,
            step_size=args.p_tube_step,
            block_steps=args.p_tube_block_steps,
            candidate_a=args.tube_a,
            profiles=DEFAULT_ASYMMETRIC_P_TUBE_PROFILES
            if args.p_tube_asymmetric_profiles
            else DEFAULT_SEGMENTED_P_TUBE_PROFILES,
            subdivisions=subdivisions,
            p_subdivisions=args.p_tube_p_subdivisions,
        )
    if args.tuned_p_tube_check:
        subdivisions = tuple(int(item) for item in args.p_tube_subdivisions.split(",") if item.strip())
        if len(subdivisions) != 4:
            raise ValueError("--p-tube-subdivisions must contain four comma-separated integers")
        report["tuned_segmented_p_tube_certificate"] = tuned_segmented_p_tube_certificate(
            start_p=args.p_tube_start,
            end_p=args.p_tube_end,
            entry_time=args.p_tube_entry_time,
            step_size=args.p_tube_step,
            block_steps=args.p_tube_block_steps,
            candidate_a=args.tube_a,
            initial_growth=float_tuple(args.tuned_p_tube_initial_growth, 4, "--tuned-p-tube-initial-growth"),
            max_growth=float_tuple(args.tuned_p_tube_max_growth, 4, "--tuned-p-tube-max-growth"),
            growth_factor=args.tuned_tube_growth_factor,
            max_attempts=args.tuned_p_tube_max_attempts,
            subdivisions=subdivisions,
            p_subdivisions=args.p_tube_p_subdivisions,
            use_cancellation_p_prime=args.p_tube_cancellation_prime,
        )
    if args.staged_union_p_tube_check:
        subdivisions = tuple(int(item) for item in args.p_tube_subdivisions.split(",") if item.strip())
        if len(subdivisions) != 4:
            raise ValueError("--p-tube-subdivisions must contain four comma-separated integers")
        report["staged_union_p_tube_certificate"] = staged_union_p_tube_certificate(
            start_p=args.p_tube_start,
            source_box_low=float_tuple(
                args.staged_union_p_tube_source_low,
                4,
                "--staged-union-p-tube-source-low",
            ),
            source_box_high=float_tuple(
                args.staged_union_p_tube_source_high,
                4,
                "--staged-union-p-tube-source-high",
            ),
            stages=staged_union_specs(args.staged_union_p_tube_stages),
            step_size=args.p_tube_step,
            block_steps=args.p_tube_block_steps,
            candidate_a=args.tube_a,
            initial_growth=float_tuple(args.tuned_p_tube_initial_growth, 4, "--tuned-p-tube-initial-growth"),
            max_growth=float_tuple(args.tuned_p_tube_max_growth, 4, "--tuned-p-tube-max-growth"),
            growth_factor=args.tuned_tube_growth_factor,
            max_attempts=args.tuned_p_tube_max_attempts,
            subdivisions=subdivisions,
            p_subdivisions=args.p_tube_p_subdivisions,
            use_cancellation_p_prime=args.p_tube_cancellation_prime,
        )
    if args.adaptive_union_p_tube_check:
        subdivisions = tuple(int(item) for item in args.p_tube_subdivisions.split(",") if item.strip())
        if len(subdivisions) != 4:
            raise ValueError("--p-tube-subdivisions must contain four comma-separated integers")
        source_path = Path(args.adaptive_union_p_tube_source_json)
        if not source_path.exists():
            raise FileNotFoundError(
                f"adaptive union source JSON not found: {source_path}; "
                "run the staged union certificate first or pass --adaptive-union-p-tube-source-json"
            )
        source_boxes = tuple(_load_union_leaf_boxes(source_path))
        report["adaptive_union_p_tube_certificate"] = adaptive_union_p_tube_certificate(
            start_p=args.p_tube_start,
            end_p=args.p_tube_end,
            source_boxes=source_boxes,
            step_size=args.p_tube_step,
            block_steps=args.p_tube_block_steps,
            candidate_a=args.tube_a,
            initial_growth=float_tuple(args.tuned_p_tube_initial_growth, 4, "--tuned-p-tube-initial-growth"),
            max_growth=float_tuple(args.tuned_p_tube_max_growth, 4, "--tuned-p-tube-max-growth"),
            growth_factor=args.tuned_tube_growth_factor,
            max_attempts=args.tuned_p_tube_max_attempts,
            subdivisions=subdivisions,
            p_subdivisions=args.p_tube_p_subdivisions,
            max_depth=args.adaptive_union_max_depth,
            max_leaf_boxes=args.adaptive_union_max_leaf_boxes,
            max_processed_boxes=args.adaptive_union_max_processed_boxes,
            use_cancellation_p_prime=args.p_tube_cancellation_prime,
            progress_callback=_print_adaptive_union_progress
            if args.adaptive_union_progress_every > 0
            else None,
            progress_every=args.adaptive_union_progress_every,
        )
        report["adaptive_union_p_tube_certificate"]["source_json"] = str(source_path)
    if args.adaptive_carried_c_union_p_tube_check:
        subdivisions = tuple(int(item) for item in args.carried_c_p_tube_subdivisions.split(",") if item.strip())
        if len(subdivisions) != 5:
            raise ValueError("--carried-c-p-tube-subdivisions must contain five comma-separated integers")
        source_path = Path(args.adaptive_union_p_tube_source_json)
        if not source_path.exists():
            raise FileNotFoundError(
                f"adaptive carried-C union source JSON not found: {source_path}; "
                "run an earlier union certificate first or pass --adaptive-union-p-tube-source-json"
            )
        source_boxes = tuple(_load_union_leaf_boxes_for_carried_c(source_path, args.p_tube_start))
        report["adaptive_carried_c_union_p_tube_certificate"] = adaptive_carried_c_union_p_tube_certificate(
            start_p=args.p_tube_start,
            end_p=args.p_tube_end,
            source_boxes=source_boxes,
            step_size=args.p_tube_step,
            block_steps=args.p_tube_block_steps,
            candidate_a=args.tube_a,
            initial_growth=float_tuple(args.carried_c_p_tube_initial_growth, 5, "--carried-c-p-tube-initial-growth"),
            max_growth=float_tuple(args.carried_c_p_tube_max_growth, 5, "--carried-c-p-tube-max-growth"),
            growth_factor=args.tuned_tube_growth_factor,
            max_attempts=args.tuned_p_tube_max_attempts,
            subdivisions=subdivisions,
            p_subdivisions=args.p_tube_p_subdivisions,
            max_depth=args.adaptive_union_max_depth,
            max_leaf_boxes=args.adaptive_union_max_leaf_boxes,
            max_processed_boxes=args.adaptive_union_max_processed_boxes,
            split_x3_on_x0_failure=args.carried_c_split_x3_on_x0_failure,
            split_x3_on_x2_failure=args.carried_c_split_x3_on_x2_failure,
            progress_callback=_print_adaptive_union_progress
            if args.adaptive_union_progress_every > 0
            else None,
            progress_every=args.adaptive_union_progress_every,
        )
        report["adaptive_carried_c_union_p_tube_certificate"]["source_json"] = str(source_path)
    if args.sampled_carried_c_p_tube_check:
        subdivisions = tuple(
            int(item)
            for item in args.sampled_carried_c_p_tube_subdivisions.split(",")
            if item.strip()
        )
        if len(subdivisions) != 5:
            raise ValueError("--sampled-carried-c-p-tube-subdivisions must contain five comma-separated integers")

        def _print_sampled_carried_c_progress(event: dict) -> None:
            print(
                "Sampled carried-C p-tube progress: "
                f"blocks={event['blocks_certified']} "
                f"certified_to_p={event['certified_to_p']:.9g} "
                f"worst_margin={event['worst_margin']:.9g} "
                f"width_5d={event['width_5d']}",
                file=sys.stderr,
                flush=True,
            )

        report["sampled_carried_c_p_tube_certificate"] = sampled_carried_c_p_tube_certificate(
            start_p=args.sampled_carried_c_p_tube_start,
            end_p=args.sampled_carried_c_p_tube_end,
            entry_time=args.sampled_carried_c_p_tube_entry_time,
            step_size=args.sampled_carried_c_p_tube_step,
            candidate_a=args.tube_a,
            radius0=float_tuple(
                args.sampled_carried_c_p_tube_radius,
                5,
                "--sampled-carried-c-p-tube-radius",
            ),
            profiles=SAMPLED_CARRIED_C_P_TUBE_PROFILE_SETS[args.sampled_carried_c_p_tube_profile_set],
            max_attempts=args.tuned_p_tube_max_attempts,
            subdivisions=subdivisions,
            p_subdivisions=args.p_tube_p_subdivisions,
            seed_step_size=args.sampled_carried_c_p_tube_seed_step,
            progress_callback=_print_sampled_carried_c_progress
            if args.sampled_carried_c_p_tube_progress_every > 0
            else None,
            progress_every=args.sampled_carried_c_p_tube_progress_every,
        )
        report["sampled_carried_c_p_tube_certificate"]["profile_set"] = (
            args.sampled_carried_c_p_tube_profile_set
        )
    if args.carried_c_p_tube_from_box_check:
        subdivisions = tuple(int(item) for item in args.carried_c_p_tube_subdivisions.split(",") if item.strip())
        if len(subdivisions) != 5:
            raise ValueError("--carried-c-p-tube-subdivisions must contain five comma-separated integers")
        if args.carried_c_p_tube_source_json is None:
            raise ValueError("--carried-c-p-tube-from-box-check requires --carried-c-p-tube-source-json")
        source_path = Path(args.carried_c_p_tube_source_json)
        if not source_path.exists():
            raise FileNotFoundError(f"carried-C p-tube source JSON not found: {source_path}")
        source_low, source_high, source_p, source_kind = _load_carried_c_corridor_source_box(source_path)
        if source_p is not None and abs(source_p - args.carried_c_p_tube_start) > 1e-12:
            raise ValueError(
                "carried-C p-tube source JSON is at "
                f"p={source_p}, but --carried-c-p-tube-start={args.carried_c_p_tube_start}"
            )
        report["carried_c_p_tube_from_box_certificate"] = tuned_carried_c_p_tube_from_box_certificate(
            start_p=args.carried_c_p_tube_start,
            end_p=args.carried_c_p_tube_end,
            start_low=source_low,
            start_high=source_high,
            step_size=args.carried_c_p_tube_step,
            block_steps=args.p_tube_block_steps,
            candidate_a=args.tube_a,
            initial_growth=float_tuple(
                args.carried_c_p_tube_initial_growth,
                5,
                "--carried-c-p-tube-initial-growth",
            ),
            max_growth=float_tuple(
                args.carried_c_p_tube_max_growth,
                5,
                "--carried-c-p-tube-max-growth",
            ),
            growth_factor=args.tuned_tube_growth_factor,
            max_attempts=args.tuned_p_tube_max_attempts,
            subdivisions=subdivisions,
            p_subdivisions=args.p_tube_p_subdivisions,
        )
        report["carried_c_p_tube_from_box_certificate"]["source_kind"] = source_kind
        report["carried_c_p_tube_from_box_certificate"]["source_json"] = str(source_path)
    if args.automatic_carried_c_p_corridor_check:
        if args.carried_c_p_corridor_source_json is not None:
            source_path = Path(args.carried_c_p_corridor_source_json)
            if not source_path.exists():
                raise FileNotFoundError(f"carried-C p-corridor source JSON not found: {source_path}")
            source_low, source_high, source_p, source_kind = _load_carried_c_corridor_source_box(source_path)
            if source_p is not None and abs(source_p - args.carried_c_p_corridor_start) > 1e-12:
                raise ValueError(
                    "carried-C p-corridor source JSON is at "
                    f"p={source_p}, but --carried-c-p-corridor-start={args.carried_c_p_corridor_start}"
                )
        else:
            source_low = float_tuple(args.carried_c_p_corridor_source_low, 5, "--carried-c-p-corridor-source-low")
            source_high = float_tuple(args.carried_c_p_corridor_source_high, 5, "--carried-c-p-corridor-source-high")
            source_kind = "cli_source_box"
        report["automatic_carried_c_p_corridor_certificate"] = automatic_carried_c_p_barrier_corridor_certificate(
            start_p=args.carried_c_p_corridor_start,
            end_p=args.carried_c_p_corridor_end,
            source_box_low=source_low,
            source_box_high=source_high,
            step_size=args.carried_c_p_corridor_step,
            candidate_a=args.tube_a,
            safety=float_tuple(
                args.carried_c_p_corridor_safety,
                5,
                "--carried-c-p-corridor-safety",
            ),
            subdivisions=tuple(
                int(value)
                for value in float_tuple(
                    args.carried_c_p_corridor_subdivisions,
                    5,
                    "--carried-c-p-corridor-subdivisions",
                )
            ),
            p_subdivisions=args.carried_c_p_corridor_p_subdivisions,
        )
        report["automatic_carried_c_p_corridor_certificate"]["source_kind"] = source_kind
        if args.carried_c_p_corridor_source_json is not None:
            report["automatic_carried_c_p_corridor_certificate"]["source_json"] = str(source_path)
    if args.carried_c_p_wall_check:
        source_box_low = None
        source_box_high = None
        source_kind = None
        wall_box_dimension = 5 if args.carried_c_p_wall_component == 4 else 4
        if args.carried_c_p_wall_source_json is not None:
            wall_source_path = Path(args.carried_c_p_wall_source_json)
            if not wall_source_path.exists():
                raise FileNotFoundError(f"carried-C p-wall source JSON not found: {wall_source_path}")
            loaded_low, loaded_high, source_p, source_kind = _load_carried_c_corridor_source_box(wall_source_path)
            if source_p is not None and abs(source_p - args.carried_c_p_wall_start) > 1e-12:
                raise ValueError(
                    "carried-C p-wall source JSON is at "
                    f"p={source_p}, but --carried-c-p-wall-start={args.carried_c_p_wall_start}"
                )
            source_box_low = tuple(loaded_low[:wall_box_dimension])
            source_box_high = tuple(loaded_high[:wall_box_dimension])
        report["carried_c_p_wall_certificate"] = carried_c_p_wall_certificate(
            start_p=args.carried_c_p_wall_start,
            end_p=args.carried_c_p_wall_end,
            p_step=args.carried_c_p_wall_step,
            candidate_a=args.tube_a,
            box_low=float_tuple(args.carried_c_p_wall_box_low, wall_box_dimension, "--carried-c-p-wall-box-low"),
            box_high=float_tuple(args.carried_c_p_wall_box_high, wall_box_dimension, "--carried-c-p-wall-box-high"),
            component=args.carried_c_p_wall_component,
            side=args.carried_c_p_wall_side,
            wall_value=args.carried_c_p_wall_value,
            source_box_low=source_box_low,
            source_box_high=source_box_high,
            subdivisions=tuple(
                int(value)
                for value in float_tuple(
                    args.carried_c_p_wall_subdivisions,
                    5,
                    "--carried-c-p-wall-subdivisions",
                )
            ),
            p_subdivisions=args.carried_c_p_wall_p_subdivisions,
        )
        if source_kind is not None:
            report["carried_c_p_wall_certificate"]["source_kind"] = source_kind
            report["carried_c_p_wall_certificate"]["source_json"] = str(wall_source_path)
    if args.p_corridor_check:
        report["affine_p_corridor_certificate"] = affine_p_corridor_certificate(
            start_p=args.p_corridor_start,
            end_p=args.p_corridor_end,
            step_size=args.p_corridor_step,
            candidate_a=args.tube_a,
            lower_start=float_tuple(args.p_corridor_lower_start, 4, "--p-corridor-lower-start"),
            upper_start=float_tuple(args.p_corridor_upper_start, 4, "--p-corridor-upper-start"),
            lower_slope=float_tuple(args.p_corridor_lower_slope, 4, "--p-corridor-lower-slope"),
            upper_slope=float_tuple(args.p_corridor_upper_slope, 4, "--p-corridor-upper-slope"),
            subdivisions=tuple(int(value) for value in float_tuple(args.p_corridor_subdivisions, 4, "--p-corridor-subdivisions")),
            p_subdivisions=args.p_corridor_p_subdivisions,
        )
    if args.p_corridor_tune:
        report["affine_p_corridor_tuning"] = tune_affine_p_corridor(
            x2_lower_slopes=float_list(args.p_corridor_tune_x2_slopes, "--p-corridor-tune-x2-slopes"),
            x1_upper_slopes=float_list(args.p_corridor_tune_x1_upper_slopes, "--p-corridor-tune-x1-upper-slopes"),
            start_p=args.p_corridor_start,
            end_p=args.p_corridor_end,
            step_size=args.p_corridor_step,
            candidate_a=args.tube_a,
            lower_start=float_tuple(args.p_corridor_lower_start, 4, "--p-corridor-lower-start"),
            upper_start=float_tuple(args.p_corridor_upper_start, 4, "--p-corridor-upper-start"),
            lower_slope=float_tuple(args.p_corridor_lower_slope, 4, "--p-corridor-lower-slope"),
            upper_slope=float_tuple(args.p_corridor_upper_slope, 4, "--p-corridor-upper-slope"),
            subdivisions=tuple(int(value) for value in float_tuple(args.p_corridor_subdivisions, 4, "--p-corridor-subdivisions")),
            p_subdivisions=args.p_corridor_p_subdivisions,
            max_runs=args.p_corridor_tune_max_runs,
        )
    if args.terminal_takeover_check:
        report["terminal_barrier_takeover_certificate"] = terminal_barrier_takeover_certificate(
            p_min=args.terminal_takeover_p_min,
            p_step=args.terminal_takeover_p_step,
            candidate_a=args.tube_a,
            box_low=float_tuple(args.terminal_takeover_box_low, 4, "--terminal-takeover-box-low"),
            box_high=float_tuple(args.terminal_takeover_box_high, 4, "--terminal-takeover-box-high"),
            x3_wall=args.terminal_takeover_x3_wall,
            subdivisions=tuple(
                int(value)
                for value in float_tuple(
                    args.terminal_takeover_subdivisions,
                    4,
                    "--terminal-takeover-subdivisions",
                )
            ),
            p_subdivisions=args.terminal_takeover_p_subdivisions,
        )
    if args.x3_zero_wall_check:
        report["x3_zero_wall_certificate"] = x3_zero_wall_certificate(
            time_range=float_tuple(args.x3_zero_wall_time_range, 2, "--x3-zero-wall-time-range"),
            x0_range=float_tuple(args.x3_zero_wall_x0_range, 2, "--x3-zero-wall-x0-range"),
            x1_range=float_tuple(args.x3_zero_wall_x1_range, 2, "--x3-zero-wall-x1-range"),
            x2_range=float_tuple(args.x3_zero_wall_x2_range, 2, "--x3-zero-wall-x2-range"),
            candidate_a=args.tube_a,
            subdivisions=tuple(
                int(value)
                for value in float_tuple(
                    args.x3_zero_wall_subdivisions,
                    4,
                    "--x3-zero-wall-subdivisions",
                )
            ),
            time_subdivisions=args.x3_zero_wall_time_subdivisions,
        )
    if args.x2_zero_factor_check:
        report["x2_zero_boundary_factor_certificate"] = x2_zero_boundary_factor_certificate(
            p_range=float_tuple(args.x2_zero_factor_p_range, 2, "--x2-zero-factor-p-range"),
            time_range=float_tuple(args.x2_zero_factor_time_range, 2, "--x2-zero-factor-time-range"),
            x3_range=float_tuple(args.x2_zero_factor_x3_range, 2, "--x2-zero-factor-x3-range"),
            candidate_a=args.tube_a,
        )
    if args.late_x3_descent_check:
        report["late_x3_descent_certificate"] = late_x3_descent_certificate(
            start_time=args.late_x3_descent_start,
            end_time=args.late_x3_descent_end,
            step_size=args.late_x3_descent_step,
            candidate_a=args.tube_a,
            radius0=float_tuple(args.late_x3_descent_radius0, 4, "--late-x3-descent-radius0"),
            safety=float_tuple(args.late_x3_descent_safety, 4, "--late-x3-descent-safety"),
            x0_target=args.late_x3_descent_x0_target,
        )
    if args.frontier_continuation_check:
        report["p_tube_frontier_continuation_certificate"] = p_tube_frontier_continuation_certificate(
            candidate_a=args.tube_a,
        )
    if args.hybrid_handoff_check:
        report["hybrid_p_frontier_handoff_certificate"] = hybrid_p_frontier_handoff_certificate(
            candidate_a=args.tube_a,
        )
    if args.p_start_slice_bridge_check:
        report["p_start_slice_from_support_certificate"] = p_start_slice_from_support_certificate(
            candidate_a=args.tube_a,
            after_time=args.support_tail_bridge_after_time,
            step_size=args.support_tail_bridge_step,
            support_radius0=float_tuple(
                args.support_tail_support_radius,
                4,
                "--support-tail-support-radius",
            ),
        )
    if args.regular_time_corridor_check:
        report["automatic_time_barrier_corridor_certificate"] = automatic_time_barrier_corridor_certificate(
            start_time=args.regular_time_start,
            end_time=args.regular_time_end,
            step_size=args.regular_time_step,
            candidate_a=args.tube_a,
            radius0=float_tuple(args.regular_time_radius0, 4, "--regular-time-radius0"),
            safety=float_tuple(args.regular_time_safety, 4, "--regular-time-safety"),
            subdivisions=tuple(
                int(value)
                for value in float_tuple(args.regular_time_subdivisions, 4, "--regular-time-subdivisions")
            ),
            time_subdivisions=args.regular_time_time_subdivisions,
        )
    if args.taylor_start_block_check:
        report["taylor_start_block_certificate"] = taylor_start_block_certificate(
            start_time=args.taylor_start_time,
            step_size=args.taylor_start_step,
            candidate_a=args.tube_a,
            radius=float_tuple(args.taylor_start_radius, 4, "--taylor-start-radius"),
            safety=float_tuple(args.taylor_start_safety, 4, "--taylor-start-safety"),
            subdivisions=tuple(
                int(value)
                for value in float_tuple(args.regular_time_subdivisions, 4, "--regular-time-subdivisions")
            ),
            time_subdivisions=args.regular_time_time_subdivisions,
        )
    if args.taylor_time_bridge_check:
        report["taylor_time_bridge_certificate"] = taylor_time_bridge_certificate(
            start_time=args.taylor_start_time,
            end_time=args.taylor_bridge_end,
            candidate_a=args.tube_a,
            radius=float_tuple(args.taylor_start_radius, 4, "--taylor-start-radius"),
            max_attempts=args.taylor_bridge_max_attempts,
            subdivisions=tuple(
                int(value)
                for value in float_tuple(args.regular_time_subdivisions, 4, "--regular-time-subdivisions")
            ),
            time_subdivisions=args.regular_time_time_subdivisions,
            progress_callback=_print_taylor_progress if args.taylor_progress_every_blocks > 0 else None,
            progress_every_blocks=args.taylor_progress_every_blocks,
        )
    if args.taylor_frontier_continuation_check:
        report["taylor_frontier_continuation_certificate"] = taylor_frontier_continuation_certificate(
            bridge_end_time=args.taylor_bridge_end,
            end_time=args.taylor_frontier_end,
            candidate_a=args.tube_a,
            radius=float_tuple(args.taylor_start_radius, 4, "--taylor-start-radius"),
            bridge_max_attempts=args.taylor_bridge_max_attempts,
            max_attempts=args.taylor_frontier_max_attempts,
            subdivisions=tuple(
                int(value)
                for value in float_tuple(args.regular_time_subdivisions, 4, "--regular-time-subdivisions")
            ),
            time_subdivisions=args.regular_time_time_subdivisions,
        )
    if args.taylor_restart_chain_check:
        bridge_certificate = (
            None
            if args.taylor_restart_bridge_json is None
            else _load_taylor_bridge_certificate(Path(args.taylor_restart_bridge_json))
        )
        report["taylor_restart_chain_certificate"] = taylor_restart_chain_certificate(
            bridge_end_time=args.taylor_bridge_end,
            end_time=args.taylor_restart_end,
            restart_interval=args.taylor_restart_interval,
            step_size=args.tube_step,
            candidate_a=args.tube_a,
            radius=float_tuple(args.taylor_start_radius, 4, "--taylor-start-radius"),
            bridge_max_attempts=args.taylor_bridge_max_attempts,
            initial_growth=float_tuple(args.tuned_tube_initial_growth, 4, "--tuned-tube-initial-growth"),
            max_growth=float_tuple(args.taylor_restart_max_growth, 4, "--taylor-restart-max-growth"),
            growth_factor=args.tuned_tube_growth_factor,
            max_attempts=args.taylor_restart_max_attempts,
            subdivisions=tuple(
                int(value)
                for value in float_tuple(args.regular_time_subdivisions, 4, "--regular-time-subdivisions")
            ),
            time_subdivisions=args.regular_time_time_subdivisions,
            bridge_certificate=bridge_certificate,
            bridge_progress_callback=_print_taylor_progress if args.taylor_progress_every_blocks > 0 else None,
            bridge_progress_every_blocks=args.taylor_progress_every_blocks,
            progress_callback=_print_taylor_restart_progress
            if args.taylor_restart_progress_every_segments > 0
            else None,
            progress_every_segments=args.taylor_restart_progress_every_segments,
        )
    if args.taylor_p_slice_audit:
        report["taylor_p_slice_convergence_audit"] = taylor_p_slice_convergence_audit(
            target_p=args.taylor_p_slice_target,
            candidate_a=args.tube_a,
            low_order=args.taylor_p_slice_low_order,
            high_order=args.taylor_p_slice_high_order,
            working_dps=args.taylor_p_slice_working_dps,
            radius0=float_tuple(args.taylor_p_slice_radius, 5, "--taylor-p-slice-radius"),
        )
    if args.taylor_p_slice_tail_audit:
        report["taylor_p_slice_tail_ratio_audit"] = taylor_p_slice_tail_ratio_audit(
            target_p=args.taylor_p_slice_target,
            candidate_a=args.tube_a,
            order=args.taylor_p_slice_tail_order,
            tail_start=args.taylor_p_slice_tail_start,
            ratio_start=args.taylor_p_slice_ratio_start,
            ratio_bound=args.taylor_p_slice_ratio_bound,
            b_sample_count=args.taylor_p_slice_b_samples,
            working_dps=args.taylor_p_slice_tail_working_dps,
            radius0=float_tuple(args.taylor_p_slice_radius, 5, "--taylor-p-slice-radius"),
        )
    if args.taylor_p_slice_cauchy_budget_audit:
        report["taylor_p_slice_cauchy_budget_audit"] = taylor_p_slice_cauchy_budget_audit(
            target_p=args.taylor_p_slice_target,
            candidate_a=args.tube_a,
            order=args.taylor_p_slice_tail_order,
            tail_start=args.taylor_p_slice_tail_start,
            b_sample_count=args.taylor_p_slice_b_samples,
            working_dps=args.taylor_p_slice_tail_working_dps,
            analytic_radii=float_list(args.taylor_p_slice_cauchy_radii, "--taylor-p-slice-cauchy-radii"),
            circle_sample_count=args.taylor_p_slice_cauchy_circle_samples,
            circle_tail_ratio_bound=args.taylor_p_slice_cauchy_circle_tail_ratio_bound,
            radius0=float_tuple(args.taylor_p_slice_radius, 5, "--taylor-p-slice-radius"),
        )
    if args.taylor_ratio_profile_audit:
        report["taylor_ratio_profile_audit"] = taylor_ratio_profile_audit(
            target_p=args.taylor_p_slice_target,
            candidate_a=args.tube_a,
            order=args.taylor_p_slice_tail_order,
            ratio_start=args.taylor_p_slice_ratio_start,
            b_sample_count=args.taylor_p_slice_b_samples,
            b_mode=args.taylor_ratio_profile_b_mode,
            working_dps=args.taylor_p_slice_tail_working_dps,
            circle_radius=args.taylor_ratio_profile_circle_radius,
            circle_ratio_bound=args.taylor_ratio_profile_circle_ratio_bound,
            p_slice_ratio_bound=args.taylor_ratio_profile_p_slice_ratio_bound,
        )
    if args.taylor_geometric_envelope_audit:
        report["taylor_geometric_envelope_audit"] = taylor_geometric_envelope_audit(
            target_p=args.taylor_p_slice_target,
            candidate_a=args.tube_a,
            order=args.taylor_p_slice_tail_order,
            tail_start=args.taylor_p_slice_tail_start,
            b_sample_count=args.taylor_p_slice_b_samples,
            b_mode=args.taylor_ratio_profile_b_mode,
            working_dps=args.taylor_p_slice_tail_working_dps,
            circle_radius=args.taylor_ratio_profile_circle_radius,
            circle_ratio_bound=args.taylor_ratio_profile_circle_ratio_bound,
            p_slice_ratio_bound=args.taylor_ratio_profile_p_slice_ratio_bound,
        )
    if args.taylor_even_parity_audit:
        report["taylor_even_parity_audit"] = taylor_even_parity_audit(
            candidate_a=args.tube_a,
            order=args.taylor_p_slice_tail_order,
            b_sample_count=args.taylor_p_slice_b_samples,
            working_dps=args.taylor_p_slice_tail_working_dps,
            complex_b_radius=args.taylor_b_cauchy_enclosure_radius,
            complex_b_sample_count=args.taylor_b_cauchy_enclosure_samples or args.taylor_b_cauchy_samples,
        )
    if args.taylor_even_s_series_audit:
        report["taylor_even_s_series_audit"] = taylor_even_s_series_audit(
            target_p=args.taylor_p_slice_target,
            candidate_a=args.tube_a,
            order=args.taylor_p_slice_tail_order,
            tail_start=args.taylor_p_slice_tail_start,
            b_sample_count=args.taylor_p_slice_b_samples,
            b_mode=args.taylor_ratio_profile_b_mode,
            working_dps=args.taylor_p_slice_tail_working_dps,
            circle_radius=args.taylor_ratio_profile_circle_radius,
            circle_ratio_bound=args.taylor_ratio_profile_circle_ratio_bound,
            p_slice_ratio_bound=args.taylor_ratio_profile_p_slice_ratio_bound,
        )
    if args.taylor_recurrence_forcing_audit:
        report["taylor_recurrence_forcing_audit"] = taylor_recurrence_forcing_audit(
            candidate_a=args.tube_a,
            order=args.taylor_p_slice_tail_order,
            tail_start=args.taylor_p_slice_tail_start,
            b_sample_count=args.taylor_p_slice_b_samples,
            b_mode=args.taylor_ratio_profile_b_mode,
            working_dps=args.taylor_p_slice_tail_working_dps,
            circle_radius=args.taylor_ratio_profile_circle_radius,
            circle_ratio_bound=args.taylor_ratio_profile_circle_ratio_bound,
        )
    if args.taylor_b_sensitivity_audit:
        report["taylor_b_sensitivity_audit"] = taylor_b_sensitivity_audit(
            target_p=args.taylor_p_slice_target,
            candidate_a=args.tube_a,
            order=args.taylor_p_slice_tail_order,
            ratio_start=args.taylor_p_slice_ratio_start,
            b_sample_count=args.taylor_p_slice_b_samples,
            working_dps=args.taylor_p_slice_tail_working_dps,
            circle_radius=args.taylor_b_sensitivity_circle_radius,
            radius0=float_tuple(args.taylor_p_slice_radius, 5, "--taylor-p-slice-radius"),
        )
    if args.taylor_p_slice_entry_budget_audit:
        report["taylor_p_slice_entry_budget_audit"] = taylor_p_slice_entry_budget_audit(
            target_p=args.taylor_p_slice_target,
            candidate_a=args.tube_a,
            order=args.taylor_p_slice_tail_order,
            tail_start=args.taylor_p_slice_tail_start,
            ratio_start=args.taylor_p_slice_ratio_start,
            ratio_bound=args.taylor_p_slice_ratio_bound,
            b_sample_count=args.taylor_p_slice_b_samples,
            working_dps=args.taylor_p_slice_tail_working_dps,
            b_cauchy_radius=args.taylor_b_cauchy_radius,
            b_circle_sample_count=args.taylor_b_cauchy_samples,
            b_outer_cauchy_radius=args.taylor_b_cauchy_outer_radius,
            b_outer_circle_sample_count=args.taylor_b_cauchy_outer_samples,
            b_enclosure_cauchy_radius=args.taylor_b_cauchy_enclosure_radius,
            b_enclosure_circle_sample_count=args.taylor_b_cauchy_enclosure_samples,
            radius0=float_tuple(args.taylor_p_slice_radius, 5, "--taylor-p-slice-radius"),
            include_direct_endpoints=not args.taylor_b_cauchy_skip_direct,
        )
    if args.taylor_p_slice_required_a_audit:
        report["taylor_p_slice_required_a_audit"] = taylor_p_slice_required_a_audit(
            target_p=args.taylor_p_slice_target,
            candidate_a=args.tube_a,
            order=args.taylor_p_slice_tail_order,
            tail_start=args.taylor_p_slice_tail_start,
            ratio_start=args.taylor_p_slice_ratio_start,
            ratio_bound=args.taylor_p_slice_ratio_bound,
            b_sample_count=args.taylor_p_slice_b_samples,
            working_dps=args.taylor_p_slice_tail_working_dps,
            b_cauchy_radius=args.taylor_b_cauchy_radius,
            b_circle_sample_count=args.taylor_b_cauchy_samples,
            b_outer_cauchy_radius=args.taylor_b_cauchy_outer_radius,
            b_outer_circle_sample_count=args.taylor_b_cauchy_outer_samples,
            b_enclosure_cauchy_radius=args.taylor_b_cauchy_enclosure_radius,
            b_enclosure_circle_sample_count=args.taylor_b_cauchy_enclosure_samples,
            radius0=float_tuple(args.taylor_p_slice_radius, 5, "--taylor-p-slice-radius"),
            include_direct_endpoints=not args.taylor_b_cauchy_skip_direct,
        )
    if args.taylor_p_slice_b_cauchy_event_audit:
        report["taylor_p_slice_b_cauchy_event_audit"] = taylor_p_slice_b_cauchy_event_audit(
            target_p=args.taylor_p_slice_target,
            candidate_a=args.tube_a,
            order=args.taylor_p_slice_tail_order,
            working_dps=args.taylor_p_slice_tail_working_dps,
            b_cauchy_radius=args.taylor_b_cauchy_radius,
            b_circle_sample_count=args.taylor_b_cauchy_samples,
            b_outer_cauchy_radius=args.taylor_b_cauchy_outer_radius,
            b_outer_circle_sample_count=args.taylor_b_cauchy_outer_samples,
            b_enclosure_cauchy_radius=args.taylor_b_cauchy_enclosure_radius,
            b_enclosure_circle_sample_count=args.taylor_b_cauchy_enclosure_samples,
            radius0=float_tuple(args.taylor_p_slice_radius, 5, "--taylor-p-slice-radius"),
            include_direct_endpoints=not args.taylor_b_cauchy_skip_direct,
        )
    if args.taylor_b_cauchy_coefficient_audit:
        report["taylor_b_cauchy_coefficient_audit"] = taylor_b_cauchy_coefficient_audit(
            candidate_a=args.tube_a,
            order=args.taylor_p_slice_tail_order,
            working_dps=args.taylor_p_slice_tail_working_dps,
            time_radius=args.taylor_b_cauchy_time_radius,
            b_cauchy_radius=args.taylor_b_cauchy_radius,
            b_circle_sample_count=args.taylor_b_cauchy_samples,
            support_radius0=float_tuple(args.support_tail_support_radius, 4, "--support-tail-support-radius"),
            include_direct_endpoints=not args.taylor_b_cauchy_skip_direct,
        )
    if args.taylor_support_time_audit:
        report["taylor_support_time_convergence_audit"] = taylor_support_time_convergence_audit(
            candidate_a=args.tube_a,
            support_time=args.taylor_support_time,
            low_order=args.taylor_support_low_order,
            high_order=args.taylor_support_high_order,
            b_sample_count=args.taylor_p_slice_b_samples,
            working_dps=args.taylor_p_slice_tail_working_dps,
            support_radius0=float_tuple(args.support_tail_support_radius, 4, "--support-tail-support-radius"),
        )
    if args.taylor_circle_residual_audit:
        report["taylor_circle_residual_audit"] = taylor_circle_residual_audit(
            candidate_a=args.tube_a,
            order=args.taylor_p_slice_tail_order,
            b_sample_count=args.taylor_p_slice_b_samples,
            working_dps=args.taylor_p_slice_tail_working_dps,
            circle_radius=args.taylor_circle_residual_radius,
            circle_sample_count=args.taylor_circle_residual_samples,
        )
    if args.taylor_p_slice_interval_ratio_audit:
        ratio_bound = args.taylor_p_slice_ratio_bound
        if ratio_bound is None:
            raise ValueError("--taylor-p-slice-interval-ratio-audit requires --taylor-p-slice-ratio-bound")
        report["taylor_p_slice_interval_ratio_audit"] = interval_taylor_finite_ratio_audit(
            target_p=args.taylor_p_slice_target,
            candidate_a=args.tube_a,
            order=args.taylor_p_slice_interval_order,
            ratio_start=args.taylor_p_slice_ratio_start,
            ratio_bound=ratio_bound,
            b_subdivisions=args.taylor_p_slice_b_subdivisions,
            working_dps=args.taylor_p_slice_interval_working_dps,
            time_padding=args.taylor_p_slice_time_padding,
        )
    if args.piecewise_corridor_check:
        report["piecewise_affine_p_corridor_certificate"] = piecewise_affine_p_corridor_certificate(
            candidate_a=args.tube_a,
        )
    if args.late_tail_closure_check:
        report["late_tail_closure_certificate"] = late_tail_closure_certificate(
            candidate_a=args.tube_a,
        )
    if args.broad_tail_closure_check:
        report["broad_tail_closure_certificate"] = broad_tail_closure_certificate(
            candidate_a=args.tube_a,
        )
    if args.support_tail_closure_check:
        report["support_tail_closure_certificate"] = support_tail_closure_certificate(
            candidate_a=args.tube_a,
            bridge_after_time=args.support_tail_bridge_after_time,
            bridge_step_size=args.support_tail_bridge_step,
            support_radius0=float_tuple(
                args.support_tail_support_radius,
                4,
                "--support-tail-support-radius",
            ),
        )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return
    print("S7 SU(2)^3 tail-defect diagnostics", flush=True)
    print(f"version: {report['version']}", flush=True)
    print(f"defect: {report['defect']}", flush=True)
    limit = report["limiting_crossing"]
    print(
        "limit crossing: "
        f"T={limit['time']:.9g}, X2={limit['x2_tail_defect']:.9g}, "
        f"X3={limit['x3_auxiliary_defect']:.9g}, status={limit['status']}",
        flush=True,
    )
    print("finite tail samples:", flush=True)
    for item in report["exact_crossings"]:
        print(
            f"  a={item['a']:9.3g} T={item['time']:.9g} "
            f"X2={item['x2_tail_defect']:.9g} X3={item['x3_auxiliary_defect']:.9g} "
            f"u={item.get('u_f2_over_f1', float('nan')):.9g} status={item['status']}",
            flush=True,
        )
    support = report["sign_support"]
    state = support["support_state"]
    print(
        "support state: "
        f"t={support['support_time']}, x0={state[0]:.9g}, x1={state[1]:.9g}, "
        f"x2={state[2]:.9g}, x3={state[3]:.9g}",
        flush=True,
    )
    print(
        "boundary signs: "
        f"x2'|x2=0={support['x2_boundary_derivative_at_support']:.9g}, "
        f"x3'|x3=-0.3={support['x3_minus_0_3_boundary_derivative_at_support']:.9g}",
        flush=True,
    )
    riccati = support["riccati"]
    print(
        "riccati check: "
        f"integral~{riccati['riccati_integral_numeric']:.9g}, "
        f"lower~{riccati['riccati_lower_bound_numeric']:.9g}",
        flush=True,
    )
    barrier = report["terminal_barrier"]
    barrier_state = barrier["limit"]
    print(
        "terminal x3=0 barrier: "
        f"t={barrier['support_time']}, limit x1={barrier_state['x1']:.9g}, "
        f"threshold={barrier_state['x1_threshold_for_x3_zero_barrier']:.9g}, "
        f"margin={barrier_state['x1_margin']:.9g}, "
        f"x3'|x3=0={barrier_state['x3_zero_boundary_derivative']:.9g}",
        flush=True,
    )
    for item in barrier["finite_candidate_A"]:
        print(
            f"  finite A sample a={item['a']:.9g}: "
            f"x1={item['x1']:.9g}, x3={item['x3']:.9g}, "
            f"margin={item['x1_margin']:.9g}, x3'|x3=0={item['x3_zero_boundary_derivative']:.9g}",
            flush=True,
        )
    perturbation = report["finite_a_perturbation"]
    print(
        "finite-a perturbation sample: "
        f"A={perturbation['candidate_A']:.9g}, "
        f"component bounds={perturbation['sampled_rhs_error_bound_at_candidate_A']}",
        flush=True,
    )
    if "moving_tube_certificate" in report:
        tube = report["moving_tube_certificate"]
        print(
            "moving tube certificate: "
            f"status={tube['status']}, A={tube['candidate_A']:.9g}, "
            f"t=[{tube['start_time']}, {tube['end_time']}], "
            f"step={tube['step_size']}, worst_margin={tube['worst_margin']:.9g}",
            flush=True,
        )
        face = tube.get("failing_face") or tube.get("worst_face")
        if face is not None:
            print(f"  face={face}", flush=True)
    if "segmented_moving_tube_certificate" in report:
        tube = report["segmented_moving_tube_certificate"]
        print(
            "segmented moving tube certificate: "
            f"status={tube['status']}, A={tube['candidate_A']:.9g}, "
            f"certified_until={tube['certified_until']:.9g}, "
            f"blocks={tube['blocks_certified']}",
            flush=True,
        )
        block = tube.get("failing_block")
        if block is not None:
            print(
                f"  failing_margin={block['worst_margin']:.9g}, "
                f"face={block.get('failing_face')}",
                flush=True,
            )
    if "tuned_segmented_moving_tube_certificate" in report:
        tube = report["tuned_segmented_moving_tube_certificate"]
        print(
            "tuned segmented moving tube certificate: "
            f"status={tube['status']}, A={tube['candidate_A']:.9g}, "
            f"certified_until={tube['certified_until']:.9g}, "
            f"blocks={tube['blocks_certified']}, "
            f"attempts={tube['tuning_attempt_count']}, "
            f"worst_margin={tube.get('worst_margin', float('nan')):.9g}",
            flush=True,
        )
        block = tube.get("failing_block")
        if block is not None:
            print(
                f"  failing_margin={block.get('worst_margin', float('nan')):.9g}, "
                f"face={block.get('failing_face')}",
                flush=True,
            )
    if "restart_tuned_time_chain_certificate" in report:
        chain = report["restart_tuned_time_chain_certificate"]
        print(
            "restart tuned time-chain certificate: "
            f"status={chain['status']}, A={chain['candidate_A']:.9g}, "
            f"certified_until={chain['certified_until']:.9g}, "
            f"segments={chain['segments_certified']}, "
            f"blocks={chain['blocks_certified']}, "
            f"attempts={chain['tuning_attempt_count']}, "
            f"worst_margin={chain.get('worst_margin', float('nan')):.9g}",
            flush=True,
        )
        block = chain.get("failing_block")
        if block is not None:
            print(
                f"  failing_margin={block.get('worst_margin', float('nan')):.9g}, "
                f"face={block.get('failing_face')}",
                flush=True,
            )
    if "automatic_time_barrier_corridor_certificate" in report:
        corridor = report["automatic_time_barrier_corridor_certificate"]
        print(
            "automatic time-corridor certificate: "
            f"status={corridor['status']}, A={corridor['candidate_A']:.9g}, "
            f"t=[{corridor['start_time']}, {corridor['end_time']}], "
            f"certified_until={corridor.get('certified_until', corridor['end_time']):.9g}, "
            f"steps={corridor['steps_certified']}/{corridor.get('steps', corridor['steps_certified'])}, "
            f"worst_margin={corridor.get('worst_margin', float('nan')):.9g}",
            flush=True,
        )
        face = corridor.get("failing_step", {}).get("failing_face") or corridor.get("worst_face")
        if face is not None:
            print(f"  face={face}", flush=True)
    if "taylor_start_block_certificate" in report:
        certificate = report["taylor_start_block_certificate"]
        step = certificate["step_certificate"]
        print(
            "Taylor start-block certificate: "
            f"status={certificate['status']}, A={certificate['candidate_A']:.9g}, "
            f"t=[{certificate['start_time']}, {certificate['end_time']}], "
            f"worst_margin={certificate.get('worst_margin', float('nan')):.9g}, "
            f"conditional={certificate['conditional']}",
            flush=True,
        )
        face = step.get("failing_face") or step.get("worst_face")
        if face is not None:
            print(f"  face={face}", flush=True)
    if "taylor_time_bridge_certificate" in report:
        bridge = report["taylor_time_bridge_certificate"]
        print(
            "Taylor time-bridge certificate: "
            f"status={bridge['status']}, A={bridge['candidate_A']:.9g}, "
            f"certified_until={bridge['certified_until']:.9g}, "
            f"blocks={bridge.get('blocks_certified', 0)}, "
            f"attempts={bridge.get('tuning_attempt_count', 0)}, "
            f"worst_margin={bridge.get('worst_margin', float('nan')):.9g}, "
            f"conditional={bridge['conditional']}",
            flush=True,
        )
        width = bridge.get("end_width") or bridge.get("current_width")
        if width is not None:
            print(f"  end_width={width}", flush=True)
        face = bridge.get("worst_face")
        if face is not None:
            print(f"  face={face}", flush=True)
    if "taylor_frontier_continuation_certificate" in report:
        frontier = report["taylor_frontier_continuation_certificate"]
        print(
            "Taylor frontier continuation certificate: "
            f"status={frontier['status']}, A={frontier['candidate_A']:.9g}, "
            f"certified_until={frontier.get('certified_until', float('nan')):.9g}, "
            f"steps={frontier.get('steps_certified', 0)}, "
            f"attempts={frontier.get('tuning_attempt_count', 0)}, "
            f"worst_margin={frontier.get('worst_margin', float('nan')):.9g}, "
            f"conditional={frontier['conditional']}",
            flush=True,
        )
        width = frontier.get("end_width") or frontier.get("current_width")
        if width is not None:
            print(f"  frontier_width={width}", flush=True)
        face = frontier.get("worst_face")
        if face is not None:
            print(f"  face={face}", flush=True)
    if "taylor_restart_chain_certificate" in report:
        chain = report["taylor_restart_chain_certificate"]
        print(
            "Taylor restart-chain certificate: "
            f"status={chain['status']}, A={chain['candidate_A']:.9g}, "
            f"certified_until={chain.get('certified_until', float('nan')):.9g}, "
            f"segments={chain.get('segments_certified', 0)}, "
            f"blocks={chain.get('blocks_certified', 0)}, "
            f"attempts={chain.get('tuning_attempt_count', 0)}, "
            f"worst_margin={chain.get('worst_margin', float('nan')):.9g}, "
            f"conditional={chain['conditional']}",
            flush=True,
        )
        width = chain.get("end_width") or chain.get("current_width")
        if width is not None:
            print(f"  restart_width={width}", flush=True)
        face = chain.get("worst_face")
        if face is not None:
            print(f"  face={face}", flush=True)
    if "taylor_p_slice_convergence_audit" in report:
        audit = report["taylor_p_slice_convergence_audit"]
        print(
            "Taylor p-slice convergence audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"p={audit['target_p']:.9g}, "
            f"orders={audit['low_order']}->{audit['high_order']}, "
            f"max_diff={audit['max_order_difference_5d']}, "
            f"max_diff/radius={audit['max_order_difference_over_radius']}",
            flush=True,
        )
    if "taylor_p_slice_tail_ratio_audit" in report:
        audit = report["taylor_p_slice_tail_ratio_audit"]
        print(
            "Taylor p-slice tail-ratio audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"p={audit['target_p']:.9g}, "
            f"order={audit['order']}, tail_start={audit['tail_start']}, "
            f"ratio_bound={audit['ratio_bound']}, "
            f"b_samples={audit['b_sample_count']}, "
            f"max_tail/radius={audit['max_tail_estimate_over_radius']}, "
            f"max_ratio={audit['max_observed_ratio_4d']}, "
            f"inside_bound={audit['observed_ratios_inside_bound']}, "
            f"time_shift_bound={audit['max_time_shift_bound_from_p_tail']:.9g}",
            flush=True,
        )
    if "taylor_p_slice_cauchy_budget_audit" in report:
        audit = report["taylor_p_slice_cauchy_budget_audit"]
        print(
            "Taylor p-slice Cauchy-budget audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"p={audit['target_p']:.9g}, "
            f"order={audit['order']}, tail_start={audit['tail_start']}, "
            f"b_samples={audit['b_sample_count']}, "
            f"viable_radii={audit['viable_analytic_radii']}, "
            f"proof_relevant_viable_radii={audit['proof_relevant_viable_analytic_radii']}, "
            f"terminal_time_ref={audit['limiting_crossing_time_reference']:.9g}, "
            f"best_radius={audit['best_radius_by_observed_floor']:.9g}, "
            f"best_max_tail_floor/radius={audit['best_max_tail_floor_over_radius']:.9g}, "
            f"best_p_circle_min={audit['best_radius_min_p_circle_abs_partial']:.9g}, "
            f"best_p_circle_certified_min={audit['best_radius_certified_min_p_circle_abs_partial']:.9g}, "
            f"best_p_circle_tail={audit['best_radius_p_circle_tail_estimate']:.9g}, "
            f"best_p_circle_rouche_margin={audit['best_radius_p_circle_rouche_margin']:.9g}, "
            f"best_p_circle_ratio={audit['best_radius_p_circle_observed_tail_ratio']:.9g}, "
            f"best_p_circle_inside_ratio_bound={audit['best_radius_p_circle_tail_inside_ratio_bound']}",
            flush=True,
        )
    if "taylor_ratio_profile_audit" in report:
        audit = report["taylor_ratio_profile_audit"]
        print(
            "Taylor ratio-profile audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"order={audit['order']}, ratio_start={audit['ratio_start']}, "
            f"b_mode={audit['b_mode']}, b_samples={audit['b_sample_count']}, "
            f"circle_R={audit['circle_radius']:.9g}, "
            f"circle_bound={audit['circle_ratio_bound']}, "
            f"p_slice_bound={audit['p_slice_ratio_bound']}, "
            f"max_circle_ratio={audit['max_circle_ratio_4d']}, "
            f"circle_inside={audit['circle_inside_bound']}, "
            f"max_p_slice_ratio={audit['max_p_slice_ratio_4d']}, "
            f"p_slice_inside={audit['p_slice_inside_bound']}",
            flush=True,
        )
    if "taylor_geometric_envelope_audit" in report:
        audit = report["taylor_geometric_envelope_audit"]
        print(
            "Taylor geometric-envelope audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"order={audit['order']}, tail_start={audit['tail_start']}, "
            f"b_mode={audit['b_mode']}, b_samples={audit['b_sample_count']}, "
            f"circle_R={audit['circle_radius']:.9g}, "
            f"circle_q={audit['circle_ratio_bound']}, "
            f"p_slice_q={audit['p_slice_ratio_bound']}, "
            f"max_circle_usage={audit['max_circle_envelope_usage_4d']}, "
            f"max_circle_strict_usage={audit['max_circle_strict_post_anchor_usage_4d']}, "
            f"max_circle_tail_sum_usage={audit['max_circle_tail_sum_usage_4d']}, "
            f"circle_inside={audit['circle_inside_envelope']}, "
            f"max_p_slice_usage={audit['max_p_slice_envelope_usage_4d']}, "
            f"max_p_slice_strict_usage={audit['max_p_slice_strict_post_anchor_usage_4d']}, "
            f"max_p_slice_tail_sum_usage={audit['max_p_slice_tail_sum_usage_4d']}, "
            f"p_slice_inside={audit['p_slice_inside_envelope']}",
            flush=True,
        )
    if "taylor_even_parity_audit" in report:
        audit = report["taylor_even_parity_audit"]
        print(
            "Taylor even-parity audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"order={audit['order']}, "
            f"real_b_samples={audit['b_sample_count']}, "
            f"complex_b_radius={audit['complex_b_radius']}, "
            f"complex_b_samples={audit['complex_b_sample_count']}, "
            f"max_odd_abs={audit['max_odd_abs_4d']}, "
            f"max_even_abs={audit['max_even_abs_4d']}",
            flush=True,
        )
    if "taylor_even_s_series_audit" in report:
        audit = report["taylor_even_s_series_audit"]
        print(
            "Taylor even-s-series audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"order={audit['order']}, tail_start_s={audit['tail_start_s_index']}, "
            f"b_mode={audit['b_mode']}, b_samples={audit['b_sample_count']}, "
            f"circle_s={audit['circle_radius_s']:.9g}, "
            f"circle_q={audit['circle_ratio_bound']}, "
            f"p_slice_q={audit['p_slice_ratio_bound']}, "
            f"max_circle_ratio={audit['max_circle_ratio_4d']}, "
            f"max_p_slice_ratio={audit['max_p_slice_ratio_4d']}, "
            f"min_inferred_circle_s={audit['min_inferred_circle_radius_s_4d']}, "
            f"terminal_s={audit['limiting_crossing_s_reference']:.9g}",
            flush=True,
        )
    if "taylor_recurrence_forcing_audit" in report:
        audit = report["taylor_recurrence_forcing_audit"]
        print(
            "Taylor recurrence-forcing audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"order={audit['order']}, tail_start_s={audit['tail_start_s_index']}, "
            f"b_mode={audit['b_mode']}, b_samples={audit['b_sample_count']}, "
            f"circle_s={audit['circle_radius_s']:.9g}, "
            f"circle_q={audit['circle_ratio_bound']}, "
            f"max_reconstruction_error={audit['max_reconstruction_error_4d']}, "
            f"max_inverse_usage={audit['max_inverse_bound_usage_4d']}, "
            f"max_solution_ratio={audit['max_solution_ratio_4d']}, "
            f"max_forcing_ratio={audit['max_forcing_ratio_4d']}",
            flush=True,
        )
    if "taylor_b_sensitivity_audit" in report:
        audit = report["taylor_b_sensitivity_audit"]
        print(
            "Taylor b-sensitivity audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"order={audit['order']}, ratio_start={audit['ratio_start']}, "
            f"circle_R={audit['circle_radius']:.9g}, "
            f"max_state_delta/radius={audit['max_state_delta_over_radius']}, "
            f"max_circle_delta_l1_rel={audit['max_circle_delta_l1_relative_to_limit_4d']}, "
            f"max_circle_tail_delta_l1_rel={audit['max_circle_tail_delta_l1_relative_to_limit_4d']}",
            flush=True,
        )
    if "taylor_p_slice_entry_budget_audit" in report:
        audit = report["taylor_p_slice_entry_budget_audit"]
        print(
            "Taylor p-slice entry-budget audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"p={audit['target_p']:.9g}, order={audit['order']}, "
            f"tail_start={audit['tail_start']}, ratio_bound={audit['ratio_bound']}, "
            f"tail/radius={audit['tail_budget_over_radius']}, "
            f"finite_b/radius={audit['finite_b_budget_over_radius']}, "
            f"combined/radius={audit['combined_budget_over_radius']}, "
            f"max_combined={audit['max_combined_budget_over_radius']:.9g}, "
            f"ratios_inside={audit['observed_ratios_inside_bound']}, "
            f"event_status={audit['event_cauchy_status']}, "
            f"event_source={audit['event_cauchy_source']}",
            flush=True,
        )
    if "taylor_p_slice_required_a_audit" in report:
        audit = report["taylor_p_slice_required_a_audit"]
        print(
            "Taylor p-slice required-A audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"minimum_A={audit['minimum_A_for_conditional_entry_budget']:.9g}, "
            f"headroom={audit['candidate_A_headroom_factor']:.9g}, "
            f"max_tail/radius={audit['max_tail_budget_over_radius']:.9g}, "
            f"max_finite_b/radius={audit['max_finite_b_budget_over_radius_at_candidate_A']:.9g}, "
            f"max_combined/radius={audit['max_combined_budget_over_radius_at_candidate_A']:.9g}, "
            f"event_source={audit['event_cauchy_source']}",
            flush=True,
        )
    if "taylor_p_slice_b_cauchy_event_audit" in report:
        audit = report["taylor_p_slice_b_cauchy_event_audit"]
        direct_delta = audit["max_direct_delta_over_radius"]
        print(
            "Taylor p-slice b-Cauchy event audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"p={audit['target_p']:.9g}, order={audit['order']}, "
            f"b_R={audit['b_cauchy_radius']:.9g}, "
            f"samples={audit['b_circle_sample_count']}, "
            f"direct_delta/radius={direct_delta if direct_delta is not None else 'skipped'}, "
            f"cauchy_delta/radius={audit['cauchy_delta_bound_over_radius']}, "
            f"empirical_cauchy_delta/radius={audit['empirical_cauchy_delta_bound_over_radius']}, "
            f"proof_cauchy_delta/radius={audit['proof_cauchy_delta_bound_over_radius']}, "
            f"proof_source={audit['proof_cauchy_source']}, "
            f"max_p_residual={audit['max_p_residual_abs']:.3g}, "
            f"min_event_p_derivative={audit['min_event_p_derivative_abs']:.9g}",
            flush=True,
        )
    if "taylor_b_cauchy_coefficient_audit" in report:
        audit = report["taylor_b_cauchy_coefficient_audit"]
        direct_delta = audit["max_direct_delta_over_support_radius"]
        print(
            "Taylor b-Cauchy coefficient audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"order={audit['order']}, t_R={audit['time_radius']:.9g}, "
            f"b_R={audit['b_cauchy_radius']:.9g}, "
            f"samples={audit['b_circle_sample_count']}, "
            f"direct_delta/radius={direct_delta if direct_delta is not None else 'skipped'}, "
            f"cauchy_delta/radius={audit['cauchy_delta_bound_over_support_radius']}",
            flush=True,
        )
    if "taylor_support_time_convergence_audit" in report:
        audit = report["taylor_support_time_convergence_audit"]
        print(
            "Taylor support-time convergence audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"t={audit['support_time']:.9g}, "
            f"orders={audit['low_order']}->{audit['high_order']}, "
            f"b_samples={audit['b_sample_count']}, "
            f"max_diff={audit['max_order_difference_4d']}, "
            f"max_diff/radius={audit['max_order_difference_over_support_radius']}",
            flush=True,
        )
    if "taylor_circle_residual_audit" in report:
        audit = report["taylor_circle_residual_audit"]
        print(
            "Taylor circle-residual audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"order={audit['order']}, R={audit['circle_radius']:.9g}, "
            f"samples={audit['circle_sample_count']}, "
            f"max_residual={audit['max_residual']:.9g}, "
            f"max_residual_4d={audit['max_residual_4d']}, "
            f"min_p_abs={audit['min_p_abs']:.9g}",
            flush=True,
        )
    if "taylor_p_slice_interval_ratio_audit" in report:
        audit = report["taylor_p_slice_interval_ratio_audit"]
        print(
            "Taylor p-slice interval-ratio audit: "
            f"status={audit['status']}, A={audit['candidate_A']:.9g}, "
            f"p={audit['target_p']:.9g}, order={audit['order']}, "
            f"ratio_start={audit['ratio_start']}, ratio_bound={audit['ratio_bound']}, "
            f"b_subdivisions={audit['b_subdivisions']}, "
            f"max_ratio={audit.get('max_ratio_upper', float('nan')):.9g}, "
            f"component_max={audit.get('component_max_ratio_upper')}, "
            f"time_hull={audit['checked_time_hull']}, "
            f"failed_subinterval={audit.get('failed_subinterval')}",
            flush=True,
        )
    if "x3_zero_wall_certificate" in report:
        wall = report["x3_zero_wall_certificate"]
        print(
            "x3=0 wall certificate: "
            f"status={wall['status']}, A={wall['candidate_A']:.9g}, "
            f"t={wall['time_range']}, x0={wall['x0_range']}, "
            f"x1_margin={wall['threshold_margin']:.9g}, "
            f"analytic_rhs_x3_upper={wall['analytic_rhs_x3_upper']:.9g}, "
            f"analytic_inward_margin={wall['analytic_inward_margin']:.9g}, "
            f"interval_status={wall['interval_status']}",
            flush=True,
        )
    if "x2_zero_boundary_factor_certificate" in report:
        wall = report["x2_zero_boundary_factor_certificate"]
        print(
            "x2=0 factor certificate: "
            f"status={wall['status']}, A={wall['candidate_A']:.9g}, "
            f"p={wall['p_range']}, x3={wall['x3_range']}, "
            f"factor_lower={wall['factor_lower_bound']:.9g}, "
            f"x2_prime_lower={wall['x2_prime_lower_bound_on_wall']:.9g}",
            flush=True,
        )
    if "late_x3_descent_certificate" in report:
        descent = report["late_x3_descent_certificate"]
        end_box = descent.get("end_box") or descent.get("time_corridor", {}).get("end_box")
        print(
            "late x3 descent certificate: "
            f"status={descent['status']}, A={descent['candidate_A']:.9g}, "
            f"t=[{descent.get('start_time')}, {descent.get('end_time')}], "
            f"conditional={descent['conditional']}",
            flush=True,
        )
        if end_box is not None:
            print(
                f"  end x0_high={end_box['high'][0]:.9g}, "
                f"x3_high={end_box['high'][3]:.9g}, "
                f"x2_low={end_box['low'][2]:.9g}, "
                f"wall_box_contained={descent.get('end_box_contained_in_wall_box')}",
                flush=True,
            )
    if "segmented_p_tube_certificate" in report:
        tube = report["segmented_p_tube_certificate"]
        print(
            "segmented p-tube certificate: "
            f"status={tube['status']}, A={tube['candidate_A']:.9g}, "
            f"p=[{tube['start_p']}, {tube['end_p']}], "
            f"certified_to_p={tube['certified_to_p']:.9g}, "
            f"blocks={tube['blocks_certified']}",
            flush=True,
        )
        block = tube.get("failing_block")
        if block is not None:
            print(
                f"  failing_margin={block['worst_margin']:.9g}, "
                f"face={block.get('failing_face')}",
                flush=True,
            )
    if "tuned_segmented_p_tube_certificate" in report:
        tube = report["tuned_segmented_p_tube_certificate"]
        print(
            "tuned segmented p-tube certificate: "
            f"status={tube['status']}, A={tube['candidate_A']:.9g}, "
            f"p=[{tube['start_p']}, {tube['end_p']}], "
            f"certified_to_p={tube['certified_to_p']:.9g}, "
            f"blocks={tube['blocks_certified']}, "
            f"attempts={tube['tuning_attempt_count']}, "
            f"worst_margin={tube.get('worst_margin', float('nan')):.9g}",
            flush=True,
        )
        block = tube.get("failing_block")
        if block is not None:
            print(
                f"  failing_margin={block.get('worst_margin', float('nan')):.9g}, "
                f"face={block.get('failing_face')}",
                flush=True,
            )
    if "sampled_carried_c_p_tube_certificate" in report:
        tube = report["sampled_carried_c_p_tube_certificate"]
        print(
            "sampled carried-C p-tube certificate: "
            f"status={tube['status']}, A={tube['candidate_A']:.9g}, "
            f"p=[{tube['start_p']}, {tube['end_p']}], "
            f"certified_to_p={tube['certified_to_p']:.9g}, "
            f"blocks={tube['blocks_certified']}, "
            f"attempts={tube['tuning_attempt_count']}, "
            f"worst_margin={tube.get('worst_margin', float('nan')):.9g}, "
            f"conditional={tube['conditional']}",
            flush=True,
        )
        if tube.get("end_box_5d") is not None:
            end_box = tube["end_box_5d"]
            print(
                f"  end_low={end_box['low']}, "
                f"end_high={end_box['high']}",
                flush=True,
            )
        block = tube.get("failing_block")
        if block is not None:
            print(
                f"  failing_margin={block.get('worst_margin', float('nan')):.9g}, "
                f"face={block.get('failing_face')}",
                flush=True,
            )
    if "carried_c_p_tube_from_box_certificate" in report:
        tube = report["carried_c_p_tube_from_box_certificate"]
        print(
            "carried-C p-tube from box certificate: "
            f"status={tube['status']}, A={tube['candidate_A']:.9g}, "
            f"p=[{tube['start_p']}, {tube['end_p']}], "
            f"certified_to_p={tube.get('certified_to_p')}, "
            f"blocks={tube['blocks_certified']}, "
            f"attempts={tube['tuning_attempt_count']}, "
            f"worst_margin={tube.get('worst_margin', float('nan')):.9g}, "
            f"source={tube.get('source_kind')}",
            flush=True,
        )
        if tube.get("end_box_5d") is not None:
            end_box = tube["end_box_5d"]
            print(
                f"  end_low={end_box['low']}, "
                f"end_high={end_box['high']}",
                flush=True,
            )
        block = tube.get("failing_block")
        if block is not None:
            print(
                f"  failing_margin={block.get('worst_margin', float('nan')):.9g}, "
                f"face={block.get('failing_face')}",
                flush=True,
            )
    if "automatic_carried_c_p_corridor_certificate" in report:
        corridor = report["automatic_carried_c_p_corridor_certificate"]
        print(
            "automatic carried-C p-corridor certificate: "
            f"status={corridor['status']}, A={corridor['candidate_A']:.9g}, "
            f"p=[{corridor['start_p']}, {corridor['end_p']}], "
            f"certified_to_p={corridor.get('certified_to_p')}, "
            f"steps={corridor['steps_certified']}/{corridor.get('steps', corridor['steps_certified'])}, "
            f"worst_margin={corridor.get('worst_margin', float('nan')):.9g}, "
            f"source={corridor.get('source_kind', corridor.get('c_source'))}",
            flush=True,
        )
        if corridor.get("end_box_5d") is not None:
            end_box = corridor["end_box_5d"]
            print(
                f"  end_low={end_box['low']}, "
                f"end_high={end_box['high']}",
                flush=True,
            )
        face = corridor.get("failing_step", {}).get("failing_face") or corridor.get("worst_face")
        if face is not None:
            print(f"  face={face}", flush=True)
    if "carried_c_p_wall_certificate" in report:
        wall = report["carried_c_p_wall_certificate"]
        print(
            "carried-C p-wall certificate: "
            f"status={wall['status']}, A={wall['candidate_A']:.9g}, "
            f"component={wall['component']}, side={wall['side']}, value={wall['wall_value']:.9g}, "
            f"p=[{wall['start_p']}, {wall['end_p']}], "
            f"certified_to_p={wall.get('certified_to_p')}, "
            f"steps={wall['steps_certified']}, "
            f"worst_margin={wall.get('worst_margin', float('nan')):.9g}, "
            f"source_contained={wall['source_box_contained']}",
            flush=True,
        )
        face = wall.get("failing_slice") or wall.get("worst_slice")
        if face is not None:
            print(f"  slice={face}", flush=True)
    if "staged_union_p_tube_certificate" in report:
        union = report["staged_union_p_tube_certificate"]
        print(
            "staged union p-tube certificate: "
            f"status={union['status']}, A={union['candidate_A']:.9g}, "
            f"start_p={union['start_p']:.9g}, "
            f"certified_to_p={union['certified_to_p']:.9g}, "
            f"leaf_boxes={union.get('leaf_box_count', union.get('leaf_boxes_certified'))}, "
            f"blocks={union['blocks_certified']}, "
            f"attempts={union['tuning_attempt_count']}, "
            f"worst_margin={union.get('worst_margin', float('nan')):.9g}",
            flush=True,
        )
        failing = union.get("failing_certificate")
        if failing is not None:
            print(
                f"  failed_stage={union.get('stage_index')}, "
                f"parent={union.get('parent_index')}, child={union.get('child_index')}, "
                f"face={failing.get('failing_block', {}).get('failing_face') or failing.get('worst_face')}",
                flush=True,
            )
    if "adaptive_union_p_tube_certificate" in report:
        union = report["adaptive_union_p_tube_certificate"]
        print(
            "adaptive union p-tube certificate: "
            f"status={union['status']}, A={union['candidate_A']:.9g}, "
            f"p=[{union['start_p']}, {union['end_p']}], "
            f"certified_to_p={union.get('certified_to_p')}, "
            f"source_boxes={union['source_box_count']}, "
            f"certified_leaves={union['certified_leaf_box_count']}, "
            f"failed_leaves={union['failed_leaf_box_count']}, "
            f"queued={union['remaining_queue_count']}, "
            f"splits={union['split_count']}, "
            f"processed={union['processed_boxes']}, "
            f"blocks={union['blocks_certified']}, "
            f"attempts={union['tuning_attempt_count']}, "
            f"worst_margin={union.get('worst_margin', float('nan')):.9g}",
            flush=True,
        )
        if union.get("stopped_reason") is not None:
            print(f"  stopped_reason={union['stopped_reason']}", flush=True)
        if union.get("failed_leaves"):
            first_failure = union["failed_leaves"][0]
            print(
                f"  first_failed_source={first_failure['source_index']}, "
                f"depth={first_failure['depth']}, "
                f"certified_to_p={first_failure.get('certified_to_p')}, "
                f"face={first_failure.get('failing_face')}",
                flush=True,
            )
    if "adaptive_carried_c_union_p_tube_certificate" in report:
        union = report["adaptive_carried_c_union_p_tube_certificate"]
        print(
            "adaptive carried-C union p-tube certificate: "
            f"status={union['status']}, A={union['candidate_A']:.9g}, "
            f"p=[{union['start_p']}, {union['end_p']}], "
            f"certified_to_p={union.get('certified_to_p')}, "
            f"source_boxes={union['source_box_count']}, "
            f"certified_leaves={union['certified_leaf_box_count']}, "
            f"failed_leaves={union['failed_leaf_box_count']}, "
            f"queued={union['remaining_queue_count']}, "
            f"splits={union['split_count']}, "
            f"processed={union['processed_boxes']}, "
            f"blocks={union['blocks_certified']}, "
            f"attempts={union['tuning_attempt_count']}, "
            f"worst_margin={union.get('worst_margin', float('nan')):.9g}",
            flush=True,
        )
        if union.get("stopped_reason") is not None:
            print(f"  stopped_reason={union['stopped_reason']}", flush=True)
        if union.get("end_hull_5d") is not None:
            print(f"  C_interval={union['end_hull_5d']['low'][4], union['end_hull_5d']['high'][4]}", flush=True)
        if union.get("failed_leaves"):
            first_failure = union["failed_leaves"][0]
            print(
                f"  first_failed_source={first_failure['source_index']}, "
                f"depth={first_failure['depth']}, "
                f"certified_to_p={first_failure.get('certified_to_p')}, "
                f"face={first_failure.get('failing_face')}",
                flush=True,
            )
    if "affine_p_corridor_certificate" in report:
        corridor = report["affine_p_corridor_certificate"]
        print(
            "affine p-corridor certificate: "
            f"status={corridor['status']}, A={corridor['candidate_A']:.9g}, "
            f"p=[{corridor['start_p']}, {corridor['end_p']}], "
            f"certified_to_p={corridor['certified_to_p']:.9g}, "
            f"steps={corridor['steps_certified']}/{corridor['steps']}, "
            f"worst_margin={corridor['worst_margin']:.9g}",
            flush=True,
        )
        print(
            f"  source_box_contained={corridor['source_box_contained']}, "
            f"face={corridor.get('failing_face') or corridor.get('worst_face')}",
            flush=True,
        )
    if "affine_p_corridor_tuning" in report:
        tuning = report["affine_p_corridor_tuning"]
        best = tuning["best"]
        if best is None:
            print(f"affine p-corridor tuning: runs={tuning['runs']}, status=no_runs", flush=True)
        else:
            print(
                "affine p-corridor tuning: "
                f"runs={tuning['runs']}, best_status={best['status']}, "
                f"best_x2_slope={best['x2_lower_slope']:.9g}, "
                f"best_x1_upper_slope={best['x1_upper_slope']:.9g}, "
                f"certified_to_p={best['certified_to_p']:.9g}, "
                f"worst_margin={best['worst_margin']:.9g}",
                flush=True,
            )
            print(f"  face={best['face']}", flush=True)
    if "terminal_barrier_takeover_certificate" in report:
        takeover = report["terminal_barrier_takeover_certificate"]
        print(
            "terminal barrier takeover: "
            f"status={takeover['status']}, A={takeover['candidate_A']:.9g}, "
            f"p=[{takeover['p_start']}, {takeover['p_min']}], "
            f"x3_wall={takeover['x3_wall']:.9g}, "
            f"worst_margin={takeover.get('worst_margin', float('nan')):.9g}",
            flush=True,
        )
        print(
            f"  source_box_contained={takeover['source_box_contained']}, "
            f"source_below_wall={takeover['source_below_wall']}, "
            f"source_x2_floor={takeover['source_x2_floor']}, "
            f"x3_zero_margin={takeover['x3_zero_margin']:.9g}, "
            f"slice={takeover.get('failing_slice') or takeover.get('worst_slice')}",
            flush=True,
        )
        small_tail = takeover["small_p_tail"]
        print(
            "  small-p tail: "
            f"p_prime_margin={small_tail['p_prime_negative_coefficient_margin']:.9g}, "
            f"x3_prime_margin_at_p_min={small_tail['x3_prime_negative_margin_at_p_min']:.9g}",
            flush=True,
        )
    if "p_tube_frontier_continuation_certificate" in report:
        continuation = report["p_tube_frontier_continuation_certificate"]
        print(
            "p-tube frontier continuation: "
            f"status={continuation['status']}, A={continuation['candidate_A']:.9g}, "
            f"p=[{continuation['start_p']}, {continuation['end_p']}], "
            f"certified_to_p={continuation['certified_to_p']:.9g}, "
            f"blocks={continuation['blocks_certified']}, "
            f"worst_margin={continuation.get('worst_margin', float('nan')):.9g}",
            flush=True,
        )
        block = continuation.get("failing_block")
        if block is not None:
            print(
                f"  failing_margin={block['worst_margin']:.9g}, "
                f"face={block.get('failing_face')}",
                flush=True,
            )
    if "hybrid_p_frontier_handoff_certificate" in report:
        handoff = report["hybrid_p_frontier_handoff_certificate"]
        print(
            "hybrid p-frontier handoff: "
            f"status={handoff['status']}, A={handoff['candidate_A']:.9g}, "
            f"conditional={handoff['conditional']}",
            flush=True,
        )
        if handoff["status"] == "certified_conditional":
            tube = handoff["p_tube"]
            corridor = handoff["affine_corridor"]
            print(
                "  stages: "
                f"p_tube_to={tube['certified_to_p']:.9g}, "
                f"corridor_to={corridor['certified_to_p']:.9g}, "
                f"frontier_x3_high={handoff['frontier_box']['high'][3]:.9g}, "
                f"corridor_margin={corridor['worst_margin']:.9g}",
                flush=True,
            )
        else:
            stage = handoff.get("stage")
            failing = handoff.get("p_tube") if stage == "p_tube" else handoff.get("affine_corridor")
            face = None if failing is None else failing.get("failing_face") or failing.get("worst_face")
            print(f"  failed_stage={stage}, face={face}", flush=True)
    if "p_start_slice_from_support_certificate" in report:
        bridge = report["p_start_slice_from_support_certificate"]
        print(
            "p-start slice bridge: "
            f"status={bridge['status']}, A={bridge['candidate_A']:.9g}, "
            f"conditional={bridge['conditional']}",
            flush=True,
        )
        print(
            "  checks: "
            f"before_above={bridge['before_above_target']}, "
            f"after_below={bridge['after_below_target']}, "
            f"contained={bridge['crossing_slab_contained_in_start_slice']}",
            flush=True,
        )
    if "piecewise_affine_p_corridor_certificate" in report:
        piecewise = report["piecewise_affine_p_corridor_certificate"]
        print(
            "piecewise affine p-corridor: "
            f"status={piecewise['status']}, A={piecewise['candidate_A']:.9g}, "
            f"certified_to_p={piecewise['certified_to_p']:.9g}, "
            f"segments={piecewise['segments_certified']}, "
            f"worst_margin={piecewise.get('worst_margin', float('nan')):.9g}",
            flush=True,
        )
        failing = piecewise.get("failing_segment")
        if failing is not None:
            print(
                f"  failing_segment={piecewise['failing_segment_index']}, "
                f"face={failing.get('failing_face') or failing.get('worst_face')}",
                flush=True,
            )
    if "late_tail_closure_certificate" in report:
        closure = report["late_tail_closure_certificate"]
        print(
            "late-tail closure certificate: "
            f"status={closure['status']}, A={closure['candidate_A']:.9g}, "
            f"conditional={closure['conditional']}",
            flush=True,
        )
        if closure["status"] == "certified_conditional":
            continuation = closure["frontier_continuation"]
            piecewise = closure["piecewise_corridor"]
            terminal = closure["terminal_takeover"]
            print(
                "  stages: "
                f"frontier_to_p={continuation['certified_to_p']:.9g}, "
                f"piecewise_to_p={piecewise['certified_to_p']:.9g}, "
                f"terminal_wall={terminal['x3_wall']:.9g}, "
                f"terminal_margin={terminal['worst_margin']:.9g}",
                flush=True,
            )
        else:
            print(f"  failed_stage={closure.get('stage')}", flush=True)
    if "broad_tail_closure_certificate" in report:
        closure = report["broad_tail_closure_certificate"]
        print(
            "broad-tail closure certificate: "
            f"status={closure['status']}, A={closure['candidate_A']:.9g}, "
            f"conditional={closure['conditional']}",
            flush=True,
        )
        if closure["status"] == "certified_conditional":
            handoff = closure["hybrid_handoff"]
            automatic = closure["automatic_corridor"]
            terminal = closure["terminal_takeover"]
            print(
                "  stages: "
                f"handoff_to_p={handoff['certified_to_p']:.9g}, "
                f"automatic_to_p={automatic['certified_to_p']:.9g}, "
                f"automatic_steps={automatic['steps_certified']}, "
                f"automatic_margin={automatic['worst_margin']:.9g}, "
                f"terminal_margin={terminal['worst_margin']:.9g}",
                flush=True,
            )
        else:
            print(f"  failed_stage={closure.get('stage')}", flush=True)
    if "support_tail_closure_certificate" in report:
        closure = report["support_tail_closure_certificate"]
        print(
            "support-tail closure certificate: "
            f"status={closure['status']}, A={closure['candidate_A']:.9g}, "
            f"conditional={closure['conditional']}",
            flush=True,
        )
        if closure["status"] == "certified_conditional":
            bridge = closure["p_start_slice_bridge"]
            broad = closure["broad_tail"]
            automatic = broad["automatic_corridor"]
            terminal = broad["terminal_takeover"]
            print(
                "  stages: "
                f"bridge_after_t={bridge['after_time']:.9g}, "
                f"broad_from_p={broad['certified_from_p']:.9g}, "
                f"automatic_to_p={automatic['certified_to_p']:.9g}, "
                f"terminal_margin={terminal['worst_margin']:.9g}",
                flush=True,
            )
        else:
            print(f"  failed_stage={closure.get('stage')}", flush=True)


if __name__ == "__main__":
    main()
