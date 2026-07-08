# Aloff-Wallach N11 Ansatz Note

Reproducibility command:

```zsh
.venv/bin/python -m experiments.aloff_wallach_ansatz
.venv/bin/python -m experiments.aloff_wallach_ansatz --json
```

## Purpose

This note records the first concrete ansatz checkpoint for the exceptional
Aloff-Wallach space

```text
N_{1,1} = SU(3) / U(1)_{1,1}.
```

The feasibility audit identified a plausible cohomogeneity-one action

```text
SO(3)_real x SO(3)_fiber.
```

The principal orbit is locally `SO(3) x SO(3)` modulo a finite principal
isotropy. The two singular base orbits come from the real `SO(3)` action on
`CP^2`: the real-line orbit `RP^2` and the null-conic orbit `CP^1`.

At a regular point the principal isotropy is modeled by the diagonal half-turn

```text
(-, -, +, -, -, +)
```

on a local coframe

```text
base_1, base_2, base_3, fiber_1, fiber_2, fiber_3.
```

Thus the principal-orbit `SU(3)` structure starts from

```text
omega(t) in a 7-dimensional invariant 2-form space,
gamma(t) in a 12-dimensional invariant 3-form space,
```

before imposing the algebraic `SU(3)` compatibility and positivity conditions.

This is not yet a full scouting ODE. It is the calibration layer: verify the
`A,B,C,D` homogeneous `N_{1,1}` nearly parallel structures and record how the
extra Sasaki-Einstein structure fits into the same cohomogeneity-one symmetry.

## Full Principal-Orbit Variables

Use the ordered model coframe

```text
base_1, base_2, base_3, fiber_1, fiber_2, fiber_3.
```

For comparison with the Ball-Oliveira homogeneous coframe, the model regular
orbit convention is

```text
base_1  = omega_2,
base_2  = omega_6,
base_3  = omega_3,
fiber_1 = omega_1,
fiber_2 = omega_5,
fiber_3 = omega_4,
normal  = omega_7.
```

The principal half-turn has signs

```text
(-, -, +, -, -, +),
```

so the full invariant `SU(3)` pair is

```text
omega =
  x1 base_1^base_2
  + x2 base_1^fiber_1
  + x3 base_1^fiber_2
  + x4 base_2^fiber_1
  + x5 base_2^fiber_2
  + x6 base_3^fiber_3
  + x7 fiber_1^fiber_2,

gamma =
  y1  base_1^base_2^base_3
  + y2  base_1^base_2^fiber_3
  + y3  base_1^base_3^fiber_1
  + y4  base_1^base_3^fiber_2
  + y5  base_1^fiber_1^fiber_3
  + y6  base_1^fiber_2^fiber_3
  + y7  base_2^base_3^fiber_1
  + y8  base_2^base_3^fiber_2
  + y9  base_2^fiber_1^fiber_3
  + y10 base_2^fiber_2^fiber_3
  + y11 base_3^fiber_1^fiber_2
  + y12 fiber_1^fiber_2^fiber_3.
```

The two explicit `omega wedge gamma = 0` equations are

```text
x1*y11 + x2*y8 - x3*y7 - x4*y4 + x5*y3 + x7*y1 = 0,
x1*y12 - x2*y10 + x3*y9 + x4*y6 - x5*y5 + x7*y2 = 0.
```

The remaining algebraic condition is Hitchin stability and normalization:

```text
K_gamma^2 = lambda(gamma) I,
lambda(gamma) < 0,
|omega^3 / 6| = sqrt(-lambda(gamma)) / 2.
```

Thus there are `19` raw invariant coefficients and a `16`-dimensional
invariant `SU(3)` locus. The command computes these checks numerically for
candidate pairs.

## Homogeneous Calibration Family

Following Ball-Oliveira, use the `SU(3)/U(1)` coframe `omega_1,...,omega_7` and
the homogeneous family

```text
phi =
  A B C (omega_123 - omega_167 + omega_257 - omega_356)
  - D omega_4 wedge (A^2 omega_15 + B^2 omega_26 + C^2 omega_37).
```

The induced metric is

```text
g =
  A^2 (omega_1^2 + omega_5^2)
  + B^2 (omega_2^2 + omega_6^2)
  + C^2 (omega_3^2 + omega_7^2)
  + D^2 omega_4^2.
```

For the `SO(3)` bundle picture over `CP^2`, the vertical directions are

```text
omega_1, omega_4, omega_5,
```

and the horizontal directions are

```text
omega_2, omega_6, omega_3, omega_7.
```

The subfamily

```text
C = B,
D = A
```

contains the tri-Sasakian nearly parallel structure. The squashed strict nearly
parallel structure is obtained from the same sign-square pattern with

```text
C^2 = B^2,
D^2 = A^2,
A^2 = 2 B^2 / 5,
A B C D < 0.
```

## Verified Known Points

The command checks `d phi - lambda psi` from the reductive
`SU(3)/U(1)_{1,1}` bracket.

With `B=C=1`:

```text
tri-Sasakian:
  A = sqrt(2)
  D = sqrt(2)
  lambda = 2

squashed strict nearly parallel:
  A = sqrt(2/5)
  D = -sqrt(2/5)
  lambda = 6/sqrt(5)
```

Both residuals are at the `1e-15` level, limited by the matrix-bracket
projection used in the lightweight verifier.

With model normal `omega_7`, the corresponding principal-orbit `SU(3)` pairs
also pass the algebraic checks:

```text
omega wedge gamma residual: 0
Hitchin complex residual:   ~1e-15 or better
volume normalization:       ~1e-15 or better
```

## Extra Sasaki-Einstein Structure

Ball-Oliveira also record a further nearly parallel structure associated with
the tri-Sasakian metric:

```text
phi_ts =
  - eta_123
  + (s / 48) (eta_1 wedge omega_1
            + eta_2 wedge omega_2
            + eta_3 wedge omega_3),

d phi_ts = 4 psi_ts.
```

This is not obtained by a hidden `D`-sign branch in the `A,B,C,D` family. For
the same square pattern as the tri-Sasakian-metric point,

```text
A^2 = 2 B^2, C^2 = B^2, D^2 = A^2,
```

the valid branch has best residual around `1e-16`, while replacing
`D = sqrt(2)` by `D = -sqrt(2)` gives best possible
`d phi = lambda psi` residual about `3.84`. The issue is structural: the
`A,B,C,D` family locks the vertical-volume coefficient to the mixed contraction
coefficients, whereas `phi_ts` changes that relative sign independently.

However, this does not require a different cohomogeneity-one action. Both

```text
eta_123
sum_i eta_i wedge omega_i
```

are invariant contractions for the fiber `SO(3)` rotating the vertical
connection triple and the self-dual curvature triple together. Hence the extra
Sasaki-Einstein form belongs to the same `SO(3)_real x SO(3)_fiber`
principal-orbit invariant `SU(3)` variable space, rather than to a different
cohomogeneity-one action.

The command also verifies the standard Sasaki-Einstein nearly-parallel form
from Geipel's source coframe. In that coframe,

```text
eta = e7,
omega = e12 + e34 + e56,
Theta^1 = e1 - i e2,
Theta^2 = e3 - i e4,
Theta^3 = e5 - i e6,
phi_SE = eta wedge omega + Re(Theta^1 wedge Theta^2 wedge Theta^3),
```

and the structure equations give

```text
d phi_SE = 4 psi_SE
```

with zero residual in the symbolic form algebra. The coframe conversion is a
Weyl-adjusted diagonal map to the Ball-Oliveira coframe:

```text
e1 =  omega_2 / 2,
e2 = -omega_6 / 2,
e3 =  omega_3 / 2,
e4 = -omega_7 / 2,
e5 = -omega_1 / sqrt(2),
e6 = -omega_5 / sqrt(2),
e7 = -omega_4 / sqrt(2).
```

One representative becomes

```text
phi_SE =
  1/(4 sqrt(2)) (
    - omega_123
    + 2 omega_145
    + omega_167
    - omega_246
    + omega_257
    - omega_347
    - omega_356
  ),
```

with metric coframe scales

```text
(1/sqrt(2), 1/2, 1/2, 1/sqrt(2), 1/sqrt(2), 1/2, 1/2)
```

on `omega_1,...,omega_7`. In this metric the model normal is still `omega_7`,
but the unit normal coform is `(1/2) omega_7`. With that normalization, the
real/imaginary phase representatives and both signs tested by the command all
pass:

```text
d phi_SE - 4 psi_SE:        ~1e-16
omega wedge gamma residual: 0
Hitchin complex residual:   0
volume normalization:       ~1e-82
```

## Next Step

The next mathematical/code task is to derive the evolution equations and
endpoint smoothness conditions in the variables above. This verifier now
supplies the two `A,B,C,D` known targets plus the extra Sasaki-Einstein target
for recovery calibration.
