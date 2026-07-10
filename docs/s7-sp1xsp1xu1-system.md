# S7 Sp(1) x Sp(1) x U(1) System

Reproducibility command:

```zsh
.venv/bin/python -m experiments.s7_sp1xsp1xu1_system
.venv/bin/python -m experiments.s7_sp1xsp1xu1_system --json
```

## Principal Orbit Algebra

Use the principal-orbit coframe

```text
a1, a2, a3, b1, b2, b3
```

where the diagonal `U(1)` rotates `(a1,a2)` and `(b1,b2)` and fixes
`a3,b3`.  The Maurer-Cartan normalization is fixed by the Podesta subchart:

```text
da1 = 6 a2^a3      db1 = 6 b2^b3
da2 = 6 a3^a1      db2 = 6 b3^b1
da3 = 6 a1^a2      db3 = 6 b1^b2
```

The invariant principal-orbit `SU(3)` variables are

```text
omega =
  x1 a12
  + x2 b12
  + x3 a3b3
  + x4 (a1b1 + a2b2)
  + x5 (a1b2 - a2b1),
```

and

```text
gamma =
  y1 a123
  + y2 a3b12
  + y3 a3^(a1b1 + a2b2)
  + y4 a3^(a1b2 - a2b1)
  + y5 b3a12
  + y6 b123
  + y7 b3^(a1b1 + a2b2)
  + y8 b3^(a1b2 - a2b1).
```

So the raw invariant ansatz has `5 + 8 = 13` functions.

## ODE System

For

```text
phi = dt^omega + gamma,
psi = 1/2 omega^2 - dt^hat(gamma),
```

with `hat(gamma)` chosen so that

```text
gamma^hat(gamma) = 4 omega^3/6,
```

the nearly parallel equations are

```text
d_6 gamma = (lambda/2) omega^2,
omega^gamma = 0,
omega^3/6 = sqrt(-lambda_Hitchin(gamma))/2,
dot(gamma) = d_6 omega - lambda hat(gamma),
omega^dot(omega) = -d_6 hat(gamma).
```

The last equation is solved in the five-dimensional invariant two-form basis.
This is the explicit 13-variable ODE layer implemented in
`experiments.s7.sp1xsp1xu1_system`.

The orientation choice for `hat(gamma)` is essential.  The round target uses
the raw Hitchin branch, while the squashed target uses the opposite branch in
this coframe.

The polynomial part of the system is already quite restrictive.  In the
`gamma` basis,

```text
d omega =
  -6 x3 a3b12
  -6 x5 a3_delta
  +6 x4 a3_epsilon
  +6 x3 b3a12
  +6 x5 b3_delta
  -6 x4 b3_epsilon.
```

The algebraic nearly-parallel equation gives

```text
-lambda x1 x3 = 0,
6(y2+y5) - lambda x1 x2 + lambda(x4^2+x5^2) = 0,
6(y4+y8) + lambda x3 x4 = 0,
6(y3+y7) - lambda x3 x5 = 0,
-lambda x2 x3 = 0.
```

Compatibility gives the two five-form equations

```text
x1 y2 + x2 y1 - 2 x4 y3 - 2 x5 y4 = 0,
x1 y6 + x2 y5 - 2 x4 y7 - 2 x5 y8 = 0.
```

Thus on the regular nearly-parallel branch with `lambda != 0` and `x3 != 0`,
the algebraic constraint already forces

```text
x1 = x2 = 0.
```

So `13` is the raw invariant form count, not the number of unconstrained ODE
degrees of freedom.  After `x1=x2=0`, the `d gamma` equation gives three sum
relations among `(y2,y5)`, `(y4,y8)`, and `(y3,y7)`.  The two compatibility
equations then become one independent equation, and the volume normalization
adds one more scalar condition.  Thus the regular branch has six algebraic
degrees before endpoint Taylor recurrences are imposed.

## Podesta Subchart

Podesta's five-function chart embeds as

```text
omega = f0 (a3b3 + a1b1 + a2b2),
gamma =
  f1 a123
  + f2 b123
  + f3 (b3a12 + a3^(a1b2 - a2b1))
  + f4 (a3b12 + b3^(a1b2 - a2b1)).
```

Equivalently,

```text
x = (0, 0, f0, f0, 0),
y = (f1, f4, 0, f3, f3, f2, 0, f4).
```

With the `6` in the Maurer-Cartan table, the algebraic equation restricts to

```text
f3 + f4 + lambda f0^2/6 = 0,
```

matching the existing Podesta audit.

## Endpoint Conditions

At the left `K+` endpoint the `a` directions collapse and the `b` directions
survive.  A Podesta-compatible regular chart is

```text
x1=t^4 X1, x2=t X2, x3=t X3, x4=t X4, x5=t^3 X5,
y1=t^4 Y1, y2=t^2 Y2, y3=t^3 Y3, y4=t^2 Y4,
y5=t^2 Y5, y6=Y6, y7=t^2 Y7, y8=t^2 Y8.
```

The Podesta subchart is recovered by

```text
x3=x4=f0,
y1=f1,
y2=y8=f4,
y4=y5=f3,
y6=f2,
other variables = 0,
```

with the familiar leading relation

```text
Y4(0) = Y5(0) = 3 X3(0) = 3 X4(0).
```

At the right `K-` endpoint the roles of `a` and `b` are swapped:

```text
x1=t X1, x2=t^4 X2, x3=t X3, x4=t X4, x5=t^3 X5,
y1=Y1, y2=t^2 Y2, y3=t^2 Y3, y4=t^2 Y4,
y5=t^2 Y5, y6=t^4 Y6, y7=t^3 Y7, y8=t^2 Y8.
```

The endpoint smoothness layer is not yet a full Taylor recurrence.  The current
module records the linear Eschenburg-Wang smooth-jet dimensions through order
six:

```text
order:       0  1  2  3   4   5   6
dimension:   1  1  5  9  17  25  41
new:         1  0  4  4   8   8  16
```

These are the smoothness degrees before imposing the nearly parallel ODE
recurrence.

## Max-Volume Matching

The practical endpoint layer is implemented in
`experiments.s7.sp1xsp1xu1_matching`.  It uses the endpoint weights above,
fixes five leading regular constants at each endpoint,

```text
A3 = X3(0), A4 = X4(0), B2 = Y2(0), B4 = Y4(0), C = surviving-volume coefficient,
```

and builds in the leading algebraic relations

```text
Y5(0) = -lambda A4^2/6 - B2,
Y8(0) = -lambda A3 A4/6 - B4.
```

Higher Taylor coefficients are fitted numerically against the full ODE and
algebraic residuals.  A left endpoint is marched with `dot u = F(u)`, while a
right endpoint is marched in the inward coordinate with `dot u = -F(u)`.

The matching section is the first stationary point of the positively oriented
principal volume `abs(omega^3/6)`.  The two one-sided states are compared at
that max-volume section.

Known exact endpoint germs recover the two homogeneous examples:

```zsh
.venv/bin/python -m experiments.s7_sp1xsp1xu1_matching --recover-known
```

Current diagnostics:

```text
round:    residual about 1e-9, interval pi/2, left/right tau pi/4
squashed: residual about 3e-9, interval pi/2, left/right tau pi/4
```

The first actual exploratory runner is target-independent by default: it samples
the ten endpoint parameters in an absolute box at fixed `lambda`.  Use
`--include-known-controls` to prepend the round and homothetically rescaled
squashed endpoint data as controls.  The `--target round` or
`--target squashed` modes are only local known-solution refinement modes.

```zsh
.venv/bin/python -m experiments.s7_sp1xsp1xu1_scout --dry-run --samples 3 --radius 40 --include-known-controls
.venv/bin/python -m experiments.s7_sp1xsp1xu1_scout --samples 2 --radius 40 --include-known-controls --workers 1
```

It writes JSONL and summary output under `output/s7_sp1xsp1xu1_scouts/`.

The broad overnight form is:

```zsh
.venv/bin/python -m experiments.s7_sp1xsp1xu1_scout \
  --samples 400 \
  --radius 40 \
  --include-known-controls \
  --workers 4 \
  --endpoint-order 3 \
  --max-germ-evaluations 100 \
  --max-step 0.02 \
  --progress-every 5
```

## Calibration

The module embeds the existing round and squashed `S7` Podesta targets and
checks the full 13-variable system at two regular sample times.  Current smoke
residuals are below `1e-60` for both targets.
