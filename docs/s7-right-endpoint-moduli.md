# S7 Right-Endpoint Moduli Charts

This note derives the terminal offset moduli for the two S7 right-end weighted
charts used by the codebase.  The derivation is deliberately tied to the
implemented `q_rhs` system, not to a separate writeup.

## Setup

Write

```text
tau = T - t
q_i(tau) = Q_i + tau^{w_i} y_i(tau)
```

at the right endpoint.  The raw ODE uses the three branch sums

```text
A = q2 + q7,
B = q3 + q6,
C = q4 + q5,
```

and

```text
p1 = -sqrt(-(A B)/(lambda C))
p2 =  sqrt(-(A C)/(lambda B))
p3 =  sqrt(-(B C)/(lambda A)).
```

The singular denominator in the rational part of every equation is
`p1 p2 p3`.  Therefore, at a collapsing endpoint, the leading numerator cores

```text
N_i = alpha_i (2 alpha_i - alpha_sum) + 2 beta_j
```

must vanish at the terminal offset `Q`.  In the implemented component order,

```text
alpha1 = q1 q8
alpha2 = q2 q7
alpha3 = q3 q6
alpha4 = q4 q5
beta1  = q1 q4 q6 q7
beta2  = q2 q3 q5 q8.
```

The eight leading cores are exactly the polynomial factors returned by
`experiments.s7.right_moduli_chart.leading_core_residual`.

## The p3 Chart

For a `p3` collapse, `B = q3 + q6` and `C = q4 + q5` vanish at the endpoint,
while `A = q2 + q7` remains nonzero.  Thus set

```text
q6 = -q3,
q5 = -q4.
```

Substituting these identities into the eight leading cores and staying on the
nonzero branch gives the relations

```text
q2 = -q1,
q3 = -q4,
q5 = q3,
q6 = q4,
q8 = -q7.
```

Therefore the regular p3 terminal offsets form the three-parameter family

```text
Q_p3(A,B,C) = ( A, -A, -B,  B, -B,  B,  C, -C).
```

The non-collapsing branch sum is

```text
q2 + q7 = C - A,
```

so a genuine p3 chart requires `C != A`.  If

```text
U = y3(0) + y6(0),
V = y4(0) + y5(0),
```

then the leading product condition is

```text
-(C - A) U V > 0
```

for `lambda > 0`.

The known round S7 endpoint is the special point

```text
A = sqrt(5)/25,
B = 2 sqrt(5)/25,
C = 19 sqrt(5)/25.
```

This gives

```text
(1, -1, -2, 2, -2, 2, 19, -19) * sqrt(5)/25.
```

## The p2 Chart

For a `p2` collapse, `A = q2 + q7` and `C = q4 + q5` vanish at the endpoint,
while `B = q3 + q6` remains nonzero.  Thus set

```text
q7 = -q2,
q5 = -q4.
```

Substituting these identities into the leading cores and staying on the
nonzero branch gives

```text
q3 = -q1,
q2 = -q4,
q5 = -q4,
q6 = -q8,
q7 = q4.
```

Therefore the regular p2 terminal offsets form the three-parameter family

```text
Q_p2(A,B,C) = ( A, -B, -A,  B, -B,  C,  B, -C).
```

The non-collapsing branch sum is

```text
q3 + q6 = C - A,
```

so a genuine p2 chart requires `C != A`.  If

```text
U = y2(0) + y7(0),
V = y4(0) + y5(0),
```

then the leading product condition is again

```text
-(C - A) U V > 0
```

for `lambda > 0`.

The known squashed S7 endpoint is the special point

```text
A = sqrt(5)/25,
B = 2 sqrt(5)/25,
C = 19 sqrt(5)/25.
```

This gives

```text
(1, -2, -1, 2, -2, 19, 2, -19) * sqrt(5)/25.
```

## Solving The Taylor Coefficients

The right endpoint moduli are not first-jet moduli at fixed offset.  The
terminal offset itself already carries the expected three right-end parameters.
This explains why the provisional fixed-offset first-jet scout was only a
diagnostic chart: it varied the wrong layer of endpoint data.

After choosing `(A,B,C)`, the weighted zero jet and first jet are not independent
coordinates.  They are part of the same regularity problem as the higher Taylor
coefficients.  In these charts the coefficient equations are coupled across
Taylor levels, so the one-layer recurrence used by the Berger endpoint is not a
valid solver.

The implemented solver therefore fixes the terminal offset using `Q_p2` or
`Q_p3`, then solves the whole weighted Taylor coefficient block simultaneously.
For a truncated order `N` it starts from the known homogeneous S7 coefficient
block, treats all `8(N+1)` weighted coefficients as unknowns, and minimizes the
sampled local ODE residual

```text
d/dtau [Q + tau^w y(tau)] - local_q_rhs(tau, Q + tau^w y(tau)).
```

The exact round and squashed points are still returned without optimization.
Small offset perturbations around both endpoints solve to sampled residual below
`1e-8` in the test suite.

This gives a usable right-end germ constructor for the honest S7 full-moduli
search.  It is a numerical simultaneous coefficient solve, not yet a symbolic
closed-form recurrence.
