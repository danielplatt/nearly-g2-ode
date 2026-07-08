# S7 SU(2)^3 Action Audit

Reproducibility command:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_action_audit
.venv/bin/python -m experiments.s7_su2_cubed_action_audit --json
```

## Result

There is a genuinely new `S7` cohomogeneity-one action worth separating from
the existing `SO(4)/Z_2^2` `q_i` action:

```text
G = Sp(1)^3
H = diagonal Sp(1)
K+ = {(q, q', q)}
K- = {(q, q', q')}
gamma(t) = (cos t, sin t) in H^2, 0 <= t <= pi/2
```

The principal orbit is

```text
G/H ~= S3 x S3
```

and both singular orbits are `S3`, with normal slice `R4`.  The topology is
not merely inferred: this is the explicit restriction of the standard
`Sp(2) x Sp(1)` action to the unit sphere `S7` in `H^2`, so the compact
cohomogeneity-one space is exactly

```text
S7 = G x_{K+} D4 union_{G/H} G x_{K-} D4.
```

## Invariant Form Chart

Podesta's invariant chart has one invariant principal-orbit 2-form

```text
omega = e25 + e36 + e47
```

and every invariant 3-form on the regular part is

```text
phi =
  f0 (e125 + e136 + e147)
  + f1 e234
  + f2 e567
  + f3 (e237 - e246 + e345)
  + f4 (e267 - e357 + e456).
```

This is much smaller and cleaner than the old eight-variable `q_i` system, but
it is a different ansatz and should get its own marcher.

## Endpoint Smoothness

At the `K+` endpoint, `t=0`, smoothness and positivity are equivalent to:

```text
f0 odd
f1,f2,f3,f4 even
f1(0)=f3(0)=f4(0)=0
f1''(0)=0
6 f0'(0)=f3''(0)
f2(0) != 0
f0'(0) != 0
f2(0) f0'(0) < 0
```

The regular variables are

```text
f0 = t h0
f1 = t^4 h1
f2 = h2
f3 = t^2 h3
f4 = t^2 h4
```

with singular initial data

```text
h0(0)=a
h1(0)=27 lambda / 4
h2(0)=-a^3 / 27
h3(0)=3a
h4(0)=-3a - lambda a^2 / 6
```

for `a != 0`.  The `K-` endpoint uses the same conditions after applying

```text
g0(s)=f0(pi/2-s)
g1(s)=f2(pi/2-s)
g2(s)=f1(pi/2-s)
g3(s)=f4(pi/2-s)
g4(s)=f3(pi/2-s).
```

## Literature

Podesta proves that this action admits a one-parameter family of
`SU(2)^3`-invariant nearly parallel `G2` structures on the one-ended manifold
`S3 x R4`.  The family connects the two locally homogeneous structures induced
from the known round and squashed nearly parallel structures on `S7`.

The compactification question is exactly the interesting one for us: Podesta
records numerical evidence that no compact `S7` extension exists beyond the
homogeneous cases, but does not prove this.  That makes this action a good
next numerical target.

Related pointers:

- Hoelscher gives the low-dimensional cohomogeneity-one classification context.
- Cvetic-Gibbons-Lu-Pope study related cohomogeneity-one `G2` and `Spin(7)`
  holonomy systems, including principal `S3 x S3` cases.
- Podesta notes that any hypothetical nonhomogeneous compact extension would
  be distinct from Boehm's cohomogeneity-one Einstein metrics on `S7`.

## Smoke Test

The audit command checks both known homogeneous solutions:

- round `S7`, `lambda=4`;
- squashed `S7`, `lambda=12/sqrt(5)`.

For both, it verifies the five-function nearly-parallel residuals at two
regular sample points and verifies the endpoint smoothness residuals at both
singular `S3` orbits.  The current smoke residuals are below `1e-60`.
