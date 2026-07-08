# Aloff-Wallach Feasibility Note

Reproducibility command:

```zsh
.venv/bin/python -m experiments.aloff_wallach_feasibility
.venv/bin/python -m experiments.aloff_wallach_feasibility --json
```

## Question

We want to know whether Aloff-Wallach spaces are a good next topology after
Berger, `S^7`, and the Stiefel feasibility obstruction.

The spaces are

```text
N_{k,l} = SU(3) / S^1_{k,l}.
```

They are attractive because the literature contains homogeneous nearly parallel
`G_2` structures on Aloff-Wallach spaces, and `N_{1,1}` is a particularly
special example with 3-Sasakian/Sasaki-Einstein geometry.

## Literature Anchor

Ball and Oliveira study invariant `G_2`-instantons for homogeneous coclosed
`G_2` structures on Aloff-Wallach spaces and distinguish strictly nearly
parallel structures on `X_{1,1}` from the tri-Sasakian one.

Aleshin computes invariants of homogeneous nearly parallel `G_2` structures on
`N_{k,l}=SU(3)/S^1_{k,l}`, including the two homogeneous structures usually
denoted `phi^+` and `phi^-`, and compares them with the structures arising from
the 3-Sasakian geometry.

References:

```text
Ball-Oliveira, Gauge theory on Aloff-Wallach spaces, arXiv:1610.04557.
Aleshin, The bar-nu invariant of G2-structures on Aloff-Wallach spaces, arXiv:2604.04605.
```

## Generic Aloff-Wallach Spaces

For generic `N_{k,l}`, the calibrated homogeneous action is the transitive
`SU(3)` action:

```text
dim SU(3) = 8,
dim S^1_{k,l} = 1,
dim N_{k,l} = 7,
cohomogeneity = 0.
```

A cohomogeneity-one endpoint-volume search needs six-dimensional principal
orbits. The usual connected proper subgroups preserving the homogeneous setup
are too small:

```text
S(U(2)U(1)): dimension 4.
SO(3): dimension 3.
T^2: dimension 2.
```

So the generic Aloff-Wallach family is not immediately ready for our
endpoint-volume workflow. This is not a mathematical negative result; it says
that the obvious homogeneous calibration does not provide a one-dimensional
singular-orbit boundary value problem.

## The Exceptional Space N_{1,1}

The space `N_{1,1}` has extra structure. Let

```text
H = S^1_{1,1} = diag(z, z, z^{-2}),
K = S(U(2)U(1)).
```

Then `H` is central in `K`, and

```text
K/H ~= SO(3),
SU(3)/K ~= CP^2.
```

This gives the fibration

```text
SO(3) -> N_{1,1} -> CP^2.
```

The real subgroup `SO(3) < SU(3)` acts on `CP^2` with cohomogeneity one. One
way to see this is to write a complex line as `[x + i y]`, with
`x,y in R^3`, and choose phase so that `x` and `y` are orthogonal. The remaining
quantity `|x|^2 - |y|^2` is a one-dimensional invariant.

Combining this base action with the right `K/H ~= SO(3)` fiber action gives a
candidate action

```text
SO(3)_real x SO(3)_fiber
```

with generic orbit dimension

```text
3 + 3 = 6,
```

hence cohomogeneity one on the seven-dimensional manifold `N_{1,1}`.

## Current Solver Compatibility

This candidate is promising, but it does not fit the current `q_i` solver
directly. The existing code assumes principal orbit

```text
SO(4) / Z_2^2
```

and uses the corresponding eight-variable endpoint charts. The candidate
`N_{1,1}` action has different principal orbits and needs a new invariant-form
basis, endpoint charts, and ODE derivation.

## Verdict

Generic Aloff-Wallach spaces are not immediately ready for this pipeline.

`N_{1,1}` is the interesting next candidate. It has a plausible
cohomogeneity-one action compatible with the maximal-volume philosophy, but the
next task is pen-and-paper plus symbolic/numerical setup:

```text
derive SO(3)_real x SO(3)_fiber invariant forms,
derive singular orbit endpoint conditions,
verify a known homogeneous N_{1,1} nearly parallel structure,
then build the max-volume scout.
```

