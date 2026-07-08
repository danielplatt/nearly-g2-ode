# Stiefel Feasibility Note

Reproducibility command:

```zsh
.venv/bin/python -m experiments.stiefel_feasibility
.venv/bin/python -m experiments.stiefel_feasibility --json
```

## Question

We wanted to move from Berger and `S^7` to the Stiefel manifold

```text
V_{5,2} = SO(5) / SO(3)_standard
```

and repeat the same workflow:

1. derive endpoint conditions,
2. verify the known homogeneous nearly parallel `G_2` structure,
3. run a small max-volume recovery calibration.

## Current Solver Assumption

The current `q_i` ODE is not a general homogeneous-space solver. It is the
cohomogeneity-one `SO(4)`-invariant system with principal orbit

```text
SO(4) / Z_2^2
```

so the regular part is one-dimensional:

```text
(t, principal orbit),  dim(principal orbit) = 6.
```

The Berger and fixed-chart `S^7` experiments fit this setup because they are
encoded as endpoint charts for this same eight-variable `q_i` system.

## Stiefel Homogeneous Calibration Data

The standard Stiefel manifold does have a known homogeneous nearly parallel
`G_2` structure. In Moreno-Portilla notation for invariant `G_2` forms on
`V_{5,2}`, the nearly parallel locus is

```text
x = a,
y = b,
a^2 + b^2 = 27/512,
z = -9/32.
```

The feasibility command verifies these algebraic defects exactly for the sample
point `theta = 0`.

This is an algebraic homogeneous calibration, not a cohomogeneity-one endpoint
calibration.

## Why The Existing Endpoint Plan Fails

View `V_{5,2}` as oriented orthonormal pairs `(x,y)` in `R^5`. Let the natural
`SO(4) < SO(5)` fix the fifth coordinate. Then the two components

```text
x_5, y_5
```

are independent invariants on the open disk

```text
x_5^2 + y_5^2 < 1.
```

For a generic pair, the `R^4` projections of `x` and `y` span a two-plane. The
generic stabilizer in `SO(4)` is therefore the `SO(2)` rotating the orthogonal
two-plane. Thus

```text
dim generic SO(4)-orbit = 6 - 1 = 5,
dim V_{5,2} = 7,
cohomogeneity = 2.
```

So standard Stiefel is not a new pair of singular endpoints for the current
`SO(4)/Z_2^2` cohomogeneity-one system. There is no single interval variable,
no pair of endpoint Taylor charts, and no one-dimensional maximal-volume
matching problem to run in the current framework.

## Consequence

The requested Stiefel plan cannot honestly proceed to the same step-3 small
max-volume calibration without deriving a different symmetry reduction.

## Calibration-Action Audit

The known Stiefel nearly parallel structure is homogeneous and `SO(5)`-invariant.
For the known homogeneous solution to be a calibration target of a reduced
cohomogeneity-one ODE, the reducing group should preserve that known structure;
in particular, it should sit inside the homogeneous automorphism group.

The standard connected candidates do not produce the needed one-dimensional
orbit space:

```text
SO(5): transitive on V_{5,2}; cohomogeneity 0.
SO(4): generic stabilizer SO(2); cohomogeneity 2.
U(2): dimension 4, too small for six-dimensional principal orbits.
SO(3): dimension 3, too small for six-dimensional principal orbits.
```

Thus the obstruction is not just that the first `SO(4)` embedding was unlucky.
Within the usual homogeneous-calibration candidates, there is no
cohomogeneity-one Stiefel action available for the current endpoint-volume
workflow.

Reasonable next directions are:

```text
1. choose a different topology that fits the current SO(4)/Z_2^2 q-system;
2. derive a new cohomogeneity-one ansatz for another group action, if one exists;
3. move to a homogeneous/invariant algebraic Stiefel study, which is useful but
   is not the same as our endpoint-volume scouting pipeline.
```
