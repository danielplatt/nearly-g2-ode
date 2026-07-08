# Aloff-Wallach N11 Endpoint Smoothness

Reproducibility command:

```zsh
.venv/bin/python -m experiments.aloff_wallach_endpoint_smoothness
.venv/bin/python -m experiments.aloff_wallach_endpoint_smoothness --json
```

## Purpose

This note derives the first endpoint smoothness conditions for the
`SO(3)_real x SO(3)_fiber` cohomogeneity-one ansatz on

```text
N_{1,1} = SU(3) / U(1)_{1,1}.
```

The derivation keeps the full principal-orbit chart from
`experiments.aloff_wallach_ansatz`:

```text
omega(t): 7 coefficients x1,...,x7
gamma(t): 12 coefficients y1,...,y12
```

No homogeneous `A,B,C,D` reduction is imposed.

## Singular Isotropy Model

At either endpoint the singular orbit has codimension two.  In the standard
graph lift used by the verifier, the collapsing and surviving principal
directions are

```text
theta = base_3 + fiber_3,
zeta  = base_3 - fiber_3.
```

On the normal ray,

```text
normal_angular = r theta,
base_3  = (theta + zeta) / 2,
fiber_3 = (theta - zeta) / 2.
```

The connected singular isotropy rotates

```text
(base_1, base_2), (fiber_1, fiber_2)
```

with weight `1` and fixes `zeta`.  The two endpoints differ in the normal
slice:

```text
RP^2 real-line endpoint:     normal weight 1
CP^1 null-conic endpoint:    normal weight 2
```

Changing the graph lift sign replaces `fiber_3` by `-fiber_3` in the displayed
linear combinations.  It does not change the dimension counts.

## Eschenburg-Wang Test

A smooth invariant `G_2` form near a singular orbit is a
`K`-equivariant smooth map

```text
V -> Lambda^3(V + n)^*
```

where `V` is the two-dimensional normal slice and `n` is the tangent space of
the singular orbit.  For each homogeneous normal-polynomial degree `p`, the
verifier solves the linear equivariance equation and evaluates the resulting
space on the chosen normal ray.  The 19 principal coefficients are then
rewritten in the smooth coframe, remembering that every occurrence of
`theta` costs one radial factor because `normal_angular = r theta`.

## Zeroth-Order Conditions

The zeroth-order endpoint values reduce to four free constants.  In the
standard graph convention,

```text
x1(0)=x2(0)=x3(0)=x4(0)=x5(0)=x6(0)=x7(0)=0,
```

and

```text
y1(0)  = A,    y2(0)  = -A,
y3(0)  = B,    y4(0)  =  C,
y5(0)  = B,    y6(0)  =  C,
y7(0)  = -C,   y8(0)  =  B,
y9(0)  = -C,   y10(0) =  B,
y11(0) = D,    y12(0) = -D.
```

Equivalently, the raw 19 endpoint coefficient values have a canonical
four-parameter chart

```text
(A, B, C, D).
```

The origin of the conditions is as follows:

```text
theta-components must vanish to at least first order:
  y1 + y2 = 0,
  y3 - y5 = 0,
  y4 - y6 = 0,
  y7 - y9 = 0,
  y8 - y10 = 0,
  y11 + y12 = 0.

constant mixed base/fiber tensors must be SO(2)-invariant:
  y8 = y3,
  y7 = -y4.

no degree-zero invariant contains the radial normal coform alone:
  all x_i(0) vanish.
```

## Weighted Jet Dimensions

Smoothness does not stop at endpoint values; it gives weighted Taylor layers.
The verifier computes these dimensions from the full 19-variable chart:

```text
RP^2 endpoint, normal weight 1:
  through order 0:  4
  through order 1:  9   (new layer: 5)
  through order 2:  21  (new layer: 12)

CP^1 endpoint, normal weight 2:
  through order 0:  4
  through order 1:  13  (new layer: 9)
  through order 2:  23  (new layer: 10)
```

These are smoothness-only counts, before imposing the algebraic `SU(3)`
constraints and before deriving the nearly-parallel evolution equations.

## Consequence

Endpoint smoothness does reduce the parameter count, but not by artificially
choosing a small homogeneous family.  The correct next chart starts from the
full 19 coefficients, rewrites them in the collapsing/surviving endpoint
coframe, and uses the weighted smooth combinations above.  The next task is to
derive the cohomogeneity-one evolution equations in this chart and then impose
the `SU(3)` algebraic constraints and ODE recurrence to find the actual scout
coordinates.
