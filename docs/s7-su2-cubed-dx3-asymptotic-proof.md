# S7 SU(2)^3 D_x3 Asymptotic Proof

This note records the current proof status for the scalar defect

```text
D_x3(a) = x3(T_a),
```

where `T_a` is the first `p=x0=0` terminal event in the scaled Podesta
`SU(2)^3` system.  Standard `K-` terminal closure forces `x3(T_a)=0`, so a
nonzero value of `D_x3` excludes compact closure.

## Statement Proved

The large-tail endpoint defect has a nonzero negative limit:

```text
lim_{|a| -> infinity} D_x3(a) < 0.
```

Consequently, there exists some `A` such that no standard `K-` compact closure
occurs in this one-parameter Podesta chart for `|a| >= A`.

This proof does not give a sharp explicit value of `A`.  The separate
`A=100000000` wall calculation is still conditional on a support-entry lemma.

## Setup

Use

```text
b = 1/a,
h0 = a p,
h1 = x1,
h2 = a^3 x2,
h3 = a x3.
```

The scaled equations have the form

```text
x' = F_0(t,x) + b R_1(t,x) + b^2 R_2(t,x) + b^3 R_3(t,x).
```

For any fixed positive `p0`, the vector field is smooth on `p >= p0`.
Therefore ordinary continuous dependence of ODE solutions gives convergence of
the finite-`b` trajectories to the `b=0` trajectory up to the first slice
`p=p0`, as long as the limiting trajectory reaches that slice transversely.

We use

```text
p0 = 0.001.
```

The limiting trajectory at this slice is numerically enclosed in the terminal
box

```text
3.59 <= t  <= 3.61,
8.5  <= x1 <= 9.5,
0.004 <= x2 <= 0.008,
-1.4 <= x3 <= -0.9.
```

## Removable Terminal Singularity

Near `p=0`, switch to `p` as independent variable.  Multiplying away the
singular powers gives

```text
dt/dp   = p^4 / A,
dx_i/dp = p H_i / A,
```

where

```text
A  = p^4 dp/dt,
H_i = p^3 dx_i/dt.
```

The functions `A,H_i` are regular at `p=0`.  On the box

```text
0 <= p <= 0.001,
|b| <= 1e-8,
3.59 <= t  <= 3.61,
8.5  <= x1 <= 9.5,
0.004 <= x2 <= 0.008,
-1.4 <= x3 <= -0.9,
```

interval evaluation gives

```text
A in [-0.013103144180239277, -0.0026925092524472277].
```

Thus `A` is bounded away from zero, and the `p`-time system extends
continuously to `p=0` inside this terminal box.

The same interval check gives the tail variation bound

```text
|Delta x3| <= 3.56642486934e-05
```

from `p=0.001` to `p=0`.

At the limiting `p=0.001` slice,

```text
x3 = -1.1664640742922274.
```

Therefore the limiting endpoint satisfies

```text
x3(T_infinity) in [-1.1664997385409208, -1.166428410043534].
```

In particular, the limiting `D_x3` endpoint value is strictly negative.

## Conclusion

By compact-interval continuous dependence up to `p=0.001`, followed by the
regular `p`-time terminal layer above, the finite-`b` endpoint values
`D_x3(a)` converge to this negative limiting endpoint value as `b=1/a -> 0`.

Hence `D_x3(a)` is nonzero for all sufficiently large `|a|`.  Since standard
`K-` closure requires `D_x3(a)=0`, this proves large-`|a|` exclusion for some
unspecified threshold `A`.

## What Is Not Yet Proved

This is not yet the explicit statement

```text
no closure for every |a| >= 100000000.
```

For that stronger statement we still need a certified support-entry estimate
showing that every `|a| >= 100000000` trajectory reaches the correlated late
region used in the scalar wall calculation:

```text
x3 <= -0.36,
C >= 1.23 p^3,
p <= 0.33.
```

That explicit finite-`A` support-entry lemma remains open.
