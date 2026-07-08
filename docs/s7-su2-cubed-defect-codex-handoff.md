# S7 SU(2)^3 Scalar Defect Handoff for Codex

This file is a Codex-ready handoff for implementing and auditing scalar defects for Podesta's `SU(2)^3` cohomogeneity-one nearly parallel `G2` ODE on `S^7`.

The goal is not to prove anything in this pass. The goal is to build a broad defect-audit framework, evaluate many necessary scalar defects numerically, discard bad candidates, and identify proof-friendly quantities for a later large-`|a|` exclusion argument.

## Core Principle

A scalar defect `D(a)` is acceptable if standard compact `K_-` closure forces

```text
D(a) = 0.
```

It does **not** need to characterize compact solutions. It may have extra zeros. It may be chart-dependent, asymmetric, nonlinear, ugly, or ad hoc. The numerical audit should find which necessary defects are actually useful.

## Mathematical Setup

Podesta's five-function reduction uses

```text
f0 = t h0,
f1 = t^4 h1,
f2 = h2,
f3 = t^2 h3,
f4 = t^2 h4,
h4 = -h3 - h0^2/6.
```

The one-ended smooth initial data are parametrized by nonzero real `a`:

```text
h0(0) = a,
h1(0) = 27/4,
h2(0) = -a^3/27,
h3(0) = 3a.
```

Known compact values in this chart:

```text
round S7:     a = -36
squashed S7:  a = 108/5
```

The positive round value `a = 36` is equivalent by an outer automorphism but does not close in this unmodified standard `K_-` chart.

The finite `h`-ODE is

```text
h4 = -h3 - h0^2/6

h0' = a0/t + b0
h1' = a1/t + b1
h2' = b2
h3' = a3/t + b3
```

where

```text
a0 = -h0 - 3 h2 h3^2 / h0^4
a1 = -4 h1 + h3^3 / h0^3
a3 = -2 h3 + 6 h0

b0 = -3/(2 h0^4) *
     (t (h3 - h4) (h1 h2 + h3 h4) - 2 t^3 h1 h4^2)

b1 = t/(2 h0^3) * (h1^2 h2 - 3 h1 h3 h4)

b2 = t/h0^3 *
     (h4 (h2 h3 - t^2 h4^2) - (1/2) h2 (h1 h2 - h3 h4))

b3 = t/(2 h0^3) *
     (h1 h2 h3 + h3^2 h4 - 2 t^2 h1 h4^2).
```

The standard `K_-` right endpoint closure condition is: at a terminal point with `f1` nonzero,

```text
f0 = f2 = f3 = f4 = 0.
```

Equivalently, at the first `f0 = 0` or `h0 = 0` terminal event `T_a`, standard closure forces

```text
h2(T_a) = 0,
h3(T_a) = 0,
h4(T_a) = 0.
```

Since `h4 = -h3 - h0^2/6`, the conditions `h0 = 0` and `h3 = 0` imply `h4 = 0`.

For large `|a|`, use scaled variables

```text
h0 = a x0,
h1 = x1,
h2 = a^3 x2,
h3 = a x3,
b = 1/a.
```

Set

```text
p = x0.
```

The scaled smooth left-end initial value in the `b = 0` limit is

```text
x(0) = (1, 27/4, -1/27, 3).
```

The limiting scaled ODE for `x = (x0, x1, x2, x3)` is

```text
x0' =
  (-x0 - 3 x2 x3^2/x0^4)/t
  - t/(4 x0^2) * (x1 x2 - x3 x0^2/6)

x1' =
  (-4 x1 + x3^3/x0^3)/t
  + t/(2 x0^3) * (x1^2 x2 + (1/2) x1 x3 x0^2)

x2' =
  t/x0^3 *
  (-x0^2 x2 x3/4 - (1/2) x1 x2^2 + t^2 x0^6/216)

x3' =
  (-2 x3 + 6 x0)/t
  + t/(2 x0^3) *
    (x1 x2 x3 - x3^2 x0^2/6 - t^2 x1 x0^4/18).
```

At the scaled first `p = 0` terminal event `T`, standard closure forces

```text
p(T) = 0,
x2(T) = 0,
x3(T) = 0,
x1(T) != 0.
```

Every scalar expression below is a candidate defect because it vanishes on this terminal set, or because it is a scalar multiple/normalization/integral equivalent of such a vanishing condition.

## Already Known Numerical Signals

The previous serious proof attempt focused on

```text
X2(a) = x2(T_a) = h2(T_a)/a^3.
```

Standard closure forces `X2(a) = 0`. Numerically, in the limiting first crossing,

```text
T_infinity ~= 3.598,
X2(T_infinity) ~= 0.006,
X3(T_infinity) ~= -1.1.
```

The `X2` signal appears asymptotically nonzero but is small and terminal-event-sensitive. The auxiliary endpoint coordinate

```text
X3(a) = x3(T_a) = h3(T_a)/a
```

is also forced to vanish and is numerically much farther from zero. It deserves priority.

The useful cancellation variable is

```text
C = x1*x2 - p^2*x3/6.
```

In the limiting system,

```text
C' = -4 C/t + 2 x2*x3^3/(t*p^3) - p^3/t + x1*t^3*p^3/108.
```

On a negative `x3` wall, `C` cancels bad singular terms in `x3'`. A previously promising tail barrier region was

```text
x3 <= -0.36,
C >= 1.23 p^3,
p <= 0.33,
t in [3.5, 4.0].
```

The proof attempt stalled because rectangular interval boxes lost correlations among `x1`, `x2`, `x3`, and `C`. This suggests that `C` or a correlated combination involving `C` may be a better proof object than raw `X2`.

One exact finite-scaled wall identity is especially simple:

```text
On x2 = 0,

x2' = t^3 p^3/216 * (1 + 6 b x3/p^2)^3,

where b = 1/a.
```

This suggests testing defects involving `x2`, `x3`, `Q = 1 + 6 b x3/p^2`, or `W = p^2 + 6 b x3`.

## Suggested Codex Implementation Interface

Implement a defect registry where each defect is a function from a terminal integration result to a scalar. Use one common object or dictionary containing endpoint data and, when available, full trajectory data.

Suggested data fields:

```python
state = {
    "a": a,
    "b": 1/a,
    "T": T,
    "p": p_T,
    "x0": p_T,
    "x1": x1_T,
    "x2": x2_T,
    "x3": x3_T,
    "h0": h0_T,
    "h1": h1_T,
    "h2": h2_T,
    "h3": h3_T,
    "h4": h4_T,
    "f0": f0_T,
    "f1": f1_T,
    "f2": f2_T,
    "f3": f3_T,
    "f4": f4_T,
    "history": {
        "t": t_array,
        "p": p_array,
        "x1": x1_array,
        "x2": x2_array,
        "x3": x3_array,
        # optional h and f arrays
    },
}
```

Helper quantities:

```python
C = x1*x2 - p*p*x3/6
Q = 1 + 6*b*x3/(p*p)       # only away from p=0
W = p*p + 6*b*x3
```

For endpoint evaluation at `p = 0`, avoid evaluating `Q` directly. Use scaled forms such as `W/b = 6*x3` when `b != 0`.

Audit each defect on:

```text
1. known compact values: a = -36 and a = 108/5;
2. nonclosing positive round-chart value: a = 36;
3. large positive and negative a values, e.g. ±100, ±250, ±500, ±1000;
4. larger values if stable;
5. the limiting b = 0 IVP.
```

For each defect, record:

```text
name,
value,
absolute value,
sign,
relative terminal-event sensitivity,
whether it vanishes at known compact values,
whether it has apparent extra zeros,
whether it tends to a nonzero finite limit,
whether it appears to diverge,
whether it has a simple derivative/barrier identity.
```

## Highest-Priority Defects

Implement these first.

| name | formula | why closure forces zero | sanity check |
|---|---|---|---|
| `D_x3` | `x3(T)` | closure gives `x3(T)=0` | Numerically large: limiting value about `-1.1`. Strongest immediate candidate. |
| `D_x2` | `x2(T)` | closure gives `x2(T)=0` | Baseline. Known to be small, about `0.006` in the limit. |
| `D_C` | `C(T) = x1*x2 - p^2*x3/6` | at `p=0`, `C=x1*x2`, and `x1 != 0` | Structured cancellation variable; likely better than raw `x2` for barriers. |
| `D_x3_C_norm2` | `x3(T)^2 + lambda*C(T)^2` | both terms vanish | Positive; no sign cancellation; dominated by strong `x3` signal. Try `lambda=1` first. |
| `D_x3_x2_norm2` | `x3(T)^2 + lambda*x2(T)^2` | both terms vanish | Crude but robust endpoint norm. |
| `D_C_plus_lam_x3` | `C(T) + lambda*x3(T)` | both summands vanish | Search for stable sign; avoid cancellation. |
| `D_x3_plus_lam_x2` | `x3(T) + lambda*x2(T)` | both summands vanish | Small `lambda` behaves like `x3`; large `lambda` tests whether `x2` helps. |
| `D_C_over_x1` | `C(T)/x1(T)` | equals `x2(T)` at `p=0` if `x1 != 0` | Same endpoint condition as `x2`, but uses `C`; can be better scaled. |
| `D_x3_damped` | `x3(T)/(1 + abs(x1(T))^r)` | numerator vanishes | Useful if `x1` grows with `|a|`; try `r=1/3,1/2,1`. |
| `D_C_damped` | `C(T)/(1 + abs(x1(T))^r)` | numerator vanishes | Bounded version of `C`. |

The strongest immediate recommendation is to test

```text
R_3C(a)^2 = x3(T)^2 + lambda*C(T)^2.
```

At `p(T)=0`, `C(T)=x1(T)*x2(T)`. Since `x1(T) != 0` under standard closure, `R_3C = 0` forces both `x3(T)=0` and `x2(T)=0`. This combines the large numerical separation of `x3` with the useful cancellation structure of `C`.

## Endpoint Coordinate Defects

Basic endpoint defects:

```text
D_x2 = x2(T)
D_x3 = x3(T)
D_C  = C(T)
```

Weighted coordinate defects:

```text
D_x1x2       = x1(T)*x2(T)
D_x1sq_x2    = x1(T)^2*x2(T)
D_x1x2_lamx3 = x1(T)*x2(T) + lambda*x3(T)
```

Squared and positive defects:

```text
D_x3_sq = x3(T)^2
D_x2_sq = x2(T)^2
D_C_sq  = C(T)^2
D_x3_x2_norm2 = x3(T)^2 + lambda*x2(T)^2
D_x3_C_norm2  = x3(T)^2 + lambda*C(T)^2
D_endpoint_max = max(abs(x2(T)), abs(x3(T)))
D_endpoint_radial = sqrt(x2(T)^2 + x3(T)^2 + C(T)^2)
```

Bounded variants:

```text
D_bounded_x3 = abs(x3(T))/(1 + abs(x3(T)))
D_bounded_C  = abs(C(T))/(1 + abs(C(T)))
D_bounded_radial = sqrt(x3(T)^2 + C(T)^2)/(1 + sqrt(x3(T)^2 + C(T)^2))
```

Advantages:

```text
- Squared/norm defects avoid sign cancellation.
- Linear defects may have simpler differential inequalities.
- Bounded defects are less sensitive to large scaling drift.
```

Degeneracies:

```text
- A scalar linear combination can vanish accidentally.
- Pure `x2` inherits the known smallness issue.
- Pure `C` is equivalent to `x2` at the endpoint but may behave better along the tail.
```

## Linear Combination Families

Implement parameter sweeps over small rational coefficients.

```text
D_linear_1 = alpha*x2(T) + beta*x3(T) + gamma*C(T)
D_linear_2 = C(T) + lambda*x3(T)
D_linear_3 = x3(T) + lambda*x2(T)
D_linear_4 = x1(T)*x2(T) + lambda*x3(T)
```

Suggested grids:

```text
lambda for C + lambda*x3:
  {-2, -1, -1/2, 1/2, 1, 2}

lambda for x3 + lambda*x2:
  {-10, -3, -1, 1, 3, 10}

(alpha,beta,gamma):
  small integer triples in {-2,-1,0,1,2}, excluding the zero triple.
```

Keep combinations with:

```text
- stable sign for large positive and negative `a`,
- no visible extra zeros near the large-tail region,
- nonzero or divergent limiting value,
- simple derivative/barrier form.
```

## Endpoint h4-Based Defects

In finite variables,

```text
h4 = -h3 - h0^2/6.
```

At a terminal `h0=0` event,

```text
h4(T) = -h3(T).
```

Thus `h4`-based endpoint defects are mostly `h3`/`x3` defects in disguise.

Candidates:

```text
D_h4       = h4(T)
D_h4_over_a = h4(T)/a        # equals -x3(T) at h0=0
D_h3_minus_h4 = h3(T) - h4(T)  # equals 2*h3(T) at h0=0
D_h3_h4_product = h3(T)*h4(T)
D_h3_h4_norm2 = h3(T)^2 + h4(T)^2
D_h3_plus_lam_h4 = h3(T) + lambda*h4(T), lambda != 1
```

Do not prioritize

```text
h3(T) + h4(T)
```

because at `h0(T)=0`,

```text
h3(T) + h4(T) = -h0(T)^2/6 = 0
```

for every first `h0=0` terminal event, whether or not standard closure holds. It is necessary but non-discriminating.

## C-Based Defects

Recall

```text
C = x1*x2 - p^2*x3/6.
```

At `p=0`,

```text
C(T) = x1(T)*x2(T).
```

Since standard closure has `x1(T) != 0`, `C(T)=0` is equivalent to `x2(T)=0` at a true terminal event. Its advantage is tail structure, not endpoint novelty.

Candidates:

```text
D_C = C(T)
D_C_over_x1 = C(T)/x1(T)
D_C_plus_lam_x3 = C(T) + lambda*x3(T)
D_C_sq_plus_x3_sq = C(T)^2 + lambda*x3(T)^2
D_C_damped_x3 = C(T)/(1 + abs(x3(T)))
D_C_damped_x1 = C(T)/(1 + abs(x1(T)))
D_sgn_a_C = sign(a)*C(T)
D_absa_gamma_C = abs(a)^gamma*C(T)
```

Barrier-gap variants:

```text
D_B_mu = C(T) - mu*p(T)^3
D_B_mu_lam = C(T) - mu*p(T)^3 + lambda*x3(T)
D_L_alpha_beta = C(T) - alpha*p(T)^3 - beta*x3(T)^2
```

At `p(T)=0`, these reduce to `C(T)` or a combination involving `C(T)` and `x3(T)`, so closure forces zero.

Suggested values:

```text
mu in {0.5, 1.0, 1.23, 1.5, 2.0}
lambda in {-2, -1, -1/2, 1/2, 1, 2}
alpha in {0.5, 1.0, 1.23, 1.5, 2.0}
beta in {-2, -1, -1/2, 0, 1/2, 1, 2}
```

The special value `mu = 1.23` is motivated by the earlier tail barrier `C >= 1.23 p^3`.

## Polynomial Defects From Singular ODE Numerators

The `1/t` pieces contain useful polynomial combinations. Multiplying away denominators gives endpoint-polynomial defects.

From

```text
x1' contains (-4*x1 + x3^3/p^3)/t,
```

use

```text
S1 = x3^3 - 4*x1*p^3.
```

At `p=0`,

```text
S1(T) = x3(T)^3.
```

Thus closure forces `S1(T)=0`. If the limiting `x3(T) ~= -1.1`, then `S1(T) ~= -1.331`, which is large.

From

```text
x3' contains (-2*x3 + 6*p)/t,
```

use

```text
S3 = 2*x3 - 6*p.
```

At `p=0`,

```text
S3(T) = 2*x3(T).
```

From

```text
x0' contains (-p - 3*x2*x3^2/p^4)/t,
```

use

```text
S0 = p^5 + 3*x2*x3^2.
```

At `p=0`,

```text
S0(T) = 3*x2(T)*x3(T)^2.
```

This vanishes under closure, but is degenerate if either `x2(T)=0` or `x3(T)=0`.

Additional polynomial candidates:

```text
D_poly_1 = x3(T)^3 + lambda*C(T)
D_poly_2 = x3(T)^3 + lambda*x1(T)*x2(T)
D_poly_3 = 2*x3(T) - 6*p(T) + lambda*C(T)
D_poly_4 = x3(T)^2 + lambda*x1(T)*x2(T)
D_poly_5 = x1(T)^2*x2(T) + lambda*x3(T)^2
```

These are worth testing because low-degree polynomials may satisfy simpler derivative inequalities than `x2` itself.

## Defects From Finite-h Numerator Terms

The finite `h`-system numerator expressions in the `b_i` terms vanish under standard closure because `h2=h3=h4=0` at the right endpoint.

Define

```text
N0 = T*(h3 - h4)*(h1*h2 + h3*h4) - 2*T^3*h1*h4^2

N1 = h1^2*h2 - 3*h1*h3*h4

N2 = h4*(h2*h3 - T^2*h4^2) - (1/2)*h2*(h1*h2 - h3*h4)

N3 = h1*h2*h3 + h3^2*h4 - 2*T^2*h1*h4^2
```

Closure forces

```text
N0(T) = N1(T) = N2(T) = N3(T) = 0.
```

Scalar defects:

```text
D_Ni = Ni(T)
D_Ni_damped = Ni(T)/(1 + abs(h1(T))^r)
D_N_linear = sum_i lambda_i*Ni(T)
D_N_norm2 = sum_i lambda_i*Ni(T)^2
```

Advantages:

```text
- These are exactly the numerator combinations controlling the singular-looking finite system.
- A stable sign in one numerator may be proof-friendly.
```

Degeneracies:

```text
- Some `Ni` may vanish when only one of `h2`, `h3`, or `h4` vanishes.
- Prefer pairs or norms if scalar `Ni` values show extra zeros.
```

Scaled limiting analogues worth testing:

```text
M1 = x1^2*x2 + (1/2)*x1*x3*p^2

M2 = -(1/4)*p^2*x2*x3 - (1/2)*x1*x2^2 + T^2*p^6/216

M3 = x1*x2*x3 - (1/6)*x3^2*p^2 - T^2*x1*p^4/18
```

At `p=0`,

```text
M1(T) = x1(T)^2*x2(T)
M2(T) = -(1/2)*x1(T)*x2(T)^2
M3(T) = x1(T)*x2(T)*x3(T)
```

Thus `M1` is essentially a weighted `x2` defect, while `M3` detects joint nonclosure.

## Ratio and Normalized Endpoint Defects

Because standard closure has `x1(T) != 0`, one can divide by powers of `x1(T)` if numerically safe.

Endpoint-normalized defects:

```text
D_x2_over_x1 = x2(T)/x1(T)
D_x2_over_x1sq = x2(T)/x1(T)^2
D_x3_over_x1 = x3(T)/x1(T)
D_x3_over_absx1_third = x3(T)/abs(x1(T))^(1/3)
D_C_over_x1 = C(T)/x1(T)
D_C_over_x1sq = C(T)/x1(T)^2
```

Safe damped versions:

```text
D_x2_damped_r = x2(T)/(1 + abs(x1(T))^r)
D_x3_damped_r = x3(T)/(1 + abs(x1(T))^r)
D_C_damped_r  = C(T)/(1 + abs(x1(T))^r)
```

Suggested exponents:

```text
r in {1/3, 1/2, 1, 2}
```

`f1`-normalized defects:

```text
D_f2_over_f1r = f2(T)/(1 + abs(f1(T))^r)
D_f3_over_f1r = f3(T)/(1 + abs(f1(T))^r)
D_f4_over_f1r = f4(T)/(1 + abs(f1(T))^r)
D_f_radial_over_f1r = sqrt(f2(T)^2 + f3(T)^2 + f4(T)^2)/(1 + abs(f1(T))^r)
```

Advantages:

```text
- Better scale control.
- Useful if raw defects grow or drift with `a`.
```

Degeneracies:

```text
- Avoid raw division by `x1` or `f1` if numerical nonclosing trajectories make the denominator small.
- Prefer `1 + abs(denominator)^r` damping for audit robustness.
```

## Large-|a| Rescaled Defects

Since closure forces `h2(T)=h3(T)=0`, any scalar multiple is also a necessary defect.

For `h2`:

```text
D_h2_scaled_gamma = abs(a)^(-gamma)*h2(T)
D_h2_sgn_scaled_gamma = sign(a)*abs(a)^(-gamma)*h2(T)
```

For `h3`:

```text
D_h3_scaled_gamma = abs(a)^(-gamma)*h3(T)
D_h3_sgn_scaled_gamma = sign(a)*abs(a)^(-gamma)*h3(T)
```

Natural exponents:

```text
gamma = 3 for h2, giving x2
gamma = 1 for h3, giving x3
```

Also deliberately test non-natural exponents:

```text
a^(-2)*h2(T) = a*x2(T)
a^(-4)*h2(T) = a^(-1)*x2(T)
h3(T) = a*x3(T)
a^(-2)*h3(T) = a^(-1)*x3(T)
```

Combined rescalings:

```text
D_absa_alpha_x3_plus_absa_beta_C = abs(a)^alpha*x3(T) + lambda*abs(a)^beta*C(T)
D_absa_alpha_x3_plus_absa_beta_x2 = abs(a)^alpha*x3(T) + lambda*abs(a)^beta*x2(T)
D_absa_alpha_radial_3C = abs(a)^alpha*sqrt(x3(T)^2 + C(T)^2)
```

Test small exponents:

```text
alpha,beta in {-2, -1, 0, 1, 2}
```

Advantages:

```text
- If the limiting defect is nonzero, multiplying by powers of `a` can produce divergence.
- Divergence may be easier to prove by contradiction.
```

Degeneracies:

```text
- Sign conventions matter for negative `a`.
- Test both `a^gamma` and `abs(a)^gamma` variants.
```

## Wall-Identity Defects From x2 = 0

The exact finite-scaled identity on `x2=0` is

```text
x2' = t^3*p^3/216 * (1 + 6*b*x3/p^2)^3.
```

Define

```text
Q = 1 + 6*b*x3/p^2
W = p^2 + 6*b*x3.
```

Endpoint candidates:

```text
D_W = W(T)
D_W_over_b = W(T)/b = 6*x3(T) at p=0, b != 0
D_x2W = x2(T)*W(T)
```

Do not prioritize raw `W(T)` for large `|a|`, because

```text
W(T) = 6*b*x3(T)
```

at `p=0`, and this can be artificially small as `b -> 0`. Prefer `W/b`.

Wall-event candidates:

Let `S2_last` be the last zero of `x2` before `T`, preferably restricted to a terminal tail region.

```text
D_S2_p = p(S2_last)
D_S2_x3 = x3(S2_last)
D_S2_W = W(S2_last)
D_S2_Q_cube = p(S2_last)^3 * Q(S2_last)^3
```

Advantages:

```text
- The derivative on the `x2=0` wall is explicit.
- If `Q` has stable sign in the tail, this may prove `x2` cannot return to zero at the endpoint.
```

Degeneracies:

```text
- Extra `x2` zeros may mislead event selection.
- Use the last zero after entering a prescribed terminal tail box, not the first zero from `t=0`.
```

## Event-Location Defects

Let

```text
T0 = first p=0 event.
```

Define event times:

```text
T2_last = last zero of x2 before T0
T3_last = last zero of x3 before T0
TC_last = last zero of C before T0
```

Closure forces `x2(T0)=x3(T0)=C(T0)=0`, so, if the relevant zero is isolated and chosen as the last tail zero,

```text
D_T0_minus_T2 = T0 - T2_last
D_T0_minus_T3 = T0 - T3_last
D_T0_minus_TC = T0 - TC_last
```

Alternative event defects:

```text
D_p_at_T2 = p(T2_last)
D_p_at_T3 = p(T3_last)
D_p_at_TC = p(TC_last)
D_x3_at_T2 = x3(T2_last)
D_x2_at_T3 = x2(T3_last)
D_C_at_T3 = C(T3_last)
```

Advantages:

```text
- Event times may be less sensitive than endpoint values near a singular crossing.
```

Degeneracies:

```text
- Missing zeros, tangential zeros, or extra oscillations.
- Encode "no last zero in terminal tail" as a large signed nonzero diagnostic value.
```

## Pre-Terminal p = epsilon Defects

Instead of sampling exactly at `p=0`, sample at

```text
T_eps = first time p(t) = eps
```

for small fixed `eps > 0`.

Exact closure implies only the limiting statements

```text
lim_{eps -> 0} x2(T_eps) = 0
lim_{eps -> 0} x3(T_eps) = 0
lim_{eps -> 0} C(T_eps) = 0
```

So fixed-`eps` values are scout diagnostics rather than exact defects. Still implement them because they may be much more stable numerically.

Scout quantities:

```text
D_x2_eps = x2(T_eps)
D_x3_eps = x3(T_eps)
D_C_eps = C(T_eps)
D_radial_3C_eps = x3(T_eps)^2 + C(T_eps)^2
```

Test

```text
eps in {1e-1, 5e-2, 1e-2, 5e-3, 1e-3}
```

Promote a fixed-`eps` scout to a proof-relevant limiting defect only if it remains bounded away from zero uniformly as `eps -> 0`.

## Integral Balance Defects

Endpoint defects can be rewritten as integral defects. These may be proof-friendly if the integrand has sign structure.

Since

```text
x2(0) = -1/27,
```

closure `x2(T)=0` gives

```text
D_I2 = -1/27 + integral_0^T x2'(t) dt = 0.
```

Using the limiting equation,

```text
D_I2 = -1/27
       + integral_0^T t/p^3 *
         (-p^2*x2*x3/4 - (1/2)*x1*x2^2 + t^2*p^6/216) dt.
```

For `x3`, since `x3(0)=3`, closure gives

```text
D_I3 = 3 + integral_0^T x3'(t) dt = 0.
```

For `C`, the initial value is

```text
C(0) = x1(0)*x2(0) - p(0)^2*x3(0)/6
     = (27/4)*(-1/27) - 3/6
     = -3/4.
```

Thus closure gives

```text
D_IC = -3/4 + integral_0^T C'(t) dt = 0.
```

The `C` equation has integrating factor `t^4`:

```text
C' = -4*C/t + 2*x2*x3^3/(t*p^3) - p^3/t + x1*t^3*p^3/108.
```

Hence

```text
(t^4*C)' =
    2*t^3*x2*x3^3/p^3
  - t^3*p^3
  + x1*t^7*p^3/108.
```

Since `t^4*C -> 0` at the left endpoint and closure gives `C(T)=0`, a necessary integral defect is

```text
D_C_IF = integral_0^T [
    2*t^3*x2*x3^3/p^3
  - t^3*p^3
  + x1*t^7*p^3/108
] dt = 0.
```

Similarly,

```text
x3' + 2*x3/t =
  6*p/t
  + t/(2*p^3) *
    (x1*x2*x3 - x3^2*p^2/6 - t^2*x1*p^4/18).
```

Therefore

```text
(t^2*x3)' =
  6*t*p
  + t^3/(2*p^3) *
    (x1*x2*x3 - x3^2*p^2/6 - t^2*x1*p^4/18).
```

Closure gives

```text
D_3_IF = integral_0^T [
    6*t*p
  + t^3/(2*p^3) *
    (x1*x2*x3 - x3^2*p^2/6 - t^2*x1*p^4/18)
] dt = 0.
```

Implementation notes:

```text
- These require trajectory data, not just endpoint data.
- Use dense output or accepted solver samples plus robust quadrature.
- Avoid direct evaluation exactly at p=0 if terms contain p^(-3).
- Either stop at p=eps and extrapolate, or use the ODE's internally evaluated derivative where available.
```

## Weighted Moment Defects

For any smooth finite weight `w(t)`, closure gives

```text
w(T)*x2(T) = 0,
w(T)*x3(T) = 0,
w(T)*C(T) = 0.
```

Thus, by integration by parts,

```text
D_2_w = w(0)*x2(0) + integral_0^T [w'(t)*x2(t) + w(t)*x2'(t)] dt = 0

D_3_w = w(0)*x3(0) + integral_0^T [w'(t)*x3(t) + w(t)*x3'(t)] dt = 0

D_C_w = w(0)*C(0) + integral_0^T [w'(t)*C(t) + w(t)*C'(t)] dt = 0
```

Useful weights:

```text
w(t) = 1
w(t) = t
w(t) = t^2
w(t) = t^4
w(t) = T - t
w(t) = exp(-alpha*t)
w(t) = 1/(1 + t^2)
```

Weights matching linear ODE terms are especially promising:

```text
w(t) = t^2 for x3
w(t) = t^4 for C
```

Caution:

```text
If w(T)=0, the resulting endpoint condition may become trivially zero for every p(T)=0 trajectory and lose information.
```

## Derivative-Type Defects

Pure derivatives such as `x2'(T)` or `x3'(T)` are not automatically forced to vanish merely by `x2(T)=x3(T)=0`. Treat them as contingent on the precise terminal Taylor model.

However, derivative defects built from squares are automatically necessary if one-sided derivatives are finite:

```text
D_dx2sq = d/dt(x2^2)(T) = 2*x2(T)*x2'(T)
D_dx3sq = d/dt(x3^2)(T) = 2*x3(T)*x3'(T)
D_dCsq  = d/dt(C^2)(T)  = 2*C(T)*C'(T)
D_d_norm_3C = d/dt(x3^2 + lambda*C^2)(T)
```

Angular or Wronskian-type defects:

```text
D_ang_x2_x3 = x2(T)*x3'(T) - x3(T)*x2'(T)
D_ang_C_x3  = C(T)*x3'(T)  - x3(T)*C'(T)
```

Closure forces these to vanish if the derivatives are finite.

Contingent Taylor-cap defects to test against known compact solutions:

```text
D_f0_prime = f0'(T) - s0
D_f2_prime = f2'(T) - s2
D_f3_prime = f3'(T) - s3
D_f4_prime = f4'(T) - s4
```

Here `s_i` should be inferred from the local smooth `K_-` terminal Taylor model. These may be powerful but require more geometric input than raw endpoint conditions.

## Rate and Blow-Up Defects Near p = 0

These are not all guaranteed by endpoint vanishing alone, but smooth closure may impose stronger rates. Treat as exploratory diagnostics unless the terminal model confirms the rate.

Candidate ratios:

```text
x2/p
x2/p^2
x2/p^3
x3/p
x3/p^2
C/p^3
```

Finite-`h` geometric ratios:

```text
h3/h0^2 = b*x3/p^2
h4/h0^2 = -b*x3/p^2 - 1/6
Q = 1 + 6*b*x3/p^2 = 1 + 6*h3/h0^2
```

If smooth `K_-` closure fixes a terminal value of `h3/h0^2`, then

```text
Q - Q_star
```

is a genuine terminal defect. Calibrate `Q_star` using the known compact values `a=-36` and `a=108/5`.

If nonclosing large-`|a|` trajectories have `x3(T) != 0` while `p -> 0`, these ratios should blow up. Blow-up may be easier to prove than a small endpoint estimate.

## Volume and Maximum-Volume Normalized Defects

The original numerical loss used scale normalization. More algebraic variants:

```text
D_V2 = f2(T)/(1 + abs(f1(T))^r)
D_V3 = f3(T)/(1 + abs(f1(T))^r)
D_V4 = f4(T)/(1 + abs(f1(T))^r)
D_V_radial = sqrt(f2(T)^2 + f3(T)^2 + f4(T)^2)/(1 + abs(f1(T))^r)
```

Using a global maximum scale:

```text
F1_max = max_{0 <= t <= T} abs(f1(t))

D_M2 = f2(T)/(1 + F1_max^r)
D_M3 = f3(T)/(1 + F1_max^r)
D_M34 = sqrt(f3(T)^2 + f4(T)^2)/(1 + F1_max^r)
```

A crude volume proxy:

```text
D_V_perp = sqrt(f2(T)^2 + f3(T)^2 + f4(T)^2)
```

All vanish under standard closure because `f2(T)=f3(T)=f4(T)=0`.

## Barrier-Gap Endpoint Defects

These reduce to ordinary endpoint defects at `p=0`, but define useful possible invariant regions before the endpoint.

Candidates:

```text
B_mu = C - mu*p^3
G_delta = x3 + delta*p
H_delta = x3 + delta*p^2
K_delta = x2 - delta*p^3
L_alpha_beta = C - alpha*p^3 - beta*x3^2
```

At `p=0`,

```text
B_mu(T) = C(T)
G_delta(T) = x3(T)
H_delta(T) = x3(T)
K_delta(T) = x2(T)
L_alpha_beta(T) = C(T) - beta*x3(T)^2
```

All vanish under standard closure.

Suggested parameter values:

```text
mu in {0.5, 1.0, 1.23, 1.5, 2.0}
delta in {-2, -1, -1/2, 1/2, 1, 2}
alpha in {0.5, 1.0, 1.23, 1.5, 2.0}
beta in {-2, -1, -1/2, 0, 1/2, 1, 2}
```

The value is not endpoint novelty; the value is that these quantities may define invariant tail barriers.

## Learned or Calibrated Scalar Defects

Use the two known compact values to calibrate low-complexity scalar expressions that vanish at known compact endpoints and are nonzero in the large limiting IVP.

Ansatz classes:

```text
D = alpha*x2 + beta*x3 + gamma*C
D = alpha*x2 + beta*x3 + gamma*C + delta*x3^2
D = alpha*C + beta*x3 + gamma*x3^3
D = alpha*x1*x2 + beta*x3 + gamma*x3^2
D = alpha*M1 + beta*S1 + gamma*S3
```

Only keep expressions whose monomials vanish on the terminal set `p=x2=x3=0`.

Practical rule:

```text
Keep `x3` or `S3` present with nonzero coefficient unless the goal is specifically to isolate `x2`.
```

Otherwise the candidate may inherit the smallness problem of `X2`.

## Sectional Shooting Defects

Instead of shooting all the way to `p=0`, choose a regular section

```text
p = p_star
```

with small fixed `p_star`, for example

```text
p_star in {0.3, 0.2, 0.1}.
```

Numerically integrate the local `K_-`-smooth terminal model backward to this section. Let `Sigma_minus` be the terminal admissible curve or surface in the section. Define scalar distances from the left-shot trajectory to `Sigma_minus`:

```text
D_Sigma = ell(x1, x2, x3, C, t) evaluated at p=p_star,
```

where `ell=0` is a fitted local equation for `Sigma_minus`.

Possible fitted forms:

```text
ell = x3 - phi3(t, x1)
ell = C - phiC(t, x1)
ell = x2 - phi2(t, x1)
ell = alpha*x2 + beta*x3 + gamma*C + delta
```

Closure forces the trajectory to lie on the terminal admissible set, so `D_Sigma=0`. This is more expensive but avoids endpoint singularity.

## Compact Shortlist for First Numerical Triage

Implement this finite list first.

```text
1.  D_x3 = x3(T)
2.  D_x2 = x2(T)
3.  D_C = C(T)
4.  D_x3_C_norm2 = x3(T)^2 + C(T)^2
5.  D_x3_x2_norm2 = x3(T)^2 + x2(T)^2
6.  D_C_plus_lam_x3 for lambda in {-2,-1,-1/2,1/2,1,2}
7.  D_x3_plus_lam_x2 for lambda in {-10,-3,-1,1,3,10}
8.  S1(T) = x3(T)^3 - 4*x1(T)*p(T)^3
9.  S3(T) = 2*x3(T) - 6*p(T)
10. D_C_over_x1 = C(T)/x1(T)
11. D_x3_damped = x3(T)/(1 + abs(x1(T)))
12. D_C_damped = C(T)/(1 + abs(x1(T)))
13. D_W_over_b = W(T)/b, equivalently 6*x3(T) at p=0
14. D_C_IF integral defect
15. D_3_IF integral defect
```

Likely best proof objects, in order:

```text
1. x3(T)
2. x3(T)^2 + C(T)^2
3. C(T) + lambda*x3(T)
4. S1(T) = x3(T)^3 - 4*x1(T)*p(T)^3
5. D_C_IF
```

Rationale:

```text
- x3 is numerically far from zero in the large limiting crossing.
- C carries cancellation structure hidden from raw endpoint coordinates.
- Positive norm defects avoid accidental sign cancellation.
- S1 amplifies the strong x3 signal cubically.
- Integral defects may expose sign structure unavailable from endpoint sampling alone.
```

## Numerical Triage Workflow

Implement a script or module with these stages.

### Stage 1: Endpoint audit

For each `a` in a grid, integrate to the first `p=0` or `h0=0` event and evaluate endpoint defects.

Recommended grid:

```text
known:      {-36, 108/5}
control:    {36}
large:      {-100, 100, -250, 250, -500, 500, -1000, 1000}
optional:   larger magnitudes if stable
limit:      b = 0 IVP
```

Output a CSV or Markdown table with one row per `(a, defect)`.

### Stage 2: Stability audit

For each candidate, estimate terminal-event sensitivity by varying:

```text
- ODE solver tolerance,
- terminal interpolation method,
- stop threshold p=epsilon,
- fixed-step versus adaptive-step sampling if both exist.
```

Prefer defects whose sign and scale are robust.

### Stage 3: Extra-zero audit

For each defect, scan finite `a` values and reject candidates that show extra zeros in the large-tail region unless their derivative/barrier structure is unusually promising.

### Stage 4: Barrier/proof audit

For surviving candidates, compute derivative expressions or sampled monotonicity diagnostics along the late tail. Prioritize candidates with:

```text
- stable sign in a tail box,
- simple wall identity,
- cancellation of bad singular terms,
- positive norm structure,
- divergence or nonzero finite limit as |a| -> infinity.
```

## Suggested Output Tables

Codex should produce at least these outputs.

### Defect value table

Columns:

```text
a,
b,
T,
defect_name,
defect_value,
abs_value,
sign,
solver_tolerance,
terminal_method,
notes
```

### Defect summary table

Columns:

```text
defect_name,
vanishes_at_known_compact,
large_positive_sign,
large_negative_sign,
limit_value,
apparent_extra_zeros,
terminal_sensitivity,
recommended_priority,
notes
```

### Tail diagnostic table

For barrier-type quantities, sample along a late-tail box such as

```text
p <= 0.33,
t in [3.5, 4.0],
x3 <= -0.36.
```

Columns:

```text
a,
t,
p,
x1,
x2,
x3,
C,
B_mu = C - mu*p^3,
Q,
W,
selected_derivative_values
```

## Important Implementation Cautions

Do not evaluate expressions containing `p^(-k)` exactly at `p=0`. Use algebraically scaled endpoint equivalents when possible.

For example:

```text
Q = 1 + 6*b*x3/p^2
```

is not endpoint-safe, but

```text
W/b = (p^2 + 6*b*x3)/b
```

has endpoint value `6*x3` when `b != 0`.

For the limiting `b=0` IVP, define the endpoint at the first `p=0` crossing and use limiting expressions in `x` variables only.

Treat fixed-`epsilon` pre-terminal samples as diagnostics, not exact defects, unless converted into an `epsilon -> 0` limiting statement.

Known compact values `a=-36` and `a=108/5` should be used as calibration checks. A defect that does not vanish there is probably implemented incorrectly, unless the integration or chart convention differs.

Positive `a=36` is not expected to close in the unmodified standard `K_-` chart, so do not use it as a compact-zero calibration point.

## Minimal Defect Registry Sketch

This is only a sketch. Adapt to the existing project layout.

```python
import math


def C_of_state(s):
    return s["x1"]*s["x2"] - (s["p"]**2)*s["x3"]/6.0


def W_of_state(s):
    return s["p"]**2 + 6.0*s["b"]*s["x3"]


def defect_x3(s):
    return s["x3"]


def defect_x2(s):
    return s["x2"]


def defect_C(s):
    return C_of_state(s)


def defect_x3_C_norm2(s, lam=1.0):
    C = C_of_state(s)
    return s["x3"]**2 + lam*C**2


def defect_C_plus_lam_x3(s, lam):
    return C_of_state(s) + lam*s["x3"]


def defect_x3_plus_lam_x2(s, lam):
    return s["x3"] + lam*s["x2"]


def defect_S1(s):
    return s["x3"]**3 - 4.0*s["x1"]*s["p"]**3


def defect_S3(s):
    return 2.0*s["x3"] - 6.0*s["p"]


def defect_C_over_x1(s):
    return C_of_state(s)/s["x1"]


def defect_x3_damped(s, r=1.0):
    return s["x3"]/(1.0 + abs(s["x1"])**r)


def defect_C_damped(s, r=1.0):
    return C_of_state(s)/(1.0 + abs(s["x1"])**r)


def defect_W_over_b(s):
    # At the terminal event p=0 this equals 6*x3 when b != 0.
    # Use the endpoint-safe equivalent if abs(b) is small.
    if abs(s["b"]) < 1e-14:
        return 6.0*s["x3"]
    return W_of_state(s)/s["b"]
```

The integral defects require history arrays and should be implemented separately.

## Success Criterion for This Codex Pass

This pass is successful if it produces:

```text
1. a defect registry containing the shortlist and enough extension hooks for parameter sweeps;
2. a numerical audit script for finite `a` and the `b=0` limiting IVP;
3. tables ranking defects by large-|a| signal strength, sign stability, extra-zero behavior, and endpoint sensitivity;
4. clear notes identifying which defects should be attempted in a later proof sprint.
```

