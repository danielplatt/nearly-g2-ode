# S7 SU(2)^3 Defect Audit

Reproducibility command:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_defect_audit --write-markdown docs/s7-su2-cubed-defect-audit.md
```

## Summary

Version: `s7-su2-cubed-defect-audit-v1`.
Step size: `2.5e-05`; seed mode: `taylor`.

Top proof candidates selected by the first-pass audit:

| defect | limit value | known compact max abs | note |
|---|---:|---:|---|
| `D_x3` | -1.17314297668 | 4.85e-05 | x3(T) |
| `D_x3_C_norm2` | 1.37917977161 | 2.35e-09 | x3(T)^2+C(T)^2 |
| `D_S1` | -1.6145549662 | 1.14e-13 | x3(T)^3-4*x1(T)*p(T)^3 |

## Numerical Ranking

| rank | defect | priority | limit abs | min large abs | large signs | extra zero? |
|---:|---|---|---:|---:|---|---|
| 1 | `D_x3` | top | 1.17314 | 0.939432 | +:negative / -:negative | False |
| 2 | `D_x3_C_norm2` | top | 1.37918 | 0.885176 | +:positive / -:positive | False |
| 3 | `D_S1` | top | 1.61455 | 0.829079 | +:negative / -:negative | False |
| 4 | `D_C_IF` | top | 9.04396 | 8.6801 | +:positive / -:positive | False |
| 5 | `D_x3_x2_norm2` | strong | 1.3763 | 0.882565 | +:positive / -:positive | False |
| 6 | `D_S3` | strong | 2.34629 | 1.87886 | +:negative / -:negative | False |
| 7 | `D_3_IF` | strong | 15.183 | 12.2903 | +:negative / -:negative | False |
| 8 | `D_W_over_b` | strong | 7.03886 | 5.63659 | +:negative / -:negative | False |
| 9 | `D_C_plus_m2_x3` | strong | 2.40028 | 1.93028 | +:positive / -:positive | False |
| 10 | `D_C_plus_p2_x3` | strong | 2.29229 | 1.82745 | +:negative / -:negative | False |
| 11 | `D_x3_plus_m10_x2` | watch | 1.23244 | 0.996707 | +:negative / -:negative | False |
| 12 | `D_C_plus_m1_x3` | watch | 1.22714 | 0.990848 | +:positive / -:positive | False |
| 13 | `D_x3_plus_m3_x2` | watch | 1.19093 | 0.956614 | +:negative / -:negative | False |
| 14 | `D_x3_plus_m1_x2` | watch | 1.17907 | 0.945159 | +:negative / -:negative | False |
| 15 | `D_x3_plus_p1_x2` | watch | 1.16721 | 0.933704 | +:negative / -:negative | False |
| 16 | `D_x3_plus_p3_x2` | watch | 1.15535 | 0.922249 | +:negative / -:negative | False |
| 17 | `D_C_plus_p1_x3` | watch | 1.11915 | 0.888016 | +:negative / -:negative | False |
| 18 | `D_x3_plus_p10_x2` | watch | 1.11385 | 0.882157 | +:negative / -:negative | False |

## Infinity Reduction

Status: `reduced_to_singular_endpoint_convergence`.
Sampled preterminal `p'` at `p=0.001`: `-6768408917.3`.

Common reduction: in scaled variables the finite equations have the form

```text
x' = F_0(t,x) + b R_1(t,x) + b^2 R_2(t,x) + b^3 R_3(t,x),  b=1/a.
```

The smooth left seed is also a regular expansion in `b`.  Therefore the
large-`|a|` limit for any endpoint defect in this table is reduced to the
singular endpoint-continuity statement: finite-`b` trajectories and their
first `p=x0=0` event converge to the limiting first crossing.  The sampled
preterminal `p'` is very negative, so the limiting trajectory is already in
a decisive terminal plunge before the singular event.  This is not a proof
of uniform event convergence, but it isolates the needed lemma.

- `D_x3` reduces to the nonzero limit `-1.17314297668` once singular endpoint convergence is proved.
- `D_x3_C_norm2` reduces to the nonzero limit `1.37917977161` once singular endpoint convergence is proved.
- `D_S1` reduces to the nonzero limit `-1.6145549662` once singular endpoint convergence is proved.

### D_x3 Terminal Tail Proof

Status: `terminal_tail_bound`.

For the final terminal layer, use `p=x0` as the independent variable.
Multiplying away the singular powers gives

```text
dt/dp = p^4/A,
dx_i/dp = p*H_i/A,
A = p^4 dp/dt,  H_i = p^3 dx_i/dt,
```

where `A` and the `H_i` are regular at `p=0`.  On the box

```text
0 <= p <= 0.001,
3.59 <= t <= 3.61,
8.5 <= x1 <= 9.5,
0.004 <= x2 <= 0.008,
-1.4 <= x3 <= -0.9,
|b| <= 1e-08
```

the interval bound is `A in [-0.013103144180239277, -0.0026925092524472277]`.
Thus the p-time system has a removable singularity there, and
`|Delta x3| <= 3.56642486934e-05` from `p=0.001` to `p=0`.
The limiting state at `p=0.001` gives the endpoint interval
`[-1.1664997385409208, -1.166428410043534]`.

So `D_x3` has a nonzero negative limiting endpoint value.  The only
remaining asymptotic input is the standard compact-interval continuous
dependence up to the fixed slice `p=0.001`.

## Uniform Tail Attempt

Status: `reduced_not_closed` for candidate `D_x3` and `A=100000000`.
Scalar barrier status: `scalar_margins_positive`.
`x3=-sigma` wall margin: `0.00346442078781`.
`C-Kp^3` limiting wall margin: `3.29549936347e-05`.
Finite-`b` grid sanity margin: `0.000169223423226`.

This is the bounded proof attempt for `D_x3`: the scalar walls are

```text
x3 = -0.36,
C = 1.23*p^3,
p <= 0.33,
t in [3.5, 4.0].
```

Inside that correlated late region, the wall estimates point the right
way for every `|a| >= A`.  If the support-entry lemma were available,
this would keep `x3` negative up to the terminal event, contradicting
the standard `K-` requirement `x3(T)=0`.

The missing part is still the support-entry/containment lemma proving every |a|>=A trajectory reaches and remains in that correlated late region.

## Calibration Samples

| source | a | status | T | x2 | x3 | C |
|---|---:|---|---:|---:|---:|---:|
| exact | -36 | crossed | 6.28317701 | -1.83184375e-10 | 4.50646286e-05 | -2.03010017e-10 |
| exact | 21.6 | crossed | 8.42977842 | 3.12596977e-09 | 4.85196048e-05 | -2.3083924e-10 |
| exact | 36 | crossed | 6.2831768 | -0.0368757974 | -1.21051854 | -0.0407076635 |
| exact | -100 | crossed | 3.72547675 | 0.00467347488 | -0.657402113 | 0.0400799817 |
| exact | 100 | crossed | 3.725477 | 0.00300407251 | -1.84827142 | 0.0257701625 |
| exact | -250 | crossed | 3.61700025 | 0.00572748539 | -0.939431787 | 0.0514159069 |
| exact | 250 | crossed | 3.61700026 | 0.00564939634 | -1.40920971 | 0.0507143224 |
| exact | -500 | crossed | 3.60235147 | 0.00590479086 | -1.05044284 | 0.0534399422 |
| exact | 500 | crossed | 3.60235148 | 0.00589695973 | -1.28533409 | 0.0533687241 |
| exact | -1000 | crossed | 3.59872539 | 0.00578979922 | -1.11421345 | 0.0519144353 |
| exact | 1000 | crossed | 3.59872529 | 0.00582197709 | -1.23062077 | 0.0524108528 |
| exact | -5000 | crossed | 3.59756381 | 0.00593484532 | -1.16056381 | 0.0540108244 |
| exact | 5000 | crossed | 3.59756381 | 0.00593405562 | -1.18412189 | 0.0540036623 |
| exact | -10000 | crossed | 3.59752504 | 0.00587303501 | -1.14492343 | 0.0529509852 |
| exact | 10000 | crossed | 3.59752504 | 0.00587425053 | -1.1565922 | 0.052961967 |
| limit |  | crossed | 3.5975244 | 0.00592944197 | -1.17314298 | 0.0539937763 |
