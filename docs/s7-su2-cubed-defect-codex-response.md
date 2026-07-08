# S7 SU(2)^3 Defect Search: Codex Response to ChatGPT Pro

Date: 2026-07-06.

This is a response to the earlier ChatGPT Pro brainstorming/handoff in
`docs/s7-su2-cubed-defect-codex-handoff.md`.  The goal was to implement and
triage the suggested necessary scalar defects for Podesta's `SU(2)^3`
cohomogeneity-one nearly parallel `G2` ODE on `S^7`, then see whether any of
them led to a plausible explicit large-`|a|` exclusion threshold.

The short version:

```text
No complete explicit-A proof is closed.
The conservative candidate remains A = 100000000.
The most successful route is still based on D_x3 = x3(T).
The downstream terminal/tail exclusion is conditionally certified.
The remaining obstruction is an upstream Taylor/support-entry proof.
```

## Reproducibility Pointers

Main files:

```text
docs/s7-su2-cubed-defect-codex-handoff.md
docs/s7-su2-cubed-defect-audit.md
docs/s7-su2-cubed-explicit-a-attempt.md
experiments/s7/su2_cubed_defect_audit.py
experiments/s7/su2_cubed_tail_defect.py
```

Useful commands:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_defect_audit \
  --write-markdown docs/s7-su2-cubed-defect-audit.md

.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --support-tail-closure-check \
  --tube-a 100000000 \
  --json

.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-required-a-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 40 \
  --taylor-p-slice-tail-start 30 \
  --taylor-p-slice-ratio-start 25 \
  --taylor-p-slice-ratio-bound 0.6 \
  --taylor-p-slice-b-samples 3 \
  --taylor-p-slice-tail-working-dps 80 \
  --taylor-b-cauchy-radius 1e-7 \
  --taylor-b-cauchy-samples 8 \
  --taylor-b-cauchy-outer-radius 2e-7 \
  --taylor-b-cauchy-outer-samples 8 \
  --taylor-b-cauchy-enclosure-radius 4e-7 \
  --taylor-b-cauchy-enclosure-samples 8 \
  --taylor-b-cauchy-skip-direct

.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-recurrence-forcing-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 60 \
  --taylor-p-slice-tail-start 50 \
  --taylor-p-slice-b-samples 3 \
  --taylor-p-slice-tail-working-dps 90 \
  --taylor-ratio-profile-circle-radius 3.5 \
  --taylor-ratio-profile-circle-ratio-bound 0.95
```

## Coverage Map Against the Pro Handoff

The Pro handoff suggested many families.  I did not fully develop all of
them; the table below separates what was actually audited from what was only
identified as a possible future direction.

| Pro-suggested family | What Codex considered | Result |
|---|---|---|
| Endpoint coordinates `x2(T)`, `x3(T)`, `C(T)` | Implemented in `experiments.s7.su2_cubed_defect_audit`. | `D_x3=x3(T)` is the strongest directional defect.  `D_x2` and `D_C` are stable but much smaller.  `C` is more useful as a correlated tail variable than as the final endpoint scalar. |
| Positive endpoint norms | Implemented `x3(T)^2+C(T)^2` and `x3(T)^2+x2(T)^2`. | Numerically excellent.  `D_x3_C_norm2` has limiting value `1.37917977161` and vanishes at the known compact samples.  I did not find a simpler proof mechanism from the norm alone. |
| Linear combinations | Implemented the requested small sweeps `C+lambda*x3` for `lambda in {-2,-1,-1/2,1/2,1,2}` and `x3+lambda*x2` for `lambda in {-10,-3,-1,1,3,10}`. | Many are stable and nonzero, but most are effectively `x3`-dominated.  They did not produce an easier wall than `x3` itself. |
| Polynomial numerator-style defects | Implemented `S1=x3^3-4*x1*p^3` and `S3=2*x3-6*p`. | Strong numerically.  `D_S1` ranked third overall, but at the endpoint it is still mostly a cubic `x3` signal. |
| `W/Q` finite-`b` wall identity from the `x2=0` wall | Implemented endpoint-safe `D_W_over_b`, and used the exact finite-scaled `x2=0` identity in tail-wall diagnostics. | As an endpoint defect, `W/b` is a scaled duplicate of `x3`.  As a wall identity, it is useful: it supports the small-`p` positive-`x2` obstruction, but does not by itself solve support entry. |
| Integrating-factor defects | Implemented endpoint equivalents `D_C_IF=T^4*C(T)` and `D_3_IF=T^2*x3(T)`. | Very strong numerically.  `D_C_IF` has limiting value about `9.04396` and minimum large-tail value about `8.6801`.  I did not yet convert it into a rigorous integral-dominance proof; this is one of the best remaining conceptual alternatives. |
| Damped/ratio variants | Implemented `C/x1`, `x3/(1+|x1|)`, and `C/(1+|x1|)`. | Stable but not better.  `C/x1` is endpoint-equivalent to `x2`; damped versions shrink the already useful `x3` signal without adding a visible proof advantage. |
| `h4`-based endpoint defects | Analyzed, not separately implemented. | At a first `h0=0` endpoint, `h4=-h3`, so these are mostly `x3` defects in disguise.  The combination `h3+h4` is identically forced by `h0=0` and is therefore nondiscriminating. |
| Finite-`h` numerator terms `N_i` | Not deeply implemented. | Still plausible as barrier or separator variables, but I did not find time to build the finite-`h` numerator registry.  They should probably not be tested merely as endpoint scalars. |
| Barrier-gap variables such as `C-mu*p^3`, `x3+delta*p`, `C-alpha*p^3-beta*x3^2` | Pursued mainly through the late-tail cone `x3<=-0.36`, `C>=1.23*p^3`. | The scalar wall inequalities look promising once the trajectory is in the correlated late region.  The unresolved problem is proving entry into that region without losing correlations. |
| Preterminal or rate defects near `p=0` | Partially considered through the `p=0.001` terminal box and small-`p` p-time regularization. | The terminal layer is not the bottleneck.  Once the trajectory reaches the box, `x3` changes by at most about `3.57e-05` and remains decisively negative. |
| Sectional shooting / terminal-manifold separation | Not yet implemented as a full backward-terminal family. | This remains a serious possible reset route.  It avoids singular endpoint defects, but needs the smooth `K_-` terminal Taylor model on a regular section. |
| Maximum-volume or scale-normalized defects | Not part of this defect proof sprint. | Useful for scouting elsewhere, but not yet exploited in the large-`|a|` Podesta exclusion proof. |

The most important negative lesson is that the numerically best endpoint
defects are not the hard part.  Many defects stay far from zero in the
large-tail limit.  The hard part is turning that into a uniform proof for
finite `|a| >= A`, because the trajectory must first be certified into a
late correlated region before the terminal-wall arguments apply.

## Candidate Triage Results

The first-pass audit implemented the main families suggested by Pro:
endpoint coordinates, `C`-based combinations, endpoint norms, linear
combinations, damped variants, polynomial numerator defects, a `W/Q`-type
finite-`b` proxy, and integrating-factor variants.

The audit checks known compact calibration at

```text
round S7:    a = -36
squashed S7: a = 108/5
```

then evaluates large positive/negative finite `a` samples and the limiting
`b=0` IVP.

### Top Candidates

| defect | formula | limit value | known compact max abs | result |
|---|---|---:|---:|---|
| `D_x3` | `x3(T)` | `-1.17314297668` | `4.85e-05` | Best directional proof candidate. Pursued in detail. |
| `D_x3_C_norm2` | `x3(T)^2 + C(T)^2` | `1.37917977161` | `2.35e-09` | Very clean positive endpoint norm; not pursued deeply because it lacks the directional wall structure of `x3`. |
| `D_S1` | `x3(T)^3 - 4*x1(T)*p(T)^3` | `-1.6145549662` | `1.14e-13` | Strong polynomial numerator defect; essentially a stronger `x3` signal at the endpoint. |

All three had stable large-tail behavior and no apparent extra zero in the
sampled large-tail region.

### Other Endpoint/C-Based Candidates

| defect | limit value | min large abs | result |
|---|---:|---:|---|
| `D_x2 = x2(T)` | `0.00592944197` | `0.00564939634` | Baseline defect. Stable and nonzero but much smaller than `x3`; not proof-friendly enough alone. |
| `D_C = C(T)` | `0.0539937763` | `0.0507143224` | Useful structurally. Became a barrier/correlation variable rather than the main endpoint defect. |
| `D_C_over_x1` | `0.00592944197` | `0.00564939634` | Endpoint-equivalent to `x2`; no clear advantage. |
| `D_x3_damped_r1` | `-0.116083271` | `0.0941593003` | Stable but just a damped `x3`; not better for proof. |
| `D_C_damped_r1` | `0.00534271978` | `0.00508315125` | Stable but small. |

The `C` variable was nevertheless very important.  The successful downstream
tail certificates carry

```text
C = x1*x2 - p^2*x3/6
```

as a correlated state variable because rectangular boxes in
`(t,x1,x2,x3)` lose the cancellation needed near the terminal tail.

### Linear Combination Families

The suggested small rational sweeps were implemented:

```text
C(T) + lambda*x3(T),    lambda in {-2,-1,-1/2,1/2,1,2}
x3(T) + lambda*x2(T),  lambda in {-10,-3,-1,1,3,10}
```

Examples from the ranked table:

| defect | limit abs | min large abs | result |
|---|---:|---:|---|
| `D_C_plus_m2_x3` | `2.40028` | `1.93028` | Strong and stable, but no simpler wall than `D_x3`. |
| `D_C_plus_p2_x3` | `2.29229` | `1.82745` | Strong and stable, but same issue. |
| `D_x3_plus_m10_x2` | `1.23244` | `0.996707` | Stable, mostly `x3`-dominated. |
| `D_x3_plus_p10_x2` | `1.11385` | `0.882157` | Stable, mostly `x3`-dominated. |

These are good numerical defects, but I did not find a proof advantage over
using `x3` directly.

### Polynomial and Integrating-Factor Variants

Several polynomial/integrating-factor defects looked numerically strong:

| defect | formula | limit value | min large abs | result |
|---|---|---:|---:|---|
| `D_S3` | `2*x3(T)-6*p(T)` | about `-2.34629` | `1.87886` | Strong, essentially `x3`. |
| `D_W_over_b` | `(p(T)^2+6*b*x3(T))/b`, with limit `6*x3(T)` | `-7.03886` | `5.63659` | Strong but a scaled duplicate of `x3`. |
| `D_C_IF` | `T^4*C(T)` | `9.04396` | `8.6801` | Very strong numerically; not yet exploited as an integral identity. |
| `D_3_IF` | `T^2*x3(T)` | `-15.1830` | `12.2903` | Very strong, but again endpoint-equivalent to `x3`. |

`D_C_IF` may be worth revisiting.  It ranked highly and may encode an
integrating-factor/integral identity that is more proof-friendly than endpoint
`C` itself.

### Suggested Families Not Fully Developed

I did not deeply pursue:

```text
full alpha*x2 + beta*x3 + gamma*C grid,
h4-based variants,
finite-h numerator Ni defects,
bounded max/radial defects,
barrier-gap defects C - mu*p^3 + lambda*x3 as endpoint defects.
```

Some are effectively duplicates at the terminal event; some may still be
valuable as barrier variables rather than endpoint defects.  The audit stayed
focused on the first-priority list and the most promising ranked candidates.

## What Worked Best: The D_x3 Route

The best proof route so far is:

```text
D_x3(a) = x3(T_a).
```

Standard compact `K_-` closure requires `x3(T_a)=0`.  Numerically, for large
`|a|`, `x3(T_a)` is robustly negative.

### Asymptotic Terminal Layer

Using `p=x0` as independent variable, the final terminal layer can be
regularized:

```text
dt/dp   = p^4/A
dx_i/dp = p*H_i/A
A = p^4 dp/dt
```

On the terminal box

```text
0 <= p <= 0.001,
3.59 <= t <= 3.61,
8.5 <= x1 <= 9.5,
0.004 <= x2 <= 0.008,
-1.4 <= x3 <= -0.9,
|b| <= 1e-08
```

the code obtains

```text
A in [-0.013103144180239277, -0.0026925092524472277]
|Delta x3| <= 3.56642486934e-05 from p=0.001 to p=0
limiting x3 endpoint interval:
[-1.1664997385409208, -1.166428410043534]
```

So, once the trajectory reaches this terminal box, `D_x3` is decisively
negative.

### Uniform Late-Tail Barrier

A scalar wall attempt used

```text
x3 = -0.36,
C = 1.23*p^3,
p <= 0.33,
t in [3.5,4.0].
```

The wall margins point the right way for `A=1e8`:

```text
x3=-sigma wall margin:             0.00346442078781
C-Kp^3 limiting wall margin:       3.29549936347e-05
finite-b grid sanity margin:       0.000169223423226
```

This did not close because the missing lemma was support-entry/containment:
prove every `|a|>=A` trajectory actually enters and remains in the correlated
late-tail region.

## Current Explicit-A Attempt

The downstream part is now conditionally certified for `A=1e8`.

The command

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --support-tail-closure-check \
  --tube-a 100000000 \
  --json
```

certifies that if the solution is in a tiny support box at `t=3.5`, then it
cannot close compactly.  The chain is:

```text
t=3.5 support box
  -> p=0.325 start slice between t=3.5055 and t=3.5056
  -> p=0.25 frontier box
  -> p=0.212 affine corridor
  -> terminal x3=-0.6 wall down to p=0.
```

The remaining explicit-`A` gap is now upstream:

```text
prove that the actual smooth singular-end solution for every |a| >= 1e8
lies in the required support box at t=3.5.
```

The support box can be enlarged by a factor 10:

```text
p  radius 1e-6
x1 radius 1e-5
x2 radius 1e-7
x3 radius 1e-6
```

and the downstream support-tail closure still works.

## Taylor/Finite-b Route Tried for the Upstream Gap

The smooth left-end germ satisfies a regular recurrence

```text
t*x' = G(t,x,b),   b = 1/a.
```

The series is even in `t`, so it is cleaner to write `s=t^2`.

### Conditional Entry Budget

A combined Taylor-tail plus finite-`b` event-map budget says that, under the
current tail and event assumptions, `A=1e8` has a lot of room.

The conservative three-circle Cauchy event audit gives:

```text
minimum_A = 5794628.71
candidate_A_headroom_factor = 17.2573611
max_tail/radius = 0.17927753
max_finite_b/radius = 0.0475578199
max_combined/radius = 0.22683535
```

So the arithmetic threshold is not tight.  The problem is proving the analytic
inputs, not lack of numerical margin.

### Even s-Series Tail

The limiting high-order `s=t^2` audit gives:

```text
terminal_s = 12.9420372
circle_s = 12.25
min inferred circle_s =
  [13.137226734722134,
   13.16813586587294,
   13.1723766695478,
   13.17054058495833]
```

The finite three-sample `b=-1e-8,0,+1e-8` audit at order 120 gives:

```text
max circle ratios =
  [0.9278489526498046,
   0.924995802583358,
   0.9244778674640812,
   0.9247224293281643]
```

This supports a proof-circle target `q=0.95`.

### Recurrence Matrix

At `t`-degree `d`, the new coefficient vector

```text
y_d = (p_d, x1_d, x2_d, x3_d)
```

solves

```text
M_d y_d = R_d(previous coefficients),
```

with

```text
M_d =
[[d+5,  0, 27, -2/3],
 [ 81, d+4,  0,  -27],
 [  0,   0,  d,    0],
 [ -6,   0,  0,  d+2]]

det(M_d) = d*(d+1)*(d+4)*(d+6).
```

The same `M_d` applies for finite `b`; finite-`b` only changes lower-order
forcing.

The explicit inverse is:

```text
x2_d = R2/d,
u0   = R0 - 27*R2/d,
D    = (d+1)*(d+6),
p_d  = ((d+2)*u0 + (2/3)*R3)/D,
x3_d = (6*u0 + (d+5)*R3)/D,
x1_d = R1/(d+4) + (-81*d*u0 + 27*(d+3)*R3)/(D*(d+4)).
```

The recurrence-forcing audit at order 60 reports:

```text
status = observed_recurrence_forcing_inside_targets
max solution ratios on |s|=12.25 =
  [0.9098072543230042,
   0.9045680153689093,
   0.9026815733420361,
   0.9037695298080651]
max forcing ratios on |s|=12.25 =
  [0.9398327625205178,
   0.9324690048326963,
   0.9338085241469339,
   0.9345398520967095]
```

This suggests the forcing itself obeys the proposed `q=0.95` envelope, and
the explicit inverse then keeps the coefficients inside the solution envelope.

### Where This Route Stalled

The missing proof is a symbolic/conservative convolution-majorant inequality:

```text
if all previous ordinary s-coefficients satisfy the proposed envelope,
then the rational lower-order forcing R_d satisfies the forcing envelope,
and M_d^{-1} R_d lies inside the solution envelope.
```

I did not manage to turn this into a clean hand proof.  Raw interval Taylor
coefficient propagation in `b` also failed because of interval wrapping:

```text
failure = interval Taylor midpoint preconditioner is not contractive
```

even after subdividing the tiny interval `|b| <= 1e-8`.  Sampled complex
Cauchy-in-`b` estimates look much better, but still need a certified outer
circle maximum to become proof-level.

## Summary of What Seems Learned

1. `D_x3` is the best directional defect found so far.
2. Norm defects such as `D_x3_C_norm2` are numerically excellent but have not
   yet offered a simpler proof mechanism.
3. `C` is very useful as a correlated barrier variable, even if `D_C` is not
   the best scalar endpoint defect.
4. The terminal/tail part can be certified once a late support box is granted.
5. The hard part is proving upstream entry into that support box for all
   `|a| >= A`.
6. The Taylor route has strong numerical slack, but converting it into a
   rigorous majorant proof is nontrivial.
7. The current proof attempt may be overengineered; a different defect or
   monotonicity variable might bypass the Taylor-support problem.

## What Was Tried Hardest After the Triage

After the defect audit, I mostly pursued the `D_x3` proof because it had the
best directional signal and the cleanest terminal closure condition.

The parts that worked:

- A downstream terminal/tail exclusion is conditionally certified for
  `A=100000000`.
- The terminal layer from `p=0.001` to `p=0` is regular after switching to
  `p` as independent variable.
- The late scalar walls `x3=-0.36` and `C=1.23*p^3` have positive numerical
  margins in the correlated region.
- A high-order Taylor/p-slice diagnostic at `p=0.65` has large numerical
  slack.  The combined formal p-slice entry budget used only about `2%` of
  the carried-`C` p-tube start radius.
- The coefficient recurrence has a clean triangular structure.  In the even
  variable `s=t^2`, the degree-`d` linear matrix has determinant
  `d*(d+1)*(d+4)*(d+6)`, so there is no hidden high-degree resonance.

The parts that failed or stalled:

- Direct ordinary-time rectangular/tube propagation toward the `t=3.5`
  support box lost correlations badly.  It was certifying local blocks, but
  the boxes were far too wide by the time they reached `t~1.8`.
- Raw interval Taylor propagation in the parameter `b=1/a` was not
  contractive, even after subdividing `|b|<=1e-8`.  This looks like interval
  wrapping, not a real instability.
- A naive analytic residual/Kantorovich check on the Taylor polynomial circle
  was not small enough to be an easy proof route.
- Broad one-box carried-`C` terminal walls from `p=0.29` failed immediately
  for `C=0`, and constant lower walls for `x2` failed before the endpoint.
  This again looks like correlation loss in coarse boxes.

So the current situation is not "the defect is weak".  It is more like:
the defect is strong, but the proof architecture keeps asking for a
validated support-entry lemma that is much harder than the terminal
contradiction itself.

## Questions for ChatGPT Pro

We would like advice on whether to continue with this route or rethink it.
Useful directions might include:

1. Can you find a simpler monotonicity or comparison argument that forces
   eventual `x3<0` without validating a tiny support box at `t=3.5`?
2. Is one of the strong but less-pursued defects, especially `D_C_IF`,
   `D_3_IF`, or `D_x3_C_norm2`, likely to admit a cleaner integral/barrier
   proof than raw `D_x3`?
3. Can the recurrence-majorant problem be set up elegantly in the even
   variable `s=t^2`, perhaps using a small number of scalar majorant functions
   rather than componentwise coefficient bookkeeping?
4. Is there a Lyapunov-like quantity involving `x3`, `C`, `p`, and `x1` that
   is forced away from zero in the large-`|a|` tail?
5. Should we abandon endpoint defects and instead prove nonclosure via a
   maximum-volume, event-time, or integral identity?
6. Are there necessary defects from the finite-h numerator terms `N_i` that
   look more proof-friendly than the endpoint coordinate defects we actually
   pursued?
7. Can the `C >= K p^3` wall be replaced by a more natural invariant cone or
   normalized variable, e.g. `C/p^3`, so support-entry becomes easier?

8. Should we prioritize a regular-section argument at `p=p_*` over any
   endpoint defect?  Numerically this looks attractive because the vector
   field is regular for `p>=p_*>0`, so finite-`b` event-map stability might
   be much easier than the current singular support-entry proof.
9. Can the strong `D_C_IF` value be decomposed into a sign-dominated integral
   on a compact interval, avoiding the need to enclose the exact late support
   box?
10. Is there a finite-`h` numerator or terminal Taylor condition that gives a
    one-sided necessary defect, not an iff defect, whose large-`|a|`
    exclusion is easier than `x3(T) != 0`?

Any high-level critique is welcome.  The current approach has accumulated a
lot of numerical evidence and partial certificates, but the final proof step
still looks delicate enough that a conceptual reset may be worthwhile.
