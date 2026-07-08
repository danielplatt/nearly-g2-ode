# S7 SU(2)^3 large-|a| exclusion: next Codex sprint

Date: 2026-07-06

This is a follow-up work order after the first Codex defect sprint. The previous sprint produced useful evidence and a conditional downstream certificate, but it narrowed too early to one proof architecture: prove a very small support box at `t=3.5`, then run the existing `D_x3` tail exclusion. The next sprint should preserve the useful downstream result, but should not keep forcing the same upstream Taylor/support-entry proof unless the alternatives below fail clearly.

## One-sentence objective

Find a robust regular-section or integral proof route showing that large-|a| trajectories enter a coarse correlated tail region that implies terminal nonclosure, preferably without proving a tiny `t=3.5` support box.

## Current state to preserve

The previous sprint found:

```text
No complete explicit-A proof is closed.
Candidate explicit threshold remains A = 100000000.
Best directional endpoint defect so far: D_x3(a) = x3(T_a).
Downstream terminal/tail exclusion is conditionally certified.
The bottleneck is upstream support-entry into a tiny box near t = 3.5.
```

The strongest numerical endpoint candidates were:

```text
D_x3        = x3(T)                         limit about -1.17314297668
D_x3_C_norm = x3(T)^2 + C(T)^2              limit about  1.37917977161
D_S1        = x3(T)^3 - 4*x1(T)*p(T)^3      limit about -1.6145549662
D_C_IF      = T^4*C(T)                      limit about  9.04396
```

The existing downstream terminal box result should be kept as a lemma. Once the trajectory reaches approximately

```text
0 <= p <= 0.001,
3.59 <= t <= 3.61,
8.5 <= x1 <= 9.5,
0.004 <= x2 <= 0.008,
-1.4 <= x3 <= -0.9,
|b| <= 1e-8,
```

then the terminal drift in `x3` is tiny and the endpoint remains decisively negative. Do not discard this. Repackage it as a downstream lemma with explicit hypotheses and slack.

The problem is not the terminal layer. The problem is proving entry into a correlated late-tail region for all `|a| >= A`.

## Main guardrail

Do not spend the sprint only increasing Taylor order, tightening the `t=3.5` support box, or optimizing `A`. The arithmetic threshold has headroom. The missing ingredient is a proof architecture that avoids or weakens the tiny support-entry requirement.

The next sprint should explore at least these four branches:

1. Regular `p=p_*` event sections and finite-`b` event-map stability.
2. The normalized cone variable `c = C/p^3`.
3. The integrating-factor identity behind `D_C_IF = T^4*C(T)`.
4. Sectional terminal-manifold separation.

Keep `D_x3` as the downstream contradiction, but do not assume raw endpoint `x3` is the only proof object.

---

# Mathematical notation

Use the scaled variables

```text
p = x0,
C = x1*x2 - p^2*x3/6,
b = 1/a.
```

At a standard compact right endpoint with first `p=0` event `T_a`, closure forces

```text
x2(T_a) = 0,
x3(T_a) = 0,
C(T_a)  = 0,
```

assuming the usual nonzero `f1` condition. Thus any scalar involving these quantities and vanishing when they vanish is a valid necessary defect.

The previous sprint already found that `x3(T)` is numerically much stronger than `x2(T)`, while `C` is structurally important because rectangular boxes lose correlations between `x1`, `x2`, and `x3`.

---

# Task 0: preserve and generalize the downstream tail lemma

## Goal

Turn the existing conditional `D_x3` tail exclusion into a reusable lemma with parameterized input hypotheses.

## Current input

The previous command

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --support-tail-closure-check \
  --tube-a 100000000 \
  --json
```

certifies a chain roughly of the form

```text
t=3.5 support box
  -> p=0.325 start slice
  -> p=0.25 frontier box
  -> p=0.212 affine corridor
  -> terminal x3=-0.6 wall down to p=0.
```

## Requested change

Do not hard-code the proof around the exact tiny `t=3.5` box. Refactor the downstream check so that it accepts a coarse correlated section/cone input, for example:

```text
p <= p0,
t in [t_minus,t_plus],
x1 in [m1,M1],
x2 in [m2,M2],
x3 <= -sigma,
C >= K*p^3,
|b| <= b0.
```

Then output the largest or most forgiving such hypotheses the current downstream machinery can tolerate.

## Acceptance output

Produce a Markdown/JSON table with columns:

```text
p0, t_range, x1_range, x2_range, x3_upper, K, b0,
terminal_x3_upper_bound, minimum_wall_margin, status
```

Prefer wide hypotheses over sharp endpoint estimates. A downstream lemma that accepts a broad cone is more useful than one that gives a slightly better terminal value from a tiny box.

---

# Task 1: replace the `t=3.5` support target by regular `p=p_*` sections

## Goal

Find a regular section `Sigma_* = {p = p_*}` where the limiting solution and finite large-|a| solutions already lie in a robust cone implying the downstream tail lemma.

## Section grid

Audit at least

```text
p_* in {0.45, 0.40, 0.35, 0.33, 0.30, 0.25}.
```

Add nearby values if the margins change abruptly.

## Quantities to record

For each `p_*`, compute the limiting `b=0` event values:

```text
t_*, x1_*, x2_*, x3_*, C_*, c_* = C_*/p_*^3.
```

Also record transversality:

```text
dp/dt at event,
abs(dp/dt),
```

and the cone margins for a range of candidate constants:

```text
x3 <= -sigma,
c  >= K,
x2 >= m2,
x1 >= m1,
t in [t_minus,t_plus].
```

Suggested initial scan:

```text
sigma in {0.20, 0.25, 0.30, 0.36, 0.45, 0.60},
K     in {0.50, 0.75, 1.00, 1.23, 1.50, 2.00}.
```

## Finite-b checks

For `|b| <= 1e-8`, sample at least

```text
b in {-1e-8, -5e-9, 0, 5e-9, 1e-8}
```

and report finite-b deviations from the limiting section values. If available, also run complex Cauchy-in-b or interval enclosures, but do not let this become the whole sprint.

## Acceptance output

Produce a ranked table:

```text
p_*, t_*, x1_*, x2_*, x3_*, C_*, c_*, abs(dp/dt),
best_sigma, x3_margin, best_K, c_margin,
finite_b_max_deviation, recommended_box_or_cone, status
```

Rank sections by proof usefulness, not by terminal proximity. Earlier sections are better if they still have robust margins, because they reduce exposure to the singular terminal layer.

## Decision rule

Choose the earliest `p_*` where:

```text
x3 <= -sigma     with comfortable margin,
c  >= K          with comfortable margin,
x2 > 0,
x1 > 0,
abs(dp/dt) not small,
```

and finite-b deviations are small relative to the margins.

---

# Task 2: prove or numerically prepare finite-b event-map stability on `p=p_*`

## Goal

Replace the long Taylor-to-`t=3.5` support proof by a compact ODE perturbation estimate up to a regular section.

On any region with `p >= p_* > 0`, the finite-b scaled ODE is regular and is a perturbation of the limiting system:

```text
dX_b/dt = F(t, X_b, b),
dX_0/dt = F(t, X_0, 0).
```

Use a Gronwall/event-map estimate rather than a high-order Taylor proof all the way to `t=3.5`.

## Desired estimate

On a compact region containing both trajectories up to section crossing, find explicit bounds

```text
||D_X F||       <= L,
||partial_b F|| <= M,
|dp_0/dt|       >= m > 0
```

so that

```text
||X_b(t)-X_0(t)|| <= exp(L*(t-t0))*(initial_error + |b|*M*(t-t0)).
```

Then use transversality to bound event-time error:

```text
|tau_b - tau_0| <= const * sup|p_b - p_0| / m.
```

## Practical version

If a fully rigorous interval proof is too large for one sprint, produce the data needed for one:

```text
chosen section p_*,
compact box used up to p_*,
interval bounds for F, D_XF, partial_bF,
transversality lower bound m,
Gronwall constant,
predicted finite-b error for |b| <= 1e-8,
margin comparison against the cone inequalities.
```

## Acceptance output

A file `docs/s7-su2-cubed-psection-stability.md` containing:

1. The selected `p_*`.
2. The compact box over which regularity is used.
3. The constants `L`, `M`, `m` if available.
4. A comparison showing finite-b event-map error is below the cone margins.
5. A clear statement of any remaining rigorous gap.

---

# Task 3: replace `C >= K*p^3` by the normalized variable `c = C/p^3`

## Goal

Avoid rectangular-box loss by carrying the correlation variable

```text
c = C/p^3.
```

The old wall was

```text
C >= 1.23*p^3,
x3 <= -0.36,
p <= 0.33,
t in [3.5,4.0].
```

In normalized form this is simply

```text
c >= 1.23,
x3 <= -0.36.
```

Use `c` as a state variable, at least in diagnostics. If its ODE is too singular in `t`, use `p` or `tau = -log p` as independent variable.

## Work items

1. Derive and verify the ODE for `c` using symbolic differentiation from `C = p^3*c`.
2. Re-express the late-tail wall checks in variables `(p,t,x1,x2,x3,c)`.
3. Audit boundary derivatives on:

```text
x3 = -sigma,
c = K.
```

4. Search for values of `(sigma,K,p0)` for which the cone is forward invariant toward the terminal event.

## Acceptance output

A table:

```text
p0, sigma, K, t_range, x1_range, x2_range,
x3_wall_margin, c_wall_margin, status
```

and a recommendation:

```text
best invariant cone candidate = {p <= ..., x3 <= ..., c >= ..., ...}
```

If `c` is numerically unstable near `p=0`, report the largest useful range, for example `p in [0.001,0.33]`, and hand off to the existing terminal layer below that.

---

# Task 4: reopen `D_C_IF`, but as an integral identity

## Goal

The previous audit found `D_C_IF = T^4*C(T)` very strong numerically, but it was not exploited as an integral identity. This is a missed opportunity.

The limiting equation for `C` is

```text
C' = -4*C/t + 2*x2*x3^3/(t*p^3) - p^3/t + x1*t^3*p^3/108.
```

Therefore

```text
(t^4*C)' = 2*t^3*x2*x3^3/p^3 - t^3*p^3 + x1*t^7*p^3/108.
```

Under closure, `C(T)=0`; the left endpoint contribution is also controlled by the integrating factor. Hence a necessary integral defect is

```text
D_C_IF_int = integral_0^T [
    2*t^3*x2*x3^3/p^3
  - t^3*p^3
  + x1*t^7*p^3/108
] dt = 0.
```

The endpoint value is numerically large. The question is whether the integral has a robust sign/dominance mechanism before the terminal layer.

## Required diagnostics

Decompose the integral into cumulative pieces:

```text
I1(s) = integral_0^s 2*t^3*x2*x3^3/p^3 dt,
I2(s) = integral_0^s -t^3*p^3 dt,
I3(s) = integral_0^s x1*t^7*p^3/108 dt.
```

For both the limiting IVP and finite large-|a| samples, output cumulative values at:

```text
p = 0.60, 0.45, 0.40, 0.35, 0.33, 0.30, 0.25, 0.20, 0.10, 0.01, 0.001, 0.
```

Also output values at comparable `t` checkpoints if useful.

## Questions to answer

1. Is `D_C_IF_int` already forced to have a large sign before the singular tail?
2. Does one term dominate, e.g. `I3` versus `I1+I2`, on a compact regular interval?
3. Is there a section `p=p_*` after which the remaining tail contribution has a simple bound too small to cancel the accumulated sign?
4. Does the finite-b perturbation preserve the sign for `|b| <= 1e-8`?

## Acceptance output

Produce:

```text
docs/s7-su2-cubed-CIF-integral-audit.md
experiments/s7/su2_cubed_CIF_integral_audit.py
```

or equivalent, with tables for the cumulative integral pieces and a clear conclusion:

```text
promising / not promising / inconclusive
```

If promising, propose the exact compact interval and inequalities needed for a proof.

---

# Task 5: audit the structured scalar `L = x3^3 - (1/2)*t^2*x1*C`

## Goal

Test one new cancellation scalar that was not pursued in the first sprint.

Define

```text
L = x3^3 - (1/2)*t^2*x1*C.
```

This is a necessary endpoint defect because closure forces `x3(T)=0` and `C(T)=0`.

The reason to test it is structural: in the limiting system, the worst `C*x3^3/p^3`-type terms cancel in `L'`.

Using `C = x1*x2 - p^2*x3/6`, the limiting derivative should simplify to

```text
L' = -1/(216*p^3*t) * [
    54*C^2*t^4*x1
  - 648*C*p^3*t^2*x1
  + 36*C*p^2*t^4*x1*x3
  + p^6*t^6*x1^2
  - 108*p^6*t^2*x1
  + 18*p^4*t^4*x1*x3^2
  - 3888*p^4*x3^2
  + 1296*p^3*x3^3
  + 36*p^2*t^2*x3^4
].
```

First verify this formula symbolically in code before using it.

## Required audit

Evaluate `L` and `L'`:

1. At the known compact values `a=-36` and `a=108/5`.
2. For finite large positive and negative `a` samples.
3. In the limiting `b=0` IVP.
4. On the late-tail boxes used in the existing `D_x3` proof.
5. On the new `p=p_*` section boxes from Task 1.

## Acceptance output

Table columns:

```text
a_or_limit, event, p, t, x1, x2, x3, C, L, Lprime, sign_margin
```

Conclusion:

```text
Does L have a stable sign?
Does L' have a stable sign on any useful tail cone?
Is L more proof-friendly than raw x3 or D_C_IF?
```

If not useful, record this and move on. Do not over-focus on `L`.

---

# Task 6: sectional terminal-manifold separation

## Goal

Try to avoid singular endpoint defects entirely. Work on a regular section `p=p_*` and separate the left-shot point from the backward terminal-admissible set.

Compact closure requires the left-shot trajectory to intersect the smooth `K_-` terminal admissible set. On a regular section this becomes a finite-dimensional separation problem.

## Work items

1. Choose one or two promising sections from Task 1, e.g.

```text
p_* = 0.33 or 0.25.
```

2. Integrate the smooth `K_-` terminal model backward to `p=p_*`, varying the terminal free parameter(s) in the relevant range.
3. Construct an enclosure or sample cloud for the terminal-admissible set `M_-` on the section.
4. Compare it to the left-shot point `L_b` on the same section.
5. Search for a scalar separator

```text
ell = alpha*x2 + beta*x3 + gamma*C + delta*t + epsilon*x1 + zeta
```

and then, if needed, quadratic corrections such as

```text
ell = linear terms + q1*x3^2 + q2*C^2 + q3*x3*C.
```

## Acceptance output

Produce a table:

```text
p_*, separator_type, coefficients,
max ell(left-shot enclosure), min ell(terminal enclosure), margin, status
```

or with the inequality reversed.

A useful result is any separator with a robust margin on a regular section, even if it is not yet fully rigorous. This may become the cleanest proof route.

---

# Task 7: keep endpoint norm defects as certificates, not primary walls

The norm defect

```text
D_x3_C_norm2 = x3(T)^2 + C(T)^2
```

is numerically excellent. But differentiating the norm directly may not give a simple invariant wall. Treat it as a clean final certificate:

```text
if x3(T) <= -eta, then D_x3_C_norm2 >= eta^2;
if C(T) >= eta, then D_x3_C_norm2 >= eta^2.
```

Use it to report the final contradiction, not necessarily to drive the barrier proof.

---

# Task 8: deprioritize duplicate x3 defects

The following are useful sanity checks but should not consume much effort:

```text
D_S3     = 2*x3(T) - 6*p(T),
D_W_over_b = (p(T)^2 + 6*b*x3(T))/b,
D_3_IF   = T^2*x3(T).
```

They are strong because they are endpoint-equivalent to `x3`. They do not solve the upstream entry problem.

---

# Task 9: finite-h numerator defects only as barriers or separators

The previous sprint did not deeply pursue finite-h numerator defects `N_i`. Do not test them merely as endpoint scalars; many will be degenerate. If time remains after the main branches, test them as:

1. Barrier variables on the late-tail cone.
2. Components of a section separator `ell`.
3. Combinations whose derivatives cancel the worst singular terms.

Scaled analogues worth checking include combinations related to

```text
x1^2*x2,
x1*x2*x3,
x3^3,
C,
```

especially if they reduce bad powers of `p` in derivatives.

---

# Reporting requirements

Produce one top-level report:

```text
docs/s7-su2-cubed-next-sprint-report.md
```

with sections:

```text
1. Executive summary
2. What was preserved from the previous D_x3 proof
3. p-section audit results
4. finite-b event-map stability results
5. normalized c=C/p^3 cone results
6. D_C_IF integral decomposition
7. L scalar audit
8. terminal-manifold separation attempt
9. recommended proof route
10. remaining gaps
```

Also produce machine-readable outputs where practical:

```text
experiments/s7/output/psection_audit.json
experiments/s7/output/CIF_integral_audit.json
experiments/s7/output/terminal_separator_audit.json
```

Every branch should end with one of:

```text
promising
not promising
inconclusive, with exact blocker
```

Avoid vague conclusions. If a branch fails, record the numerical reason or proof obstacle precisely.

---

# Minimal success criterion for this sprint

A successful sprint does not need to close the proof. It should produce at least one of the following:

1. A regular `p=p_*` cone-entry route that makes the `t=3.5` support box unnecessary.
2. A convincing `D_C_IF` integral dominance mechanism on a compact interval.
3. A robust section separator between the left-shot trajectory and the backward terminal-admissible set.
4. A clear negative result showing that all three alternatives fail, with data strong enough to justify returning to the Taylor-support proof.

The preferred outcome is a proof architecture of the form

```text
finite-b event-map stability to p=p_*
  -> entry into correlated cone in (x3, c=C/p^3)
  -> existing or generalized tail lemma
  -> x3(T) < 0 or x3(T)^2 + C(T)^2 > 0
  -> contradiction to compact K_- closure.
```
