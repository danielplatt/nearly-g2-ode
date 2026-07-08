# S7 SU(2)^3 Tail Defect

Reproducibility command:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --json
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --tube-check
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --p-tube-check
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --hybrid-handoff-check
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --taylor-start-block-check
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --regular-time-corridor-check
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --taylor-restart-chain-check
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --x3-zero-wall-check
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --late-x3-descent-check
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --broad-tail-closure-check
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --support-tail-closure-check
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --adaptive-union-p-tube-check
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --sampled-carried-c-p-tube-check
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --carried-c-p-tube-from-box-check --carried-c-p-tube-source-json output/s7_tail_proof/sampled_carried_c_p_tube_0.65_to_0.3255_tight_attempts120.json
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --automatic-carried-c-p-corridor-check
```

The useful shooting defect for a tail exclusion is not the heuristic loss from
the scout.  Let `T_a` be the first zero of `f0`, equivalently `h0`, for the
one-ended Podesta solution with left parameter `a`.  In scaled variables

```text
h0 = a x0
h1 = x1
h2 = a^3 x2
h3 = a x3
```

define

```text
X2(a) = x2(T_a) = h2(T_a) / a^3.
```

A standard `K-` terminal closure requires `f2(T_a)=0`, hence `X2(a)=0`.  Thus
`X2` is an exact shooting defect for the tail problem.  Numerically, the
large-`|a|` limiting scaled IVP has first crossing

```text
T_infinity ~= 3.598
X2(T_infinity) ~= 0.006
X3(T_infinity) ~= -1.1
```

and conservative finite samples such as `a=+-250, +-500` have the same positive
`X2` sign.

The tail loss should not be treated as monotone in `a`.  The robust numerical
fact is instead asymptotic: after scaling by `h0=a x0`, `h2=a^3 x2`,
`h3=a x3`, the finite equations converge to a limiting singular IVP, and the
limiting terminal defects are bounded away from zero.  Some fixed-step samples
near the singular event are sensitive to step size, so broad finite-`a` plots
are useful diagnostics but not a monotonicity proof.

For example, with a finer terminal step

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --step-size 0.00005 \
  --a-values=-10000,-5000,-1000,1000,5000,10000
```

the mid-large samples have the expected positive scaled tail defect:

```text
a=-10000: X2 ~= 0.00591779, X3 ~= -1.16981
a= -5000: X2 ~= 0.00592846, X3 ~= -1.16160
a= -1000: X2 ~= 0.00596574, X3 ~= -1.10472
a=  1000: X2 ~= 0.00596789, X3 ~= -1.22203
a=  5000: X2 ~= 0.00592749, X3 ~= -1.18518
a= 10000: X2 ~= 0.00591713, X3 ~= -1.18160
```

This is strong evidence for convergence to the nonzero limiting defect, but it
is visibly not the kind of evidence that would support a global monotonicity
claim.

## Scaled Limit

Write

```text
h4 = -h3 - h0^2/6 = -a x3 - a^2 x0^2/6.
```

For `|a| -> infinity`, the exact scaled equations converge on compact
subintervals before `x0=0` to

```text
x0' =
  (-x0 - 3 x2 x3^2 / x0^4) / t
  - t (x1 x2 - x3 x0^2 / 6) / (4 x0^2)

x1' =
  (-4 x1 + x3^3 / x0^3) / t
  + t (x1^2 x2 + x1 x3 x0^2 / 2) / (2 x0^3)

x2' =
  t (-x0^2 x2 x3 / 4 - x1 x2^2 / 2 + t^2 x0^6 / 216) / x0^3

x3' =
  (-2 x3 + 6 x0) / t
  + t (x1 x2 x3 - x3^2 x0^2 / 6 - t^2 x1 x0^4 / 18) / (2 x0^3).
```

The initial data are independent of `a`:

```text
x0(0)=1
x1(0)=27/4
x2(0)=-1/27
x3(0)=3.
```

The current numerical limiting crossing is

```text
T_infinity = 3.5975043...
x1(T_infinity) = 9.05...
x2(T_infinity) = 0.00615...
x3(T_infinity) = -1.12...
```

## Proof Target

It would be enough to prove a lemma of the following form.

```text
Lemma.  The limiting IVP has a transverse first zero of x0 at
T_infinity in [3.59, 3.61], and at that zero x2 >= 0.004.
```

The numerical evidence for the lemma is stronger than the final value alone.
At `t0=3.5`,

```text
x0 = 0.329445368...
x1 = 6.49140718...
x2 = 0.010237118...
x3 = -0.367687579...
```

Two boundary-sign facts then point to a hand-checkable proof:

```text
on x2=0:      x2' = t^3 x0^3 / 216 > 0,
at x3=-0.3:  x3' ~= -2.866 < 0    (at the support state).
```

Thus `x2` cannot cross back through zero while `x0>0`, and the observed
negative `x3` barrier should persist in the last interval after `t0`.
When `x3 <= 0` and `x1,x2>0`, the limiting equation gives the lower inequality

```text
x2' >= - t x1 x2^2 / (2 x0^3).
```

Equivalently,

```text
(1/x2)' <= t x1 / (2 x0^3).
```

Numerically integrating the right side from `t0=3.5` to the limiting first
crossing gives

```text
integral ~= 97.96
x2(T_infinity) >= 1 / (1/x2(t0) + integral) ~= 0.00511.
```

This is the most promising route: rigorously bound the support state, the
`x3=-0.3` barrier, and the Riccati integral by interval Taylor integration.

For finite positive `a`, the exact `x2=0` boundary derivative contains lower
order terms and can change sign extremely close to `x0=0`; the boundary-barrier
argument should therefore be used for the limiting IVP.  The finite-tail
exclusion should then be transferred from the limiting problem by continuous
dependence of the scaled equations and of the transverse event `z=x0^5`.

## Auxiliary Terminal Barrier

There is a cleaner auxiliary obstruction from the other terminal equations.
Right `K-` closure requires `f3=f4=0`.  At an `f0=0` terminal event this is
equivalent, in the scaled variables, to

```text
x3 = 0.
```

On the boundary `x3=0`, all finite-`a` correction terms vanish and the exact
scaled equation gives

```text
x3' = x0 (6/t - t^3 x1/36).
```

Thus while `x0>0`, the boundary `x3=0` is inward whenever

```text
x1 > 216/t^4.
```

At the later support time `t=3.58`, the limiting trajectory has

```text
x0 ~= 0.227560179
x1 ~= 7.56152399
x2 ~= 0.008151319
x3 ~= -0.750051201
216/t^4 ~= 1.31498756
x3'|x3=0 ~= -1.81168264.
```

The finite samples at `a=+-10000` lie in the same region:

```text
a=-10000: x1 ~= 7.56112415, x3 ~= -0.74510447
a= 10000: x1 ~= 7.56112415, x3 ~= -0.75479511
```

This is better than the `x2` sign as a final-end barrier, because the `x3=0`
formula is exact for the finite scaled family.  What remains is still a
validated tube estimate proving that for every `|a|>=A`, not merely the sampled
values, the trajectory enters this late support box and keeps `x1 > 216/t^4`
until `x0` reaches zero.

The broader one-way wall can now be checked directly by interval arithmetic:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --x3-zero-wall-check \
  --tube-a 100000000
```

With the default box

```text
t in [3.02, 3.5]
x0 in [0.30, 0.56]
x1 in [5.0, 7.0]
x2 in [0.005, 0.02]
x3 = 0
```

the command reports

```text
status = certified
x1 threshold margin ~= 2.403
analytic upper bound for x3' ~= -1.03
interval diagnostic upper bound for x3' ~= -0.182
```

The decisive proof condition is analytic, not the interval diagnostic: on
`x3=0` the finite-`a` corrections vanish exactly, and
`x3'=x0(6/t - t^3 x1/36)`.  Thus the wall remains one-way for arbitrarily
small positive `x0` as long as `x1 > 216/t^4`.  The interval diagnostic is kept
as a guard against implementation mistakes, but it can become overconservative
near tiny `x0`.  This does not by itself prove that `x3` reaches zero from the
current `t=3.021` frontier; it isolates the remaining late-regular task to a
descent/crossing estimate for `x3>0`.

That descent stage is now separately certified by

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --late-x3-descent-check \
  --tube-a 100000000
```

It runs an automatic ordinary-time barrier corridor from `t=3.021` to `t=3.45`
with start radius

```text
(1e-4, 1e-3, 1e-5, 1e-4)
```

and safety

```text
(1e-2, 1e-1, 1e-3, 1e-2).
```

The certified end box is

```text
x0 in [0.317516812, 0.392785597]
x1 in [4.828903906, 8.038151051]
x2 in [0.006827124, 0.014499726]
x3 in [-0.600333836, -0.012808657]
```

so the bridge proves `x3<0` and `x0<0.4` at `t=3.45`, conditional on the
`t=3.021` start box.  A fresh p-time tube from `p=0.3905` to `p=0.325`
certifies with a narrow nominal start slice, but the crossing slab produced by
this broad ordinary-time corridor is too wide to initialize that p-tube.  The
next handoff task is therefore to replace this broad descent corridor by a
tighter centered/tuned tube near the `p=0.3905` crossing, or to derive a
coarser p-time barrier that accepts the wide crossing slab.

## Moving-Tube Verifier

The command now includes an experimental interval face checker:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --tube-check \
  --tube-a 100000000 \
  --tube-start 3.5 \
  --tube-end 3.58 \
  --tube-step 0.0001 \
  --tube-subdivisions 2,2,1,2
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --segmented-tube-check \
  --tube-a 100000000 \
  --tube-subdivisions 2,2,1,2
```

It builds a linearly moving rectangular tube around nominal samples at
`b=1/a` values `-1/A`, `0`, and `1/A`, then evaluates the exact scaled RHS over
each slab with interval arithmetic.  For a lower face it checks

```text
F_j([t_i,t_{i+1}], face box, [-1/A,1/A]) >= lower_face_slope,
```

and for an upper face it checks

```text
F_j([t_i,t_{i+1}], face box, [-1/A,1/A]) <= upper_face_slope.
```

Therefore a successful tube certificate is conditional:

```text
if the start box contains the true state, then the end box contains the true
state for every |a| >= A.
```

This is the right computational shape for the proof, but the current simple
rectangular tube does not yet close the full `3.5 -> 3.58` segment.  With
`A=100000000` and the default tube-growth constants, the first failing face is
the lower `x3` face on the slab `[3.5023,3.5024]`, with margin about
`-2.04e-6`.  Subdividing the interval boxes and using asymmetric `x3` radii
moves the obstruction but does not remove it; a representative half-step run
with subdivisions `(2,2,1,2)` and asymmetric `x3` growth fails on the lower
`x3` face near `t=3.5113`, with margin about `-8.9e-6`.  A short slab, for
example `3.5 -> 3.5001`, is certified by the same code; the remaining work is
to replace the ad hoc rectangles by a sharper tube or by better variables.

The segmented verifier chains local tubes and carries the certified end box
forward.  With `A=100000000`, block size `0.001`, and subdivisions `(2,2,1,2)`,
it certifies the implication from the `t=3.5` start box up to `t=3.527`.  The
next block fails on the lower `x0` face at `[3.527,3.5271]`, with margin about
`-3.3e-4`; widening the lower `x0` side moves the bottleneck back to the lower
`x3` face.  This makes the current frontier precise: the remaining proof needs
a non-rectangular or transformed-variable tube for the coupled `x0,x3`
decrease.

## Transformed Tail Tube

The late segment is much better conditioned if `x0` itself is used as the
independent variable.  Write `p=x0` and `y=(t,x1,x2,x3)`.  While `x0' < 0`,
the equivalent p-time system is

```text
dy/dp = (1, x1', x2', x3') / x0'.
```

Near the terminal crossing the individual t-time derivatives blow up, but the
p-time ratios stay moderate.  Along the limiting trajectory,

```text
t=3.58,  p=0.22756: dx1/dp ~= -11.23, dx2/dp ~= 0.0186, dx3/dp ~= 3.42
t=3.597, p=0.11177: dx1/dp ~= -7.07,  dx2/dp ~= 0.0096, dx3/dp ~= 1.86
```

Since `p` decreases, positive `dx3/dp` means `x3` moves further negative in
forward time.  The module now includes a segmented p-time interval face checker:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --p-tube-check \
  --tube-a 100000000 \
  --p-tube-start 0.305 \
  --p-tube-end 0.25 \
  --p-tube-step 0.00005
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --p-tube-check \
  --p-tube-asymmetric-profiles \
  --tube-a 100000000 \
  --p-tube-start 0.305 \
  --p-tube-end 0.2 \
  --p-tube-step 0.00005
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --tuned-p-tube-check \
  --tube-a 100000000 \
  --p-tube-entry-time 2.6 \
  --p-tube-start 0.65 \
  --p-tube-end 0.423 \
  --p-tube-step 0.0005 \
  --tuned-p-tube-max-growth 20,200,2,50 \
  --tuned-p-tube-max-attempts 120
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --staged-union-p-tube-check \
  --tube-a 100000000 \
  --p-tube-start 0.423 \
  --p-tube-step 0.0005 \
  --tuned-p-tube-max-growth 20,200,2,50 \
  --tuned-p-tube-max-attempts 120
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --p-corridor-check \
  --tube-a 100000000 \
  --p-corridor-start 0.25 \
  --p-corridor-end 0.2 \
  --p-corridor-step 0.0005
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --p-corridor-tune \
  --tube-a 100000000 \
  --p-corridor-start 0.25 \
  --p-corridor-end 0.2 \
  --p-corridor-step 0.0005 \
  --p-corridor-tune-x2-slopes=0.02505,0.025075,0.0251 \
  --p-corridor-tune-x1-upper-slopes=-150,-151 \
  --p-corridor-tune-max-runs 6
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --terminal-takeover-check \
  --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --frontier-continuation-check \
  --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --piecewise-corridor-check \
  --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --late-tail-closure-check \
  --tube-a 100000000
```

The checker evaluates the transformed RHS over interval boxes and rejects any
box where the interval for `x0'` touches zero.  Very short p-time slabs certify
cleanly.  With `A=100000000`, step size `0.00005`, and one-step local blocks,
the current p-time checker certifies from `p=0.305` down to `p=0.25`.  The
certified end box is

```text
t  in [3.55532378, 3.58457378]
x1 in [6.79807514, 7.82007514]
x2 in [0.00562601, 0.01154101]
x3 in [-0.72800600, -0.61398507]
```

On this box the cancellation quantity below has interval lower bound `~1.257`,
so the `q=x1=2` wall would still point upward with a large margin.  Attempts to
continue the same rectangular p-time tube below `p=0.25` fail by ordinary
width growth, not by a singular terminal obstruction.  An experimental
asymmetric profile menu, which can widen lower and upper faces separately,
extends the same proof object to about `p=0.2464`; beyond that the carried box
has lost too much correlation for the current rectangular method.  The next
refinement is to combine this certified frontier with scalar barriers, or to
use a sharper piecewise p-time enclosure in a cancellation-adapted variable
such as

```text
C = x3^3 + (t^2/2) x1^2 x2,
```

which controls the singular part of the `x1` equation.

There is now also an automatic tuned p-time tube.  Starting from the nominal
`p=0.65` slice near `t=2.64`, it grows only the face/component that fails the
local p-time check.  With `A=100000000`, step size `0.0005`, and max growth
`(20,200,2,50)`, the command above certifies the conditional bridge

```text
p = 0.65 -> 0.423
blocks = 454
tuning attempts = 13538
worst margin ~= 4.05e-5
end box:
  t  in [3.202583542, 3.603053032]
  x1 in [3.014649320, 9.469593608]
  x2 in [0.008307881, 0.016164379]
  x3 in [-0.641008515, 0.407673277]
```

The saved artifact is
`output/s7_tail_proof/tuned_p_tube_0.65_to_0.423.json`.

Asking for `p=0.42` fails at `p=0.423`/`0.4225`, close to the nominal
`x3=0` crossing, with a tiny lower-`t` face failure after larger growth caps.
This does not close the large-tail proof, but it moves the rigorous
transformed-time frontier from the old static p-tube failure near `p=0.606` to
the natural `x3` transition region.

The first finite-union continuation through that transition is also certified.
It partitions the certified `p=0.423` box into `2 x 2 x 1 x 4 = 16` boxes and
uses midpoint-centred p-time tubes on each branch:

```text
p = 0.423 -> 0.4
leaf boxes = 16
blocks = 736
tuning attempts = 37392
worst margin ~= 3.95e-6
end hull:
  t  in [3.216851743, 3.796034947]
  x1 in [2.302863131, 12.289395242]
  x2 in [0.007160847, 0.017682503]
  x3 in [-1.020262936, 0.491745838]
```

The saved artifact is
`output/s7_tail_proof/staged_union_p_tube_0.423_to_0.4.json`.

The adaptive finite-union continuation now splits only failed carried boxes and
can load the saved `leaf_boxes` from either a staged-union or adaptive-union
JSON file:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --adaptive-union-p-tube-check \
  --tube-a 100000000 \
  --p-tube-start 0.4 \
  --p-tube-end 0.395 \
  --p-tube-step 0.0005 \
  --tuned-p-tube-max-growth 20,200,2,50 \
  --tuned-p-tube-max-attempts 120 \
  --adaptive-union-max-depth 2 \
  --json > output/s7_tail_proof/adaptive_union_p_tube_0.4_to_0.395.json \
  2> output/s7_tail_proof/adaptive_union_p_tube_0.4_to_0.395.log
```

This certifies the short continuation:

```text
p = 0.4 -> 0.395
source boxes = 16
certified leaves = 24
failed leaves = 0
processed boxes = 28
blocks = 264
tuning attempts = 16610
certified worst margin ~= 7.36e-8
end hull:
  t  in [3.219597599, 3.883941643]
  x1 in [2.058515403, 13.300441922]
  x2 in [0.006619965, 0.018162530]
  x3 in [-1.101673772, 0.528214261]
```

The next continuation uses the additional cancellation quantity

```text
C = x1*x2 - p^2*x3/6.
```

The p-time denominator is

```text
p' =
  -p/t - 3*x2*x3^2/(t*p^4)
  - t*C/(4*p^2)
  + finite-|a| corrections.
```

On boxes where `x2 >= 0` and `C > 0`, the last two displayed terms give a
rigorous negative upper bound for `p'`.  The finite-`|a|` corrections are added
back by absolute-value interval bounds.  This removes the fake
`p'=0` denominators that appear if the same broad box is evaluated without the
correlation.

With the cancellation-aware denominator and adaptive splitting in `t,x1,x2`,
the saved `p=0.395` leaves certify all the way to `p=0.39`:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --adaptive-union-p-tube-check \
  --adaptive-union-p-tube-source-json output/s7_tail_proof/adaptive_union_p_tube_0.4_to_0.395.json \
  --tube-a 100000000 \
  --p-tube-start 0.395 \
  --p-tube-end 0.39 \
  --p-tube-step 0.0005 \
  --tuned-p-tube-max-growth 20,200,2,50 \
  --tuned-p-tube-max-attempts 120 \
  --adaptive-union-max-depth 3 \
  --p-tube-cancellation-prime \
  --json > output/s7_tail_proof/adaptive_union_p_tube_0.395_to_0.39_cprime_splitq.json \
  2> output/s7_tail_proof/adaptive_union_p_tube_0.395_to_0.39_cprime_splitq.log
```

The certified payload reports:

```text
p = 0.395 -> 0.39
source boxes = 24
certified leaves = 68
failed leaves = 0
processed boxes = 76
blocks = 734
tuning attempts = 43309
certified worst margin ~= 2.42e-5
end hull:
  t  in [3.221689639, 3.936475893]
  x1 in [1.806834098, 14.316210319]
  x2 in [0.006368667, 0.018754131]
  x3 in [-1.182715411, 0.574049872]
```

Starting from that clean frontier, a shorter continuation to `p=0.3885` also
certifies:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --adaptive-union-p-tube-check \
  --adaptive-union-p-tube-source-json output/s7_tail_proof/adaptive_union_p_tube_0.395_to_0.39_cprime_splitq.json \
  --tube-a 100000000 \
  --p-tube-start 0.39 \
  --p-tube-end 0.3885 \
  --p-tube-step 0.0005 \
  --tuned-p-tube-max-growth 20,200,2,50 \
  --tuned-p-tube-max-attempts 120 \
  --adaptive-union-max-depth 3 \
  --p-tube-cancellation-prime \
  --json > output/s7_tail_proof/adaptive_union_p_tube_0.39_to_0.3885_cprime_splitq.json \
  2> output/s7_tail_proof/adaptive_union_p_tube_0.39_to_0.3885_cprime_splitq.log
```

with summary

```text
p = 0.39 -> 0.3885
source boxes = 68
certified leaves = 138
failed leaves = 0
processed boxes = 148
blocks = 429
tuning attempts = 24189
certified worst margin ~= 1.01e-6
end hull:
  t  in [3.222167536, 3.959740564]
  x1 in [1.699115712, 14.621273913]
  x2 in [0.006244291, 0.018930104]
  x3 in [-1.206921468, 0.595456032]
```

The next requested slab, `p=0.3885 -> 0.3875`, does not close with the current
axis-aligned finite union:

```text
status = failed
source boxes = 138
certified leaves = 572
failed leaves = 110
remaining queue = 0
split count = 110
worst failed attempt margin ~= -0.606
```

The failures are concentrated on `t` faces.  Direct diagnostics show that some
carried four-dimensional boxes have already forgotten enough product
correlation that interval evaluation of `C=x1*x2-p^2*x3/6` can have a negative
lower endpoint even though the center trajectory remains in the positive-`C`
regime.  The next proof-building step should therefore carry `C` as an
additional state/corridor variable, rather than simply increasing rectangular
split depth.

The carried-`C` implementation is now available:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --adaptive-carried-c-union-p-tube-check \
  --adaptive-union-p-tube-source-json output/s7_tail_proof/adaptive_union_p_tube_0.395_to_0.39_cprime_splitq.json \
  --tube-a 100000000 \
  --p-tube-start 0.39 \
  --p-tube-end 0.3895 \
  --p-tube-step 0.0005 \
  --tuned-p-tube-max-attempts 140 \
  --adaptive-union-max-depth 5 \
  --json > output/s7_tail_proof/adaptive_carried_c_union_p_tube_0.39_to_0.3895_sharpC_depth5.json \
  2> output/s7_tail_proof/adaptive_carried_c_union_p_tube_0.39_to_0.3895_sharpC_depth5.log
```

This augments each p-slice box to `(t,x1,x2,x3,C)`, evolves the fifth variable
by

```text
dC/dp = x2 dx1/dp + x1 dx2/dp - p x3/3 - p^2 dx3/dp/6,
```

and evaluates the p-time denominator on the constrained intersection of the
carried `C` interval with the algebraic interval for `x1*x2-p^2*x3/6`.  The
intersection is legitimate for the actual trajectory, which lies on the
constraint graph, and it avoids treating the whole independent five-dimensional
rectangle as physical.

There is one more important handoff sharpening.  After a block certifies, the
outgoing carried `C` interval is intersected again with the algebraic
end-slice interval.  This is rigorous for the same reason: at the next p-slice
the true trajectory still lies on the constraint graph.  Without this handoff
sharpening, the next half-step `p=0.3895 -> 0.389` produced a large queue
explosion; with it, the same step certifies without further splitting.

The first sharpened carried-`C` half-step certifies:

```text
p = 0.39 -> 0.3895
source boxes = 68
certified leaves = 134
failed leaves = 0
processed boxes = 140
split count = 6
tuning attempts = 9367
worst margin ~= 6.09e-7
C handoff lower bound improves from negative to ~= 4.35e-7
```

Continuing from the sharpened five-dimensional leaves gives the current
certified chain:

```text
p = 0.3895 -> 0.389
certified leaves = 134
split count = 0
worst margin ~= 5.89e-5

p = 0.389 -> 0.3885
certified leaves = 134
split count = 0
worst margin ~= 7.42e-6

p = 0.3885 -> 0.388
certified leaves = 155
split count = 3
worst margin ~= 4.75e-5

p = 0.388 -> 0.3875
certified leaves = 155
split count = 0
worst margin ~= 6.17e-5

p = 0.3875 -> 0.387
certified leaves = 198
split count = 5
worst margin ~= 6.07e-5

p = 0.387 -> 0.3865
certified leaves = 198
split count = 0
worst margin ~= 3.42e-5

p = 0.3865 -> 0.386
certified leaves = 212
split count = 2
worst margin ~= 9.02e-6

p = 0.386 -> 0.3855
certified leaves = 226
split count = 2
worst margin ~= 8.35e-5

p = 0.3855 -> 0.385
certified leaves = 226
split count = 0
worst margin ~= 1.19e-5

p = 0.385 -> 0.3845
certified leaves = 255
split count = 3
worst margin ~= 5.85e-5

p = 0.3845 -> 0.3825
certified leaves = 269
split count = 2
worst margin ~= 4.65e-6

p = 0.3825 -> 0.3805
certified leaves = 340
split count = 9
worst margin ~= 5.02e-6

p = 0.3805 -> 0.38
certified leaves = 340
split count = 0
worst margin ~= 5.54e-4

p = 0.38 -> 0.378
certified leaves = 340
split count = 0
worst margin ~= 1.48e-5

p = 0.378 -> 0.3775
certified leaves = 347
split count = 1
worst margin ~= 2.22e-4

p = 0.3775 -> 0.377
certified leaves = 556
split count = 9
worst margin ~= 1.24e-5
```

Thus the old failed `p=0.39 -> 0.385` target is now certified, and the current
frontier is `p=0.377`.  The segment `p=0.3805 -> 0.378` uses the wider local
growth profile

```text
initial_growth = (0.2, 5, 0.1, 0.5, 0.5)
max_growth     = (50, 500, 20, 100, 50)
```

The segment below `p=0.378` additionally uses the graph-prime denominator
sharpening: the verifier intersects the carried-`C` expression for `p'` with
the expanded expression obtained by substituting
`C=x1*x2-p^2*x3/6`, after first narrowing the box by interval consequences of
that same graph identity.  This is rigorous because both expressions enclose
the same true `p'` on the constraint graph.  It removes the previous
component-0 fake-denominator obstruction and certifies the two half-steps
`0.378 -> 0.3775 -> 0.377`.

The next quarter-step `p=0.377 -> 0.37675` has not been certified by the
current rectangular p-time boxes.  A bounded diagnostic with the graph-prime
sharpening, `x3` RHS subdivision, and adaptive `x3` splitting on component-0
failures reached the processed-box cap with 978 certified leaves and 706 queued
boxes.  The unresolved worst faces are again component-0 lower faces, so the
current obstruction is renewed axis-aligned branching rather than evidence that
the limiting tail is approaching terminal closure.

An alternate narrow-chain diagnostic now starts directly from the sampled
`b in [-1/A,1/A]` family at `p=0.65` and carries `C` in a single
sample-centered p-time tube.  With the default robust profile list, the command

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --sampled-carried-c-p-tube-check \
  --sampled-carried-c-p-tube-end 0.325 \
  --sampled-carried-c-p-tube-progress-every 50 \
  --json > output/s7_tail_proof/sampled_carried_c_p_tube_0.65_to_0.325.json \
  2> output/s7_tail_proof/sampled_carried_c_p_tube_0.65_to_0.325.log
```

certifies the conditional bridge

```text
p = 0.65 -> 0.325
blocks = 650
worst margin ~= 7.67e-6
end t in [3.4036, 3.6542]
end x1 in [3.5352, 10.8726]
end x2 in [0.006607, 0.013096]
end x3 in [-1.0377, 0.01771]
```

This bypasses the wide finite-union split tree, but it is still conditional on
the explicit sampled start box at `p=0.65` being validated from the compact
Taylor-side proof.  Tighter staged profiles give a much sharper conditional
box down to `p=0.3255`,

```text
t in [3.4274, 3.5981]
x1 in [4.5945, 9.0975]
x2 in [0.007835, 0.012076]
x3 in [-0.7874, -0.0851]
```

and the existing automatic 4D p-corridor can continue that sharper box to about
`p=0.276` with finer subdivisions before another small component-0 face
failure.

There is now a carried-`C` automatic p-corridor:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --automatic-carried-c-p-corridor-check \
  --carried-c-p-corridor-end 0.32
```

With the default broad sampled source box above and the tuned fifth-coordinate
safety `(0.002,0.005,0.00005,0.0005,4.0)`, it certifies `p=0.325 -> 0.320`.
That broad box is not a terminal-quality handoff because it still has
`x3_high > 0`.  Starting instead from the sharper staged-profile diagnostic box
at `p=0.3255`, the same carried-`C` automatic corridor certifies to
`p=0.298`, with

```text
x2_low ~= 0.004116
x3_high ~= -0.1017
```

and then fails on a tiny component-0 lower-face balance in the next half-step.
This is good evidence that carrying `C` is the right corridor variable.  At
that point the next useful proof object was a staged sampled carried-`C`
command with validated restarts, rather than more brute-force subdivision of
the old union boxes.

That missing sharper source is now command-reproducible.  The tight profile set
requires a larger late-block tuning budget than the CLI default:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --sampled-carried-c-p-tube-check \
  --sampled-carried-c-p-tube-profile-set tight \
  --sampled-carried-c-p-tube-end 0.3255 \
  --sampled-carried-c-p-tube-progress-every 50 \
  --tuned-p-tube-max-attempts 120 \
  --json > output/s7_tail_proof/sampled_carried_c_p_tube_0.65_to_0.3255_tight_attempts120.json \
  2> output/s7_tail_proof/sampled_carried_c_p_tube_0.65_to_0.3255_tight_attempts120.log
```

It certifies

```text
p = 0.65 -> 0.3255
blocks = 649
tuning attempts = 42335
worst margin ~= 1.97e-6
end x2 in [0.007835, 0.012075]
end x3 in [-0.7874, -0.08505]
```

Feeding that JSON into the carried-`C` automatic corridor gives a reproducible
two-stage continuation:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --automatic-carried-c-p-corridor-check \
  --carried-c-p-corridor-source-json output/s7_tail_proof/sampled_carried_c_p_tube_0.65_to_0.3255_tight_attempts120.json \
  --carried-c-p-corridor-start 0.3255 \
  --carried-c-p-corridor-end 0.298 \
  --json > output/s7_tail_proof/automatic_carried_c_p_corridor_0.3255_to_0.298_from_tight_sampled.json

.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --automatic-carried-c-p-corridor-check \
  --carried-c-p-corridor-source-json output/s7_tail_proof/automatic_carried_c_p_corridor_0.3255_to_0.298_from_tight_sampled.json \
  --carried-c-p-corridor-start 0.298 \
  --carried-c-p-corridor-end 0.29 \
  --carried-c-p-corridor-step 0.0001 \
  --carried-c-p-corridor-safety 0.01,0.005,0.00005,0.0005,4.0 \
  --json > output/s7_tail_proof/automatic_carried_c_p_corridor_0.298_to_0.29_from_tight_sampled.json
```

The second stage certifies

```text
p = 0.298 -> 0.29
worst margin ~= 1.10e-2
end x2_low ~= 0.001558
end x3_high ~= -0.09599
```

From this `p=0.29` box the new carried-`C` p-wall checker certifies that
`x2=0` is an inward lower wall down to `p=0.005`:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --carried-c-p-wall-check \
  --carried-c-p-wall-component 2 \
  --carried-c-p-wall-side lower \
  --carried-c-p-wall-value 0 \
  --carried-c-p-wall-end 0.005 \
  --carried-c-p-wall-source-json output/s7_tail_proof/automatic_carried_c_p_corridor_0.298_to_0.29_from_tight_sampled.json \
  --json > output/s7_tail_proof/carried_c_p_wall_x2_zero_0.29_to_0.005.json
```

The wall margin is small but positive:

```text
status = certified_conditional
p = 0.29 -> 0.005
worst margin ~= 3.16e-8
```

The corresponding `x3=0` wall is not yet certified all the way down in the same
box: direct carried-`C` interval evaluation fails near `p=0.1`.  So the current
tail proof has a rigorous positive-`x2` floor almost to the singular end, but
still needs either a sharper `x3=0` wall argument, an analytic tiny-p closure,
or a better late corridor for the remaining `x3` obstruction.

The last p-time `x2` denominator failure is artificial.  On the ordinary-time
wall `x2=0`, the exact finite-`A` equation factors as

```text
x2' = t^3 p^3 / 216 * (1 + 6 b x3 / p^2)^3,   b = 1/a.
```

Thus for `A=100000000`, `x3 >= -1.51623`, and `p >= 3.5e-4`, the factor is
bounded below by about `0.257`.  The command

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --x2-zero-factor-check \
  --json > output/s7_tail_proof/x2_zero_factor_0.29_to_0.00035.json
```

records this as

```text
status = certified
p = 3.5e-4 -> 0.29
factor lower bound ~= 0.25736
x2' lower bound on the wall ~= 1.38e-13
```

This gives a clean pen-and-paper/computer-certified explanation for the
positive `x2` wall down to the finite-`A` scale where the factor could change
sign.  It does not by itself close the proof, because the remaining obstruction
is now the `x3=0` wall and the need to keep `x1` above `216/t^4` in a proven
late box.  Numerically the nominal trajectories have `x1` increasing to about
`9.08`, so this is a correlation/enclosure problem rather than evidence for a
terminal closure.

The chain is still conditional on the earlier source boxes, and it is still far
from the existing terminal-takeover region, but it removes the first major
obstruction after `p=0.39`.

## Piecewise Barrier Corridor

The next proof attempt is a corridor rather than a trajectory tube.  Starting
from the certified `p=0.25` box, use affine barriers in `p` for
`(t,x1,x2,x3)` and check the p-time vector field on each face.  For decreasing
`p`, a lower face `L(p)` is inward when

```text
G_j(p,y) <= L_j'(p),
```

and an upper face `U(p)` is inward when

```text
U_j'(p) <= G_j(p,y),
```

where `G=dy/dp`.  This is the same face logic as the p-time tube, but the
barriers are chosen by hand rather than centered on nominal samples.

A near-certifying first corridor from `p=0.25` to `p=0.20` uses the rough
slopes

```text
x1 lower slope ~= 30
x2 lower slope ~= 0.025075
x3 lower slope ~= 12
t upper slope  ~= -12
x1 upper slope ~= -150
x2 upper slope ~= -0.12
```

with starting lower values around

```text
x1 >= 6.7, x2 >= 0.0055, x3 >= -0.75.
```

The command implements this corridor as `affine_p_corridor_certificate`.  With
`A=100000000`, step size `0.0005`, p-subdivisions `2`, and state subdivisions
`(2,2,2,2)`, it certifies down to `p=0.2175`.  The first failing face is the
lower `x2` wall on `[0.217,0.2175]`, with margin about `-1.4e-5`.
Increasing the lower `x2` slope fixes that face but moves the bottleneck to
the upper `x1` wall nearby.  Thus the remaining problem is now a two-face
balancing issue for a piecewise barrier corridor, not the original singular
terminal blow-up.

The `--p-corridor-tune` mode scans the two active slopes in this balance.  A
small run over `x2` lower slopes `(0.02505, 0.025075, 0.0251)` and `x1` upper
slopes `(-150, -151)` confirms the tradeoff: increasing the lower `x2` slope
pushes the certified frontier from about `p=0.2175` to about `p=0.217`, but the
failing face switches to the upper `x1` wall with a much larger negative
margin.  This suggests the next successful corridor should be genuinely
piecewise, resetting the upper `x1` ceiling and the lower `x2` floor after the
first near-bottleneck, rather than using one affine corridor over the full
`0.25 -> 0.20` interval.

There is now also a separate terminal-takeover check.  It verifies a
conditional terminal wall:
assuming the late trajectory remains in the coarse box

```text
t  in [3.5, 4.0]
x1 in [2.0, 30.0]
x2 in [0.001, 0.03]
x3 in [-4.0, -0.5],
```

the wall `x3=-0.5` is inward for every `|a| >= 100000000` on
`0.001 <= p <= 0.25`.  The default interval check gives worst p-time face
margin

```text
dx3/dp on x3=-0.5 >= 1.53e-5.
```

For the final tiny layer `0 < p <= 0.001`, the command uses explicit
coefficient inequalities on the same wall.  With `sigma=0.5`, `b=1/a`, and the
coarse box above, the dominant singular terms give

```text
p'  <= -1.8744599999e-4 / p^4 < 0,
x3' <= -1.749997e-3 / p^3 + 0.288 < 0.
```

Thus the terminal wall mechanism itself is now certified down to `p=0`,
conditional on preserving the coarse late box and the positive `x2` floor.

The first part of that missing piece is now certified.  Starting from the
previous p-tube frontier at `p=0.25`, the command

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --frontier-continuation-check \
  --tube-a 100000000
```

continues the conditional p-tube to

```text
p = 0.23595
worst face margin ~= 2.10e-4
```

with end box

```text
t  in [3.54370577, 3.60193577]
x1 in [6.54286432, 8.38936432]
x2 in [0.00277211, 0.01320461]
x3 in [-0.84081364, -0.62259268].
```

From that box, the draft piecewise affine corridor certifies three more
segments while keeping the upper wall `x3 <= -0.6`, down to

```text
p = 0.20995.
```

At that point the stronger terminal wall can take over.  The composed command

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --late-tail-closure-check \
  --tube-a 100000000
```

certifies the whole late tail from the `p=0.25` frontier to terminal:

```text
status = certified_conditional
A = 100000000
frontier continuation: p=0.25 -> 0.23595
piecewise corridor:    p=0.23595 -> 0.20995
terminal wall:         x3=-0.6 down to p=0
terminal wall margin:  ~= 1.59e-6
```

This is still conditional: it starts from the older narrow `p=0.25` frontier
box.  The broader route below is the current path toward a full large-tail
certificate.

There is now a second, independent handoff certificate aimed at that gap.  The
command

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --hybrid-handoff-check \
  --tube-a 100000000
```

uses a narrow p-time tube from `p=0.325` to `p=0.272`, then switches to a
single affine corridor from `p=0.272` to `p=0.25`.  With the current tuned
frontier it reports

```text
status = certified_conditional
p_tube:          p=0.325 -> 0.272
affine corridor: p=0.272 -> 0.25
frontier:
  t  in [3.545, 3.60]
  x1 in [6.35, 8.30]
  x2 in [0.0052, 0.012]
  x3 in [-0.78, -0.58]
corridor margin ~= 1.60e-2
```

This broader frontier now has its own stable tail certificate.  The command

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --broad-tail-closure-check \
  --tube-a 100000000
```

composes:

```text
hybrid handoff:       p=0.325 -> 0.25
automatic p-corridor: p=0.25  -> 0.212
terminal wall:        p=0.212 -> 0
```

The automatic p-corridor greedily proposes each affine face slope from interval
RHS bounds at the current box and then verifies that step with the ordinary
affine face checker.  The default certificate uses `760` p-steps of size
`5e-5` and reports

```text
status = certified_conditional
A = 100000000
automatic corridor margin ~= 2.37e-5
terminal wall margin      ~= 9.46e-6
conditional = p_0_325_start_slice_box_contains_true_state
```

Thus the broad tail from the `p=0.325` slice to terminal is now closed.  The
support-to-slice bridge is also certified.  The command

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --support-tail-closure-check \
  --tube-a 100000000
```

first certifies a t-time moving tube from the support box at `t=3.5` through the
slab `[3.5055,3.5056]`.  The slab has `p` above `0.325` on the left face and
below `0.325` on the right face, and the whole crossing slab is contained in
the enlarged `p=0.325` start box used by the hybrid handoff.  The composed
result is

```text
status = certified_conditional
A = 100000000
bridge:              t=3.5 -> p=0.325
hybrid handoff:      p=0.325 -> 0.25
automatic corridor:  p=0.25  -> 0.212
terminal wall:       p=0.212 -> 0
terminal wall margin ~= 9.46e-6
conditional = support_start_box_contains_true_state
```

The remaining proof obligation for a full large-`|a|` exclusion is now earlier:
certify that the original scaled finite-`a` IVP reaches the support start box at
`t=3.5` for every `|a| >= 100000000`.

The singular-end Taylor seed has now been made explicit.  With `b=1/a`, the
smooth scaled solution has

```text
x(t,b) = (1, 27/4, -1/27, 3) + c2(b) t^2 + O(t^4),
```

where

```text
c2_0 = -5/96  + (27/2) b^2
c2_1 = -27/128 - (729/8) b^2
c2_2 =  5/432 + b/4
c2_3 = -23/64 - (27/8) b + (81/4) b^2.
```

These coefficients come from matching the t-linear terms after the
regular-singular cancellations in the scaled ODE.  The command

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-start-block-check \
  --tube-a 100000000
```

builds the `t=0.001` start box from this Taylor polynomial with radius
`1e-8` in each component, then certifies the first ordinary-time slab:

```text
status = certified_conditional
A = 100000000
t = 0.001 -> 0.00105
worst margin ~= 2.00e-4
conditional = taylor_remainder_is_inside_radius
```

Thus the remaining singular-end task has been localized to a Taylor remainder
bound at `t=0.001`, followed by a shaped validated integration from `0.00105`
to the support box.

The staged Taylor bridge now starts from that same Taylor box instead of the
uncorrected numerical epsilon state used by quick diagnostics:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-time-bridge-check \
  --taylor-bridge-end 0.01 \
  --tube-a 100000000
```

The short version certifies

```text
status = certified_conditional
A = 100000000
t = 0.001 -> 0.01
blocks = 100
worst margin ~= 4.41e-7
conditional = taylor_remainder_is_inside_radius
```

The staged profile has since been tightened.  A progress-audited run with the
same interval face checks reaches `t=2.0` with a much narrower box:

```text
t = 0.001 -> 2.0
end-width ~= (0.0196, 0.957, 0.00185, 0.132)
worst margin ~= 1.04e-8
```

Starting from a radius large enough to contain that `t=2.0` box, the direct
tuned continuation reaches about

```text
t = 2.0 -> 2.603
width at t=2.6 ~= (0.189, 9.66, 0.0137, 1.68)
failure just after t=2.603: lower x2 face
```

This moves the rectangular proof frontier much later, but still not to the
support box.  The true parameter-family spread is tiny by comparison, so the
remaining compact-bridge proof probably still needs to keep correlations, for
example with a shaped/variational tube, before trying to enter the late
`t=3.021`/`t=3.5` support certificates.

The restart-chain architecture can now also be composed directly with the
Taylor bridge:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-restart-chain-check \
  --taylor-bridge-end 0.01 \
  --taylor-restart-end 0.012 \
  --taylor-restart-interval 0.001 \
  --tube-step 0.001 \
  --tube-a 100000000
```

The displayed smoke command certifies

```text
status = certified_conditional
A = 100000000
t = 0.001 -> 0.012
segments = 2
blocks = 2
worst margin ~= 3.48e-4
conditional = taylor_remainder_is_inside_radius
```

This is mostly a bookkeeping/proof-object improvement: the long restart-chain
diagnostic is no longer a separate experiment starting from an unrelated fresh
box. The cached composed mode now certifies through `t=2.6` and then fails just
after `t=2.604` on the longer `2.75` target, because the carried axis-aligned
box has become too wide. Thus the proof gap is no longer bookkeeping: it is a
compact-time correlation problem between `t~2.6` and the already-certified
support-tail entry at `t=3.5`.

For longer attempts, the expensive Taylor bridge prefix can be saved once and
then reused:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-time-bridge-check \
  --taylor-bridge-end 2.0 \
  --taylor-progress-every-blocks 250 \
  --json > output/s7_tail_proof/taylor_bridge_2.0.json \
  2> output/s7_tail_proof/taylor_bridge_2.0.log

.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-restart-chain-check \
  --taylor-bridge-end 2.0 \
  --taylor-restart-end 2.6 \
  --taylor-restart-bridge-json output/s7_tail_proof/taylor_bridge_2.0.json \
  --taylor-restart-progress-every-segments 1 \
  --tube-step 0.001 \
  --tube-a 100000000 \
  --json > output/s7_tail_proof/taylor_restart_2.0_to_2.6_retry.json \
  2> output/s7_tail_proof/taylor_restart_2.0_to_2.6_retry.log
```

The progress flag writes human-readable bridge status lines to stderr, so the
redirected stdout file remains valid JSON.  For example, use
`2> output/s7_tail_proof/taylor_bridge_2.0.log` to keep the progress log.

The current cached artifacts give the honest composed frontier:

```text
t = 0.001 -> 2.0:
  status = certified_conditional
  blocks = 2130
  attempts = 73497
  worst margin ~= 1.04e-8
  end width ~= (0.01956, 0.95412, 0.0018445, 0.13171)

t = 2.0 -> 2.3, using the saved bridge:
  status = certified_conditional
  segments = 6
  attempts = 15202
  worst margin ~= 8.25e-7
  end width ~= (0.04513, 2.6800, 0.003877, 0.39077)

t = 2.0 -> 2.6, using the saved bridge:
  status = certified_conditional
  segments = 12
  blocks = 600
  retry count = 34
  worst margin ~= 8.25e-7
  end width ~= (0.29445, 16.87445, 0.01867, 2.35201)

t = 2.0 -> 2.75, using the saved bridge:
  status = failed
  certified_until ~= 2.604
  retry count = 38
  current width ~= (0.39058, 23.21721, 0.02203, 2.59976)
  failing face = lower x0
  margin ~= -2.96e-3
```

Using a smaller restart interval `0.025` did not improve the pre-retry
frontier, and retry subdivisions now show that the earlier lower-`x2` failure
was an enclosure-shape issue.  The current rigorous composed bridge reaches
farther than the Taylor-only prefix, but the axis-aligned restart boxes still
lose too much correlation well before the late support box.  The next proof
improvement should be a shaped/variational tube or a component inequality for
the broad lower-`x0`/lower-`x2` faces after `t~2.6`.

There is also now an ordinary-time automatic affine-corridor checker for the
regular part of that remaining bridge:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --regular-time-corridor-check \
  --regular-time-start 0.5 \
  --regular-time-end 0.6 \
  --regular-time-step 0.001 \
  --tube-a 100000000
```

For the default component safety `(0.005, 0.05, 0.0005, 0.005)`, this certifies
the conditional t-time corridor

```text
status = certified
A = 100000000
t = 0.5 -> 0.6
steps = 100
worst margin ~= 4.72e-4
conditional = start_box_contains_true_state
```

This proves the regular-time checker can produce genuine interval barriers, but
it does not yet close the whole `0.001 -> 3.5` bridge.  A scalar Gronwall audit
was also tried: the naive row-sum Lipschitz constant on a modest tube near
`t=3.5` is already about `2.3e3`, so a one-number continuous-dependence bound is
far too crude.  The remaining compact-time bridge needs a centered/shaped
validated integrator or a sharper componentwise differential inequality.

The centered moving-tube checker now also has an automatic local profile tuner.
It uses the same face inequalities as the hand-profile checker, but when a lower
face fails it grows only that lower component radius, and when an upper face
fails it grows only that upper component radius.  The reproducible command

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --tuned-tube-check \
  --tube-a 100000000 \
  --tube-start 2.0 \
  --tube-end 2.75 \
  --tube-step 0.001 \
  --tube-block-steps 1 \
  --tuned-tube-max-attempts 60 \
  --tuned-tube-max-growth 0.5,5.0,0.5,0.5
```

certifies

```text
status = certified
A = 100000000
t = 2.0 -> 2.75
blocks = 750
tuning attempts = 17908
worst margin ~= 3.77e-7
conditional = initial_start_box_contains_true_state
```

Starting fresh at `t=2.75`, the same tuned checker certifies `2.75 -> 3.0` and
then reaches about `3.245` before failing.  Thus the current obstruction is not
a local failure of the ODE barriers after `2.75`; it is accumulated width in the
carried box.  The next proof-building target is a restart/containment lemma:
prove that the carried box at `2.75` is contained in a larger fresh centered box
that can be used as a new support for the later tuned tube.

That restart/containment mechanism is now implemented explicitly.  The command

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --restart-tuned-chain-check \
  --tube-a 100000000 \
  --tube-start 2.0 \
  --tube-end 2.95 \
  --tube-step 0.001 \
  --restart-interval 0.05 \
  --tuned-tube-max-attempts 200 \
  --tuned-tube-max-growth 2,20,20,2
```

certifies each segment, then replaces the carried box by a fresh centered box
around the three nominal `b=-1/A,0,1/A` samples, with radius chosen large enough
to contain the carried box.  It reports

```text
status = failed
A = 100000000
certified_until = 2.936
segments = 18
blocks = 936
tuning attempts = 30347
worst margin ~= -3.80e-4
failing face = lower x2 on [2.936, 2.937]
```

The status is `failed` because the requested endpoint was `2.95`, but the
certificate is still a rigorous conditional bridge from `2.0` to `2.936`.
The remaining regular-time gap is therefore `2.936 -> 3.5`, plus the earlier
bridges into `t=2.0`.

A stronger growth-cap audit pushes this same restart-chain architecture a bit
farther:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --restart-tuned-chain-check \
  --tube-a 100000000 \
  --tube-start 2.0 \
  --tube-end 3.1 \
  --tube-step 0.001 \
  --restart-interval 0.05 \
  --tuned-tube-max-attempts 450 \
  --tuned-tube-max-growth 20,200,50,10
```

This reports

```text
status = failed
certified_until = 3.021
segments = 20
blocks = 1021
tuning attempts = 37759
worst margin ~= -9.03e-4
failing face = lower x1 on [3.021, 3.022]
```

The stronger run shows that the restart-chain proof can cross `t=3.0`, but the
boxes are now wide enough that continuing by simply raising growth caps is not a
good proof strategy.  The next late-regular bridge should use a sharper
component inequality, most likely for the coupled `x1,x2,x3` lower faces, or a
changed coordinate before attempting `3.021 -> 3.5`.

## Candidate Explicit Tail

The finite scaled equations have an exact perturbation expansion

```text
x' = F_infinity(t,x) + b R1(t,x) + b^2 R2(t,x) + b^3 R3(t,x),
b = 1/a.
```

The command verifies this identity against the exact scaled equations.  Along
the limiting support samples up to `t0=3.5`, the componentwise sampled error
bound for `A=10000` is

```text
|error| <= (0.00484, 0.01289, 0.000099, 0.01061).
```

The downstream certified tail currently uses the more conservative explicit
threshold

```text
Candidate theorem.  No standard K- closure occurs for |a| >= 100000000.
```

The tail part of this candidate theorem is now certified from the support box
to terminal.  To turn the whole statement into a proof, we still need a rigorous
enclosure showing that the finite `|a|>=100000000` scaled trajectories enter
the support box at `t=3.5`, starting from the actual smooth singular-end Taylor
data.

Using `z=x0^5` as the event variable, standard continuous dependence for ODEs
with a transverse event would then give an `A` such that for all `|a| >= A`,
the exact scaled first crossing satisfies

```text
x2(T_a) >= 0.002.
```

But standard `K-` closure requires

```text
f2(T_a) = a^3 x2(T_a) = 0,
```

which contradicts the lower bound.  This proves the tail exclusion.

The remaining hard part is the lemma.  It is now a bounded problem for the
limiting four-dimensional IVP plus a finite-`a` tube estimate up to the late
support slice.  It looks suitable either for interval Taylor integration or for
a sharper differential-inequality argument.  This does not yet prove the finite
interval exclusion.

## Pen-And-Paper Reduction Attempt

The amount of interval-tube machinery above is a sign that the current proof
object is not yet the right one.  A more plausible paper proof should use a
small number of scalar barriers.  The most useful scalar found so far is the
cancellation variable

```text
C = x1*x2 - p^2*x3/6,  p=x0.
```

This is the same combination that controls the singular part of `p'`.  In the
limiting system, direct algebra gives

```text
C' = -4 C/t + 2 x2*x3^3/(t*p^3) - p^3/t + x1*t^3*p^3/108.
```

The more important cancellation occurs on a negative `x3` wall.  Put
`x3=-sigma`, with `sigma>0`.  Since

```text
x1*x2 = C - p^2*sigma/6
```

on that wall, the two `p^2*sigma^2/6` terms cancel in the `x3'` equation.  For
the finite scaled family, with `|b|<=beta`, this gives the upper estimate

```text
x3' <= (2 sigma + 6 p)/t
       - t sigma C/(2 p^3)
       + beta*t/(2 p^3) * (sigma^3 + 2 t^2 Q p^2 sigma/3),
```

provided `x1 <= Q`; the omitted `-t^3*x1*p/36` and `b^2` terms are favorable.
Thus `x3=-sigma` is an inward wall once `C` is large enough relative to `p^3`.

For the first concrete values tried,

```text
A = 100000000, beta = 1/A
sigma = 0.3
t in [3.5,4.0]
p <= 0.34
x1 <= 30
C >= 0.06
```

the scalar bound at the worst endpoint is

```text
(2 sigma + 6 p)/t       <= 0.7542857143
t sigma C/(2 p^3)       >= 0.8014451455
finite-|a| error term   <= 0.0000056608
upper bound for x3'     <= -0.0471537704.
```

The constants improve if the negative wall is placed where the actual support
box already lies.  The certified support box at `t=3.5` has

```text
p_high  = 0.3294454679
x1_low  = 6.4914061822
x2_low  = 0.0102371048
x3_high = -0.3676870810
C_low   = 0.0731043083.
```

Thus it is natural to use the wall

```text
x3 = -0.36,  C >= 1.23 p^3,  p <= 0.33.
```

At `p=0.33`, the same scalar `x3` wall estimate gives

```text
(2 sigma + 6 p)/t       <= 0.7714285714
t sigma K/2             >= 0.7749000000
finite-|a| error term   <= 0.000006... 
upper bound for x3'     <= -0.00346...
```

The lower `C-Kp^3` wall is also promising.  On `C=Kp^3` write
`x3=-u p`.  Since `x2>=0`,

```text
sigma/p <= u <= 6K.
```

The correct parametrization on this wall is

```text
x2 = p^3 (K-u/6) / x1.
```

For the limiting system, direct simplification gives

```text
d(C-Kp^3)/dt
  = p^3 * [(-K - 1)/t
            + x1 t^3/108
            + 3 K^2 t/4
            + ((K-u/6) u^2 (9K-2u))/(x1 t)].
```

For `0 <= u <= 6K`, the final one-variable factor satisfies

```text
(K-u/6) u^2 (9K-2u) >= -5.272 K^4.
```

Hence, on the lower wall,

```text
d(C-Kp^3)/dt
  >= p^3 * [(-K - 1)/t
            + x1 t^3/108
            + 3 K^2 t/4
            - 5.272 K^4/(x1 t)].
```

With `K=1.23`, this bracket is already positive for `x1>=1` and
`t in [3.5,4.0]` in the limiting calculation; at `x1=2` the bracket margin is
about `2.40`.  A finite `b=1e-8` grid sanity check over the parametrized wall
gives a minimum sampled `d(C-Kp^3)/dt` margin about `3.75e-4` for `x1>=2`;
this is not a proof, but it
supports the hand inequality and shows the finite correction is not the
bottleneck.

So a broad terminal proof can replace the precise `p=0.325` handoff if we can
prove the following compact support/barrier lemma:

```text
For every |a| >= 100000000, after the trajectory first enters p<=0.33,
while p>0 and t<=4, one has
  t >= 3.5,
  2 <= x1 <= 30,
  C >= 1.23 p^3,
  x2 remains positive,
  and x3 <= -0.36.
```

Below `p = sigma/(6K) ~= 0.04878`, the lower `C` barrier is automatic from
`x3<=-sigma` and `x2>=0`, because

```text
C = x1*x2 - p^2*x3/6 >= sigma p^2/6 >= K p^3.
```

After that point the existing terminal-takeover inequalities already prove the
final small-`p` exclusion once a negative `x3` wall and a positive `x2` floor
are preserved.

This is not yet a proof: the `C` and `x3` walls now have plausible scalar
inequalities, but the auxiliary coarse lower bound `x1>=2` still needs its own
barrier or a replacement that avoids a separate `x1` floor.  The next analytic
task is therefore sharper and much smaller than the earlier box-search problem:

```text
Prove a coupled late-region barrier for
  x3 <= -0.36,
  C >= 1.23 p^3,
  x1 >= 2,
  x2 >= 0,
starting from the certified t=3.5 support box.
```

A naive rectangular lower wall for `x1` is not expected to work: if `x3` is
allowed to become very negative independently of `x2` and `C`, the `x1'`
equation has a large negative `x3^3/p^3` term.  Thus the remaining lemma should
preserve a correlation, not merely a box.  The promising variables are
`C`, `x2`, and a scaled negative variable such as `u=-x3/p`; on the actual
`C=Kp^3` boundary, `x2>=0` already forces `u<=6K`, exactly the correlation
that made the scalar `C` wall work.

This would leave only a compact support-entry estimate from the Taylor start to
the broad `t=3.5`, `p<=0.34`, `C>=0.06`, `x3<=-0.3` region.  That compact part
may still need validated numerics, but the terminal exclusion would then be a
short scalar inequality rather than a large collection of fitted p-time boxes.
