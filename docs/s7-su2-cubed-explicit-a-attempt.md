# S7 SU(2)^3 Explicit-A Attempt

Date: 2026-07-05.

Goal: find a concrete threshold `A` such that the Podesta `SU(2)^3`
one-parameter tail has no standard compact `K-` closure for every
`|a| >= A`.

The current conservative candidate remains

```text
A = 100000000.
```

No complete explicit-`A` proof is closed yet.  This note records the latest
successful certificate repairs and the remaining obstruction.

## Current Sharp Status

The downstream part is now better than the older summary below suggested.
The command

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --support-tail-closure-check \
  --tube-a 100000000 \
  --json
```

returns `support_tail_closure_certificate.status =
certified_conditional`.  It proves, for `A=1e8`, that the trajectory cannot
close compactly once the following tiny support box at `t=3.5` has been
validated:

```text
p  in [0.3294452679037324, 0.3294454679037324]
x1 in [6.491406182277336, 6.491408182277336]
x2 in [0.010237108127091615, 0.010237128127091614]
x3 in [-0.36768767855941524, -0.36768747855941526]
```

More explicitly, the conditional chain is:

```text
t=3.5 support box
  -> p=0.325 start slice between t=3.5055 and t=3.5056
  -> p=0.25 frontier box
  -> p=0.212 affine corridor
  -> terminal x3=-0.6 wall down to p=0.
```

The terminal wall stage has positive margin about `9.459351615812532e-06`.
So the remaining explicit-`A` gap is not the terminal tail anymore.  It is the
upstream support-entry statement: prove that the actual smooth singular-end
solution for every `|a| >= 1e8` lies in the displayed support box at `t=3.5`.

The short support-to-`p=0.325` bridge also tolerates a larger support box.  The
new reproducible command

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --support-tail-closure-check \
  --tube-a 100000000 \
  --support-tail-support-radius 1e-6,1e-5,1e-7,1e-6 \
  --json
```

uses a 10x larger support radius.  The support bridge still has
`before_above_target=True`, `after_below_target=True`, and
`crossing_slab_contained_in_start_slice=True`.  At 20x, the same fixed
`after_time=3.5056` no longer has the whole final p-interval below `0.325`, so
10x is the current clean enlargement.

## Empirical Defect Threshold

The scalar defect

```text
D_x3(a) = x3(T_a)
```

is already numerically negative soon after the known compact parameters.
With Taylor-seeded integration and step size `5e-5`:

| a | x2(T) | x3(T) | C(T) |
|---:|---:|---:|---:|
| -40 | 6.605e-4 | -9.978e-2 | 2.922e-3 |
| -50 | 2.201e-3 | -2.686e-1 | 1.460e-2 |
| -100 | 4.668e-3 | -6.597e-1 | 4.008e-2 |
| 40 | -3.805e-2 | -2.726 | -1.683e-1 |
| 100 | 2.978e-3 | -1.851 | 2.556e-2 |

This is only numerical evidence.  It suggests that a small empirical
threshold may exist, but the proof attempt below still targets the much safer
`A=1e8`.

## High-Order Left Germ

The second-order scaled Taylor seed is not enough for a clean compact bridge.
The code now contains a reproducible coefficient recurrence for the regular equation

```text
t x' = G(t, x, b),  b = 1/a,
```

in `experiments.s7.su2_cubed_tail_defect.scaled_taylor_coefficients`.
Degree 40 gives the following fixed-`p` slice at `p=0.65`:

```text
b=-1e-8:
  (t, x1, x2, x3) =
  (2.640415759530347, 5.679207694818107,
   0.008333592337988095, 0.9753289660446369)

b=0:
  (2.6404157595303, 5.679207694818237,
   0.008333599701946177, 0.9753287680733078)

b=+1e-8:
  (2.640415759530347, 5.679207694818107,
   0.008333607065905682, 0.975328570101909)
```

This suggests a better route than the old rectangular bridge to `t=3.5`:
use a high-order singular Taylor germ to validate a start slice near
`p=0.65`, then switch to carried-`C` p-time certificates.

The current reproducible convergence audit is:

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-audit \
  --tube-a 100000000
```

For the default `p=0.65` slice, comparing Taylor orders `30 -> 40` gives

```text
max_diff =
  [1.2370238078318607e-06,
   5.608562593906186e-06,
   2.389283264292441e-08,
   1.5913721895799426e-07,
   1.0015860676254595e-07]
max_diff/radius =
  [0.12370238078318606,
   0.056085625939061856,
   0.02389283264292441,
   0.015913721895799426,
   0.010015860676254595]
```

A stricter follow-up,

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-audit \
  --tube-a 100000000 \
  --taylor-p-slice-low-order 40 \
  --taylor-p-slice-high-order 50
```

shrinks the observed difference to

```text
max_diff =
  [4.1676410766910976e-08,
   1.8195763296091627e-07,
   7.892310624479926e-10,
   7.595904794044372e-09,
   3.500723093352587e-09]
max_diff/radius =
  [0.004167641076691098,
   0.0018195763296091627,
   0.0007892310624479926,
   0.0007595904794044371,
   0.0003500723093352586]
```

So the p-slice route has substantial numerical slack.  This is still not a
rigorous start-slice certificate: the remaining step is to replace the
observed order-to-order convergence by a Taylor remainder bound, or by a
validated high-order series enclosure, uniformly for `|b| <= 1e-8`.

The follow-up tail-ratio audit is:

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-tail-audit \
  --tube-a 100000000
```

With `order=60`, `tail_start=50`, and `ratio_start=45`, the parity-aware
formal geometric tail estimate gives

```text
max_tail/radius =
  [4.083807511496553e-05,
   5.930577694693053e-05,
   8.384828030500245e-06,
   0.0001589218972495815,
   2.089497834813165e-05]
max observed same-parity ratio =
  [0.5177950863574734,
   0.5148132985405564,
   0.5137396750806918,
   0.5143588595393823]
formal p-event time-shift bound from the p-tail =
  1.55116937e-09
```

This is still a formal/observed estimate, not a proof.  But it says that a
successful same-parity geometric majorant beginning around order 50 would have
far more than enough room to fit inside the existing carried-`C` p-tube start
box.

The more proof-shaped finite check is:

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-tail-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 70 \
  --taylor-p-slice-tail-start 50 \
  --taylor-p-slice-ratio-start 45 \
  --taylor-p-slice-ratio-bound 0.53
```

This treats `q=0.53` as the candidate same-parity geometric majorant ratio.
The observed finite window through order 70 remains below it for
`b=-1e-8,0,1e-8`:

```text
inside_bound = True
max observed same-parity ratio =
  [0.5206861303380034,
   0.5180691582261967,
   0.5172594851593756,
   0.5177061334196337]
max_tail/radius using q=0.53 =
  [4.189855417590606e-05,
   6.122207290001888e-05,
   8.67491318736112e-06,
   0.0001642106624816875,
   2.1591837516724603e-05]
formal p-event time-shift bound from the p-tail =
  1.59144999e-09
```

The same `q=0.53` tail estimate was also checked on a five-point symmetric
`b` grid:

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-tail-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 60 \
  --taylor-p-slice-tail-start 50 \
  --taylor-p-slice-ratio-start 45 \
  --taylor-p-slice-ratio-bound 0.53 \
  --taylor-p-slice-b-samples 5
```

It gives

```text
inside_bound = True
max observed same-parity ratio =
  [0.5177950863574734,
   0.5148132985405564,
   0.5137396750806918,
   0.5143588595393823]
max_tail/radius using q=0.53 =
  [4.189855422157228e-05,
   6.122207296674624e-05,
   8.67491319681611e-06,
   0.00016421066266066457,
   2.159183754068536e-05]
```

Thus a plausible rigorous target is now very concrete: prove that, uniformly
for `|b| <= 1e-8`, the same-parity Taylor tails after order 50 are dominated
by the first omitted term times the geometric ratio `q=0.53`.

The maintained tail-ratio audit now also records where the worst observed
same-parity ratios occur.  In the order-60, three-sample run above, the
componentwise worst witnesses are all at the last checked pair `58 -> 60`:

```text
max observed same-parity ratio =
  [0.5177950863574734,
   0.5148132985405564,
   0.5137396750806918,
   0.5143588595393823]
worst degree witnesses =
  [(b=0,       58 -> 60),
   (b=0,       58 -> 60),
   (b=1e-8,    58 -> 60),
   (b=-1e-8,   58 -> 60)]
```

This is useful but also a warning: the finite ratio evidence has not yet
found a terminal plateau.  A rigorous proof still has to explain the
coefficient recurrence, not just sample it farther out.

### Cauchy-Budget Diagnostic

A second maintained diagnostic now asks a different question: if we try to
prove the Taylor remainder by a standard Cauchy disk bound, how large would
the disk bound have to be?  The command

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-cauchy-budget-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 80 \
  --taylor-p-slice-tail-start 70 \
  --taylor-p-slice-b-samples 3 \
  --taylor-p-slice-cauchy-radii 3.2,3.3,3.4,3.45,3.5 \
  --taylor-p-slice-cauchy-circle-samples 720 \
  --taylor-p-slice-cauchy-circle-tail-ratio-bound 0.95
```

returns

```text
status = observed_cauchy_budget_has_proof_relevant_viable_radius
limiting terminal time reference = 3.5975043
viable radii = [3.3, 3.4, 3.45, 3.5]
proof-relevant viable radii = [3.3, 3.4, 3.45, 3.5]
best radius by observed coefficient floor = 3.5
best max observed Cauchy-floor/radius = 0.00365903334
sampled min |p_N(t)| on |t|=3.5 = 0.331014491
derivative-certified min |p_N(t)| on |t|=3.5 = 0.317648051
formal p-circle tail estimate using q=0.95 = 0.0035984569
formal Rouche margin = 0.314049594
```

Interpretation: at the real `p=0.65` slice the time is about `2.64`, so a
uniform analytic disk out to roughly `|t| <= 3.5`, together with a Cauchy
circle bound not much worse than what the observed coefficients already
force, would leave about a factor `270` of tail-radius slack after order 70.
This is still not a proof.  It is a sharper proof target: prove a uniform
analytic disk bound below the real terminal crossing for `|b| <= 1e-8`, or
replace it by an equivalent coefficient-majorant argument.

The denominator check is also encouraging: for the truncated `p_N` series,
sampling the circle and subtracting the angular derivative loss still leaves
`|p_N| >= 0.3176`.  A Rouche-style proof would still have to add a certified
tail bound for `p - p_N`; with a same-parity circle-tail ratio `q=0.95`,
the formal tail estimate is only `0.00360`, leaving margin about `0.314`.
So the denominator part now has a concrete proof target:

```text
on |t| = 3.5, after degree 70, prove the p-series same-parity tail ratio
is at most 0.95, uniformly for |b| <= 1e-8.
```

The expected real-terminal ratio scale is approximately
`(3.5 / 3.5975)^2 = 0.9465`, so `q=0.95` is a more realistic infinite-tail
target than the tighter finite-window value `q=0.93`.

The previous `R=3.58`, degree-60/tail-50 target used `q=0.96`; an order-80
check showed the p-circle ratio already reaches `0.96121756149084` at
degree `78 -> 80`.  So that target was too tight.  Moving inward to `R=3.5`
and using more Taylor coefficients is a more realistic proof route.

The finite ratio profile for the revised target is now reproducible via

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-ratio-profile-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 80 \
  --taylor-p-slice-ratio-start 70 \
  --taylor-p-slice-b-samples 3 \
  --taylor-ratio-profile-circle-radius 3.5 \
  --taylor-ratio-profile-circle-ratio-bound 0.95 \
  --taylor-ratio-profile-p-slice-ratio-bound 0.53
```

It gives

```text
status = observed_ratios_inside_bounds
max p-circle ratios =
  [0.9187381111905675,
   0.9146409280717309,
   0.9135302349063585,
   0.9141200192478031]
max p-slice ratios =
  [0.522877870387067,
   0.5205460563938639,
   0.5199139318853971,
   0.5202495935681836]
```

A one-sample `b=0` scratch extension to orders 90 and 100 stayed below even
the tighter `q=0.93` p-circle level.  We still use `q=0.95` as the proposed
proof target because the expected real-terminal ratio scale is closer to
`0.9465`:

```text
order 90:  p-circle ratio = 0.9217562722213091 at 88 -> 90
order 100: p-circle ratio = 0.9241845618291651 at 98 -> 100
```

This is still finite-window evidence, not the missing induction or majorant
proof.  It does make the proposed ratio bound less fragile than the discarded
`R=3.58` target.

The ratio evidence was later extended in two directions.  First, the full
finite-`b` three-sample profile was pushed to order 120, checking both the
p-slice target `q <= 0.60` from degree 50 and the proof-circle target
`q <= 0.95` on `|t|=3.5`:

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-ratio-profile-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 120 \
  --taylor-p-slice-ratio-start 50 \
  --taylor-p-slice-b-samples 3 \
  --taylor-p-slice-tail-working-dps 110 \
  --taylor-ratio-profile-circle-radius 3.5 \
  --taylor-ratio-profile-circle-ratio-bound 0.95 \
  --taylor-ratio-profile-p-slice-ratio-bound 0.6
```

It returns

```text
status = observed_ratios_inside_bounds
max p-circle ratios =
  [0.9278489526498046,
   0.924995802583358,
   0.9244778674640812,
   0.9247224293281643]
max p-slice ratios =
  [0.5280630883741984,
   0.5264392861040336,
   0.5261445156913882,
   0.5262837022398857]
```

Second, a new limiting-only mode makes deeper `b=0` ratio probes less
expensive:

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-ratio-profile-audit \
  --taylor-ratio-profile-b-mode limit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 160 \
  --taylor-p-slice-ratio-start 50 \
  --taylor-p-slice-tail-working-dps 140 \
  --taylor-ratio-profile-circle-radius 3.5 \
  --taylor-ratio-profile-circle-ratio-bound 0.95 \
  --taylor-ratio-profile-p-slice-ratio-bound 0.6
```

It gives

```text
status = observed_ratios_inside_bounds
max p-circle ratios =
  [0.9324646858398841,
   0.930275942227144,
   0.9299764429239128,
   0.9301060894941814]
max p-slice ratios =
  [0.5306900227653013,
   0.5294443515722672,
   0.5292738986913553,
   0.5293476839428207]
```

The worst observed ratios are still well below the proposed proof bounds.
They are drifting upward slowly, as expected from the nearby terminal
singularity, but not toward either proposed wall at these orders.

The ratio profile was then converted into a more explicit finite-window
geometric-envelope check.  For each component and parity, the audit anchors
the envelope at the first omitted term after degree 50 and checks whether all
later computed terms stay below

```text
first omitted same-parity term * q^k.
```

In fact, the recurrence has an even-parity simplification.  If the state
series contains only even powers of `t`, then every expression in
`t*x' = G(t,x,b)` also contains only even powers: the explicit `t` factors in
the finite-`b` perturbation terms occur in odd powers before the final
multiplication by `t`, and the limiting terms contain only even powers.  Since
the initial coefficients are even, uniqueness of the triangular coefficient
solve forces all odd Taylor coefficients to vanish for every real or complex
`b`.

The diagnostic

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-even-parity-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 40 \
  --taylor-p-slice-tail-working-dps 80 \
  --taylor-p-slice-b-samples 3 \
  --taylor-b-cauchy-enclosure-radius 4e-7 \
  --taylor-b-cauchy-enclosure-samples 8
```

confirms this on the real `|b| <= 1e-8` samples and on the complex
`|b|=4e-7` event-enclosure circle:

```text
status = observed_odd_coefficients_zero
max odd coefficient abs = [0.0, 0.0, 0.0, 0.0]
```

Thus the actual Taylor-tail majorant is an even-subsequence problem.  The
"same-parity" wording in the diagnostics is conservative; the odd subsequence
is identically zero.

Equivalently, write the germ as an ordinary series in

```text
s = t^2.
```

Then `t*x'(t) = 2*s*dX/ds`, and the coefficient recurrence should be proved as
an ordinary `s`-series majorant, not as a same-parity `t`-series majorant.  This
is a cleaner target because the terminal time satisfies

```text
T_infinity^2 ~= 12.9420372,
```

while the proof circle `|t|=3.5` is the `s`-circle `|s|=12.25`.

The finite three-sample diagnostic

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-even-s-series-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 120 \
  --taylor-p-slice-tail-start 50 \
  --taylor-p-slice-b-samples 3 \
  --taylor-p-slice-tail-working-dps 110 \
  --taylor-ratio-profile-circle-radius 3.5 \
  --taylor-ratio-profile-circle-ratio-bound 0.95 \
  --taylor-ratio-profile-p-slice-ratio-bound 0.6
```

returns

```text
status = observed_s_series_inside_targets
tail_start_s = 25
circle_s = 12.25
max circle ratios =
  [0.9278489526498046,
   0.924995802583358,
   0.9244778674640812,
   0.9247224293281643]
max p-slice ratios =
  [0.5280630883741984,
   0.5264392861040336,
   0.5261445156913881,
   0.5262837022398856]
min inferred circle_s =
  [13.202579972759297,
   13.243303338012785,
   13.25072284705177,
   13.247218420883286]
terminal_s = 12.9420372
```

The deeper limiting-only check

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-even-s-series-audit \
  --taylor-ratio-profile-b-mode limit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 160 \
  --taylor-p-slice-tail-start 50 \
  --taylor-p-slice-tail-working-dps 140 \
  --taylor-ratio-profile-circle-radius 3.5 \
  --taylor-ratio-profile-circle-ratio-bound 0.95 \
  --taylor-ratio-profile-p-slice-ratio-bound 0.6
```

returns

```text
status = observed_s_series_inside_targets
tail_start_s = 25
circle_s = 12.25
max circle ratios =
  [0.9324646858398841,
   0.930275942227144,
   0.9299764429239128,
   0.9301060894941814]
max p-slice ratios =
  [0.5306900227653013,
   0.5294443515722671,
   0.5292738986913552,
   0.5293476839428206]
min inferred circle_s =
  [13.137226734722134,
   13.16813586587294,
   13.1723766695478,
   13.17054058495833]
terminal_s = 12.9420372
```

Thus the observed `s`-series coefficient ratios imply a nearest-singularity
scale beyond the limiting terminal value, with a small but visible margin.  The
next proof target is to justify an `s`-series geometric majorant by the
coefficient recurrence, uniformly for the finite `|b| <= 1e-8` tube.

The triangular coefficient solve has a helpful closed-form linear part.  At
`t`-degree `d` the new coefficient vector

```text
y_d = (p_d, x1_d, x2_d, x3_d)
```

is solved from

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
```

for the limiting and finite-`b` recurrences.  The determinant factors as

```text
det(M_d) = d*(d+1)*(d+4)*(d+6).
```

Moreover, writing `R_d=(R0,R1,R2,R3)`,

```text
x2_d = R2/d,
u0   = R0 - 27*R2/d,
D    = (d+1)*(d+6),
p_d  = ((d+2)*u0 + (2/3)*R3)/D,
x3_d = (6*u0 + (d+5)*R3)/D,
x1_d = R1/(d+4) + (-81*d*u0 + 27*(d+3)*R3)/(D*(d+4)).
```

So there is no hidden high-degree resonance.  The remaining induction should
bound the lower-order convolution vector `R_d` under a proposed ordinary
`s`-series envelope and then feed it through this explicit inverse.  The
finite-`b` perturbation terms do not alter `M_d`; they only add lower-order
forcing terms carrying powers `b,b^2,b^3`.

The corresponding diagnostic is:

```text
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

It returns

```text
status = observed_recurrence_forcing_inside_targets
max reconstruction error =
  [7.670458539527698e-93,
   6.136366831622158e-92,
   0.0,
   1.9176146348819244e-93]
max inverse-bound usage =
  [1.0,
   0.8233120741010939,
   1.0000000000000002,
   1.0]
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

This is not yet the symbolic induction, but it is a sharper target than the
raw coefficient-ratio check: the lower-order forcing itself appears to satisfy
the same proof-circle `q=0.95` envelope.  The remaining missing inequality is
therefore a convolution-majorant statement of the following form:

```text
if all previous ordinary s-coefficients satisfy the proposed envelope,
then the rational lower-order forcing R_d satisfies the displayed forcing
envelope, and the explicit inverse above returns y_d inside the solution
envelope.
```

For the full finite-`b` three-sample window,

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-geometric-envelope-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 120 \
  --taylor-p-slice-tail-start 50 \
  --taylor-p-slice-b-samples 3 \
  --taylor-p-slice-tail-working-dps 110 \
  --taylor-ratio-profile-circle-radius 3.5 \
  --taylor-ratio-profile-circle-ratio-bound 0.95 \
  --taylor-ratio-profile-p-slice-ratio-bound 0.6
```

returns

```text
status = observed_terms_inside_geometric_envelopes
max strict proof-circle envelope usage =
  [0.9535821446214192,
   0.9475626945495688,
   0.9451614573972486,
   0.946590485148854]
max proof-circle tail-sum usage =
  [0.5544617498433901,
   0.5286225288285096,
   0.520475724191603,
   0.5250623253912419]
max strict p-slice envelope usage =
  [0.859288490735743,
   0.8538642656739858,
   0.851700471542432,
   0.8529881918576904]
max p-slice tail-sum usage =
  [0.8270710967339429,
   0.8216894629830235,
   0.8196300603400131,
   0.8208426395421468]
```

The deeper limiting-only order-160 envelope check,

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-geometric-envelope-audit \
  --taylor-ratio-profile-b-mode limit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 160 \
  --taylor-p-slice-tail-start 50 \
  --taylor-p-slice-tail-working-dps 140 \
  --taylor-ratio-profile-circle-radius 3.5 \
  --taylor-ratio-profile-circle-ratio-bound 0.95 \
  --taylor-ratio-profile-p-slice-ratio-bound 0.6
```

also stays inside:

```text
max strict proof-circle envelope usage =
  [0.9535821446214192,
   0.9475626945495688,
   0.945161456715412,
   0.9465904849491621]
max proof-circle tail-sum usage =
  [0.5837178550930738,
   0.5533003823394625,
   0.5440755909948812,
   0.5492309077669907]
max strict p-slice envelope usage =
  [0.859288490735743,
   0.8538642656739858,
   0.8517004709279868,
   0.8529881916777136]
max p-slice tail-sum usage =
  [0.8270710968569763,
   0.8216894630885377,
   0.8196300598675172,
   0.8208426394786906]
```

This is still finite evidence, but it is closer to the desired proof shape:
the remaining Taylor-tail task is now to justify this same-parity geometric
envelope by induction from the coefficient recurrence.

### Finite-b Sensitivity Diagnostic

The revised Cauchy/ratio target is useful only if the finite family
`|b|=|1/a| <= 1e-8` is genuinely a small perturbation of the limiting
`b=0` Taylor germ.  The maintained command

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-b-sensitivity-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 80 \
  --taylor-p-slice-ratio-start 70 \
  --taylor-p-slice-b-samples 3 \
  --taylor-b-sensitivity-circle-radius 3.5
```

compares the endpoints `b=-1e-8,0,1e-8` against the limiting germ.  It gives

```text
status = finite_b_state_delta_inside_start_radius
max state delta / p-tube start radius =
  [4.796163466380676e-09,
   1.305622276959184e-09,
   0.007363959345085824,
   0.019797140593524887,
   0.0055761941376852855]
max circle coefficient l1 delta / limiting l1 =
  [4.2384227794746856e-14,
   9.086846001367982e-14,
   2.3773649148754175e-07,
   7.190388669197883e-08]
max circle tail coefficient l1 delta / limiting tail l1 =
  [2.534977484755207e-12,
   2.55986851995067e-12,
   7.451686595755823e-07,
   2.5259289070973173e-07]
```

Interpretation: at `A=1e8`, the finite-`b` motion of the carried-C p-slice is
well inside the downstream start radius; the worst coordinate uses about
`2%` of the radius.  On the proof circle `|t|=3.5`, the high-degree tail
coefficient perturbation is below `1e-6` relative in the worst component.

This makes the finite-`A` part look like a small perturbation problem rather
than a separate instability.  It still has to be upgraded from endpoint
sampling to a uniform interval or analytic coefficient bound over the whole
interval `|b| <= 1e-8`.

The existing interval Taylor coefficient audit is not yet sharp enough for
that upgrade at order 80.  The commands

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-interval-ratio-audit \
  --tube-a 100000000 \
  --taylor-p-slice-interval-order 80 \
  --taylor-p-slice-ratio-start 70 \
  --taylor-p-slice-ratio-bound 0.53 \
  --taylor-p-slice-b-subdivisions 1 \
  --taylor-p-slice-interval-working-dps 90

.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-interval-ratio-audit \
  --tube-a 100000000 \
  --taylor-p-slice-interval-order 80 \
  --taylor-p-slice-ratio-start 70 \
  --taylor-p-slice-ratio-bound 0.53 \
  --taylor-p-slice-b-subdivisions 16 \
  --taylor-p-slice-interval-working-dps 90
```

both fail with

```text
failure = interval Taylor midpoint preconditioner is not contractive
```

The 16-subinterval run already narrows the first checked interval to
`[-1e-8,-8.75e-9]`, so this is probably interval wrapping in the coefficient
solve rather than a real finite-`b` instability.  The next proof attempt
should therefore use either an explicit perturbation recurrence for
`c_n(b)-c_n(0)` or a coefficient-majorant argument with `b` treated as a small
parameter, not the current raw interval linear solve.

### Complex-b Cauchy Perturbation Diagnostic

The current replacement for the failed raw interval recurrence is a sampled
Cauchy estimate in the small parameter `b`.  For each Taylor coefficient
`c_n(b)`, put

```text
g_n(b) = c_n(b) - c_n(0).
```

If `g_n` is certified on a complex circle `|b|=B`, then the
maximum-modulus principle applied to `g_n(b)/b` gives

```text
|g_n(b)| <= (|b|/B) max_{|z|=B} |g_n(z)|
```

for `|b| <= 1/A`.  This avoids the interval linear-solve wrapping that made
the direct interval Taylor audit fail.

The original tiny support box at `t=3.5` is too tight for this finite-`b`
perturbation: at order 40, the direct endpoint comparison already uses about
`3.89` times the original `x3` radius.  This is not fatal, because the
downstream support-tail certificate had already been checked with the 10x
larger support radius

```text
(1e-6, 1e-5, 1e-7, 1e-6).
```

Against that 10x support box, the order-80 finite-degree Cauchy-in-`b`
diagnostic is:

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-b-cauchy-coefficient-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 80 \
  --taylor-p-slice-tail-working-dps 90 \
  --taylor-b-cauchy-time-radius 3.5 \
  --taylor-b-cauchy-radius 1e-7 \
  --taylor-b-cauchy-samples 4 \
  --support-tail-support-radius 1e-6,1e-5,1e-7,1e-6 \
  --taylor-b-cauchy-skip-direct
```

It returns

```text
status = sampled_b_cauchy_delta_inside_support_radius
cauchy_delta/radius =
  [7.591467369217093e-07,
   1.0576271155819335e-06,
   0.9098173287302712,
   0.712791426602341]
```

The corresponding lower-order checks are stable:

```text
order 40:
  direct_delta/radius =
    [6.272760089132134e-08,
     8.650857807879218e-08,
     0.03450746591249221,
     0.3893462796611402]
  cauchy_delta/radius =
    [6.27785312184339e-07,
     8.655591922930218e-07,
     0.9087560181822663,
     0.7058862219565238]

order 60:
  direct_delta/radius =
    [7.127631818093505e-08,
     9.920952948050398e-08,
     0.033727106732439616,
     0.39442627386421236]
  cauchy_delta/radius =
    [7.129825591706308e-07,
     9.92782856892916e-07,
     0.909536393301281,
     0.7109665123403226]
```

This gives a much more concrete explicit-`A` route:

1. use the 10x support box at `t=3.5`;
2. prove the complex-`b` circle maxima used above, replacing the four sampled
   points by a certified circle maximum;
3. combine that finite-degree `b` perturbation with the existing
   `R=3.5`, order-80, tail-after-70 Taylor majorant in the `t` variable;
4. feed the resulting support box into the already certified downstream
   support-tail closure chain.

The worst finite-degree margin is now the `x2` support radius: the sampled
Cauchy-in-`b` bound uses about `91%` of the 10x `x2` radius before adding the
remaining Taylor tail in `t`.  So the next proof step should either certify a
slightly sharper complex-`b` circle maximum, use a larger support radius if
the downstream bridge tolerates it, or reduce the finite-degree perturbation
budget by increasing the b-circle radius/sampling certificate.

An earlier optimistic budget with radii `4.0,4.2,4.4,4.6,4.8` had much more
slack, but those radii pass the limiting real terminal crossing at
`T ~= 3.5975`.  They are therefore coefficient-size diagnostics, not a
legitimate ordinary Cauchy disk route for the compact proof.

### Residual Validation Attempt

I also tested whether a direct a-posteriori analytic validation around the
degree-80 Taylor polynomial might be easy.  The diagnostic

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-circle-residual-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 80 \
  --taylor-p-slice-b-samples 3 \
  --taylor-circle-residual-radius 3.5 \
  --taylor-circle-residual-samples 120
```

samples the defect

```text
t*P'(t) - t*f(t,P(t),b)
```

on `|t|=3.5`.  It returns

```text
status = sampled_residual_not_small
max residual = 1.52196038
max residual by component =
  [0.13441412577410083,
   1.5219603782489015,
   0.0018018406915424933,
   0.45344404520853243]
min sampled |p_N| = 0.331014491
```

So a naive Newton/Kantorovich validation on the circle does not look like the
easy route: the raw equation residual is not tiny in sup norm.  This does not
contradict the coefficient-tail budget, because the residual diagnostic is a
different norm and does not use the triangular Taylor recurrence.  It does
suggest that the next proof attempt should stay with a coefficient recurrence
or majorant argument rather than a black-box analytic residual theorem.

An interval-coefficient attempt was added as a reproducible finite-window
diagnostic:

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-interval-ratio-audit \
  --tube-a 100000000 \
  --taylor-p-slice-interval-order 20 \
  --taylor-p-slice-ratio-start 10 \
  --taylor-p-slice-ratio-bound 0.53 \
  --taylor-p-slice-b-subdivisions 20 \
  --taylor-p-slice-interval-working-dps 80
```

This currently returns

```text
status = interval_finite_ratios_failed
failed_subinterval =
  {'index': 0,
   'b_range': [-1e-08, -9e-09],
   'failure': 'interval Taylor midpoint preconditioner is not contractive'}
```

So plain interval Taylor coefficient propagation in `b` is too
overconservative even at moderate order.  This does not contradict the
sampled `q=0.53` evidence; it says the proof should probably use a hand or
symbolic majorant for the recurrence, or a sharper parameter-dependent
coefficient enclosure, rather than raw interval Gaussian solves on every
coefficient.

A Taylor-polynomial-centered ordinary-time tube was also tested from the
already-certified `t=0.01` bridge.  It failed immediately on early lower-face
checks even with generous radius growth; the time-coordinate interval
dependence is still too lossy there.  This makes the p-slice Taylor-tail
majorant look like the cleaner upstream proof target.

## Repaired Carried-C Chain

The existing saved carried-`C` p-tube

```text
output/s7_tail_proof/sampled_carried_c_p_tube_0.65_to_0.3255_tight.json
```

is marked failed for target `p=0.3255`, but its last certified box reaches
`p=0.3265`.  A later retry with more tuning attempts,

```text
output/s7_tail_proof/sampled_carried_c_p_tube_0.65_to_0.3255_tight_attempts120.json
```

does certify the sharper conditional tube all the way to `p=0.3255`, with
worst margin about `1.9677515554171465e-06`.

Starting from that last certified 5D box, two repaired automatic carried-`C`
corridors certify:

```text
p = 0.3265 -> 0.298   step = 2.5e-4   worst margin = 0.0026893291363964122
p = 0.298  -> 0.29    step = 5e-5     worst margin = 0.0032411718159143232
```

The resulting `p=0.29` 4D box is

```text
low  = (3.4391924881470897, 1.1136896739940916,
        0.000781944407090743, -1.6205653341549169)
high = (3.7442793524129394, 15.037333680341968,
        0.014833024272630795, -0.08797215325164864)
```

and the carried `C` interval is approximately

```text
C in [0.0021039197598916753, 0.24341137302927038].
```

## What Is Certified After p=0.29

From the repaired `p=0.29` source box, the widened carried-`C` wall proves

```text
x2 >= 0
```

down to `p=0.01`:

```text
status = certified_conditional
worst margin = 3.201538397554474e-07
source_box_contained = True
```

The existing small-`p` factor certificate then handles the remaining
`p <= 0.01` layer for the `x2=0` wall.

Thus the positive `x2` terminal obstruction is close to a composed proof once
the `p=0.65` start slice is validated.

## Remaining Obstruction

The `x3` terminal defect is not yet closed from the repaired `p=0.29` box if
one insists on using that broad one-box route.  However the separate
`support-tail` composition closes the terminal tail from a much sharper
`t=3.5` support box, so this is no longer the main explicit-`A` obstruction.

Simple rectangular terminal takeovers with walls

```text
x3 = -0.02, -0.05, -0.08
```

all fail immediately near `p=0.29`, even though the source satisfies
`x3 <= -0.0879`.  The failure is caused by wide boxes that allow nonphysical
combinations of `t`, `x1`, `x2`, `x3`, and `C`.

A one-box automatic carried-`C` continuation from `p=0.29` also fails
immediately if it is asked to move all the way to `p=0.28`.  Very small steps
can move slightly:

```text
p = 0.29 -> 0.2895   step = 2.5e-5   certified
p = 0.29 -> 0.2898   step = 1e-5     certified
```

but the enclosure widens quickly.  A shallow adaptive carried-`C` union run
from `p=0.29` to `p=0.28` began certifying split leaves, but the queue grew
large and the run was stopped as exploratory rather than completed.

Two additional shortcut diagnostics were checked:

- The `x3=0` wall from the earlier `p=0.3265` certified box succeeds down to
  `p=0.005` if one assumes a maintained lower bound `x1 >= 4`; the wall margin
  in that check is about `1.6066734213918436e-07`.
- A linear wall of the form `C + lambda*x3 = 0` is not promising in the broad
  late boxes.  For example, `lambda=2` is already negative on the certified
  `p=0.3255` source box, but interval evaluation of
  `(C + lambda*x3)_p` on coarse terminal boxes has large sign-indefinite
  ranges.  This looks like another correlation-loss issue, not a scalar wall
  that closes the proof by itself.

## Current Best Path

The most promising explicit-`A` route is now:

1. Prove a rigorous high-order Taylor/restart enclosure from the singular
   left endpoint to the 10x support box at `t=3.5`.
2. Reuse the already certified `support-tail` chain from that support box to
   terminal.

The alternative p-slice route remains useful as independent evidence:

1. Validate the `p=0.65` start slice for every `|a| >= 1e8`.
2. Reuse the certified sampled carried-`C` tube to `p=0.3255`.
3. Connect it to the already certified `p=0.325`/support-tail terminal chain,
   or continue the carried-`C` tube with finite unions.

So the explicit threshold `A=1e8` is substantially closer than before.  The
unproved part has been narrowed to a single upstream enclosure/entry lemma,
not the terminal exclusion.

## 2026-07-06 Follow-Up

Two more diagnostics narrow the shape of the remaining obstruction.

First, the ordinary-time restart chain is not the right way to hit the tiny
`t=3.5` support box.  A capped run of

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-restart-chain-check \
  --tube-a 100000000 \
  --taylor-progress-every-blocks 200 \
  --taylor-restart-progress-every-segments 1
```

timed out after 180 seconds while still in the initial bridge, certified to
about `t=1.87`.  Starting restarts earlier,

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-restart-chain-check \
  --tube-a 100000000 \
  --taylor-bridge-end 1.5 \
  --taylor-restart-end 2.0 \
  --taylor-progress-every-blocks 400 \
  --taylor-restart-progress-every-segments 1
```

timed out after 300 seconds with certification to `t=1.8`.  This was not a
mathematical failure: all reported blocks were certified.  But the
axis-aligned widths had already grown to approximately

```text
[0.01173743067431321, 0.535728010052984,
 0.0011022595431175123, 0.0674062448376378]
```

at `t=1.8`, much wider than the eventual support box.  So the direct
ordinary-time rectangular restart method is too lossy for support entry.  The
p-slice route remains the sharper upstream target.

Second, the downstream support-tail certificate tolerates a much wider
anisotropic `x2` support radius while keeping the narrow `p`, `x1`, and `x3`
radii:

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --support-tail-closure-check \
  --tube-a 100000000 \
  --support-tail-support-radius 1e-6,1e-5,5e-6,1e-6 \
  --support-tail-bridge-after-time 3.5056 \
  --support-tail-bridge-step 1e-4
```

returns

```text
support-tail closure certificate: status=certified_conditional
stages: bridge_after_t=3.5056, broad_from_p=0.325,
        automatic_to_p=0.212, terminal_margin=9.45935162e-06
```

This matters because the complex-`b` Cauchy diagnostic mostly stressed the
`x2` support radius.  The larger anisotropic support box leaves considerably
more room for the finite-`b` perturbation without weakening the already
certified terminal chain.

I also tested whether the broad `p=0.29` carried-`C` box could prove a more
direct terminal obstruction.  Positive constant lower walls for `x2` fail:

```text
x2 = 0.001   fails around p=0.225
x2 = 0.0005  fails around p=0.18
x2 = 0.0002  fails around p=0.135
x2 = 0.0001  fails around p=0.105
```

The code now also allows carried-`C` wall checks on component `4`, but the
one-box `C=0` lower wall from the saved `p=0.29` source fails immediately:

```text
carried-C p-wall certificate: status=failed
component=4, side=lower, value=0
certified_to_p=0.29
worst_margin=-15.1964374
```

This should be read as correlation loss in the broad one-box enclosure, not
as evidence that `C` is mathematically useless.  It says that any terminal
`C`-wall proof would need a narrower union or a tailored corridor, not the
current coarse box.

Current best target after these checks:

1. Validate the high-order `p=0.65` Taylor slice for every `|b| <= 1e-8`.
2. Use the already certified carried-`C` tube/corridor to reach `p=0.29`.
3. Either finish with the existing `x2 >= 0`/small-`p` wall plus a sharper
   last-layer estimate, or connect back into the certified support-tail chain.

The explicit candidate threshold remains

```text
A = 100000000.
```

but it is still conditional on the upstream Taylor/p-slice validation.

## 2026-07-06 P-Slice Complex-b Event Diagnostic

The upstream p-slice validation now has a sharper finite-`A` diagnostic.
The command

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-b-cauchy-event-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 40 \
  --taylor-p-slice-tail-working-dps 80 \
  --taylor-b-cauchy-radius 1e-7 \
  --taylor-b-cauchy-samples 4
```

samples the complex circle `|b|=1e-7`, solves the finite Taylor polynomial
event equation `p(t,b)=0.65` on that circle, and applies the formal
maximum-modulus shrink factor `(1e-8)/(1e-7)=0.1` to the event data

```text
Y(b) = (t, x1, x2, x3, C)|_{p=0.65}.
```

At order 40 it gives

```text
status = sampled_b_cauchy_event_delta_inside_start_radius
direct_delta/radius =
  [4.707345624410664e-09,
   1.2967404927621828e-09,
   0.007363959508149831,
   0.01979713990518661,
   0.005576194043316328]
cauchy_delta/radius =
  [4.7709947270950906e-08,
   1.3006710759621186e-08,
   0.007363965989991731,
   0.01979717210543269,
   0.005576199016717224]
max p-event residual = 1.05e-81
```

A cheaper order-60 check with two complex samples and direct endpoint
recomputation skipped,

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-b-cauchy-event-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 60 \
  --taylor-p-slice-tail-working-dps 90 \
  --taylor-b-cauchy-radius 1e-7 \
  --taylor-b-cauchy-samples 2 \
  --taylor-b-cauchy-skip-direct
```

is numerically identical at the displayed scale:

```text
cauchy_delta/radius =
  [4.7703865948744883e-08,
   1.3006498415640866e-08,
   0.007363965826795208,
   0.019797172695770155,
   0.005576199104178516]
max p-event residual = 7.35e-188
```

The corresponding order-80 two-sample run did not finish within a 420-second
cap, so this diagnostic needs optimization before it can be used at the full
order-80 tail setting.  Still, the order-40/order-60 agreement is strong
evidence that the finite-`b` motion of the p-slice event map is not the
bottleneck: the worst coordinate uses only about `2%` of the existing
carried-`C` p-tube start radius.

This improves the proof target:

1. certify the sampled complex-`b` circle maxima for the event map, or replace
   them by a recurrence/majorant in `b`;
2. combine that finite-`b` event bound with the existing p-slice Taylor tail
   majorant target;
3. feed the resulting p-slice box into the already certified carried-`C`
   p-tube.

## 2026-07-06 Combined P-Slice Entry Budget

The finite-`b` event motion and the Taylor tail now have a single combined
budget diagnostic.  The command

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-entry-budget-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 60 \
  --taylor-p-slice-tail-start 50 \
  --taylor-p-slice-ratio-start 45 \
  --taylor-p-slice-ratio-bound 0.53 \
  --taylor-p-slice-b-samples 3 \
  --taylor-p-slice-tail-working-dps 90 \
  --taylor-b-cauchy-radius 1e-7 \
  --taylor-b-cauchy-samples 2 \
  --taylor-b-cauchy-skip-direct
```

uses the formal p-tail time-shift bound for the time coordinate, the formal
geometric tail for `x1,x2,x3,C`, and the complex-`b` event Cauchy diagnostic
for finite-`A` motion of the degree-60 p-slice event map.

It returns

```text
status = formal_entry_budget_inside_start_radius

tail/radius =
  [0.00015914499882702815,
   6.122207296674624e-05,
   8.67491319681611e-06,
   0.00016421066266066457,
   2.159183754068536e-05]

finite_b/radius =
  [4.7703865948744883e-08,
   1.3006498415640866e-08,
   0.007363965826795208,
   0.019797172695770155,
   0.005576199104178516]

combined/radius =
  [0.0001591927026929769,
   6.123507946516189e-05,
   0.007372640739992025,
   0.01996138335843082,
   0.005597790941719201]

max combined radius use = 0.0199613834
```

Thus, at the current formal level, the full p-slice entry budget uses only
about `2.0%` of the certified carried-`C` p-tube start radius.  The bottleneck
is no longer numerical slack.  It is proving the two analytic inputs:

1. the complex-`b` event map maximum on `|b|=1e-7`;
2. the same-parity Taylor tail majorant after order 50 with ratio `0.53`.

If those two inputs are certified, the upstream handoff to the existing
carried-`C` p-tube has ample room for `A=1e8`.

The complex-`b` event-map audit was also rerun at order 40 with eight equally
spaced angles on `|b|=1e-7`:

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-b-cauchy-event-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 40 \
  --taylor-p-slice-tail-working-dps 80 \
  --taylor-b-cauchy-radius 1e-7 \
  --taylor-b-cauchy-samples 8 \
  --taylor-b-cauchy-skip-direct
```

The sampled maximum is unchanged from the four-axis run:

```text
cauchy_delta/radius =
  [4.7709947270950906e-08,
   1.3006710759621186e-08,
   0.007363965989991731,
   0.01979717210543269,
   0.005576199016717224]
```

The audit now also reports an empirical adjacent-angle variation allowance.
This is not a rigorous maximum principle certificate, but it is a useful
guard against missing a narrow peak between sampled angles:

```text
empirical_cauchy_delta/radius =
  [8.144299956681005e-08,
   2.220350447074293e-08,
   0.01018203571784243,
   0.027373231539293025,
   0.007710119488722653]
```

Even with this empirical interpolation allowance, the finite-`b` event-map
budget uses only about `2.74%` of the carried-`C` start radius.  The next
proof-level improvement would be to replace this empirical adjacent-angle
allowance by a true angular derivative bound, for example using an outer
complex-`b` circle and Cauchy's derivative estimate for the event map.

That outer-circle diagnostic is now implemented.  With inner radius `1e-7`,
outer radius `2e-7`, and eight angles on each circle,

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-b-cauchy-event-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 40 \
  --taylor-p-slice-tail-working-dps 80 \
  --taylor-b-cauchy-radius 1e-7 \
  --taylor-b-cauchy-samples 8 \
  --taylor-b-cauchy-outer-radius 2e-7 \
  --taylor-b-cauchy-outer-samples 8 \
  --taylor-b-cauchy-skip-direct
```

returns

```text
proof_cauchy_delta/radius =
  [1.2264760068790413e-07,
   3.3437038329994655e-08,
   0.013147617004544214,
   0.03534586275582822,
   0.009955739815371381]

proof_source = sampled_outer_circle_cauchy_angular_bound
```

This is still conditional on replacing the sampled outer-circle maximum by a
certified one, but the angular-variation part now has the right Cauchy shape:
the outer circle controls the derivative on the inner circle.  Even with this
more conservative allowance, the finite-`b` event-map budget uses only about
`3.54%` of the carried-`C` start radius.

A lower-order combined entry check using the same outer-circle source also
fits comfortably:

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-entry-budget-audit \
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
  --taylor-b-cauchy-skip-direct
```

gives

```text
combined/radius =
  [0.158511034458512,
   0.06592576229457212,
   0.022873439650454963,
   0.21462339300000285,
   0.033597360194445564]

max combined radius use = 0.214623393
```

The larger value here comes from the deliberately lower-order tail after
degree 30, not from the finite-`b` event map.  It is nevertheless well inside
the start box.

One further ratio sanity check changes the preferred tail target.  A direct
`b=0` high-order probe gave

```text
order 80:
  max p-slice ratios from degree 50:
    p:  0.522877870387067     at 78 -> 80
    x1: 0.5205460563938639    at 78 -> 80
    x2: 0.5199139317355432    at 78 -> 80
    x3: 0.5202495935302599    at 78 -> 80

order 90:
  max p-slice ratios from degree 50:
    p:  0.5245955847095568    at 88 -> 90
    x1: 0.5224929277611662    at 88 -> 90
    x2: 0.5219857099235997    at 88 -> 90
    x3: 0.5222461176156626    at 88 -> 90
```

These are still below `0.53`, but they are rising.  Since the expected
nearest-singularity scale could be above `0.53`, the better proof target is a
looser ratio such as `q=0.60`.  The combined entry budget remains essentially
unchanged:

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-entry-budget-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 60 \
  --taylor-p-slice-tail-start 50 \
  --taylor-p-slice-ratio-start 45 \
  --taylor-p-slice-ratio-bound 0.6 \
  --taylor-p-slice-b-samples 3 \
  --taylor-p-slice-tail-working-dps 90 \
  --taylor-b-cauchy-radius 1e-7 \
  --taylor-b-cauchy-samples 2 \
  --taylor-b-cauchy-skip-direct
```

returns

```text
tail/radius =
  [0.00018699537362175804,
   7.193593573592684e-05,
   1.0193023006258928e-05,
   0.00019294752862628089,
   2.537040911139736e-05]

finite_b/radius =
  [4.7703865948744883e-08,
   1.3006498415640866e-08,
   0.007363965826795208,
   0.019797172695770155,
   0.005576199104178516]

combined/radius =
  [0.00018704307748770675,
   7.194894223434248e-05,
   0.007374158849801467,
   0.019990120224396436,
   0.005601569513289913]

max combined radius use = 0.0199901202
```

So the current recommended analytic target is:

```text
after order 50, prove a same-parity p-slice tail ratio <= 0.60
```

rather than the tighter `0.53`.  This should be substantially easier to prove
and still leaves about `98%` of the start-box radius unused.

## 2026-07-06 Conditional Required-A Arithmetic

The p-slice entry budget now has an explicit threshold calculator.  It keeps
the Taylor-tail budget fixed and uses the fact that the complex-`b` event-map
budget scales linearly with `1/A`.  In other words, it answers:

```text
assuming the stated tail majorant and event-map maximum are certified,
how large does A need to be for the p=0.65 handoff box to fit?
```

The conservative lower-order outer-circle command

```text
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
  --taylor-b-cauchy-skip-direct
```

returns

```text
status = candidate_A_fits_conditional_entry_budget
minimum_A = 4306676.62
candidate_A_headroom_factor = 23.2197606
max_tail/radius = 0.17927753
max_finite_b/radius = 0.0353458628
max_combined/radius = 0.214623393
```

Thus the currently proposed explicit threshold

```text
A = 100000000
```

is not close to the conditional entry-budget boundary.  Under the present
outer-circle event and `q=0.60` Taylor-tail assumptions, the p-slice handoff
would already fit for `A` a little above `4.31e6`; `1e8` leaves about a
factor `23` of slack.

This does not close the theorem.  It sharpens the remaining proof task:

1. certify the complex-`b` event-map maximum on the chosen outer circle;
2. prove a same-parity Taylor tail majorant compatible with `q=0.60`;
3. then the explicit numerical threshold arithmetic already supports
   `A=1e8`.

The event-map side was then made one layer less empirical by adding a third
complex-`b` circle.  The command

```text
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
```

uses the `4e-7` circle to control angular variation on the `2e-7` circle,
then uses that enclosed outer-circle bound to control angular variation on
the `1e-7` circle.  It returns

```text
status = candidate_A_fits_conditional_entry_budget
minimum_A = 5794628.71
candidate_A_headroom_factor = 17.2573611
max_tail/radius = 0.17927753
max_finite_b/radius = 0.0475578199
max_combined/radius = 0.22683535
```

This is more conservative than the two-circle calculation, but still leaves
`A=1e8` with a factor `17` of conditional headroom.  The only sampled maximum
left on the event side is now the outermost `4e-7` circle.

The corresponding standalone event audit also records the event-root
simplicity margin:

```text
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect \
  --taylor-p-slice-b-cauchy-event-audit \
  --tube-a 100000000 \
  --taylor-p-slice-tail-order 40 \
  --taylor-p-slice-tail-working-dps 80 \
  --taylor-b-cauchy-radius 1e-7 \
  --taylor-b-cauchy-samples 8 \
  --taylor-b-cauchy-outer-radius 2e-7 \
  --taylor-b-cauchy-outer-samples 8 \
  --taylor-b-cauchy-enclosure-radius 4e-7 \
  --taylor-b-cauchy-enclosure-samples 8 \
  --taylor-b-cauchy-skip-direct
```

returns

```text
proof_source = sampled_enclosure_circle_nested_cauchy
proof_cauchy_delta/radius =
  [2.403574449949711e-07,
   6.552869959789399e-08,
   0.01769009476501496,
   0.047557819876193244,
   0.013395429919872984]
max p-event residual = 1.05e-81
min |d p_N / dt| at the event samples = 0.263272645
```

Thus the finite Taylor event root is far from multiple on the sampled
inner/outer/enclosing circles.  This supports the implicit-function part of
the event-map proof target; the remaining event-side missing input is still a
certified maximum for the outermost circle rather than the sampled maximum.
