# S7 SU(2)^3 Defect Brainstorming Handoff

This note is a prompt and context package for brainstorming scalar defects for
the Podesta `SU(2)^3` cohomogeneity-one `S^7` problem.  The goal is not a
rigorous proof at this stage.  The goal is to generate many candidate
quantities, including unintuitive ones, that might be easier to use in a later
large-parameter exclusion proof.

## Copy-Paste Prompt

```text
I am studying Podesta's SU(2)^3 cohomogeneity-one nearly parallel G2 ODE on
S^7.  I want help brainstorming scalar "defects" for a large-|a| exclusion
strategy.

Important: a defect does NOT need to characterize solutions.  I only need
necessary defects:

  compact smooth right endpoint => D(a) = 0.

It is fine if D(a)=0 is not sufficient.  It is also fine if the defect is
chart-dependent, asymmetric, ugly, or unintuitive, provided smooth terminal
closure really forces it to vanish.  No rigor is required at this stage.  I
want ideas, the more the better.  Crazy ideas are welcome, because we can test
and discard them quickly.

The desired brainstorming output:

1. Many possible scalar defects D(a), preferably grouped by type:
   endpoint coordinates, endpoint ratios, derivatives at the terminal event,
   linear or nonlinear combinations, volume/max-volume quantities, conserved or
   almost-conserved quantities, monotonicity/barrier variables, rescaled
   quantities, event-location quantities, and anything else you can think of.

2. For each candidate, explain briefly why compact standard K- closure should
   imply D(a)=0.

3. For each candidate, give a quick sanity check: possible advantages,
   possible degeneracies, and whether its large-|a| behavior might be easier to
   prove than the raw endpoint conditions.

4. Do not try to give a rigorous proof.  The next step after brainstorming will
   be numerical triage: evaluate each candidate for many finite a values and
   in the large-|a| limiting IVP, discard defects with extra zeros, and keep
   those that appear bounded away from zero or tend to a nonzero/infinite
   limit.

Here is the mathematical setup.

Podesta's five-function reduction uses

  f0 = t h0,
  f1 = t^4 h1,
  f2 = h2,
  f3 = t^2 h3,
  f4 = t^2 h4,
  h4 = -h3 - h0^2/6.

The one-ended smooth initial data are parametrized by a nonzero real number a:

  h0(0) = a,
  h1(0) = 27/4,
  h2(0) = -a^3/27,
  h3(0) = 3a.

Known direct compact values in this chart are:

  round S7:    a = -36,
  squashed S7: a = 108/5.

The positive round value a=36 is equivalent by an outer automorphism, but it
does not close in this unmodified standard K- chart.

The finite h-ODE is:

  h4 = -h3 - h0^2/6

  h0' = a0/t + b0
  h1' = a1/t + b1
  h2' = b2
  h3' = a3/t + b3

where

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

The standard K- right endpoint closure condition is that at a terminal point
with f1 nonzero,

  f0 = f2 = f3 = f4 = 0.

Equivalently, at a first f0=0 or h0=0 terminal event T_a, this forces

  h2(T_a) = 0,
  h3(T_a) = 0,
  h4(T_a) = 0.

Since h4=-h3-h0^2/6, h0=0 and h3=0 also imply h4=0.

For large |a| we used scaled variables

  h0 = a x0,
  h1 = x1,
  h2 = a^3 x2,
  h3 = a x3,
  b = 1/a.

The exact scaled finite-a ODE is a polynomial perturbation in b of the limiting
scaled ODE.  The limiting ODE for x=(x0,x1,x2,x3), with p=x0, is:

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

The scaled smooth left-end initial value is

  x(0) = (1, 27/4, -1/27, 3)

in the b=0 limit.

What has already been tried:

1. The original scout used the scale-normalized endpoint proxy

     loss =
       sqrt((f0/scale_f0)^2 + (f2/scale_f)^2
            + (f3/scale_f)^2 + (f4/scale_f)^2),

     scale_f = max(1, |f1|),
     scale_f0 = max(1, |f1|^(2/3)).

   This is good for finding numerical endpoint signals, but it is not an ideal
   proof object.

2. The first serious proof attempt focused on the necessary defect

     X2(a) = x2(T_a) = h2(T_a)/a^3,

   where T_a is the first x0=0 crossing.  Standard K- closure forces X2(a)=0.
   Numerically, the limiting first crossing has approximately

     T_infinity ~= 3.598,
     X2(T_infinity) ~= 0.006,
     X3(T_infinity) ~= -1.1.

   Finite samples such as a=+-250, +-500 and larger values have the same
   positive X2 sign, but the fixed-step terminal samples are sensitive near the
   singular crossing.  This looks asymptotically nonzero, but proving a global
   finite-A exclusion from X2 alone became awkward.

3. The auxiliary endpoint defect

     X3(a) = x3(T_a) = h3(T_a)/a

   is also forced to vanish by standard K- closure and is numerically much
   farther from zero in the limiting crossing, around -1.1.  It has not yet
   received the same level of defect-audit attention as X2.

4. A useful cancellation variable found during proof search was

     C = x1*x2 - p^2*x3/6,  p=x0.

   In the limiting system,

     C' =
       -4 C/t + 2 x2*x3^3/(t*p^3) - p^3/t + x1*t^3*p^3/108.

   On a negative x3 wall, the combination C cancels bad singular terms in x3'.
   A promising barrier region near the late tail was

     x3 <= -0.36,
     C >= 1.23 p^3,
     p <= 0.33,
     t in [3.5,4.0].

   But the proof got stuck because rectangular interval boxes lost correlations
   between x1, x2, x3, and C.  This suggests C or a related correlated quantity
   may be a good defect/barrier ingredient, even if X2 itself is not the best
   proof object.

5. One exact wall identity is especially simple:

   On x2=0,

     x2' = t^3 p^3/216 * (1 + 6 b x3/p^2)^3

   in the finite scaled family, where b=1/a.  This may or may not suggest
   useful defects involving x2, the factor 1+6b x3/p^2, or derivatives at the
   terminal event.

Please brainstorm broadly.  Again: no rigor is needed now.  I want a large menu
of necessary scalar defects and related proof-friendly quantities, including
weird combinations.  The final answer should be idea-rich rather than polished.
```

## Notes For Later Numerical Triage

After collecting suggestions, the intended local workflow is:

1. Implement a small defect-audit script that evaluates each candidate on:
   finite positive and negative `a`, the known round/squashed values, and the
   limiting scaled IVP.

2. Discard necessary defects that visibly have extra zeros in the large-tail
   region.

3. Prefer defects with one of these signatures:
   - `D(a) -> D_infinity != 0`;
   - `|D(a)| -> infinity`;
   - a stable sign for both large positive and large negative `a`;
   - a simple differential equation or favorable barrier wall;
   - robustness to terminal event step size.

4. Only after this triage try to prove the easier asymptotic statement.  A
   uniform `|a|>=A` exclusion should come last.

## Why This Is Worth Doing

The earlier sprint did not fail because the numerics were useless.  It taught
several valuable facts:

- The original endpoint loss is a good scout signal but a poor proof object.
- The large-`|a|` scaled limit is the right conceptual frame.
- The terminal condition gives several necessary scalar vanishing conditions,
  not just `X2=0`.
- The auxiliary endpoint coordinate `X3` is numerically farther from zero than
  `X2` and deserves attention.
- The cancellation variable `C=x1*x2-p^2*x3/6` exposes structure that is hidden
  in the raw endpoint coordinates.
- Correlations between variables matter; wide rectangular boxes can make true
  scalar barriers look false.

So the next good step is a defect brainstorm, not more force applied to the
same `X2` proof attempt.
