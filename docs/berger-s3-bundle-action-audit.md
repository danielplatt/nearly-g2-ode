# Berger S3-Bundle Action Audit

Reproducibility command:

```zsh
.venv/bin/python -m experiments.berger_s3_bundle_action_audit
.venv/bin/python -m experiments.berger_s3_bundle_action_audit --json
```

## Purpose

This note checks whether the diffeomorphism between the Berger space

```text
B^7 = SO(5)/SO(3)_irr
```

and an `S3`-bundle over `S4` gives a second seven-dimensional
cohomogeneity-one action suitable for the nearly `G2` endpoint search.

## Topology Check

Goette-Kitchloo-Shankar identify the Berger space with an `S3`-bundle over
`S4`. In the Crowley-Escher notation this is the orientation convention

```text
M_{-1,10}
```

so the Euler class is `10` and

```text
p1(TM) = 2(10 + 2(-1)) = 16 = 6 mod 10.
```

In Grove-Ziller `M_{k,l}` notation the matching representative is

```text
M_{6,4}
```

because `k + l = 10` and `4l = 16 = 6 mod 10`.

One deterministic Grove-Ziller slope choice is

```text
p_- = 13,  p_+ = -11,
q_- = -7,  q_+ = 9,
```

which gives

```text
k = (p_-^2 - p_+^2)/8 = 6,
l = -(q_-^2 - q_+^2)/8 = 4.
```

## Action Check

Grove-Ziller construct a cohomogeneity-one action on the principal `SO(4)`
bundle over `S4`. In this Berger-matching case that principal bundle has
dimension `10`, with effective group `SO(4) x SO(3)` and singular orbits of
codimension `2`.

That is useful geometry, but it is not a seven-dimensional nearly `G2`
endpoint problem.

On the associated `S3`-bundle `M_{6,4}`, the induced action is the `SO(3)`
action described by Grove-Ziller. For the slopes above its finite orbit types
include

```text
1, Z2, D2, D3, D10, D1, D10.
```

The generic orbit is therefore three-dimensional, so the action has
cohomogeneity `4` on the seven-dimensional Berger-sized manifold.

## Literature Notes

- Goette-Kitchloo-Shankar compute the Berger diffeomorphism type as an
  `S3`-bundle over `S4`.
- Crowley-Escher classify the `M_{m,n}` bundles and give the characteristic
  class formulas used above.
- Grove-Ziller give cohomogeneity-one structures on the principal `SO(4)`
  bundles and use them to construct nonnegative-curvature metrics on the
  associated sphere bundles.
- Grove-Ziller also discuss earlier positive-Ricci connection-type metrics of
  Nash and Poor, and the Derdzinski-Rigas restriction for positive curvature
  connection metrics on `S3`-bundles over `S4`.

I did not find a known nearly, closed, or coclosed `G2` solution attached to
this associated-bundle action in this audit pass.

## Endpoint Smoothness Status

There is no new seven-dimensional endpoint smoothness chart to derive from
this route:

- the Grove-Ziller cohomogeneity-one action lives on a ten-dimensional
  principal bundle;
- the associated seven-dimensional Berger-sized sphere bundle has
  cohomogeneity `4`, not cohomogeneity `1`.

Thus the smoke test conclusion is negative but useful: the topology matches,
while the action does not produce a new Berger `G2` ODE ansatz.
