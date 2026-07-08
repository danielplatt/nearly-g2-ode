# S7 SU(2)^3 Podesta Scout

Reproducibility commands:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_scout --recover-known
.venv/bin/python -m experiments.s7_su2_cubed_scout --dry-run
.venv/bin/python -m experiments.s7_su2_cubed_scout
```

This scout uses Podesta's `SU(2)^3` five-function nearly parallel `G2`
equations, normalized to `lambda=1`.  Smoothness at the left singular `S3`
is represented by

```text
f0=t h0
f1=t^4 h1
f2=h2
f3=t^2 h3
f4=t^2 h4
h4=-h3-h0^2/6
```

with the single free endpoint parameter

```text
h0(0)=a
h1(0)=27/4
h2(0)=-a^3/27
h3(0)=3a
```

The scout marches this one-ended solution and searches for a standard `K-`
closure, where `f0`, `f2`, `f3`, and `f4` vanish while `f1` stays nonzero.
The implementation requires the putative terminal point to occur away from the
left seed and to have nonzero terminal `f1`; this filters the degenerate
`a -> 0` limit, where the left seed itself otherwise produces a fake tiny
residual.
With this terminal chart, the direct homogeneous checks are

```text
round S7:    a=-36
squashed S7: a=108/5
```

The positive round value `a=36` is equivalent by Podesta's outer automorphism,
but it does not close in the unmodified standard `K-` chart.  The scout records
this explicitly so the sign convention is not hidden.
