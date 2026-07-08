# S7 SU(2)^3 Next-Sprint Audit

Reproducibility command:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_next_sprint_audit
```

## Executive summary

Top-ranked route: **D_C_IF integral route**.

Status: `promising`.

CIF cumulative integral has stable positive sign in the sampled large-tail family.

The short conclusion is that the `D_C_IF` integral route is currently the cleanest new proof target.  The regular p-section cone route found a strong `p=0.33` cone entry and positive normalized-`c` wall margins, but the available one-number event-map stability bound is still too crude to certify finite `|b|<=1e-8`.  Terminal-manifold separation remains blocked by the missing backward `K_-` terminal chart.

## What Was Preserved From The Previous D_x3 Proof

The existing conditional downstream tail exclusion is preserved.  Once a trajectory is in the late correlated region, the previous `D_x3` terminal/tail mechanism still gives the contradiction to compact `K_-` closure.  This sprint only tries to replace the upstream tiny `t=3.5` support-entry step.

## p-section audit results

| p | status | t(limit) | x3(limit) | c(limit) | best sigma | x3 margin | best K | c margin | finite-b max dev |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.45 | not promising | 3.2720355 | 0.1114248 | 0.72356241 | n/a | n/a | 0.5 | 0.22356195 | 4.5744163e-07 |
| 0.4 | not promising | 3.383866 | -0.08821575 | 1.114853 | n/a | n/a | 1 | 0.1148524 | 5.7110824e-07 |
| 0.35 | not promising | 3.4715449 | -0.28653494 | 1.7063789 | 0.25 | 0.036534555 | 1.5 | 0.20637822 | 7.2526621e-07 |
| 0.33 | promising | 3.499291 | -0.36550342 | 2.0344257 | 0.36 | 0.0055030266 | 2 | 0.034424852 | 8.0229672e-07 |
| 0.3 | promising | 3.5330206 | -0.48282761 | 2.6804444 | 0.45 | 0.032827191 | 2 | 0.68044348 | 9.3909234e-07 |
| 0.25 | promising | 3.5699486 | -0.67099524 | 4.4625314 | 0.6 | 0.070994769 | 2 | 2.4625302 | 1.2420112e-06 |

Recommended section: `p=0.33` with cone `{'p_max': 0.33, 'x3_upper': -0.36, 'c_lower': 1.23, 'x1_lower_observed': 6.48649328277198, 'x2_lower_observed': 0.010248564681910459}`.

## finite-b event-map stability results

Status: `inconclusive, with exact blocker`.

Sampled `L`: `2173.6543`.
Sampled `partial_b` max: `116.48839`.
Transversality `m`: `0.26334162`.
Regular segment: `p=0.65` to `p=0.33`.
Predicted state error from crude Gronwall: `inf`.

Blocker: sampled row-sum Gronwall bound is too pessimistic for a proof certificate

## normalized c=C/p^3 cone results

Status: `promising`.

`x3` wall margin: `0.0034644208`.
`c` wall limiting hdot lower: `3.2954994e-05`.
finite-b grid hdot margin: `0.00016922342`.

## D_C_IF integral decomposition

Status: `promising`.

Endpoint `T^4 C(T)` in limit: `9.0677729`.
Integral total at `p=0.33`: `10.962356`.
Integral total at `p=0.25`: `11.325283`.

| p | I1 | I2 | I3 | total | endpoint t^4 C |
|---:|---:|---:|---:|---:|---:|
| 0.6 | -1.2978714 | -6.5446852 | 8.7190419 | 0.87648534 | 0.87648523 |
| 0.45 | -1.1115102 | -8.419729 | 17.088868 | 7.557629 | 7.557629 |
| 0.4 | -1.1113901 | -8.738263 | 19.204791 | 9.3551382 | 9.3551383 |
| 0.35 | -1.1242132 | -8.9266701 | 20.676906 | 10.626022 | 10.626023 |
| 0.33 | -1.1461926 | -8.9729734 | 21.081522 | 10.962356 | 10.962356 |
| 0.3 | -1.2185258 | -9.0191784 | 21.513707 | 11.276003 | 11.276004 |
| 0.25 | -1.4993445 | -9.0547619 | 21.87939 | 11.325283 | 11.325286 |
| 0.2 | -2.0072015 | -9.0648402 | 21.994488 | 10.922447 | 10.922458 |
| 0.1 | -3.2845134 | -9.0669804 | 22.021177 | 9.6696834 | 9.6724624 |
| 0.01 | -3.8828003 | -9.0669889 | 22.021295 | 9.0715054 | 9.0742844 |
| 0.001 | -3.8892467 | -9.0669889 | 22.021295 | 9.0650591 | 9.0678381 |
| 0 | -3.8893118 | -9.0669889 | 22.021295 | 9.0649939 | 9.0677729 |

## L scalar audit

Status: `not promising`.

Formula/chain-rule max discrepancy on audited sections: `5.6941847e-09`.

Blocker: L or Lprime does not show a single useful sign across the audited regular sections

## terminal-manifold separation attempt

Status: `inconclusive, with exact blocker`.

Blocker: no implemented backward K_- terminal Taylor chart in the current Podesta SU(2)^3 code

| p | separator type | best feature | margin | status |
|---:|---|---|---:|---|
| 0.33 | coordinate_proxy | x1 | 3.7772654 | inconclusive, with exact blocker |
| 0.25 | coordinate_proxy | x1 | 5.0131217 | inconclusive, with exact blocker |

## recommended proof route

| rank | route | status | reason |
|---:|---|---|---|
| 1 | D_C_IF integral route | `promising` | CIF cumulative integral has stable positive sign in the sampled large-tail family. |
| 2 | p-section cone route | `inconclusive, with exact blocker` | Section p=0.33 enters the cone, but the simple Gronwall/event-map proof remains inconclusive: sampled row-sum Gronwall bound is too pessimistic for a proof certificate |
| 3 | terminal-manifold separation route | `inconclusive, with exact blocker` | no implemented backward K_- terminal Taylor chart in the current Podesta SU(2)^3 code |

## remaining gaps

- Upgrade the p-section finite-b event-map stability estimate from sampled diagnostics to a proof.  The current one-number Gronwall bound is intentionally crude and may be too pessimistic.
- Turn the normalized `c=C/p^3` cone wall margins into a full entry-and-invariance lemma from the selected regular section.
- If pursuing `D_C_IF`, prove an integral dominance estimate on a compact interval and a tail bound showing later pieces cannot cancel the accumulated sign.
- Derive or implement the backward smooth `K_-` terminal chart before treating terminal-manifold separation as more than a proxy diagnostic.
