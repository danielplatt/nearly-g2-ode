# S7 Cohomogeneity-One Action Census

Reproducibility command:

```zsh
.venv/bin/python -m experiments.s7_action_census --write-markdown docs/s7-action-census.md
```

## Executive recommendation

Top next action: `sp1_2_u1_intermediate`.

Best balance of novelty and tractability: it contains the Podesta five-function chart as a subfamily but allows U(1)-invariant deformations on the same S3 x S3 orbit foliation.

Immediate task:

> Derive the U(1)-invariant SU(3)-structure algebra on S3 x S3 and verify round/squashed restrictions.

The point is to leave the large-|a| proof route alone and instead test a new S7 symmetry reduction.  The best first target is not a wholly unrelated topology, but a weaker-invariance S3 x S3 action that contains the already-tested Podesta chart as a calibration subfamily.

## Step 1: Candidate actions

| key | group | principal orbit | singular orbits | source |
| --- | --- | --- | --- | --- |
| so4_z2_q_system | SO(4) | SO(4)/Z_2^2 | two lower-dimensional SO(4)-orbits in the q_i chart | existing q-system action |
| sp1_3_diag_podesta | Sp(1)^3 | S3 x S3 | S3 and S3 | Podesta SU(2)^3 action |
| sp1_2_u1_intermediate | Sp(1) x Sp(1) x U(1) | S3 x S3 | S3 and S3 | intermediate S3 x S3 symmetry ladder |
| su3_u1_complex_sum | S(U(3) x U(1)) | S5 x S1 | S5 and S1 | complex linear sum action on C3 + C |
| sp1_2_left_sum | Sp(1) x Sp(1) | S3 x S3 | S3 and S3 | minimal S3 x S3 sum action on H + H |
| g2_principal_s6 | G2 | S6 = G2/SU(3) | point and point | simple-group action fixing the real octonion coordinate |
| real_sum_s1_s5 | SO(2) x SO(6) | S1 x S5 | S1 and S5 | real linear sum action on R^{p+1}+R^{q+1} |
| real_sum_s2_s4 | SO(3) x SO(5) | S2 x S4 | S2 and S4 | real linear sum action on R^{p+1}+R^{q+1} |
| real_sum_s3_s3 | SO(4) x SO(4) | S3 x S3 | S3 and S3 | real linear sum action on R^{p+1}+R^{q+1} |

## Step 2: Duplicate and viability filter

| key | status | duplicate/equivalence note | rationale |
| --- | --- | --- | --- |
| so4_z2_q_system | already-tested | implemented Berger/S7 q_i action | This is the S7 action already used by the fixed-chart and full-moduli q_i searches. |
| sp1_3_diag_podesta | already-tested | larger normal extension of the S3 x S3 orbit foliation | This is the recent five-function Podesta system; it recovered only round and squashed compact closures. |
| sp1_2_u1_intermediate | new-candidate | same orbit foliation as Podesta, but strictly weaker invariance | Best balance of novelty and tractability: it contains the Podesta five-function chart as a subfamily but allows U(1)-invariant deformations on the same S3 x S3 orbit foliation. |
| su3_u1_complex_sum | new-candidate | not equivalent to the tested S3 x S3 or q_i actions | A moderate 10-function ansatz with a different principal orbit.  Round S7 should calibrate it; squashed visibility is unclear, so validation is weaker than for the S3 x S3 ladder. |
| sp1_2_left_sum | new-candidate | same orbit foliation as Podesta, weakest invariance in this ladder | Maximal search space on the S3 x S3 foliation, but the 35-function raw ansatz is probably too large until the U(1)-intermediate case has taught us the algebra. |
| g2_principal_s6 | new-candidate | not equivalent, but expected to be uniqueness-rigid | Tiny and excellent as a sanity check, but Cleyton-Swann's simple-group picture makes it a poor place to expect a new compact nearly-parallel G2 structure. |
| real_sum_s1_s5 | discard | symmetric-space linear action with no invariant G2-form room | The invariant principal 2- and 3-form dimensions do not both survive, so a stable invariant G2 form cannot be built in the dt^omega+gamma ansatz. |
| real_sum_s2_s4 | discard | symmetric-space linear action with no invariant G2-form room | The invariant principal 2- and 3-form dimensions do not both survive, so a stable invariant G2 form cannot be built in the dt^omega+gamma ansatz. |
| real_sum_s3_s3 | discard | symmetric-space linear action with no invariant G2-form room | The invariant principal 2- and 3-form dimensions do not both survive, so a stable invariant G2 form cannot be built in the dt^omega+gamma ansatz. |

## Step 3: Invariant form dimensions

| key | invariant 2-forms | invariant 3-forms | total functions | stable room? |
| --- | ---: | ---: | ---: | --- |
| `so4_z2_q_system` | n/a | n/a | 8 | True |
| `sp1_3_diag_podesta` | 1 | 4 | 5 | True |
| `sp1_2_u1_intermediate` | 5 | 8 | 13 | True |
| `su3_u1_complex_sum` | 4 | 6 | 10 | True |
| `sp1_2_left_sum` | 15 | 20 | 35 | True |
| `g2_principal_s6` | 1 | 2 | 3 | True |
| `real_sum_s1_s5` | 0 | 0 | 0 | False |
| `real_sum_s2_s4` | 1 | 0 | 1 | False |
| `real_sum_s3_s3` | 0 | 2 | 2 | False |

## Step 4: Known-solution visibility

| key | round visible | squashed visible | endpoint profile |
| --- | --- | --- | --- |
| so4_z2_q_system | yes | yes | implemented p2/p3 and Berger endpoint charts |
| sp1_3_diag_podesta | yes | yes | codimension-4 endpoints; one-parameter smooth left germ after normalization |
| sp1_2_u1_intermediate | yes | yes | codimension-4 endpoints on both sides; broader than Podesta but still symmetry-reduced |
| su3_u1_complex_sum | yes | not-known | asymmetric codimension-2 and codimension-6 singular endpoints |
| sp1_2_left_sum | yes | yes | codimension-4 endpoints with large endpoint smoothness representation |
| g2_principal_s6 | yes | no | two point singular orbits; very small sine-cone style endpoint problem |
| real_sum_s1_s5 | no | no | linear sphere endpoints; dimensionally fails invariant G2 ansatz |
| real_sum_s2_s4 | no | no | linear sphere endpoints; dimensionally fails invariant G2 ansatz |
| real_sum_s3_s3 | no | no | linear sphere endpoints; dimensionally fails invariant G2 ansatz |

## Step 5: Ranking

| rank | key | score | total functions | reason to do it next |
| ---: | --- | ---: | ---: | --- |
| 1 | `sp1_2_u1_intermediate` | 98 | 13 | Best balance of novelty and tractability: it contains the Podesta five-function chart as a subfamily but allows U(1)-invariant deformations on the same S3 x S3 orbit foliation. |
| 2 | `su3_u1_complex_sum` | 80 | 10 | A moderate 10-function ansatz with a different principal orbit.  Round S7 should calibrate it; squashed visibility is unclear, so validation is weaker than for the S3 x S3 ladder. |
| 3 | `g2_principal_s6` | 53 | 3 | Tiny and excellent as a sanity check, but Cleyton-Swann's simple-group picture makes it a poor place to expect a new compact nearly-parallel G2 structure. |
| 4 | `sp1_2_left_sum` | 51 | 35 | Maximal search space on the S3 x S3 foliation, but the 35-function raw ansatz is probably too large until the U(1)-intermediate case has taught us the algebra. |

Actions not in the ranking are either already tested or dimensionally unsuitable for an invariant G2 ansatz.

## Practical next sprint

1. Work on `sp1_2_u1_intermediate` first.
2. Build the U(1)-invariant 2-form and 3-form basis on the S3 x S3 principal orbit.
3. Restrict that basis to the diagonal Sp(1)-invariant subspace and verify that it reproduces the existing Podesta five-function chart.
4. Express the round and squashed S7 homogeneous solutions in the new coordinates.
5. Only after those two calibrations pass, derive endpoint smoothness and a cheap scout.

## References Used

- Hoelscher, `Classification of Cohomogeneity One Manifolds in Low Dimensions`, arXiv:0712.1327.
- Cleyton-Swann, `Cohomogeneity-one G2-structures`, arXiv:math/0111056.
- Podesta, `Nearly parallel G2-structures with large symmetry group`, arXiv:1905.03077.
- Existing local audits: `docs/s7-su2-cubed-action-audit.md`, `docs/s7-su2-cubed-podesta-scout.md`, and `docs/2026-07-07-handover.md`.
