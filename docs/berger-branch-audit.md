# Berger Branch Closure Audit

Generated from local experiment artifacts.

Reproduce with:

```zsh
.venv/bin/python -m experiments.berger_branch_audit --write-markdown docs/berger-branch-audit.md
```

## Canonical Runs

| Run | Scouts | Best scout | Refinement | Best final | Verifications | Branch | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| near vertex | near vertex: 103680 seeds, 53284 ok/50396 fail | 0.025210347 (seed 67427) | 89 selected; failed=3, inconclusive=5, recovered_berger=81 | 2.7639587e-12 (seed 103676) | 172 | a>0, c<0; mu=(-1,-1); p=(-1, 1, 1), right_p=global | berger-recovered |
| near cell-center | near cell-center: 20736 seeds, 11593 ok/9143 fail | 0.028360347 (seed 9930) | 30 selected; failed=1, inconclusive=1, recovered_berger=28 | 2.7639595e-12 (seed 20149) | 58 | a>0, c<0; mu=(-1,-1); p=(-1, 1, 1), right_p=global | berger-recovered |
| symmetric-alpha-omega vertex | symmetric-alpha-omega vertex: 250880 seeds, 123336 ok/127544 fail | 0.031984794 (seed 166502) | 86 selected; failed=3, inconclusive=5, recovered_berger=78 | 2.7639587e-12 (seed 250876) | 166 | a>0, c<0; mu=(-1,-1); p=(-1, 1, 1), right_p=global | berger-recovered |
| symmetric-alpha-omega cell-center | symmetric-alpha-omega cell-center: 54756 seeds, 29403 ok/25353 fail | 0.023719533 (seed 9598) | 32 selected; inconclusive=2, recovered_berger=30 | 2.7639579e-12 (seed 12744) | 64 | a>0, c<0; mu=(-1,-1); p=(-1, 1, 1), right_p=global | berger-recovered |
| positive-ac standard | positive-ac vertex: 250880 seeds, 153588 ok/97292 fail<br>positive-ac cell-center: 54756 seeds, 41675 ok/13081 fail | 0.39074614 (seed 0) | none | n/a | 0 | a>0, c>0; mu=(-1,-1); p=(-1, 1, 1), right_p=global | no-low-scout-signal |
| positive-ac boundary | positive-ac-boundary cell-center: 91125 seeds, 71432 ok/19693 fail<br>positive-ac-boundary vertex: 393216 seeds, 252714 ok/140502 fail | 0.15061996 (seed 10794) | none | n/a | 0 | a>0, c>0; mu=(-1,-1); p=(-1, 1, 1), right_p=global | no-low-scout-signal |
| positive-ac boundary-v2 | positive-ac-boundary-v2 vertex: 262144 seeds, 254208 ok/7936 fail | 0.051866785 (seed 15420) | 76 selected; failed=76 | 0.0030252399 (seed 25700) | 0 | a>0, c>0; mu=(-1,-1); p=(-1, 1, 1), right_p=global | collapsed-tail |
| negative-ac standard | negative-ac vertex: 250880 seeds, 208800 ok/42080 fail | 2.439706 (seed 140629) | none | n/a | 0 | a<0, c<0; mu=(-1,-1); p=(-1, 1, 1), right_p=global | no-low-scout-signal |
| mixed-mu short | mixed-mu-short vertex: 32400 seeds, 30384 ok/2016 fail | 0.2237561 (seed 0) | none | n/a | 0 | a>0, c<0; mu=(1,1); p=(1, 1, 1), right_p=(-1, 1, -1) | no-low-scout-signal |
| mixed-mu boundary | mixed-mu-boundary vertex: 32400 seeds, 32400 ok/0 fail | 0.060974981 (seed 3276) | 21 selected; failed=20, inconclusive=1 | 0.009467176 (seed 10558) | 0 | a>0, c<0; mu=(1,1); p=(1, 1, 1), right_p=(-1, 1, -1) | collapsed-tail |

## Branch Coverage

The global mirrored `p`/`mu` probe finds only the already explored Berger branch:

- `p=(-1, 1, 1), left_mu=-1, right_mu=-1, left=0.0, right=0.0`

The non-default opposite-`mu` scoutable branch requires endpoint-local square-root signs:

- `left_p=(1, 1, 1), right_p=(-1, 1, -1), left_mu=1, right_mu=1, left=0.0, right=0.0`

The physical endpoint branches covered by the grid artifacts are:

- Default Berger component, including the original near grid and the symmetric physical `alpha/omega` follow-up.
- Positive `a,c` component with `3a-c>0`, including standard, boundary, and boundary-v2 strips.
- Negative `a,c` component with `3a-c>0`.
- Mixed endpoint-local opposite-`mu` component with `left p=(1,1,1)` and `right p=(-1,1,-1)`.

## Conclusion

No non-Berger nearly G2 candidate survived refinement or high-order verification in the explored Berger branches.

The Berger-near and symmetric `alpha/omega` grids recover Berger. The positive `a,c` and mixed-`mu` boundary follow-ups enter short-interval collapsed tails with nonzero residuals. The remaining standard branches have no low scout signal. No explored branch produced a verified non-Berger candidate.

