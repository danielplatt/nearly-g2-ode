# Nearly G2 ODE Exploration

This repository contains a high-precision numerical exploration code for the nearly parallel `G_2` `q`-system coming from the cohomogeneity-one equations. The geometric raw `q` equations are the source of truth, while endpoint-adapted weighted `y` variables are used on both singular ends to build Taylor series and march inward to a common match point.

## Setup

Create a local virtual environment in the repository root:

```zsh
python3 -m venv .venv
```

Activate it:

```zsh
source .venv/bin/activate
```

Install the requirements:

```zsh
pip install -r requirements.txt
```

## Run Named Experiments

From the repository root, with the virtual environment activated:

```zsh
python -m experiments.berger
python -m experiments.berger_opposite_end_check
python -m experiments.doubled_sphere
python -m experiments.berger_jacobian
python -m experiments.mirror_search
python -m experiments.mirror_recovery_calibration
python -m experiments.mirror_covering_calibration
python -m experiments.mirror_guarded_covering_search
python -m experiments.mirror_sweep
python -m experiments.mirror_sweep_v2
python -m experiments.mirror_sweep_v3
python -m experiments.mirror_local_grid_v3
python -m experiments.non_mirrored_berger_jacobian
python -m experiments.non_mirrored_recovery_calibration
python -m experiments.non_mirrored_grid_search
python -m experiments.non_mirrored_grid_refine
python -m experiments.berger_branch_audit
python -m experiments.berger_s3_bundle_action_audit
python -m experiments.berger_max_volume_calibration
python -m experiments.berger_max_volume_scout --dry-run
python -m experiments.non_mirrored_search
python -m experiments.non_mirrored_surrogate_search
python -m experiments.non_mirrored_surrogate_wide_search
python -m experiments.fh_s6_max_volume_match --recover-round
python -m experiments.fh_s6_max_volume_match --recover-exotic
python -m experiments.fh_s6_terminal_shooting --recover-round
python -m experiments.fh_s6_terminal_shooting --recover-exotic
python -m experiments.fh_s6_max_volume_scout --dry-run
python -m experiments.fh_s6_terminal_scout --dry-run
python -m experiments.s7.round_validation
python -m experiments.s7.round_recovery_calibration
python -m experiments.s7.squashed_validation
python -m experiments.s7.squashed_recovery_calibration
python -m experiments.s7.right_chart_comparison
python -m experiments.s7.right_endpoint_moduli_probe
python -m experiments.s7.scout_search
python -m experiments.s7.full_moduli_firstjet_scout
python -m experiments.s7.full_moduli_offset_scout
python -m experiments.s7_max_volume_calibration
python -m experiments.s7_max_volume_scout --dry-run
python -m experiments.s7_action_census
python -m experiments.s7_su2_cubed_action_audit
python -m experiments.s7_su2_cubed_defect_audit
python -m experiments.s7_su2_cubed_scout --dry-run
python -m experiments.s7_su2_cubed_tail_defect
python -m experiments.stiefel_feasibility
python -m experiments.aloff_wallach_feasibility
python -m experiments.aloff_wallach_ansatz
python -m experiments.aloff_wallach_endpoint_smoothness
python -m experiments.aloff_wallach_homogeneous_recovery
python -m experiments.aloff_wallach_recovery_calibration --dry-run
python -m experiments.aloff_wallach_scout --dry-run
```

If you prefer not to activate the virtual environment, run:

```zsh
.venv/bin/python -m experiments.berger
.venv/bin/python -m experiments.berger_opposite_end_check
.venv/bin/python -m experiments.doubled_sphere
.venv/bin/python -m experiments.berger_jacobian
.venv/bin/python -m experiments.mirror_search
.venv/bin/python -m experiments.mirror_recovery_calibration
.venv/bin/python -m experiments.mirror_covering_calibration
.venv/bin/python -m experiments.mirror_guarded_covering_search
.venv/bin/python -m experiments.mirror_sweep
.venv/bin/python -m experiments.mirror_sweep_v2
.venv/bin/python -m experiments.mirror_sweep_v3
.venv/bin/python -m experiments.mirror_local_grid_v3
.venv/bin/python -m experiments.non_mirrored_berger_jacobian
.venv/bin/python -m experiments.non_mirrored_recovery_calibration
.venv/bin/python -m experiments.non_mirrored_grid_search
.venv/bin/python -m experiments.non_mirrored_grid_refine
.venv/bin/python -m experiments.berger_branch_audit
.venv/bin/python -m experiments.berger_s3_bundle_action_audit
.venv/bin/python -m experiments.berger_max_volume_calibration
.venv/bin/python -m experiments.berger_max_volume_scout --dry-run
.venv/bin/python -m experiments.non_mirrored_search
.venv/bin/python -m experiments.non_mirrored_surrogate_search
.venv/bin/python -m experiments.non_mirrored_surrogate_wide_search
.venv/bin/python -m experiments.fh_s6_max_volume_match --recover-round
.venv/bin/python -m experiments.fh_s6_max_volume_match --recover-exotic
.venv/bin/python -m experiments.fh_s6_terminal_shooting --recover-round
.venv/bin/python -m experiments.fh_s6_terminal_shooting --recover-exotic
.venv/bin/python -m experiments.fh_s6_max_volume_scout --dry-run
.venv/bin/python -m experiments.fh_s6_terminal_scout --dry-run
.venv/bin/python -m experiments.s7.round_validation
.venv/bin/python -m experiments.s7.round_recovery_calibration
.venv/bin/python -m experiments.s7.squashed_validation
.venv/bin/python -m experiments.s7.squashed_recovery_calibration
.venv/bin/python -m experiments.s7.right_chart_comparison
.venv/bin/python -m experiments.s7.right_endpoint_moduli_probe
.venv/bin/python -m experiments.s7.scout_search
.venv/bin/python -m experiments.s7.full_moduli_firstjet_scout
.venv/bin/python -m experiments.s7.full_moduli_offset_scout
.venv/bin/python -m experiments.s7_max_volume_calibration
.venv/bin/python -m experiments.s7_max_volume_scout --dry-run
.venv/bin/python -m experiments.s7_action_census
.venv/bin/python -m experiments.s7_su2_cubed_action_audit
.venv/bin/python -m experiments.s7_su2_cubed_defect_audit
.venv/bin/python -m experiments.s7_su2_cubed_scout --dry-run
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect
.venv/bin/python -m experiments.stiefel_feasibility
.venv/bin/python -m experiments.aloff_wallach_feasibility
.venv/bin/python -m experiments.aloff_wallach_ansatz
.venv/bin/python -m experiments.aloff_wallach_endpoint_smoothness
.venv/bin/python -m experiments.aloff_wallach_homogeneous_recovery
.venv/bin/python -m experiments.aloff_wallach_recovery_calibration --dry-run
.venv/bin/python -m experiments.aloff_wallach_scout --dry-run
```

`experiments.berger` performs the validated homogeneous Berger check.
`experiments.berger_opposite_end_check` compares the independently marched
Berger solutions at `t = 0.1` and `t = pi/3 - 0.1`, then checks the opposite
endpoint one-jet asymptotics in raw `q`. `experiments.doubled_sphere`
performs the mirrored round-`S^7` candidate experiment; large errors or branch
failure are possible and are part of the mathematical exploration.
`experiments.berger_jacobian` checks local regularity of the Berger mirror-closing
equations, and `experiments.mirror_search` performs a seeded basin search for
other mirror-complete candidates. The search first runs cheap scout evaluations
in deterministic annular regions, then applies staged Newton refinement to the
best seeds and verifies promoted candidates at higher Taylor orders. It may take
several minutes; the output classifies seeds as flowing back to Berger,
finite-order artifacts, branch failures, inconclusive cases, or possible
non-Berger candidates.
`experiments.mirror_recovery_calibration` deliberately perturbs Berger in the
4D mirrored coordinates and asks whether the same Newton/refinement machinery
recovers it. A healthy local recovery table means the refinement step is
working near a known root; broad-box failure does not imply nonexistence, only
that random scouting is unlikely to hit a small basin. If local shells fail to
recover Berger, the search coordinates or continuation strategy need revision
before interpreting negative searches. The calibration writes JSONL checkpoints
under `output/mirror_calibration/`; rerunning the command resumes the newest
compatible unfinished checkpoint, reusing completed seed classifications and
broad-box scout evaluations. The current calibration recipe uses an order-14
correction stage for near-Berger low-order Newton attractors, so finite-order
fake roots are filtered before a seed is counted as recovered.
`experiments.mirror_covering_calibration` is the deterministic follow-up: it
covers `[-1,1]^4` by 10,000 cell-centered scouts, checks the 16 closest grid
points to Berger as an oracle calibration, and separately tests whether blind
scout/contraction selection rediscovers Berger. It writes resumable JSONL output
under `output/mirror_covering_calibration/`.
`experiments.mirror_guarded_covering_search` is the next mirrored non-Berger
search. It keeps the calibrated 10,000-point core covering, adds 40,000
deterministic Halton scouts, probes the best 1,500 with one guarded Newton step,
and refines selected candidates with the two guardrails `m > 0.01` and
`max(|u|, |v|, |r|, |s|) <= 4`. It is intended as a roughly 7-hour terminal run
and writes resumable output under `output/mirror_guarded_covering_search/`.
`experiments.mirror_sweep` is the longer version intended for multi-hour runs:
it sweeps wider annuli, forces per-region refinement quotas, and continuously
writes JSONL checkpoints plus a final summary under `output/mirror_sweeps/`.
`experiments.mirror_sweep_v2` is a deeper follow-up sweep that biases sampling
toward far and tail regions suggested by the first sweep.
`experiments.mirror_sweep_v3` is a multi-hour follow-up with the hard midpoint
floor `m > 0.01`, so it avoids the collapsed-tail regime seen in V2 while
continuing to bias toward negative `u` and negative `v`. If a V3 run stops
during scout evaluation, rerunning the same command resumes the newest
incomplete V3 JSONL checkpoint instead of repeating completed scout seeds.
`experiments.mirror_local_grid_v3` performs a small floor-aware grid refinement
around the best V3 near-floor candidates and verifies the best local grid points
at higher Taylor order.
`experiments.non_mirrored_berger_jacobian` starts the non-mirrored workflow: it
keeps `lambda = 6/sqrt(5)`, allows the total interval length to vary, and reports
the full `8 x 7` two-sided matching Jacobian at Berger.
`experiments.non_mirrored_recovery_calibration` is the corresponding 7D recovery
test: it perturbs Berger on deterministic shells plus a blind local box, runs
staged Gauss-Newton refinement, and reports how far from Berger the non-mirrored
machinery still rediscovers the known solution. It writes resumable JSONL output
under `output/non_mirrored_calibration/`. `experiments.non_mirrored_grid_search`
uses that recovery radius to run a calibrated 7D near-grid scout search with
max grid spacing `0.4`, covering the near box with 103,680 order-4 scout
evaluations. The parent process writes all JSONL checkpoints, while optional
worker processes only evaluate independent scout residuals:

```zsh
.venv/bin/python -m experiments.non_mirrored_grid_search --dry-run
.venv/bin/python -m experiments.non_mirrored_grid_search --workers 4
.venv/bin/python -m experiments.non_mirrored_grid_search --shift cell-center --workers 4
.venv/bin/python -m experiments.non_mirrored_grid_search --region symmetric-alpha-omega --workers 4
.venv/bin/python -m experiments.non_mirrored_grid_search --region positive-ac --workers 4
.venv/bin/python -m experiments.non_mirrored_grid_search --region negative-ac --workers 4
.venv/bin/python -m experiments.non_mirrored_grid_search --region mixed-mu-short --spacing 0.6 --workers 4
.venv/bin/python -m experiments.non_mirrored_grid_search --region mixed-mu-boundary --spacing 0.6 --workers 4
.venv/bin/python -m experiments.non_mirrored_grid_search --region positive-ac-boundary --workers 4
.venv/bin/python -m experiments.non_mirrored_grid_search --region positive-ac-boundary-v2 --workers 4
```

The grid-search checkpoint resumes automatically; use `--no-resume` to force a
fresh output file. The default `--shift vertex` runs the calibrated vertex grid;
`--shift cell-center` runs the midpoint grid inside the same box, using the
centers of the default grid cells. Use `--region symmetric-alpha-omega` for the
Berger follow-up grid that keeps the same log boxes for `a,c,d,f,T` but scans
physical `alpha` and `omega` symmetrically over
`[-sqrt(5)/20, sqrt(5)/20]`, corresponding to
`r_left,r_right in [-3.5, 1.5]`. Use `--region positive-ac` for the exploratory
second real endpoint branch with `a>0`, `c>0`, and `3a-c>0`; this keeps the
same log boxes for the even endpoint scales and the same symmetric physical
`alpha/omega` interval. Use `--region negative-ac` for the remaining implemented
real `ac>0` component with `a<0`, `c<0`, and `3a-c>0`; it uses a base point with
`c/a=12`, keeps the whole standard log box inside the branch, and scans the same
symmetric physical `alpha/omega` interval. Use `--region mixed-mu-short` for the
opposite-`mu` exploratory branch with endpoint-local square-root signs
`left p=(1,1,1)` and `right p=(-1,1,-1)`. It keeps the same standard even
endpoint scale box and symmetric physical `alpha/omega` interval, but restricts
the interval coordinate to `s in [-2.0,-0.4]`, since the branch-valid smoke tests
showed the Berger-length midpoint exits the real branch. This region uses an
order-6 scout config at 50 dps and is intended as an overnight-class terminal
run; the recommended first pass is `--spacing 0.6`, which gives 32,400 scouts
before any follow-up refinement. Use `--region mixed-mu-boundary` for the
broader follow-up strip after the first mixed-mu scout improved at the low-scale
and short-interval boundary:
`u_left,v_left,u_right,v_right in [-1.8,-0.6]`,
`r_left,r_right in [-3.5,1.5]`, and `s in [-3.2,-1.6]`. The recommended first
pass is again `--spacing 0.6`, giving 32,400 order-6 scouts. Use
`--region positive-ac-boundary`
for the focused follow-up strip suggested by the positive-ac vertex and cell-center scouts:
`u_left,v_left,u_right,v_right in [-1.4,-0.2]`,
`r_left,r_right in [-4.3,1.5]`, and `s in [-1.2,0.8]`. Output is written under
`output/non_mirrored_grid_searches/`. Use `--region positive-ac-boundary-v2`
for the next lower strip after the boundary scout improved at the low edge:
`u_left,v_left,u_right,v_right in [-2.2,-1.0]`,
`r_left,r_right in [-4.3,1.5]`, and `s in [-2.0,-0.8]`.
Before launching any opposite-`mu` Berger scout, run the branch probe:

```zsh
.venv/bin/python -m experiments.mu_branch_probe
```

It enumerates the mirrored Berger square-root sign choices and left/right `mu`
choices, writes a summary under `output/berger_mu_branch_probes/`, and reports
both the global-`p` obstruction and the mixed endpoint-local branch used by
`--region mixed-mu-short`.

`experiments.non_mirrored_grid_refine` processes a completed grid-search JSONL
without rerunning the 103,680-scout phase. By default it selects the newest
completed `non-mirrored-grid-v1` checkpoint, chooses a deterministic balanced
batch of 50 promising local minima/asymmetric scouts, and refines them through
the calibrated order-6/order-10/order-14 ladder with order-14 and order-18
verification. It writes resumable JSONL and summary output under
`output/non_mirrored_grid_refinements/`:

```zsh
.venv/bin/python -m experiments.non_mirrored_grid_refine --dry-run
.venv/bin/python -m experiments.non_mirrored_grid_refine
.venv/bin/python -m experiments.non_mirrored_grid_refine --scout-jsonl output/non_mirrored_grid_searches/20260523-174121-seed1729-non-mirrored-grid-v1.jsonl
.venv/bin/python -m experiments.non_mirrored_grid_refine --selection-mode local-minima --scout-jsonl output/non_mirrored_grid_searches/20260523-174121-seed1729-non-mirrored-grid-v1.jsonl
.venv/bin/python -m experiments.non_mirrored_grid_refine --selection-mode local-minima --max-newton-coordinate 4 --scout-jsonl output/non_mirrored_grid_searches/20260530-171855-seed1729-non-mirrored-grid-v1.jsonl
```

Use `--selection-mode local-minima` for the follow-up evidence run over all
canonical branch-valid local minima already present in the completed scout
checkpoint; it does not rerun the 103,680-scout phase. Add
`--local-minimum-max-residual 0.15` or another positive cutoff to refine only
local minima below a chosen scout residual. Use `--max-newton-coordinate 4`
when refining the symmetric physical `alpha/omega` grid, since that scout box
intentionally extends to `r_left,r_right = -3.5`.
`experiments.berger_branch_audit` summarizes the curated Berger scout/refinement
artifacts, regenerates the `mu`/square-root branch coverage data, and writes the
closure report:

```zsh
.venv/bin/python -m experiments.berger_branch_audit
.venv/bin/python -m experiments.berger_branch_audit --write-markdown docs/berger-branch-audit.md
.venv/bin/python -m experiments.berger_branch_audit --write-json output/berger_branch_audits
```

`experiments.berger_s3_bundle_action_audit` checks the tempting
Grove-Ziller `S3`-bundle route to another Berger action. It verifies that
the Berger topology matches the `M_{6,4}` sphere-bundle representative, but
also records the obstruction: Grove-Ziller's cohomogeneity-one action is on
the ten-dimensional principal `SO(4)` bundle, while the induced action on the
seven-dimensional associated `S3`-bundle is only cohomogeneity four:

```zsh
.venv/bin/python -m experiments.berger_s3_bundle_action_audit
.venv/bin/python -m experiments.berger_s3_bundle_action_audit --json
```

`experiments.non_mirrored_search` runs the corresponding JSONL-checkpointed
seeded search with independent left and right endpoint data; output is written
under `output/non_mirrored_searches/`.
`experiments.non_mirrored_surrogate_search` first evaluates 5000 true cheap
Taylor solves, trains XGBoost branch/residual surrogates, uses them only to
propose further points, and then sends the best verified proposals back through
the real Newton/refinement pipeline. Its output is written under
`output/non_mirrored_surrogate/`. `experiments.non_mirrored_surrogate_wide_search`
is the wider V2 follow-up: it evaluates 20,000 new true cheap labels, adds
signed residual-vector surrogate heads, forces more asymmetric region quotas,
and uses timeout-guarded refinement. Its output is written under
`output/non_mirrored_surrogate_wide/`.
`experiments.fh_s6_max_volume_match` is the Foscolo-Haskins nearly Kahler
`S^6` benchmark. It integrates the `S^2`-closing and `S^3`-closing singular
families to their maximal-volume principal orbit, matches the resulting
hyperboloid coordinates up to the FH reflection symmetries, and writes JSONL
output under `output/fh_s6_max_volume_matches/`:

```zsh
.venv/bin/python -m experiments.fh_s6_max_volume_match --recover-round
.venv/bin/python -m experiments.fh_s6_max_volume_match --recover-exotic
.venv/bin/python -m experiments.fh_s6_max_volume_match --evaluate --a 1.7320508075688772 --b 1.5
```

This is the stable FH-style matching calibration; a direct terminal-singular
BVP comparison is available as `experiments.fh_s6_terminal_shooting`. It marches
from both singular Taylor seeds to an interior slice, applies an explicit FH
terminal symmetry to the `S^2` side, and matches all seven ODE variables by
damped Gauss-Newton:

```zsh
.venv/bin/python -m experiments.fh_s6_terminal_shooting --recover-round
.venv/bin/python -m experiments.fh_s6_terminal_shooting --recover-exotic
.venv/bin/python -m experiments.fh_s6_terminal_shooting --evaluate --a 1.7320508075688772 --b 1.5 --match-time 0.77289845 --transform round-terminal
```

This is the deliberately naive comparison with the G2-style terminal shooting
strategy. It is not yet a blind rediscovery scout, but it verifies that the
terminal-singular formulation can recover both known FH `S^6` solutions from
nearby guesses.
For blind-ish overnight rediscovery runs, use the scout commands. The
max-volume scout covers `a,b in [0.25,2.4]` in log coordinates, while the
terminal scout covers the same `a,b` range plus `match_time in [0.35,1.65]`
and both implemented terminal symmetries:

```zsh
.venv/bin/python -m experiments.fh_s6_max_volume_scout --dry-run
.venv/bin/python -m experiments.fh_s6_max_volume_scout --workers 4
.venv/bin/python -m experiments.fh_s6_max_volume_scout --spacing 0.002 --workers 4
.venv/bin/python -m experiments.fh_s6_terminal_scout --dry-run
.venv/bin/python -m experiments.fh_s6_terminal_scout --workers 4
.venv/bin/python -m experiments.fh_s6_terminal_scout --spacing 0.02 --workers 4
```

Both scouts write resumable JSONL output and summaries under
`output/fh_s6_max_volume_scouts/` and `output/fh_s6_terminal_scouts/`,
respectively. Rerunning the same command resumes the newest compatible
incomplete checkpoint; use `--no-resume` to force a new file. The summaries list
the best residuals and nearest-neighbor local minima, but do not use the known
FH root locations for selection. The default spacings are useful first-pass
checks; the two explicit `--spacing` commands above are the intended overnight
rediscovery grids.

`experiments.berger_max_volume_calibration` and
`experiments.s7_max_volume_calibration` are the G2 analogue of the
Foscolo-Haskins maximal-volume matching test. Each side is marched only until
the principal-orbit volume is stationary; the matching residual compares the raw
`q` states there, and the original interval length is reconstructed as
`tau_left + tau_right` rather than imposed as a midpoint condition:

```zsh
.venv/bin/python -m experiments.berger_max_volume_calibration
.venv/bin/python -m experiments.s7_max_volume_calibration
```

For blind follow-up scouts, the Berger runner searches six endpoint coordinates
because the interval is reconstructed from the two maximal-volume events. The S7
runner searches the existing fixed-right-chart 3D boxes for the round `p_3` and
squashed `p_2` targets:

```zsh
.venv/bin/python -m experiments.berger_max_volume_scout --dry-run --region near --spacing 0.4
.venv/bin/python -m experiments.berger_max_volume_scout --region near --spacing 0.4 --workers 4
.venv/bin/python -m experiments.berger_max_volume_scout --region symmetric-alpha-omega --spacing 0.4 --workers 4
.venv/bin/python -m experiments.s7_max_volume_scout --dry-run --spacing 0.075
.venv/bin/python -m experiments.s7_max_volume_scout --spacing 0.075 --workers 4
.venv/bin/python -m experiments.s7_max_volume_scout --region positive-ac --spacing 0.075 --workers 4
```

These scouts write resumable JSONL output and summaries under
`output/berger_max_volume_scouts/` and `output/s7_max_volume_scouts/`.
Rerunning the same command resumes the newest compatible incomplete checkpoint,
and the summaries list best scouts plus nearest-neighbor local minima.

`experiments.stiefel_feasibility` records the current status of the standard
Stiefel manifold `V_{5,2}=SO(5)/SO(3)`. It verifies the known homogeneous
nearly parallel algebraic parameter conditions, but also shows that the natural
`SO(4)` action on `V_{5,2}` is cohomogeneity two, not cohomogeneity one. Thus
standard Stiefel is not currently a valid endpoint-chart target for the
`SO(4)/Z_2^2` `q_i` marcher:

```zsh
.venv/bin/python -m experiments.stiefel_feasibility
.venv/bin/python -m experiments.stiefel_feasibility --json
```

`experiments.aloff_wallach_feasibility` performs the analogous first-pass audit
for Aloff-Wallach spaces. Generic `N_{k,l}=SU(3)/S^1_{k,l}` has known
homogeneous nearly parallel `G_2` structures, but the calibrated `SU(3)` action
is transitive and the usual proper connected subgroups are too small for
six-dimensional principal orbits. The exceptional `N_{1,1}` case remains
promising: the audit records the fibration `SO(3) -> N_{1,1} -> CP^2` and the
candidate `SO(3)_real x SO(3)_fiber` cohomogeneity-one action. This is not
ready for the current `SO(4)/Z_2^2` `q_i` equations; it needs a new invariant
form basis and endpoint charts first.

```zsh
.venv/bin/python -m experiments.aloff_wallach_feasibility
.venv/bin/python -m experiments.aloff_wallach_feasibility --json
```

`experiments.aloff_wallach_ansatz` is the next calibration checkpoint for
`N_{1,1}`. It records the `SO(3)_real x SO(3)_fiber` cohomogeneity-one route,
uses the Ball-Oliveira `A,B,C,D` homogeneous form conventions, computes the
`SU(3)/U(1)_{1,1}` exterior derivative from the matrix bracket, and verifies the
known tri-Sasakian-metric and squashed nearly parallel structures. It also
records the full model principal-orbit invariant `SU(3)` variables: 7 `omega`
coefficients, 12 `gamma` coefficients, the two explicit
`omega wedge gamma = 0` equations, and the Hitchin volume normalization. The two
`A,B,C,D` known structures pass these model principal-orbit algebraic checks.
The extra Sasaki-Einstein nearly parallel form outside the `A,B,C,D` family is
converted from Geipel's Sasaki-Einstein source coframe to Ball-Oliveira
coordinates, verified by `d phi = 4 psi`, and checked in the same model
principal-orbit `SU(3)` variables.

```zsh
.venv/bin/python -m experiments.aloff_wallach_ansatz
.venv/bin/python -m experiments.aloff_wallach_ansatz --json
```

`experiments.aloff_wallach_endpoint_smoothness` derives the first singular
endpoint restrictions for the same full 19-variable chart.  It splits
`base_3/fiber_3` into the collapsing circle and surviving singular-orbit
direction, applies the Eschenburg-Wang equivariant-polynomial test at the
`RP^2` and null-conic `CP^1` endpoints, and records the resulting weighted
smooth jet dimensions.  Smoothness alone reduces the 19 raw endpoint values to
four zeroth-order constants; the ODE and algebraic `SU(3)` constraints still
need to be imposed before choosing scout coordinates.

```zsh
.venv/bin/python -m experiments.aloff_wallach_endpoint_smoothness
.venv/bin/python -m experiments.aloff_wallach_endpoint_smoothness --json
```

`experiments.aloff_wallach_homogeneous_recovery` is the direct calibration for
the `SO(3)_real x SO(3)_fiber` action model.  It constructs the exact
homogeneous trajectory in the action basis, uses the derived product
Maurer-Cartan scales `base=-1`, `fiber=-2`, and integrates from both ends to an
interior slice.  In this action the squashed `A,B,C,D` structure is recovered,
while the tri-Sasakian `A,B,C,D` form is rejected because it is not invariant
under the selected right-fiber `SO(3)` action:

```zsh
.venv/bin/python -m experiments.aloff_wallach_homogeneous_recovery
.venv/bin/python -m experiments.aloff_wallach_homogeneous_recovery --targets squashed --no-write
```

`experiments.aloff_wallach_recovery_calibration` is the known-solution
calibration layer for the experimental `N_{1,1}` endpoint chart.  It starts
from the verified tri-Sasakian and squashed homogeneous `A,B,C,D` structures,
rescales them to the requested normalized `lambda`, and tests deterministic
left/right endpoint sign branches with resumable JSONL output under
`output/aloff_wallach_recovery_calibrations/`.  This is the Aloff-Wallach
analogue of first asking whether the numerical pipeline can rediscover the
known solutions before trusting broad scouts.

```zsh
.venv/bin/python -m experiments.aloff_wallach_recovery_calibration --dry-run
.venv/bin/python -m experiments.aloff_wallach_recovery_calibration --branch-mode all-signs --scale-multipliers 0.75,1,1.25 --structure-scales 0.5,1,2 --workers 4
```

`experiments.aloff_wallach_scout` is the first endpoint-reduced
maximal-volume scout for this topology.  It searches the eight smooth endpoint
constants

```text
left_A, left_B, left_C, left_D, right_A, right_B, right_C, right_D
```

coming from the `RP^2` and null-conic `CP^1` endpoint smoothness charts.  The
higher endpoint Taylor layers are not exposed as scout coordinates; they are
fitted internally against the invariant `SU(3)` nearly-parallel evolution
equations, then each side is marched to a maximal-volume orbit and compared.
This is intentionally more experimental than the calibrated Berger/S7 runners:
it is the first plumbing check for the new Aloff-Wallach ODE layer, and failed
germ or march attempts are ordinary scout classifications.

```zsh
.venv/bin/python -m experiments.aloff_wallach_scout --dry-run
.venv/bin/python -m experiments.aloff_wallach_scout --radius 1 --spacing 1 --workers 4
```

`experiments.s7.round_validation` validates the derived round-`S^7` oracle using
the fixed-data `p_3` right endpoint chart. The right endpoint is not the Berger
chart: it has constant offset proportional to `(1,-1,-2,2,-2,2,19,-19)` and
weights `(2,2,1,1,1,1,2,2)`.
`experiments.s7.squashed_validation` validates the derived squashed-`S^7`
oracle using the new fixed-data `p_2` right endpoint chart.  The right endpoint
is not the Berger chart: it has constant offset proportional to
`(1,-2,-1,2,-2,19,2,-19)` and weights `(2,1,2,1,1,2,1,2)`.
`experiments.s7.right_chart_comparison` marches both Berger and round-`S^7`
left-end data toward `t = pi/3` and compares the observed terminal asymptotics
with the Berger right chart.
`experiments.s7.right_endpoint_moduli_probe` checks whether the fixed S7
`p_2`/`p_3` right charts have enough parameterized Taylor data for an honest
full-moduli scout. It validates that the explicit round/squashed homogeneous
right seeds still solve the weighted equations, then disables the explicit
homogeneous-series shortcut to compare the old one-layer recurrence with a
global endpoint coefficient solve:

```zsh
.venv/bin/python -m experiments.s7.right_endpoint_moduli_probe
.venv/bin/python -m experiments.s7.right_endpoint_moduli_probe --write-json
```

At present this is a readiness probe rather than a long scout: the fixed
homogeneous right seeds are valid and the global coefficient solve removes the
local Taylor residual, but a parameterized S7 `p_2`/`p_3` right-coordinate family
is still needed before we should run a Berger-style 7D S7 search.
`experiments.s7.round_recovery_calibration` and
`experiments.s7.squashed_recovery_calibration` are light 3D recovery tests
around the two known fixed-chart solutions. They perturb the left endpoint
coordinates `(u,v,r)` with the fixed right chart and interval held at the known
value, then run a cheap
order-8/order-10/order-14 Newton ladder to confirm that the known round and
squashed solutions are recoverable. Output is written under
`output/s7_recovery_calibration/`.
`experiments.s7.scout_search` is the first long S7 scout grid. It searches both
the round `p_3` and squashed `p_2` fixed right charts by default, using the
3D box `u,v in [-1.2,1.2]` and `r in [-2.5,2.5]` with max grid spacing
`0.075`, for 148,104 order-6 scout evaluations:

```zsh
.venv/bin/python -m experiments.s7.scout_search --dry-run
.venv/bin/python -m experiments.s7.scout_search --workers 4
.venv/bin/python -m experiments.s7.scout_search --targets round --workers 4
.venv/bin/python -m experiments.s7.scout_search --targets squashed --workers 4
.venv/bin/python -m experiments.s7.scout_search --region positive-ac --workers 4
```

The scout output is written under `output/s7_scout_searches/`.
Rerunning the same scout command resumes the newest compatible incomplete
checkpoint; use `--no-resume` to force a fresh run.
Use `--region positive-ac` for the follow-up S7 scout on the other implemented
real left-end branch. It keeps the same fixed right charts, sets
`a = a0 exp(u)`, `c = 3a rho` with `rho in [0.05,0.4]`, and scans
`alpha = (sqrt(5)/50) r` symmetrically over
`[-7 sqrt(5)/100, 7 sqrt(5)/100]`. With the default spacing this is an
37,620-point order-6 scout over both targets.
`experiments.s7.full_moduli_firstjet_scout` is the first two-ended S7 follow-up.
It keeps the S7 terminal offset fixed, varies the left endpoint coordinates,
three right first-jet germ coordinates, and the interval scale, then solves a
numerical right-end Taylor germ before each order-6 scout march. This is a
7D evidence-gathering scout, not yet a closed-form p2/p3 offset-moduli search:

```zsh
.venv/bin/python -m experiments.s7.full_moduli_firstjet_scout --dry-run
.venv/bin/python -m experiments.s7.full_moduli_firstjet_scout --workers 4
.venv/bin/python -m experiments.s7.full_moduli_firstjet_scout --targets round --workers 4
.venv/bin/python -m experiments.s7.full_moduli_firstjet_scout --targets squashed --workers 4
```

The default grid has `4^7` seeds per target, or 32,768 seeds over both targets,
and writes JSONL output under `output/s7_full_moduli_firstjet_scouts/`. In
restricted environments the process pool falls back to threads; for normal
terminal runs the process executor is preferred.
`experiments.s7.full_moduli_offset_scout` is the honest S7 full-moduli scout
using the derived p2/p3 terminal-offset charts. It varies the left endpoint
coordinates, the right endpoint offset moduli `(A,B,C)`, and the interval scale,
then solves the right Taylor coefficient block before each order-6 scout march:

```zsh
.venv/bin/python -m experiments.s7.full_moduli_offset_scout --dry-run
.venv/bin/python -m experiments.s7.full_moduli_offset_scout --workers 4
.venv/bin/python -m experiments.s7.full_moduli_offset_scout --targets round --workers 4
.venv/bin/python -m experiments.s7.full_moduli_offset_scout --targets squashed --workers 4
```

The default grid has `4^7` seeds per target, or 32,768 seeds over both targets,
and writes JSONL output under `output/s7_full_moduli_offset_scouts/`.
`experiments.s7.full_moduli_offset_refine` processes a completed terminal-offset
scout checkpoint without rerunning the 7D grid. It selects target-wise local
minima, runs damped Gauss-Newton in the seven scaled coordinates, and verifies
at orders 14 and 18:

```zsh
.venv/bin/python -m experiments.s7.full_moduli_offset_refine --dry-run
.venv/bin/python -m experiments.s7.full_moduli_offset_refine
.venv/bin/python -m experiments.s7.full_moduli_offset_refine --max-residual 0.1
.venv/bin/python -m experiments.s7.full_moduli_offset_refine --scout-jsonl output/s7_full_moduli_offset_scouts/20260615-172112-seed1729-s7-full-moduli-offset-scout-v1.jsonl
```

The refinement output is written under
`output/s7_full_moduli_offset_refinements/`; rerunning the same command resumes
the newest compatible incomplete checkpoint and skips already classified seeds.

`experiments.s7.full_moduli_firstjet_refine` processes a completed full-moduli
first-jet scout checkpoint without rerunning the 7D grid. It selects the
target-wise nearest-neighbor local minima and re-evaluates those same 7D points
at orders 8, 10, and 14 with fresh finite-order calibration:

```zsh
.venv/bin/python -m experiments.s7.full_moduli_firstjet_refine --dry-run
.venv/bin/python -m experiments.s7.full_moduli_firstjet_refine --workers 4
.venv/bin/python -m experiments.s7.full_moduli_firstjet_refine --orders 8,10 --workers 4
.venv/bin/python -m experiments.s7.full_moduli_firstjet_refine --scout-jsonl output/s7_full_moduli_firstjet_scouts/20260614-170152-seed1729-s7-full-moduli-firstjet-scout-v1.jsonl --workers 4
```

The diagnostic output is written under
`output/s7_full_moduli_firstjet_refinements/`.

`experiments.s7_action_census` records the next S7 action-selection audit after
the Podesta large-`|a|` proof route stalled.  It lists candidate
cohomogeneity-one actions, filters already-tested and dimensionally unsuitable
actions, computes invariant principal-orbit 2-form and 3-form dimensions, checks
known round/squashed visibility, and ranks the viable next scouting targets:

```zsh
.venv/bin/python -m experiments.s7_action_census
.venv/bin/python -m experiments.s7_action_census --write-markdown docs/s7-action-census.md
```

The current recommendation is the intermediate
`Sp(1) x Sp(1) x U(1)` action on the same `S3 x S3` orbit foliation as
Podesta's `Sp(1)^3` action: it keeps round and squashed calibration visibility,
strictly weakens the already-tested symmetry, and has a moderate 13-function
invariant `dt^omega + gamma` chart.

`experiments.s7_su2_cubed_action_audit` checks the separate Podesta
`SU(2)^3` cohomogeneity-one action on `S7`. Its principal orbit is
`S3 x S3`, both singular orbits are `S3`, and the compact group diagram is
exactly the unit sphere `S7` in `H^2`. The command records the five-function
invariant form chart, endpoint smoothness conditions, and verifies both round
and squashed `S7` in that chart:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_action_audit
.venv/bin/python -m experiments.s7_su2_cubed_action_audit --json
```

This is a readiness check for a dedicated Podesta-system marcher, not a long
scout yet.

`experiments.s7_su2_cubed_scout` is the first dedicated Podesta-system marcher
for that action. It fixes `lambda=1`, uses the one-parameter left singular
`S3` endpoint data, detects standard `K-` terminal closure, and scans the scalar
parameter `a`. The direct homogeneous terminal checks are `a=-36` for round
`S7` and `a=108/5` for squashed `S7`; the positive round value is equivalent by
Podesta's outer automorphism but does not close in the unmodified standard
terminal chart.

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_scout --recover-known
.venv/bin/python -m experiments.s7_su2_cubed_scout --dry-run
.venv/bin/python -m experiments.s7_su2_cubed_scout
```

The scout writes JSONL and summary output under `output/s7_su2_cubed_scouts/`.

`experiments.s7_su2_cubed_tail_defect` tests a scale-free shooting defect for
excluding large `|a|` in the same Podesta family. If `T_a` is the first zero of
`f0`, standard `K-` closure requires `f2(T_a)=0`; after scaling
`h0=a x0`, `h2=a^3 x2`, the exact tail defect is
`X2(a)=x2(T_a)`. The command compares finite large-`|a|` samples with the
large-`|a|` limiting ODE:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --json
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --tube-check --tube-a 100000000 --tube-subdivisions 2,2,1,2
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --segmented-tube-check --tube-a 100000000 --tube-subdivisions 2,2,1,2
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --tuned-tube-check --tube-a 100000000 --tube-start 2.0 --tube-end 2.75 --tube-step 0.001 --tube-block-steps 1 --tuned-tube-max-attempts 60 --tuned-tube-max-growth 0.5,5.0,0.5,0.5
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --restart-tuned-chain-check --tube-a 100000000 --tube-start 2.0 --tube-end 2.95 --tube-step 0.001 --restart-interval 0.05 --tuned-tube-max-attempts 200 --tuned-tube-max-growth 2,20,20,2
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --restart-tuned-chain-check --tube-a 100000000 --tube-start 2.0 --tube-end 3.1 --tube-step 0.001 --restart-interval 0.05 --tuned-tube-max-attempts 450 --tuned-tube-max-growth 20,200,50,10
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --p-tube-check --tube-a 100000000 --p-tube-start 0.305 --p-tube-end 0.25 --p-tube-step 0.00005
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --p-tube-check --p-tube-asymmetric-profiles --tube-a 100000000 --p-tube-start 0.305 --p-tube-end 0.2 --p-tube-step 0.00005
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --tuned-p-tube-check --tube-a 100000000 --p-tube-entry-time 2.6 --p-tube-start 0.65 --p-tube-end 0.423 --p-tube-step 0.0005 --tuned-p-tube-max-growth 20,200,2,50 --tuned-p-tube-max-attempts 120
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --staged-union-p-tube-check --tube-a 100000000 --p-tube-start 0.423 --p-tube-step 0.0005 --tuned-p-tube-max-growth 20,200,2,50 --tuned-p-tube-max-attempts 120
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --adaptive-union-p-tube-check --tube-a 100000000 --p-tube-start 0.4 --p-tube-end 0.395 --p-tube-step 0.0005 --tuned-p-tube-max-growth 20,200,2,50 --tuned-p-tube-max-attempts 120 --adaptive-union-max-depth 2
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --adaptive-union-p-tube-check --adaptive-union-p-tube-source-json output/s7_tail_proof/adaptive_union_p_tube_0.4_to_0.395.json --tube-a 100000000 --p-tube-start 0.395 --p-tube-end 0.39 --p-tube-step 0.0005 --tuned-p-tube-max-growth 20,200,2,50 --tuned-p-tube-max-attempts 120 --adaptive-union-max-depth 3 --p-tube-cancellation-prime
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --adaptive-union-p-tube-check --adaptive-union-p-tube-source-json output/s7_tail_proof/adaptive_union_p_tube_0.395_to_0.39_cprime_splitq.json --tube-a 100000000 --p-tube-start 0.39 --p-tube-end 0.3885 --p-tube-step 0.0005 --tuned-p-tube-max-growth 20,200,2,50 --tuned-p-tube-max-attempts 120 --adaptive-union-max-depth 3 --p-tube-cancellation-prime
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --adaptive-carried-c-union-p-tube-check --adaptive-union-p-tube-source-json output/s7_tail_proof/adaptive_union_p_tube_0.395_to_0.39_cprime_splitq.json --tube-a 100000000 --p-tube-start 0.39 --p-tube-end 0.3895 --p-tube-step 0.0005 --tuned-p-tube-max-attempts 140 --adaptive-union-max-depth 5
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --sampled-carried-c-p-tube-check --sampled-carried-c-p-tube-profile-set tight --sampled-carried-c-p-tube-end 0.3255 --tuned-p-tube-max-attempts 120
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --carried-c-p-tube-from-box-check --carried-c-p-tube-source-json output/s7_tail_proof/sampled_carried_c_p_tube_0.65_to_0.3255_tight_attempts120.json --carried-c-p-tube-start 0.3255 --carried-c-p-tube-end 0.32 --carried-c-p-tube-step 0.0001 --carried-c-p-tube-initial-growth 0.01,0.2,0.002,0.02,0.02 --carried-c-p-tube-max-growth 20,200,5,50,10 --tuned-p-tube-max-attempts 200 --carried-c-p-tube-subdivisions 1,1,1,2,1
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --automatic-carried-c-p-corridor-check --carried-c-p-corridor-end 0.32
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --carried-c-p-wall-check --carried-c-p-wall-component 2 --carried-c-p-wall-side lower --carried-c-p-wall-value 0 --carried-c-p-wall-end 0.005 --carried-c-p-wall-source-json output/s7_tail_proof/automatic_carried_c_p_corridor_0.298_to_0.29_from_tight_sampled.json
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --x2-zero-factor-check
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --p-corridor-check --tube-a 100000000 --p-corridor-start 0.25 --p-corridor-end 0.2 --p-corridor-step 0.0005
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --p-corridor-tune --tube-a 100000000 --p-corridor-start 0.25 --p-corridor-end 0.2 --p-corridor-step 0.0005 --p-corridor-tune-x2-slopes=0.02505,0.025075,0.0251 --p-corridor-tune-x1-upper-slopes=-150,-151 --p-corridor-tune-max-runs 6
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --terminal-takeover-check --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --frontier-continuation-check --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --hybrid-handoff-check --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --p-start-slice-bridge-check --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --taylor-even-parity-audit --tube-a 100000000 --taylor-p-slice-tail-order 40 --taylor-p-slice-tail-working-dps 80 --taylor-p-slice-b-samples 3 --taylor-b-cauchy-enclosure-radius 4e-7 --taylor-b-cauchy-enclosure-samples 8
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --taylor-even-s-series-audit --taylor-ratio-profile-b-mode limit --tube-a 100000000 --taylor-p-slice-tail-order 160 --taylor-p-slice-tail-start 50 --taylor-p-slice-tail-working-dps 140 --taylor-ratio-profile-circle-radius 3.5 --taylor-ratio-profile-circle-ratio-bound 0.95 --taylor-ratio-profile-p-slice-ratio-bound 0.6
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --taylor-recurrence-forcing-audit --tube-a 100000000 --taylor-p-slice-tail-order 60 --taylor-p-slice-tail-start 50 --taylor-p-slice-b-samples 3 --taylor-p-slice-tail-working-dps 90 --taylor-ratio-profile-circle-radius 3.5 --taylor-ratio-profile-circle-ratio-bound 0.95
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --taylor-ratio-profile-audit --taylor-ratio-profile-b-mode limit --tube-a 100000000 --taylor-p-slice-tail-order 160 --taylor-p-slice-ratio-start 50 --taylor-p-slice-tail-working-dps 140 --taylor-ratio-profile-circle-radius 3.5 --taylor-ratio-profile-circle-ratio-bound 0.95 --taylor-ratio-profile-p-slice-ratio-bound 0.6
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --taylor-geometric-envelope-audit --taylor-ratio-profile-b-mode limit --tube-a 100000000 --taylor-p-slice-tail-order 160 --taylor-p-slice-tail-start 50 --taylor-p-slice-tail-working-dps 140 --taylor-ratio-profile-circle-radius 3.5 --taylor-ratio-profile-circle-ratio-bound 0.95 --taylor-ratio-profile-p-slice-ratio-bound 0.6
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --taylor-p-slice-required-a-audit --tube-a 100000000 --taylor-p-slice-tail-order 40 --taylor-p-slice-tail-start 30 --taylor-p-slice-ratio-start 25 --taylor-p-slice-ratio-bound 0.6 --taylor-p-slice-b-samples 3 --taylor-p-slice-tail-working-dps 80 --taylor-b-cauchy-radius 1e-7 --taylor-b-cauchy-samples 8 --taylor-b-cauchy-outer-radius 2e-7 --taylor-b-cauchy-outer-samples 8 --taylor-b-cauchy-enclosure-radius 4e-7 --taylor-b-cauchy-enclosure-samples 8 --taylor-b-cauchy-skip-direct
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --taylor-start-block-check --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --taylor-time-bridge-check --taylor-bridge-end 0.01 --taylor-progress-every-blocks 20 --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --taylor-frontier-continuation-check --taylor-bridge-end 0.01 --taylor-frontier-end 0.011 --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --taylor-restart-chain-check --taylor-bridge-end 0.01 --taylor-restart-end 0.012 --taylor-restart-interval 0.001 --tube-step 0.001 --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --taylor-time-bridge-check --taylor-bridge-end 2.0 --taylor-progress-every-blocks 250 --json > output/s7_tail_proof/taylor_bridge_2.0.json 2> output/s7_tail_proof/taylor_bridge_2.0.log
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --taylor-restart-chain-check --taylor-bridge-end 2.0 --taylor-restart-end 2.6 --taylor-restart-bridge-json output/s7_tail_proof/taylor_bridge_2.0.json --taylor-restart-progress-every-segments 1 --tube-step 0.001 --tube-a 100000000 --json > output/s7_tail_proof/taylor_restart_2.0_to_2.6_retry.json 2> output/s7_tail_proof/taylor_restart_2.0_to_2.6_retry.log
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --regular-time-corridor-check --regular-time-start 0.5 --regular-time-end 0.6 --regular-time-step 0.001 --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --x3-zero-wall-check --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --late-x3-descent-check --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --piecewise-corridor-check --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --late-tail-closure-check --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --broad-tail-closure-check --tube-a 100000000
.venv/bin/python -m experiments.s7_su2_cubed_tail_defect --support-tail-closure-check --tube-a 100000000
```

`experiments.s7_su2_cubed_defect_audit` is the broader follow-up to the
single-`X2` tail-defect attempt. It evaluates a registry of necessary scalar
defects (`x3`, `x2`, `C`, positive endpoint norms, linear combinations,
polynomial numerator defects, and endpoint-equivalent integrating-factor
defects), ranks them by known-solution calibration and large-`|a|` behavior,
then records the current bounded `x3` uniform-exclusion attempt:

```zsh
.venv/bin/python -m experiments.s7_su2_cubed_defect_audit
.venv/bin/python -m experiments.s7_su2_cubed_defect_audit --write-markdown docs/s7-su2-cubed-defect-audit.md
.venv/bin/python -m experiments.s7_su2_cubed_defect_audit --write-json output/s7_su2_cubed_defect_audits/audit.json --write-csv output/s7_su2_cubed_defect_audits/defects.csv
```

The optional `--tube-check` mode runs a conditional interval moving-tube face
check for the late tail segment. It is a proof-building diagnostic: success
certifies the displayed end box assuming the displayed start box has already
been validated. The `--segmented-tube-check` variant chains local tubes and
reports the certified frontier. The `--tuned-tube-check` variant uses the failed
face diagnostics to tune local centered-tube growth; the displayed command
certifies the conditional bridge `t=2.0 -> 2.75` with worst margin about
`3.77e-7`. The `--restart-tuned-chain-check` variant explicitly contains
carried boxes in fresh centered restart boxes; the displayed command certifies
to `t=2.936` before failing on the lower `x2` face, and the stronger growth-cap
variant certifies to `t=3.021` before the lower `x1`/`x3` faces become too wide
for this rectangular tube strategy. The `--p-tube-check` variant switches to
`p=x0` as the independent variable for the singular terminal tail. The
`--p-tube-asymmetric-profiles` option is an experimental variant that lets
lower and upper tube faces widen at different rates. The `--tuned-p-tube-check`
variant applies the same failed-face tuning in p-time; the displayed early
`p=0.65 -> 0.423` command is conditional on validating the `p=0.65` slice, and
moves the transformed-time frontier to the natural `x3` transition region
before a lower-`t`/lower-`x2` face failure. The `--staged-union-p-tube-check`
variant then partitions the certified `p=0.423` source box into 16 midpoint
centred p-time tubes and certifies all branches to `p=0.4` with worst margin
about `3.95e-6`. The `--adaptive-union-p-tube-check` mode continues a saved
finite union by splitting only failed boxes. The optional
`--p-tube-cancellation-prime` flag sharpens the transformed denominator with
`C=x1*x2-p^2*x3/6` on boxes with `x2>=0`; using it certifies the saved
finite-union frontier to `p=0.3885`. Attempts below that point show the present
four-dimensional rectangular p-time tubes losing the `C` correlation, so the
next proof step is to carry `C` as a state/corridor variable rather than just
increase split depth. The `--adaptive-carried-c-union-p-tube-check` mode is the
first version of that idea: it adds `C` as a fifth variable and constrains the
denominator with the intersection of carried and algebraic `C` intervals. It
also sharpens the outgoing `C` handoff by intersecting with the algebraic
end-slice interval. With that handoff, the carried-`C` chain now certifies past
the old failed target, from `p=0.39` down to `p=0.377`. The segment below
`p=0.3805` needs a wider local growth profile plus a graph-prime denominator
sharpening that intersects the carried-`C` and expanded `p'` formulas. The next
quarter-step `p=0.377 -> 0.37675` is not yet certified by the current
rectangular p-time boxes; diagnostics point to renewed component-0 branching,
not to a sampled trajectory approaching terminal closure. The
`--sampled-carried-c-p-tube-check` command is a complementary narrow-chain
diagnostic: with the default profiles it certifies the conditional sampled tube
from `p=0.65` to `p=0.325` for `A=100000000`, though the end box is still too
broad to finish the terminal proof. The
`--sampled-carried-c-p-tube-profile-set tight --tuned-p-tube-max-attempts 120`
variant reproduces the sharper `p=0.3255` box. The
`--carried-c-p-tube-from-box-check` command continues a tuned carried-`C`
p-time tube from such a saved five-dimensional certificate, giving a
reproducible way to test whether the tight source box can be pushed farther
without falling back to the broader affine corridor. The
`--automatic-carried-c-p-corridor-check` command checks the corresponding
automatic affine p-corridor while carrying `C`; from that sharper source it now
reaches `p=0.29` with `x3_high<0`. The `--carried-c-p-wall-check` mode then
certifies the `x2=0` lower wall from `p=0.29` to `p=0.005`; the remaining
large-tail gap is the `x3` wall/tiny-p closure. The
`--x2-zero-factor-check` command records the exact finite-`A` factorization of
`x2'` on `x2=0`, certifying the ordinary-time `x2` wall down to `p=3.5e-4`
for the current late `x3` range. The `--p-corridor-check`
mode checks an affine barrier corridor from the certified p-tube frontier. The
`--p-corridor-tune` mode scans the active corridor slopes around the current
lower-`x2`/upper-`x1` bottleneck. The `--terminal-takeover-check` mode verifies
the conditional late `x3=-0.5` wall certificate down to `p=0`, assuming the
coarse late box and positive `x2` floor are preserved. The
`--frontier-continuation-check` mode continues the certified p-tube frontier
from `p=0.25` to `p=0.23595`. The `--hybrid-handoff-check` mode certifies a
separate p-time handoff from the `p=0.325` slice to a broader `p=0.25` frontier
box
`t in [3.545,3.60]`, `x1 in [6.35,8.30]`, `x2 in [0.0052,0.012]`,
`x3 in [-0.78,-0.58]`, conditional on validating that `p=0.325` start slice.
`--broad-tail-closure-check` composes that hybrid handoff with an automatic
760-step p-time affine corridor to `p=0.212` and the terminal `x3=-0.6` wall;
it conditionally excludes closure from the `p=0.325` start slice down to
`p=0` for `|a| >= 100000000`.
`--p-start-slice-bridge-check` verifies that the t-time support tube crosses
into that `p=0.325` start slice between `t=3.5055` and `t=3.5056`.
`--taylor-start-block-check` uses the explicit smooth singular-end expansion
`x=(1,27/4,-1/27,3)+c2(1/a)t^2+O(t^4)` to certify the first slab
`t=0.001 -> 0.00105`, conditional on the Taylor remainder lying in the displayed
`1e-8` start radius.
`--taylor-time-bridge-check`, `--taylor-frontier-continuation-check`, and
`--taylor-restart-chain-check` compose that Taylor start with tuned
ordinary-time face barriers. The restart-chain variant preserves the Taylor
conditional across centered restart boxes; the displayed short command is a
fast smoke certificate. The current long proof-building run certifies through
`t=2.6` and stops just after `t=2.604` on a `2.75` target because the carried
axis-aligned box is too wide. Use `--taylor-restart-bridge-json` to reuse a
saved `--taylor-time-bridge-check --json` payload instead of recomputing the
expensive Taylor bridge prefix for every restart-chain attempt.
`--taylor-progress-every-blocks` writes progress lines to stderr, so stdout can
still be redirected into a clean JSON certificate. The current rectangular
frontier reaches about `t=2.603` in the direct continuation, so the remaining
compact bridge still needs sharper correlation control.
With retry subdivisions enabled, the cached restart-chain proof certifies
through `t=2.6`. A longer `t=2.75` target currently stops just after
`t=2.604`, where the carried axis-aligned box is already very wide and the
lower `x0` face fails. This honest composed certificate confirms that the next
gap is a correlation/shaped-tube problem after `t~2.6`, not a numerical sign
change in the sampled trajectory.
`--regular-time-corridor-check` is the ordinary-time analogue of the automatic
p-corridor; the displayed command certifies the short conditional bridge
`t=0.5 -> 0.6` with worst margin about `4.72e-4`.
`--support-tail-closure-check` composes the support bridge with the broad-tail
certificate, conditionally excluding closure from the support box at `t=3.5`
down to terminal. The remaining large-`|a|` proof obligation is to validate the
support start box itself from the original scaled IVP.
`--piecewise-corridor-check` then checks the current affine-corridor chain down
to `p=0.20995` from the older narrow frontier. The
`--late-tail-closure-check` command composes those stages with the stronger
`x3=-0.6` terminal wall and conditionally excludes closure from the `p=0.25`
frontier box down to `p=0`.

`experiments.s7.scout_refine` processes a completed S7 scout checkpoint without
rerunning the grid. It selects target-wise nearest-neighbor local minima, by
default with scout residual `< 0.15`, and promotes them through the calibrated
order-8/order-10/order-14 recovery ladder with order-14/order-18 verification:

```zsh
.venv/bin/python -m experiments.s7.scout_refine --dry-run
.venv/bin/python -m experiments.s7.scout_refine
.venv/bin/python -m experiments.s7.scout_refine --max-residual 0.2
.venv/bin/python -m experiments.s7.scout_refine --max-residual none
.venv/bin/python -m experiments.s7.scout_refine --scout-jsonl output/s7_scout_searches/20260612-132718-seed1729-s7-scout-v1.jsonl
```

The refinement output is written under `output/s7_scout_refinements/`.
Rerunning the same refinement command resumes the newest compatible incomplete
checkpoint; use `--no-resume` to force a fresh run.

Each script performs a baseline and a refined two-sided run and prints:

- left and right endpoint parameter data
- the Taylor settings
- the left and right patch centres
- branch-condition diagnostics on both sides
- midpoint values for raw `q` from both sides
- the midpoint `q` mismatch vector and norm
- the midpoint values of `l(t)` from both sides
- refinement agreement digits

For the Berger validation, the primary success criterion is that the left and right raw `q` states match at the midpoint `t = pi/6`. A secondary diagnostic is that the two-sided values of `l(pi/6)` agree and are numerically close to `0`.

For the doubled-sphere run, no small-error threshold is imposed yet; the output is
diagnostic evidence for the conjectural candidate.

The legacy command still runs the Berger validation:

```zsh
.venv/bin/python run_exploration.py
```

Representative output:

```text
Berger opposite-end diagnostic with eps = 0.1
T = pi/3 = 1.04719755119659774615421446109

Direct raw q agreement at t = 0.1
  left/right patch counts = 2 / 3
  ||q_left(t) - q_right(t)||_inf = 0.00000000171729372002500315993433946165
  mismatch = [-2.955539769431202e-10, -1.109512228030688e-9, 1.717293720025003e-9, -1.301000307404835e-10, 3.947590034128571e-10, 3.666436139323014e-10, -1.042626945149192e-9, 2.534288444678177e-10]

Direct raw q agreement at t = 0.947197551196597746154214461093
  left/right patch counts = 3 / 2
  ||q_left(t) - q_right(t)||_inf = 0.00000000171729372002500315993433946165
  mismatch = [-2.534288444678177e-10, -1.301000307404835e-10, -3.666436139323014e-10, -1.109512228030688e-9, -1.042626945149192e-9, -1.717293720025003e-9, 3.947590034128571e-10, 2.955539769431202e-10]

Endpoint asymptotic checks in raw q
  left -> right offset error = 0.0339275215479939986072330379724
  left -> right one-jet error = 0.000186840390165042049173928362035
  right -> left offset error = 0.0339275215479939986072330379724
  right -> left one-jet error = 0.000186840390165042049173928362035
```

## Right-End Berger Chart Comparison

To march Berger and round-`S^7` left-end data toward `t = pi/3` and compare the observed terminal asymptotics with the Berger right chart, run:

```zsh
python -m experiments.s7_right_chart_comparison
```

Or without activating the virtual environment:

```zsh
.venv/bin/python -m experiments.s7_right_chart_comparison
```

Representative output:

```text
Berger right-chart asymptotic expectation
  q1 = 3f + O(s^2), q2 = f + O(s^2)
  q3,q4,q5,q6 = O(s)
  q7 = -3d + O(s^2), q8 = -d + O(s^2)
  chart-form constraints at s=0:
    q1 - 3q2 = 0, q3=q4=q5=q6=0, q7 - 3q8 = 0
  validated Berger right offset = [0.2012461179749811, 0.06708203932499369, 0.0, 0.0, 0.0, 0.0, 0.3354101966249685, 0.1118033988749895]
  right-chart weights = [2, 2, 1, 1, 1, 1, 2, 2]

Case 1: Berger left data
  expectation: should match the validated Berger right chart
  scale used in q/scale: 0.11180339887498948482
  expected terminal q / scale: [9/5, 3/5, 0, 0, 0, 0, 3, 1]
  eps     form defect     fixed Berger offset error     Berger one-jet error      observed q / scale
  0.2       0.07284988394                 0.07284988394          0.001639830271      [1.7301361, 0.59108718, 0.43308282, -0.25252122, -0.65158917, 0.28159073, 2.8194583, 1.0698635]
  0.1       0.03392567627                 0.03392567627          0.000188685672      [1.7821376, 0.59794341, 0.24780831, -0.13337206, -0.30344047, 0.14123031, 2.9542114, 1.0178612]
  0.05      0.01624718844                 0.01624718844          2.736873651e-5      [1.7955123, 0.59949772, 0.13140928, -0.068068612, -0.14531927, 0.070192841, 2.9885035, 1.0044856]
  0.02     0.006295741289                0.006295741289          4.301174442e-5      [1.799285, 0.59992107, 0.054450992, -0.02717977, -0.056310822, 0.028257522, 2.9981458, 1.0007118]
  conclusion: should match the validated Berger right chart

Case 2: round-S7 left data tested against the Berger right chart
  expectation: should not have Berger-right chart form; q3,q4,q5,q6 do not tend to zero
  scale used in q/scale: 0.089442719099991587856
  expected terminal q / scale: [1, -2, -1, 2, -2, 19, 2, -19]
  eps     form defect     fixed Berger offset error     Berger one-jet error      observed q / scale
  0.2         4.064412112                   1.611119957             1.578347235      [0.99999994, -1.6359618, -0.96013393, -2.1964159, -2.3241719, 18.012869, 5.209285, -13.410741]
  0.1         5.051675269                   1.683381255             1.685393716      [0.99999969, -1.8220884, -0.99000894, -0.016310862, -2.1679204, 18.750812, 3.7671227, -17.570775]
  0.05        5.262967551                   1.779076949             1.779580064      [0.99999945, -1.9121849, -0.99750095, 1.0178444, -2.0853161, 18.937557, 2.9197111, -18.640685]
  0.02        5.295298689                   1.806065951             1.806146449      [0.99999915, -1.9651632, -0.99960023, 1.6140612, -2.0344374, 18.99001, 2.3759456, -18.942431]
  conclusion: should not have Berger-right chart form; q3,q4,q5,q6 do not tend to zero
```

## Run The Tests

With the virtual environment activated:

```zsh
pytest -q
```

## Repository Layout

- `problem/`: geometric `q` equations, left/right weighted charts, endpoint data, and singular-end Taylor seeds
- `solver/`: generic truncated-series utilities and the two-sided weighted marcher
- `tests/`: geometry, endpoint-seed, matching, and generic Taylor-engine tests
- `run_exploration.py`: reusable reporting helpers plus the legacy Berger entrypoint
- `experiments/`: named experiment entrypoints
- `experiments/berger_space/`: target-specific Berger validation and search entrypoints
- `experiments/s7/`: guarded round-`S^7` validation entrypoints
- `experiments/shared/`: common helpers used by long searches
- `runs/`: compatibility shims for older named experiment commands
