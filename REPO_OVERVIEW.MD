# Repository Overview

This repository is currently doing two related, but conceptually separate,
explorations of the same nearly parallel `G_2` ODE machinery.

The shared numerical core should stay topology-agnostic where possible: the raw
geometric `q` equations are the source of truth, weighted endpoint charts remove
the singular endpoint behaviour, and the solver marches weighted Taylor patches
from singular orbits into the interior.

## Folder Structure

The active cleanup target is to keep the numerical core shared while grouping
new experiment entrypoints by topology:

```text
problem/
  q_system.py              # raw geometric ODE, shared by all targets
  charts.py                # endpoint weighted charts
  initial_data.py          # endpoint jet formulas and named presets
  taylor_seed.py           # singular-end weighted Taylor seeds
  types.py                 # small shared containers

solver/
  series.py                # generic truncated Taylor algebra
  march.py                 # generic weighted patch marching
  mirror_shooting.py       # mirror-symmetric residuals
  mirror_refinement.py     # mirror-symmetric Newton refinement
  two_sided_shooting.py    # non-mirrored two-sided residuals
  two_sided_refinement.py  # overdetermined two-sided Gauss-Newton

experiments/
  berger_space/
    validation.py
    mirror_jacobian.py
    mirror_search.py
    mirror_sweep.py
    mirror_sweep_v2.py
    mirror_sweep_v3.py
    mirror_local_grid_v3.py
    non_mirrored_jacobian.py
    non_mirrored_search.py
    non_mirrored_surrogate_search.py
    non_mirrored_surrogate_wide_search.py

  s7/
    round_validation.py
    squashed_validation.py
    topology_search.py

  shared/
    mirror_sweep_common.py
    non_mirrored_common.py
    non_mirrored_surrogate_common.py

output/
  berger_space/
  s7/
```

The historical flat `experiments/` modules are kept as compatibility shims so
old commands and old JSON provenance remain meaningful. New work should use the
target-specific packages directly.

## Exploration Target 1: Berger Space Topology

The Berger-space target is the current mature validation and search target.
Here `lambda = 6/sqrt(5)` is used as the scale gauge. The known homogeneous
Berger solution is encoded by the default endpoint data, and the numerical
checks are:

- two-sided weighted Taylor marching from both singular ends;
- raw `q` matching at the midpoint;
- the Berger diagnostic `l(pi/6) = 0`;
- branch-condition monitoring along both marches.

The validated Berger run succeeds. The left and right raw `q` states match at
the midpoint to the expected Taylor accuracy, and the two values of `l(pi/6)`
are numerically close to zero. This is our main regression test that the raw
`q` equations, weighted charts, singular seeds, and patch marcher are all
cohering.

We then searched for other nearly parallel `G_2` structures on the same Berger
space topology in two stages.

First, we imposed mirror symmetry. The experiments include a local Jacobian
check at Berger, seeded basin searches, long annular sweeps, a version with the
hard floor `m > 0.01`, and a local grid refinement near the best small-`m`
leads. The outcome so far is negative: candidates either flow back to Berger,
hit branch failures, appear to be finite-order artifacts, or sit near the
small-interval floor. No stable non-Berger mirror-complete candidate has emerged.

Second, we removed mirror symmetry while keeping the Berger topology. The
two-sided residual became the full eight-component raw `q` mismatch, with seven
scaled coordinates for independent left data, independent right data, and the
interval length. We tried direct seeded search and then XGBoost-assisted
surrogate searches trained on thousands of cheap true Taylor solves. These runs
also produced no convincing non-Berger candidate so far. The best low-order
leads again appear suspicious, especially when they occur near small interval
lengths or fail to stabilize under higher-order verification.

These computations are evidence, not proof. The practical conclusion is that
the Berger topology has been explored seriously enough that moving to another
topology is mathematically reasonable.

## Exploration Target 2: `S^7` Topology

The next natural target is `S^7`. The source notes indicate that at least the
round `S^7` endpoint jets should be cleanly derivable. There is also a standard
homogeneous squashed `S^7` nearly parallel `G_2` structure, so this topology
should provide two known reference examples if we can translate both into the
current `q` and endpoint-chart conventions.

The immediate goal is validation rather than search:

- derive the left and right endpoint data for round `S^7`;
- add a named round-`S^7` preset without disturbing the Berger defaults;
- run the same two-sided weighted march-and-match diagnostic;
- compare the numerical solution against any explicit homogeneous formula that
  can be extracted from the notes;
- only then repeat the process for the squashed `S^7` structure.

Once the two known `S^7` examples are validated, the search problem becomes much
better posed. We can then decide whether to search near the round structure, near
the squashed structure, or globally across the corresponding endpoint-parameter
space.

The `S^7` entrypoints now validate both derived reference examples: round `S^7`
uses a fixed `p_3` right endpoint chart, while squashed `S^7` uses a fixed `p_2`
right endpoint chart.

## Current Navigation

The most important files today are:

- `README.md`: setup instructions and runnable experiment commands;
- `run_exploration.py`: reusable two-sided reporting helper and legacy Berger
  entrypoint;
- `problem/`: the geometric formulas and endpoint data;
- `solver/`: Taylor marching and shooting/refinement machinery;
- `experiments/`: the actual validation and search scripts;
- `experiments/berger_space/`: Berger validation and search scripts;
- `experiments/s7/`: guarded `S^7` validation scripts;
- `experiments/shared/`: common long-search helpers;
- `output/`: ignored JSONL and summary artifacts from long searches.

The `O3O10-main/` directory is the copied Buttsworth-Hodgkinson reference
repository. It is useful for method comparison, but it is not part of the active
runtime path.
