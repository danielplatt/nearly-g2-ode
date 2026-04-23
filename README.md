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

## Run The Exploration Script

From the repository root, with the virtual environment activated:

```zsh
python run_exploration.py
```

If you prefer not to activate the virtual environment, run:

```zsh
.venv/bin/python run_exploration.py
```

The script performs a baseline and a refined two-sided Berger validation and prints:

- left and right endpoint parameter data
- the Taylor settings
- the left and right patch centres
- branch-condition diagnostics on both sides
- midpoint values for raw `q` from both sides
- the midpoint `q` mismatch vector and norm
- the midpoint values of `l(t)` from both sides
- refinement agreement digits

For the Berger validation, the primary success criterion is that the left and right raw `q` states match at the midpoint `t = pi/6`. A secondary diagnostic is that the two-sided values of `l(pi/6)` agree and are numerically close to `0`.

## Run The Tests

With the virtual environment activated:

```zsh
pytest -q
```

## Repository Layout

- `problem/`: geometric `q` equations, left/right weighted charts, endpoint data, and singular-end Taylor seeds
- `solver/`: generic truncated-series utilities and the two-sided weighted marcher
- `tests/`: geometry, endpoint-seed, matching, and generic Taylor-engine tests
- `run_exploration.py`: the single direct-run orchestration script
