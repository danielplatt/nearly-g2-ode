# Nearly G2 ODE Exploration

This repository contains a first high-precision numerical exploration code for the weighted `q`-system coming from the cohomogeneity-one Einstein equations. The geometric `q` equations are treated as the source of truth, the weighted `y` variables are used for singular-end bookkeeping, and the midpoint diagnostics are reported in both variables.

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

The script performs a baseline and a refined midpoint march and prints:

- the parameter point
- the Taylor settings
- the patch centres
- branch-condition diagnostics
- midpoint values for `q`
- recovered midpoint values for `y`
- recovered midpoint derivatives `y'`
- refinement agreement digits

## Run The Tests

With the virtual environment activated:

```zsh
pytest -q
```

## Repository Layout

- `problem/`: geometric `q` equations, weighted maps, initial data, and singular-end Taylor seed
- `solver/`: generic truncated-series utilities and piecewise Taylor marcher
- `tests/`: problem-layer, singular-seed, and midpoint-march tests
- `run_exploration.py`: the single direct-run orchestration script
