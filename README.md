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


## Berger Left/Right Check

To compare the independently marched Berger solutions at `t = 0.1` and `t = pi/3 - 0.1`, run:

```zsh
python -m experiments.berger_opposite_end_check
```

Or without activating the virtual environment:

```zsh
.venv/bin/python -m experiments.berger_opposite_end_check
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
- `run_exploration.py`: the single direct-run orchestration script
