# The storage capacity of the Ising perceptron: verification of the outstanding numerical conditions

Krauth and Mezard (1989) predicted that the storage capacity of the Ising
perceptron concentrates at an explicit constant alpha_* ~ 0.833078600. Ding
and Sun (arXiv:1809.07742) proved the matching lower bound and Huang
(arXiv:2404.18902) the matching upper bound, each conditional on a numerical
hypothesis about a low-dimensional variational problem that had been checked
only in floating point. This repository contains rigorous interval-arithmetic
verifications of both hypotheses and of the parameter rectangle they share,
together with the accompanying paper.

The certificates are ball-arithmetic computations in Arb (via `python-flint`);
every decimal constant enters as an exact rational, the parameters
(alpha, q, psi) enter as balls covering the certified rectangle, and every
verdict is a certified ball comparison.

## Contents

```
paper/          the paper (LaTeX source and PDF)
verification/   the verification programs
verification/results/   logs and summaries of the certified runs
```

The main components:

- `block1_gardner.py` - the Ding-Sun parameter rectangle (their
  Proposition 1.3), thirteen certified inequalities.
- `huang_sweep.py`, `huang_sweep2.py` - Huang's Condition 1.3 on the bulk of
  the compact moment body (two-stage adaptive sweep; the moment-coordinate
  reparametrization and the per-cell convex-duality bound are described in
  the paper).
- `huang_region1.py` - Huang's Condition 1.3 on a star region around the
  degenerate maximizer (ray-concavity certificates; the entropy Hessian is
  bounded through sublevel-set localizations of the dual and is never
  inverted).
- `block3a_grid.py`, `block2_near_one.py`, `block3bc.py` - the Ding-Sun
  Condition 1.2: value grids, the near-one analysis (with corrected
  constants; see the paper), and the middle interval with its degenerate
  zero.
- `verify_all.py` - re-checks the whole chain against the run logs and
  re-runs the quick blocks.
- `core.py`, `huanggrid.py`, `huang_hessian.py`, `dsfun.py` - certified
  special functions, quadrature, and the integral evaluators.
- `huang_np.py` - a nonrigorous floating-point companion used only to select
  per-cell duals, tilts and candidate boxes; no certified inequality depends
  on it.

## Reproducing

Python 3 with `python-flint` (>= 0.9), `numpy`, `scipy`.

```
cd verification
python verify_all.py          # re-check the chain from the included logs
python huang_sweep.py 6 48    # Region II stage 1 (~2 minutes, parallel)
python huang_region1.py 6     # Region I star certificates (~25 minutes)
python huang_sweep2.py 6      # Region II stage 2 (~3 minutes)
python block1_gardner.py
python block2_near_one.py
python block3a_grid.py 3      # several hours (a few slow tail cells)
python block3bc.py par 8      # the Ding-Sun middle interval
```

Set `HUANG_GRID_N=2700` in the environment for the sweep resolutions used in
the logs.
