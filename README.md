# The storage capacity of the Ising perceptron: verification of the outstanding numerical conditions

Krauth and Mezard (1989) predicted that the storage capacity of the Ising
perceptron concentrates at an explicit constant alpha_* ~ 0.833078600. Ding
and Sun (Probab. Theory Related Fields 193 (2025), 627-715; conference
version STOC 2019; arXiv:1809.07742) proved the matching lower bound, and
Huang (arXiv:2404.18902) the matching upper bound. Each result is conditional
on a numerical hypothesis about an explicit low-dimensional variational
problem: Ding-Sun's Condition 1.2, which in its published form requires a
univariate rate function to be negative away from its two zeros together
with strict curvature at the degenerate zero and a one-sided derivative
condition at the endpoint, and Huang's Condition 1.3, the nonpositivity of a
two-variable rate function over the whole plane. This repository contains
rigorous interval-arithmetic verifications of both conditions and of the
parameter rectangle (alpha_*, q_*, psi_*) they share, together with the
accompanying paper.

The certificates are ball-arithmetic computations in Arb (via
`python-flint`); every decimal constant enters as an exact rational, the
parameters (alpha, q, psi) enter as balls covering the certified rectangle,
and every verdict is a certified ball comparison.

## Contents

```
paper/                  the paper (LaTeX source and PDF)
verification/           the verification programs and their unit tests
verification/results/   canonical certificates and the complete raw record
                        trees of the certified runs
```

The main components:

- `block1_gardner.py` - the Ding-Sun parameter rectangle (their
  Proposition 1.3), thirteen certified inequalities.
- `huang_sweep.py`, `huang_sweep2.py` - Huang's Condition 1.3 on the bulk of
  the compact moment body (two-stage adaptive sweep; the moment-coordinate
  reparametrization and the per-cell convex-duality bound are described in
  the paper).
- `huang_region1.py`, `huang_star_interior.py` - Huang's Condition 1.3 on a
  star region around the degenerate maximizer (ray-concavity certificates;
  the entropy Hessian is bounded through sublevel-set localizations of the
  dual and is never inverted).
- `block3a_grid.py`, `block2_near_one.py`, `block3bc.py` - the Ding-Sun
  Condition 1.2: value grids, the near-one analysis (with corrected
  constants; see the paper), and the middle interval with its degenerate
  zero. The curvature clause follows from the certified central
  second-derivative bound; the endpoint-derivative clause is analytic
  (see the paper's assembly section).
- `verify_all.py` - replays and validates every certificate against the
  frozen results, ending in `ALL CERTIFICATES PASS`.
- `huang_region1_verify.py`, `huang_sweep_verify.py`,
  `block3a_singlerun.py`, `block3bc_aux_verify.py` - the exact certificate
  verifiers. Each certificate pins the SHA-256 of the program files that
  produced it (its `source_sha256` fields), which can be checked directly
  against this tree.

## Reproducing

The attested runs used Python 3.12.3 with `python-flint` 0.9.0 (FLINT 3.6.0)
at 50-bit ball precision. The unit suite and the certificate verifiers run
on any platform with those packages:

```
cd verification
python -B -m unittest discover -s tests -p "test_*.py"
python -B verify_all.py
```

One unit test checks Git provenance of a frozen source view and expects to
run inside a checkout; the remaining 139 tests and all certificate
validation are self-contained. Exact certificate replay (byte-identical
ball packets) is guaranteed only on the attested runtime recorded in the
receipts; on other builds of the same library versions the verifiers still
validate schemas, geometry, coverage, and signs.

## Receipts

The final source-bound verification receipt binds the complete development
tree at source commit `5dd6c5a03197e73ed53fa2594fed599d31913193` under the
pinned runtime, with verdict `ALL CERTIFICATES PASS` (140 unit tests, the
manuscript inventory, and all nine theorem rows). Receipt file SHA-256
`6cda830ce7a1e76949d4706a82e109f3bb051aee4a1c4ca188149f23254e59a3`, payload
identity
`f2f9b0ebd3281893e4412bb3b6cd430a8987680c9b1f6f0f54313af749b8f744`. The
verification programs, certificates, and raw records in this repository are
byte-identical to that receipt-bound closure; the paper incorporates
subsequent wording-only revisions. Receipt files are retained with the
development tree rather than republished here.

## Citation

Y. Shmalo, The storage capacity of the Ising perceptron: verification of
the outstanding numerical conditions, 2026. Verification programs,
canonical certificates, and raw records at this repository.
