"""Finite-size satisfiability scan for the zero-margin Ising perceptron.

A theory-free numerical cross-check of the certified threshold
alpha* = 0.8330785995...: for small N, estimate

    P_sat(N, alpha) = P( exists J in {-1,+1}^N : G J >= 0 rowwise ),
    G an (M x N) iid standard Gaussian matrix, M = round(alpha N),

by exhaustive search (vectorized frontier BFS over spins with the standard
partial-sum + remaining-l1-mass pruning; exact, no heuristics, both signs of
every spin searched -- fixing J_1 would bias P_sat downward because J -> -J
reverses the inequalities rather than preserving them).  Spins are reordered
by decreasing column l1 mass, which changes nothing mathematically and
tightens pruning.

Finite-size scaling for this model approaches alpha* slowly from above
(Krauth--Mezard 1989 saw apparent thresholds near 0.85 for N ~ 20-25), so
the check asserts only the coarse shape: P_sat is ~1 well below alpha*,
decays through the neighborhood of alpha*, and the crossing point moves
toward alpha* as N grows.  Anything sharper needs far larger N than exact
search allows.

Writes one JSON line per (N, alpha) cell to the output file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def satisfiable(G, node_budget=6_000_000):
    """Exact: does some J in {-1,1}^N satisfy G J >= 0 (all rows)?

    Frontier DFS over spins with pruning: a partial assignment with partial
    sums p is viable iff p_r + l1_rest_r >= 0 for every row r.  Returns
    (verdict, nodes) where verdict is True/False, or None if the node budget
    was exceeded (censored).
    """
    M, N = G.shape
    order = np.argsort(-np.abs(G).sum(axis=0))
    G = G[:, order]                      # spin relabeling; exactness kept
    l1_rest = np.abs(G)[:, ::-1].cumsum(axis=1)[:, ::-1]  # incl. column j
    frontier = np.zeros((1, M), dtype=G.dtype)
    nodes = 1
    for j in range(N):
        rest = l1_rest[:, j] - np.abs(G[:, j])   # mass strictly after j
        plus = frontier + G[:, j][None, :]
        minus = frontier - G[:, j][None, :]
        frontier = np.concatenate([plus, minus], axis=0)
        nodes += frontier.shape[0]
        keep = (frontier + rest[None, :] >= 0).all(axis=1)
        frontier = frontier[keep]
        if frontier.shape[0] == 0:
            return False, nodes
        if nodes > node_budget:
            return None, nodes
    return frontier.shape[0] > 0, nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260713)
    ap.add_argument("--sizes", default="20,24,28")
    ap.add_argument("--alphas", default="0.70,0.76,0.82,0.8331,0.88,0.94,1.00")
    ap.add_argument("--samples", type=int, default=40)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sizes = [int(s) for s in args.sizes.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    with open(args.out, "w") as out:
        for N in sizes:
            for alpha in alphas:
                M = max(1, round(alpha * N))
                sat = unsat = censored = 0
                t0 = time.time()
                for _ in range(args.samples):
                    G = rng.standard_normal((M, N)).astype(np.float32)
                    verdict, _ = satisfiable(G)
                    if verdict is None:
                        censored += 1
                    elif verdict:
                        sat += 1
                    else:
                        unsat += 1
                row = {
                    "N": N, "alpha": alpha, "M": M,
                    "samples": args.samples, "sat": sat, "unsat": unsat,
                    "censored": censored,
                    "p_sat": sat / max(1, sat + unsat),
                    "seconds": round(time.time() - t0, 2),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(row, file=sys.stderr)


if __name__ == "__main__":
    main()
