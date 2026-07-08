"""Rigorous verification of Ding-Sun Proposition p:grid.result part (a):

  PG(lm) = H(lm) + P(lm) < 0 for lm in [0.2, 0.98],
  QG(lm) = H(lm) + Q(lm) < 0 for lm in [lmin, -0.125],

using their tau grids (transcribed verbatim from the source), the
lambda(tau) = ell(A(tau)) sandwich, and certified integrals for every cell.

Cell recipe (tau_lo, tau_hi):
  lam ball = hull of ell_range(A(tau_lo)), ell_range(A(tau_hi))
             [ell is increasing in A, A increasing in tau]
  A ball   = [A(tau_lo), A(tau_hi)]
  PG_ub    = frH_upper(A ball) + P_upper(lam ball)  (branch tau > 0)
  QG_ub    = frH_upper(A ball) + Q_upper(lam ball)  (branch tau < 0)

Run with worker parallelism:  python block3a_grid.py [nworkers]
Results go to results/block3a.log, one line per cell.
"""

import os
import sys
import time
from multiprocessing import Pool

GRID_POS = []  # (tau_lo, tau_hi) as decimal strings, positive branch
GRID_NEG = []  # negative branch (QG)


def seq(start, stop, step):
    """Their [[start, stop, step]] notation, endpoints as printed."""
    out = []
    # decimal-safe stepping
    from decimal import Decimal
    lo = Decimal(start)
    hi = Decimal(stop)
    st = Decimal(step)
    v = lo
    while v <= hi:
        out.append(str(v))
        v += st
    return out


def build_grids():
    t_pos = (seq('0.24', '0.284', '0.001')
             + seq('0.285', '0.315', '0.002')
             + seq('0.318', '0.342', '0.003')
             + seq('0.346', '0.366', '0.004')
             + seq('0.371', '0.386', '0.005')
             + seq('0.392', '0.404', '0.006')
             + ['0.411', '0.418', '0.425', '0.433', '0.441']
             + seq('0.45', '0.57', '0.01')
             + seq('0.59', '0.67', '0.02')
             + seq('0.7', '0.76', '0.03')
             + seq('0.8', '0.94', '0.04')
             + ['0.95', '0.98', '0.99'])
    t_neg = (seq('0.18', '0.209', '0.001')
             + seq('0.21', '0.236', '0.002')
             + seq('0.238', '0.268', '0.003')
             + seq('0.271', '0.343', '0.004')
             + seq('0.347', '0.419', '0.006')
             + seq('0.425', '0.513', '0.008')
             + seq('0.52', '0.77', '0.01')
             + ['0.78', '0.8', '0.82', '0.84', '0.86', '0.89', '0.93', '1'])
    pos_cells = list(zip(t_pos[:-1], t_pos[1:]))
    neg_cells = [('-' + b, '-' + a) for a, b in zip(t_neg[:-1], t_neg[1:])]
    return pos_cells, neg_cells


def _eval_cell(kind, tau_lo, tau_hi, depth):
    from flint import arb
    from core import dec
    import dsfun
    from decimal import Decimal
    A_hi = dsfun.A_of_tau(dec(tau_hi))
    if tau_lo == '-1':
        A_lo = arb(0)          # A(tau) -> 0 as tau -> -1
    else:
        A_lo = dsfun.A_of_tau(dec(tau_lo))
    val = dsfun.PG_cell(A_lo, A_hi) if kind == 'PG' \
        else dsfun.QG_cell(A_lo, A_hi)
    if val < 0:
        return True, str(val), 1
    if depth <= 0:
        return False, str(val), 1
    mid = str((Decimal(tau_lo) + Decimal(tau_hi)) / 2)
    ok1, v1, n1 = _eval_cell(kind, tau_lo, mid, depth - 1)
    if not ok1:
        return False, v1, n1 + 1
    ok2, v2, n2 = _eval_cell(kind, mid, tau_hi, depth - 1)
    return ok2, f"{v1} | {v2}", n1 + n2 + 1


def cell_worker(job):
    kind, tau_lo, tau_hi = job
    t0 = time.time()
    from core import set_prec
    set_prec(60)
    ok, val, ncalls = _eval_cell(kind, tau_lo, tau_hi, depth=5)
    dt = time.time() - t0
    return (kind, tau_lo, tau_hi, f"n={ncalls}", val, ok, dt)


def init_worker():
    # per-process one-time constants
    from core import set_prec
    import dsfun
    set_prec(60)
    dsfun._hlb()
    dsfun._i0lb()


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, os.cpu_count() - 2)
    pos_cells, neg_cells = build_grids()
    jobs = ([('PG', a, b) for a, b in pos_cells]
            + [('QG', a, b) for a, b in neg_cells])
    print(f"{len(jobs)} cells ({len(pos_cells)} PG, {len(neg_cells)} QG), "
          f"{nw} workers", flush=True)
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    fails = 0
    with Pool(nw, initializer=init_worker) as pool, \
            open('results/block3a.log', 'w') as log:
        for i, res in enumerate(pool.imap_unordered(cell_worker, jobs)):
            kind, a, b, lam, val, ok, dt = res
            line = f"{'PASS' if ok else 'FAIL'} {kind} [{a},{b}] lam={lam} val={val} ({dt:.1f}s)"
            log.write(line + "\n")
            log.flush()
            if not ok:
                fails += 1
                print(line, flush=True)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(jobs)} done, {fails} fails, "
                      f"{time.time()-t0:.0f}s elapsed", flush=True)
    print(f"done: {len(jobs)} cells, {fails} fails, {time.time()-t0:.0f}s",
          flush=True)

    # boundary checks tying the grid to the lambda intervals of part (a)
    from core import set_prec, dec, report
    import dsfun
    set_prec(80)
    b1 = dsfun.ell_range(dsfun.A_of_tau(dec('0.24')))
    b2 = dsfun.ell_range(dsfun.A_of_tau(dec('0.99')))
    b3 = dsfun.ell_range(dsfun.A_of_tau(dec('-0.18')))
    ok = report("lam(0.24) < 0.2", dec('0.2') - b1, '>0')
    ok &= report("lam(0.99) > 0.98", b2 - dec('0.98'), '>0')
    ok &= report("lam(-0.18) > -0.125", b3 - dec('-0.125'), '>0')
    if fails or not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
