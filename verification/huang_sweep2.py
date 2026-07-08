"""Stage-2 sweep: certify S_* < 0 on (old EXCL box) minus the Region-I star.

The original Region II sweep (huang_sweep.py) certifies S_* < 0 on K minus
EXCL_OLD = [0.95, 1.30] x [0.45, 0.66].  Region I (huang_region1.py) gives
S_* <= 0 on the star region {a* + t v(th): 0 <= t <= T(th)} with the
angle-dependent radii T_LONG/T_MID/T_CORE around the weak direction.  This
driver closes the remainder: every cell of EXCL_OLD that is not fully inside
the star is certified by the same dual-tangent bound (eval_cell), bisecting
down to MIN_SIDE = 1e-3 near the star boundary.

Star containment test (conservative): a cell is skipped only if the maximal
corner distance from a* is <= the minimum of T(th) over the cell's angular
range minus a safety margin; T is piecewise constant in the angular distance
to +-w, so the minimum over a range is the minimum over the zones it meets.

Run:  python huang_sweep2.py [nworkers]
"""

import os
import sys
import time
import math
from multiprocessing import Pool

import huang_np as nr
import huang_sweep as sw
from huang_region1 import (A1S, A2S, W_ANG, WEDGE_HALF, CONE_MID,
                           T_LONG, T_MID, T_CORE, omega_of)

# Stage-1 (huang_sweep.py) treats MIN_SIDE cells that OVERLAP its exclusion
# box as covered by this stage; those cells extend at most MIN_SIDE_1 = 0.002
# beyond the box, so this stage sweeps the box EXPANDED by 0.004 on every
# side, closing the sliver.
EXCL_OLD = (0.95 - 0.004, 1.30 + 0.004, 0.45 - 0.004, 0.66 + 0.004)
MIN_SIDE = 1e-3
SAFE = 5e-7      # includes the star-origin uncertainty (true a* vs decimals)


def T_eff(theta):
    """Effective certified star radius for a single direction.

    Region I certifies each angular CELL out to min(t1, T(theta_center)),
    and cells can straddle the T-zone boundaries; a direction whose cell
    center fell in the smaller zone is certified only to the smaller radius.
    Cells adjacent to the WEDGE_HALF boundary come from 0.016-rad chunks
    (huang_region1.bands emits chunk width 0.016 for omega <= WEDGE_HALF +
    0.05, on both sides) split into at least two cells, so their width is
    <= 0.008; cells adjacent to CONE_MID come from 0.05-rad chunks, width
    <= 0.025.  Shrinking each zone by strictly more than the adjacent cell
    width makes T_eff(theta) <= every certificate that could apply:
      omega <= WEDGE_HALF - 0.012            -> T_LONG
      omega <= CONE_MID  - 0.030             -> T_MID
      otherwise                              -> T_CORE."""
    om = omega_of(theta)
    if om <= WEDGE_HALF - 0.012 - SAFE:
        return T_LONG
    if om <= CONE_MID - 0.030 - SAFE:
        return T_MID
    return T_CORE


def T_min_over(th_lo, th_hi):
    """Min of T_eff over [th_lo, th_hi] (piecewise constant; dense sampling
    plus the shrunken zone edges make boundary handling conservative)."""
    tmin = None
    n = 128
    for k in range(n + 1):
        th = th_lo + (th_hi - th_lo) * k / n
        t = T_eff(th)
        tmin = t if tmin is None else min(tmin, t)
    return tmin


def in_star(a1lo, a1hi, a2lo, a2hi):
    """True only if the cell is certainly inside the Region-I star."""
    corners = [(a1lo, a2lo), (a1lo, a2hi), (a1hi, a2lo), (a1hi, a2hi)]
    rmax = max(math.hypot(c1 - A1S, c2 - A2S) for (c1, c2) in corners)
    if rmax < T_CORE - SAFE:
        return True                       # inside the everywhere-radius
    # if the cell contains a*, the T_CORE test above is the only safe one
    if a1lo - SAFE <= A1S <= a1hi + SAFE and a2lo - SAFE <= A2S <= a2hi + SAFE:
        return False
    ths = [math.atan2(c2 - A2S, c1 - A1S) for (c1, c2) in corners]
    # angular interval of the cell as seen from a* (cell is away from a*,
    # so the span is < pi and the corner hull brackets it)
    th0, th1 = min(ths), max(ths)
    if th1 - th0 > math.pi:               # wrapped; rotate branch
        ths = [t + 2 * math.pi if t < 0 else t for t in ths]
        th0, th1 = min(ths), max(ths)
    return rmax <= T_min_over(th0, th1) - SAFE


def cert_cell(job, depth=0):
    a1lo, a1hi, a2lo, a2hi = job
    import huanggrid as hg
    if in_star(a1lo, a1hi, a2lo, a2hi):
        return True, 0, None              # Region I covers it
    if hg.outside_K(a1lo, a1hi, a2lo, a2hi):
        return True, 0, None
    val = sw.eval_cell(a1lo, a1hi, a2lo, a2hi)
    if not (val is not None and val < 0):
        # small cells near the maximizer: mean-value form restores the
        # dual-tangent / constraint gradient cancellation
        if (a1hi - a1lo) < 0.004 and (a2hi - a2lo) < 0.004:
            val2 = sw.eval_cell_mv(a1lo, a1hi, a2lo, a2hi)
            if val2 is not None:
                val = val2
    if val is not None and val < 0:
        return True, 1, None
    if (a1hi - a1lo) < MIN_SIDE and (a2hi - a2lo) < MIN_SIDE:
        return False, 1, (str(val), job)
    if (a1hi - a1lo) >= (a2hi - a2lo):
        m = 0.5 * (a1lo + a1hi)
        subs = [(a1lo, m, a2lo, a2hi), (m, a1hi, a2lo, a2hi)]
    else:
        m = 0.5 * (a2lo + a2hi)
        subs = [(a1lo, a1hi, a2lo, m), (a1lo, a1hi, m, a2hi)]
    ok_all, ncells, worst = True, 0, None
    for s in subs:
        ok, nc, w = cert_cell(s, depth + 1)
        ok_all = ok_all and ok
        ncells += nc
        if not ok and worst is None:
            worst = w
    return ok_all, ncells, worst


def worker(job):
    from core import set_prec
    set_prec(50)
    t0 = time.time()
    ok, nc, worst = cert_cell(job)
    return (job, ok, nc, worst, time.time() - t0)


def _init():
    from core import set_prec
    set_prec(50)


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, os.cpu_count() - 2)
    a1lo, a1hi, a2lo, a2hi = EXCL_OLD
    n1, n2 = 20, 12
    jobs = []
    for i in range(n1):
        for j in range(n2):
            jobs.append((a1lo + (a1hi - a1lo) * i / n1,
                         a1lo + (a1hi - a1lo) * (i + 1) / n1,
                         a2lo + (a2hi - a2lo) * j / n2,
                         a2lo + (a2hi - a2lo) * (j + 1) / n2))
    print(f"{len(jobs)} top cells over EXCL_OLD, {nw} workers, "
          f"MIN_SIDE={MIN_SIDE}", flush=True)
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    fails = 0
    total = 0
    with Pool(nw, initializer=_init) as pool, \
            open('results/huang_sweep2.log', 'w') as log:
        for k, res in enumerate(pool.imap_unordered(worker, jobs)):
            job, ok, nc, worst, dt = res
            total += nc
            if not ok:
                fails += 1
                line = f"FAIL {job} worst={worst}"
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            if (k + 1) % 24 == 0:
                print(f"  {k+1}/{len(jobs)} top cells, {total} leaves, "
                      f"{fails} fails, {time.time()-t0:.0f}s", flush=True)
        log.write(f"total_leaves={total} fails={fails}\n")
    print(f"DONE: {total} leaves, {fails} fails, {time.time()-t0:.0f}s",
          flush=True)
    if fails:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
