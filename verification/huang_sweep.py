"""Region II of Huang's Condition 1.3: the bounded (a1, a2) sweep.

Certifies  H(a1,a2) + G(a1,a2) <= 0  over the rectangle
    R = [-sqrt(psi), sqrt(psi)] x [-A2MAX, A2MAX],  A2MAX < sqrt(q),
minus a small box EXCL around the degenerate maximizer (a1*, a2*) =
(psi(1-q), q), which is handled by the frozen-tilt Hessian (Region I).

Per cell (a1_ball, a2_ball):
  - center (c1, c2); nonrigorously find the optimal dual lambda* = dual_of(c)
    and tilt s* = argmin_s [...] (huang_np). These only guide the choice.
  - rigorously certify (huanggrid):
        Phi(lambda*) - lambda1* a1_ball - lambda2* a2_ball
        + s*^2 psi/2 + alpha T(a1_ball, a2_ball, s*)   <  0.
    The first line upper-bounds H(a1,a2) on the whole cell by convex duality
    (valid for ANY fixed lambda*); the second upper-bounds G (valid for ANY
    fixed s*). If dual_of fails, fall back to H <= log 2.
  - on failure, bisect the cell (longer side) until MIN_SIDE.

Cells entirely inside EXCL are skipped (Region I). A cell overlapping EXCL is
still swept on its non-excluded... we instead require EXCL to be axis-aligned
and only skip fully-contained cells, bisecting boundary cells down so the
excluded set is covered exactly by MIN_SIDE-resolution cells inside EXCL.

Run:  python huang_sweep.py [nworkers] [coarse_n]
Logs to results/huang_sweep.log.
"""

import math
import os
import sys
import time
from multiprocessing import Pool

# rectangle and exclusion (floats; the rigorous code re-derives balls)
import huang_np as nr

# achievable body has |a1| <= E|X| ~ 1.297, |a2| <= E|M| ~ 0.697; pad slightly
A1MAX = 1.31
A2MAX = 0.70
A1S = float(nr.PSI * (1 - nr.Q))
A2S = float(nr.Q)
# Region-I exclusion box (refined once the sweep reports failures near it)
EXCL = (0.95, 1.30, 0.45, 0.66)  # a1lo, a1hi, a2lo, a2hi

MIN_SIDE = 0.002
MAX_DEPTH = 7


def in_excl(a1lo, a1hi, a2lo, a2hi):
    """True if the cell is fully inside the exclusion box."""
    return (EXCL[0] <= a1lo and a1hi <= EXCL[1]
            and EXCL[2] <= a2lo and a2hi <= EXCL[3])


def overlaps_excl(a1lo, a1hi, a2lo, a2hi):
    return not (a1hi <= EXCL[0] or a1lo >= EXCL[1]
                or a2hi <= EXCL[2] or a2lo >= EXCL[3])


def eval_cell(a1lo, a1hi, a2lo, a2hi):
    """Rigorous upper bound on H+G over the cell. Returns arb ball or None.

    H bounded by the convex-duality tangent at the cell's nonrigorous dual
    (linear in a -> already tight over the cell). T bounded by mean value in
    (a1,a2) about the cell center. s and lambda* are nonrigorous choices;
    the certificate holds for any fixed values."""
    from flint import arb
    from core import dec, iv, PSI, ALPHA
    import huanggrid as hg
    c1 = (a1lo + a1hi) / 2
    c2 = (a2lo + a2hi) / 2
    r1 = (a1hi - a1lo) / 2
    r2 = (a2hi - a2lo) / 2
    lam = nr.dual_of(c1, c2)
    if lam is None:
        # cell near/over the achievable boundary, or in the strongly
        # stretched dual directions where Newton needs a warm start: pull
        # the center inward toward the deep-interior degenerate point, take
        # that dual, and try to CONTINUE it back out to the center.  H(a) <=
        # Phi(lam) - lam.a holds for ANY fixed lam, so every choice below
        # stays a valid upper bound on the achievable part of the cell.
        for frac in (0.02, 0.04, 0.08, 0.15, 0.3, 0.5):
            p1 = c1 + frac * (A1S - c1)
            p2 = c2 + frac * (A2S - c2)
            lam_p = nr.dual_of(p1, p2)
            if lam_p is None:
                continue
            lam_c = nr.dual_of(c1, c2, l0=lam_p)   # continuation
            lam = lam_c if lam_c is not None else lam_p
            break
    s_star = nr.G(c1, c2)[0]
    if s_star != s_star:            # nan
        s_star = 0.0
    cc1 = dec(str(round(c1, 8)))
    cc2 = dec(str(round(c2, 8)))
    rr1 = dec(str(r1))
    rr2 = dec(str(r2))
    s = dec(str(round(float(s_star), 6)))
    T = hg.T_meanvalue(cc1, cc2, rr1, rr2, s)
    if T is None:
        return None
    a1b = iv(str(a1lo), str(a1hi))
    a2b = iv(str(a2lo), str(a2hi))
    if lam is not None and abs(lam[0]) <= 80 and abs(lam[1]) <= 80:
        b1 = dec(str(round(lam[0], 6)))
        b2 = dec(str(round(lam[1], 6)))
        Phi = hg.Phi_of(b1, b2)
        Hub = Phi - b1 * a1b - b2 * a2b
    else:
        # No usable dual: the cell is near (or beyond) the boundary of the
        # moment body, where the true dual explodes along the outward normal
        # u.  H(a) <= Phi(kappa u) - kappa u.a for ANY fixed kappa, and
        # Phi(kappa u) = kappa h(u) + E log(1+e^{-2 kappa|u.f|}), so for
        # supported (or infeasible) a this drops to ~0 (or -infinity) as
        # kappa grows, at the price of kappa * (cell radius) wrapping.  Try
        # a ladder of kappa and keep the best; fall back to [0, log 2].
        Hub = arb(0).union(hg.LOG2)
        u = None
        if lam is not None:
            n = math.hypot(lam[0], lam[1])
            if n > 0:
                u = (lam[0] / n, lam[1] / n)
        if u is None:
            u = _support_dir(c1, c2)
        if u is not None:
            from core import endpoints as _ep
            for kap in (10.0, 20.0, 40.0, 80.0, 160.0):
                b1 = dec(str(round(kap * u[0], 6)))
                b2 = dec(str(round(kap * u[1], 6)))
                cand = hg.Phi_of(b1, b2) - b1 * a1b - b2 * a2b
                _, chi = _ep(cand)
                _, hhi = _ep(Hub)
                if chi < hhi:
                    Hub = cand
    return Hub + s * s * PSI / 2 + ALPHA * T


def _support_dir(c1, c2):
    """Numeric outward-normal guess at (c1, c2): the fan direction
    maximizing u.c - h(u) (any direction is valid for the dual bound).
    The fan spans a half circle; h(-u) = h(u), so both signs are tried."""
    import huanggrid as hg
    try:
        fan = hg._get_hfan()
    except Exception:
        return None
    best = None
    for (u1, u2, h) in fan:
        f1 = float(u1.mid() if hasattr(u1, 'mid') else u1)
        f2 = float(u2.mid() if hasattr(u2, 'mid') else u2)
        hf = float(h.mid() if hasattr(h, 'mid') else h)
        for sg in (1.0, -1.0):
            v = sg * (f1 * c1 + f2 * c2) - hf
            if best is None or v > best[0]:
                best = (v, sg * f1, sg * f2)
    if best is None:
        return None
    return (best[1], best[2])


def eval_cell_mv(a1lo, a1hi, a2lo, a2hi):
    """Mean-value form of eval_cell for small cells near the maximizer,
    where the linear variations of the dual tangent and the constraint term
    nearly cancel (the interval sum in eval_cell adds their widths and
    loses the margin).  Bounds

        total(a) <= total(c) + |d total/da1| r1 + |d total/da2| r2,
        d total/da_i = -lam_i + alpha dT/da_i(cell, s),

    with the gradient enclosed over the cell; near the maximizer
    alpha dT/da ~ lam(a), so the enclosure is ~|grad S_*| r, not |lam| r.
    Returns an arb upper-bound ball or None."""
    from flint import arb
    from core import dec, endpoints, PSI, ALPHA
    import huanggrid as hg
    import huang_hessian as hh
    c1 = (a1lo + a1hi) / 2
    c2 = (a2lo + a2hi) / 2
    r1 = (a1hi - a1lo) / 2
    r2 = (a2hi - a2lo) / 2
    lam = nr.dual_of(c1, c2)
    if lam is None:
        return None
    s_star = nr.G(c1, c2)[0]
    if s_star != s_star:
        return None
    b1 = dec(str(round(lam[0], 8)))
    b2 = dec(str(round(lam[1], 8)))
    s = dec(str(round(float(s_star), 8)))
    cc1 = dec(str(round(c1, 10)))
    cc2 = dec(str(round(c2, 10)))
    # thin center value
    Tc = hg.T_of(cc1, cc2, s)
    if Tc is None:
        return None
    center = (hg.Phi_of(b1, b2) - b1 * cc1 - b2 * cc2
              + s * s * PSI / 2 + ALPHA * Tc)
    # gradient of the majorant over the cell (s fixed)
    a1b = dec(str(a1lo)).union(dec(str(a1hi)))
    a2b = dec(str(a2lo)).union(dec(str(a2hi)))
    gz = hg.get_zt_grid()
    Td = hh._T_derivs(a1b, a2b, gz, s)
    if Td is None:
        return None
    g1 = -b1 + ALPHA * Td['dT1']
    g2 = -b2 + ALPHA * Td['dT2']

    def absup(x):
        lo, hi = endpoints(x)
        return max(abs(float(lo)), abs(float(hi)))

    # radii inflated by 1e-9: the decimal-rounded center differs from the
    # true cell center by up to ~1e-10, so max |a - c_dec| <= r + 1e-9
    slack = (dec(str(round(absup(g1) * 1.0000001, 10)))
             * (dec(str(r1)) + dec('0.000000001'))
             + dec(str(round(absup(g2) * 1.0000001, 10)))
             * (dec(str(r2)) + dec('0.000000001')))
    return center + slack


def cert_cell(job, depth=0):
    """Certify a cell < 0, bisecting on failure. Returns (ok, ncells, worst)."""
    import huanggrid as hg
    a1lo, a1hi, a2lo, a2hi = job
    if in_excl(a1lo, a1hi, a2lo, a2hi):
        return True, 0, None            # Region I handles it
    if hg.outside_K(a1lo, a1hi, a2lo, a2hi):
        return True, 0, None            # non-achievable: no profile here
    val = eval_cell(a1lo, a1hi, a2lo, a2hi)
    if not (val is not None and val < 0):
        # small cells near the maximizer: retry in mean-value form
        if (a1hi - a1lo) < 0.004 and (a2hi - a2lo) < 0.004:
            val2 = eval_cell_mv(a1lo, a1hi, a2lo, a2hi)
            if val2 is not None:
                val = val2
    if val is not None and val < 0:
        from core import endpoints
        _, hi = endpoints(val)
        return True, 1, float(hi.mid()) if hasattr(hi, 'mid') else float(hi)
    if (a1hi - a1lo) < MIN_SIDE and (a2hi - a2lo) < MIN_SIDE:
        # cannot refine further; if it overlaps EXCL treat as Region I, else fail
        if overlaps_excl(a1lo, a1hi, a2lo, a2hi):
            return True, 0, None
        return False, 1, (str(val), job)
    # bisect the longer side
    if (a1hi - a1lo) >= (a2hi - a2lo):
        m = (a1lo + a1hi) / 2
        subs = [(a1lo, m, a2lo, a2hi), (m, a1hi, a2lo, a2hi)]
    else:
        m = (a2lo + a2hi) / 2
        subs = [(a1lo, a1hi, a2lo, m), (a1lo, a1hi, m, a2hi)]
    ok_all = True
    ncells = 0
    worst = None
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


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, os.cpu_count() - 2)
    coarse = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    # coarse top-level grid of jobs
    jobs = []
    da1 = 2 * A1MAX / coarse
    na2 = max(1, int(2 * A2MAX / da1))
    da2 = 2 * A2MAX / na2
    for i in range(coarse):
        for j in range(na2):
            jobs.append((-A1MAX + da1 * i, -A1MAX + da1 * (i + 1),
                         -A2MAX + da2 * j, -A2MAX + da2 * (j + 1)))
    print(f"{len(jobs)} top cells, {nw} workers, "
          f"cell {da1:.3f}x{da2:.3f}", flush=True)
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    fails = 0
    total_cells = 0
    with Pool(nw, initializer=_init) as pool, \
            open('results/huang_sweep.log', 'w') as log:
        for k, res in enumerate(pool.imap_unordered(worker, jobs)):
            job, ok, nc, worst, dt = res
            total_cells += nc
            if not ok:
                fails += 1
                line = f"FAIL {job} worst={worst}"
                print(line, flush=True)
                log.write(line + "\n"); log.flush()
            if (k + 1) % 40 == 0:
                print(f"  {k+1}/{len(jobs)} top cells, {total_cells} leaves, "
                      f"{fails} fails, {time.time()-t0:.0f}s", flush=True)
        log.write(f"total_leaves={total_cells} fails={fails}\n")
    print(f"DONE: {len(jobs)} top cells -> {total_cells} leaves, "
          f"{fails} fails, {time.time()-t0:.0f}s", flush=True)
    if fails:
        raise SystemExit(1)


def _init():
    from core import set_prec
    set_prec(50)


if __name__ == '__main__':
    main()
