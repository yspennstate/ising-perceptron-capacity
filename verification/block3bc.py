"""Rigorous verification of Ding-Sun Proposition p:grid.result parts (b), (c):
the middle interval lambda in [-0.125, 0.2] around the degenerate point 0.

Structure (their c:G assembly; spec in notes/dingsun_verification_spec.md):

  PG(lm) = H(lm) + P(lm),  P(lm) = -psi(1-q) lm/(1+lm) + I(lm) - I(0),
  with the analytic identities PG(0) = 0 and PG'(0) = 0 (H'(0) = 0 since
  A(0) = 1; P'(0) = 0 is the stationarity of the replica functional at the
  symmetric point -- Ding-Sun's construction).

  part (c):  PG''(lm) < 0 for lm in [-0.03, 0.05]
             ==> 0 is a strict local max with PG(0) = 0
             ==> PG < 0 on [-0.03, 0.05] \\ {0}.
  part (b):  PG'(lm) < 0 on [0.05, 0.2] and PG'(lm) > 0 on [-0.125, -0.03]
             ==> PG decreases away from the max
             ==> PG < 0 there (PG at +-ends of (c)'s interval already < 0).

Derivative formulas (tau-parametrized cells; lam = ell(A(tau)) sandwich):
  H'(lm)  = -(1-q) log A / 2
  H''(lm) = -(1-q) / (2 A ell'(A))
  P'(lm)  = -psi(1-q)/(1+lm)^2 + I'(lm)         [dsfun.I_prime_box, s = 0]
  P''(lm) =  2 psi(1-q)/(1+lm)^3 + I''(lm)      [dsfun.I_second_box]

Cells are adaptive (bisect on failure); coverage is pinned by the boundary
checks lam_up/lam_lo at the tau-endpoints, exactly as in block 3a.

Run:  python block3bc.py [nworkers]
"""

import os
import sys
import time
from multiprocessing import Pool

from flint import arb
from core import set_prec, dec, endpoints, report, PSI, Q, ALPHA
import dsfun

MIN_W = 2e-4


def _cell(tau_lo, tau_hi):
    A = dsfun.A_of_tau(dec(tau_lo)).union(dsfun.A_of_tau(dec(tau_hi)))
    lam = dsfun.lam_cell(dec(tau_lo), dec(tau_hi))
    return A, lam


def dPG_cell(tau_lo, tau_hi):
    """Enclosure of PG'(lm) over the cell."""
    A, lam = _cell(tau_lo, tau_hi)
    H1 = -(1 - Q) * A.log() / 2
    P1 = -PSI * (1 - Q) / ((1 + lam) * (1 + lam)) \
        + dsfun.I_prime_box(lam, arb(0), zlo=-7.5, zhi=7.5, xhi=14,
                            inner_tol_bits=20, outer_tol_bits=17)
    return H1 + P1


def d2PG_cell(tau_lo, tau_hi):
    """Enclosure of PG''(lm) over the cell."""
    A, lam = _cell(tau_lo, tau_hi)
    ellp = dsfun.ell_prime(A)
    lo, _ = endpoints(ellp)
    if not (lo > 0):
        return None
    H2 = -(1 - Q) / (2 * A * ellp)
    P2 = 2 * PSI * (1 - Q) / ((1 + lam) ** 3) \
        + dsfun.I_second_box(lam, inner_tol_bits=14, outer_tol_bits=12)
    return H2 + P2


def _adaptive(tau_lo, tau_hi, evalf, want, tag):
    """Certify evalf < 0 (want='<0') or > 0 on [tau_lo, tau_hi], bisecting."""
    stack = [(tau_lo, tau_hi)]
    ncell = 0
    while stack:
        lo, hi = stack.pop()
        v = evalf(f"{lo:.6f}", f"{hi:.6f}")
        ok = False
        if v is not None:
            vlo, vhi = endpoints(v)
            ok = (vhi < 0) if want == '<0' else (vlo > 0)
        if ok:
            ncell += 1
            continue
        if (hi - lo) <= MIN_W:
            return False, ncell, (lo, hi, str(v)[:30])
        m = 0.5 * (lo + hi)
        stack.append((lo, m))
        stack.append((m, hi))
    return True, ncell, None


def job(args):
    set_prec(60)
    kind = args[0]
    t0 = time.time()
    if kind == 'b_pos':
        ok, nc, bad = _adaptive(0.06, 0.26, dPG_cell, '<0', kind)
    elif kind == 'b_neg':
        ok, nc, bad = _adaptive(-0.19, -0.03, dPG_cell, '>0', kind)
    else:
        ok, nc, bad = _adaptive(-0.043, 0.078, d2PG_cell, '<0', kind)
    return kind, ok, nc, bad, time.time() - t0


def boundaries():
    """lam-coverage pins (rigorous; ell increasing)."""
    set_prec(60)
    ok = True
    b = dsfun.ell_range(dsfun.A_of_tau(dec('0.26')))
    ok &= report("lam_lo(0.26) > 0.2", b - dec('0.2'), '>0')
    b = dsfun.ell_range(dsfun.A_of_tau(dec('0.06')))
    ok &= report("lam_up(0.06) < 0.05", dec('0.05') - b, '>0')
    b = dsfun.ell_range(dsfun.A_of_tau(dec('-0.19')))
    ok &= report("lam_up(-0.19) < -0.125", dec('-0.125') - b, '>0')
    b = dsfun.ell_range(dsfun.A_of_tau(dec('-0.03')))
    ok &= report("lam_lo(-0.03) > -0.03", b - dec('-0.03'), '>0')
    b = dsfun.ell_range(dsfun.A_of_tau(dec('-0.043')))
    ok &= report("lam_up(-0.043) < -0.03", dec('-0.03') - b, '>0')
    b = dsfun.ell_range(dsfun.A_of_tau(dec('0.078')))
    ok &= report("lam_lo(0.078) > 0.05", b - dec('0.05'), '>0')
    return ok


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    os.makedirs('results', exist_ok=True)
    lines = []

    def say(s):
        print(s, flush=True)
        lines.append(s)

    say("boundary pins:")
    okb = boundaries()
    say(f"boundary pins ok: {okb}")
    jobs = [('b_pos',), ('b_neg',), ('c',)]
    t0 = time.time()
    allok = bool(okb)
    with Pool(min(nw, 3), initializer=_init) as pool:
        for kind, ok, nc, bad, dt in pool.imap_unordered(job, jobs):
            tag = 'PASS' if ok else 'FAIL'
            say(f"{tag} {kind}: {nc} cells ({dt:.0f}s) "
                f"{'' if ok else bad}")
            allok = allok and ok
    say(f"{'ALL PASS' if allok else 'FAILURES'} ({time.time()-t0:.0f}s)")
    with open(os.path.join('results', 'block3bc.log'), 'w') as f:
        f.write("\n".join(lines) + "\n")
    if not allok:
        raise SystemExit(1)


def _init():
    set_prec(60)


def main_par():
    """Parallel driver: split each interval into chunks and pool ALL chunks
    (the per-interval jobs of main() parallelize poorly: one worker each)."""
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, os.cpu_count() - 2)
    os.makedirs('results', exist_ok=True)
    lines = []

    def say(s):
        print(s, flush=True)
        lines.append(s)

    say("boundary pins:")
    okb = boundaries()
    say(f"boundary pins ok: {okb}")
    chunks = []
    for (kind, lo, hi, n) in (('b_pos', 0.06, 0.26, 24),
                              ('b_neg', -0.19, -0.03, 20),
                              ('c', -0.043, 0.078, 16)):
        for k in range(n):
            chunks.append((kind, lo + (hi - lo) * k / n,
                           lo + (hi - lo) * (k + 1) / n))
    t0 = time.time()
    allok = bool(okb)
    kinds = {}
    with Pool(nw, initializer=_init) as pool:
        for kind, ok, nc, bad, dt in pool.imap_unordered(_chunk_job, chunks):
            kinds.setdefault(kind, [0, 0])
            kinds[kind][0] += nc
            kinds[kind][1] += 0 if ok else 1
            if not ok:
                say(f"FAIL {kind} chunk: {bad}")
                allok = False
    for kind, (nc, nf) in sorted(kinds.items()):
        say(f"{'PASS' if nf == 0 else 'FAIL'} {kind}: {nc} cells "
            f"({nf} failed chunks)")
    say(f"{'ALL PASS' if allok else 'FAILURES'} ({time.time()-t0:.0f}s)")
    with open(os.path.join('results', 'block3bc.log'), 'w') as f:
        f.write("\n".join(lines) + "\n")
    if not allok:
        raise SystemExit(1)


def _chunk_job(args):
    kind, lo, hi = args
    set_prec(60)
    t0 = time.time()
    if kind == 'c':
        ok, nc, bad = _adaptive(lo, hi, d2PG_cell, '<0', kind)
    elif kind == 'b_pos':
        ok, nc, bad = _adaptive(lo, hi, dPG_cell, '<0', kind)
    else:
        ok, nc, bad = _adaptive(lo, hi, dPG_cell, '>0', kind)
    return kind, ok, nc, bad, time.time() - t0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'par':
        main_par()
    else:
        main()
