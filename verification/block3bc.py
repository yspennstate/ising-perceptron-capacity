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


def dPG_cell(tau_lo, tau_hi, bits=(20, 17)):
    """Enclosure of PG'(lm) over the cell."""
    A, lam = _cell(tau_lo, tau_hi)
    H1 = -(1 - Q) * A.log() / 2
    P1 = -PSI * (1 - Q) / ((1 + lam) * (1 + lam)) \
        + dsfun.I_prime_box(lam, arb(0), zlo=-7.5, zhi=7.5, xhi=14,
                            inner_tol_bits=bits[0], outer_tol_bits=bits[1])
    return H1 + P1


def dPG_cell_neg(tau_lo, tau_hi):
    """Negative branch: its inner integrals run far slower, and its margins
    (>= ~5e-3 away from the join) tolerate a looser tolerance."""
    return dPG_cell(tau_lo, tau_hi, bits=(15, 14))


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


def _adaptive(tau_lo, tau_hi, evalf, want, tag, min_w=None):
    """Certify evalf < 0 (want='<0') or > 0 on [tau_lo, tau_hi], bisecting."""
    if min_w is None:
        min_w = MIN_W
    stack = [(tau_lo, tau_hi)]
    ncell = 0
    while stack:
        lo, hi = stack.pop()
        v = evalf(f"{lo:.8f}", f"{hi:.8f}")
        ok = False
        if v is not None:
            vlo, vhi = endpoints(v)
            ok = (vhi < 0) if want == '<0' else (vlo > 0)
        if ok:
            ncell += 1
            continue
        if (hi - lo) <= min_w:
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
        ok, nc, bad = _adaptive(-0.19, -0.03, dPG_cell_neg, '>0', kind)
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
    only = os.environ.get('B3BC_ONLY')
    chunks = []
    for (kind, lo, hi, n) in (('b_pos', 0.06, 0.26, 24),
                              ('b_neg', -0.19, -0.03, 20),
                              ('c', -0.043, 0.078, 16)):
        if only and kind not in only.split(','):
            continue
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
        ok, nc, bad = _adaptive(lo, hi, dPG_cell_neg_mv, '>0', kind, min_w=5e-6)
    return kind, ok, nc, bad, time.time() - t0


def main_kcell():
    """block3bc.py kcell LO HI OUT: one coarse |PG''| bound cell."""
    lo, hi, out = sys.argv[2], sys.argv[3], sys.argv[4]
    set_prec(60)
    v = d2PG_cell(f"{float(lo):.6f}", f"{float(hi):.6f}")
    vlo, vhi = endpoints(v)
    K = max(abs(float(vlo)), abs(float(vhi)))
    with open(out, 'w') as f:
        f.write(str(K) + chr(10))


def main_chunk():
    """Run ONE chunk in an isolated process: block3bc.py chunk KIND LO HI OUT.
    Writes 'OK cells' or 'FAIL detail' to OUT."""
    kind, lo, hi, out = sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), \
        sys.argv[5]
    set_prec(60)
    if kind == 'c':
        ok, nc, bad = _adaptive(lo, hi, d2PG_cell, '<0', kind)
    elif kind == 'b_pos':
        ok, nc, bad = _adaptive(lo, hi, dPG_cell, '<0', kind)
    else:
        ok, nc, bad = _adaptive(lo, hi, dPG_cell_neg_mv, '>0', kind, min_w=5e-6)
    with open(out, 'w') as f:
        f.write(f"{'OK' if ok else 'FAIL'} {kind} {lo} {hi} cells={nc} "
                f"{'' if ok else bad}\n")


def main_iso(nw=6):
    """Driver: every chunk in its own subprocess (full isolation; a crashed
    chunk is retried once, then reported)."""
    import subprocess
    import tempfile
    os.makedirs('results', exist_ok=True)
    lines = []

    def say(s):
        print(s, flush=True)
        lines.append(s)

    say("boundary pins:")
    okb = boundaries()
    say(f"boundary pins ok: {okb}")
    only = os.environ.get('B3BC_ONLY')
    if (not only or 'b_neg' in only) and not os.environ.get('B3BC_K'):
        import subprocess as sp0
        import tempfile as tf0
        here0 = os.path.dirname(os.path.abspath(__file__))
        tmp0 = tf0.mkdtemp(prefix='b3bck_')
        ps = []
        for kk in range(6):
            lo0 = -0.20 + 0.18 * kk / 6
            hi0 = -0.20 + 0.18 * (kk + 1) / 6
            out0 = os.path.join(tmp0, f"k{kk}.txt")
            ps.append((sp0.Popen(
                [sys.executable, os.path.join(here0, 'block3bc.py'), 'kcell',
                 repr(lo0), repr(hi0), out0],
                stdout=sp0.DEVNULL, stderr=sp0.DEVNULL,
                creationflags=getattr(sp0, 'IDLE_PRIORITY_CLASS', 0)), out0))
        Kv = 0.0
        for p0, out0 in ps:
            p0.wait()
            Kv = max(Kv, float(open(out0).read().strip()))
        os.environ['B3BC_K'] = f"{Kv * 1.0000001:.8f}"
        say(f"|PG''| bound K = {os.environ['B3BC_K']}")
    nbneg = 20
    if os.environ.get('B3BC_K'):
        # size b_neg cells so one mean-value eval certifies:
        # K w/2 + eval width (~6e-4) <= margin (~2e-3); lam ~ 0.55 tau here
        Kv0 = float(os.environ['B3BC_K'])
        wlam = 2 * (0.0020 - 0.0006) / max(Kv0, 0.1)
        nbneg = max(20, int(0.16 / (wlam / 0.55)) + 1)
    chunks = []
    for (kind, lo, hi, n) in (('b_pos', 0.06, 0.26, 24),
                              ('b_neg', -0.19, -0.03, nbneg),
                              ('c', -0.043, 0.078, 16)):
        if only and kind not in only.split(','):
            continue
        for k in range(n):
            chunks.append((kind, lo + (hi - lo) * k / n,
                           lo + (hi - lo) * (k + 1) / n))
    t0 = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    tmp = tempfile.mkdtemp(prefix='b3bc_')
    procs = {}
    pending = list(enumerate(chunks))
    tries = {}
    results = {}
    while pending or procs:
        while pending and len(procs) < nw:
            i, (kind, lo, hi) = pending.pop(0)
            out = os.path.join(tmp, f"{i}.txt")
            flags = getattr(subprocess, 'IDLE_PRIORITY_CLASS', 0)
            p = subprocess.Popen(
                [sys.executable, os.path.join(here, 'block3bc.py'), 'chunk',
                 kind, repr(lo), repr(hi), out],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=flags)
            procs[i] = (p, kind, lo, hi, out)
        time.sleep(5)
        for i in list(procs):
            p, kind, lo, hi, out = procs[i]
            if p.poll() is None:
                continue
            del procs[i]
            if os.path.exists(out):
                results[i] = open(out).read().strip()
                say(f"[{len(results)}/{len(chunks)}] {results[i]}")
            else:
                tries[i] = tries.get(i, 0) + 1
                if tries[i] <= 1:
                    say(f"chunk {i} ({kind} {lo:.4f}) crashed rc={p.returncode}; retrying")
                    pending.append((i, (kind, lo, hi)))
                else:
                    results[i] = f"FAIL {kind} {lo} {hi} crashed twice"
                    say(results[i])
    nfail = sum(1 for r in results.values() if r.startswith('FAIL'))
    allok = bool(okb) and nfail == 0
    say(f"{'ALL PASS' if allok else 'FAILURES'} ({time.time()-t0:.0f}s, "
        f"{nfail} failed chunks)")
    with open(os.path.join('results', 'block3bc.log'), 'w') as f:
        f.write("\n".join(lines) + "\n")
    if not allok:
        raise SystemExit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'par':
        main_par()
    elif len(sys.argv) > 1 and sys.argv[1] == 'kcell':
        main_kcell()
    elif len(sys.argv) > 1 and sys.argv[1] == 'chunk':
        main_chunk()
    elif len(sys.argv) > 1 and sys.argv[1] == 'iso':
        main_iso(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    else:
        main()


# ---------------------------------------------------------------------------
# mean-value evaluation for the negative branch: the ball-lambda evaluation
# of I' wraps by ~25 per unit lambda-width (dependency, not quadrature), so
# cells cannot certify directly.  Instead: dPG over a cell is enclosed by
# dPG(center, thin) +- K w/2, with K a one-time rigorous bound on |PG''|
# over the whole interval (its own ball evaluations are wide but only an
# upper MAGNITUDE is needed).
# ---------------------------------------------------------------------------

_KBOUND = None


def _pgpp_bound(tau_lo=-0.20, tau_hi=-0.02, ncell=6):
    """Rigorous K >= sup |PG''| over the tau-interval (coarse cells)."""
    global _KBOUND
    if _KBOUND is None and os.environ.get('B3BC_K'):
        _KBOUND = float(os.environ['B3BC_K'])
    if _KBOUND is not None:
        return _KBOUND
    K = 0.0
    for k in range(ncell):
        lo = tau_lo + (tau_hi - tau_lo) * k / ncell
        hi = tau_lo + (tau_hi - tau_lo) * (k + 1) / ncell
        v = d2PG_cell(f"{lo:.6f}", f"{hi:.6f}")
        if v is None:
            raise RuntimeError("PG'' bound cell failed domain check")
        vlo, vhi = endpoints(v)
        K = max(K, abs(float(vlo)), abs(float(vhi)))
    _KBOUND = K * 1.0000001
    return _KBOUND


def dPG_cell_neg_mv(tau_lo, tau_hi):
    """Mean-value enclosure of PG' over the cell (negative branch)."""
    K = _pgpp_bound()
    lo = float(tau_lo)
    hi = float(tau_hi)
    mid = 0.5 * (lo + hi)
    c = dPG_cell(f"{mid:.8f}", f"{mid:.8f}", bits=(20, 17))
    if c is None:
        return None
    # lambda half-width of the cell (ell' <= 1.1 on this range is NOT
    # assumed; use the certified lam_cell hull)
    lamb = dsfun.lam_cell(dec(f"{lo:.6f}"), dec(f"{hi:.6f}"))
    llo, lhi = endpoints(lamb)
    w = float(lhi) - float(llo)
    slack = dec(f"{K * w / 2 * 1.0000001:.12f}")
    return c + slack.union(-slack)
