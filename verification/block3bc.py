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
import argparse
from multiprocessing import Pool
from fractions import Fraction

import flint
from flint import arb
from core import set_prec, dec, endpoints, report, PSI, Q, ALPHA
import dsfun
import block3bc_exact as exact
from block3bc_exact import (SCHEMA_VERSION, arb_packet, as_fraction,
                            apply_worker_policy,
                            b_neg_boundaries, b_pos_boundaries, c_boundaries,
                            file_sha256, fraction_arb, fraction_record,
                            fraction_from_record, fraction_text,
                            intervals_from_boundaries,
                             isolated_subprocess_results, lane_indices,
                             load_json, parse_fraction_text,
                             packet_fraction_endpoints, payload_sha256,
                             runtime_record, source_hashes,
                             validate_runtime_record,
                             write_json_atomic)

MIN_W = Fraction(1, 5000)
MIN_W_BNEG = Fraction(1, 200000)
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')


def _replay_source_paths():
    return {
        'block3bc.py': __file__,
        'block3bc_exact.py': exact.__file__,
        'core.py': sys.modules['core'].__file__,
        'dsfun.py': dsfun.__file__,
    }


def _cell(tau_lo, tau_hi):
    tau_lo, tau_hi = as_fraction(tau_lo), as_fraction(tau_hi)
    if not tau_lo <= tau_hi:
        raise ValueError("inverted tau cell")
    lo, hi = fraction_arb(tau_lo), fraction_arb(tau_hi)
    A = dsfun.A_of_tau(lo).union(dsfun.A_of_tau(hi))
    lam = dsfun.lam_cell(lo, hi)
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


def _adaptive(tau_lo, tau_hi, evalf, want, tag, min_w=None,
              return_leaves=False):
    """Certify a sign on exact rational cells using dyadic bisection."""
    tau_lo, tau_hi = as_fraction(tau_lo), as_fraction(tau_hi)
    if min_w is None:
        min_w = MIN_W
    min_w = as_fraction(min_w)
    stack = [(tau_lo, tau_hi)]
    leaves = []
    while stack:
        lo, hi = stack.pop()
        v = evalf(lo, hi)
        ok = False
        if v is not None:
            vlo, vhi = endpoints(v)
            ok = (vhi < 0) if want == '<0' else (vlo > 0)
        if ok:
            leaves.append(dict(tau_lo=fraction_record(lo),
                               tau_hi=fraction_record(hi),
                               value=arb_packet(v), sign=want))
            continue
        if (hi - lo) <= min_w:
            bad = dict(tau_lo=fraction_record(lo),
                       tau_hi=fraction_record(hi),
                       value=None if v is None else arb_packet(v))
            return False, leaves if return_leaves else len(leaves), bad
        m = (lo + hi) / 2
        stack.append((lo, m))
        stack.append((m, hi))
    leaves.sort(key=lambda row: fraction_from_record(row['tau_lo']))
    return True, leaves if return_leaves else len(leaves), None


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
                creationflags=(getattr(sp0, 'CREATE_NO_WINDOW', 0)
                               | getattr(sp0, 'BELOW_NORMAL_PRIORITY_CLASS', 0))), out0))
        Kv = 0.0
        for p0, out0 in ps:
            p0.wait()
            Kv = max(Kv, float(open(out0).read().strip()))
        os.environ['B3BC_K'] = f"{Kv * 1.0000001:.8f}"
        say(f"|PG''| bound K = {os.environ['B3BC_K']}")
    rng = os.environ.get('B3BC_RANGE')   # 'lo,hi' tau-override for b_neg
    bneg_lo, bneg_hi = -0.19, -0.03
    if rng:
        bneg_lo, bneg_hi = (float(x) for x in rng.split(','))
    nbneg = 20
    if os.environ.get('B3BC_K'):
        # size b_neg cells so one mean-value eval certifies:
        # K w/2 + eval width (~6e-4) <= margin (~2e-3); lam ~ 0.55 tau here
        Kv0 = float(os.environ['B3BC_K'])
        wlam = 2 * (0.0020 - 0.0006) / max(Kv0, 0.1)
        nbneg = max(20, int((bneg_hi - bneg_lo) / (wlam / 0.55)) + 1)
    chunks = []
    for (kind, lo, hi, n) in (('b_pos', 0.06, 0.26, 24),
                              ('b_neg', bneg_lo, bneg_hi, nbneg),
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
            flags = (getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                     | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
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


# ---------------------------------------------------------------------------
# mean-value evaluation for the negative branch: the ball-lambda evaluation
# of I' wraps by ~25 per unit lambda-width (dependency, not quadrature), so
# cells cannot certify directly.  Instead: dPG over a cell is enclosed by
# dPG(center, thin) +- K w/2, with K a one-time rigorous bound on |PG''|
# over the whole interval (its own ball evaluations are wide but only an
# upper MAGNITUDE is needed).
# ---------------------------------------------------------------------------

_KBOUND = None


def _pgpp_bound():
    """Return the exact rational K bound selected by the fresh aux proof."""
    global _KBOUND
    if _KBOUND is None and os.environ.get('B3BC_K'):
        _KBOUND = parse_fraction_text(os.environ['B3BC_K'])
    if _KBOUND is not None:
        return _KBOUND
    raise RuntimeError("fresh Block3bc auxiliary manifest/K is required")


def dPG_cell_neg_mv(tau_lo, tau_hi, k_bound=None):
    """Mean-value enclosure of PG' over the cell (negative branch):
    dPG(center) + PG''(xi)(lam - lam_center) with |PG''| <= K.  The
    distance factor makes no centering assumption: it is the certified
    maximum of |lam - lam_center| over the cell, from the lam hulls of
    the cell and of its exact rational midpoint."""
    k_bound = _pgpp_bound() if k_bound is None else as_fraction(k_bound)
    K = fraction_arb(k_bound)
    lo, hi = as_fraction(tau_lo), as_fraction(tau_hi)
    if not lo <= hi:
        raise ValueError("inverted negative-branch cell")
    mid = (lo + hi) / 2
    c = dPG_cell(mid, mid, bits=(20, 17))
    if c is None:
        return None
    lamb = dsfun.lam_cell(fraction_arb(lo), fraction_arb(hi))
    llo, lhi = endpoints(lamb)
    r = dsfun.lam_cell(fraction_arb(mid), fraction_arb(mid))
    rlo, rhi = endpoints(r)
    dist = arb(lhi) - arb(rlo)
    d2 = arb(rhi) - arb(llo)
    if d2 > dist:
        dist = d2
    slack = K * dist
    return c + slack.union(-slack)


# ---------------------------------------------------------------------------
# Source-bound exact replay path.  Legacy text drivers above remain only for
# historical inspection; release evidence is produced exclusively here.
# ---------------------------------------------------------------------------

def _part_boundaries(part, k_run):
    if part == 'b_pos':
        return b_pos_boundaries()
    if part == 'c':
        return c_boundaries()
    if part == 'b_neg':
        return b_neg_boundaries(k_run)
    raise ValueError(f"unknown Block3bc part {part}")


def _require_leaf_cover(leaves, top_lo, top_hi):
    reach = as_fraction(top_lo)
    seen = set()
    for leaf in leaves:
        lo = fraction_from_record(leaf['tau_lo'])
        hi = fraction_from_record(leaf['tau_hi'])
        if (lo, hi) in seen or lo != reach or not lo < hi:
            raise ValueError("adaptive leaves have a duplicate, gap, or overlap")
        seen.add((lo, hi))
        reach = hi
    if reach != as_fraction(top_hi):
        raise ValueError("adaptive leaves do not reach the top-cell endpoint")


def _replay_top_cell(job):
    part, index, lo_record, hi_record, k_record = job
    set_prec(60)
    lo, hi = fraction_from_record(lo_record), fraction_from_record(hi_record)
    k_run = fraction_from_record(k_record)
    t0 = time.time_ns()
    if part == 'b_pos':
        evalf, want, min_w = dPG_cell, '<0', MIN_W
    elif part == 'c':
        evalf, want, min_w = d2PG_cell, '<0', MIN_W
    elif part == 'b_neg':
        evalf = lambda a, b: dPG_cell_neg_mv(a, b, k_run)
        want, min_w = '>0', MIN_W_BNEG
    else:
        raise ValueError(part)
    ok, leaves, bad = _adaptive(lo, hi, evalf, want, part,
                                min_w=min_w, return_leaves=True)
    if ok:
        _require_leaf_cover(leaves, lo, hi)
    return dict(index=int(index), tau_lo=fraction_record(lo),
                tau_hi=fraction_record(hi), ok=bool(ok), leaves=leaves,
                failure=bad,
                runtime_milliseconds=(time.time_ns() - t0) // 1_000_000)


def _replay_job_record(part, result, job_input, aux_hash,
                       source_before=None, source_after=None, runtime=None):
    if source_before is None:
        source_before = source_hashes(_replay_source_paths())
    if source_after is None:
        source_after = source_before
    if runtime is None:
        runtime = runtime_record(60, 1)
    result = _validate_replay_child(
        part, result.get('index'), job_input, result)
    payload = {
        'schema_version': SCHEMA_VERSION,
        'kind': 'block3bc_replay_job',
        'part': part,
        'index': result['index'],
        'input': job_input,
        'aux_manifest_sha256': aux_hash,
        'source_sha256': source_before,
        'source_sha256_after': source_after,
        'runtime': runtime,
        'result': result,
    }
    payload['job_sha256'] = payload_sha256(payload, omit=('job_sha256',))
    return payload


def _validate_replay_job(data, part, index, job_input, aux_hash, hashes,
                         require_current_runtime=True,
                         containing_runtime=None):
    required = {
        'schema_version', 'kind', 'part', 'index', 'input',
        'aux_manifest_sha256', 'source_sha256', 'source_sha256_after',
        'runtime', 'result', 'job_sha256',
    }
    if (not isinstance(data, dict) or set(data) != required
            or not isinstance(data.get('index'), int)
            or isinstance(data.get('index'), bool)
            or data.get('schema_version') != SCHEMA_VERSION
            or data.get('kind') != 'block3bc_replay_job'
            or data.get('part') != part
            or data.get('index') != index
            or data.get('input') != job_input
            or data.get('aux_manifest_sha256') != aux_hash
            or data.get('source_sha256') != hashes
            or data.get('source_sha256_after') != hashes
            or data.get('job_sha256') != payload_sha256(
                data, omit=('job_sha256',))):
        raise ValueError(f"stale/corrupt replay record {part} {index}")
    runtime = data.get('runtime', {})
    try:
        validate_runtime_record(runtime, 60, workers=1)
    except ValueError as exc:
        raise ValueError(
            f"incompatible replay runtime {part} {index}") from exc
    if require_current_runtime and runtime != runtime_record(60, 1):
        raise ValueError(f"incompatible replay runtime {part} {index}")
    if (containing_runtime is not None
            and exact.runtime_identity(runtime)
            != exact.runtime_identity(containing_runtime)):
        raise ValueError(f"replay job/shard runtime mismatch {part} {index}")
    result = data.get('result', {})
    if result.get('index') != index:
        raise ValueError("replay resume result index mismatch")
    return _validate_replay_child(part, index, job_input, result)


def _validate_replay_child(part, index, expected, result):
    if (not isinstance(result, dict)
            or not isinstance(result.get('index'), int)
            or isinstance(result.get('index'), bool)
            or result.get('index') != index):
        raise ValueError('replay result index/schema mismatch')
    if set(result) != {
            'index', 'tau_lo', 'tau_hi', 'ok', 'leaves', 'failure',
            'runtime_milliseconds'}:
        raise ValueError('replay result schema mismatch')
    if not isinstance(result['ok'], bool):
        raise ValueError('replay result has non-boolean status')
    runtime_ms = result['runtime_milliseconds']
    if (not isinstance(runtime_ms, int) or isinstance(runtime_ms, bool)
            or runtime_ms < 0):
        raise ValueError('invalid replay child runtime')
    if (result['tau_lo'] != expected['tau_lo']
            or result['tau_hi'] != expected['tau_hi']
            or not isinstance(result['leaves'], list)):
        raise ValueError('replay result/input mismatch')
    top_lo = fraction_from_record(expected['tau_lo'])
    top_hi = fraction_from_record(expected['tau_hi'])
    want = '>0' if part == 'b_neg' else '<0'
    if result['ok']:
        if result['failure'] is not None:
            raise ValueError('successful replay result carries a failure')
        _require_leaf_cover(result['leaves'], top_lo, top_hi)
        for leaf in result['leaves']:
            if (not isinstance(leaf, dict)
                    or set(leaf) != {'tau_lo', 'tau_hi', 'value', 'sign'}
                    or leaf['sign'] != want):
                raise ValueError('replay leaf schema/sign mismatch')
            value_lo, value_hi = packet_fraction_endpoints(leaf['value'])
            if not ((value_hi < 0) if want == '<0' else (value_lo > 0)):
                raise ValueError('replay leaf packet does not certify its sign')
    elif not isinstance(result['failure'], dict):
        raise ValueError('failed replay result has no failure witness')
    return result


def replay_part(part, aux_manifest_path, workers, lane=0, lanes=1,
                output=None, timeout_seconds=14400, retries=1):
    """Replay one exact part/lane and write a strict source-bound artifact."""
    from block3bc_aux_verify import verify_manifest

    aux = verify_manifest(aux_manifest_path, require_complete=True)
    if (not isinstance(workers, int) or isinstance(workers, bool)
            or workers <= 0):
        raise ValueError('workers must be a positive plain integer')
    worker_runtime = runtime_record(60, 1, fresh_flint=True)
    validate_runtime_record(worker_runtime, 60, workers=1)
    k_run = aux['k_run']
    aux_hash = aux['manifest']['manifest_sha256']
    boundaries0 = _part_boundaries(part, k_run)
    intervals = intervals_from_boundaries(boundaries0)
    indices = lane_indices(len(intervals), lane, lanes)
    jobs = [(part, i, fraction_record(intervals[i][0]),
             fraction_record(intervals[i][1]), fraction_record(k_run))
            for i in indices]
    hashes = source_hashes(_replay_source_paths())
    if output is None:
        outdir = os.path.join(RESULTS_DIR, 'block3bc_replay')
        output = os.path.join(
            outdir, f'{part}.lane-{int(lane)}-of-{int(lanes)}.json')
    output = os.path.abspath(output)
    record_dir = output + '.records'
    os.makedirs(record_dir, exist_ok=True)
    job_inputs = {
        i: {'tau_lo': lo_record, 'tau_hi': hi_record,
            'k_run': fraction_record(k_run)}
        for _, i, lo_record, hi_record, _ in jobs
    }
    records_by_index = {}
    record_payloads = {}
    pending = []
    for job in jobs:
        i = job[1]
        path = os.path.join(record_dir, f'{part}-{i:04d}.json')
        if os.path.exists(path):
            saved = load_json(path)
            result = _validate_replay_job(
                saved, part, i, job_inputs[i], aux_hash, hashes)
            records_by_index[i] = result
            record_payloads[i] = (path, saved)
        else:
            pending.append(job)
    worker_count = max(1, min(workers, len(pending) or 1))
    specs = []
    for job in pending:
        _, i, lo_record, hi_record, k_record = job
        child_args = [
            '_replay_job', '--part', part, '--index', str(i),
            '--aux-manifest-sha256=' + aux_hash,
            '--lo=' + fraction_text(fraction_from_record(lo_record)),
            '--hi=' + fraction_text(fraction_from_record(hi_record)),
            '--k-run=' + fraction_text(fraction_from_record(k_record))]
        command = exact.isolated_python_command(
            __file__, worker_runtime, child_args)
        specs.append((i, command))

    def validate_child(index, saved):
        _validate_replay_job(
            saved, part, index, job_inputs[index], aux_hash, hashes,
            require_current_runtime=True)

    for i, saved in isolated_subprocess_results(
            specs, worker_count, record_dir,
            timeout_seconds=timeout_seconds, retries=retries,
            result_validator=validate_child):
        result = saved['result']
        if result.get('index') != i:
            raise ValueError("isolated replay result index mismatch")
        path = os.path.join(record_dir, f'{part}-{i:04d}.json')
        write_json_atomic(path, saved, overwrite=False)
        records_by_index[i] = result
        record_payloads[i] = (path, saved)
    if set(records_by_index) != set(indices):
        raise ValueError("incomplete replay job-record set")
    source_after = source_hashes(_replay_source_paths())
    worker_runtime_after = runtime_record(60, 1, fresh_flint=True)
    if source_after != hashes or worker_runtime_after != worker_runtime:
        raise RuntimeError('replay producer source/runtime changed mid-run')
    shard_runtime = runtime_record(
        60, worker_count, fresh_flint=True)
    if (exact.runtime_identity(shard_runtime)
            != exact.runtime_identity(worker_runtime)):
        raise RuntimeError('replay shard/worker runtime identity mismatch')
    records = [records_by_index[i] for i in indices]
    job_artifacts = []
    for i in indices:
        path, saved = record_payloads[i]
        job_artifacts.append({
            'index': i,
            'file': os.path.relpath(path, os.path.dirname(output)).replace('\\', '/'),
            'file_sha256': file_sha256(path),
            'job_sha256': saved['job_sha256'],
        })
    payload = {
        'schema_version': SCHEMA_VERSION,
        'kind': 'block3bc_replay_shard',
        'part': part,
        'lane': {'index': int(lane), 'count': int(lanes)},
        'indices': indices,
        'schedule_boundaries': [fraction_record(x) for x in boundaries0],
        'schedule_sha256': payload_sha256(
            [fraction_record(x) for x in boundaries0], omit=()),
        'k_run': fraction_record(k_run),
        'aux_manifest_sha256': aux_hash,
        'source_sha256': hashes,
        'runtime': shard_runtime,
        'tolerances': {'precision_bits': 60,
                       'I_prime_inner_bits': 20,
                       'I_prime_outer_bits': 17,
                       'I_second_inner_bits': 14,
                       'I_second_outer_bits': 12},
        'records': records,
        'job_artifacts': job_artifacts,
        'failures': sum(not row['ok'] for row in records),
    }
    payload['artifact_sha256'] = payload_sha256(
        payload, omit=('artifact_sha256',))
    write_json_atomic(output, payload)
    if payload['failures']:
        raise RuntimeError(f"{payload['failures']} {part} top cells failed")
    return output, payload


def main_replay(argv=None):
    parser = argparse.ArgumentParser(
        description='Exact manifested Ding-Sun Block3bc replay')
    parser.add_argument('--part', required=True,
                        choices=('b_pos', 'b_neg', 'c'))
    parser.add_argument('--aux-manifest', required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--lane', type=int, default=0)
    parser.add_argument('--lanes', type=int, default=1)
    parser.add_argument('--timeout-seconds', type=int, default=14400)
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    output, payload = replay_part(
        args.part, args.aux_manifest, args.workers,
        lane=args.lane, lanes=args.lanes, output=args.output,
        timeout_seconds=args.timeout_seconds, retries=args.retries)
    print(f"PASS {args.part}: {len(payload['records'])} top cells, "
          f"{sum(len(x['leaves']) for x in payload['records'])} leaves; "
          f"{output}", flush=True)


def main_replay_job(argv=None):
    parser = argparse.ArgumentParser(description='isolated Block3bc top cell')
    parser.add_argument('--part', required=True,
                        choices=('b_pos', 'b_neg', 'c'))
    parser.add_argument('--index', required=True, type=int)
    parser.add_argument('--aux-manifest-sha256', required=True)
    parser.add_argument('--lo', required=True)
    parser.add_argument('--hi', required=True)
    parser.add_argument('--k-run', required=True)
    parser.add_argument('--result-file', required=True)
    args = parser.parse_args(argv)
    apply_worker_policy()
    set_prec(60)
    source_before = source_hashes(_replay_source_paths())
    runtime_before = runtime_record(60, 1, fresh_flint=True)
    validate_runtime_record(runtime_before, 60, workers=1)
    job_input = {
        'tau_lo': fraction_record(parse_fraction_text(args.lo)),
        'tau_hi': fraction_record(parse_fraction_text(args.hi)),
        'k_run': fraction_record(parse_fraction_text(args.k_run)),
    }
    result = _replay_top_cell((
        args.part, args.index,
        job_input['tau_lo'], job_input['tau_hi'], job_input['k_run']))
    source_after = source_hashes(_replay_source_paths())
    runtime_after = runtime_record(60, 1, fresh_flint=True)
    if source_after != source_before or runtime_after != runtime_before:
        raise RuntimeError('replay child source/runtime changed mid-job')
    if (not isinstance(args.aux_manifest_sha256, str)
            or len(args.aux_manifest_sha256) != 64
            or any(ch not in '0123456789abcdef'
                   for ch in args.aux_manifest_sha256)):
        raise ValueError('invalid auxiliary manifest identity')
    saved = _replay_job_record(
        args.part, result, job_input, args.aux_manifest_sha256,
        source_before, source_after, runtime_before)
    write_json_atomic(args.result_file, saved, overwrite=False)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '_replay_job':
        main_replay_job(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == 'replay':
        main_replay(sys.argv[2:])
    else:
        raise SystemExit(
            'legacy Block3bc drivers are disabled; use the exact replay '
            'subcommand with a fresh auxiliary manifest')
