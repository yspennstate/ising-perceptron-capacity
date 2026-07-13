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
import uuid
from fractions import Fraction
from multiprocessing import Pool

os.environ.setdefault('HUANG_GRID_N', '2700')

# rectangle and exclusion (floats; the rigorous code re-derives balls)
import huang_np as nr
import block3bc_exact as exact

# achievable body has |a1| <= E|X| ~ 1.297, |a2| <= E|M| ~ 0.697; pad slightly
A1MAX = 1.31
A2MAX = 0.70
A1S = float(nr.PSI * (1 - nr.Q))
A2S = float(nr.Q)
# Region-I exclusion box (refined once the sweep reports failures near it)
EXCL = (0.95, 1.30, 0.45, 0.66)  # a1lo, a1hi, a2lo, a2hi

MIN_SIDE = 0.002
MAX_DEPTH = 7
GRID_N = 2700
DIRECT_CHOICE_DIGITS = 6
MEAN_VALUE_CHOICE_DIGITS = 8
TILT_MIN = 0
TILT_MAX = 4
DIRECT_LAMBDA_ABS_MAX = 160
MEAN_VALUE_LAMBDA_ABS_STRICT_MAX = 300
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
SCHEMA_VERSION = 4
EVIDENCE_MODEL = 'replayable-exact-rectangle-proof-tree-v2'
_WORKER_RUNTIME = None


def _float_fraction(value):
    """Exact decimal geometry used by every Arb reconstruction in this sweep."""
    return Fraction(str(float(value)))


def _float_record(value):
    return exact.fraction_record(_float_fraction(value))


def cell_record(cell):
    return [_float_record(value) for value in cell]


def exact_mid_radius(lo, hi):
    """Return an Arb center/radius whose interval contains [lo, hi].

    ``lo`` and ``hi`` are the decimal spellings used for the certified cell.
    Re-deriving both center and radius from those same exact rationals avoids
    pairing a rounded center with an uninflated float radius, which can miss a
    declared edge by a few ulps.
    """
    from core import dec

    blo = dec(str(lo))
    bhi = dec(str(hi))
    return (blo + bhi) / 2, (bhi - blo) / 2


def _choice_record(text):
    """Canonical exact rational for a decimalized numerical guide choice."""
    return exact.fraction_record(Fraction(text))


def _fixed_decimal(value, digits):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('majorant guide choice is not finite')
    text = f'{value:.{digits}f}'.rstrip('0').rstrip('.')
    return '0' if text in ('', '-0') else text


def _with_majorant_witness(value, witness, enabled):
    return (value, witness) if enabled else value


def majorant_witness_policy():
    return {
        'direct_choice_digits': DIRECT_CHOICE_DIGITS,
        'mean_value_choice_digits': MEAN_VALUE_CHOICE_DIGITS,
        'tilt_min': _float_record(TILT_MIN),
        'tilt_max': _float_record(TILT_MAX),
        'direct_lambda_abs_max': _float_record(DIRECT_LAMBDA_ABS_MAX),
        'mean_value_lambda_abs_strict_max': _float_record(
            MEAN_VALUE_LAMBDA_ABS_STRICT_MAX),
    }


def in_excl(a1lo, a1hi, a2lo, a2hi):
    """True if the cell is fully inside the exclusion box."""
    return (EXCL[0] <= a1lo and a1hi <= EXCL[1]
            and EXCL[2] <= a2lo and a2hi <= EXCL[3])


def overlaps_excl(a1lo, a1hi, a2lo, a2hi):
    return not (a1hi <= EXCL[0] or a1lo >= EXCL[1]
                or a2hi <= EXCL[2] or a2lo >= EXCL[3])


def eval_cell(a1lo, a1hi, a2lo, a2hi, return_witness=False):
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
    if (lam is not None
            and not all(math.isfinite(float(value)) for value in lam)):
        lam = None
    s_star = nr.G(c1, c2)[0]
    if (not math.isfinite(float(s_star))
            or not TILT_MIN <= float(s_star) <= TILT_MAX):
        s_star = 0.0
    # These must be derived from the certified endpoints together.  The old
    # round(c, 8) + raw-float-radius pairing missed one first-cell edge by
    # 3.33e-9, making the mean-value enclosure formally incomplete.
    cc1, rr1 = exact_mid_radius(a1lo, a1hi)
    cc2, rr2 = exact_mid_radius(a2lo, a2hi)
    s_text = _fixed_decimal(s_star, DIRECT_CHOICE_DIGITS)
    s = dec(s_text)
    T = hg.T_meanvalue(cc1, cc2, rr1, rr2, s)
    if T is None:
        return _with_majorant_witness(None, None, return_witness)
    a1b = iv(str(a1lo), str(a1hi))
    a2b = iv(str(a2lo), str(a2hi))
    if lam is not None and abs(lam[0]) <= 80 and abs(lam[1]) <= 80:
        b1_text = _fixed_decimal(lam[0], DIRECT_CHOICE_DIGITS)
        b2_text = _fixed_decimal(lam[1], DIRECT_CHOICE_DIGITS)
        b1 = dec(b1_text)
        b2 = dec(b2_text)
        Phi = hg.Phi_of(b1, b2)
        Hub = Phi - b1 * a1b - b2 * a2b
        witness = {
            'dual_mode': 'fixed_tangent',
            'lambda1': _choice_record(b1_text),
            'lambda2': _choice_record(b2_text),
            'tilt_s': _choice_record(s_text),
        }
    else:
        # No usable dual: the cell is near (or beyond) the boundary of the
        # moment body, where the true dual explodes along the outward normal
        # u.  H(a) <= Phi(kappa u) - kappa u.a for ANY fixed kappa, and
        # Phi(kappa u) = kappa h(u) + E log(1+e^{-2 kappa|u.f|}), so for
        # supported (or infeasible) a this drops to ~0 (or -infinity) as
        # kappa grows, at the price of kappa * (cell radius) wrapping.  Try
        # a ladder of kappa and keep the best; fall back to [0, log 2].
        Hub = arb(0).union(hg.LOG2)
        witness = {
            'dual_mode': 'entropy_cap',
            'tilt_s': _choice_record(s_text),
        }
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
                b1_text = _fixed_decimal(
                    kap * u[0], DIRECT_CHOICE_DIGITS)
                b2_text = _fixed_decimal(
                    kap * u[1], DIRECT_CHOICE_DIGITS)
                b1 = dec(b1_text)
                b2 = dec(b2_text)
                cand = hg.Phi_of(b1, b2) - b1 * a1b - b2 * a2b
                _, chi = _ep(cand)
                _, hhi = _ep(Hub)
                if chi < hhi:
                    Hub = cand
                    witness = {
                        'dual_mode': 'fixed_tangent',
                        'lambda1': _choice_record(b1_text),
                        'lambda2': _choice_record(b2_text),
                        'tilt_s': _choice_record(s_text),
                    }
    value = Hub + s * s * PSI / 2 + ALPHA * T
    return _with_majorant_witness(value, witness, return_witness)


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


def eval_cell_mv(a1lo, a1hi, a2lo, a2hi, return_witness=False):
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
    lam = nr.dual_of(c1, c2)
    if (lam is None or not all(math.isfinite(float(value)) for value in lam)
            or abs(lam[0]) >= MEAN_VALUE_LAMBDA_ABS_STRICT_MAX
            or abs(lam[1]) >= MEAN_VALUE_LAMBDA_ABS_STRICT_MAX):
        return _with_majorant_witness(None, None, return_witness)
    s_star = nr.G(c1, c2)[0]
    if (not math.isfinite(float(s_star))
            or not TILT_MIN <= float(s_star) <= TILT_MAX):
        return _with_majorant_witness(None, None, return_witness)
    b1_text = _fixed_decimal(lam[0], MEAN_VALUE_CHOICE_DIGITS)
    b2_text = _fixed_decimal(lam[1], MEAN_VALUE_CHOICE_DIGITS)
    s_text = _fixed_decimal(s_star, MEAN_VALUE_CHOICE_DIGITS)
    if (abs(Fraction(b1_text)) >= MEAN_VALUE_LAMBDA_ABS_STRICT_MAX
            or abs(Fraction(b2_text))
            >= MEAN_VALUE_LAMBDA_ABS_STRICT_MAX):
        return _with_majorant_witness(None, None, return_witness)
    b1 = dec(b1_text)
    b2 = dec(b2_text)
    s = dec(s_text)
    witness = {
        'lambda1': _choice_record(b1_text),
        'lambda2': _choice_record(b2_text),
        'tilt_s': _choice_record(s_text),
    }
    cc1, rr1 = exact_mid_radius(a1lo, a1hi)
    cc2, rr2 = exact_mid_radius(a2lo, a2hi)
    # thin center value
    Tc = hg.T_of(cc1, cc2, s)
    if Tc is None:
        return _with_majorant_witness(None, None, return_witness)
    center = (hg.Phi_of(b1, b2) - b1 * cc1 - b2 * cc2
              + s * s * PSI / 2 + ALPHA * Tc)
    # gradient of the majorant over the cell (s fixed)
    a1b = dec(str(a1lo)).union(dec(str(a1hi)))
    a2b = dec(str(a2lo)).union(dec(str(a2hi)))
    gz = hg.get_zt_grid()
    Td = hh._T_derivs(a1b, a2b, gz, s)
    if Td is None:
        return _with_majorant_witness(None, None, return_witness)
    g1 = -b1 + ALPHA * Td['dT1']
    g2 = -b2 + ALPHA * Td['dT2']

    def absup(x):
        # The upper endpoint of |x| is already an outward-rounded Arb bound;
        # do not serialize through float/round here.
        _, hi = endpoints(abs(x))
        return hi

    slack = absup(g1) * rr1 + absup(g2) * rr2
    return _with_majorant_witness(
        center + slack, witness, return_witness)


def cert_cell(job, depth=0):
    """Return an exact binary proof tree for one declared rectangle."""
    import huanggrid as hg
    a1lo, a1hi, a2lo, a2hi = job
    geometry = cell_record(job)
    if in_excl(a1lo, a1hi, a2lo, a2hi):
        return {'kind': 'delegate_sweep2', 'cell': geometry}
    outside = hg.outside_K_witness(a1lo, a1hi, a2lo, a2hi)
    if outside is not None:
        return {'kind': 'outside_K', 'cell': geometry, 'witness': outside}
    val, majorant_witness = eval_cell(
        a1lo, a1hi, a2lo, a2hi, return_witness=True)
    method = 'direct'
    if not (val is not None and val < 0):
        # small cells near the maximizer: retry in mean-value form
        if (a1hi - a1lo) < 0.004 and (a2hi - a2lo) < 0.004:
            val2, witness2 = eval_cell_mv(
                a1lo, a1hi, a2lo, a2hi, return_witness=True)
            if val2 is not None:
                val = val2
                majorant_witness = witness2
                method = 'mean_value'
    if val is not None and val < 0:
        return {
            'kind': 'negative',
            'cell': geometry,
            'method': method,
            'majorant_witness': majorant_witness,
            'value_packet': exact.arb_packet(val),
        }
    if (a1hi - a1lo) < MIN_SIDE and (a2hi - a2lo) < MIN_SIDE:
        # cannot refine further; if it overlaps EXCL treat as Region I, else fail
        if overlaps_excl(a1lo, a1hi, a2lo, a2hi):
            return {'kind': 'delegate_sweep2', 'cell': geometry}
        return {'kind': 'failure', 'cell': geometry}
    # bisect the longer side
    if (a1hi - a1lo) >= (a2hi - a2lo):
        m = (a1lo + a1hi) / 2
        axis = 1
        subs = [(a1lo, m, a2lo, a2hi), (m, a1hi, a2lo, a2hi)]
    else:
        m = (a2lo + a2hi) / 2
        axis = 2
        subs = [(a1lo, a1hi, a2lo, m), (a1lo, a1hi, m, a2hi)]
    return {
        'kind': 'split',
        'cell': geometry,
        'axis': axis,
        'split_at': _float_record(m),
        'children': [cert_cell(subs[0], depth + 1),
                     cert_cell(subs[1], depth + 1)],
    }


def tree_counts(tree):
    if tree['kind'] == 'split':
        left = tree_counts(tree['children'][0])
        right = tree_counts(tree['children'][1])
        return tuple(a + b for a, b in zip(left, right))
    return (int(tree['kind'] == 'negative'),
            int(tree['kind'] == 'outside_K'),
            int(tree['kind'] == 'delegate_sweep2'),
            int(tree['kind'] == 'failure'))


def worker(job):
    from core import set_prec
    set_prec(50)
    source_before = exact.source_hashes(proof_source_paths())
    runtime = (_WORKER_RUNTIME if _WORKER_RUNTIME is not None else
               exact.runtime_record(50, 1))
    if len(job) == 5:
        index, job = job[0], job[1:]
    else:
        index = None
    t0 = time.time_ns()
    tree = cert_cell(job)
    source_after = exact.source_hashes(proof_source_paths())
    if source_after != source_before:
        raise RuntimeError('Sweep1 worker source changed mid-job')
    counts = tree_counts(tree)
    return {
        'index': index,
        'cell': cell_record(job),
        'verdict': 'PASS' if counts[3] == 0 else 'FAIL',
        'tree': tree,
        'runtime_milliseconds': (time.time_ns() - t0) // 1_000_000,
        'worker_source_sha256_before': source_before,
        'worker_source_sha256_after': source_after,
        'worker_runtime': runtime,
    }


def build_jobs(coarse=48):
    """Deterministic declared top-cell schedule."""
    jobs = []
    da1 = 2 * A1MAX / coarse
    na2 = max(1, int(2 * A2MAX / da1))
    da2 = 2 * A2MAX / na2
    for i in range(coarse):
        for j in range(na2):
            jobs.append((-A1MAX + da1 * i, -A1MAX + da1 * (i + 1),
                         -A2MAX + da2 * j, -A2MAX + da2 * (j + 1)))
    return jobs, da1, da2


def schedule_records(coarse=48):
    return [
        {'index': index, 'cell': cell_record(job)}
        for index, job in enumerate(build_jobs(coarse)[0])]


def proof_source_paths():
    import core
    import huanggrid as hg
    import huang_hessian as hh
    return {'huang_sweep.py': __file__,
            'block3bc_exact.py': exact.__file__, 'core.py': core.__file__,
            'huanggrid.py': hg.__file__, 'huang_hessian.py': hh.__file__,
            'huang_np.py': nr.__file__}


def main():
    import huanggrid as hg
    from core import set_prec
    set_prec(50)
    if hg.GRID_N != GRID_N:
        raise RuntimeError(
            f'HUANG_GRID_N must be exactly {GRID_N}, got {hg.GRID_N}')
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, os.cpu_count() - 2)
    coarse = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    if coarse != 48:
        raise ValueError('final Sweep1 proof policy requires coarse=48')
    if nw <= 0:
        raise ValueError('Sweep1 worker count must be positive')
    jobs, da1, da2 = build_jobs(coarse)
    print(f"{len(jobs)} top cells, {nw} workers, "
          f"cell {da1:.3f}x{da2:.3f}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    started = time.time()
    run_id = uuid.uuid4().hex
    sources = exact.source_hashes(proof_source_paths())
    runtime = exact.runtime_record(50, nw, fresh_flint=True)
    exact.validate_runtime_record(runtime, 50, workers=nw)
    fails = 0
    records = []
    schedule = schedule_records(coarse)
    with Pool(nw, initializer=_init) as pool, \
            open(os.path.join(RESULTS_DIR, 'huang_sweep.log'), 'w',
                 encoding='utf-8', newline='\n') as log:
        indexed_jobs = [(index, *job) for index, job in enumerate(jobs)]
        for k, res in enumerate(pool.imap_unordered(worker, indexed_jobs)):
            res['run_id'] = run_id
            if (res['worker_source_sha256_before'] != sources
                    or res['worker_source_sha256_after'] != sources
                    or exact.runtime_identity(res['worker_runtime'])
                    != exact.runtime_identity(runtime)):
                raise RuntimeError('Sweep1 worker attestation mismatch')
            records.append(res)
            if res['verdict'] != 'PASS':
                fails += 1
                line = f"FAIL index={res['index']}"
                print(line, flush=True)
                log.write(line + "\n"); log.flush()
            if (k + 1) % 40 == 0:
                print(f"  {k+1}/{len(jobs)} top cells, {fails} fails, "
                      f"{time.time()-started:.0f}s", flush=True)
        log.write(f"jobs={len(records)} fails={fails}\n")
    records.sort(key=lambda row: row['index'])
    sources_after = exact.source_hashes(proof_source_paths())
    runtime_after = exact.runtime_record(50, nw, fresh_flint=True)
    if sources_after != sources or runtime_after != runtime:
        raise RuntimeError('Sweep1 producer source/runtime changed mid-run')
    totals = [tree_counts(record['tree']) for record in records]
    import huanggrid as hg
    h10 = hg.h_support_upper(1, 0)
    h01 = hg.h_support_upper(0, 1)
    payload = {
        'schema_version': SCHEMA_VERSION,
        'kind': 'huang_sweep1_certificate',
        'evidence_model': EVIDENCE_MODEL,
        'run_id': run_id,
        'source_sha256': sources,
        'source_sha256_after': sources_after,
        'source_set_sha256': exact.payload_sha256(sources, omit=()),
        'runtime': runtime,
        'policy': {
            'coarse': coarse,
            'A1MAX': _float_record(A1MAX),
            'A2MAX': _float_record(A2MAX),
            'EXCL': [_float_record(x) for x in EXCL],
            'MIN_SIDE': _float_record(MIN_SIDE),
            'huang_grid_n': GRID_N,
            'majorant_witness': majorant_witness_policy(),
        },
        'schedule': schedule,
        'schedule_sha256': exact.payload_sha256(schedule, omit=()),
        'domain_guards': {
            'h10_packet': exact.arb_packet(h10),
            'h10_limit': _float_record(A1MAX),
            'h01_packet': exact.arb_packet(h01),
            'h01_limit': _float_record(A2MAX),
            'degenerate_identity': 'huang-analytic-Sstar(1,0)=0',
        },
        'records': records,
        'derived_summary': {
            'jobs': len(records),
            'failures': fails,
            'negative_leaves': sum(x[0] for x in totals),
            'outside_K_leaves': sum(x[1] for x in totals),
            'delegated_leaves': sum(x[2] for x in totals),
        },
    }
    payload['certificate_sha256'] = exact.payload_sha256(
        payload, omit=('certificate_sha256',))
    exact.write_json_atomic(
        os.path.join(RESULTS_DIR, 'huang_sweep.json'), payload)
    print(f"DONE: {len(jobs)} top cells, {fails} fails, "
          f"{time.time()-started:.0f}s", flush=True)
    if fails:
        raise SystemExit(1)


def _init():
    global _WORKER_RUNTIME
    from core import set_prec
    set_prec(50)
    exact.apply_worker_policy()
    _WORKER_RUNTIME = exact.runtime_record(50, 1, fresh_flint=True)


if __name__ == '__main__':
    main()
