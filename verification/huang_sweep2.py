"""Stage-2 sweep: certify S_* < 0 on (old EXCL box) minus the Region-I star.

The original Region II sweep (huang_sweep.py) certifies S_* < 0 on K minus
EXCL_OLD = [0.95, 1.30] x [0.45, 0.66].  Region I (huang_region1.py) gives
S_* <= 0 on the star region {a* + t v(th): 0 <= t <= T(th)} with the
angle-dependent radii T_LONG/T_MID/T_CORE around the weak direction.  This
driver closes the remainder: every cell of EXCL_OLD that is not fully inside
the star is certified by the same dual-tangent bound (eval_cell), bisecting
down to MIN_SIDE = 1e-3 near the star boundary.

Star containment is certified without `atan2` or angular sampling.  The
maximal corner radius is enclosed by Arb, and every corner must satisfy the
two affine half-cone inequalities for one common signed weak axis.  Convexity
then puts the whole rectangle inside the shrunken Region-I disk or cone.

Run:  python huang_sweep2.py [nworkers]
"""

import os
import sys
import time
import math
import uuid
from multiprocessing import Pool

import huang_np as nr
import huang_sweep as sw
import block3bc_exact as exact
import huang_region1 as r1
from huang_region1 import (A1S, A2S, W_ANG, WEDGE_HALF, CONE_MID,
                           T_LONG, T_MID, T_CORE)

# Stage-1 (huang_sweep.py) treats MIN_SIDE cells that OVERLAP its exclusion
# box as covered by this stage; those cells extend at most MIN_SIDE_1 = 0.002
# beyond the box, so this stage sweeps the box EXPANDED by 0.004 on every
# side, closing the sliver.
EXCL_OLD = (0.95 - 0.004, 1.30 + 0.004, 0.45 - 0.004, 0.66 + 0.004)
MIN_SIDE = 1e-3
SAFE = 5e-7      # includes the star-origin uncertainty (true a* vs decimals)
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
SCHEMA_VERSION = 4
EVIDENCE_MODEL = 'replayable-exact-rectangle-proof-tree-v2'
_WORKER_RUNTIME = None


_REGION1_STAR_FIELDS = (
    'W_ANG', 'WEDGE_HALF', 'CONE_MID', 'T_LONG', 'T_MID', 'T_CORE',
    'A1S', 'A2S', 'origin_radius')


def region1_star_policy():
    star = {
        'W_ANG': r1._float_record(r1.W_ANG),
        'WEDGE_HALF': r1._float_record(r1.WEDGE_HALF),
        'CONE_MID': r1._float_record(r1.CONE_MID),
        'T_LONG': r1._float_record(r1.T_LONG),
        'T_MID': r1._float_record(r1.T_MID),
        'T_CORE': r1._float_record(r1.T_CORE),
        'A1S': r1._float_record(r1.A1S),
        'A2S': r1._float_record(r1.A2S),
        'origin_radius': r1._decimal_record('0.0000001'),
    }
    if tuple(star) != _REGION1_STAR_FIELDS:
        raise RuntimeError('Region-I star policy field order drifted')
    return star


def star_witness(a1lo, a1hi, a2lo, a2hi):
    """Return a packetized rectangle-in-star witness, or ``None``.

    Cone membership is checked by affine inequalities at all four corners;
    radius and the true-origin uncertainty are enclosed by Arb.  No atan2 or
    sampled angular classifier is part of the certificate.
    """
    from flint import arb
    from core import dec, endpoints, sq_any, sqrt_nonneg

    origin = dec('0.0000001').union(-dec('0.0000001'))
    nominal_a1 = dec(str(A1S))
    nominal_a2 = dec(str(A2S))
    corners = []
    radii = []
    for c1, c2 in ((a1lo, a2lo), (a1lo, a2hi),
                   (a1hi, a2lo), (a1hi, a2hi)):
        # Keep the declared decimal rectangle exact.  Subtracting in
        # binary64 first can move an endpoint by one ulp outside its proof.
        d1 = dec(str(c1)) - nominal_a1 + origin
        d2 = dec(str(c2)) - nominal_a2 + origin
        corners.append((d1, d2))
        radii.append(sqrt_nonneg(sq_any(d1) + sq_any(d2)))
    radius_box = radii[0]
    for value in radii[1:]:
        radius_box = radius_box.union(value)

    def radius_below(limit):
        _, upper = endpoints(radius_box)
        return upper < dec(str(limit))

    core_limit = T_CORE - SAFE
    if radius_below(core_limit):
        return {
            'mode': 'core_radius',
            'radius_max_packet': exact.arb_packet(radius_box),
            'radius_limit': sw._float_record(core_limit),
        }

    axis_angle = dec(str(W_ANG))
    w1 = axis_angle.cos()
    w2 = axis_angle.sin()

    def cone(mode, half_angle, radius_limit):
        if not radius_below(radius_limit):
            return None
        tangent = dec(str(half_angle)).tan()
        for sign in (1, -1):
            projections, left_guards, right_guards = [], [], []
            for d1, d2 in corners:
                projection = sign * (w1 * d1 + w2 * d2)
                cross = sign * (w1 * d2 - w2 * d1)
                projections.append(projection)
                left_guards.append(tangent * projection + cross)
                right_guards.append(tangent * projection - cross)
            def hull(values):
                out = values[0]
                for value in values[1:]:
                    out = out.union(value)
                return out
            projection_box = hull(projections)
            left_box = hull(left_guards)
            right_box = hull(right_guards)
            if (endpoints(projection_box)[0] > 0
                    and endpoints(left_box)[0] > 0
                    and endpoints(right_box)[0] > 0):
                return {
                    'mode': mode,
                    'axis_sign': sign,
                    'half_angle': sw._float_record(half_angle),
                    'radius_max_packet': exact.arb_packet(radius_box),
                    'radius_limit': sw._float_record(radius_limit),
                    'projection_min_packet': exact.arb_packet(projection_box),
                    'left_cone_packet': exact.arb_packet(left_box),
                    'right_cone_packet': exact.arb_packet(right_box),
                }
        return None

    long_witness = cone(
        'long_cone', WEDGE_HALF - 0.012 - SAFE, T_LONG - SAFE)
    if long_witness is not None:
        return long_witness
    return cone('mid_cone', CONE_MID - 0.030 - SAFE, T_MID - SAFE)


def in_star(a1lo, a1hi, a2lo, a2hi):
    return star_witness(a1lo, a1hi, a2lo, a2hi) is not None


def cert_cell(job, depth=0):
    a1lo, a1hi, a2lo, a2hi = job
    import huanggrid as hg
    geometry = sw.cell_record(job)
    star = star_witness(a1lo, a1hi, a2lo, a2hi)
    if star is not None:
        return {
            'kind': 'delegate_region1',
            'cell': geometry,
            'witness': star,
        }
    outside = hg.outside_K_witness(a1lo, a1hi, a2lo, a2hi)
    if outside is not None:
        return {'kind': 'outside_K', 'cell': geometry, 'witness': outside}
    val, majorant_witness = sw.eval_cell(
        a1lo, a1hi, a2lo, a2hi, return_witness=True)
    method = 'direct'
    if not (val is not None and val < 0):
        # small cells near the maximizer: mean-value form restores the
        # dual-tangent / constraint gradient cancellation
        if (a1hi - a1lo) < 0.004 and (a2hi - a2lo) < 0.004:
            val2, witness2 = sw.eval_cell_mv(
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
        return {'kind': 'failure', 'cell': geometry}
    if (a1hi - a1lo) >= (a2hi - a2lo):
        m = 0.5 * (a1lo + a1hi)
        axis = 1
        subs = [(a1lo, m, a2lo, a2hi), (m, a1hi, a2lo, a2hi)]
    else:
        m = 0.5 * (a2lo + a2hi)
        axis = 2
        subs = [(a1lo, a1hi, a2lo, m), (a1lo, a1hi, m, a2hi)]
    return {
        'kind': 'split',
        'cell': geometry,
        'axis': axis,
        'split_at': sw._float_record(m),
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
            int(tree['kind'] == 'delegate_region1'),
            int(tree['kind'] == 'failure'))


def worker(job):
    from core import set_prec
    set_prec(50)
    sources = exact.source_hashes(proof_source_paths())
    runtime = (_WORKER_RUNTIME if _WORKER_RUNTIME is not None else
               exact.runtime_record(50, 1))
    if len(job) == 5:
        index, job = job[0], job[1:]
    else:
        index = None
    t0 = time.time_ns()
    tree = cert_cell(job)
    sources_after = exact.source_hashes(proof_source_paths())
    if sources_after != sources:
        raise RuntimeError('Sweep2 worker source changed mid-job')
    counts = tree_counts(tree)
    return {
        'index': index,
        'cell': sw.cell_record(job),
        'verdict': 'PASS' if counts[3] == 0 else 'FAIL',
        'tree': tree,
        'runtime_milliseconds': (time.time_ns() - t0) // 1_000_000,
        'worker_source_sha256_before': sources,
        'worker_source_sha256_after': sources_after,
        'worker_runtime': runtime,
    }


def _init():
    global _WORKER_RUNTIME
    from core import set_prec
    set_prec(50)
    exact.apply_worker_policy()
    _WORKER_RUNTIME = exact.runtime_record(50, 1, fresh_flint=True)


def build_jobs():
    a1lo, a1hi, a2lo, a2hi = EXCL_OLD
    n1, n2 = 20, 12
    jobs = []
    for i in range(n1):
        for j in range(n2):
            jobs.append((a1lo + (a1hi - a1lo) * i / n1,
                         a1lo + (a1hi - a1lo) * (i + 1) / n1,
                         a2lo + (a2hi - a2lo) * j / n2,
                         a2lo + (a2hi - a2lo) * (j + 1) / n2))
    return jobs, n1, n2


def schedule_records():
    return [
        {'index': index, 'cell': sw.cell_record(job)}
        for index, job in enumerate(build_jobs()[0])]


def proof_source_paths():
    import core
    import huanggrid as hg
    import huang_hessian as hh
    import huang_region1 as r1
    return {'huang_sweep2.py': __file__, 'huang_sweep.py': sw.__file__,
            'huang_region1.py': r1.__file__,
            'block3bc_exact.py': exact.__file__, 'core.py': core.__file__,
            'huanggrid.py': hg.__file__, 'huang_hessian.py': hh.__file__,
            'huang_np.py': nr.__file__}


def main():
    import huanggrid as hg
    from core import set_prec
    set_prec(50)
    if hg.GRID_N != sw.GRID_N:
        raise RuntimeError(
            f'HUANG_GRID_N must be exactly {sw.GRID_N}, got {hg.GRID_N}')
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, os.cpu_count() - 2)
    if nw <= 0:
        raise ValueError('Sweep2 worker count must be positive')
    jobs, n1, n2 = build_jobs()
    print(f"{len(jobs)} top cells over EXCL_OLD, {nw} workers, "
          f"MIN_SIDE={MIN_SIDE}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    started = time.time()
    run_id = uuid.uuid4().hex
    sources = exact.source_hashes(proof_source_paths())
    runtime = exact.runtime_record(50, nw, fresh_flint=True)
    exact.validate_runtime_record(runtime, 50, workers=nw)
    fails = 0
    records = []
    schedule = schedule_records()
    with Pool(nw, initializer=_init) as pool, \
            open(os.path.join(RESULTS_DIR, 'huang_sweep2.log'), 'w',
                 encoding='utf-8', newline='\n') as log:
        indexed_jobs = [(index, *job) for index, job in enumerate(jobs)]
        for k, res in enumerate(pool.imap_unordered(worker, indexed_jobs)):
            res['run_id'] = run_id
            if (res['worker_source_sha256_before'] != sources
                    or res['worker_source_sha256_after'] != sources
                    or exact.runtime_identity(res['worker_runtime'])
                    != exact.runtime_identity(runtime)):
                raise RuntimeError('Sweep2 worker attestation mismatch')
            records.append(res)
            if res['verdict'] != 'PASS':
                fails += 1
                line = f"FAIL index={res['index']}"
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            if (k + 1) % 24 == 0:
                print(f"  {k+1}/{len(jobs)} top cells, {fails} fails, "
                      f"{time.time()-started:.0f}s", flush=True)
        log.write(f"jobs={len(records)} fails={fails}\n")
    records.sort(key=lambda row: row['index'])
    sources_after = exact.source_hashes(proof_source_paths())
    runtime_after = exact.runtime_record(50, nw, fresh_flint=True)
    if sources_after != sources or runtime_after != runtime:
        raise RuntimeError('Sweep2 producer source/runtime changed mid-run')
    totals = [tree_counts(record['tree']) for record in records]
    payload = {
        'schema_version': SCHEMA_VERSION,
        'kind': 'huang_sweep2_certificate',
        'evidence_model': EVIDENCE_MODEL,
        'run_id': run_id,
        'source_sha256': sources,
        'source_sha256_after': sources_after,
        'source_set_sha256': exact.payload_sha256(sources, omit=()),
        'runtime': runtime,
        'policy': {
            'n1': n1,
            'n2': n2,
            'EXCL_OLD': [sw._float_record(x) for x in EXCL_OLD],
            'MIN_SIDE': sw._float_record(MIN_SIDE),
            'SAFE': sw._float_record(SAFE),
            'huang_grid_n': sw.GRID_N,
            'majorant_witness': sw.majorant_witness_policy(),
            'region1_star': region1_star_policy(),
            'region1_sdot_model': r1.SDOT_MODEL,
            'region1_sdot_coefficients': [
                r1._decimal_record(text) for text in r1.SDOT_C_TEXT],
        },
        'schedule': schedule,
        'schedule_sha256': exact.payload_sha256(schedule, omit=()),
        'records': records,
        'derived_summary': {
            'jobs': len(records),
            'failures': fails,
            'negative_leaves': sum(x[0] for x in totals),
            'outside_K_leaves': sum(x[1] for x in totals),
            'region1_delegated_leaves': sum(x[2] for x in totals),
        },
    }
    payload['certificate_sha256'] = exact.payload_sha256(
        payload, omit=('certificate_sha256',))
    exact.write_json_atomic(
        os.path.join(RESULTS_DIR, 'huang_sweep2.json'), payload)
    print(f"DONE: {len(records)} jobs, {fails} fails, "
          f"{time.time()-started:.0f}s",
          flush=True)
    if fails:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
