"""Generate fresh, sharded auxiliary evidence for Ding--Sun Block 3b/c."""

from __future__ import annotations

import argparse
import os
import sys
import time

from core import endpoints, set_prec
import core
import dsfun
import block3bc_exact as exact
from block3bc_exact import (SCHEMA_VERSION, arb_packet, ell_boundaries,
                            apply_worker_policy,
                            file_sha256, fraction_arb, fraction_record,
                            fraction_from_record, fraction_text,
                            intervals_from_boundaries, k_nodes, lane_indices,
                            isolated_subprocess_results, load_json,
                            parse_fraction_text, payload_sha256,
                            runtime_record, source_hashes,
                            validate_runtime_record,
                            write_json_atomic)


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results', 'block3bc_aux')
PRECISION_BITS = 100
DEFAULT_K_RUN = '21/2'


def _source_paths():
    return {
        'block3bc_aux_generate.py': __file__,
        'block3bc_exact.py': exact.__file__,
        'core.py': core.__file__,
        'dsfun.py': dsfun.__file__,
    }


def _ell_job(job):
    index, lo_record, hi_record = job
    set_prec(PRECISION_BITS)
    t0 = time.time_ns()
    lo, hi = fraction_from_record(lo_record), fraction_from_record(hi_record)
    alo = dsfun.A_of_tau(fraction_arb(lo))
    ahi = dsfun.A_of_tau(fraction_arb(hi))
    acell = alo.union(ahi)
    value = dsfun.ell_prime(acell, n=300)
    vlo, _ = endpoints(value)
    if not (vlo > 0):
        raise ValueError("ell-prime lower endpoint is not positive")
    return dict(index=int(index), tau_lo=fraction_record(lo),
                tau_hi=fraction_record(hi), A=arb_packet(acell),
                value=arb_packet(value), ok=True,
                runtime_milliseconds=(time.time_ns() - t0) // 1_000_000)


def _k_job(job):
    index, lambda_record = job
    set_prec(PRECISION_BITS)
    t0 = time.time_ns()
    lam = fraction_from_record(lambda_record)
    value = dsfun.I_second_box(
        fraction_arb(lam), inner_tol_bits=18, outer_tol_bits=15)
    if value is None:
        raise ValueError("I_second_box returned None")
    return dict(index=int(index), lambda_value=fraction_record(lam),
                value=arb_packet(value), ok=True,
                runtime_milliseconds=(time.time_ns() - t0) // 1_000_000)


def _job_record(kind, result, job_input, source_before=None,
                source_after=None, runtime=None):
    if source_before is None:
        source_before = source_hashes(_source_paths())
    if source_after is None:
        source_after = source_before
    if runtime is None:
        runtime = runtime_record(PRECISION_BITS, 1)
    result = _validate_child_result(kind, result.get('index'), job_input, result)
    payload = {
        'schema_version': SCHEMA_VERSION,
        'kind': 'block3bc_aux_job',
        'aux_kind': kind,
        'index': result['index'],
        'input': job_input,
        'source_sha256': source_before,
        'source_sha256_after': source_after,
        'runtime': runtime,
        'result': result,
    }
    payload['job_sha256'] = payload_sha256(payload, omit=('job_sha256',))
    return payload


def _validate_job_record(data, kind, index, job_input, hashes,
                         require_current_runtime=True,
                         containing_runtime=None):
    required = {
        'schema_version', 'kind', 'aux_kind', 'index', 'input',
        'source_sha256', 'source_sha256_after', 'runtime', 'result',
        'job_sha256',
    }
    if (not isinstance(data, dict) or set(data) != required
            or not isinstance(data.get('index'), int)
            or isinstance(data.get('index'), bool)
            or data.get('schema_version') != SCHEMA_VERSION
            or data.get('kind') != 'block3bc_aux_job'
            or data.get('aux_kind') != kind
            or data.get('index') != index
            or data.get('input') != job_input
            or data.get('source_sha256') != hashes
            or data.get('source_sha256_after') != hashes
            or data.get('job_sha256') != payload_sha256(
                data, omit=('job_sha256',))):
        raise ValueError(f"stale/corrupt resume record {kind} {index}")
    runtime = data.get('runtime', {})
    try:
        validate_runtime_record(runtime, PRECISION_BITS, workers=1)
    except ValueError as exc:
        raise ValueError(
            f"incompatible resume runtime {kind} {index}") from exc
    if require_current_runtime and runtime != runtime_record(PRECISION_BITS, 1):
        raise ValueError(f"incompatible resume runtime {kind} {index}")
    if (containing_runtime is not None
            and exact.runtime_identity(runtime)
            != exact.runtime_identity(containing_runtime)):
        raise ValueError(f"job/shard runtime mismatch {kind} {index}")
    result = data.get('result', {})
    if result.get('index') != index:
        raise ValueError("resume result index mismatch")
    return _validate_child_result(kind, index, job_input, result)


def _validate_child_result(kind, index, job_input, result):
    if (not isinstance(result, dict)
            or not isinstance(result.get('index'), int)
            or isinstance(result.get('index'), bool)
            or result.get('index') != index):
        raise ValueError('auxiliary result index/schema mismatch')
    if result.get('ok') is not True:
        raise ValueError('auxiliary result is not a success')
    runtime_ms = result.get('runtime_milliseconds')
    if (not isinstance(runtime_ms, int) or isinstance(runtime_ms, bool)
            or runtime_ms < 0):
        raise ValueError('invalid auxiliary child runtime')
    if kind == 'ell_prime':
        if set(result) != {
                'index', 'tau_lo', 'tau_hi', 'A', 'value', 'ok',
                'runtime_milliseconds'}:
            raise ValueError('auxiliary ell-prime result schema mismatch')
        if (result['tau_lo'] != job_input['tau_lo']
                or result['tau_hi'] != job_input['tau_hi']):
            raise ValueError('auxiliary ell-prime result/input mismatch')
        exact.packet_fraction_endpoints(result['A'])
        value_lo, _ = exact.packet_fraction_endpoints(result['value'])
        if value_lo <= 0:
            raise ValueError('auxiliary ell-prime packet is not positive')
    elif kind == 'k_grid':
        if set(result) != {
                'index', 'lambda_value', 'value', 'ok',
                'runtime_milliseconds'}:
            raise ValueError('auxiliary K-grid result schema mismatch')
        if result['lambda_value'] != job_input['lambda_value']:
            raise ValueError('auxiliary K-grid result/input mismatch')
        exact.packet_fraction_endpoints(result['value'])
    else:
        raise ValueError('unknown auxiliary child kind')
    return result


def generate_shard(kind, lane=0, lanes=1, workers=1, output=None,
                   timeout_seconds=14400, retries=1):
    set_prec(PRECISION_BITS)
    # Refuse an unsupported numerical runtime before inspecting resume state
    # or launching any expensive auxiliary job.  The resulting job and shard
    # records bind the full python-flint package tree (including native code).
    if (not isinstance(workers, int) or isinstance(workers, bool)
            or workers <= 0):
        raise ValueError('workers must be a positive plain integer')
    worker_runtime = runtime_record(
        PRECISION_BITS, 1, fresh_flint=True)
    validate_runtime_record(worker_runtime, PRECISION_BITS, workers=1)
    if output is None:
        output = os.path.join(
            RESULTS_DIR,
            f'{kind}.lane-{int(lane)}-of-{int(lanes)}.json')
    output = os.path.abspath(output)
    record_dir = output + '.records'
    hashes = source_hashes(_source_paths())
    if kind == 'ell_prime':
        schedule = ell_boundaries()
        cells = intervals_from_boundaries(schedule)
        indices = lane_indices(len(cells), lane, lanes)
        jobs = [(i, fraction_record(cells[i][0]),
                 fraction_record(cells[i][1])) for i in indices]
        job_inputs = {i: {'tau_lo': lo, 'tau_hi': hi}
                      for i, lo, hi in jobs}
        prefix = 'ellp'
        schedule_payload = [fraction_record(x) for x in schedule]
    elif kind == 'k_grid':
        schedule = k_nodes()
        indices = lane_indices(len(schedule), lane, lanes)
        jobs = [(i, fraction_record(schedule[i])) for i in indices]
        job_inputs = {i: {'lambda_value': lam} for i, lam in jobs}
        prefix = 'i2'
        schedule_payload = [fraction_record(x) for x in schedule]
    else:
        raise ValueError(kind)

    os.makedirs(record_dir, exist_ok=True)
    records_by_index = {}
    record_payloads = {}
    pending = []
    for job in jobs:
        i = job[0]
        path = os.path.join(record_dir, f'{prefix}-{i:03d}.json')
        if os.path.exists(path):
            saved = load_json(path)
            result = _validate_job_record(
                saved, kind, i, job_inputs[i], hashes)
            records_by_index[i] = result
            record_payloads[i] = (path, saved)
        else:
            pending.append(job)

    worker_count = max(1, min(workers, len(pending) or 1))
    specs = []
    for job in pending:
        i = job[0]
        child_args = ['_job', '--aux-kind', kind, '--index', str(i)]
        if kind == 'ell_prime':
            child_args += [
                '--lo=' + fraction_text(fraction_from_record(job[1])),
                '--hi=' + fraction_text(fraction_from_record(job[2]))]
        else:
            child_args += [
                '--lambda-value=' + fraction_text(
                    fraction_from_record(job[1]))]
        command = exact.isolated_python_command(
            __file__, worker_runtime, child_args)
        specs.append((i, command))

    def validate_child(index, saved):
        _validate_job_record(
            saved, kind, index, job_inputs[index], hashes,
            require_current_runtime=True)

    for i, saved in isolated_subprocess_results(
            specs, worker_count, record_dir,
            timeout_seconds=timeout_seconds, retries=retries,
            result_validator=validate_child):
        result = saved['result']
        if result.get('index') != i:
            raise ValueError("isolated auxiliary result index mismatch")
        path = os.path.join(record_dir, f'{prefix}-{i:03d}.json')
        write_json_atomic(path, saved, overwrite=False)
        records_by_index[i] = result
        record_payloads[i] = (path, saved)

    if set(records_by_index) != set(indices):
        raise ValueError("incomplete auxiliary job-record set")
    source_after = source_hashes(_source_paths())
    worker_runtime_after = runtime_record(
        PRECISION_BITS, 1, fresh_flint=True)
    if source_after != hashes or worker_runtime_after != worker_runtime:
        raise RuntimeError('auxiliary producer source/runtime changed mid-run')
    shard_runtime = runtime_record(
        PRECISION_BITS, worker_count, fresh_flint=True)
    if (exact.runtime_identity(shard_runtime)
            != exact.runtime_identity(worker_runtime)):
        raise RuntimeError('auxiliary shard/worker runtime identity mismatch')
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
    records.sort(key=lambda row: row['index'])
    payload = {
        'schema_version': SCHEMA_VERSION,
        'kind': 'block3bc_aux_shard',
        'aux_kind': kind,
        'lane': {'index': int(lane), 'count': int(lanes)},
        'indices': indices,
        'schedule': schedule_payload,
        'schedule_sha256': payload_sha256(schedule_payload, omit=()),
        'source_sha256': hashes,
        'runtime': shard_runtime,
        'tolerances': ({'ell_prime_cells': 300}
                       if kind == 'ell_prime' else
                       {'I_second_inner_bits': 18,
                        'I_second_outer_bits': 15}),
        'records': records,
        'job_artifacts': job_artifacts,
        'failures': sum(not row.get('ok', False) for row in records),
    }
    payload['artifact_sha256'] = payload_sha256(
        payload, omit=('artifact_sha256',))
    write_json_atomic(output, payload)
    if payload['failures']:
        raise RuntimeError(f"{payload['failures']} {kind} jobs failed")
    return output, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name, kind in (('ell-prime', 'ell_prime'), ('k-grid', 'k_grid')):
        p = sub.add_parser(name)
        p.set_defaults(aux_kind=kind)
        p.add_argument('--lane', type=int, default=0)
        p.add_argument('--lanes', type=int, default=1)
        p.add_argument('--workers', type=int, default=1)
        p.add_argument('--timeout-seconds', type=int, default=14400)
        p.add_argument('--retries', type=int, default=1)
        p.add_argument('--output')
    p = sub.add_parser('_job')
    p.add_argument('--aux-kind', required=True, choices=('ell_prime', 'k_grid'))
    p.add_argument('--index', required=True, type=int)
    p.add_argument('--lo')
    p.add_argument('--hi')
    p.add_argument('--lambda-value')
    p.add_argument('--result-file', required=True)
    p = sub.add_parser('finalize')
    p.add_argument('--ell-shard', action='append', required=True)
    p.add_argument('--k-shard', action='append', required=True)
    p.add_argument('--k-run', default=DEFAULT_K_RUN)
    p.add_argument('--output', default=os.path.join(RESULTS_DIR, 'manifest.json'))
    args = parser.parse_args(argv)
    if args.command == '_job':
        apply_worker_policy()
        set_prec(PRECISION_BITS)
        source_before = source_hashes(_source_paths())
        runtime_before = runtime_record(
            PRECISION_BITS, 1, fresh_flint=True)
        validate_runtime_record(
            runtime_before, PRECISION_BITS, workers=1)
        if args.aux_kind == 'ell_prime':
            if args.lo is None or args.hi is None or args.lambda_value is not None:
                raise SystemExit("invalid ell-prime child inputs")
            result = _ell_job((args.index,
                               fraction_record(parse_fraction_text(args.lo)),
                               fraction_record(parse_fraction_text(args.hi))))
        else:
            if args.lambda_value is None or args.lo is not None or args.hi is not None:
                raise SystemExit("invalid K-grid child inputs")
            result = _k_job((
                args.index, fraction_record(parse_fraction_text(args.lambda_value))))
        if result.get('ok') is not True:
            raise RuntimeError('isolated auxiliary job did not certify success')
        source_after = source_hashes(_source_paths())
        runtime_after = runtime_record(
            PRECISION_BITS, 1, fresh_flint=True)
        if source_after != source_before or runtime_after != runtime_before:
            raise RuntimeError('auxiliary child source/runtime changed mid-job')
        saved = _job_record(
            args.aux_kind, result,
            ({'tau_lo': result['tau_lo'], 'tau_hi': result['tau_hi']}
             if args.aux_kind == 'ell_prime' else
             {'lambda_value': result['lambda_value']}),
            source_before, source_after, runtime_before)
        write_json_atomic(args.result_file, saved, overwrite=False)
        return
    if args.command == 'finalize':
        from block3bc_aux_verify import build_manifest
        manifest = build_manifest(args.ell_shard, args.k_shard,
                                  args.k_run, args.output)
        print(f"PASS auxiliary manifest K<{args.k_run}: {args.output} "
              f"{manifest['manifest_sha256']}", flush=True)
        return
    output, payload = generate_shard(
        args.aux_kind, args.lane, args.lanes, args.workers, args.output,
        timeout_seconds=args.timeout_seconds, retries=args.retries)
    print(f"PASS {args.aux_kind}: {len(payload['records'])} records; "
          f"{output}", flush=True)


if __name__ == '__main__':
    main()
