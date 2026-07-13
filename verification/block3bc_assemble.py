"""Strict assembler/verifier for fresh exact Block 3b/c replay artifacts."""

from __future__ import annotations

import argparse
import glob
import os
import sys

import flint

import core
import dsfun
import block3bc
import block3bc_exact as exact
import block3bc_aux_verify
from block3bc_exact import (SCHEMA_VERSION, b_neg_boundaries,
                            b_pos_boundaries, c_boundaries, file_sha256,
                            fraction_from_record, fraction_record,
                            intervals_from_boundaries, lane_indices, load_json,
                            packet_endpoints, payload_sha256, runtime_record,
                            source_hashes, validate_runtime_record,
                            write_json_atomic)


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
DEFAULT_AUX = os.path.join(RESULTS_DIR, 'block3bc_aux', 'manifest.json')
DEFAULT_REPLAY_DIR = os.path.join(RESULTS_DIR, 'block3bc_replay')
DEFAULT_CERTIFICATE = os.path.join(RESULTS_DIR, 'block3bc_certificate.json')
PRECISION_BITS = 60


def _replay_source_paths():
    return {
        'block3bc.py': block3bc.__file__,
        'block3bc_exact.py': exact.__file__,
        'core.py': core.__file__,
        'dsfun.py': dsfun.__file__,
    }


def _certificate_source_paths():
    return {
        **_replay_source_paths(),
        'block3bc_assemble.py': __file__,
        'block3bc_aux_verify.py': block3bc_aux_verify.__file__,
    }


def _part_boundaries(part, k_run):
    if part == 'b_pos':
        return b_pos_boundaries()
    if part == 'c':
        return c_boundaries()
    if part == 'b_neg':
        return b_neg_boundaries(k_run)
    raise ValueError(f"unknown replay part {part}")


def verify_replay_shard(path, aux):
    path = exact.require_plain_regular_file(path)
    data = load_json(path)
    if (data.get('schema_version') != SCHEMA_VERSION
            or data.get('kind') != 'block3bc_replay_shard'):
        raise ValueError("invalid Block3bc replay shard schema")
    if data.get('artifact_sha256') != payload_sha256(
            data, omit=('artifact_sha256',)):
        raise ValueError("replay shard payload hash mismatch")
    if data.get('failures') != 0:
        raise ValueError("replay shard contains failures")
    if data.get('source_sha256') != source_hashes(_replay_source_paths()):
        raise ValueError("stale replay source hash")
    runtime = data.get('runtime', {})
    try:
        validate_runtime_record(
            runtime, PRECISION_BITS, allow_any_workers=True)
    except ValueError as exc:
        raise ValueError("replay runtime mismatch") from exc
    if data.get('tolerances') != {
            'precision_bits': 60,
            'I_prime_inner_bits': 20,
            'I_prime_outer_bits': 17,
            'I_second_inner_bits': 14,
            'I_second_outer_bits': 12}:
        raise ValueError("replay tolerance policy mismatch")
    if (data.get('aux_manifest_sha256')
            != aux['manifest']['manifest_sha256']):
        raise ValueError("wrong auxiliary manifest binding")
    if fraction_from_record(data.get('k_run')) != aux['k_run']:
        raise ValueError("wrong replay K_run")

    part = data.get('part')
    boundaries = _part_boundaries(part, aux['k_run'])
    expected_schedule = [fraction_record(x) for x in boundaries]
    if (data.get('schedule_boundaries') != expected_schedule
            or data.get('schedule_sha256') != payload_sha256(
                expected_schedule, omit=())):
        raise ValueError("replay schedule mismatch")
    intervals = intervals_from_boundaries(boundaries)
    lane = data.get('lane', {})
    expected_indices = lane_indices(
        len(intervals), lane.get('index'), lane.get('count'))
    if data.get('indices') != expected_indices:
        raise ValueError("replay lane index mismatch")
    records = data.get('records', [])
    if [row.get('index') for row in records] != expected_indices:
        raise ValueError("replay record index mismatch")
    job_artifacts = data.get('job_artifacts', [])
    if [row.get('index') for row in job_artifacts] != expected_indices:
        raise ValueError("replay job-artifact index mismatch")
    shard_dir = exact.require_plain_directory(path.parent)
    hashes = source_hashes(_replay_source_paths())
    by_index = {row['index']: row for row in records}
    for artifact in job_artifacts:
        i = artifact['index']
        try:
            job_path = exact.resolve_regular_file_under(
                shard_dir, artifact['file'])
        except ValueError as exc:
            raise ValueError("replay job path escapes shard directory") from exc
        if file_sha256(job_path) != artifact['file_sha256']:
            raise ValueError("missing or changed replay job artifact")
        saved = load_json(job_path)
        if saved.get('job_sha256') != artifact['job_sha256']:
            raise ValueError("replay job identity mismatch")
        top_lo, top_hi = intervals[i]
        job_input = {'tau_lo': fraction_record(top_lo),
                     'tau_hi': fraction_record(top_hi),
                     'k_run': fraction_record(aux['k_run'])}
        result = block3bc._validate_replay_job(
            saved, part, i, job_input,
            aux['manifest']['manifest_sha256'], hashes,
            require_current_runtime=False, containing_runtime=runtime)
        if result != by_index[i]:
            raise ValueError("replay shard/job result mismatch")

    want = '>0' if part == 'b_neg' else '<0'
    for row in records:
        i = row['index']
        top_lo, top_hi = intervals[i]
        if (row.get('ok') is not True or row.get('failure') is not None
                or fraction_from_record(row['tau_lo']) != top_lo
                or fraction_from_record(row['tau_hi']) != top_hi):
            raise ValueError("invalid replay top-cell record")
        leaves = row.get('leaves', [])
        leaf_intervals = [(fraction_from_record(x['tau_lo']),
                           fraction_from_record(x['tau_hi']))
                          for x in leaves]
        exact.require_exact_partition(leaf_intervals, (top_lo, top_hi))
        for leaf in leaves:
            if leaf.get('sign') != want:
                raise ValueError("wrong replay leaf sign")
            vlo, vhi = packet_endpoints(leaf['value'])
            if not ((vhi < 0) if want == '<0' else (vlo > 0)):
                raise ValueError("replay leaf packet does not certify its sign")
    return data


def _evaluate(aux_path, shard_paths):
    aux = block3bc_aux_verify.verify_manifest(aux_path, require_complete=True)
    merged = {part: {} for part in ('b_pos', 'b_neg', 'c')}
    artifacts = []
    for path in shard_paths:
        data = verify_replay_shard(path, aux)
        part = data['part']
        artifacts.append({
            'file': os.path.relpath(os.path.abspath(path), RESULTS_DIR).replace('\\', '/'),
            'file_sha256': file_sha256(path),
            'artifact_sha256': data['artifact_sha256'],
            'part': part,
            'lane': data['lane'],
        })
        for row in data['records']:
            i = row['index']
            if i in merged[part]:
                raise ValueError(f"duplicate {part} top cell {i}")
            merged[part][i] = row

    summary = {}
    for part in ('b_pos', 'b_neg', 'c'):
        boundaries = _part_boundaries(part, aux['k_run'])
        expected = set(range(len(boundaries) - 1))
        if set(merged[part]) != expected:
            raise ValueError(f"incomplete {part} replay union")
        records = [merged[part][i] for i in sorted(merged[part])]
        exact.require_exact_schedule(records, boundaries)
        summary[part] = {
            'top_cells': len(records),
            'leaves': sum(len(row['leaves']) for row in records),
        }
    if not block3bc.boundaries():
        raise ValueError("Block3bc boundary pins failed")
    artifacts.sort(key=lambda row: (row['part'], row['lane']['count'],
                                    row['lane']['index'], row['file']))
    return aux, summary, artifacts


def assemble(aux_path, shard_paths, output=DEFAULT_CERTIFICATE):
    aux, summary, artifacts = _evaluate(aux_path, shard_paths)
    payload = {
        'schema_version': SCHEMA_VERSION,
        'kind': 'block3bc_certificate',
        'source_sha256': source_hashes(_certificate_source_paths()),
        'runtime': runtime_record(PRECISION_BITS),
        'aux_manifest': os.path.relpath(
            os.path.abspath(aux_path), RESULTS_DIR).replace('\\', '/'),
        'aux_manifest_sha256': aux['manifest']['manifest_sha256'],
        'k_run': fraction_record(aux['k_run']),
        'replay_artifacts': artifacts,
        'summary': summary,
        'boundary_pins': True,
        'verdict': 'ALL PASS',
    }
    payload['certificate_sha256'] = payload_sha256(
        payload, omit=('certificate_sha256',))
    write_json_atomic(output, payload)
    verify_certificate(output)
    return payload


def _resolve_result_path(relative):
    try:
        return exact.resolve_regular_file_under(RESULTS_DIR, relative)
    except ValueError as exc:
        raise ValueError("artifact path escapes results directory") from exc


def verify_certificate(path=DEFAULT_CERTIFICATE):
    path = exact.require_plain_regular_file(path)
    data = load_json(path)
    if (data.get('schema_version') != SCHEMA_VERSION
            or data.get('kind') != 'block3bc_certificate'
            or data.get('verdict') != 'ALL PASS'
            or data.get('boundary_pins') is not True):
        raise ValueError("invalid Block3bc certificate schema/verdict")
    if data.get('certificate_sha256') != payload_sha256(
            data, omit=('certificate_sha256',)):
        raise ValueError("Block3bc certificate hash mismatch")
    if data.get('source_sha256') != source_hashes(_certificate_source_paths()):
        raise ValueError("stale Block3bc certificate source hash")
    runtime = data.get('runtime', {})
    try:
        validate_runtime_record(runtime, PRECISION_BITS)
    except ValueError as exc:
        raise ValueError("Block3bc certificate runtime mismatch") from exc
    aux_path = _resolve_result_path(data.get('aux_manifest'))
    shard_paths = []
    for artifact in data.get('replay_artifacts', []):
        shard = _resolve_result_path(artifact['file'])
        if file_sha256(shard) != artifact['file_sha256']:
            raise ValueError("missing or changed Block3bc replay artifact")
        shard_paths.append(shard)
    aux, summary, artifacts = _evaluate(aux_path, shard_paths)
    if (data.get('aux_manifest_sha256') != aux['manifest']['manifest_sha256']
            or fraction_from_record(data.get('k_run')) != aux['k_run']
            or data.get('summary') != summary
            or data.get('replay_artifacts') != artifacts):
        raise ValueError("Block3bc certificate/raw evidence mismatch")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aux-manifest', default=DEFAULT_AUX)
    parser.add_argument('--shard', action='append')
    parser.add_argument('--output', default=DEFAULT_CERTIFICATE)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--certificate', default=DEFAULT_CERTIFICATE)
    args = parser.parse_args(argv)
    if args.check_only:
        data = verify_certificate(args.certificate)
        print(f"PASS Block3bc certificate {data['certificate_sha256']}")
        return
    shards = args.shard or sorted(glob.glob(os.path.join(
        DEFAULT_REPLAY_DIR, '*.json')))
    if not shards:
        raise SystemExit("no Block3bc replay shards supplied")
    data = assemble(args.aux_manifest, shards, args.output)
    print(f"PASS Block3bc assembly {data['certificate_sha256']}")


if __name__ == '__main__':
    main()
