"""Strict verifier and finalizer for Block 3b/c auxiliary evidence."""

from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction

import flint
from flint import arb

import core
from core import endpoints, set_prec, PSI, Q
import dsfun
import block3bc_exact as exact
import block3bc_aux_generate as generator
from block3bc_exact import (SCHEMA_VERSION, arb_packet, as_fraction,
                            ell_boundaries, file_sha256, fraction_arb,
                            fraction_from_record, fraction_record,
                             intervals_from_boundaries,
                             k_nodes, load_json, packet_abs_upper, packet_ball,
                             packet_endpoints, packet_fraction_endpoints,
                             payload_sha256, runtime_record, parse_fraction_text,
                             source_hashes, validate_runtime_record,
                             write_json_atomic)


HERE = os.path.dirname(os.path.abspath(__file__))
PRECISION_BITS = 100


def _shard_source_paths():
    return {
        'block3bc_aux_generate.py': os.path.join(
            HERE, 'block3bc_aux_generate.py'),
        'block3bc_exact.py': exact.__file__,
        'core.py': core.__file__,
        'dsfun.py': dsfun.__file__,
    }


def _manifest_source_paths():
    return {
        **_shard_source_paths(),
        'block3bc_aux_verify.py': __file__,
        'block3bc.py': os.path.join(HERE, 'block3bc.py'),
        'block3bc_assemble.py': os.path.join(HERE, 'block3bc_assemble.py'),
    }


def _require_packet(packet):
    lo_q, hi_q = packet_fraction_endpoints(packet)
    if lo_q > hi_q:
        raise ValueError("invalid packet")
    return fraction_arb(lo_q), fraction_arb(hi_q)


def verify_shard(path, expected_kind=None):
    path = exact.require_plain_regular_file(path)
    data = load_json(path)
    if (data.get('schema_version') != SCHEMA_VERSION
            or data.get('kind') != 'block3bc_aux_shard'):
        raise ValueError("invalid auxiliary shard schema")
    if data.get('artifact_sha256') != payload_sha256(
            data, omit=('artifact_sha256',)):
        raise ValueError("auxiliary shard payload hash mismatch")
    kind = data.get('aux_kind')
    if expected_kind is not None and kind != expected_kind:
        raise ValueError("wrong auxiliary shard kind")
    if kind not in ('ell_prime', 'k_grid') or data.get('failures') != 0:
        raise ValueError("failed or unknown auxiliary shard")
    expected_tolerances = ({'ell_prime_cells': 300}
                           if kind == 'ell_prime' else
                           {'I_second_inner_bits': 18,
                            'I_second_outer_bits': 15})
    if data.get('tolerances') != expected_tolerances:
        raise ValueError("auxiliary shard tolerance policy mismatch")
    if data.get('source_sha256') != source_hashes(_shard_source_paths()):
        raise ValueError("stale auxiliary shard source hash")
    runtime = data.get('runtime', {})
    try:
        validate_runtime_record(
            runtime, PRECISION_BITS, allow_any_workers=True)
    except ValueError as exc:
        raise ValueError("auxiliary shard runtime mismatch") from exc

    schedule0 = ell_boundaries() if kind == 'ell_prime' else k_nodes()
    schedule_payload = [fraction_record(x) for x in schedule0]
    if (data.get('schedule') != schedule_payload
            or data.get('schedule_sha256') != payload_sha256(
                schedule_payload, omit=())):
        raise ValueError("auxiliary shard schedule mismatch")
    lane = data.get('lane', {})
    expected_indices = exact.lane_indices(
        len(schedule0) - 1 if kind == 'ell_prime' else len(schedule0),
        lane.get('index'), lane.get('count'))
    if data.get('indices') != expected_indices:
        raise ValueError("auxiliary lane indices mismatch")
    records = data.get('records', [])
    if [row.get('index') for row in records] != expected_indices:
        raise ValueError("auxiliary record index mismatch")
    artifacts = data.get('job_artifacts', [])
    if [row.get('index') for row in artifacts] != expected_indices:
        raise ValueError("auxiliary job-artifact index mismatch")
    hashes = source_hashes(_shard_source_paths())
    shard_dir = exact.require_plain_directory(path.parent)
    by_index = {row['index']: row for row in records}
    for artifact in artifacts:
        i = artifact['index']
        try:
            job_path = exact.resolve_regular_file_under(
                shard_dir, artifact['file'])
        except ValueError as exc:
            raise ValueError("auxiliary job path escapes shard directory") from exc
        if file_sha256(job_path) != artifact['file_sha256']:
            raise ValueError("missing or changed auxiliary job artifact")
        saved = load_json(job_path)
        if saved.get('job_sha256') != artifact['job_sha256']:
            raise ValueError("auxiliary job identity mismatch")
        if kind == 'ell_prime':
            job_input = {'tau_lo': fraction_record(schedule0[i]),
                         'tau_hi': fraction_record(schedule0[i + 1])}
        else:
            job_input = {'lambda_value': fraction_record(schedule0[i])}
        result = generator._validate_job_record(
            saved, kind, i, job_input, hashes,
            require_current_runtime=False, containing_runtime=runtime)
        if result != by_index[i]:
            raise ValueError("auxiliary shard/job result mismatch")
    for row in records:
        if row.get('ok') is not True:
            raise ValueError("failed auxiliary record")
        i = row['index']
        if kind == 'ell_prime':
            if (fraction_from_record(row['tau_lo']) != schedule0[i]
                    or fraction_from_record(row['tau_hi']) != schedule0[i + 1]):
                raise ValueError("ell-prime cell mismatch")
            _require_packet(row['A'])
            vlo, _ = _require_packet(row['value'])
            if not (vlo > 0):
                raise ValueError("ell-prime packet is not positive")
        else:
            if fraction_from_record(row['lambda_value']) != schedule0[i]:
                raise ValueError("K-grid node mismatch")
            _require_packet(row['value'])
    return data


def _merge_shards(paths, kind):
    schedule0 = ell_boundaries() if kind == 'ell_prime' else k_nodes()
    expected_count = len(schedule0) - 1 if kind == 'ell_prime' else len(schedule0)
    merged = {}
    artifacts = []
    for path in paths:
        data = verify_shard(path, kind)
        artifacts.append({
            'file': os.path.basename(path),
            'file_sha256': file_sha256(path),
            'artifact_sha256': data['artifact_sha256'],
            'aux_kind': kind,
            'lane': data['lane'],
        })
        for row in data['records']:
            i = row['index']
            if i in merged:
                raise ValueError(f"duplicate {kind} record {i}")
            merged[i] = row
    if set(merged) != set(range(expected_count)):
        raise ValueError(f"incomplete {kind} shard union")
    return [merged[i] for i in range(expected_count)], artifacts


def _derive(ell_records, k_records, k_run):
    set_prec(PRECISION_BITS)
    k_run = as_fraction(k_run)
    ell_schedule = ell_boundaries()
    k_schedule = k_nodes()
    if len(ell_records) != 16 or len(k_records) != 59:
        raise ValueError("wrong auxiliary record count")
    for i, row in enumerate(ell_records):
        if (row.get('index') != i
                or fraction_from_record(row['tau_lo']) != ell_schedule[i]
                or fraction_from_record(row['tau_hi']) != ell_schedule[i + 1]):
            raise ValueError("ell-prime manifest schedule mismatch")
    for i, row in enumerate(k_records):
        if (row.get('index') != i
                or fraction_from_record(row['lambda_value']) != k_schedule[i]):
            raise ValueError("K-grid manifest schedule mismatch")

    # Select extrema in exact Fraction arithmetic.  Comparing rounded Arb
    # endpoint balls can be inconclusive when two candidates overlap and may
    # otherwise retain a non-extremal value.
    ell_lower_fraction = min(
        packet_fraction_endpoints(row['value'])[0]
        for row in ell_records)
    if ell_lower_fraction <= 0:
        raise ValueError("ell-prime lower bound is not positive")
    ell_lower = fraction_arb(ell_lower_fraction)

    max_i2_fraction = max(
        max(abs(lo), abs(hi))
        for lo, hi in (packet_fraction_endpoints(row['value'])
                       for row in k_records))
    max_i2 = fraction_arb(max_i2_fraction)

    spacing = k_schedule[1] - k_schedule[0]
    if any(k_schedule[i + 1] - k_schedule[i] != spacing
           for i in range(len(k_schedule) - 1)):
        raise ValueError("K grid is not uniform")
    lam_domain = fraction_arb(k_schedule[0]).union(
        fraction_arb(k_schedule[-1]))
    i3 = dsfun.I_third_fullbound(lam_domain)
    i3_packet = arb_packet(i3)
    _, i3_up = packet_endpoints(i3_packet)
    border = 2 * PSI * (1 - Q) / ((1 + lam_domain) ** 3)
    border_packet = arb_packet(border)
    _, border_up = packet_endpoints(border_packet)

    tau_cover_lo, tau_cover_hi = Fraction(-19, 100), Fraction(-3, 100)
    lam_cover = dsfun.lam_cell(fraction_arb(tau_cover_lo),
                               fraction_arb(tau_cover_hi))
    cover_packet = arb_packet(lam_cover)
    cover_lo, cover_hi = packet_endpoints(cover_packet)
    left_margin = cover_lo - fraction_arb(k_schedule[0])
    right_margin = fraction_arb(k_schedule[-1]) - cover_hi
    if not (cover_lo >= fraction_arb(k_schedule[0])
            and cover_hi <= fraction_arb(k_schedule[-1])
            and left_margin > 0 and right_margin > 0):
        raise ValueError("K grid does not cover lambda(tau) for b_neg")

    A_eval = dsfun.A_of_tau(fraction_arb(ell_schedule[0]))
    A_min = fraction_arb(Fraction(2, 3))
    # exp(2 atanh(tau)) = (1+tau)/(1-tau), hence A(-1/5)=2/3
    # algebraically.  The two independently rounded Arb representations need
    # only overlap; neither tiny ball need contain the other's rounding ball.
    if not A_eval.overlaps(A_min):
        raise ValueError("A(-1/5) is inconsistent with the exact value 2/3")
    A_packet = arb_packet(A_min)
    A_min_lo, _ = packet_endpoints(A_packet)
    one_minus_q = 1 - Q
    _, one_minus_q_up = endpoints(one_minus_q)
    h2 = one_minus_q_up / (2 * A_min_lo * ell_lower)
    h2_packet = arb_packet(h2)
    _, h2_up = packet_endpoints(h2_packet)
    spacing_arb = fraction_arb(spacing)
    k_bound = max_i2 + spacing_arb * i3_up / 2 + border_up + h2_up
    k_packet = arb_packet(k_bound)
    _, k_upper = packet_endpoints(k_packet)
    if not (k_upper < fraction_arb(k_run)):
        raise ValueError(f"K upper bound {k_upper} is not below K_run {k_run}")

    return {
        'lambda_domain': [fraction_record(k_schedule[0]),
                          fraction_record(k_schedule[-1])],
        'lambda_cover': cover_packet,
        'lambda_left_margin': arb_packet(left_margin),
        'lambda_right_margin': arb_packet(right_margin),
        'spacing': fraction_record(spacing),
        'ell_prime_lower': arb_packet(ell_lower),
        'max_abs_I_second': arb_packet(max_i2),
        'I_third_bound': i3_packet,
        'border_bound': border_packet,
        'A_min': A_packet,
        'H_second_abs_bound': h2_packet,
        'K_bound': k_packet,
        'K_run': fraction_record(k_run),
    }


def build_manifest(ell_paths, k_paths, k_run, output):
    output_parent = exact.require_plain_directory(
        os.path.dirname(os.path.abspath(output)))
    for path in [*ell_paths, *k_paths]:
        shard = exact.require_plain_regular_file(path)
        if exact.require_plain_directory(shard.parent) != output_parent:
            raise ValueError('auxiliary shards and manifest must share a directory')
    ell_records, ell_artifacts = _merge_shards(ell_paths, 'ell_prime')
    k_records, k_artifacts = _merge_shards(k_paths, 'k_grid')
    k_run = parse_fraction_text(k_run) if isinstance(k_run, str) \
        else as_fraction(k_run)
    derived = _derive(ell_records, k_records, k_run)
    payload = {
        'schema_version': SCHEMA_VERSION,
        'kind': 'block3bc_aux_manifest',
        'source_sha256': source_hashes(_manifest_source_paths()),
        'runtime': runtime_record(PRECISION_BITS),
        'parameter_policy': {
            'precision_bits': PRECISION_BITS,
            'I_second_inner_bits': 18,
            'I_second_outer_bits': 15,
            'ell_prime_cells': 300,
        },
        'ell_schedule': [fraction_record(x) for x in ell_boundaries()],
        'k_schedule': [fraction_record(x) for x in k_nodes()],
        'ell_records': ell_records,
        'k_records': k_records,
        'derived': derived,
        'k_run': fraction_record(k_run),
        'input_artifacts': ell_artifacts + k_artifacts,
    }
    payload['manifest_sha256'] = payload_sha256(payload)
    write_json_atomic(output, payload)
    verify_manifest(output, require_complete=True)
    return payload


def verify_manifest(path, require_complete=True):
    path = exact.require_plain_regular_file(path)
    data = load_json(path)
    if (data.get('schema_version') != SCHEMA_VERSION
            or data.get('kind') != 'block3bc_aux_manifest'):
        raise ValueError("invalid Block3bc auxiliary manifest schema")
    if data.get('manifest_sha256') != payload_sha256(data):
        raise ValueError("auxiliary manifest hash mismatch")
    if data.get('source_sha256') != source_hashes(_manifest_source_paths()):
        raise ValueError("stale auxiliary manifest source hash")
    if data.get('parameter_policy') != {
            'precision_bits': PRECISION_BITS,
            'I_second_inner_bits': 18,
            'I_second_outer_bits': 15,
            'ell_prime_cells': 300}:
        raise ValueError("auxiliary manifest parameter policy mismatch")
    runtime = data.get('runtime', {})
    try:
        validate_runtime_record(runtime, PRECISION_BITS)
    except ValueError as exc:
        raise ValueError("auxiliary manifest runtime mismatch") from exc
    if data.get('ell_schedule') != [fraction_record(x) for x in ell_boundaries()]:
        raise ValueError("ell-prime manifest schedule mismatch")
    if data.get('k_schedule') != [fraction_record(x) for x in k_nodes()]:
        raise ValueError("K-grid manifest schedule mismatch")
    k_run = fraction_from_record(data.get('k_run'))
    derived = _derive(data.get('ell_records', []),
                      data.get('k_records', []), k_run)
    if data.get('derived') != derived:
        raise ValueError("auxiliary derived-bound mismatch")

    if require_complete:
        root = exact.require_plain_directory(path.parent)
        artifacts = data.get('input_artifacts', [])
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("auxiliary manifest has no input artifacts")
        reconstructed = {'ell_prime': {}, 'k_grid': {}}
        seen_files = set()
        for artifact in artifacts:
            kind = artifact.get('aux_kind')
            if kind not in reconstructed:
                raise ValueError("unknown auxiliary input-artifact kind")
            if artifact.get('file') in seen_files:
                raise ValueError("duplicate auxiliary input artifact")
            seen_files.add(artifact.get('file'))
            try:
                shard = exact.resolve_regular_file_under(
                    root, artifact['file'])
            except ValueError as exc:
                raise ValueError(
                    "auxiliary shard path escapes manifest directory") from exc
            if file_sha256(shard) != artifact['file_sha256']:
                raise ValueError(f"missing or changed auxiliary shard {shard}")
            shard_data = verify_shard(shard, kind)
            if (shard_data['artifact_sha256'] != artifact['artifact_sha256']
                    or shard_data['lane'] != artifact.get('lane')):
                raise ValueError("auxiliary shard identity mismatch")
            for row in shard_data['records']:
                i = row['index']
                if i in reconstructed[kind]:
                    raise ValueError(f"duplicate reconstructed {kind} record {i}")
                reconstructed[kind][i] = row
        expected_counts = {'ell_prime': 16, 'k_grid': 59}
        embedded = {'ell_prime': data.get('ell_records', []),
                    'k_grid': data.get('k_records', [])}
        for kind, count in expected_counts.items():
            if set(reconstructed[kind]) != set(range(count)):
                raise ValueError(f"incomplete reconstructed {kind} union")
            rows = [reconstructed[kind][i] for i in range(count)]
            if rows != embedded[kind]:
                raise ValueError(f"embedded {kind} records differ from shards")
    return {'manifest': data, 'k_run': k_run, 'derived': derived}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--allow-missing-shards', action='store_true')
    args = parser.parse_args(argv)
    out = verify_manifest(args.manifest,
                          require_complete=not args.allow_missing_shards)
    print(f"PASS Block3bc auxiliary manifest {out['manifest']['manifest_sha256']} "
          f"K_run={exact.fraction_text(out['k_run'])}")


if __name__ == '__main__':
    main()
