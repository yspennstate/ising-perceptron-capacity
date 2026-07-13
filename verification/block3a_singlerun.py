"""Verifier and certificate assembler for a frozen-source Block3a run."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
from decimal import Decimal, InvalidOperation

import block3a_assemble as A
import block3a_run as run
import block3bc_exact as exact


SCHEMA_VERSION = 2
EVIDENCE_MODEL = 'single-run-frozen-source-v2'
_ELAPSED_RE = re.compile(r'(?:0|[1-9][0-9]*)\.[0-9]{6}\Z')


def source_hashes():
    """Hash the complete producer/proof/verifier closure for this mode."""
    paths = {
        name: pathlib.Path(run.HERE) / name
        for name in run.PROOF_SOURCE_NAMES
    }
    paths.update({
        'block3a_assemble.py': pathlib.Path(A.__file__).resolve(),
        'block3a_run.py': pathlib.Path(run.__file__).resolve(),
        'block3a_singlerun.py': pathlib.Path(__file__).resolve(),
    })
    return exact.source_hashes(paths)


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f'duplicate JSON key {key}')
        out[key] = value
    return out


def _reject_number(text):
    raise ValueError(f'noncanonical JSON number {text}')


def _load_canonical_json_bytes(raw, label):
    try:
        value = json.loads(
            raw.decode('utf-8', errors='strict'),
            parse_float=_reject_number, parse_constant=_reject_number,
            object_pairs_hook=_unique_object)
    except Exception as exc:
        raise ValueError(f'invalid {label} JSON') from exc
    if raw != exact.canonical_json_bytes(value) + b'\n':
        raise ValueError(f'noncanonical {label} JSON')
    return value


def _resolve_under(root, relative):
    try:
        return exact.resolve_regular_file_under(root, relative)
    except ValueError as exc:
        raise ValueError('artifact path is not a confined regular file') from exc


def _read_descriptor(root, descriptor, label):
    if (not isinstance(descriptor, dict)
            or set(descriptor) != {'file', 'sha256', 'bytes'}):
        raise ValueError(f'invalid {label} artifact descriptor')
    path = _resolve_under(root, descriptor['file'])
    if (not isinstance(descriptor['bytes'], int)
            or isinstance(descriptor['bytes'], bool)
            or descriptor['bytes'] < 0
            or not isinstance(descriptor['sha256'], str)
            or re.fullmatch(r'[0-9a-f]{64}', descriptor['sha256']) is None):
        raise ValueError(f'invalid {label} artifact identity')
    raw = path.read_bytes()
    if (descriptor['bytes'] != len(raw)
            or descriptor['sha256'] != hashlib.sha256(raw).hexdigest()):
        raise ValueError(f'{label} artifact identity mismatch')
    return path, raw


def _parse_utc(text):
    if not isinstance(text, str):
        raise ValueError('timestamp must be text')
    value = dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('timestamp must be timezone-aware')
    return value.astimezone(dt.timezone.utc)


def _runtime_identity(runtime):
    return {key: value for key, value in runtime.items()
            if key != 'workers'}


def parse_structured_log_bytes(raw, schedule, *, run_id=None,
                               expected_source=None,
                               producer_runtime=None):
    """Validate canonical JSONL and every exact negative Arb leaf packet."""
    if not raw or not raw.endswith(b'\n') or b'\r' in raw:
        raise ValueError('Block3a JSONL is empty, partial, or noncanonical')
    expected = {row['index']: row for row in schedule}
    records = {}
    max_upper = None
    total_leaves = 0
    for lineno, line in enumerate(raw[:-1].split(b'\n'), 1):
        if not line:
            raise ValueError(f'blank Block3a JSONL line {lineno}')
        try:
            row = json.loads(
                line.decode('utf-8', errors='strict'),
                parse_float=_reject_number, parse_constant=_reject_number,
                object_pairs_hook=_unique_object)
        except Exception as exc:
            raise ValueError(f'invalid Block3a JSONL line {lineno}') from exc
        if exact.canonical_json_bytes(row) != line:
            raise ValueError(f'noncanonical Block3a JSONL line {lineno}')
        required = {
            'schema_version', 'run_id', 'index', 'kind', 'tau_lo', 'tau_hi',
            'verdict', 'recursive_calls', 'leaf_bounds', 'elapsed_seconds',
            'worker_runtime', 'worker_source_sha256',
        }
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f'invalid Block3a row schema at line {lineno}')
        index = row['index']
        if (not isinstance(index, int) or isinstance(index, bool)
                or index not in expected or index in records):
            raise ValueError(f'invalid/duplicate Block3a index at line {lineno}')
        target = expected[index]
        if (row['schema_version'] != 1 or row['verdict'] != 'PASS'
                or not isinstance(row['run_id'], str)
                or re.fullmatch(r'[0-9a-f]{32}', row['run_id']) is None
                or (run_id is not None and row['run_id'] != run_id)
                or row['kind'] != target['kind']
                or row['tau_lo'] != target['tau_lo']
                or row['tau_hi'] != target['tau_hi']):
            raise ValueError(f'failed or mismatched Block3a row {index}')
        try:
            exact.validate_runtime_record(
                row['worker_runtime'], A.PROOF_PRECISION_BITS, workers=1)
        except ValueError as exc:
            raise ValueError(f'invalid worker runtime in row {index}') from exc
        if (producer_runtime is not None
                and _runtime_identity(row['worker_runtime'])
                != _runtime_identity(producer_runtime)):
            raise ValueError(f'worker/producer runtime mismatch in row {index}')
        if (not isinstance(row['worker_source_sha256'], dict)
                or (expected_source is not None
                    and row['worker_source_sha256'] != expected_source)):
            raise ValueError(f'worker proof source mismatch in row {index}')
        calls = row['recursive_calls']
        leaves = row['leaf_bounds']
        if (not isinstance(calls, int) or isinstance(calls, bool)
                or calls <= 0 or calls > 2 ** (A.MAX_DEPTH + 1) - 1
                or calls % 2 != 1 or not isinstance(leaves, list)
                or len(leaves) != (calls + 1) // 2):
            raise ValueError(f'invalid recursion partition in Block3a row {index}')
        for packet in leaves:
            _, upper = exact.packet_fraction_endpoints(packet)
            if upper >= 0:
                raise ValueError(f'nonnegative leaf upper bound in row {index}')
            max_upper = upper if max_upper is None else max(max_upper, upper)
        elapsed = row['elapsed_seconds']
        if not isinstance(elapsed, str) or _ELAPSED_RE.fullmatch(elapsed) is None:
            raise ValueError(f'invalid elapsed time in Block3a row {index}')
        try:
            if Decimal(elapsed) < 0:
                raise ValueError(f'negative elapsed time in Block3a row {index}')
        except InvalidOperation as exc:
            raise ValueError(f'invalid elapsed time in Block3a row {index}') from exc
        total_leaves += len(leaves)
        records[index] = row
    if set(records) != set(expected):
        raise ValueError('Block3a JSONL does not exactly cover the schedule')
    return records, total_leaves, max_upper


def _validate_receipt(receipt_path):
    receipt_path = exact.require_plain_regular_file(receipt_path)
    run_root = exact.require_plain_directory(receipt_path.parent)
    if receipt_path.name != 'receipt.json':
        raise ValueError('Block3a receipt has the wrong identity')
    receipt_raw = receipt_path.read_bytes()
    receipt_file_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    receipt = _load_canonical_json_bytes(receipt_raw, 'Block3a receipt')
    required = {
        'schema_version', 'kind', 'run_id', 'launch', 'finished_utc',
        'launcher_pid', 'exit_code', 'timed_out', 'source_unchanged',
        'proof_source_sha256_after', 'frozen_source_sha256_after',
        'artifacts', 'receipt_sha256',
    }
    if (not isinstance(receipt, dict) or set(receipt) != required
            or receipt['schema_version'] != SCHEMA_VERSION
            or receipt['kind'] != 'block3a_single_run_receipt'
            or receipt['receipt_sha256'] != exact.payload_sha256(
                receipt, omit=('receipt_sha256',))):
        raise ValueError('invalid Block3a run receipt')
    if (not isinstance(receipt['exit_code'], int)
            or isinstance(receipt['exit_code'], bool)
            or receipt['exit_code'] != 0
            or receipt['timed_out'] is not False
            or receipt['source_unchanged'] is not True):
        raise ValueError('Block3a producer did not finish cleanly')
    if (not isinstance(receipt['launcher_pid'], int)
            or isinstance(receipt['launcher_pid'], bool)
            or receipt['launcher_pid'] <= 0):
        raise ValueError('invalid Block3a launcher process identity')
    if (not isinstance(receipt['run_id'], str)
            or re.fullmatch(r'[0-9a-f]{32}', receipt['run_id']) is None):
        raise ValueError('invalid Block3a run identity')

    if (not isinstance(receipt['launch'], dict)
            or receipt['launch'].get('file') != 'launch.json'):
        raise ValueError('Block3a launch descriptor has the wrong path')
    launch_path, launch_raw = _read_descriptor(
        run_root, receipt['launch'], 'Block3a launch')
    launch = _load_canonical_json_bytes(launch_raw, 'Block3a launch')
    launch_required = {
        'schema_version', 'kind', 'run_id', 'created_utc',
        'proof_source_sha256', 'frozen_source_sha256',
        'runner_source_sha256', 'runtime', 'workers', 'timeout_seconds',
        'command', 'cwd', 'output', 'import_policy', 'launch_sha256',
    }
    if (not isinstance(launch, dict) or set(launch) != launch_required
            or launch['schema_version'] != SCHEMA_VERSION
            or launch['kind'] != 'block3a_single_run_launch'
            or launch['launch_sha256'] != exact.payload_sha256(
                launch, omit=('launch_sha256',))
            or launch['run_id'] != receipt['run_id']):
        raise ValueError('invalid Block3a launch manifest')
    if _parse_utc(launch['created_utc']) > _parse_utc(receipt['finished_utc']):
        raise ValueError('Block3a run timestamps are reversed')

    current_proof = run.source_hashes(run.HERE)
    frozen = run_root / 'frozen_source'
    frozen_proof = run.validate_frozen_tree(frozen)
    if (launch['proof_source_sha256'] != current_proof
            or launch['frozen_source_sha256'] != current_proof
            or receipt['proof_source_sha256_after'] != current_proof
            or receipt['frozen_source_sha256_after'] != current_proof
            or frozen_proof != current_proof):
        raise ValueError('Block3a frozen/current proof source mismatch')
    if launch['runner_source_sha256'] != exact.file_sha256(run.__file__):
        raise ValueError('Block3a producer source mismatch')

    runtime = launch['runtime']
    workers = launch['workers']
    try:
        exact.validate_runtime_record(
            runtime, A.PROOF_PRECISION_BITS, workers=workers)
    except ValueError as exc:
        raise ValueError('Block3a producer runtime/policy mismatch') from exc
    if (not isinstance(workers, int) or isinstance(workers, bool)
            or not 1 <= workers <= 64):
        raise ValueError('Block3a producer runtime/policy mismatch')
    if (not isinstance(launch['timeout_seconds'], int)
            or isinstance(launch['timeout_seconds'], bool)
            or launch['timeout_seconds'] <= 0):
        raise ValueError('Block3a timeout policy mismatch')
    command = launch['command']
    expected_command = [
        runtime.get('executable'), '-I', '-S', '-B', '-c', run.BOOTSTRAP,
        'frozen_source', 'python_flint_site', str(workers),
        '--output', 'block3a.jsonl',
        '--run-id', receipt['run_id'],
    ]
    if (command != expected_command or launch['output'] != 'block3a.jsonl'
            or launch['cwd'] != 'frozen_source'
            or launch['import_policy'] != 'isolated-bootstrap-v3-no-site'):
        raise ValueError('Block3a producer command mismatch')

    artifacts = receipt['artifacts']
    if not isinstance(artifacts, dict) or set(artifacts) != {
            'log', 'stdout', 'stderr'}:
        raise ValueError('Block3a run artifact set mismatch')
    expected_files = {
        'log': 'block3a.jsonl',
        'stdout': 'stdout.txt',
        'stderr': 'stderr.txt',
    }
    if (any(not isinstance(artifacts[key], dict) for key in expected_files)
            or any(artifacts[key].get('file') != filename
                   for key, filename in expected_files.items())):
        raise ValueError('Block3a artifact path is not launch-bound')
    _, log_raw = _read_descriptor(run_root, artifacts['log'], 'Block3a log')
    _read_descriptor(run_root, artifacts['stdout'], 'Block3a stdout')
    _read_descriptor(run_root, artifacts['stderr'], 'Block3a stderr')
    records, total_leaves, max_upper = parse_structured_log_bytes(
        log_raw, A.canonical_schedule(), run_id=receipt['run_id'],
        expected_source=current_proof, producer_runtime=runtime)
    return (receipt, launch, records, total_leaves, max_upper,
            receipt_file_sha256)


def _evaluate(root, receipt_relative):
    receipt_path = _resolve_under(root, receipt_relative)
    (receipt, launch, records, total_leaves, max_upper,
     receipt_file_sha256) = _validate_receipt(receipt_path)
    boundary_pins = A.compute_boundary_pins()
    return {
        'schema_version': SCHEMA_VERSION,
        'kind': 'block3a_certificate',
        'verdict': 'ALL PASS',
        'evidence_model': EVIDENCE_MODEL,
        'policy': {
            'precision_bits': A.PROOF_PRECISION_BITS,
            'max_depth': A.MAX_DEPTH,
            'boundary_precision_bits': A.BOUNDARY_PRECISION_BITS,
            'expected_python_flint': A.EXPECTED_PYTHON_FLINT,
            'expected_flint': A.EXPECTED_FLINT,
        },
        'source_sha256': source_hashes(),
        'verifier_runtime': A._runtime_record(),
        'receipt': {
            'file': pathlib.PurePosixPath(receipt_relative).as_posix(),
            'file_sha256': receipt_file_sha256,
            'receipt_sha256': receipt['receipt_sha256'],
            'launch_sha256': launch['launch_sha256'],
            'run_id': receipt['run_id'],
        },
        'schedule': {
            'cells': A.SCHEDULE_COUNT,
            'pg_cells': A.PG_COUNT,
            'qg_cells': A.QG_COUNT,
            'schedule_sha256': A.EXPECTED_SCHEDULE_SHA256,
        },
        'coverage': {
            'cells': len(records),
            'failures': 0,
            'duplicates': 0,
            'total_recursive_leaves': total_leaves,
            'max_leaf_upper': exact.fraction_record(max_upper),
            'indices_sha256': A.payload_sha256(
                list(range(A.SCHEDULE_COUNT)), omit=()),
        },
        'boundary_pins': boundary_pins,
    }


def assemble(*, receipt, certificate):
    certificate = pathlib.Path(certificate).absolute()
    root = exact.require_plain_directory(certificate.parent)
    receipt_path = exact.require_plain_regular_file(receipt)
    try:
        receipt_relative = receipt_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError('Block3a receipt must live below certificate root') from exc
    if certificate.exists():
        raise FileExistsError('certificate destination already exists')
    payload = _evaluate(root, receipt_relative)
    payload['certificate_sha256'] = A.payload_sha256(payload)
    A._write_bytes_no_clobber(
        certificate, A.canonical_json_bytes(payload) + b'\n')
    verify_certificate(certificate)
    return payload


def verify_certificate(path):
    path = exact.require_plain_regular_file(path)
    data = A.load_certificate(path)
    if (data.get('schema_version') != SCHEMA_VERSION
            or data.get('kind') != 'block3a_certificate'
            or data.get('verdict') != 'ALL PASS'
            or data.get('evidence_model') != EVIDENCE_MODEL):
        raise ValueError('invalid frozen-run Block3a certificate schema/verdict')
    if data.get('certificate_sha256') != A.payload_sha256(data):
        raise ValueError('Block3a certificate payload hash mismatch')
    expected = _evaluate(path.parent, data.get('receipt', {}).get('file'))
    without_hash = {key: value for key, value in data.items()
                    if key != 'certificate_sha256'}
    if without_hash != expected:
        raise ValueError('Block3a certificate/evidence mismatch')
    return data


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    make = sub.add_parser('assemble')
    make.add_argument('--receipt', required=True)
    make.add_argument('--certificate', required=True)
    check = sub.add_parser('verify')
    check.add_argument('certificate')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'assemble':
        data = assemble(receipt=args.receipt, certificate=args.certificate)
    else:
        data = verify_certificate(args.certificate)
    print(data['certificate_sha256'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
