"""No-clobber, source-frozen producer for a Block3a single-run receipt."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import uuid

import block3bc_exact as exact


HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / 'results'
PROOF_SOURCE_NAMES = (
    'block3a_grid.py',
    'block3bc_exact.py',
    'core.py',
    'dsfun.py',
)
BOOTSTRAP = (
    "import os,runpy,sys;root=sys.argv[1];"
    "cache=root+'/.proof-empty-pycache';"
    "assert not os.path.lexists(cache);"
    "sys.pycache_prefix=cache;sys.dont_write_bytecode=True;"
    "site=sys.argv[2];script=root+'/block3a_grid.py';"
    "sys.path.insert(0,root);sys.path.append(site);"
    "sys.argv=[script]+sys.argv[3:];"
    "runpy.run_path(script,run_name='__main__')"
)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def source_hashes(root):
    root = exact.require_plain_directory(root)
    return {
        name: exact.file_sha256(root / name)
        for name in sorted(PROOF_SOURCE_NAMES)
    }


def validate_frozen_tree(root):
    """Require exactly the four regular proof-source files and nothing else."""
    root = exact.require_plain_directory(root)
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != set(PROOF_SOURCE_NAMES):
        raise ValueError('frozen source tree has missing or extra entries')
    for entry in entries:
        try:
            exact.require_plain_regular_file(entry)
        except ValueError as exc:
            raise ValueError(f'frozen source entry is not a regular file: {entry.name}')
    return source_hashes(root)


def _child_env():
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP',
                'PYTHONINSPECT', 'PYTHONUSERBASE'):
        env.pop(key, None)
    env.update({
        'PYTHONNOUSERSITE': '1',
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONHASHSEED': '0',
    })
    return env


def _gated_child(encoded):
    """Windows gate: wait inside the Job before launching the real proof."""
    try:
        payload = json.loads(base64.urlsafe_b64decode(
            encoded.encode('ascii')).decode('utf-8', errors='strict'))
        if (not isinstance(payload, dict) or set(payload) != {'argv', 'cwd'}
                or not isinstance(payload['argv'], list)
                or not payload['argv']
                or not all(isinstance(item, str) and item
                           for item in payload['argv'])
                or not isinstance(payload['cwd'], str)
                or not payload['cwd']):
            return 124
        if sys.stdin.buffer.read(1) != b'\x01':
            return 125
        exact.apply_worker_policy()
        flags = (getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                 | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
        child = subprocess.Popen(
            payload['argv'], cwd=payload['cwd'], stdin=subprocess.DEVNULL,
            env=_child_env(), creationflags=flags)
        return child.wait()
    except BaseException:
        return 126


def _inside_results(path):
    try:
        return exact.require_confined_directory(
            RESULTS, path, must_exist=False)
    except ValueError as exc:
        raise ValueError('Block3a run directory must be below results/') from exc


def _artifact(path, root):
    root = exact.require_plain_directory(root)
    path = pathlib.Path(os.path.abspath(os.fspath(path)))
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError('Block3a artifact escapes run directory') from exc
    path = exact.resolve_regular_file_under(root, relative)
    return {
        'file': relative,
        'sha256': exact.file_sha256(path),
        'bytes': path.stat().st_size,
    }


def produce(run_dir, workers, timeout_seconds):
    if (not isinstance(workers, int) or isinstance(workers, bool)
            or not 1 <= workers <= 64):
        raise ValueError('workers must be a plain integer in 1..64')
    if (not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool) or timeout_seconds <= 0):
        raise ValueError('timeout must be a positive plain integer')
    exact.apply_worker_policy()

    run_dir = _inside_results(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    exact.require_confined_directory(RESULTS, run_dir, must_exist=True)
    frozen = run_dir / 'frozen_source'
    frozen.mkdir()
    before = source_hashes(HERE)
    for name in PROOF_SOURCE_NAMES:
        source = HERE / name
        target = frozen / name
        with source.open('rb') as inp, target.open('xb') as out:
            shutil.copyfileobj(inp, out)
    frozen_before = validate_frozen_tree(frozen)
    if frozen_before != before:
        raise RuntimeError('frozen Block3a source copy mismatch')

    run_id = uuid.uuid4().hex
    log = run_dir / 'block3a.jsonl'
    stdout_path = run_dir / 'stdout.txt'
    stderr_path = run_dir / 'stderr.txt'
    runtime = exact.runtime_record(60, workers=workers)
    flint_site = str(pathlib.Path(runtime['python_flint_root']).parent)
    actual_command = [
        sys.executable, '-I', '-S', '-B', '-c', BOOTSTRAP, str(frozen),
        flint_site,
        str(workers), '--output', str(log), '--run-id', run_id,
    ]
    command = [
        sys.executable, '-I', '-S', '-B', '-c', BOOTSTRAP, 'frozen_source',
        'python_flint_site',
        str(workers), '--output', 'block3a.jsonl', '--run-id', run_id,
    ]
    launch = {
        'schema_version': 2,
        'kind': 'block3a_single_run_launch',
        'run_id': run_id,
        'created_utc': utc_now(),
        'proof_source_sha256': before,
        'frozen_source_sha256': frozen_before,
        'runner_source_sha256': exact.file_sha256(__file__),
        'runtime': runtime,
        'workers': workers,
        'timeout_seconds': timeout_seconds,
        'command': command,
        'cwd': 'frozen_source',
        'output': log.relative_to(run_dir).as_posix(),
        'import_policy': 'isolated-bootstrap-v3-no-site',
    }
    launch['launch_sha256'] = exact.payload_sha256(
        launch, omit=('launch_sha256',))
    launch_path = run_dir / 'launch.json'
    exact.write_json_atomic(launch_path, launch, overwrite=False)

    creationflags = 0
    if os.name == 'nt':
        creationflags = (getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                         | getattr(subprocess,
                                   'BELOW_NORMAL_PRIORITY_CLASS', 0))
    timed_out = False
    process = None
    job = None
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        if os.name == 'nt':
            payload = exact.canonical_json_bytes({
                'argv': actual_command, 'cwd': str(frozen)})
            encoded = base64.urlsafe_b64encode(payload).decode('ascii')
            wrapper = [sys.executable, '-B', str(pathlib.Path(__file__).resolve()),
                       '_gated-child', encoded]
            process = subprocess.Popen(
                wrapper, cwd=HERE, stdin=subprocess.PIPE,
                stdout=stdout, stderr=stderr, creationflags=creationflags,
                start_new_session=False, bufsize=0)
            try:
                job = exact._attach_windows_kill_job(process)
                if job is None or process.stdin is None:
                    raise RuntimeError('gated launcher exited before Job assignment')
                process.stdin.write(b'\x01')
                process.stdin.flush()
                process.stdin.close()
            except BaseException:
                if job is not None:
                    exact._close_windows_job(job, terminate=True)
                    job = None
                elif process.poll() is None:
                    process.kill()
                process.wait()
                raise
        else:
            process = subprocess.Popen(
                actual_command, cwd=frozen, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, env=_child_env(),
                start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == 'nt' and job is not None:
                exact._close_windows_job(job, terminate=True)
                job = None
            elif os.name != 'nt':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            returncode = process.wait()
        finally:
            exact._close_windows_job(job)

    frozen_after = validate_frozen_tree(frozen)
    current_after = source_hashes(HERE)
    source_unchanged = (before == frozen_after == current_after)
    artifacts = {
        'stdout': _artifact(stdout_path, run_dir),
        'stderr': _artifact(stderr_path, run_dir),
    }
    if log.is_file():
        artifacts['log'] = _artifact(log, run_dir)
    receipt = {
        'schema_version': 2,
        'kind': 'block3a_single_run_receipt',
        'run_id': run_id,
        'launch': _artifact(launch_path, run_dir),
        'finished_utc': utc_now(),
        'launcher_pid': process.pid,
        'exit_code': returncode,
        'timed_out': timed_out,
        'source_unchanged': source_unchanged,
        'proof_source_sha256_after': current_after,
        'frozen_source_sha256_after': frozen_after,
        'artifacts': artifacts,
    }
    receipt['receipt_sha256'] = exact.payload_sha256(
        receipt, omit=('receipt_sha256',))
    receipt_path = run_dir / 'receipt.json'
    exact.write_json_atomic(receipt_path, receipt, overwrite=False)
    print(receipt['receipt_sha256'], flush=True)
    if timed_out:
        return 124
    if returncode != 0 or not source_unchanged:
        return returncode if returncode != 0 else 125
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--timeout-seconds', type=int, default=21600)
    return parser


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == '_gated-child':
        if len(raw) != 2:
            return 123
        return _gated_child(raw[1])
    args = build_parser().parse_args(raw)
    return produce(args.run_dir, args.workers, args.timeout_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
