"""Detached finisher: waits for strict Block3a and Block3bc certificates,
then runs the complete source-bound verifier and records the outcome.

Runs windowless (spawned with CREATE_NO_WINDOW) and below normal priority.
Writes results/FINISHED.txt on success, results/FINISH_FAILED.txt on any
failure, and appends progress lines to results/auto_finish.log either way.
The git/tag/zip steps stay manual (notes/finish.ps1) - a human or a wake
session runs them after reading the outcome.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
LOG = os.path.join(RES, 'auto_finish.log')
CREATE_FLAGS = (getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))


def say(msg):
    with open(LOG, 'a') as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def read_any(path):
    raw = open(path, 'rb').read()
    if raw[:2] in (bytes([0xFF, 0xFE]), bytes([0xFE, 0xFF])):
        return raw.decode('utf-16', errors='ignore')
    return raw.decode('utf-8', errors='ignore')


def terminal(path):
    try:
        t = read_any(path)
    except OSError:
        return None
    if 'ALL PASS' in t:
        return 'pass'
    if 'FAILURES' in t:
        return 'fail'
    return None


def certificate_commands():
    return (
        [sys.executable, '-B', 'block3a_singlerun.py', 'verify',
         os.path.join(RES, 'block3a_certificate.json')],
        [sys.executable, '-B', 'block3bc_assemble.py', '--check-only',
         '--certificate', os.path.join(RES, 'block3bc_certificate.json')],
    )


def certificate_paths():
    return (
        os.path.join(RES, 'block3a_certificate.json'),
        os.path.join(RES, 'block3bc_certificate.json'),
    )


def write_failure(message):
    with open(os.path.join(RES, 'FINISH_FAILED.txt'), 'w') as stream:
        stream.write(message)


def main():
    say('auto-finisher armed (Block3a/Block3bc certificate-gated)')
    # Never preserve a sentinel from an earlier source/artifact revision.
    for stale in (os.path.join(RES, 'FINISHED.txt'),
                  os.path.join(RES, 'FINISH_FAILED.txt')):
        try:
            os.remove(stale)
        except FileNotFoundError:
            pass
    while True:
        missing = [path for path in certificate_paths()
                   if not os.path.isfile(path)]
        if not missing:
            break
        say('waiting for missing certificates: '
            + ', '.join(os.path.basename(path) for path in missing))
        time.sleep(900)

    # Once a certificate exists, a verifier failure is terminal: it can mean
    # source drift, tampering, or the wrong arithmetic runtime.  Retrying the
    # same bytes forever would hide a real proof failure.
    block3a_command, block3bc_command = certificate_commands()
    r0 = subprocess.run(
        block3a_command,
        cwd=HERE, capture_output=True, text=True,
        creationflags=CREATE_FLAGS)
    r1 = subprocess.run(
        block3bc_command,
        cwd=HERE, capture_output=True, text=True,
        creationflags=CREATE_FLAGS)
    if r0.returncode != 0 or r1.returncode != 0:
        say(f'certificate verification failed: Block3a rc={r0.returncode}, '
            f'Block3bc rc={r1.returncode}')
        write_failure(
            'certificate verification failed; run this finisher with the '
            'artifacts\' exact Python/flint runtime\n\n'
            + 'Block3a stdout:\n' + r0.stdout[-2500:]
            + '\nBlock3a stderr:\n' + r0.stderr[-2500:]
            + '\nBlock3bc stdout:\n' + r1.stdout[-2500:]
            + '\nBlock3bc stderr:\n' + r1.stderr[-2500:])
        return 1
    say('both certificate verifiers report complete')
    r2 = subprocess.run([sys.executable, '-B', 'verify_all.py'],
                        cwd=HERE, capture_output=True, text=True,
                        creationflags=CREATE_FLAGS)
    say(f'verify_all rc={r2.returncode}')
    tail = r2.stdout[-2500:]
    if r2.returncode != 0:
        write_failure('verify_all failed\n' + tail + r2.stderr[-1000:])
        return 1
    open(os.path.join(RES, 'FINISHED.txt'), 'w').write(
        'ALL CERTIFICATES COMPLETE - hardened verifier return code 0\n\n' + tail +
        '\nRemaining manual steps: notes/finish.ps1 (tag, zip, sync).\n')
    say('FINISHED written; hardened verifier passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
