"""Exact serialization and deterministic schedules for Block 3b/c.

Proof endpoints are rational numbers.  Binary floats are deliberately
rejected at this boundary: a nearest-rounded endpoint can shrink a certified
cell or turn exact coverage into an epsilon-based guess.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from fractions import Fraction

import flint
from flint import arb

from core import rat


SCHEMA_VERSION = 2
PACKET_DIGITS = 60
_INT_RE = re.compile(r'(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z')
_UINT_RE = re.compile(r'(?:0|[1-9][0-9]*)\Z')
_DEC_RE = re.compile(
    r'-?(?:0|[1-9][0-9]*)\.[0-9]+\Z')
_SHA256_RE = re.compile(r'[0-9a-f]{64}\Z')
EXPECTED_PYTHON_FLINT = '0.9.0'
EXPECTED_FLINT = '3.6.0'
_FLINT_TREE_IDENTITY = None
ISOLATED_BOOTSTRAP = (
    "import os,runpy,sys;root=sys.argv[1];site=sys.argv[2];"
    "cache=root+'/.proof-empty-pycache';"
    "assert not os.path.lexists(cache);"
    "sys.pycache_prefix=cache;sys.dont_write_bytecode=True;"
    "name=sys.argv[3];script=root+'/'+name;"
    "sys.path.insert(0,root);sys.path.append(site);"
    "sys.argv=[script]+sys.argv[4:];"
    "runpy.run_path(script,run_name='__main__')"
)


def as_fraction(value) -> Fraction:
    """Accept only already-exact internal rational types."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Fraction(value, 1)
    raise TypeError(f"unsupported rational value {type(value).__name__}")


def _canonical_integer(text, nonnegative=False) -> int:
    if not isinstance(text, str):
        raise TypeError("manifest integer must be a string")
    regex = _UINT_RE if nonnegative else _INT_RE
    if regex.fullmatch(text) is None:
        raise ValueError(f"noncanonical integer {text!r}")
    return int(text)


def parse_fraction_text(text: str) -> Fraction:
    """Parse a strict CLI integer, finite decimal, or reduced ratio."""
    if not isinstance(text, str) or text != text.strip():
        raise ValueError("fraction text has whitespace or a non-string type")
    if '/' in text:
        if text.count('/') != 1:
            raise ValueError("invalid ratio")
        num_text, den_text = text.split('/')
        num = _canonical_integer(num_text)
        den = _canonical_integer(den_text, nonnegative=True)
        if den <= 0:
            raise ValueError("ratio denominator must be positive")
        value = Fraction(num, den)
        if fraction_text(value) != text:
            raise ValueError("ratio is not reduced/canonical")
        return value
    if _INT_RE.fullmatch(text):
        return Fraction(_canonical_integer(text), 1)
    if _DEC_RE.fullmatch(text):
        return Fraction(text)
    raise ValueError(f"invalid exact fraction text {text!r}")


def fraction_from_record(value) -> Fraction:
    if not isinstance(value, dict) or set(value) != {'num', 'den'}:
        raise ValueError("invalid rational record schema")
    num = _canonical_integer(value['num'])
    den = _canonical_integer(value['den'], nonnegative=True)
    if den <= 0:
        raise ValueError("rational denominator must be positive")
    out = Fraction(num, den)
    if str(out.numerator) != value['num'] or str(out.denominator) != value['den']:
        raise ValueError("rational record is not reduced/canonical")
    return out


def fraction_value(value) -> Fraction:
    return fraction_from_record(value) if isinstance(value, dict) \
        else as_fraction(value)


def fraction_record(value) -> dict:
    value = as_fraction(value)
    return {'num': str(value.numerator), 'den': str(value.denominator)}


def fraction_text(value) -> str:
    value = as_fraction(value)
    return (str(value.numerator) if value.denominator == 1
            else f"{value.numerator}/{value.denominator}")


def fraction_arb(value):
    value = as_fraction(value)
    return rat(value.numerator, value.denominator)


def arb_packet(value, digits=PACKET_DIGITS) -> dict:
    """Serialize an outward decimal enclosure returned by Arb itself."""
    if not isinstance(value, arb):
        raise TypeError("arb_packet requires an Arb value")
    if not value.is_finite():
        raise ValueError("cannot serialize a nonfinite Arb value")
    if (not isinstance(digits, int) or isinstance(digits, bool)
            or digits <= 0):
        raise ValueError("packet digits must be a positive integer")
    mid, rad, exp10 = value.mid_rad_10exp(digits)
    return {
        'format': 'arb-midrad10-v1',
        'mid10': str(mid),
        'rad10': str(rad),
        'exp10': int(exp10),
        'digits': int(digits),
    }


def packet_fraction_endpoints(packet):
    required = {'format', 'mid10', 'rad10', 'exp10', 'digits'}
    if not isinstance(packet, dict) or set(packet) != required:
        raise ValueError("invalid Arb packet schema")
    if packet['format'] != 'arb-midrad10-v1':
        raise ValueError("unknown Arb packet format")
    mid = _canonical_integer(packet['mid10'])
    rad = _canonical_integer(packet['rad10'], nonnegative=True)
    if (not isinstance(packet['exp10'], int)
            or isinstance(packet['exp10'], bool)):
        raise ValueError("packet exponent must be a plain integer")
    if (not isinstance(packet['digits'], int)
            or isinstance(packet['digits'], bool) or packet['digits'] <= 0):
        raise ValueError("packet digits must be a positive plain integer")
    exp10 = packet['exp10']
    scale = (Fraction(10 ** exp10, 1) if exp10 >= 0
             else Fraction(1, 10 ** (-exp10)))
    lo, hi = (mid - rad) * scale, (mid + rad) * scale
    return lo, hi


def packet_endpoints(packet):
    lo, hi = packet_fraction_endpoints(packet)
    return fraction_arb(lo), fraction_arb(hi)


def packet_ball(packet):
    lo, hi = packet_endpoints(packet)
    return lo.union(hi)


def packet_contains(packet, value) -> bool:
    return bool(packet_ball(packet).contains(arb(value)))


def packet_abs_upper(packet):
    lo, hi = packet_fraction_endpoints(packet)
    return fraction_arb(max(abs(lo), abs(hi)))


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key}")
        out[key] = value
    return out


def canonical_json_bytes(value) -> bytes:
    def reject_float(obj):
        if isinstance(obj, float):
            raise TypeError("JSON floats are forbidden in proof artifacts")
        if isinstance(obj, dict):
            for key, item in obj.items():
                if not isinstance(key, str):
                    raise TypeError("JSON object keys must be strings")
                reject_float(item)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                reject_float(item)
    reject_float(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def payload_sha256(value, omit=('manifest_sha256',)) -> str:
    if isinstance(value, dict):
        value = {k: v for k, v in value.items() if k not in set(omit)}
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _plain_stat(path, *, directory=None):
    """Return lstat(path), rejecting every link/reparse indirection."""
    path = pathlib.Path(path)
    try:
        info = path.lstat()
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f'missing artifact path {path}') from exc
    reparse = bool(getattr(info, 'st_file_attributes', 0) & 0x400)
    if stat.S_ISLNK(info.st_mode) or reparse:
        raise ValueError(f'link/reparse artifact path is forbidden: {path}')
    if directory is True and not stat.S_ISDIR(info.st_mode):
        raise ValueError(f'artifact path is not a directory: {path}')
    if directory is False and not stat.S_ISREG(info.st_mode):
        raise ValueError(f'artifact path is not a regular file: {path}')
    return info


def require_plain_directory(path):
    path = pathlib.Path(os.path.abspath(os.fspath(path)))
    _plain_stat(path, directory=True)
    return path


def require_plain_regular_file(path):
    path = pathlib.Path(os.path.abspath(os.fspath(path)))
    _plain_stat(path, directory=False)
    return path


def _canonical_relative_path(relative):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('artifact path must be canonical relative POSIX text')
    rel = pathlib.PurePosixPath(relative)
    if (rel.is_absolute() or relative != rel.as_posix()
            or any(part in ('', '.', '..') or ':' in part
                   for part in rel.parts)):
        raise ValueError('artifact path is not canonical and confined')
    return rel


def resolve_regular_file_under(root, relative):
    """Resolve a canonical relative file without following any link component."""
    root = require_plain_directory(root)
    rel = _canonical_relative_path(relative)
    current = root
    for index, part in enumerate(rel.parts):
        current = current / part
        _plain_stat(current, directory=(index < len(rel.parts) - 1))
    real_root = root.resolve(strict=True)
    real_file = current.resolve(strict=True)
    try:
        real_file.relative_to(real_root)
    except ValueError as exc:
        raise ValueError('artifact path escapes its root') from exc
    return current


def require_confined_directory(root, target, *, must_exist=True):
    """Reject symlink/junction components between root and a target directory."""
    root = require_plain_directory(root)
    target = pathlib.Path(os.path.abspath(os.fspath(target)))
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ValueError('directory escapes its required root') from exc
    current = root
    missing = False
    for part in relative.parts:
        current = current / part
        try:
            _plain_stat(current, directory=True)
        except ValueError:
            if must_exist or current.exists() or current.is_symlink():
                raise
            missing = True
        if missing and (current.exists() or current.is_symlink()):
            raise ValueError('noncanonical directory chain')
    if must_exist:
        real_root = root.resolve(strict=True)
        try:
            target.resolve(strict=True).relative_to(real_root)
        except ValueError as exc:
            raise ValueError('directory resolves outside its required root') from exc
    return target


def isolated_python_command(script, runtime, argv):
    """Build a no-site child command pinned to one source and flint tree."""
    script = require_plain_regular_file(script)
    if not isinstance(argv, (list, tuple)) or not all(
            isinstance(item, str) and item for item in argv):
        raise ValueError('isolated child argv must contain nonempty strings')
    validate_runtime_record(
        runtime, runtime.get('precision_bits'), workers=1)
    root = require_plain_directory(script.parent)
    if os.path.lexists(root / '.proof-empty-pycache'):
        raise ValueError('isolated proof cache prefix must be absent')
    flint_root = require_plain_directory(runtime['python_flint_root'])
    flint_site = str(flint_root.parent)
    return [
        runtime['executable'], '-I', '-S', '-B', '-c', ISOLATED_BOOTSTRAP,
        str(root), flint_site, script.name, *argv,
    ]


def load_json(path):
    with open(path, 'rb') as stream:
        raw = stream.read()
    if raw.startswith(b'\xef\xbb\xbf'):
        raise ValueError("BOM is forbidden")
    text = raw.decode('utf-8', errors='strict')
    value = json.loads(text, parse_float=_reject_constant,
                       parse_constant=_reject_constant,
                       object_pairs_hook=_unique_object)
    if raw != canonical_json_bytes(value) + b'\n':
        raise ValueError("JSON artifact is not canonical")
    return value


def write_json_atomic(path, value, overwrite=True):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + '.',
                               suffix='.tmp', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical_json_bytes(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(tmp, path)
        else:
            # Atomic no-clobber publication on the same filesystem.  A
            # concurrent writer wins or loses cleanly; neither can overwrite
            # the other's immutable job record.
            os.link(tmp, path)
            os.unlink(tmp)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def source_hashes(paths) -> dict:
    return {
        name: file_sha256(require_plain_regular_file(path))
        for name, path in sorted(paths.items())
    }


def python_flint_tree_identity(*, fresh=False):
    """Hash the producer's complete python-flint package, including native code."""
    global _FLINT_TREE_IDENTITY
    if _FLINT_TREE_IDENTITY is not None and not fresh:
        return _FLINT_TREE_IDENTITY
    module_file = getattr(flint, '__file__', None)
    if not module_file:
        raise ValueError('python-flint module has no filesystem identity')
    module_path = pathlib.Path(os.path.abspath(module_file))
    require_plain_regular_file(module_path)
    raw_root = require_plain_directory(module_path.parent)
    root = raw_root.resolve(strict=True)
    digest = hashlib.sha256()
    files = []
    for path in root.rglob('*'):
        stat = path.lstat()
        reparse = bool(getattr(stat, 'st_file_attributes', 0) & 0x400)
        if path.is_symlink() or reparse:
            raise ValueError('python-flint package contains a link/reparse entry')
        # ``-B`` prevents new bytecode writes but Python may still execute an
        # existing cache.  Hash every regular package file, including pyc,
        # so the recorded tree is the tree the interpreter could consume.
        if not path.is_file():
            continue
        files.append(path)
    if not files:
        raise ValueError('python-flint package tree is empty')
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode('utf-8')
        digest.update(len(relative).to_bytes(4, 'big'))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    _FLINT_TREE_IDENTITY = (str(root), digest.hexdigest())
    return _FLINT_TREE_IDENTITY


def runtime_record(precision_bits, workers=None, *, fresh_flint=False) -> dict:
    flint_root, flint_tree_sha256 = python_flint_tree_identity(
        fresh=fresh_flint)
    out = {
        'runtime_schema_version': 2,
        'host': platform.node(),
        'python': sys.version.split()[0],
        'executable': sys.executable,
        'python_executable_sha256': file_sha256(sys.executable),
        'python_flint_root': flint_root,
        'python_flint_tree_sha256': flint_tree_sha256,
        'python_flint': flint.__version__,
        'flint': flint.__FLINT_VERSION__,
        'precision_bits': int(precision_bits),
    }
    if workers is not None:
        out['workers'] = int(workers)
    return out


def runtime_identity(runtime):
    if not isinstance(runtime, dict):
        raise ValueError('runtime identity requires a record')
    return {key: value for key, value in runtime.items() if key != 'workers'}


def validate_runtime_record(runtime, precision_bits, workers=None,
                            allow_any_workers=False):
    """Validate a portable producer runtime against fixed proof policy.

    Host/path/binary identity is retained and schema-checked, while the
    verifier may run on another machine.  Producer-side resume validators can
    additionally require equality with ``runtime_record(...)``.
    """
    if (flint.__version__ != EXPECTED_PYTHON_FLINT
            or flint.__FLINT_VERSION__ != EXPECTED_FLINT):
        raise ValueError('current verifier runtime violates fixed proof policy')
    base = {
        'runtime_schema_version', 'host', 'python', 'executable',
        'python_executable_sha256',
        'python_flint_root', 'python_flint_tree_sha256',
        'python_flint', 'flint', 'precision_bits',
    }
    expected = base | ({'workers'} if workers is not None
                       or allow_any_workers else set())
    if not isinstance(runtime, dict) or set(runtime) != expected:
        raise ValueError('runtime record schema mismatch')
    for key in ('host', 'python', 'executable', 'python_flint_root'):
        if not isinstance(runtime[key], str) or not runtime[key]:
            raise ValueError(f'invalid runtime {key}')
    for key in ('python_executable_sha256', 'python_flint_tree_sha256'):
        if (not isinstance(runtime[key], str)
                or _SHA256_RE.fullmatch(runtime[key]) is None):
            raise ValueError(f'invalid runtime {key}')
    if (runtime['runtime_schema_version'] != 2
            or runtime['python_flint'] != EXPECTED_PYTHON_FLINT
            or runtime['flint'] != EXPECTED_FLINT
            or runtime['precision_bits'] != precision_bits):
        raise ValueError('runtime proof policy mismatch')
    if ((workers is not None or allow_any_workers)
            and (not isinstance(runtime['workers'], int)
                 or isinstance(runtime['workers'], bool)
                 or runtime['workers'] <= 0)):
        raise ValueError('runtime workers must be a positive plain integer')
    if workers is not None and runtime['workers'] != workers:
        raise ValueError('runtime worker count mismatch')
    return runtime


def apply_worker_policy():
    """Owner-first priority/affinity policy for isolated proof workers."""
    if os.name == 'nt':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p,
                                               ctypes.c_uint32]
        kernel32.SetPriorityClass.restype = ctypes.c_bool
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_bool
        handle = kernel32.GetCurrentProcess()
        below_normal = 0x00004000
        if not kernel32.SetPriorityClass(handle, below_normal):
            raise OSError("SetPriorityClass(BelowNormal) failed")
        # The machine-wide compute doctrine reserves CPUs 8+ for the owner.
        if not kernel32.SetProcessAffinityMask(handle, 0xFF):
            raise OSError("SetProcessAffinityMask(0xFF) failed")
    else:
        try:
            os.nice(15)
        except (AttributeError, OSError):
            pass


def _attach_windows_kill_job(process):
    """Put a Windows child in a kill-on-close Job Object."""
    if os.name != 'nt' or process.poll() is not None:
        return None
    import ctypes
    from ctypes import wintypes

    class LARGE_INTEGER(ctypes.Structure):
        _fields_ = [('QuadPart', ctypes.c_longlong)]

    class BASIC_LIMITS(ctypes.Structure):
        _fields_ = [
            ('PerProcessUserTimeLimit', LARGE_INTEGER),
            ('PerJobUserTimeLimit', LARGE_INTEGER),
            ('LimitFlags', wintypes.DWORD),
            ('MinimumWorkingSetSize', ctypes.c_size_t),
            ('MaximumWorkingSetSize', ctypes.c_size_t),
            ('ActiveProcessLimit', wintypes.DWORD),
            ('Affinity', ctypes.c_size_t),
            ('PriorityClass', wintypes.DWORD),
            ('SchedulingClass', wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ('ReadOperationCount', ctypes.c_ulonglong),
            ('WriteOperationCount', ctypes.c_ulonglong),
            ('OtherOperationCount', ctypes.c_ulonglong),
            ('ReadTransferCount', ctypes.c_ulonglong),
            ('WriteTransferCount', ctypes.c_ulonglong),
            ('OtherTransferCount', ctypes.c_ulonglong),
        ]

    class EXTENDED_LIMITS(ctypes.Structure):
        _fields_ = [
            ('BasicLimitInformation', BASIC_LIMITS),
            ('IoInfo', IO_COUNTERS),
            ('ProcessMemoryLimit', ctypes.c_size_t),
            ('JobMemoryLimit', ctypes.c_size_t),
            ('PeakProcessMemoryUsed', ctypes.c_size_t),
            ('PeakJobMemoryUsed', ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [
        wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        limits = EXTENDED_LIMITS()
        limits.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
                job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.AssignProcessToJobObject(
                job, wintypes.HANDLE(int(process._handle))):
            error = ctypes.get_last_error()
            if process.poll() is not None:
                kernel32.CloseHandle(job)
                return None
            raise ctypes.WinError(error)
        return job
    except BaseException:
        kernel32.CloseHandle(job)
        raise


def _close_windows_job(job, terminate=False):
    if os.name != 'nt' or job is None:
        return
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    if terminate:
        kernel32.TerminateJobObject(wintypes.HANDLE(job), 1)
    kernel32.CloseHandle(wintypes.HANDLE(job))


def isolated_subprocess_results(specs, workers, workdir,
                                timeout_seconds=14400, retries=1,
                                result_validator=None):
    """Run jobs in fresh hidden subprocesses and yield canonical results.

    Each spec is ``(key, command_prefix)``.  ``--result-file PATH`` is
    appended.  Native crashes, malformed results, and timeouts are retried
    without poisoning a long-lived Pool.  A yielded result file remains on
    disk until the consumer resumes iteration after publishing its durable
    parent-level record.
    """
    if (not isinstance(workers, int) or isinstance(workers, bool)
            or workers <= 0):
        raise ValueError("workers must be a positive plain integer")
    if timeout_seconds <= 0 or retries < 0:
        raise ValueError("invalid timeout/retry policy")
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    pending = [(key, list(command), 0) for key, command in specs]
    active = {}
    failure_evidence = {}
    create_flags = 0
    if os.name == 'nt':
        create_flags = (getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                        | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))

    def launch(item):
        key, command, attempt = item
        token = uuid.uuid4().hex
        result_path = os.path.join(workdir, f'.worker-{key}-{token}.json')
        stderr_path = os.path.join(workdir, f'.worker-{key}-{token}.stderr')
        stderr_stream = open(stderr_path, 'wb')
        process = None
        job = None
        try:
            process = subprocess.Popen(
                command + ['--result-file', result_path], cwd=workdir,
                stdin=subprocess.DEVNULL, stdout=stderr_stream,
                stderr=stderr_stream, creationflags=create_flags,
                start_new_session=(os.name != 'nt'))
            job = _attach_windows_kill_job(process)
        except BaseException:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            stderr_stream.close()
            _close_windows_job(job, terminate=True)
            raise
        active[process.pid] = {
            'process': process, 'key': key, 'command': command,
            'attempt': attempt, 'result': result_path,
            'stderr': stderr_path, 'stderr_stream': stderr_stream,
            'job': job,
            'started': time.monotonic(),
        }

    def stop_state(state):
        process = state['process']
        if process.poll() is None:
            if os.name == 'nt' and state['job'] is not None:
                _close_windows_job(state['job'], terminate=True)
                state['job'] = None
            elif os.name != 'nt':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if not state['stderr_stream'].closed:
            state['stderr_stream'].close()
        _close_windows_job(state['job'])
        state['job'] = None

    try:
        while pending or active:
            while pending and len(active) < workers:
                launch(pending.pop(0))
            progressed = False
            for pid in list(active):
                state = active[pid]
                process = state['process']
                elapsed = time.monotonic() - state['started']
                timed_out = elapsed > timeout_seconds
                if process.poll() is None and not timed_out:
                    continue
                if timed_out and process.poll() is None:
                    stop_state(state)
                else:
                    process.wait()
                    state['stderr_stream'].close()
                    _close_windows_job(state['job'])
                    state['job'] = None
                del active[pid]

                reason = None
                result = None
                if timed_out:
                    reason = f'deadline exceeded after {elapsed:.3f}s'
                elif process.returncode != 0:
                    reason = f'child exit code {process.returncode}'
                elif not os.path.isfile(state['result']):
                    reason = 'child produced no result file'
                else:
                    try:
                        result = load_json(state['result'])
                        if result_validator is not None:
                            result_validator(state['key'], result)
                    except Exception as exc:
                        reason = (
                            f'invalid child result: {type(exc).__name__}: {exc}')

                if reason is None:
                    progressed = True
                    acknowledged = False
                    try:
                        yield state['key'], result
                        acknowledged = True
                    finally:
                        if acknowledged:
                            for path in (state['result'], state['stderr']):
                                try:
                                    os.unlink(path)
                                except FileNotFoundError:
                                    pass
                    continue

                diagnostic = (
                    f'runner attempt={state["attempt"]} elapsed={elapsed:.3f}s '
                    f'reason={reason}\n')
                with open(state['stderr'], 'ab') as evidence_stream:
                    evidence_stream.write(diagnostic.encode('utf-8', 'replace'))
                evidence = {
                    'attempt': state['attempt'], 'reason': reason,
                    'stderr': state['stderr'],
                    'result': (state['result']
                               if os.path.isfile(state['result']) else None),
                }
                failure_evidence.setdefault(state['key'], []).append(evidence)
                if state['attempt'] < retries:
                    pending.append((state['key'], state['command'],
                                    state['attempt'] + 1))
                    progressed = True
                    continue
                details = '; '.join(
                    f"attempt={item['attempt']} reason={item['reason']} "
                    f"stderr={item['stderr']} result={item['result']}"
                    for item in failure_evidence[state['key']])
                raise RuntimeError(
                    f"isolated job {state['key']} failed after "
                    f"{state['attempt'] + 1} attempt(s): {details}")
            if active and not progressed:
                time.sleep(0.2)
    finally:
        for state in list(active.values()):
            stop_state(state)
        active.clear()


def k_nodes():
    """59 exact nodes covering [-0.131, -0.0212]."""
    return tuple(Fraction(-37990 + 549 * j, 290000) for j in range(59))


def ell_boundaries():
    """16 exact cells covering tau in [-0.20, -0.02]."""
    return tuple(Fraction(-160 + 9 * j, 800) for j in range(17))


def c_boundaries():
    """16 exact cells covering tau in [-0.043, 0.078]."""
    return tuple(Fraction(-688 + 121 * j, 16000) for j in range(17))


def b_pos_boundaries():
    """24 exact cells covering tau in [0.06, 0.26]."""
    return tuple(Fraction(3, 50) + Fraction(j, 120) for j in range(25))


def ceil_fraction(value) -> int:
    value = as_fraction(value)
    return -(-value.numerator // value.denominator)


def b_neg_count(k_run) -> int:
    """Deterministic top-cell count from the certified rational K_run."""
    value = Fraction(220, 7) * as_fraction(k_run)
    return max(20, value.numerator // value.denominator + 1)


def b_neg_boundaries(k_run):
    ncell = b_neg_count(k_run)
    return tuple(Fraction(-19, 100) + Fraction(4 * j, 25 * ncell)
                 for j in range(ncell + 1))


def intervals_from_boundaries(boundaries):
    boundaries = [as_fraction(x) for x in boundaries]
    return list(zip(boundaries[:-1], boundaries[1:]))


def require_exact_schedule(records, boundaries, lo_key='tau_lo',
                           hi_key='tau_hi'):
    """Reject duplicates, omissions, reorderings, overlaps, and any gap."""
    expected = intervals_from_boundaries(boundaries)
    actual = []
    for record in records:
        actual.append((fraction_value(record[lo_key]),
                       fraction_value(record[hi_key])))
    if actual != expected:
        raise ValueError("records do not equal the exact canonical schedule")
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate exact interval")
    return True


def require_exact_partition(intervals, target):
    intervals = [(fraction_value(a), fraction_value(b)) for a, b in intervals]
    target = (fraction_value(target[0]), fraction_value(target[1]))
    if not intervals or intervals[0][0] != target[0]:
        raise ValueError("partition does not start at target")
    seen = set()
    reach = target[0]
    for interval in intervals:
        lo, hi = interval
        if interval in seen or lo != reach or not lo < hi:
            raise ValueError("partition duplicate, gap, overlap, or reversal")
        seen.add(interval)
        reach = hi
    if reach != target[1]:
        raise ValueError("partition does not end at target")
    return True


def require_exact_union(intervals, target):
    intervals = [(fraction_value(a), fraction_value(b)) for a, b in intervals]
    target = (fraction_value(target[0]), fraction_value(target[1]))
    if len(intervals) != len(set(intervals)):
        raise ValueError("duplicate interval")
    intervals.sort()
    if not intervals or intervals[0][0] != target[0]:
        raise ValueError("union does not start at target")
    reach = target[0]
    for lo, hi in intervals:
        if lo > reach or not lo < hi:
            raise ValueError("union has a gap or reversed interval")
        reach = max(reach, hi)
    if reach != target[1]:
        raise ValueError("union does not end exactly at target")
    return True


def lane_indices(count, lane, lanes):
    if any(not isinstance(x, int) or isinstance(x, bool)
           for x in (count, lane, lanes)):
        raise ValueError("lane/count fields must be plain integers")
    if lanes <= 0 or not 0 <= lane < lanes:
        raise ValueError("invalid lane")
    return list(range(lane, count, lanes))
