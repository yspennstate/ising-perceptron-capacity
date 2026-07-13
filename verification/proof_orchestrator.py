"""Hidden, crash-safe local continuation of the perceptron proof run.

The orchestrator is deliberately conservative.  It runs at most one heavy
phase at a time, derives progress from source-bound artifacts rather than its
state journal, and never starts a phase unless a frozen source manifest still
matches the exact bytes on disk.  It does not start Azure work.  The optional
local reverse-tail fallback requires an explicit, expiring no-overlap
clearance issued outside this program.

Normal production launch uses pythonw.exe (or a VBS wrapper) and ``run``.
``status`` and ``dry-run`` never create child processes.  ``freeze`` creates
the immutable source manifest and is intended to be run only after all proof
source edits have stopped.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass


HERE = pathlib.Path(__file__).resolve().parent
PROOF_ROOT = HERE.parent
RESULTS = HERE / "results"
DEFAULT_RUN_DIR = RESULTS / "proof_orchestrator"
DEFAULT_RECOVERY = pathlib.Path(
    r"C:\Users\owner\Documents\Codex\2026-07-10\recovery"
    r"\perceptron-azure-block3a-20260710"
)

SCHEMA_VERSION = 1
WATCHER_PID = 16896
WATCHER_CREATE_TIME = 1783669125.430185
WINDOWS_AFFINITY = tuple(range(8))
WINDOWS_AFFINITY_MASK = "0xFF"
LOCAL_FILL_WORKERS = 24
TAIL_WORKERS = 4
HUANG_WORKERS = 8
BLOCK3BC_WORKERS = 8
BLOCK3BC_TIMEOUT = 21600
BLOCK3BC_RETRIES = 2
HUANG_TIMEOUT = 48 * 60 * 60
LOCAL_RECOVERY_TIMEOUT = 30 * 24 * 60 * 60
BLOCK3BC_PHASE_TIMEOUT = 7 * 24 * 60 * 60
TAIL_FALLBACK_TIMEOUT = 7 * 24 * 60 * 60
BLOCK3A_ASSEMBLY_TIMEOUT = 2 * 60 * 60
FINAL_VERIFY_TIMEOUT = 12 * 60 * 60
WAITING_EXIT = 3

MESH_SCRIPT = pathlib.Path(
    r"C:\Users\owner\ai-memories-and-functionality"
    r"\12_cognitive_architecture\agent_mesh\agent_mesh.py")
MESH_AGENT = "codex-proof-orchestrator-run-20260710"
MESH_PARENT = "codex-root-recovery-20260710"
MESH_PROCESS_RESOURCE = (
    "external:process/local/perceptron-proof-orchestrator-20260710")
LOCAL_CUTOVER_RESOURCE = (
    "external:process/local/perceptron-block3a-cutover-20260710")
TAIL_CONTROL_RESOURCE = (
    "external:process/azure/trading-linux-az/perceptron-block3a-20260710")
MESH_PULSE_SECONDS = 10 * 60
WATCHER_STALL_SECONDS = 30 * 60
LOCAL_FILL_HANDOFF_GRACE_SECONDS = 60

ORIGINAL_WORKER_PID = "27528"
ORIGINAL_WORKER_CREATE_TIME = "1783635613.445872"
PREFIX_THRESHOLD = "187"

SOURCE_NAMES = (
    "block1_gardner.py",
    "block2_near_one.py",
    "block3a_grid.py",
    "block3a_run.py",
    "block3a_singlerun.py",
    "block3a_assemble.py",
    "core.py",
    "dsfun.py",
    "huang_region1.py",
    "huang_region1_verify.py",
    "huang_star_interior.py",
    "huang_sweep.py",
    "huang_sweep2.py",
    "huang_sweep_verify.py",
    "huanggrid.py",
    "huang_hessian.py",
    "huang_np.py",
    "block3bc_exact.py",
    "block3bc_aux_generate.py",
    "block3bc_aux_verify.py",
    "block3bc.py",
    "block3bc_assemble.py",
    "verify_all.py",
    "proof_orchestrator.py",
)


class OrchestratorError(RuntimeError):
    pass


class MeshConflict(OrchestratorError):
    pass


class ChildPhaseError(OrchestratorError):
    pass


class WaitingForGate(OrchestratorError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value) -> bytes:
    def reject(obj):
        if isinstance(obj, float):
            raise TypeError("floats are forbidden in orchestrator artifacts")
        if isinstance(obj, dict):
            if not all(isinstance(key, str) for key in obj):
                raise TypeError("JSON keys must be strings")
            for item in obj.values():
                reject(item)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                reject(item)

    reject(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value, self_key="manifest_sha256") -> str:
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if key != self_key}
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key}")
        out[key] = value
    return out


def load_canonical_json(path):
    raw = pathlib.Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("BOM is forbidden")
    value = json.loads(
        raw.decode("utf-8", errors="strict"),
        parse_float=lambda text: (_ for _ in ()).throw(
            ValueError(f"JSON float is forbidden: {text}")),
        parse_constant=lambda text: (_ for _ in ()).throw(
            ValueError(f"JSON constant is forbidden: {text}")),
        object_pairs_hook=_unique_object,
    )
    if raw != canonical_json_bytes(value) + b"\n":
        raise ValueError("JSON is not canonical")
    return value


def atomic_write_json(path, value, *, overwrite=True) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    tmp = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
            tmp.unlink()
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def append_event(path, event, **fields) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"at_utc": utc_now(), "event": event, **fields}
    raw = canonical_json_bytes(row) + b"\n"
    with path.open("ab") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def parse_utc(text: str) -> dt.datetime:
    if not isinstance(text, str):
        raise ValueError("timestamp must be text")
    value = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(dt.timezone.utc)


def pythonw_executable() -> str:
    executable = pathlib.Path(sys.executable)
    if os.name == "nt":
        candidate = executable.with_name("pythonw.exe")
        if candidate.is_file():
            return str(candidate)
    return str(executable)


def console_python_executable() -> str:
    executable = pathlib.Path(sys.executable)
    if os.name == "nt":
        candidate = executable.with_name("python.exe")
        if candidate.is_file():
            return str(candidate)
    return str(executable)


def hidden_creation_flags() -> int:
    if os.name != "nt":
        return 0
    return (getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))


def runtime_fingerprint() -> dict:
    import flint
    import numpy
    import scipy

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "python_flint": flint.__version__,
        "flint": flint.__FLINT_VERSION__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }


def git_provenance(paths) -> dict:
    relative = []
    for path in paths.values():
        try:
            relative.append(pathlib.Path(path).resolve().relative_to(PROOF_ROOT))
        except ValueError as exc:
            raise OrchestratorError(
                f"frozen source is outside proof root: {path}") from exc
    names = [item.as_posix() for item in sorted(relative)]

    def capture(*args) -> bytes:
        result = subprocess.run(
            ["git", *args], cwd=str(PROOF_ROOT), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            creationflags=hidden_creation_flags(), timeout=30,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", "replace").strip()
            raise OrchestratorError(f"git provenance failed: {detail}")
        return result.stdout

    head = capture("rev-parse", "HEAD").decode("ascii", "strict").strip()
    diff = capture("diff", "--no-ext-diff", "--binary", "HEAD", "--", *names)
    index = capture("ls-files", "--stage", "--", *names)
    untracked = capture(
        "ls-files", "--others", "--exclude-standard", "--", *names)
    return {
        "head": head,
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_index_sha256": hashlib.sha256(index).hexdigest(),
        "untracked_sources": [
            row for row in untracked.decode("utf-8", "strict").splitlines()
            if row],
    }


def mesh_command(args, *, timeout=20, expect_json=True):
    if not MESH_SCRIPT.is_file():
        raise OrchestratorError(f"agent mesh CLI is missing: {MESH_SCRIPT}")
    command = [console_python_executable(), str(MESH_SCRIPT), *args]
    result = subprocess.run(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, timeout=timeout,
        creationflags=hidden_creation_flags(),
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode(
            "utf-8", "replace").strip()
        error = MeshConflict if result.returncode == 3 else OrchestratorError
        raise error(
            f"agent mesh command failed ({result.returncode}): {detail}")
    text = result.stdout.decode("utf-8", "strict")
    return json.loads(text) if expect_json else text


def apply_owner_policy() -> None:
    if os.name != "nt":
        with contextlib.suppress(OSError):
            os.nice(15)
        return
    import psutil

    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(list(WINDOWS_AFFINITY))


class Singleton:
    """Crash-released singleton; never relies on a stale PID file."""

    def __init__(self, name="Global\\PerceptronProofOrchestratorV1"):
        self.name = name
        self.handle = None
        self.stream = None

    def __enter__(self):
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateMutexW.argtypes = [
                ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
            kernel32.CreateMutexW.restype = wintypes.HANDLE
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel32.CreateMutexW(None, False, self.name)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
                kernel32.CloseHandle(handle)
                raise OrchestratorError("another proof orchestrator is active")
            self.handle = handle
        else:
            import fcntl

            lock = pathlib.Path(tempfile.gettempdir()) / "perceptron-proof.lock"
            self.stream = lock.open("a+b")
            try:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                self.stream.close()
                raise OrchestratorError(
                    "another proof orchestrator is active") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        if os.name == "nt" and self.handle is not None:
            import ctypes
            ctypes.windll.kernel32.CloseHandle(self.handle)
            self.handle = None
        if self.stream is not None:
            self.stream.close()
            self.stream = None


@dataclass(frozen=True)
class Settings:
    run_dir: pathlib.Path = DEFAULT_RUN_DIR
    recovery_dir: pathlib.Path = DEFAULT_RECOVERY
    frozen_manifest: pathlib.Path | None = None
    tail_clearance: pathlib.Path | None = None
    watcher_pid: int = WATCHER_PID
    watcher_create_time: float = WATCHER_CREATE_TIME
    poll_seconds: int = 30
    once: bool = False

    def normalized(self):
        run_dir = pathlib.Path(self.run_dir).resolve()
        recovery = pathlib.Path(self.recovery_dir).resolve()
        frozen = (pathlib.Path(self.frozen_manifest).resolve()
                  if self.frozen_manifest else run_dir / "frozen_sources.json")
        clearance = (pathlib.Path(self.tail_clearance).resolve()
                     if self.tail_clearance else None)
        return Settings(
            run_dir=run_dir, recovery_dir=recovery,
            frozen_manifest=frozen, tail_clearance=clearance,
            watcher_pid=self.watcher_pid,
            watcher_create_time=self.watcher_create_time,
            poll_seconds=self.poll_seconds, once=self.once,
        )


def mesh_resources(settings: Settings) -> list[str]:
    settings = settings.normalized()
    result_root = RESULTS.resolve()
    resources = [str(result_root), MESH_PROCESS_RESOURCE]
    try:
        settings.run_dir.relative_to(result_root)
    except ValueError:
        resources.insert(1, str(settings.run_dir))
    return resources


def recovery_mesh_resources(settings: Settings) -> list[str]:
    settings = settings.normalized()
    provenance = settings.recovery_dir / "block3a_local_fill_provenance.json"
    return [
        str(settings.recovery_dir / "block3a_local_fill.log"),
        str(provenance),
        str(provenance.with_name(provenance.name + ".tmp")),
        LOCAL_CUTOVER_RESOURCE,
    ]


def remote_tail_mesh_resources(settings: Settings) -> list[str]:
    settings = settings.normalized()
    return [
        str(settings.recovery_dir / "block3a_azure_tail.log"),
        str(settings.recovery_dir / "block3a_azure_tail_provenance.json"),
    ]


class MeshLease:
    """Fail-closed lease for every file and process the controller may mutate."""

    def __init__(self, settings: Settings, *, agent=MESH_AGENT):
        self.settings = settings.normalized()
        self.agent = agent
        self.resources = mesh_resources(self.settings)
        self.recovery_resources = recovery_mesh_resources(self.settings)
        self.remote_tail_resources = remote_tail_mesh_resources(self.settings)
        self.joined = False
        self.recovery_claimed = False
        self.remote_tail_claimed = False
        self.last_pulse = 0.0

    def join_and_claim(self):
        mesh_command([
            "hello", "--agent", self.agent, "--runtime", "codex",
            "--parent", MESH_PARENT, "--status", "working", "--ttl", "45m",
            "--cwd", str(PROOF_ROOT), "--task",
            "Run the hidden, source-frozen perceptron proof continuation",
        ])
        self.joined = True
        mesh_command(
            ["digest", "--agent", self.agent], expect_json=False)
        argv = ["claim", "--agent", self.agent]
        for resource in self.resources:
            argv.extend(["--resource", resource])
        argv.extend([
            "--mode", "exclusive", "--ttl", "45m", "--purpose",
            "Own the unattended perceptron proof pipeline and its evidence",
        ])
        mesh_command(argv)
        self.last_pulse = time.monotonic()

    def claim_recovery(self):
        if self.recovery_claimed:
            self.assert_resources(*self.recovery_resources)
            return
        argv = ["claim", "--agent", self.agent]
        for resource in self.recovery_resources:
            argv.extend(["--resource", resource])
        argv.extend([
            "--mode", "exclusive", "--ttl", "45m", "--purpose",
            "Take over Block3a recovery only after the live watcher exits",
        ])
        mesh_command(argv)
        self.recovery_claimed = True

    def release_recovery(self):
        if not self.recovery_claimed:
            return
        argv = ["release", "--agent", self.agent]
        for resource in self.recovery_resources:
            argv.extend(["--resource", resource])
        mesh_command(argv)
        self.recovery_claimed = False

    def claim_remote_tail(self):
        if self.remote_tail_claimed:
            self.assert_resources(*self.remote_tail_resources)
            return
        argv = ["claim", "--agent", self.agent]
        for resource in self.remote_tail_resources:
            argv.extend(["--resource", resource])
        argv.extend([
            "--mode", "exclusive", "--ttl", "45m", "--purpose",
            "Accept immutable Azure reverse-tail evidence after publication",
        ])
        mesh_command(argv)
        self.remote_tail_claimed = True

    def release_remote_tail(self):
        if not self.remote_tail_claimed:
            return
        argv = ["release", "--agent", self.agent]
        for resource in self.remote_tail_resources:
            argv.extend(["--resource", resource])
        mesh_command(argv)
        self.remote_tail_claimed = False

    def pulse(self, *, force=False):
        if not self.joined:
            raise OrchestratorError("cannot pulse an unjoined mesh lease")
        if force or time.monotonic() - self.last_pulse >= MESH_PULSE_SECONDS:
            mesh_command([
                "pulse", "--agent", self.agent, "--status", "working",
                "--task", "Run the hidden, source-frozen perceptron proof continuation",
            ])
            self.last_pulse = time.monotonic()

    def assert_resources(self, *resources):
        if not self.joined:
            raise OrchestratorError("mutation attempted before mesh join")
        argv = ["assert", "--agent", self.agent]
        for resource in resources:
            argv.extend(["--resource", str(resource)])
        mesh_command(argv)

    def bye(self, status, summary):
        if not self.joined:
            return
        try:
            mesh_command([
                "bye", "--agent", self.agent, "--status", status,
                "--summary", summary,
            ])
        finally:
            self.joined = False


def assert_external_claim(controller, resource, lease_id):
    result = mesh_command([
        "assert", "--agent", controller, "--resource", resource])
    claims = result.get("claims", []) if isinstance(result, dict) else []
    if not any(row.get("claim_id") == lease_id
               and row.get("mode") == "exclusive" for row in claims):
        raise OrchestratorError(
            "tail clearance is not backed by its exact exclusive mesh lease")
    return result


def source_paths() -> dict[str, pathlib.Path]:
    return {name: HERE / name for name in SOURCE_NAMES}


def frozen_payload(paths=None, *, recovery_dir=None) -> dict:
    del recovery_dir  # Legacy split-run recovery is not final v2 evidence.
    production = paths is None
    paths = source_paths() if production else {
        name: pathlib.Path(path).resolve() for name, path in paths.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen sources: {missing}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "perceptron_frozen_source_manifest",
        "created_utc": utc_now(),
        "run_id": uuid.uuid4().hex,
        "runtime": runtime_fingerprint(),
        "source_root": str(PROOF_ROOT.resolve()),
        "recovery_root": None,
        "git": git_provenance(paths) if production else None,
        "policy": {
            "windows_affinity_mask": WINDOWS_AFFINITY_MASK,
            "priority": "BelowNormal",
            "huang_workers": HUANG_WORKERS,
            "block3bc_workers": BLOCK3BC_WORKERS,
            "block3bc_lanes": 1,
            "block3bc_timeout_seconds": BLOCK3BC_TIMEOUT,
            "block3bc_retries": BLOCK3BC_RETRIES,
        },
        "source_sha256": {
            name: file_sha256(path) for name, path in sorted(paths.items())},
        "recovery_sha256": {},
    }
    payload["manifest_sha256"] = payload_sha256(payload)
    return payload


def write_frozen_manifest(path, *, recovery_dir=None) -> dict:
    payload = frozen_payload(recovery_dir=recovery_dir)
    atomic_write_json(path, payload, overwrite=False)
    return payload


def verify_frozen_manifest(path, *, paths=None, recovery_dir=None) -> dict:
    data = load_canonical_json(path)
    if (data.get("schema_version") != SCHEMA_VERSION
            or data.get("kind") != "perceptron_frozen_source_manifest"
            or data.get("manifest_sha256") != payload_sha256(data)):
        raise OrchestratorError("invalid frozen source manifest")
    expected_policy = {
        "windows_affinity_mask": WINDOWS_AFFINITY_MASK,
        "priority": "BelowNormal",
        "huang_workers": HUANG_WORKERS,
        "block3bc_workers": BLOCK3BC_WORKERS,
        "block3bc_lanes": 1,
        "block3bc_timeout_seconds": BLOCK3BC_TIMEOUT,
        "block3bc_retries": BLOCK3BC_RETRIES,
    }
    if data.get("policy") != expected_policy:
        raise OrchestratorError("frozen source policy mismatch")
    if data.get("runtime") != runtime_fingerprint():
        raise OrchestratorError("frozen arithmetic runtime mismatch")
    current_paths = source_paths() if paths is None else {
        name: pathlib.Path(value).resolve() for name, value in paths.items()}
    actual = {name: file_sha256(value)
              for name, value in sorted(current_paths.items())}
    if data.get("source_sha256") != actual:
        raise OrchestratorError("proof source drift after freeze")
    if paths is None:
        if data.get("source_root") != str(PROOF_ROOT.resolve()):
            raise OrchestratorError("frozen proof root mismatch")
        if data.get("recovery_root") is not None:
            raise OrchestratorError("legacy recovery root is not final evidence")
        git = data.get("git")
        if (not isinstance(git, dict)
                or set(git) != {"head", "source_diff_sha256",
                                "source_index_sha256", "untracked_sources"}
                or not isinstance(git["untracked_sources"], list)
                or any(not isinstance(item, str)
                       for item in git["untracked_sources"])
                or len(git["head"]) not in (40, 64)
                or any(len(git[key]) != 64 for key in (
                    "source_diff_sha256", "source_index_sha256"))):
            raise OrchestratorError("invalid frozen git provenance")
        if data.get("recovery_sha256") != {}:
            raise OrchestratorError("legacy recovery sources are not final evidence")
    return data


def archive_existing(path, archive_dir) -> pathlib.Path | None:
    path = pathlib.Path(path)
    if not path.exists():
        return None
    archive_dir = pathlib.Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(path)
    destination = archive_dir / f"{path.name}.{digest}.archive"
    if destination.exists():
        if file_sha256(destination) != digest:
            raise OrchestratorError("archive name collision")
        path.unlink()
        return destination
    os.link(path, destination)
    path.unlink()
    return destination


def archive_snapshot(path, archive_dir) -> pathlib.Path | None:
    """Hard-link a file into the archive without removing the live name."""
    path = pathlib.Path(path)
    if not path.exists():
        return None
    archive_dir = pathlib.Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(path)
    destination = archive_dir / f"{path.name}.{digest}.archive"
    if destination.exists():
        if file_sha256(destination) != digest:
            raise OrchestratorError("archive name collision")
        return destination
    os.link(path, destination)
    return destination


def tree_sha256(path) -> str:
    root = pathlib.Path(path)
    if not root.is_dir() or root.is_symlink():
        raise OrchestratorError(f"unsafe archive tree: {root}")
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_symlink():
            raise OrchestratorError(f"symlink in archive tree: {item}")
        relative = item.relative_to(root).as_posix().encode("utf-8")
        if item.is_dir():
            digest.update(b"D\0" + relative + b"\0")
        elif item.is_file():
            digest.update(b"F\0" + relative + b"\0")
            digest.update(bytes.fromhex(file_sha256(item)))
        else:
            raise OrchestratorError(f"unsupported archive tree entry: {item}")
    return digest.hexdigest()


def archive_tree(path, archive_dir) -> pathlib.Path | None:
    path = pathlib.Path(path)
    if not path.exists():
        return None
    digest = tree_sha256(path)
    archive_dir = pathlib.Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / f"{path.name}.{digest}.archive-dir"
    counter = 1
    while destination.exists():
        destination = archive_dir / (
            f"{path.name}.{digest}.{counter}.archive-dir")
        counter += 1
    os.replace(path, destination)
    return destination


def atomic_publish(source, destination, archive_dir) -> None:
    source = pathlib.Path(source)
    destination = pathlib.Path(destination)
    raw = source.read_bytes()
    if destination.exists() and destination.read_bytes() == raw:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Preserve the old bytes while keeping the canonical name live.  If
        # replacement fails, readers still see the old certified artifact.
        archive_snapshot(destination, archive_dir)
        os.replace(tmp_name, destination)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command_plan(settings: Settings) -> list[dict]:
    settings = settings.normalized()
    pyw = pythonw_executable()
    rec = settings.recovery_dir
    run = settings.run_dir
    aux = run / "block3bc_aux"
    replay = run / "block3bc_replay"
    aux_manifest = aux / "manifest.json"
    commands = [
        {"phase": "LOCAL_FILL_RECOVERY", "argv": [
            pyw, "-B", str(rec / "manifest_runner.py"),
            "--manifest", str(rec / "manifest_local_under235.json"),
            "--output", str(rec / "block3a_local_fill.log"),
            "--provenance", str(rec / "block3a_local_fill_provenance.json"),
            "--workers", "24", "--seed-log", str(rec / "local_prefix.log"),
            "--below-normal", "--affinity", "0,1,2,3,4,5,6,7"]},
        {"phase": "HUANG_STAR_INTERIOR", "argv": [
            pyw, "-B", str(HERE / "huang_star_interior.py"), "generate",
            "--output", str(RESULTS / "huang_star_interior.json")]},
        {"phase": "HUANG_REGION1", "argv": [
            pyw, "-B", str(HERE / "huang_region1.py"),
            str(HUANG_WORKERS)]},
        {"phase": "HUANG_SWEEP1", "argv": [
            pyw, "-B", str(HERE / "huang_sweep.py"),
            str(HUANG_WORKERS), "48"]},
        {"phase": "HUANG_SWEEP2", "argv": [
            pyw, "-B", str(HERE / "huang_sweep2.py"),
            str(HUANG_WORKERS)]},
        {"phase": "HUANG_BUNDLE", "argv": [
            pyw, "-B", str(HERE / "huang_sweep_verify.py"), "bundle",
            "--output", str(RESULTS / "huang_bundle.json"),
            "--star-interior", str(RESULTS / "huang_star_interior.json"),
            "--region1", str(RESULTS / "huang_region1.json"),
            "--sweep1", str(RESULTS / "huang_sweep.json"),
            "--sweep2", str(RESULTS / "huang_sweep2.json")]},
        {"phase": "BLOCK3BC_AUX_ELL", "argv": [
            pyw, "-B", str(HERE / "block3bc_aux_generate.py"), "ell-prime",
            "--lane", "0", "--lanes", "1", "--workers",
            str(BLOCK3BC_WORKERS),
            "--timeout-seconds", "21600", "--retries", "2",
            "--output", str(aux / "ell_prime.lane-0-of-1.json")]},
        {"phase": "BLOCK3BC_AUX_K", "argv": [
            pyw, "-B", str(HERE / "block3bc_aux_generate.py"), "k-grid",
            "--lane", "0", "--lanes", "1", "--workers",
            str(BLOCK3BC_WORKERS),
            "--timeout-seconds", "21600", "--retries", "2",
            "--output", str(aux / "k_grid.lane-0-of-1.json")]},
        {"phase": "BLOCK3BC_AUX_FINALIZE", "argv": [
            pyw, "-B", str(HERE / "block3bc_aux_generate.py"), "finalize",
            "--ell-shard", str(aux / "ell_prime.lane-0-of-1.json"),
            "--k-shard", str(aux / "k_grid.lane-0-of-1.json"),
            "--k-run", "21/2", "--output", str(aux_manifest)]},
    ]
    for part in ("b_pos", "b_neg", "c"):
        commands.append({"phase": f"BLOCK3BC_REPLAY_{part.upper()}", "argv": [
            pyw, "-B", str(HERE / "block3bc.py"), "replay",
            "--part", part, "--aux-manifest", str(aux_manifest),
            "--workers", str(BLOCK3BC_WORKERS),
            "--lane", "0", "--lanes", "1",
            "--timeout-seconds", "21600", "--retries", "2",
            "--output", str(replay / f"{part}.lane-0-of-1.json")]})
    commands.extend([
        {"phase": "BLOCK3BC_ASSEMBLE", "argv": [
            pyw, "-B", str(HERE / "block3bc_assemble.py"),
            "--aux-manifest", str(aux_manifest),
            "--shard", str(replay / "b_pos.lane-0-of-1.json"),
            "--shard", str(replay / "b_neg.lane-0-of-1.json"),
            "--shard", str(replay / "c.lane-0-of-1.json"),
            "--output", str(run / "block3bc_certificate.json")]},
        {"phase": "TAIL_LOCAL_FALLBACK", "argv": [
            pyw, "-B", str(rec / "manifest_runner.py"),
            "--manifest", str(rec / "manifest_remote_tail.json"),
            "--output", str(run / "block3a_tail_fallback.log"),
            "--provenance", str(run / "block3a_tail_fallback_provenance.json"),
            "--workers", "4", "--below-normal", "--affinity",
            "0,1,2,3,4,5,6,7"]},
        {"phase": "VERIFY_ALL", "argv": [
            pyw, "-B", str(HERE / "verify_all.py")]},
    ])
    return commands


def wrapped_child_command(argv, cwd) -> list[str]:
    payload = canonical_json_bytes({
        "argv": list(argv),
        "cwd": str(pathlib.Path(cwd).resolve()),
    })
    encoded = base64.urlsafe_b64encode(payload).decode("ascii")
    return [
        pythonw_executable(), "-B", str(pathlib.Path(__file__).resolve()),
        "_job-child", encoded,
    ]


def job_child_main(encoded) -> int:
    """Wait for parent Job assignment, then create the real phase child."""
    try:
        raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8", "strict"))
        if (not isinstance(payload, dict) or set(payload) != {"argv", "cwd"}
                or not isinstance(payload["argv"], list)
                or not payload["argv"]
                or not all(isinstance(item, str) and item
                           for item in payload["argv"])
                or not isinstance(payload["cwd"], str)):
            return 124
        gate = sys.stdin.buffer.read(1)
        if gate != b"\x01":
            return 125
        apply_owner_policy()
        child = subprocess.Popen(
            payload["argv"], cwd=payload["cwd"], shell=False,
            stdin=subprocess.DEVNULL, creationflags=hidden_creation_flags())
        return child.wait()
    except BaseException:
        return 126


class Orchestrator:
    def __init__(self, settings: Settings, *, dry_run=False, mesh=None,
                 external_claim_checker=None):
        self.settings = settings.normalized()
        self.dry_run = dry_run
        self.state_path = self.settings.run_dir / "state.json"
        self.events_path = self.settings.run_dir / "events.jsonl"
        self.logs_dir = self.settings.run_dir / "logs"
        self.archive_dir = self.settings.run_dir / "archive"
        self._frozen = None
        self.mesh = mesh
        self.external_claim_checker = (
            external_claim_checker or assert_external_claim)
        self._watcher_progress = None
        self._watcher_workers = {}
        self._watcher_count_mismatch_since = None

    def _assert(self, *resources):
        if self.mesh is not None:
            self.mesh.assert_resources(*resources)

    def _pulse(self, *, force=False):
        if self.mesh is not None:
            self.mesh.pulse(force=force)

    def _append_event(self, event, **fields):
        self._assert(self.events_path.parent, self.events_path)
        append_event(self.events_path, event, **fields)

    def _archive_file(self, path):
        self._assert(path, self.archive_dir)
        return archive_existing(path, self.archive_dir)

    def _archive_tree(self, path):
        self._assert(path, self.archive_dir)
        return archive_tree(path, self.archive_dir)

    def _archive_path(self, path):
        path = pathlib.Path(path)
        if path.is_dir() and not path.is_symlink():
            return self._archive_tree(path)
        return self._archive_file(path)

    def _publish(self, source, destination):
        self._assert(destination.parent, destination, self.archive_dir)
        return atomic_publish(source, destination, self.archive_dir)

    def set_state(self, phase, status, **details):
        state = {
            "schema_version": SCHEMA_VERSION,
            "kind": "perceptron_proof_orchestrator_state",
            "phase": phase,
            "status": status,
            "updated_utc": utc_now(),
            "details": details,
        }
        self._assert(self.state_path.parent, self.state_path, self.events_path)
        atomic_write_json(self.state_path, state)
        append_event(self.events_path, "state", phase=phase, status=status,
                     details=details)

    def verify_sources(self):
        self._frozen = verify_frozen_manifest(
            self.settings.frozen_manifest,
            recovery_dir=self.settings.recovery_dir)
        return self._frozen

    def _wait_or_raise(self, phase, reason):
        self._pulse()
        if pathlib.Path(self.settings.frozen_manifest).is_file():
            self.verify_sources()
        self.set_state(phase, "waiting", reason=reason)
        if self.settings.once:
            raise WaitingForGate(reason)
        time.sleep(self.settings.poll_seconds)

    def _expected_watcher_command(self):
        rec = self.settings.recovery_dir
        return [
            pythonw_executable(), "watch_local_cutover.py",
            "--pid", ORIGINAL_WORKER_PID,
            "--create-time", ORIGINAL_WORKER_CREATE_TIME,
            "--threshold", PREFIX_THRESHOLD,
            "--live-log", str((PROOF_ROOT / "results" / "block3a.log").resolve()),
            "--snapshot", str(rec / "local_prefix.log"),
            "--status", str(rec / "cutover_status.log"),
            "--manifest", str(rec / "manifest_local_under235.json"),
            "--output", str(rec / "block3a_local_fill.log"),
            "--provenance", str(rec / "block3a_local_fill_provenance.json"),
            "--workers", str(LOCAL_FILL_WORKERS),
        ]

    def _validate_watcher_worker_count(self, provenance_state, count, now):
        """Allow only a brief one-worker Pool replacement handoff.

        multiprocessing.Pool with worker recycling can expose 23 or 25 direct
        children while an old worker exits and its replacement starts.  The
        controller only waits during this state; a mismatch that persists, or
        any larger deviation, remains a fail-closed error.
        """
        if provenance_state != "running":
            self._watcher_count_mismatch_since = None
            if count > LOCAL_FILL_WORKERS:
                raise OrchestratorError("local-fill watcher has excess children")
            return
        if count == LOCAL_FILL_WORKERS:
            self._watcher_count_mismatch_since = None
            return
        if count not in (LOCAL_FILL_WORKERS - 1, LOCAL_FILL_WORKERS + 1):
            raise OrchestratorError(
                f"local-fill watcher has {count} workers, expected 24")
        if self._watcher_count_mismatch_since is None:
            self._watcher_count_mismatch_since = now
            return
        if now - self._watcher_count_mismatch_since > \
                LOCAL_FILL_HANDOFF_GRACE_SECONDS:
            raise OrchestratorError(
                f"local-fill watcher worker handoff stayed at {count} "
                f"for over {LOCAL_FILL_HANDOFF_GRACE_SECONDS} seconds")

    def _watcher_child_role(self, child):
        """Accept only exact Pool workers or the watcher's mesh heartbeat."""
        command = child.cmdline()
        if not command:
            raise OrchestratorError(
                f"local-fill watcher child {child.pid} has no command line")
        executable = pathlib.Path(child.exe()).resolve()
        command_executable = pathlib.Path(command[0]).resolve()
        cwd = pathlib.Path(child.cwd()).resolve()
        if executable != command_executable or cwd != self.settings.recovery_dir:
            raise OrchestratorError(
                f"unexpected local-fill watcher child {child.pid} executable/cwd")

        pyw = pathlib.Path(pythonw_executable()).resolve()
        if executable == pyw:
            worker = re.fullmatch(
                r"from multiprocessing\.spawn import spawn_main; "
                rf"spawn_main\(parent_pid={self.settings.watcher_pid}, "
                r"pipe_handle=\d+\)",
                command[2] if len(command) == 4 else "")
            if (len(command) == 4 and command[1] == "-c" and worker
                    and command[3] == "--multiprocessing-fork"):
                return "worker"

        python = pathlib.Path(console_python_executable()).resolve()
        # Match the argv emitted by mesh_command() exactly.  In particular,
        # do not resolve a Windows-style configured path on a POSIX watcher:
        # pathlib would reinterpret it as a relative POSIX filename.
        prefix = [str(python), str(MESH_SCRIPT)]
        if executable == python and command[:2] == prefix:
            args = command[2:]
            if args == ["pulse", "--agent", MESH_PARENT]:
                return "mesh"
            watcher_resources = {
                "external:process/27528-tree",
                LOCAL_CUTOVER_RESOURCE,
                str(self.settings.recovery_dir),
            }
            if (len(args) == 5 and args[:3] == ["assert", "--agent", MESH_PARENT]
                    and args[3] == "--resource" and args[4] in watcher_resources):
                return "mesh"
        raise OrchestratorError(
            f"unexpected local-fill watcher child {child.pid} command")

    def watcher_alive(self):
        import psutil

        try:
            process = psutil.Process(self.settings.watcher_pid)
            if abs(process.create_time() - self.settings.watcher_create_time) > 0.5:
                raise OrchestratorError("local-fill watcher PID was reused")
            command = process.cmdline()
            expected = self._expected_watcher_command()
            if command != expected:
                raise OrchestratorError("unexpected local-fill watcher command")
            if (pathlib.Path(process.exe()).resolve()
                    != pathlib.Path(pythonw_executable()).resolve()
                    or pathlib.Path(process.cwd()).resolve()
                    != self.settings.recovery_dir):
                raise OrchestratorError(
                    "unexpected local-fill watcher executable/cwd")
            descendants = process.children(recursive=False)
            workers = []
            helpers = []
            worker_cpu = []
            if os.name == "nt":
                allowed = set(WINDOWS_AFFINITY)
                if (not set(process.cpu_affinity()).issubset(allowed)
                        or process.nice() not in (
                            psutil.BELOW_NORMAL_PRIORITY_CLASS,
                            psutil.IDLE_PRIORITY_CLASS)):
                    raise OrchestratorError(
                        "local-fill watcher escaped BelowNormal/affinity 0xFF")
            for item in descendants:
                try:
                    role = self._watcher_child_role(item)
                    if os.name == "nt":
                        if not set(item.cpu_affinity()).issubset(allowed):
                            raise OrchestratorError(
                                f"process {item.pid} escaped affinity 0xFF")
                        if item.nice() not in (
                                psutil.BELOW_NORMAL_PRIORITY_CLASS,
                                psutil.IDLE_PRIORITY_CLASS):
                            raise OrchestratorError(
                                f"process {item.pid} is above BelowNormal")
                    if role == "worker":
                        cpu = item.cpu_times()
                        workers.append(item)
                        worker_cpu.append(cpu.user + cpu.system)
                    else:
                        helpers.append(item)
                except psutil.NoSuchProcess:
                    continue
            if len(helpers) > 1:
                raise OrchestratorError(
                    "local-fill watcher has multiple simultaneous mesh helpers")
            self._watcher_workers = {
                item.pid: item.create_time() for item in workers}
            provenance_state = None
            provenance = self.settings.recovery_dir / \
                "block3a_local_fill_provenance.json"
            if provenance.is_file():
                with contextlib.suppress(Exception):
                    provenance_state = json.loads(
                        provenance.read_text("utf-8")).get("state")
            now = time.monotonic()
            self._validate_watcher_worker_count(
                provenance_state, len(workers), now)

            signature = tuple(sorted(item.pid for item in workers))
            cpu_total = sum(worker_cpu)
            previous = self._watcher_progress
            if not workers or provenance_state != "running":
                self._watcher_progress = None
            elif (previous is None or previous[0] != signature
                  or cpu_total > previous[2] + 0.01):
                self._watcher_progress = (signature, now, cpu_total)
            elif now - previous[1] > WATCHER_STALL_SECONDS:
                raise OrchestratorError(
                    "local-fill worker tree made no CPU progress for 30 minutes")
            return True, workers
        except psutil.NoSuchProcess:
            self._watcher_progress = None
            self._watcher_count_mismatch_since = None
            return False, []

    def orphan_worker_pids(self):
        import psutil

        found = set()
        for pid, created in self._watcher_workers.items():
            try:
                process = psutil.Process(pid)
                if abs(process.create_time() - created) <= 0.5:
                    found.add(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        marker = f"spawn_main(parent_pid={self.settings.watcher_pid},"
        for process in psutil.process_iter(["pid", "cmdline"]):
            try:
                command = process.info.get("cmdline") or []
                if any(marker in item for item in command):
                    found.add(process.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return sorted(found)

    def _manifest_runner(self):
        return load_module(
            self.settings.recovery_dir / "manifest_runner.py",
            "proof_orchestrator_manifest_runner")

    def validate_local_fill(self) -> bool:
        rec = self.settings.recovery_dir
        needed = [
            rec / "manifest_runner.py", rec / "manifest_local_under235.json",
            rec / "local_prefix.log", rec / "block3a_local_fill.log",
            rec / "block3a_local_fill_provenance.json",
            rec / "original_process_attestation.json",
            rec / "cutover_status.log", rec / "watch_local_cutover.py",
        ]
        if not all(path.is_file() for path in needed):
            return False
        import block3a_assemble as assemble

        try:
            preliminary = assemble.load_evidence_json(
                rec / "block3a_local_fill_provenance.json")
        except Exception:
            return False
        if preliminary.get("state") != "complete":
            return False
        try:
            schedule = assemble.canonical_schedule()
            assemble.validate_manifest(
                rec / "manifest_local_under235.json", schedule,
                list(range(assemble.LOCAL_MANIFEST_COUNT)),
                assemble.EXPECTED_LOCAL_MANIFEST_SHA256, "local")
            _, _, prefix_runtime = assemble.validate_prefix_chain(
                rec / "local_prefix.log",
                rec / "original_process_attestation.json",
                rec / "cutover_status.log", rec / "watch_local_cutover.py",
                schedule)
            fill = assemble.parse_log(
                rec / "block3a_local_fill.log", schedule, "local fill")
            local_indices = list(range(
                assemble.PREFIX_COUNT, assemble.LOCAL_MANIFEST_COUNT))
            if set(fill) != set(local_indices):
                raise ValueError("local fill does not own exactly indices 187..234")
            assemble.validate_runner_provenance(
                rec / "block3a_local_fill_provenance.json",
                rec / "manifest_local_under235.json",
                rec / "block3a_local_fill.log", fill, local_indices,
                assemble.LOCAL_MANIFEST_COUNT, LOCAL_FILL_WORKERS,
                assemble.PREFIX_COUNT, prefix_runtime)
            return True
        except Exception as exc:
            raise OrchestratorError(
                f"completed local-fill evidence is invalid: {exc}") from exc

    def _attach_job(self, process):
        import block3bc_exact as exact
        return exact._attach_windows_kill_job(process)  # audited shared helper

    def _close_job(self, job, terminate=False):
        import block3bc_exact as exact
        exact._close_windows_job(job, terminate=terminate)

    def _stop_process(self, process, job):
        # If lease renewal itself failed, the safest action is still to stop
        # the already-owned child rather than let uncoordinated work continue.
        with contextlib.suppress(Exception):
            self._assert(MESH_PROCESS_RESOURCE)
        if process.poll() is not None:
            self._close_job(job)
            return
        if os.name == "nt" and job is not None:
            self._close_job(job, terminate=True)
            job = None
        elif os.name != "nt":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=15)
        if process.poll() is None:
            process.kill()
            process.wait()
        self._close_job(job)

    def _phase_mutation_targets(self, phase):
        rec = self.settings.recovery_dir
        run = self.settings.run_dir
        mapping = {
            "LOCAL_FILL_RECOVERY": [
                rec / "block3a_local_fill.log",
                rec / "block3a_local_fill_provenance.json",
                rec / "block3a_local_fill_provenance.json.tmp"],
            "HUANG_STAR_INTERIOR": [RESULTS / "huang_star_interior.json"],
            "HUANG_REGION1": [
                RESULTS / "huang_region1.json", RESULTS / "huang_region1.log"],
            "HUANG_SWEEP1": [
                RESULTS / "huang_sweep.json", RESULTS / "huang_sweep.log"],
            "HUANG_SWEEP2": [
                RESULTS / "huang_sweep2.json", RESULTS / "huang_sweep2.log"],
            "HUANG_BUNDLE": [
                RESULTS / "huang_bundle.json",
                RESULTS / "huang_bundle.json.tmp"],
            "BLOCK3BC_AUX_ELL": [run / "block3bc_aux"],
            "BLOCK3BC_AUX_K": [run / "block3bc_aux"],
            "BLOCK3BC_AUX_FINALIZE": [run / "block3bc_aux" / "manifest.json"],
            "BLOCK3BC_REPLAY_B_POS": [run / "block3bc_replay"],
            "BLOCK3BC_REPLAY_B_NEG": [run / "block3bc_replay"],
            "BLOCK3BC_REPLAY_C": [run / "block3bc_replay"],
            "BLOCK3BC_ASSEMBLE": [run / "block3bc_certificate.json"],
            "TAIL_LOCAL_FALLBACK": [
                run / "block3a_tail_fallback.log",
                run / "block3a_tail_fallback_provenance.json"],
            "BLOCK3A_ASSEMBLE": [
                RESULTS / "block3a.log", RESULTS / "block3a_certificate.json",
                run / "block3a_evidence"],
        }
        return mapping.get(phase, [])

    def run_child(self, phase, argv, *, cwd=None, timeout=None, guard=None,
                  guard_interval=60):
        if self.dry_run:
            self._append_event("dry_run_child", phase=phase, argv=argv)
            return 0
        self.verify_sources()
        self._pulse()
        if guard is not None:
            guard()
        targets = [self.logs_dir / f"{phase.lower()}.log",
                   self.settings.run_dir, RESULTS, MESH_PROCESS_RESOURCE,
                   *self._phase_mutation_targets(phase)]
        self._assert(*targets)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log = self.logs_dir / f"{phase.lower()}.log"
        flags = hidden_creation_flags()
        started = time.monotonic()
        last_guard = 0.0
        last_source_check = 0.0
        with log.open("ab") as stream:
            stream.write((f"\n=== {utc_now()} {phase} ===\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
            child_cwd = pathlib.Path(cwd or HERE).resolve()
            launch_argv = (wrapped_child_command(argv, child_cwd)
                           if os.name == "nt" else argv)
            process = subprocess.Popen(
                launch_argv, cwd=str(child_cwd), shell=False,
                stdin=(subprocess.PIPE if os.name == "nt"
                       else subprocess.DEVNULL),
                stdout=stream, stderr=stream, creationflags=flags,
                start_new_session=(os.name != "nt"), bufsize=0,
            )
            job = None
            try:
                job = self._attach_job(process)
                if os.name == "nt":
                    if job is None or process.stdin is None:
                        raise OrchestratorError(
                            "phase launcher exited before Job assignment")
                    process.stdin.write(b"\x01")
                    process.stdin.flush()
                    process.stdin.close()
                self.set_state(phase, "running", pid=process.pid,
                               command_sha256=hashlib.sha256(
                                   "\0".join(argv).encode()).hexdigest())
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if timeout is not None and elapsed > timeout:
                        raise ChildPhaseError(f"{phase} timed out")
                    if elapsed - last_source_check >= 60:
                        self.verify_sources()
                        self._pulse()
                        last_source_check = elapsed
                    if (guard is not None
                            and elapsed - last_guard >= guard_interval):
                        guard()
                        last_guard = elapsed
                    time.sleep(0.2)
                process.wait()
                if guard is not None:
                    guard()
                self._close_job(job)
                job = None
                if process.returncode != 0:
                    raise ChildPhaseError(
                        f"{phase} exited {process.returncode}; see {log}")
            except BaseException:
                self._stop_process(process, job)
                raise
        self.verify_sources()
        self._pulse()
        return 0

    def run_child_with_retries(self, phase, argv, *, attempts=3,
                               retry_delay=30, **kwargs):
        for attempt in range(1, attempts + 1):
            try:
                return self.run_child(phase, argv, **kwargs)
            except ChildPhaseError as exc:
                if attempt == attempts:
                    raise
                self.set_state(
                    phase, "retrying", attempt=attempt,
                    next_attempt=attempt + 1, error=str(exc))
                deadline = time.monotonic() + retry_delay
                while time.monotonic() < deadline:
                    self.verify_sources()
                    self._pulse()
                    time.sleep(min(5, max(0, deadline - time.monotonic())))

    def _phase_command(self, name):
        for row in command_plan(self.settings):
            if row["phase"] == name:
                return row["argv"]
        raise KeyError(name)

    def ensure_local_fill(self):
        while True:
            alive, children = self.watcher_alive()
            if alive:
                self._wait_or_raise(
                    "WAIT_LOCAL_FILL",
                    f"watcher {self.settings.watcher_pid} has {len(children)} children")
                continue
            orphans = self.orphan_worker_pids()
            if orphans:
                self._wait_or_raise(
                    "WAIT_LOCAL_ORPHANS",
                    f"watcher exited but worker PIDs remain: {orphans}")
                continue
            if self.mesh is not None:
                try:
                    self.mesh.claim_recovery()
                except MeshConflict as exc:
                    self._wait_or_raise(
                        "WAIT_RECOVERY_HANDOFF",
                        f"live watcher ownership has not handed off: {exc}")
                    continue
                orphans = self.orphan_worker_pids()
                if orphans:
                    self.mesh.release_recovery()
                    self._wait_or_raise(
                        "WAIT_LOCAL_ORPHANS",
                        f"workers appeared during recovery handoff: {orphans}")
                    continue
            if self.validate_local_fill():
                self.set_state("LOCAL_FILL", "complete")
                return
            self.run_child_with_retries(
                "LOCAL_FILL_RECOVERY",
                self._phase_command("LOCAL_FILL_RECOVERY"),
                cwd=self.settings.recovery_dir,
                timeout=LOCAL_RECOVERY_TIMEOUT,
            )
            if not self.validate_local_fill():
                raise OrchestratorError("local fill runner did not complete evidence")

    def ensure_frozen_sources(self):
        if not pathlib.Path(self.settings.frozen_manifest).is_file():
            reason = f"create frozen manifest at {self.settings.frozen_manifest}"
            self.set_state("WAIT_SOURCE_FREEZE", "waiting", reason=reason)
            raise WaitingForGate(reason)
        self.verify_sources()
        self.set_state("SOURCE_FREEZE", "complete",
                       manifest_sha256=self._frozen["manifest_sha256"])

    @staticmethod
    def _star_interior_valid(path) -> bool:
        if not pathlib.Path(path).is_file():
            return False
        try:
            import huang_star_interior as interior
            interior.verify_certificate(path)
            return True
        except Exception:
            return False

    @staticmethod
    def _region1_valid(path) -> bool:
        if not pathlib.Path(path).is_file():
            return False
        try:
            import verify_all
            ok, _ = verify_all.validate_region1_manifest(path)
            return bool(ok)
        except Exception:
            return False

    @staticmethod
    def _sweep_valid(path, stage) -> bool:
        if not pathlib.Path(path).is_file():
            return False
        try:
            import huang_sweep
            import huang_sweep2
            import verify_all
            module = huang_sweep if stage == 1 else huang_sweep2
            ok, _, _ = verify_all.validate_sweep_manifest(path, module, stage)
            return bool(ok)
        except Exception:
            return False

    @staticmethod
    def _huang_bundle_valid(path) -> bool:
        if not pathlib.Path(path).is_file():
            return False
        try:
            import huang_sweep_verify
            huang_sweep_verify.verify_bundle(path)
            return True
        except Exception:
            return False

    def ensure_huang(self):
        phases = [
            ("HUANG_STAR_INTERIOR", RESULTS / "huang_star_interior.json",
             lambda path: self._star_interior_valid(path)),
            ("HUANG_REGION1", RESULTS / "huang_region1.json",
             lambda path: self._region1_valid(path)),
            ("HUANG_SWEEP1", RESULTS / "huang_sweep.json",
             lambda path: self._sweep_valid(path, 1)),
            ("HUANG_SWEEP2", RESULTS / "huang_sweep2.json",
             lambda path: self._sweep_valid(path, 2)),
        ]
        for phase, artifact, validator in phases:
            self.verify_sources()
            if validator(artifact):
                self.set_state(phase, "complete", resumed=True)
                continue
            if artifact.exists():
                self._archive_file(artifact)
            legacy_log = artifact.with_suffix(".log")
            if legacy_log.exists():
                self._archive_file(legacy_log)
            self.run_child(
                phase, self._phase_command(phase), timeout=HUANG_TIMEOUT)
            if not validator(artifact):
                raise OrchestratorError(f"{phase} produced invalid evidence")
            self.set_state(phase, "complete", resumed=False)
        bundle_path = RESULTS / "huang_bundle.json"
        self.verify_sources()
        if self._huang_bundle_valid(bundle_path):
            self.set_state("HUANG_BUNDLE", "complete", resumed=True)
            return
        if bundle_path.exists():
            self._archive_file(bundle_path)
        self.run_child(
            "HUANG_BUNDLE", self._phase_command("HUANG_BUNDLE"),
            timeout=HUANG_TIMEOUT)
        if not self._huang_bundle_valid(bundle_path):
            raise OrchestratorError(
                "HUANG_BUNDLE produced invalid delegation evidence")
        self.set_state("HUANG_BUNDLE", "complete", resumed=False)

    @staticmethod
    def _aux_resume_records_valid(kind, output):
        record_dir = pathlib.Path(str(output) + ".records")
        if not record_dir.exists():
            return True
        if not record_dir.is_dir() or record_dir.is_symlink():
            return False
        try:
            import block3bc_aux_generate as generate
            import block3bc_exact as exact

            generate.set_prec(generate.PRECISION_BITS)
            hashes = exact.source_hashes(generate._source_paths())
            if kind == "ell_prime":
                cells = generate.intervals_from_boundaries(
                    generate.ell_boundaries())
                indices = generate.lane_indices(len(cells), 0, 1)
                job_inputs = {
                    i: {"tau_lo": generate.fraction_record(cells[i][0]),
                        "tau_hi": generate.fraction_record(cells[i][1])}
                    for i in indices}
                prefix = "ellp"
            else:
                nodes = generate.k_nodes()
                indices = generate.lane_indices(len(nodes), 0, 1)
                job_inputs = {
                    i: {"lambda_value": generate.fraction_record(nodes[i])}
                    for i in indices}
                prefix = "i2"
            for i in indices:
                path = record_dir / f"{prefix}-{i:03d}.json"
                if path.exists():
                    generate._validate_job_record(
                        exact.load_json(path), kind, i, job_inputs[i], hashes)
            return True
        except Exception:
            return False

    @staticmethod
    def _replay_resume_records_valid(part, output, aux):
        record_dir = pathlib.Path(str(output) + ".records")
        if not record_dir.exists():
            return True
        if not record_dir.is_dir() or record_dir.is_symlink():
            return False
        try:
            import block3bc
            import block3bc_exact as exact

            k_run = aux["k_run"]
            intervals = exact.intervals_from_boundaries(
                block3bc._part_boundaries(part, k_run))
            indices = exact.lane_indices(len(intervals), 0, 1)
            job_inputs = {
                i: {"tau_lo": exact.fraction_record(intervals[i][0]),
                    "tau_hi": exact.fraction_record(intervals[i][1]),
                    "k_run": exact.fraction_record(k_run)}
                for i in indices}
            proof_paths = {
                "block3bc.py": block3bc.__file__,
                "block3bc_exact.py": exact.__file__,
                "core.py": sys.modules["core"].__file__,
                "dsfun.py": block3bc.dsfun.__file__,
            }
            hashes = exact.source_hashes(proof_paths)
            aux_hash = aux["manifest"]["manifest_sha256"]
            for i in indices:
                path = record_dir / f"{part}-{i:04d}.json"
                if path.exists():
                    block3bc._validate_replay_job(
                        exact.load_json(path), part, i, job_inputs[i],
                        aux_hash, hashes)
            return True
        except Exception:
            return False

    def ensure_block3bc(self):
        import block3bc_assemble as assemble
        import block3bc_aux_verify as aux_verify

        run = self.settings.run_dir
        aux_dir = run / "block3bc_aux"
        replay_dir = run / "block3bc_replay"
        ell = aux_dir / "ell_prime.lane-0-of-1.json"
        grid = aux_dir / "k_grid.lane-0-of-1.json"
        manifest = aux_dir / "manifest.json"

        for phase, path, kind in (
                ("BLOCK3BC_AUX_ELL", ell, "ell_prime"),
                ("BLOCK3BC_AUX_K", grid, "k_grid")):
            valid = False
            if path.is_file():
                with contextlib.suppress(Exception):
                    aux_verify.verify_shard(path, kind)
                    valid = True
            if not valid:
                records = pathlib.Path(str(path) + ".records")
                if not self._aux_resume_records_valid(kind, path):
                    if records.exists():
                        self._archive_path(records)
                if path.exists():
                    self._archive_file(path)
                self.run_child(
                    phase, self._phase_command(phase),
                    timeout=BLOCK3BC_PHASE_TIMEOUT)
                aux_verify.verify_shard(path, kind)

        manifest_valid = False
        if manifest.is_file():
            with contextlib.suppress(Exception):
                aux_verify.verify_manifest(manifest, require_complete=True)
                manifest_valid = True
        if not manifest_valid:
            if manifest.exists():
                self._archive_file(manifest)
            self.run_child(
                "BLOCK3BC_AUX_FINALIZE",
                self._phase_command("BLOCK3BC_AUX_FINALIZE"),
                timeout=BLOCK3A_ASSEMBLY_TIMEOUT)
            aux_verify.verify_manifest(manifest, require_complete=True)

        aux = aux_verify.verify_manifest(manifest, require_complete=True)
        shards = []
        for part in ("b_pos", "b_neg", "c"):
            phase = f"BLOCK3BC_REPLAY_{part.upper()}"
            path = replay_dir / f"{part}.lane-0-of-1.json"
            valid = False
            if path.is_file():
                with contextlib.suppress(Exception):
                    assemble.verify_replay_shard(path, aux)
                    valid = True
            if not valid:
                records = pathlib.Path(str(path) + ".records")
                if not self._replay_resume_records_valid(part, path, aux):
                    if records.exists():
                        self._archive_path(records)
                if path.exists():
                    self._archive_file(path)
                self.run_child(
                    phase, self._phase_command(phase),
                    timeout=BLOCK3BC_PHASE_TIMEOUT)
                assemble.verify_replay_shard(path, aux)
            shards.append(path)

        run_certificate = run / "block3bc_certificate.json"
        valid = False
        if run_certificate.is_file():
            with contextlib.suppress(Exception):
                assemble.verify_certificate(run_certificate)
                valid = True
        if not valid:
            if run_certificate.exists():
                self._archive_file(run_certificate)
            self.run_child(
                "BLOCK3BC_ASSEMBLE", self._phase_command("BLOCK3BC_ASSEMBLE"),
                timeout=BLOCK3A_ASSEMBLY_TIMEOUT)
            assemble.verify_certificate(run_certificate)
        canonical = RESULTS / "block3bc_certificate.json"
        self._publish(run_certificate, canonical)
        assemble.verify_certificate(canonical)
        self.set_state("BLOCK3BC", "complete")

    def validate_tail_clearance(self):
        path = self.settings.tail_clearance
        if path is None or not pathlib.Path(path).is_file():
            raise WaitingForGate("no machine-neutral tail evidence or fallback clearance")
        data = load_canonical_json(path)
        required = {
            "schema_version", "kind", "decision", "lease_id", "controller",
            "issued_utc", "expires_utc", "remote_manifest_sha256",
            "resource", "clearance_sha256",
        }
        if not isinstance(data, dict) or set(data) != required:
            raise OrchestratorError("invalid tail fallback clearance schema")
        if (data["schema_version"] != 1
                or data["kind"] != "block3a_tail_fallback_clearance"
                or data["decision"]
                != "remote_tail_inactive_local_fallback_exclusive"
                or data["resource"] != TAIL_CONTROL_RESOURCE
                or data["clearance_sha256"]
                != payload_sha256(data, "clearance_sha256")):
            raise OrchestratorError("invalid tail fallback clearance")
        manifest = self.settings.recovery_dir / "manifest_remote_tail.json"
        if data["remote_manifest_sha256"] != file_sha256(manifest):
            raise OrchestratorError("tail clearance is for another manifest")
        now = dt.datetime.now(dt.timezone.utc)
        if not (parse_utc(data["issued_utc"]) <= now < parse_utc(data["expires_utc"])):
            raise WaitingForGate("tail fallback clearance is not currently valid")
        if not data["lease_id"] or not data["controller"]:
            raise OrchestratorError("tail clearance lacks lease identity")
        self.external_claim_checker(
            data["controller"], data["resource"], data["lease_id"])
        return data

    def validate_tail_evidence(self, paths) -> bool:
        log_path, provenance_path = map(pathlib.Path, paths)
        if not log_path.is_file() or not provenance_path.is_file():
            return False
        import block3a_assemble as assemble

        try:
            schedule = assemble.canonical_schedule()
            manifest = self.settings.recovery_dir / "manifest_remote_tail.json"
            remote_indices_desc = list(range(
                assemble.SCHEDULE_COUNT - 1,
                assemble.LOCAL_MANIFEST_COUNT - 1, -1))
            assemble.validate_manifest(
                manifest, schedule, remote_indices_desc,
                assemble.EXPECTED_REMOTE_MANIFEST_SHA256, "remote")
            records = assemble.parse_log(log_path, schedule, "reverse tail")
            expected = list(range(
                assemble.LOCAL_MANIFEST_COUNT, assemble.SCHEDULE_COUNT))
            if set(records) != set(expected):
                return False
            assemble.validate_runner_provenance(
                provenance_path, manifest, log_path, records, expected,
                assemble.REMOTE_TAIL_COUNT, TAIL_WORKERS, 0)
            return True
        except Exception:
            return False

    @property
    def tail_receipt_path(self):
        return self.settings.run_dir / "block3a_tail_fallback_receipt.json"

    @property
    def remote_tail_paths(self):
        rec = self.settings.recovery_dir
        return (rec / "block3a_azure_tail.log",
                rec / "block3a_azure_tail_provenance.json")

    def _fallback_receipt_valid(self, paths) -> bool:
        if not self.tail_receipt_path.is_file():
            return False
        try:
            data = load_canonical_json(self.tail_receipt_path)
            required = {
                "schema_version", "kind", "clearance_sha256",
                "remote_manifest_sha256", "log_sha256", "provenance_sha256",
                "completed_utc", "receipt_sha256",
            }
            return bool(
                isinstance(data, dict) and set(data) == required
                and data["schema_version"] == 1
                and data["kind"] == "block3a_tail_fallback_receipt"
                and data["receipt_sha256"]
                == payload_sha256(data, "receipt_sha256")
                and data["remote_manifest_sha256"] == file_sha256(
                    self.settings.recovery_dir / "manifest_remote_tail.json")
                and data["log_sha256"] == file_sha256(paths[0])
                and data["provenance_sha256"] == file_sha256(paths[1])
                and parse_utc(data["completed_utc"]))
        except Exception:
            return False

    def _write_fallback_receipt(self, clearance, paths):
        payload = {
            "schema_version": 1,
            "kind": "block3a_tail_fallback_receipt",
            "clearance_sha256": clearance["clearance_sha256"],
            "remote_manifest_sha256": file_sha256(
                self.settings.recovery_dir / "manifest_remote_tail.json"),
            "log_sha256": file_sha256(paths[0]),
            "provenance_sha256": file_sha256(paths[1]),
            "completed_utc": utc_now(),
        }
        payload["receipt_sha256"] = payload_sha256(payload, "receipt_sha256")
        if self.tail_receipt_path.exists():
            self._archive_file(self.tail_receipt_path)
        self._assert(self.tail_receipt_path.parent, self.tail_receipt_path)
        atomic_write_json(self.tail_receipt_path, payload, overwrite=False)

    def tail_paths(self):
        remote = self.remote_tail_paths
        if self.validate_tail_evidence(remote):
            return remote
        fallback = (self.settings.run_dir / "block3a_tail_fallback.log",
                    self.settings.run_dir / "block3a_tail_fallback_provenance.json")
        if (self.validate_tail_evidence(fallback)
                and self._fallback_receipt_valid(fallback)):
            return fallback
        return None

    def ensure_tail(self):
        while True:
            paths = self.tail_paths()
            if paths is not None:
                if paths == self.remote_tail_paths and self.mesh is not None:
                    try:
                        self.mesh.claim_remote_tail()
                    except MeshConflict as exc:
                        self._wait_or_raise(
                            "WAIT_REMOTE_TAIL_HANDOFF",
                            f"Azure evidence publisher still owns tail: {exc}")
                        continue
                    if not self.validate_tail_evidence(paths):
                        self.mesh.release_remote_tail()
                        raise OrchestratorError(
                            "remote tail changed during ownership handoff")
                return paths
            try:
                clearance = self.validate_tail_clearance()
            except WaitingForGate as exc:
                self._wait_or_raise("WAIT_REVERSE_TAIL", str(exc))
                continue
            if self.tail_receipt_path.exists():
                self._archive_file(self.tail_receipt_path)

            def clearance_guard():
                current = self.validate_tail_clearance()
                if (current["clearance_sha256"]
                        != clearance["clearance_sha256"]):
                    raise OrchestratorError(
                        "tail clearance changed during fallback computation")
                return current

            self.run_child(
                "TAIL_LOCAL_FALLBACK", self._phase_command("TAIL_LOCAL_FALLBACK"),
                timeout=TAIL_FALLBACK_TIMEOUT,
                guard=clearance_guard, guard_interval=5)
            fallback = (
                self.settings.run_dir / "block3a_tail_fallback.log",
                self.settings.run_dir / "block3a_tail_fallback_provenance.json")
            if not self.validate_tail_evidence(fallback):
                raise OrchestratorError("tail fallback produced no evidence")
            # The post-exit guard in run_child proves the exact mesh lease was
            # live through completion.  Only then make the fallback resumable.
            self._write_fallback_receipt(clearance, fallback)
            if self.tail_paths() != fallback:
                raise OrchestratorError("tail fallback receipt validation failed")
            return fallback

    def ensure_block3a_certificate(self):
        import block3a_singlerun

        certificate = RESULTS / "block3a_certificate.json"
        if not certificate.is_file():
            raise OrchestratorError(
                "final policy requires a Block3a single-run frozen-source "
                "v2 certificate; legacy assembly is disabled")
        try:
            block3a_singlerun.verify_certificate(certificate)
        except Exception as exc:
            raise OrchestratorError(
                "Block3a v2 certificate failed exact replay; refusing to "
                "archive it or downgrade to legacy evidence") from exc
        self.set_state("BLOCK3A", "complete", resumed=True)

    def final_verify(self):
        self.run_child(
            "VERIFY_ALL", self._phase_command("VERIFY_ALL"),
            timeout=FINAL_VERIFY_TIMEOUT)
        self.set_state("DONE", "complete")

    def run(self):
        apply_owner_policy()
        with Singleton():
            if self.mesh is None:
                self.mesh = MeshLease(self.settings)
            lease_ready = False
            result = 1
            bye_status = "offline"
            bye_summary = "Proof orchestrator exited before acquiring its lease"
            try:
                self.mesh.join_and_claim()
                lease_ready = True
                self._assert(self.settings.run_dir)
                self.settings.run_dir.mkdir(parents=True, exist_ok=True)
                self.ensure_frozen_sources()
                self.ensure_block3a_certificate()
                self.ensure_huang()
                self.ensure_block3bc()
                self.final_verify()
                result = 0
                bye_status = "done"
                bye_summary = "All proof certificates and final verification completed"
            except WaitingForGate as exc:
                if lease_ready:
                    self._append_event("waiting_exit", reason=str(exc))
                result = WAITING_EXIT
                bye_status = "waiting"
                bye_summary = f"Proof orchestrator paused at a gate: {exc}"
            except BaseException as exc:
                if lease_ready:
                    with contextlib.suppress(Exception):
                        self.set_state("FAILED", "failed",
                                       error=f"{type(exc).__name__}: {exc}")
                bye_summary = (
                    f"Proof orchestrator stopped safely: {type(exc).__name__}: {exc}")
            finally:
                if self.mesh is not None and getattr(self.mesh, "joined", False):
                    try:
                        self.mesh.bye(bye_status, bye_summary)
                    except Exception:
                        result = 1
            return result

    def status(self):
        state = None
        if self.state_path.is_file():
            with contextlib.suppress(Exception):
                state = load_canonical_json(self.state_path)
        alive = False
        children = 0
        with contextlib.suppress(Exception):
            alive, descendants = self.watcher_alive()
            children = len(descendants)
        return {
            "state": state,
            "watcher_alive": alive,
            "watcher_children": children,
            "frozen_manifest_exists": pathlib.Path(
                self.settings.frozen_manifest).is_file(),
            "reverse_tail_exists": self.tail_paths() is not None,
            "block3a_certificate_exists": (
                RESULTS / "block3a_certificate.json").is_file(),
            "block3bc_certificate_exists": (
                RESULTS / "block3bc_certificate.json").is_file(),
        }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--recovery-dir", default=str(DEFAULT_RECOVERY))
    parser.add_argument("--frozen-manifest")
    parser.add_argument("--tail-clearance")
    parser.add_argument("--watcher-pid", type=int, default=WATCHER_PID)
    parser.add_argument("--watcher-create-time", type=float,
                        default=WATCHER_CREATE_TIME)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    sub.add_parser("dry-run")
    sub.add_parser("status")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--output")
    return parser


def settings_from_args(args):
    return Settings(
        run_dir=pathlib.Path(args.run_dir),
        recovery_dir=pathlib.Path(args.recovery_dir),
        frozen_manifest=(pathlib.Path(args.frozen_manifest)
                         if args.frozen_manifest else None),
        tail_clearance=(pathlib.Path(args.tail_clearance)
                        if args.tail_clearance else None),
        watcher_pid=args.watcher_pid,
        watcher_create_time=args.watcher_create_time,
        poll_seconds=args.poll_seconds,
        once=args.once,
    ).normalized()


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "_job-child":
        if len(raw_argv) != 2:
            return 123
        return job_child_main(raw_argv[1])
    args = build_parser().parse_args(raw_argv)
    settings = settings_from_args(args)
    if args.command == "freeze":
        output = pathlib.Path(args.output).resolve() if args.output \
            else pathlib.Path(settings.frozen_manifest)
        payload = write_frozen_manifest(
            output, recovery_dir=settings.recovery_dir)
        print(f"frozen source manifest {payload['manifest_sha256']} {output}")
        return 0
    orchestrator = Orchestrator(settings, dry_run=args.command == "dry-run")
    if args.command == "status":
        print(json.dumps(orchestrator.status(), indent=2, sort_keys=True))
        return 0
    if args.command == "dry-run":
        print(json.dumps(command_plan(settings), indent=2))
        return 0
    return orchestrator.run()


if __name__ == "__main__":
    raise SystemExit(main())
