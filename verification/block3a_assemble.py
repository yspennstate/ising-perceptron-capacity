"""Fail-closed assembler and verifier for the sharded Block-3a replay.

This module binds the frozen original-process prefix, the local manifest fill,
and the machine-neutral reverse-tail shard.  The historical text rows are trusted
execution verdicts: ``str(arb)`` is not a lossless interval serialization, so
the row values cannot be independently reconstructed.  The certificate
therefore pins the exact proof sources, runtimes, execution tools, manifests,
raw evidence, exact schedule ownership, and the cheap endpoint calculations.

No default command touches the canonical results.  ``assemble`` requires all
destinations explicitly and publishes with no-clobber semantics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from fractions import Fraction

import flint
from flint import arb

import block3a_grid
import core
import dsfun


SCHEMA_VERSION = 1
PROOF_PRECISION_BITS = 60
MAX_DEPTH = 5
BOUNDARY_PRECISION_BITS = 80
SCHEDULE_COUNT = 247
PG_COUNT = 115
QG_COUNT = 132
PREFIX_COUNT = 187
ATTESTED_PREFIX_COUNT = 163
LOCAL_MANIFEST_COUNT = 235
REMOTE_TAIL_COUNT = 12

HERE = pathlib.Path(__file__).resolve().parent
PROOF_ROOT = HERE.parent

EXPECTED_PROOF_SOURCE_SHA256 = {
    "block3a_grid.py": "09c3cac1b29c48a45ae29b095338661506f04f544d38695c67b9810f19860231",
    "core.py": "f018024ed3754a207cd4ca8265f25b093445f26f5b174ba987fcea6ac8688e59",
    "dsfun.py": "883c7139b0dbcf4fae4c3728acc60f3d4f4135a47798175973999291b6e0a5b8",
}
EXPECTED_SCHEDULE_SHA256 = (
    "94f916fbf0ef4362cabd0fbef5430fa2bb94517b7176676386243f6253f47f3f"
)
EXPECTED_LOCAL_MANIFEST_SHA256 = (
    "4777f2d12065f7fdf6d7ae82ae4e9fc83cdf554a5444742ee6f1f81bde3e3336"
)
EXPECTED_REMOTE_MANIFEST_SHA256 = (
    "ef977c3d49ef23e726518ae6b24c6e5672125afa8c6d7f6b7db08b0858adeb57"
)
EXPECTED_RUNNER_SHA256 = (
    "a0f08e5090d27d880d53976d4207324b3a2d26195b418e6a4e5382e11cdc3fc4"
)
EXPECTED_WATCHER_SHA256 = (
    "ffd80acb98ddbb47408cb963b91d1bf3e5099eb18200abcecc1a9441b7069ed0"
)
EXPECTED_PREFIX_SHA256 = (
    "1d193b1b0aa65054bcbcde1a77e4a9dd11a9d911f38ac06afd7da0890efe6ac9"
)
EXPECTED_ATTESTATION_SHA256 = (
    "5612f8df5f53d511e512969d4e332a88dfcdd331e50bf2ba498772c030a778e4"
)
EXPECTED_PYTHON_FLINT = "0.9.0"
EXPECTED_FLINT = "3.6.0"

INPUT_NAMES = {
    "prefix_log": "local_prefix.log",
    "attestation": "original_process_attestation.json",
    "cutover_status": "cutover_status.log",
    "local_manifest": "manifest_local_under235.json",
    "local_log": "block3a_local_fill.log",
    "local_provenance": "block3a_local_fill_provenance.json",
    "remote_manifest": "manifest_remote_tail.json",
    "remote_log": "block3a_azure_tail.log",
    "remote_provenance": "block3a_azure_tail_provenance.json",
    "runner_source": "manifest_runner.py",
    "watcher_source": "watch_local_cutover.py",
}

_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\Z")
_LINE_RE = re.compile(
    r"^(PASS|FAIL) (PG|QG) \[([^,]+),([^\]]+)\] "
    r"lam=n=([1-9][0-9]*) val=(.+) \(([0-9]+(?:\.[0-9]+)?)s\)$"
)
_ARMED_RE = re.compile(r"armed pid=([0-9]+) threshold=([0-9]+)")
_TRIGGER_RE = re.compile(r"cutover triggered at ([0-9]+) certified cells")
_SUSPEND_RE = re.compile(r"suspended parent and ([0-9]+) descendants")
_SNAPSHOT_RE = re.compile(r"snapshot cells=([0-9]+) sha256=([0-9a-f]{64})")
_STATUS_LINE_RE = re.compile(
    r"([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}) (.+)\Z"
)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key}")
        out[key] = value
    return out


def load_evidence_json(path):
    """Load noncanonical staging JSON without accepting ambiguity."""
    raw = pathlib.Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"BOM is forbidden in {path}")
    return json.loads(
        raw.decode("utf-8", errors="strict"),
        parse_float=Decimal,
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )


def canonical_json_bytes(value) -> bytes:
    def walk(obj):
        if isinstance(obj, (float, Decimal)):
            raise TypeError("floats are forbidden in canonical certificates")
        if isinstance(obj, dict):
            for key, item in obj.items():
                if not isinstance(key, str):
                    raise TypeError("JSON object keys must be strings")
                walk(item)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)

    walk(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value, omit=("certificate_sha256",)) -> str:
    if isinstance(value, dict):
        omitted = set(omit)
        value = {key: item for key, item in value.items() if key not in omitted}
    return _sha_bytes(canonical_json_bytes(value))


def load_certificate(path):
    raw = pathlib.Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("certificate BOM is forbidden")
    value = json.loads(
        raw.decode("utf-8", errors="strict"),
        parse_float=_reject_constant,
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if raw != canonical_json_bytes(value) + b"\n":
        raise ValueError("certificate is not canonical JSON")
    return value


def _write_bytes_no_clobber(path, raw: bytes) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    linked = False
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp_name, path)
        linked = True
        os.unlink(tmp_name)
    except Exception:
        if linked:
            path.unlink(missing_ok=True)
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _copy_no_clobber(source, destination) -> None:
    _write_bytes_no_clobber(destination, pathlib.Path(source).read_bytes())


def _plain_int(value, name, minimum=None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be a plain integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} is below its minimum")
    return value


def _nonempty_text(value, name) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha_text(value, name) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} is not a lowercase SHA-256")
    return value


def _expect_keys(value, expected, name) -> None:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"invalid {name} schema")


def _parse_utc(value, name) -> dt.datetime:
    _nonempty_text(value, name)
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid {name}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} is not timezone-aware")
    return parsed.astimezone(dt.timezone.utc)


def _nonnegative_decimal(value, name) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise ValueError(f"{name} must be an exact JSON number")
    try:
        out = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if not out.is_finite() or out < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return out


def _proof_source_paths() -> dict[str, pathlib.Path]:
    return {
        "block3a_grid.py": pathlib.Path(block3a_grid.__file__).resolve(),
        "core.py": pathlib.Path(core.__file__).resolve(),
        "dsfun.py": pathlib.Path(dsfun.__file__).resolve(),
    }


def current_source_hashes(include_assembler=False) -> dict[str, str]:
    paths = _proof_source_paths()
    if include_assembler:
        paths["block3a_assemble.py"] = pathlib.Path(__file__).resolve()
    return {name: file_sha256(path) for name, path in sorted(paths.items())}


def _validate_trust_environment() -> None:
    if current_source_hashes() != EXPECTED_PROOF_SOURCE_SHA256:
        raise ValueError("Block3a proof source hash mismatch")
    if flint.__version__ != EXPECTED_PYTHON_FLINT:
        raise ValueError("python-flint runtime mismatch")
    if flint.__FLINT_VERSION__ != EXPECTED_FLINT:
        raise ValueError("FLINT runtime mismatch")


def canonical_schedule() -> list[dict]:
    pos, neg = block3a_grid.build_grids()
    if len(pos) != PG_COUNT or len(neg) != QG_COUNT:
        raise ValueError("Block3a branch schedule count mismatch")
    jobs = ([('PG', lo, hi) for lo, hi in pos]
            + [('QG', lo, hi) for lo, hi in neg])
    if len(jobs) != SCHEDULE_COUNT:
        raise ValueError("Block3a total schedule count mismatch")
    schedule = [
        {"index": index, "kind": kind, "tau_lo": lo, "tau_hi": hi}
        for index, (kind, lo, hi) in enumerate(jobs)
    ]
    if payload_sha256(schedule, omit=()) != EXPECTED_SCHEDULE_SHA256:
        raise ValueError("Block3a canonical schedule hash mismatch")
    return schedule


def _artifact(path, root) -> dict:
    path = pathlib.Path(path).resolve()
    root = pathlib.Path(root).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact escapes certificate root: {path}") from exc
    return {
        "file": relative.as_posix(),
        "file_sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _resolve_artifact(root, relative) -> pathlib.Path:
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or pathlib.PurePosixPath(relative).is_absolute()):
        raise ValueError("invalid relative artifact path")
    parts = pathlib.PurePosixPath(relative).parts
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("artifact path traversal is forbidden")
    root = pathlib.Path(root).resolve()
    path = root.joinpath(*parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact path escapes certificate root") from exc
    return path


def _complete_text_lines(raw: bytes, label: str) -> list[str]:
    """Accept one consistent newline convention, never mixed/partial text."""
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{label} is empty or lacks its final newline")
    crlf_count = raw.count(b"\r\n")
    lf_count = raw.count(b"\n")
    if b"\r" in raw:
        if crlf_count != lf_count or raw.replace(b"\r\n", b"").find(b"\r") >= 0:
            raise ValueError(f"{label} has mixed or lone-CR newlines")
        body = raw[:-2]
        parts = body.split(b"\r\n")
    else:
        body = raw[:-1]
        parts = body.split(b"\n")
    if any(not part for part in parts):
        raise ValueError(f"{label} contains a blank line")
    return [part.decode("utf-8", errors="strict") for part in parts]


def parse_log_bytes(raw: bytes, schedule, label="Block3a log") -> dict[int, dict]:
    lines = _complete_text_lines(raw, label)
    by_key = {
        (row["kind"], row["tau_lo"], row["tau_hi"]): row["index"]
        for row in schedule
    }
    records = {}
    for lineno, line in enumerate(lines, 1):
        match = _LINE_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed {label} line {lineno}")
        verdict, kind, lo, hi, ncalls_text, value_text, elapsed_text = match.groups()
        if verdict != "PASS":
            raise ValueError(f"failed cell in {label} line {lineno}")
        key = (kind, lo, hi)
        if key not in by_key:
            raise ValueError(f"unexpected cell in {label} line {lineno}")
        index = by_key[key]
        if index in records:
            raise ValueError(f"duplicate cell {index} in {label}")
        ncalls = int(ncalls_text)
        if ncalls > 2 ** (MAX_DEPTH + 1) - 1 or ncalls % 2 != 1:
            raise ValueError(f"invalid recursive call count in {label} line {lineno}")
        leaves = value_text.split(" | ")
        if len(leaves) != (ncalls + 1) // 2:
            raise ValueError(f"value/call-count mismatch in {label} line {lineno}")
        if any(not leaf.startswith("[") or not leaf.endswith("]")
               or any(ord(char) < 32 or ord(char) == 127 for char in leaf)
               for leaf in leaves):
            raise ValueError(f"malformed Arb leaf text in {label} line {lineno}")
        try:
            elapsed = Decimal(elapsed_text)
        except InvalidOperation as exc:
            raise ValueError(f"invalid elapsed time in {label} line {lineno}") from exc
        if not elapsed.is_finite() or elapsed < 0:
            raise ValueError(f"invalid elapsed time in {label} line {lineno}")
        records[index] = {
            "index": index,
            "kind": kind,
            "tau_lo": lo,
            "tau_hi": hi,
            "line": line,
        }
    return records


def parse_log(path, schedule, label="Block3a log") -> dict[int, dict]:
    return parse_log_bytes(pathlib.Path(path).read_bytes(), schedule, label)


def validate_manifest(path, schedule, expected_indices, expected_sha, label):
    if file_sha256(path) != expected_sha:
        raise ValueError(f"{label} manifest raw hash mismatch")
    data = load_evidence_json(path)
    _expect_keys(
        data,
        {"created_utc", "description", "entries", "schedule_total", "schema",
         "source_sha256"},
        f"{label} manifest",
    )
    if data["schema"] != 1 or data["schedule_total"] != SCHEDULE_COUNT:
        raise ValueError(f"{label} manifest policy mismatch")
    _parse_utc(data["created_utc"], f"{label} manifest created_utc")
    _nonempty_text(data["description"], f"{label} manifest description")
    if data["source_sha256"] != EXPECTED_PROOF_SOURCE_SHA256:
        raise ValueError(f"{label} manifest source hash mismatch")
    expected_entries = [schedule[index] for index in expected_indices]
    if data["entries"] != expected_entries:
        raise ValueError(f"{label} manifest schedule mismatch")
    return data


def _validate_status(path, prefix_sha, prefix_count, process, workers):
    raw = pathlib.Path(path).read_bytes()
    lines = _complete_text_lines(raw, "cutover status")
    events = []
    for index, line in enumerate(lines):
        outer = _STATUS_LINE_RE.fullmatch(line)
        if outer is None:
            raise ValueError("malformed cutover status line")
        timestamp = dt.datetime.strptime(outer.group(1), "%Y-%m-%d %H:%M:%S")
        events.append((index, timestamp, outer.group(2)))

    def matches(regex):
        out = []
        for index, timestamp, message in events:
            match = regex.fullmatch(message)
            if match is not None:
                out.append((index, timestamp, match.groups()))
        return out

    armed = matches(_ARMED_RE)
    triggered = matches(_TRIGGER_RE)
    suspended = matches(_SUSPEND_RE)
    snapshots = matches(_SNAPSHOT_RE)
    if not armed or len(triggered) != 1 or len(suspended) != 1 or len(snapshots) != 1:
        raise ValueError("cutover status lacks required events")
    pid = process["pid"]
    eligible_armed = [event for event in armed
                      if int(event[2][0]) == pid
                      and int(event[2][1]) == prefix_count]
    if not eligible_armed:
        raise ValueError("cutover status PID/threshold mismatch")
    armed_event = eligible_armed[-1]
    trigger_event, suspend_event, snapshot_event = (
        triggered[0], suspended[0], snapshots[0]
    )
    ordered = [armed_event, trigger_event, suspend_event, snapshot_event]
    if ([event[0] for event in ordered] != sorted(event[0] for event in ordered)
            or [event[1] for event in ordered]
            != sorted(event[1] for event in ordered)):
        raise ValueError("cutover status events are out of order")
    if int(trigger_event[2][0]) != prefix_count:
        raise ValueError("cutover trigger count mismatch")
    if int(suspend_event[2][0]) < workers:
        raise ValueError("cutover did not suspend the worker tree")
    count, digest = snapshot_event[2]
    if int(count) != prefix_count or digest != prefix_sha:
        raise ValueError("cutover snapshot record mismatch")


def validate_prefix_chain(prefix_path, attestation_path, status_path,
                          watcher_path, schedule):
    prefix_raw = pathlib.Path(prefix_path).read_bytes()
    prefix_sha = _sha_bytes(prefix_raw)
    if prefix_sha != EXPECTED_PREFIX_SHA256:
        raise ValueError("frozen local prefix hash mismatch")
    records = parse_log_bytes(prefix_raw, schedule, "local prefix")
    if set(records) != set(range(PREFIX_COUNT)):
        raise ValueError("local prefix is not exactly schedule indices 0..186")
    if file_sha256(attestation_path) != EXPECTED_ATTESTATION_SHA256:
        raise ValueError("original-process attestation raw hash mismatch")
    data = load_evidence_json(attestation_path)
    _expect_keys(
        data,
        {"capture", "kind", "live_log_observation", "process", "runtime",
         "schema_version", "source_files"},
        "original-process attestation",
    )
    if (data["schema_version"] != 1
            or data["kind"] != "block3a_original_process_attestation"):
        raise ValueError("original-process attestation identity mismatch")

    capture = data["capture"]
    _expect_keys(capture, {"captured_at_utc", "method", "process_alive",
                           "worker_children"}, "attestation capture")
    if capture["process_alive"] is not True:
        raise ValueError("attested original process was not alive")
    workers = _plain_int(capture["worker_children"], "worker_children", 1)
    if workers != 24:
        raise ValueError("original process worker count mismatch")

    process = data["process"]
    _expect_keys(
        process,
        {"affinity_mask", "command_line", "creation_unix_milliseconds",
         "executable", "parent_pid_at_capture", "pid", "priority_class",
         "reported_launch_workdir"},
        "attested process",
    )
    pid = _plain_int(process["pid"], "attested PID", 1)
    _plain_int(process["parent_pid_at_capture"], "attested parent PID", 1)
    creation_ms = _plain_int(
        process["creation_unix_milliseconds"], "process creation time", 1
    )
    executable = pathlib.Path(_nonempty_text(process["executable"], "executable"))
    if executable.resolve() != pathlib.Path(sys.executable).resolve():
        raise ValueError("attested executable is not the verifier executable")
    command = _nonempty_text(process["command_line"], "command line").strip()
    expected_command = f'"{process["executable"]}" verification/block3a_grid.py 24'
    if command != expected_command:
        raise ValueError("attested Block3a command line mismatch")
    if pathlib.Path(process["reported_launch_workdir"]).resolve() != PROOF_ROOT:
        raise ValueError("attested Block3a working directory mismatch")
    if process["priority_class"] != "BelowNormal" or process["affinity_mask"] != "0xFF":
        raise ValueError("attested local governance policy mismatch")

    runtime = data["runtime"]
    _expect_keys(runtime, {"executable_sha256", "flint", "implementation",
                           "python", "python_flint"}, "attested runtime")
    if (runtime["implementation"] != platform.python_implementation()
            or runtime["python"] != platform.python_version()
            or runtime["python_flint"] != EXPECTED_PYTHON_FLINT
            or runtime["flint"] != EXPECTED_FLINT):
        raise ValueError("attested local runtime mismatch")
    if (file_sha256(executable) != _sha_text(
            runtime["executable_sha256"], "attested executable hash")):
        raise ValueError("attested executable hash mismatch")

    created = dt.datetime.fromtimestamp(creation_ms / 1000, tz=dt.timezone.utc)
    captured = _parse_utc(capture["captured_at_utc"], "captured_at_utc")
    if captured <= created:
        raise ValueError("attestation capture predates process creation")
    _nonempty_text(capture["method"], "attestation method")

    source_files = data["source_files"]
    if not isinstance(source_files, dict) or set(source_files) != set(
            EXPECTED_PROOF_SOURCE_SHA256):
        raise ValueError("attested source-file set mismatch")
    for name, source_path in _proof_source_paths().items():
        row = source_files[name]
        _expect_keys(row, {"length", "mtime_utc", "sha256"},
                     f"attested source {name}")
        if (_plain_int(row["length"], f"{name} length", 1) != source_path.stat().st_size
                or _sha_text(row["sha256"], f"{name} hash")
                != EXPECTED_PROOF_SOURCE_SHA256[name]):
            raise ValueError(f"attested source identity mismatch for {name}")
        if _parse_utc(row["mtime_utc"], f"{name} mtime") >= created:
            raise ValueError(f"attested source {name} was not frozen before launch")

    observation = data["live_log_observation"]
    _expect_keys(observation, {"fail_lines", "length", "pass_lines", "path",
                               "sha256"}, "live-log observation")
    if (_plain_int(observation["fail_lines"], "observed fails", 0) != 0
            or _plain_int(observation["pass_lines"], "observed passes", 1)
            != ATTESTED_PREFIX_COUNT):
        raise ValueError("attested live-log row counts mismatch")
    observed_length = _plain_int(observation["length"], "observed length", 1)
    if observed_length > len(prefix_raw) or prefix_raw[observed_length - 1:observed_length] != b"\n":
        raise ValueError("attested live-log byte boundary is invalid")
    observed_sha = _sha_text(observation["sha256"], "observed log hash")
    if _sha_bytes(prefix_raw[:observed_length]) != observed_sha:
        raise ValueError("attested live log is not a byte prefix of the snapshot")
    observed = parse_log_bytes(
        prefix_raw[:observed_length], schedule, "attested live-log prefix"
    )
    if set(observed) != set(range(ATTESTED_PREFIX_COUNT)):
        raise ValueError("attested live log is not indices 0..162")
    if pathlib.Path(observation["path"]).name != "block3a.log":
        raise ValueError("unexpected attested live-log name")

    if file_sha256(watcher_path) != EXPECTED_WATCHER_SHA256:
        raise ValueError("cutover watcher source hash mismatch")
    _validate_status(status_path, prefix_sha, PREFIX_COUNT, process, workers)
    return records, {
        "pid": pid,
        "workers": workers,
        "observed_rows": ATTESTED_PREFIX_COUNT,
        "observed_bytes": observed_length,
        "snapshot_rows": PREFIX_COUNT,
    }, runtime


def validate_runner_provenance(path, manifest_path, output_path, log_records,
                               expected_indices, manifest_count, expected_workers,
                               seeded_rows, local_runtime=None):
    data = load_evidence_json(path)
    required = {
        "completed_this_run", "covered_manifest_cells", "elapsed_seconds",
        "fails_this_run", "flint", "host", "manifest", "manifest_cells",
        "manifest_sha256", "output", "platform", "preexisting_cells",
        "python", "python_flint", "runner_sha256", "schema", "source_sha256",
        "started_utc", "state", "updated_utc", "workers",
    }
    _expect_keys(data, required, "runner provenance")
    if data["schema"] != 1 or data["state"] != "complete":
        raise ValueError("runner provenance is not complete schema 1")
    if data["source_sha256"] != EXPECTED_PROOF_SOURCE_SHA256:
        raise ValueError("runner provenance source hash mismatch")
    if data["runner_sha256"] != EXPECTED_RUNNER_SHA256:
        raise ValueError("runner provenance source identity mismatch")
    if data["manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("runner provenance manifest hash mismatch")
    if pathlib.Path(data["manifest"]).name != pathlib.Path(manifest_path).name:
        raise ValueError("runner provenance manifest name mismatch")
    if (pathlib.Path(_nonempty_text(data["output"], "runner output path")).name
            != pathlib.Path(output_path).name):
        raise ValueError("runner provenance output name mismatch")
    if (_plain_int(data["manifest_cells"], "manifest_cells", 1) != manifest_count
            or _plain_int(data["covered_manifest_cells"],
                          "covered_manifest_cells", 0) != manifest_count):
        raise ValueError("runner provenance coverage count mismatch")
    preexisting = _plain_int(data["preexisting_cells"], "preexisting_cells", 0)
    completed = _plain_int(data["completed_this_run"], "completed_this_run", 0)
    if preexisting < seeded_rows or preexisting + completed != manifest_count:
        raise ValueError("runner provenance resume accounting mismatch")
    if _plain_int(data["fails_this_run"], "fails_this_run", 0) != 0:
        raise ValueError("runner provenance records failures")
    if _plain_int(data["workers"], "workers", 1) != expected_workers:
        raise ValueError("runner worker policy mismatch")
    _nonnegative_decimal(data["elapsed_seconds"], "elapsed_seconds")
    started = _parse_utc(data["started_utc"], "runner started_utc")
    updated = _parse_utc(data["updated_utc"], "runner updated_utc")
    if updated < started:
        raise ValueError("runner provenance timestamps are reversed")
    if (data["python_flint"] != EXPECTED_PYTHON_FLINT
            or data["flint"] != EXPECTED_FLINT):
        raise ValueError("runner arithmetic runtime mismatch")
    python_text = _nonempty_text(data["python"], "runner Python")
    if _VERSION_RE.fullmatch(python_text.split()[0]) is None:
        raise ValueError("runner Python version is malformed")
    if local_runtime is not None and python_text.split()[0] != local_runtime["python"]:
        raise ValueError("local fill Python differs from original-process runtime")
    _nonempty_text(data["host"], "runner host")
    _nonempty_text(data["platform"], "runner platform")
    if set(log_records) != set(expected_indices):
        raise ValueError("runner output does not match its owned indices")
    return {
        "host": data["host"],
        "platform": data["platform"],
        "python": python_text,
        "python_flint": data["python_flint"],
        "flint": data["flint"],
        "workers": data["workers"],
        "preexisting_cells": preexisting,
        "completed_this_run": completed,
    }


def _canonical_integer(text, nonnegative=False) -> int:
    regex = (r"(?:0|[1-9][0-9]*)\Z" if nonnegative
             else r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")
    if not isinstance(text, str) or re.fullmatch(regex, text) is None:
        raise ValueError("noncanonical packet integer")
    return int(text)


def arb_packet(value, digits=60) -> dict:
    if not isinstance(value, arb) or not value.is_finite():
        raise ValueError("boundary value is not a finite Arb ball")
    mid, rad, exp10 = value.mid_rad_10exp(digits)
    return {
        "format": "arb-midrad10-v1",
        "mid10": str(mid),
        "rad10": str(rad),
        "exp10": int(exp10),
        "digits": int(digits),
    }


def packet_fraction_endpoints(packet) -> tuple[Fraction, Fraction]:
    _expect_keys(packet, {"format", "mid10", "rad10", "exp10", "digits"},
                 "Arb packet")
    if packet["format"] != "arb-midrad10-v1":
        raise ValueError("unknown Arb packet format")
    mid = _canonical_integer(packet["mid10"])
    rad = _canonical_integer(packet["rad10"], nonnegative=True)
    exp10 = _plain_int(packet["exp10"], "packet exponent")
    _plain_int(packet["digits"], "packet digits", 1)
    scale = Fraction(10 ** exp10, 1) if exp10 >= 0 else Fraction(1, 10 ** (-exp10))
    return (mid - rad) * scale, (mid + rad) * scale


def compute_boundary_pins() -> dict:
    old_precision = flint.ctx.prec
    try:
        core.set_prec(BOUNDARY_PRECISION_BITS)
        differences = {
            "lambda_0.24_lt_0.2": (
                core.dec("0.2")
                - dsfun.ell_range(dsfun.A_of_tau(core.dec("0.24")))
            ),
            "lambda_0.99_gt_0.98": (
                dsfun.ell_range(dsfun.A_of_tau(core.dec("0.99")))
                - core.dec("0.98")
            ),
            "lambda_-0.18_gt_-0.125": (
                dsfun.ell_range(dsfun.A_of_tau(core.dec("-0.18")))
                - core.dec("-0.125")
            ),
        }
        out = {"precision_bits": BOUNDARY_PRECISION_BITS, "checks": {}}
        for name, value in differences.items():
            if not (value > 0):
                raise ValueError(f"Block3a endpoint pin failed: {name}")
            packet = arb_packet(value)
            lo, _ = packet_fraction_endpoints(packet)
            if lo <= 0:
                raise ValueError(f"serialized endpoint packet lost its sign: {name}")
            out["checks"][name] = {"sign": ">0", "difference": packet}
        return out
    finally:
        flint.ctx.prec = old_precision


def _runtime_record() -> dict:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "python_flint": flint.__version__,
        "flint": flint.__FLINT_VERSION__,
        "boundary_precision_bits": BOUNDARY_PRECISION_BITS,
    }


def _evaluate(root, inputs, installed_log_relative):
    _validate_trust_environment()
    if not isinstance(inputs, dict) or set(inputs) != set(INPUT_NAMES):
        raise ValueError("certificate input map mismatch")
    paths = {key: _resolve_artifact(root, value) for key, value in inputs.items()}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"missing Block3a evidence artifact: {path}")
    if file_sha256(paths["runner_source"]) != EXPECTED_RUNNER_SHA256:
        raise ValueError("manifest runner source hash mismatch")
    if file_sha256(paths["watcher_source"]) != EXPECTED_WATCHER_SHA256:
        raise ValueError("cutover watcher source hash mismatch")

    schedule = canonical_schedule()
    validate_manifest(
        paths["local_manifest"], schedule, list(range(LOCAL_MANIFEST_COUNT)),
        EXPECTED_LOCAL_MANIFEST_SHA256, "local",
    )
    validate_manifest(
        paths["remote_manifest"], schedule,
        list(range(SCHEDULE_COUNT - 1, LOCAL_MANIFEST_COUNT - 1, -1)),
        EXPECTED_REMOTE_MANIFEST_SHA256, "remote",
    )
    prefix, prefix_summary, prefix_runtime = validate_prefix_chain(
        paths["prefix_log"], paths["attestation"], paths["cutover_status"],
        paths["watcher_source"], schedule,
    )
    local = parse_log(paths["local_log"], schedule, "local fill")
    remote = parse_log(paths["remote_log"], schedule, "reverse tail")
    local_indices = list(range(PREFIX_COUNT, LOCAL_MANIFEST_COUNT))
    remote_indices = list(range(LOCAL_MANIFEST_COUNT, SCHEDULE_COUNT))
    if set(local) != set(local_indices):
        raise ValueError("local fill is not exactly indices 187..234")
    if set(remote) != set(remote_indices):
        raise ValueError("reverse tail is not exactly indices 235..246")
    if set(prefix) & set(local) or set(prefix) & set(remote) or set(local) & set(remote):
        raise ValueError("Block3a shard overlap")
    if set(prefix) | set(local) | set(remote) != set(range(SCHEDULE_COUNT)):
        raise ValueError("Block3a shard union is incomplete")

    local_runtime = validate_runner_provenance(
        paths["local_provenance"], paths["local_manifest"], paths["local_log"],
        local, local_indices, LOCAL_MANIFEST_COUNT, 24, PREFIX_COUNT,
        prefix_runtime,
    )
    remote_runtime = validate_runner_provenance(
        paths["remote_provenance"], paths["remote_manifest"], paths["remote_log"],
        remote, remote_indices, REMOTE_TAIL_COUNT, 4, 0,
    )

    merged = {**prefix, **local, **remote}
    final_raw = "".join(merged[index]["line"] + "\n"
                        for index in range(SCHEDULE_COUNT)).encode("utf-8")
    installed_relative = pathlib.PurePosixPath(installed_log_relative).as_posix()
    installed_path = _resolve_artifact(root, installed_relative)
    artifacts = {key: _artifact(path, root) for key, path in sorted(paths.items())}
    boundary_pins = compute_boundary_pins()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "block3a_certificate",
        "verdict": "ALL PASS",
        "policy": {
            "precision_bits": PROOF_PRECISION_BITS,
            "max_depth": MAX_DEPTH,
            "boundary_precision_bits": BOUNDARY_PRECISION_BITS,
            "evidence_model": "source-bound-trusted-execution",
        },
        "source_sha256": current_source_hashes(include_assembler=True),
        "runtime": _runtime_record(),
        "schedule": {
            "cells": SCHEDULE_COUNT,
            "pg_cells": PG_COUNT,
            "qg_cells": QG_COUNT,
            "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        },
        "inputs": dict(sorted(inputs.items())),
        "artifacts": artifacts,
        "prefix_chain": prefix_summary,
        "shards": [
            {
                "role": "local_prefix",
                "line_count": len(prefix),
                "indices": list(range(PREFIX_COUNT)),
                "log_sha256": artifacts["prefix_log"]["file_sha256"],
                "runtime": prefix_runtime,
            },
            {
                "role": "local_fill",
                "line_count": len(local),
                "indices": local_indices,
                "log_sha256": artifacts["local_log"]["file_sha256"],
                "runtime": local_runtime,
            },
            {
                "role": "reverse_tail",
                "line_count": len(remote),
                "indices": remote_indices,
                "log_sha256": artifacts["remote_log"]["file_sha256"],
                "runtime": remote_runtime,
            },
        ],
        "coverage": {
            "cells": SCHEDULE_COUNT,
            "failures": 0,
            "duplicates": 0,
            "indices_sha256": payload_sha256(list(range(SCHEDULE_COUNT)), omit=()),
        },
        "installed_log": {
            "file": installed_relative,
            "file_sha256": _sha_bytes(final_raw),
            "bytes": len(final_raw),
            "line_count": SCHEDULE_COUNT,
        },
        "boundary_pins": boundary_pins,
    }
    return payload, final_raw, installed_path


def assemble(*, prefix_log, attestation, cutover_status, local_manifest,
             local_log, local_provenance, remote_manifest, remote_log,
             remote_provenance, runner_source, watcher_source, evidence_dir,
             output_log, certificate):
    certificate = pathlib.Path(certificate).resolve()
    root = certificate.parent
    evidence_dir = pathlib.Path(evidence_dir).resolve()
    try:
        evidence_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("evidence directory must be inside certificate directory") from exc
    if evidence_dir == root:
        raise ValueError("evidence directory cannot equal certificate directory")
    output_log = pathlib.Path(output_log).resolve()
    try:
        output_relative = output_log.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("installed log must be inside certificate directory") from exc
    if certificate == output_log:
        raise ValueError("certificate and installed log destinations alias")
    for target, name in ((certificate, "certificate"), (output_log, "installed log")):
        try:
            target.relative_to(evidence_dir)
        except ValueError:
            pass
        else:
            raise ValueError(f"{name} cannot be inside the evidence directory")
    sources = {
        "prefix_log": prefix_log,
        "attestation": attestation,
        "cutover_status": cutover_status,
        "local_manifest": local_manifest,
        "local_log": local_log,
        "local_provenance": local_provenance,
        "remote_manifest": remote_manifest,
        "remote_log": remote_log,
        "remote_provenance": remote_provenance,
        "runner_source": runner_source,
        "watcher_source": watcher_source,
    }
    source_paths = {key: pathlib.Path(path).resolve()
                    for key, path in sources.items()}
    if any(not path.is_file() for path in source_paths.values()):
        raise ValueError("one or more Block3a input artifacts are missing")
    if len(set(source_paths.values())) != len(source_paths):
        raise ValueError("Block3a input artifacts alias each other")
    destination_names = dict(INPUT_NAMES)
    # Preserve runner-recorded log basenames.  The reverse tail may execute
    # on Azure or locally; its provenance host, not a renamed file, records
    # where the computation actually ran.
    for key in ("local_log", "remote_log"):
        destination_names[key] = source_paths[key].name
    if len(set(destination_names.values())) != len(destination_names):
        raise ValueError("Block3a evidence destination names collide")
    if certificate.exists() or output_log.exists() or evidence_dir.exists():
        raise FileExistsError("a Block3a publication destination already exists")

    root.mkdir(parents=True, exist_ok=True)
    evidence_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(tempfile.mkdtemp(
        prefix=evidence_dir.name + ".", suffix=".tmp",
        dir=str(evidence_dir.parent),
    )).resolve()
    published_evidence = False
    created_output = False
    created_certificate = False
    try:
        temporary_inputs = {}
        for key, source in source_paths.items():
            destination = temporary / destination_names[key]
            _copy_no_clobber(source, destination)
            temporary_inputs[key] = destination.relative_to(root).as_posix()

        # Validate the complete copied byte set before publishing any durable
        # evidence path.  The payload is rebuilt after the atomic directory
        # rename so it never records temporary names.
        _evaluate(root, temporary_inputs, output_relative)
        os.replace(temporary, evidence_dir)
        published_evidence = True
        inputs = {
            key: (evidence_dir / destination_names[key]).relative_to(root).as_posix()
            for key in source_paths
        }
        payload, final_raw, _ = _evaluate(root, inputs, output_relative)
        _write_bytes_no_clobber(output_log, final_raw)
        created_output = True
        payload["certificate_sha256"] = payload_sha256(payload)
        _write_bytes_no_clobber(
            certificate, canonical_json_bytes(payload) + b"\n"
        )
        created_certificate = True
        verify_certificate(certificate)
        return payload
    except Exception:
        if created_certificate:
            certificate.unlink(missing_ok=True)
        if created_output:
            output_log.unlink(missing_ok=True)
        if published_evidence and evidence_dir.is_dir():
            shutil.rmtree(evidence_dir)
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise


def verify_certificate(path):
    path = pathlib.Path(path).resolve()
    data = load_certificate(path)
    if (data.get("schema_version") != SCHEMA_VERSION
            or data.get("kind") != "block3a_certificate"
            or data.get("verdict") != "ALL PASS"):
        raise ValueError("invalid Block3a certificate schema/verdict")
    if data.get("certificate_sha256") != payload_sha256(data):
        raise ValueError("Block3a certificate payload hash mismatch")
    expected, final_raw, installed_path = _evaluate(
        path.parent, data.get("inputs"), data.get("installed_log", {}).get("file")
    )
    if not installed_path.is_file() or installed_path.read_bytes() != final_raw:
        raise ValueError("installed Block3a log differs from raw evidence")
    without_hash = {key: value for key, value in data.items()
                    if key != "certificate_sha256"}
    if without_hash != expected:
        raise ValueError("Block3a certificate/raw evidence mismatch")
    return data


def _build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("assemble")
    for option in (
        "prefix-log", "attestation", "cutover-status", "local-manifest",
        "local-log", "local-provenance", "remote-manifest", "remote-log",
        "remote-provenance", "runner-source", "watcher-source",
        "evidence-dir", "output-log", "certificate",
    ):
        make.add_argument("--" + option, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--certificate", required=True)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.command == "verify":
        data = verify_certificate(args.certificate)
        print(f"Block3a certificate PASS {data['certificate_sha256']}")
        return 0
    data = assemble(
        prefix_log=args.prefix_log,
        attestation=args.attestation,
        cutover_status=args.cutover_status,
        local_manifest=args.local_manifest,
        local_log=args.local_log,
        local_provenance=args.local_provenance,
        remote_manifest=args.remote_manifest,
        remote_log=args.remote_log,
        remote_provenance=args.remote_provenance,
        runner_source=args.runner_source,
        watcher_source=args.watcher_source,
        evidence_dir=args.evidence_dir,
        output_log=args.output_log,
        certificate=args.certificate,
    )
    print(f"Block3a certificate assembled {data['certificate_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
