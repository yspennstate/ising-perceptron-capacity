#!/usr/bin/env python3
"""Create or check a source-bound final-verification receipt.

``run`` belongs on a clean checkout under the attested arithmetic runtime. It
runs every command from a read-only commit archive. The unit suite alone gets
a source-commit Git view so production provenance tests still work; the
manuscript-inventory and theorem-level ``verify_all.py`` commands remain
Git-free. ``check`` is portable and lets the Windows packager bind the receipt
to an immutable commit archive without repeating hours of proof computation on
the owner's workstation.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
import time
import zipfile


SCHEMA_VERSION = 2
KIND = "ising_perceptron_final_verification_receipt"
VERDICT = "ALL_CERTIFICATES_PASS"
ATTESTATION_SCOPE = "operator-controlled-run-mixup-detection-v1"
EXPECTED_EXECUTABLE_SHA256 = (
    "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118")
EXPECTED_FLINT_TREE_SHA256 = (
    "80779bae258d9d4ab5ad12ab18aacf6670aef98287fb23bd7af8b94c9af61051")
MAX_RECEIPT_BYTES = 32 * 1024 * 1024
MAX_COMMAND_OUTPUT_BYTES = 8 * 1024 * 1024
MINIMUM_UNIT_TESTS = 140
MAX_WORKERS = 4
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
HERE = pathlib.Path(__file__).resolve().parent
PROOF_ROOT = HERE.parent
REPO_ROOT = PROOF_ROOT.parent
IMMUTABLE_ARCHIVE_SCOPE = "immutable_commit_archive"
IMMUTABLE_ARCHIVE_GIT_SCOPE = "immutable_archive_with_source_git_view"
COMMANDS = (
    ("unit_tests",
     ("-B", "-m", "unittest", "discover", "-s", "tests", "-p",
      "test_*.py"),
     "OK", IMMUTABLE_ARCHIVE_GIT_SCOPE),
    ("paper_inventory", ("-B", "update_paper_counts.py", "--check"),
     "PASS paper inventory matches verified certificates",
     IMMUTABLE_ARCHIVE_SCOPE),
    ("verify_all", ("-B", "verify_all.py"), "ALL CERTIFICATES PASS",
     IMMUTABLE_ARCHIVE_SCOPE),
)


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("ascii")


def _payload_sha256(value) -> str:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _unit_inventory_ok(output: str) -> bool:
    matches = re.findall(
        r"^Ran ([0-9]+) tests? in [^\r\n]+$", output, re.MULTILINE)
    return (len(matches) == 1
            and int(matches[0]) >= MINIMUM_UNIT_TESTS)


def _load(path: pathlib.Path):
    with path.open("rb") as handle:
        raw = handle.read(MAX_RECEIPT_BYTES + 1)
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ValueError("final receipt exceeds the size limit")
    return json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=_strict_object)


def _unsafe_link(path: pathlib.Path) -> bool:
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _regular_file_map(root: pathlib.Path, *, label: str):
    if not root.is_dir() or _unsafe_link(root):
        raise ValueError(f"{label} directory is missing or unsafe")
    out = {}
    casefolded = set()
    for path in sorted(root.rglob("*")):
        if _unsafe_link(path):
            raise ValueError(f"{label} symlink/reparse point is forbidden: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        folded = relative.casefold()
        if relative in out or folded in casefolded:
            raise ValueError(f"duplicate/case-colliding {label} path: {relative}")
        casefolded.add(folded)
        out[relative] = {
            "sha256": _file_sha256(path),
            "size": path.stat().st_size,
        }
    if not out:
        raise ValueError(f"receipt cannot bind an empty {label} closure")
    return out


def _tree_identity(proof_root: pathlib.Path):
    files = _regular_file_map(proof_root, label="proof tree")
    return {
        "algorithm": "sha256(canonical(path,size,sha256)-map)-v1",
        "sha256": hashlib.sha256(_canonical_bytes(files)).hexdigest(),
        "file_count": len(files),
        "byte_count": sum(item["size"] for item in files.values()),
    }


def _result_hashes(proof_root: pathlib.Path):
    results = proof_root / "verification" / "results"
    files = _regular_file_map(results, label="result")
    return {f"verification/results/{name}": value for name, value in files.items()}


def _git(*args):
    env = os.environ.copy()
    for key in (
            "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
            "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        env.pop(key, None)
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    return subprocess.run(
        ["git", "--no-replace-objects", *args], cwd=REPO_ROOT,
        env=env, capture_output=True, encoding="utf-8", errors="strict",
        check=False)


def _require_clean_commit(source_commit: str):
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a full lowercase SHA-1")
    head = _git("rev-parse", "HEAD")
    if head.returncode or head.stdout.strip() != source_commit:
        raise ValueError("checkout HEAD does not equal the requested commit")
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status.returncode or status.stdout:
        raise ValueError("exact-verification checkout is not clean")


@contextlib.contextmanager
def _immutable_commit_archive(source_commit: str):
    """Yield a read-only extraction of exactly ``source_commit:perceptron``."""
    with tempfile.TemporaryDirectory(prefix="ising-perceptron-final-") as td:
        temporary = pathlib.Path(td)
        archive = temporary / "source.zip"
        extraction = temporary / "tree"
        extraction.mkdir()
        listing = _git("ls-tree", "-r", source_commit, "--", "perceptron")
        if listing.returncode or any(line.startswith("160000 ")
                                     for line in listing.stdout.splitlines()):
            raise ValueError("proof tree contains an unsupported gitlink")
        command = _git(
            "archive", "--format=zip", f"--output={archive}",
            source_commit, "perceptron")
        if command.returncode or not archive.is_file():
            raise ValueError("could not create exact final-verification archive")
        seen = set()
        with zipfile.ZipFile(archive) as zipped:
            for info in zipped.infolist():
                name = info.filename
                pure = pathlib.PurePosixPath(name)
                folded = name.casefold()
                mode = (info.external_attr >> 16) & 0xFFFF
                if (not name.startswith("perceptron/")
                        or pure.is_absolute() or ".." in pure.parts
                        or "\\" in name or ":" in name
                        or folded in seen or stat.S_ISLNK(mode)):
                    raise ValueError(f"unsafe final-verification archive path: {name}")
                seen.add(folded)
            zipped.extractall(extraction)
        proof_root = extraction / "perceptron"
        if not proof_root.is_dir() or proof_root.is_symlink():
            raise ValueError("exact archive is missing the proof root")
        paths = sorted(proof_root.rglob("*"), key=lambda p: len(p.parts),
                       reverse=True)
        files = [path for path in paths if path.is_file()]
        directories = [path for path in paths if path.is_dir()]
        for path in files:
            path.chmod(0o444)
        for path in directories:
            path.chmod(0o555)
        proof_root.chmod(0o555)
        try:
            yield proof_root
        finally:
            proof_root.chmod(0o755)
            for path in reversed(directories):
                path.chmod(0o755)
            for path in files:
                path.chmod(0o644)


def _command_environment(base_env, execution_scope, proof_root, git_dir):
    env = dict(base_env)
    if execution_scope == IMMUTABLE_ARCHIVE_SCOPE:
        return env
    if execution_scope != IMMUTABLE_ARCHIVE_GIT_SCOPE:
        raise ValueError(
            f"unsupported final-verification execution scope: {execution_scope}")
    if not pathlib.Path(git_dir).is_dir():
        raise ValueError("source checkout Git directory is missing")
    env.update({
        "GIT_DIR": str(pathlib.Path(git_dir).resolve()),
        "GIT_WORK_TREE": str(pathlib.Path(proof_root).resolve().parent),
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    })
    return env


def _require_archive_git_view(source_commit, proof_root, env):
    work_tree = pathlib.Path(proof_root).resolve().parent

    def capture(*args):
        result = subprocess.run(
            ["git", "--no-replace-objects", *args], cwd=work_tree, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, encoding="utf-8", errors="strict",
            check=False)
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ValueError(f"archive Git view failed: {detail}")
        return result.stdout.strip()

    if capture("rev-parse", "HEAD") != source_commit:
        raise ValueError("archive Git view HEAD does not equal source commit")
    observed_work_tree = pathlib.Path(
        capture("rev-parse", "--show-toplevel")).resolve()
    if observed_work_tree != work_tree:
        raise ValueError("archive Git view work tree mismatch")
    capture("diff", "--quiet", "--no-ext-diff", source_commit, "--",
            "perceptron")
    capture("diff", "--cached", "--quiet", "--no-ext-diff",
            source_commit, "--", "perceptron")
    if capture("ls-files", "--others", "--exclude-standard", "--",
               "perceptron"):
        raise ValueError("archive Git view contains untracked proof files")


def _run_command(name, arguments, marker, execution_scope, git_head, env,
                 execution_root):
    argv = [sys.executable, *arguments]
    started = time.monotonic_ns()
    process = subprocess.Popen(
        argv, cwd=execution_root, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", bufsize=1)
    lines = []
    output_bytes = 0
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
        output_bytes += len(line.encode("utf-8"))
        if output_bytes > MAX_COMMAND_OUTPUT_BYTES:
            process.kill()
            process.wait()
            raise RuntimeError(f"verification output limit exceeded: {name}")
    returncode = process.wait()
    output = "".join(lines)
    elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
    record = {
        "name": name,
        "argv": argv,
        "returncode": returncode,
        "elapsed_milliseconds": elapsed_ms,
        "output": output,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "required_marker": marker,
        "execution_scope": execution_scope,
        "git_head": git_head,
    }
    final_line = next((line.strip() for line in reversed(lines)
                       if line.strip()), "")
    if (returncode or final_line != marker
            or (name == "unit_tests" and not _unit_inventory_ok(output))):
        raise RuntimeError(f"final verification command failed: {name}")
    return record


def _parse_timestamp(value):
    if not isinstance(value, str):
        raise ValueError("receipt timestamp is not a string")
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("receipt timestamp is not timezone-aware")
    return parsed


def check_receipt(path, source_commit, proof_root):
    path = pathlib.Path(path).resolve()
    proof_root = pathlib.Path(proof_root).resolve()
    if not isinstance(source_commit, str) or COMMIT_RE.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a full lowercase SHA-1")
    data = _load(path)
    expected_keys = {
        "schema_version", "kind", "source_commit", "started_at",
        "finished_at", "runtime", "commands", "proof_tree",
        "result_sha256", "verdict", "attestation_scope", "receipt_sha256",
    }
    if not isinstance(data, dict) or set(data) != expected_keys:
        raise ValueError("final receipt schema mismatch")
    if (not _plain_int(data["schema_version"])
            or data["schema_version"] != SCHEMA_VERSION
            or data["kind"] != KIND
            or data["verdict"] != VERDICT
            or data["attestation_scope"] != ATTESTATION_SCOPE
            or data["source_commit"] != source_commit
            or data["receipt_sha256"] != _payload_sha256(data)):
        raise ValueError("final receipt envelope mismatch")
    if _parse_timestamp(data["finished_at"]) < _parse_timestamp(data["started_at"]):
        raise ValueError("final receipt timestamps are reversed")

    runtime = data["runtime"]
    required_runtime = {
        "runtime_schema_version", "host", "python", "executable",
        "python_executable_sha256", "python_flint_root",
        "python_flint_tree_sha256", "python_flint", "flint",
        "precision_bits", "workers",
    }
    if not isinstance(runtime, dict) or set(runtime) != required_runtime:
        raise ValueError("final receipt runtime schema mismatch")
    if (not _plain_int(runtime["runtime_schema_version"])
            or runtime["runtime_schema_version"] != 2
            or runtime["python"] != "3.12.3"
            or runtime["python_flint"] != "0.9.0"
            or runtime["flint"] != "3.6.0"
            or not _plain_int(runtime["precision_bits"])
            or runtime["precision_bits"] != 50
            or runtime["python_executable_sha256"]
            != EXPECTED_EXECUTABLE_SHA256
            or runtime["python_flint_tree_sha256"]
            != EXPECTED_FLINT_TREE_SHA256
            or not isinstance(runtime["workers"], int)
            or isinstance(runtime["workers"], bool)
            or runtime["workers"] <= 0
            or runtime["workers"] > MAX_WORKERS):
        raise ValueError("final receipt runtime policy mismatch")
    for key in ("host", "executable", "python_flint_root"):
        if not isinstance(runtime[key], str) or not runtime[key]:
            raise ValueError(f"invalid final receipt runtime {key}")
    for key in ("python_executable_sha256", "python_flint_tree_sha256"):
        if not isinstance(runtime[key], str) or SHA_RE.fullmatch(runtime[key]) is None:
            raise ValueError(f"invalid final receipt runtime {key}")

    records = data["commands"]
    if not isinstance(records, list) or len(records) != len(COMMANDS):
        raise ValueError("final receipt command list mismatch")
    for record, (name, arguments, marker, execution_scope) in zip(
            records, COMMANDS):
        if not isinstance(record, dict) or set(record) != {
                "name", "argv", "returncode", "elapsed_milliseconds",
                "output", "output_sha256", "required_marker",
                "execution_scope", "git_head"}:
            raise ValueError("final receipt command schema mismatch")
        output = record["output"]
        final_line = (next((line.strip() for line in reversed(output.splitlines())
                            if line.strip()), "")
                      if isinstance(output, str) else "")
        if (record["name"] != name
                or not isinstance(record["argv"], list)
                or not record["argv"]
                or record["argv"][0] != runtime["executable"]
                or record["argv"][1:] != list(arguments)
                or not _plain_int(record["returncode"])
                or record["returncode"] != 0
                or record["required_marker"] != marker
                or record["execution_scope"] != execution_scope
                or record["git_head"] != (
                    source_commit
                    if execution_scope == IMMUTABLE_ARCHIVE_GIT_SCOPE
                    else None)
                or not isinstance(record["elapsed_milliseconds"], int)
                or isinstance(record["elapsed_milliseconds"], bool)
                or record["elapsed_milliseconds"] < 0
                or not isinstance(output, str)
                or len(output.encode("utf-8")) > MAX_COMMAND_OUTPUT_BYTES
                or final_line != marker
                or (name == "unit_tests" and not _unit_inventory_ok(output))
                or record["output_sha256"]
                != hashlib.sha256(output.encode("utf-8")).hexdigest()):
            raise ValueError(f"invalid final receipt command: {name}")

    observed_tree = data["proof_tree"]
    expected_tree = _tree_identity(proof_root)
    if (not isinstance(observed_tree, dict)
            or set(observed_tree) != {
                "algorithm", "sha256", "file_count", "byte_count"}
            or not _plain_int(observed_tree["file_count"])
            or observed_tree["file_count"] <= 0
            or not _plain_int(observed_tree["byte_count"])
            or observed_tree["byte_count"] < 0
            or observed_tree != expected_tree):
        raise ValueError("final receipt does not bind the exact proof tree")
    observed_results = data["result_sha256"]
    expected_results = _result_hashes(proof_root)
    if (not isinstance(observed_results, dict)
            or observed_results != expected_results
            or any(not isinstance(value, dict)
                   or set(value) != {"sha256", "size"}
                   or not isinstance(value["sha256"], str)
                   or SHA_RE.fullmatch(value["sha256"]) is None
                   or not isinstance(value["size"], int)
                   or isinstance(value["size"], bool) or value["size"] < 0
                   for value in observed_results.values())):
        raise ValueError("final receipt does not bind the exact result closure")
    return data


def run_receipt(source_commit, receipt, workers):
    receipt = pathlib.Path(receipt).resolve()
    if receipt.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {receipt}")
    if receipt == REPO_ROOT or REPO_ROOT in receipt.parents:
        raise ValueError("final receipt must be written outside the checkout")
    if not receipt.parent.is_dir():
        raise ValueError("final receipt parent does not exist")
    _require_clean_commit(source_commit)

    with _immutable_commit_archive(source_commit) as proof_root:
        verification_root = proof_root / "verification"
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(verification_root))
        try:
            import block3bc_exact as exact
        finally:
            sys.path.pop(0)
        runtime = exact.runtime_record(50, workers, fresh_flint=True)
        if (sys.version_info[:3] != (3, 12, 3)
                or exact.flint.__version__ != "0.9.0"
                or exact.flint.__FLINT_VERSION__ != "3.6.0"
                or runtime["python_executable_sha256"]
                != EXPECTED_EXECUTABLE_SHA256
                or runtime["python_flint_tree_sha256"]
                != EXPECTED_FLINT_TREE_SHA256):
            raise ValueError("receipt producer is not the exact arithmetic runtime")
        tree_before = _tree_identity(proof_root)
        if _tree_identity(PROOF_ROOT) != tree_before:
            raise ValueError(
                "clean checkout proof tree differs from the commit archive")
        started = dt.datetime.now(dt.timezone.utc)
        inherited = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "TZ",
                     "SYSTEMROOT")
        env = {key: os.environ[key] for key in inherited if key in os.environ}
        env.update({
            "HUANG_GRID_N": "2700",
            "HUANG_REGION1_REPLAY_WORKERS": str(workers),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        })
        git_dir_result = _git("rev-parse", "--absolute-git-dir")
        git_dir = pathlib.Path(git_dir_result.stdout.strip()).resolve()
        if git_dir_result.returncode or not git_dir.is_dir():
            raise ValueError("could not resolve source checkout Git directory")
        commands = []
        for name, arguments, marker, execution_scope in COMMANDS:
            command_env = _command_environment(
                env, execution_scope, proof_root, git_dir)
            git_head = (source_commit
                        if execution_scope == IMMUTABLE_ARCHIVE_GIT_SCOPE
                        else None)
            if git_head is not None:
                _require_archive_git_view(
                    source_commit, proof_root, command_env)
            commands.append(_run_command(
                name, arguments, marker, execution_scope, git_head,
                command_env, verification_root))
            if git_head is not None:
                _require_archive_git_view(
                    source_commit, proof_root, command_env)
            _require_clean_commit(source_commit)
            if _tree_identity(PROOF_ROOT) != tree_before:
                raise ValueError(
                    f"clean checkout proof tree changed during {name}")
            runtime_after = exact.runtime_record(
                50, workers, fresh_flint=True)
            if runtime_after != runtime:
                raise ValueError(
                    f"exact arithmetic runtime changed during {name}")
        tree_after = _tree_identity(proof_root)
        if tree_after != tree_before:
            raise ValueError("proof tree changed during final verification")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "source_commit": source_commit,
            "started_at": started.isoformat().replace("+00:00", "Z"),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat().replace(
                "+00:00", "Z"),
            "runtime": runtime,
            "commands": commands,
            "proof_tree": tree_before,
            "result_sha256": _result_hashes(proof_root),
            "verdict": VERDICT,
            "attestation_scope": ATTESTATION_SCOPE,
        }
        payload["receipt_sha256"] = _payload_sha256(payload)
        with receipt.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2,
                      ensure_ascii=True)
            handle.write("\n")
        check_receipt(receipt, source_commit, proof_root)
    _require_clean_commit(source_commit)
    print(f"PASS final verification receipt {payload['receipt_sha256']} {receipt}")
    return payload


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--source-commit", required=True)
    run.add_argument("--receipt", required=True)
    run.add_argument("--workers", type=int, default=3)
    check = sub.add_parser("check")
    check.add_argument("--source-commit", required=True)
    check.add_argument("--receipt", required=True)
    check.add_argument("--proof-root", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "run":
        if args.workers <= 0 or args.workers > MAX_WORKERS:
            raise ValueError(f"workers must be in 1..{MAX_WORKERS}")
        run_receipt(args.source_commit, args.receipt, args.workers)
    else:
        data = check_receipt(
            args.receipt, args.source_commit, args.proof_root)
        print(f"PASS final verification receipt {data['receipt_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
