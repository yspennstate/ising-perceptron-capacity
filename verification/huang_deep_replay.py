"""Independent execution replay for the Huang Region-I and sweep evidence.

The production Huang JSON files are intentionally small execution summaries.
Their lightweight validators establish schedule and source consistency, but
they do not reconstruct every certified job.  This separate auditor reruns
every deterministic job from the frozen proof sources, compares the complete
essential result with the production artifact, and publishes hash-bound lane
records plus one canonical report.

This file is deliberately outside ``proof_orchestrator.SOURCE_NAMES``.  It can
therefore be installed while the v2 proof controller is running without
changing any frozen source or controller-owned result.
"""

from __future__ import annotations

import argparse
import _thread
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import platform
import subprocess
import sys
import threading
import time
import traceback


# Keep every numerical subprocess single-threaded.  Parallelism is controlled
# explicitly by isolated lanes and is confined to CPUs 0--7.
for _name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"


HERE = pathlib.Path(__file__).resolve().parent
PROOF_ROOT = HERE.parent
DEFAULT_FROZEN = HERE / "results" / "proof_orchestrator" / \
    "frozen_sources_v2.json"
DEFAULT_STATE = HERE / "results" / "proof_orchestrator" / "state.json"
DEFAULT_OUTPUT = HERE / "audit_results" / "huang_deep_replay.json"

SCHEMA_VERSION = 1
MESH_AGENT = "codex-huang-deep-replay-20260710"
MESH_PARENT = "codex-root-recovery-20260710"
MESH_PROCESS_RESOURCE = \
    "external:process/local/perceptron-huang-deep-replay-20260710"
MESH_TASK_RESOURCE = "external:scheduled-task/PerceptronHuangDeepReplay"

EXPECTED_FROZEN_MANIFEST_SHA256 = \
    "eef0587697e0d22e038e848ebc02712f5d63079b0b52d533536574817afbf315"
EXPECTED_FROZEN_FILE_SHA256 = \
    "28929eb1e3d244f246dfdf34a7e4685efc84a256131bc15a9bb17f0a3b61bcb4"

SCOPE_POLICY = {
    "region1": {
        "count": 1404,
        "lanes": 64,
        "schedule_sha256":
            "075615cce1341fc75c3bc462c2fdb632f55384d2e76201531f33089247b77517",
        "artifact": "huang_region1.json",
        "canaries": (0, 700, 1403),
    },
    "sweep1": {
        "count": 1200,
        "lanes": 24,
        "schedule_sha256":
            "bb7e2d41343c13e2e98dfd2f6f435eb8c51569a7fc2232dca282a88013adce83",
        "artifact": "huang_sweep.json",
        "canaries": (0, 600, 1199),
    },
    "sweep2": {
        "count": 240,
        "lanes": 12,
        "schedule_sha256":
            "f1dacd97ae2a6e3fe0e27033ceedf1cf6ada352605d6ff0d5f15c37d7d123f61",
        "artifact": "huang_sweep2.json",
        "canaries": (0, 120, 239),
    },
}


class AuditError(RuntimeError):
    pass


class NotReady(AuditError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key}")
        out[key] = value
    return out


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number {text}")
    return value


def _plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_json_relaxed(path):
    """Strict JSON parser that permits ordinary finite floats.

    Production Region-I artifacts predate the canonical no-float serializer,
    so they cannot be read by block3bc_exact.load_json.  We still reject BOMs,
    duplicate keys, non-finite constants, and invalid UTF-8.
    """
    raw = pathlib.Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("BOM is forbidden")
    return json.loads(
        raw.decode("utf-8", "strict"),
        parse_float=_finite_float,
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )


def normalize_value(value):
    """Convert replay output to canonical JSON without binary floats."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if _plain_int(value):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite replay float")
        return repr(value)
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("replay dictionaries require string keys")
        return {key: normalize_value(item) for key, item in value.items()}
    raise TypeError(f"unsupported replay value {type(value).__name__}")


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_linklike(path: pathlib.Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _safe_resolve(path, label, *, must_exist=False) -> pathlib.Path:
    """Resolve only after rejecting symlinks/junctions in every component."""
    raw = pathlib.Path(os.path.abspath(os.fspath(path)))
    current = pathlib.Path(raw.anchor)
    for part in raw.parts[1:]:
        current /= part
        if _is_linklike(current):
            raise AuditError(f"{label} may not traverse a symlink or junction")
    if must_exist and not raw.exists():
        raise AuditError(f"missing {label}: {raw}")
    return raw.resolve(strict=must_exist)


def _exact_module():
    import block3bc_exact as exact
    return exact


def _orchestrator_module():
    import proof_orchestrator as orchestrator
    return orchestrator


def canonical_jobs(scope) -> tuple:
    if scope == "region1":
        import huang_region1 as module
        jobs = tuple(module.bands())
    elif scope == "sweep1":
        import huang_sweep as module
        jobs = tuple(module.build_jobs(48)[0])
    elif scope == "sweep2":
        import huang_sweep2 as module
        jobs = tuple(module.build_jobs()[0])
    else:
        raise ValueError(f"unknown replay scope {scope}")
    return jobs


def schedule_record(scope) -> dict:
    exact = _exact_module()
    jobs = canonical_jobs(scope)
    schedule = [[repr(value) for value in job] for job in jobs]
    record = {
        "count": len(jobs),
        "schedule_sha256": exact.payload_sha256(schedule, omit=()),
    }
    policy = SCOPE_POLICY[scope]
    if record["count"] != policy["count"]:
        raise AuditError(
            f"{scope} schedule count drift: {record['count']} != "
            f"{policy['count']}")
    if record["schedule_sha256"] != policy["schedule_sha256"]:
        raise AuditError(f"{scope} schedule hash drift")
    return record


def normalize_result(scope, index, job, raw) -> dict:
    job_record = [repr(value) for value in job]
    if scope == "region1":
        if not isinstance(raw, dict):
            raise TypeError("Region-I replay did not return a dictionary")
        return {
            "index": int(index),
            "job": job_record,
            "result": normalize_value(raw),
        }
    if scope in ("sweep1", "sweep2"):
        if not isinstance(raw, tuple) or len(raw) != 5:
            raise TypeError("sweep replay returned an invalid tuple")
        observed_job, ok, leaves, worst, _runtime_ms = raw
        if tuple(observed_job) != tuple(job):
            raise AuditError(f"{scope} worker returned another job")
        if not isinstance(ok, bool) or not _plain_int(leaves):
            raise AuditError(f"{scope} worker returned invalid verdict metadata")
        return {
            "index": int(index),
            "job": job_record,
            "result": {
                "ok": bool(ok),
                "leaves": int(leaves),
                "worst": None if worst is None else repr(worst),
            },
        }
    raise ValueError(f"unknown replay scope {scope}")


def replay_one(scope, index) -> dict:
    jobs = canonical_jobs(scope)
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("replay index must be a plain integer")
    if not 0 <= index < len(jobs):
        raise IndexError(index)
    job = jobs[index]
    if scope == "region1":
        import huang_region1 as module
        raw = module.band_job(job)
    elif scope == "sweep1":
        import huang_sweep as module
        raw = module.worker(job)
    elif scope == "sweep2":
        import huang_sweep2 as module
        raw = module.worker(job)
    else:
        raise ValueError(f"unknown replay scope {scope}")
    return normalize_result(scope, index, job, raw)


def _production_records(scope, artifact) -> list[dict]:
    jobs = canonical_jobs(scope)
    artifact = _safe_resolve(
        artifact, f"{scope} production artifact", must_exist=True)
    data = load_json_relaxed(artifact)

    if scope == "region1":
        import huang_region1 as module
        import verify_all
        ok, detail = verify_all.validate_region1_manifest(data, module)
        if not ok:
            raise AuditError(f"Region-I structural validation failed: {detail}")
        rows = data.get("results")
        if not isinstance(rows, list):
            raise AuditError("Region-I results are missing")
        by_band = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("band"), list):
                raise AuditError("invalid Region-I production row")
            key = tuple(row["band"])
            if key in by_band:
                raise AuditError("duplicate Region-I production band")
            by_band[key] = row
        output = []
        for index, job in enumerate(jobs):
            row = by_band.get(tuple(job))
            if row is None:
                raise AuditError(f"missing Region-I production index {index}")
            output.append(normalize_result(scope, index, job, row))
        if len(by_band) != len(jobs):
            raise AuditError("Region-I production schedule has extra rows")
        return output

    import verify_all
    if scope == "sweep1":
        import huang_sweep as module
        stage = 1
    elif scope == "sweep2":
        import huang_sweep2 as module
        stage = 2
    else:
        raise ValueError(scope)
    ok, detail, _total = verify_all.validate_sweep_manifest(
        artifact, module, stage)
    if not ok:
        raise AuditError(f"{scope} structural validation failed: {detail}")
    rows = data.get("records")
    if not isinstance(rows, list):
        raise AuditError(f"{scope} records are missing")
    by_index = {}
    for row in rows:
        if not isinstance(row, dict):
            raise AuditError(f"invalid {scope} production row")
        index = row.get("index")
        if (not _plain_int(index)
                or index in by_index):
            raise AuditError(f"invalid or duplicate {scope} index")
        by_index[index] = row
    output = []
    for index, job in enumerate(jobs):
        row = by_index.get(index)
        if row is None:
            raise AuditError(f"missing {scope} production index {index}")
        expected_cell = [repr(value) for value in job]
        if row.get("cell") != expected_cell:
            raise AuditError(f"{scope} production cell mismatch at {index}")
        if (not isinstance(row.get("ok"), bool)
                or not _plain_int(row.get("leaves"))
                or row.get("leaves") <= 0
                or row.get("worst") is not None
                and not isinstance(row.get("worst"), str)):
            raise AuditError(f"{scope} production metadata is not canonical")
        output.append({
            "index": index,
            "job": expected_cell,
            "result": {
                "ok": row["ok"],
                "leaves": row["leaves"],
                "worst": row.get("worst"),
            },
        })
    if len(by_index) != len(jobs):
        raise AuditError(f"{scope} production schedule has extra rows")
    return output


def _record_is_success(scope, record) -> bool:
    result = record.get("result", {})
    if scope == "region1":
        return (result.get("ok") is True
                and _plain_int(result.get("nfail"))
                and result.get("nfail") == 0
                and _plain_int(result.get("cells"))
                and result.get("cells") >= 0
                and isinstance(result.get("lbox"), list)
                and len(result.get("lbox")) == 4
                and all(isinstance(value, str)
                        for value in result.get("lbox"))
                and isinstance(result.get("root_certificate"), dict)
                and bool(result.get("root_certificate")))
    return (result.get("ok") is True
            and _plain_int(result.get("leaves"))
            and result.get("leaves") > 0)


def _verified_frozen(path) -> dict:
    orchestrator = _orchestrator_module()
    path = _safe_resolve(path, "frozen manifest", must_exist=True)
    data = orchestrator.verify_frozen_manifest(path)
    if data.get("manifest_sha256") != EXPECTED_FROZEN_MANIFEST_SHA256:
        raise AuditError("unexpected frozen manifest identity")
    if file_sha256(path) != EXPECTED_FROZEN_FILE_SHA256:
        raise AuditError("unexpected frozen manifest file hash")
    return data


def _batch_payload(scope, lane, lanes, frozen_path, artifact_path,
                   artifact_sha256, auditor_sha256) -> dict:
    exact = _exact_module()
    exact.apply_worker_policy()
    if file_sha256(__file__) != auditor_sha256:
        raise AuditError("auditor source changed before worker replay")
    frozen = _verified_frozen(frozen_path)
    artifact_path = _safe_resolve(
        artifact_path, f"{scope} production artifact", must_exist=True)
    if file_sha256(artifact_path) != artifact_sha256:
        raise AuditError("production artifact changed before worker replay")
    schedule = schedule_record(scope)
    jobs = canonical_jobs(scope)
    if not (0 <= lane < lanes == SCOPE_POLICY[scope]["lanes"]):
        raise AuditError("invalid replay lane")
    production = _production_records(scope, artifact_path)
    indices = list(range(lane, len(jobs), lanes))
    records = []
    for index in indices:
        replayed = replay_one(scope, index)
        if exact.payload_sha256(replayed, omit=()) \
                != exact.payload_sha256(production[index], omit=()):
            raise AuditError(
                f"{scope} replay differs from production at index {index}")
        if not _record_is_success(scope, replayed):
            raise AuditError(f"{scope} replay failed at index {index}")
        records.append(replayed)
    production_subset = [production[index] for index in indices]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "huang_deep_replay_batch",
        "scope": scope,
        "lane": lane,
        "lanes": lanes,
        "indices": indices,
        "schedule": schedule,
        "auditor_sha256": auditor_sha256,
        "frozen_manifest_sha256": frozen["manifest_sha256"],
        "frozen_manifest_file_sha256": file_sha256(frozen_path),
        "source_sha256": frozen["source_sha256"],
        "runtime": _orchestrator_module().runtime_fingerprint(),
        "policy": {
            "priority": "BelowNormal",
            "windows_affinity_mask": "0xFF",
            "single_threaded_numeric_libraries": True,
        },
        "artifact_sha256": artifact_sha256,
        "production_records_sha256": exact.payload_sha256(
            production_subset, omit=()),
        "records": records,
    }
    payload["batch_sha256"] = exact.payload_sha256(
        payload, omit=("batch_sha256",))
    return payload


def validate_batch(key, payload, *, scope=None, lane=None, lanes=None,
                   frozen_path=DEFAULT_FROZEN, artifact_path=None,
                   artifact_sha256=None, auditor_sha256=None) -> dict:
    exact = _exact_module()
    if not isinstance(payload, dict):
        raise AuditError("batch is not an object")
    required = {
        "schema_version", "kind", "scope", "lane", "lanes", "indices",
        "schedule", "auditor_sha256", "frozen_manifest_sha256",
        "frozen_manifest_file_sha256", "source_sha256", "runtime",
        "policy", "artifact_sha256", "production_records_sha256",
        "records", "batch_sha256",
    }
    if set(payload) != required:
        raise AuditError("invalid batch schema")
    if (not _plain_int(payload["schema_version"])
            or payload["schema_version"] != SCHEMA_VERSION
            or payload["kind"] != "huang_deep_replay_batch"
            or payload["batch_sha256"] != exact.payload_sha256(
                payload, omit=("batch_sha256",))):
        raise AuditError("invalid batch identity or hash")
    actual_scope = payload["scope"]
    actual_lane = payload["lane"]
    actual_lanes = payload["lanes"]
    if actual_scope not in SCOPE_POLICY:
        raise AuditError("unknown batch scope")
    if scope is not None and actual_scope != scope:
        raise AuditError("batch scope mismatch")
    if lane is not None and actual_lane != lane:
        raise AuditError("batch lane mismatch")
    if lanes is not None and actual_lanes != lanes:
        raise AuditError("batch lane-count mismatch")
    if key is None or str(key) != f"{actual_scope}-{actual_lane}":
        raise AuditError("isolated worker key mismatch")
    expected_auditor = auditor_sha256 or file_sha256(__file__)
    if (payload["auditor_sha256"] != expected_auditor
            or file_sha256(__file__) != expected_auditor):
        raise AuditError("batch auditor source mismatch")
    frozen = _verified_frozen(frozen_path)
    if (payload["frozen_manifest_sha256"] != frozen["manifest_sha256"]
            or payload["frozen_manifest_file_sha256"]
            != file_sha256(frozen_path)
            or payload["source_sha256"] != frozen["source_sha256"]
            or payload["runtime"]
            != _orchestrator_module().runtime_fingerprint()):
        raise AuditError("batch source or runtime mismatch")
    schedule = payload["schedule"]
    if (not isinstance(schedule, dict)
            or set(schedule) != {"count", "schedule_sha256"}
            or not _plain_int(schedule["count"])
            or payload["schedule"] != schedule_record(actual_scope)):
        raise AuditError("batch schedule mismatch")
    policy = SCOPE_POLICY[actual_scope]
    expected_policy = {
        "priority": "BelowNormal",
        "windows_affinity_mask": "0xFF",
        "single_threaded_numeric_libraries": True,
    }
    if (not isinstance(payload["policy"], dict)
            or set(payload["policy"]) != set(expected_policy)
            or payload["policy"].get("priority") != "BelowNormal"
            or payload["policy"].get("windows_affinity_mask") != "0xFF"
            or payload["policy"].get(
                "single_threaded_numeric_libraries") is not True):
        raise AuditError("batch owner policy mismatch")
    if not (_plain_int(actual_lane) and _plain_int(actual_lanes)
            and 0 <= actual_lane < actual_lanes == policy["lanes"]):
        raise AuditError("invalid batch lane geometry")
    expected_indices = list(range(
        actual_lane, policy["count"], actual_lanes))
    if (not isinstance(payload["indices"], list)
            or any(not _plain_int(index) for index in payload["indices"])
            or payload["indices"] != expected_indices):
        raise AuditError("batch index coverage mismatch")
    records = payload["records"]
    if (not isinstance(records, list)
            or any(not isinstance(row, dict)
                   or not _plain_int(row.get("index")) for row in records)
            or [row.get("index") for row in records] != expected_indices):
        raise AuditError("batch records do not cover their exact indices")
    if any(not _record_is_success(actual_scope, row) for row in records):
        raise AuditError("batch contains a failed replay")
    if artifact_path is None or artifact_sha256 is None:
        raise AuditError("path-backed production validation is required")
    if payload["artifact_sha256"] != artifact_sha256:
        raise AuditError("batch artifact hash mismatch")
    artifact_path = _safe_resolve(
        artifact_path, f"{actual_scope} production artifact",
        must_exist=True)
    if file_sha256(artifact_path) != payload["artifact_sha256"]:
        raise AuditError("batch production artifact drift")
    production = _production_records(actual_scope, artifact_path)
    subset = [production[index] for index in expected_indices]
    if exact.payload_sha256(records, omit=()) != exact.payload_sha256(
            subset, omit=()):
        raise AuditError("batch records differ from production")
    if payload["production_records_sha256"] != exact.payload_sha256(
            subset, omit=()):
        raise AuditError("batch production-record hash mismatch")
    return {
        "scope": actual_scope,
        "lane": actual_lane,
        "count": len(records),
        "batch_sha256": payload["batch_sha256"],
    }


def _controller_done(state_path) -> dict:
    state = load_json_relaxed(state_path)
    if (not isinstance(state, dict)
            or state.get("kind") != "perceptron_proof_orchestrator_state"
            or state.get("phase") != "DONE"
            or state.get("status") != "complete"):
        raise NotReady("proof controller has not reached DONE")
    return state


def _artifact_paths(overrides=None) -> dict[str, pathlib.Path]:
    result = {
        scope: _safe_resolve(
            HERE / "results" / policy["artifact"],
            f"{scope} production artifact")
        for scope, policy in SCOPE_POLICY.items()
    }
    if overrides:
        for scope, path in overrides.items():
            if scope not in SCOPE_POLICY:
                raise ValueError(f"unknown artifact override {scope}")
            result[scope] = _safe_resolve(
                path, f"{scope} production artifact")
    return result


def _safe_batch_path(audit_root, scope, lane, lanes):
    if scope not in SCOPE_POLICY or not _plain_int(lane) \
            or not _plain_int(lanes):
        raise AuditError("invalid batch path coordinates")
    root = _safe_resolve(audit_root, "audit root")
    batch_root = root / "batches"
    _safe_resolve(batch_root, "batch directory")
    path = batch_root / f"{scope}-lane-{lane:03d}-of-{lanes:03d}.json"
    if _is_linklike(path):
        raise AuditError("batch file may not be a symlink or junction")
    if _safe_resolve(path, "batch file").parent != batch_root:
        raise AuditError("batch path escaped the audit root")
    return path


class _MeshPulse:
    def __init__(self, orchestrator, interval=300):
        self.orchestrator = orchestrator
        self.interval = interval
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(self.interval):
            try:
                self.orchestrator.mesh_command([
                    "pulse", "--agent", MESH_AGENT, "--status", "working",
                    "--task", "Deep-replay every frozen Huang certificate job",
                ])
            except BaseException as exc:  # surfaced before every publication
                if self.stop.is_set():
                    return
                self.error = exc
                _thread.interrupt_main()
                return

    def __enter__(self):
        self.thread.start()
        return self

    def check(self):
        if self.error is not None:
            raise AuditError(f"mesh heartbeat failed: {self.error}")

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive() and exc_type is None:
            raise AuditError("mesh heartbeat thread did not terminate")
        if self.error is not None and exc_type is None:
            raise AuditError(f"mesh heartbeat failed: {self.error}")


def _mesh_join_and_claim(audit_root, read_resources):
    orchestrator = _orchestrator_module()
    joined = False
    try:
        orchestrator.mesh_command([
            "hello", "--agent", MESH_AGENT, "--runtime", "codex",
            "--parent", MESH_PARENT, "--status", "working", "--ttl", "45m",
            "--cwd", str(PROOF_ROOT), "--task",
            "Deep-replay every frozen Huang certificate job",
        ])
        joined = True
        orchestrator.mesh_command(
            ["digest", "--agent", MESH_AGENT], expect_json=False)
        argv = [
            "claim", "--agent", MESH_AGENT,
            "--resource", str(_safe_resolve(audit_root, "audit root")),
            "--resource", MESH_PROCESS_RESOURCE,
            "--resource", MESH_TASK_RESOURCE,
        ]
        for resource in sorted({str(path) for path in read_resources}):
            argv.extend(["--resource", resource])
        argv.extend([
            "--mode", "exclusive", "--ttl", "45m", "--purpose",
            "Freeze every source/evidence byte and own all deep-replay output",
        ])
        orchestrator.mesh_command(argv)
        return orchestrator
    except BaseException:
        if joined:
            _mesh_bye(orchestrator, "offline",
                      "Huang deep replay could not acquire its full lease")
        raise


def _mesh_assert(orchestrator, *resources):
    argv = ["assert", "--agent", MESH_AGENT]
    for resource in resources:
        argv.extend(["--resource", str(pathlib.Path(resource).resolve())
                     if not str(resource).startswith("external:")
                     else str(resource)])
    orchestrator.mesh_command(argv)


def _mesh_bye(orchestrator, status, summary):
    with contextlib.suppress(Exception):
        orchestrator.mesh_command([
            "bye", "--agent", MESH_AGENT, "--status", status,
            "--summary", summary,
        ])


def _disable_scheduled_task(orchestrator):
    if os.name != "nt":
        return
    _mesh_assert(orchestrator, MESH_TASK_RESOURCE)
    result = subprocess.run(
        ["schtasks.exe", "/Change", "/TN", "PerceptronHuangDeepReplay",
         "/Disable"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, timeout=30,
        creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0)
                       | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)),
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode(
            "utf-8", "replace").strip()
        raise AuditError(f"could not disable completed audit task: {detail}")


def _validate_production_artifacts(artifacts):
    import huang_star_interior as star
    star_path = HERE / "results" / "huang_star_interior.json"
    star.verify_certificate(star_path)
    production = {
        scope: _production_records(scope, path)
        for scope, path in artifacts.items()
    }
    return star_path.resolve(), production


def _frozen_read_resources(frozen, frozen_path, state_path, artifacts,
                           star_path):
    resources = {
        _safe_resolve(__file__, "auditor source", must_exist=True),
        _safe_resolve(frozen_path, "frozen manifest", must_exist=True),
        _safe_resolve(state_path, "controller state", must_exist=True),
        _safe_resolve(star_path, "star certificate", must_exist=True),
    }
    source_root = _safe_resolve(
        frozen["source_root"], "frozen source root", must_exist=True)
    for name in frozen["source_sha256"]:
        resources.add(_safe_resolve(
            source_root / "verification" / name,
            f"frozen source {name}", must_exist=True))
    recovery_root = _safe_resolve(
        frozen["recovery_root"], "frozen recovery root", must_exist=True)
    for name in frozen["recovery_sha256"]:
        resources.add(_safe_resolve(
            recovery_root / name, f"frozen recovery source {name}",
            must_exist=True))
    for scope, path in artifacts.items():
        resources.add(_safe_resolve(
            path, f"{scope} production artifact", must_exist=True))
    return resources


def run_audit(frozen_manifest=DEFAULT_FROZEN, artifacts=None,
              audit_root=None, workers=8, state_path=DEFAULT_STATE,
              output_path=None) -> dict:
    exact = _exact_module()
    orchestrator = _orchestrator_module()
    frozen_manifest = _safe_resolve(
        frozen_manifest, "frozen manifest", must_exist=True)
    state_path = _safe_resolve(
        state_path, "controller state", must_exist=True)
    if (frozen_manifest != _safe_resolve(
            DEFAULT_FROZEN, "default frozen manifest", must_exist=True)
            or state_path != _safe_resolve(
                DEFAULT_STATE, "default controller state", must_exist=True)
            or artifacts is not None):
        raise AuditError(
            "the v2 deep replay accepts only its exact production inputs")
    output_path = _safe_resolve(
        output_path or DEFAULT_OUTPUT, "deep-replay report")
    audit_root = _safe_resolve(
        audit_root or output_path.parent, "audit root")
    if output_path.parent != audit_root:
        raise AuditError("report must be a direct child of the audit root")
    if not isinstance(workers, int) or isinstance(workers, bool) \
            or not 1 <= workers <= 8:
        raise ValueError("workers must be a plain integer in [1, 8]")
    auditor_sha_start = file_sha256(__file__)

    controller_state = _controller_done(state_path)
    frozen = _verified_frozen(frozen_manifest)
    artifact_paths = _artifact_paths()
    for path in artifact_paths.values():
        if not path.is_file():
            raise AuditError(f"missing production artifact {path}")

    star_path = _safe_resolve(
        HERE / "results" / "huang_star_interior.json",
        "star certificate", must_exist=True)
    read_resources = _frozen_read_resources(
        frozen, frozen_manifest, state_path, artifact_paths, star_path)
    singleton = orchestrator.Singleton(
        "Global\\PerceptronHuangDeepReplayV1")
    singleton.__enter__()
    mesh_joined = False
    started = utc_now()
    success = False
    try:
        orchestrator = _mesh_join_and_claim(audit_root, read_resources)
        mesh_joined = True
        # A current PASS report makes a manual/repeated invocation idempotent,
        # but it is verified only after acquiring the same exclusive evidence
        # window used by a fresh replay.
        if output_path.is_file():
            verified = _verify_report_unlocked(output_path, current=True)
            _disable_scheduled_task(orchestrator)
            success = True
            return verified
        controller_state = _controller_done(state_path)
        state_sha_before = file_sha256(state_path)
        frozen = _verified_frozen(frozen_manifest)
        artifact_hashes_before = {
            scope: file_sha256(path)
            for scope, path in artifact_paths.items()}
        star_path, production = _validate_production_artifacts(artifact_paths)
        star_sha_before = file_sha256(star_path)
        schedule = {scope: schedule_record(scope) for scope in SCOPE_POLICY}
        exact.apply_worker_policy()
        with _MeshPulse(orchestrator) as pulse:
            _mesh_assert(orchestrator, audit_root)
            audit_root.mkdir(parents=True, exist_ok=True)
            batch_root = audit_root / "batches"
            _mesh_assert(orchestrator, batch_root)
            batch_root.mkdir(parents=True, exist_ok=True)

            canaries = {}
            for scope, policy in SCOPE_POLICY.items():
                rows = []
                for index in policy["canaries"]:
                    pulse.check()
                    replayed = replay_one(scope, index)
                    if exact.payload_sha256(
                            replayed, omit=()) != exact.payload_sha256(
                                production[scope][index], omit=()):
                        raise AuditError(
                            f"{scope} canary differs at index {index}")
                    if not _record_is_success(scope, replayed):
                        raise AuditError(
                            f"{scope} canary failed at index {index}")
                    rows.append(replayed)
                canaries[scope] = {
                    "indices": list(policy["canaries"]),
                    "records_sha256": exact.payload_sha256(rows, omit=()),
                }

            phase_reports = {}
            all_batch_hashes = {}
            for scope, policy in SCOPE_POLICY.items():
                lanes = policy["lanes"]
                specs = []
                batch_payloads = {}
                for lane in range(lanes):
                    batch_path = _safe_batch_path(
                        audit_root, scope, lane, lanes)
                    if batch_path.is_file():
                        payload = exact.load_json(batch_path)
                        validate_batch(
                            f"{scope}-{lane}", payload, scope=scope,
                            lane=lane, lanes=lanes,
                            frozen_path=frozen_manifest,
                            artifact_path=artifact_paths[scope],
                            artifact_sha256=artifact_hashes_before[scope],
                            auditor_sha256=auditor_sha_start,
                        )
                        batch_payloads[lane] = payload
                        continue
                    command = [
                        sys.executable, "-B", str(pathlib.Path(__file__).resolve()),
                        "worker", "--scope", scope, "--lane", str(lane),
                        "--lanes", str(lanes), "--frozen-manifest",
                        str(frozen_manifest), "--artifact",
                        str(artifact_paths[scope]), "--artifact-sha256",
                        artifact_hashes_before[scope], "--auditor-sha256",
                        auditor_sha_start,
                    ]
                    specs.append((f"{scope}-{lane}", command))

                def validator(key, payload, _scope=scope, _lanes=lanes):
                    lane_value = int(key.rsplit("-", 1)[1])
                    validate_batch(
                        key, payload, scope=_scope, lane=lane_value,
                        lanes=_lanes, frozen_path=frozen_manifest,
                        artifact_path=artifact_paths[_scope],
                        artifact_sha256=artifact_hashes_before[_scope],
                        auditor_sha256=auditor_sha_start,
                    )

                workdir = audit_root / "workers" / scope
                _mesh_assert(orchestrator, workdir)
                workdir.mkdir(parents=True, exist_ok=True)
                for key, payload in exact.isolated_subprocess_results(
                        specs, workers, workdir,
                        timeout_seconds=10800, retries=1,
                        result_validator=validator):
                    pulse.check()
                    lane = int(key.rsplit("-", 1)[1])
                    batch_path = _safe_batch_path(
                        audit_root, scope, lane, lanes)
                    if file_sha256(__file__) != auditor_sha_start:
                        raise AuditError("auditor source changed during replay")
                    _mesh_assert(orchestrator, batch_path)
                    exact.write_json_atomic(batch_path, payload, overwrite=False)
                    batch_payloads[lane] = payload

                if set(batch_payloads) != set(range(lanes)):
                    raise AuditError(f"{scope} does not cover every lane")
                records = []
                batch_files = []
                for lane in range(lanes):
                    payload = batch_payloads[lane]
                    records.extend(payload["records"])
                    path = _safe_batch_path(audit_root, scope, lane, lanes)
                    batch_files.append({
                        "name": path.name,
                        "file_sha256": file_sha256(path),
                        "batch_sha256": payload["batch_sha256"],
                    })
                records.sort(key=lambda row: row["index"])
                expected_indices = list(range(policy["count"]))
                if [row["index"] for row in records] != expected_indices:
                    raise AuditError(f"{scope} final index coverage mismatch")
                if exact.payload_sha256(
                        records, omit=()) != exact.payload_sha256(
                            production[scope], omit=()):
                    raise AuditError(f"{scope} final replay differs from production")
                if any(not _record_is_success(scope, row) for row in records):
                    raise AuditError(f"{scope} final replay contains failures")
                phase_reports[scope] = {
                    "schedule": schedule[scope],
                    "replayed": len(records),
                    "failures": 0,
                    "metadata_mismatches": 0,
                    "record_set_sha256": exact.payload_sha256(
                        records, omit=()),
                    "production_record_set_sha256": exact.payload_sha256(
                        production[scope], omit=()),
                    "batches": batch_files,
                }
                all_batch_hashes[scope] = batch_files

            pulse.check()
            artifact_hashes_after = {
                scope: file_sha256(path)
                for scope, path in artifact_paths.items()}
            if artifact_hashes_after != artifact_hashes_before:
                raise AuditError("production artifacts changed during replay")
            if file_sha256(__file__) != auditor_sha_start:
                raise AuditError("auditor source changed during replay")
            _verified_frozen(frozen_manifest)
            import huang_star_interior as star
            star.verify_certificate(star_path)
            if file_sha256(star_path) != star_sha_before:
                raise AuditError("star certificate changed during replay")
            controller_state_after = _controller_done(state_path)
            if (file_sha256(state_path) != state_sha_before
                    or controller_state_after != controller_state):
                raise AuditError("controller state changed during replay")

            report = {
                "schema_version": SCHEMA_VERSION,
                "kind": "huang_deep_replay_report",
                "verdict": "PASS",
                "started_utc": started,
                "completed_utc": utc_now(),
                "auditor_sha256": auditor_sha_start,
                "frozen_manifest": {
                    "manifest_sha256": frozen["manifest_sha256"],
                    "file_sha256": file_sha256(frozen_manifest),
                    "source_sha256": frozen["source_sha256"],
                },
                "controller_state": {
                    "file_sha256": state_sha_before,
                    "updated_utc": controller_state.get("updated_utc"),
                },
                "runtime": orchestrator.runtime_fingerprint(),
                "policy": {
                    "workers": workers,
                    "priority": "BelowNormal",
                    "windows_affinity_mask": "0xFF",
                    "single_threaded_numeric_libraries": True,
                    "lane_timeout_seconds": 10800,
                    "retries": 1,
                },
                "star_interior": {
                    "path": star_path.name,
                    "file_sha256": star_sha_before,
                    "exact_replay": True,
                },
                "production_artifacts_before": artifact_hashes_before,
                "production_artifacts_after": artifact_hashes_after,
                "canaries": canaries,
                "phases": phase_reports,
                "batch_files": all_batch_hashes,
            }
            report["report_sha256"] = exact.payload_sha256(
                report, omit=("report_sha256",))
            pulse.check()
            _mesh_assert(orchestrator, output_path)
            exact.write_json_atomic(output_path, report, overwrite=False)
            verified = _verify_report_unlocked(output_path, current=True)
            failure_path = output_path.with_suffix(".failure.json")
            if failure_path.exists() or failure_path.is_symlink():
                _mesh_assert(orchestrator, failure_path)
                failure_path.unlink()
        _disable_scheduled_task(orchestrator)
        success = True
        return verified
    except BaseException as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "kind": "huang_deep_replay_failure",
            "verdict": "FAIL",
            "started_utc": started,
            "failed_utc": utc_now(),
            "auditor_sha256": file_sha256(__file__),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-20:],
        }
        failure["failure_sha256"] = exact.payload_sha256(
            failure, omit=("failure_sha256",))
        failure_path = output_path.with_suffix(".failure.json")
        with contextlib.suppress(Exception):
            _mesh_assert(orchestrator, failure_path)
            exact.write_json_atomic(failure_path, failure, overwrite=True)
        raise
    finally:
        if mesh_joined:
            _mesh_bye(
                orchestrator, "done" if success else "offline",
                "Huang deep replay completed" if success
                else "Huang deep replay stopped safely",
            )
        singleton.__exit__(None, None, None)


def _is_hex64(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _parse_utc(value):
    if not isinstance(value, str):
        raise AuditError("report timestamp is not a string")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise AuditError("invalid report timestamp") from exc
    if parsed.tzinfo is None:
        raise AuditError("report timestamp lacks a timezone")
    return parsed


def _verify_report_unlocked(path, rerun=False, current=False) -> dict:
    """Verify the report, every batch hash, and production correspondence.

    ``rerun`` is retained as a CLI compatibility flag; report verification is
    always path-backed and batch-complete.  It never launches numerical jobs.
    """
    del rerun
    exact = _exact_module()
    orchestrator = _orchestrator_module()
    path = _safe_resolve(path, "deep-replay report", must_exist=True)
    report = exact.load_json(path)
    required = {
        "schema_version", "kind", "verdict", "started_utc",
        "completed_utc", "auditor_sha256", "frozen_manifest",
        "controller_state", "runtime", "policy", "star_interior",
        "production_artifacts_before", "production_artifacts_after",
        "canaries", "phases", "batch_files", "report_sha256",
    }
    if set(report) != required:
        raise AuditError("invalid deep-replay report schema")
    if (not _plain_int(report["schema_version"])
            or report["schema_version"] != SCHEMA_VERSION
            or report["kind"] != "huang_deep_replay_report"
            or report["verdict"] != "PASS"
            or not _is_hex64(report["report_sha256"])
            or report["report_sha256"] != exact.payload_sha256(
                report, omit=("report_sha256",))):
        raise AuditError("invalid deep-replay report identity or hash")
    if _parse_utc(report["completed_utc"]) < _parse_utc(report["started_utc"]):
        raise AuditError("deep-replay timestamps are reversed")
    if not _is_hex64(report["auditor_sha256"]):
        raise AuditError("invalid auditor hash")

    frozen_schema = {"manifest_sha256", "file_sha256", "source_sha256"}
    if (not isinstance(report["frozen_manifest"], dict)
            or set(report["frozen_manifest"]) != frozen_schema):
        raise AuditError("invalid frozen-manifest report schema")
    frozen = _verified_frozen(DEFAULT_FROZEN)
    if (report["frozen_manifest"]["manifest_sha256"]
            != frozen["manifest_sha256"]
            or report["frozen_manifest"]["file_sha256"]
            != file_sha256(DEFAULT_FROZEN)
            or report["frozen_manifest"]["source_sha256"]
            != frozen["source_sha256"]):
        raise AuditError("report frozen-source binding mismatch")
    if report["auditor_sha256"] != file_sha256(__file__):
        raise AuditError("report auditor source drift")

    expected_runtime = orchestrator.runtime_fingerprint()
    if report["runtime"] != expected_runtime:
        raise AuditError("report arithmetic runtime drift")
    policy = report["policy"]
    if (not isinstance(policy, dict)
            or set(policy) != {
                "workers", "priority", "windows_affinity_mask",
                "single_threaded_numeric_libraries", "lane_timeout_seconds",
                "retries"}
            or not _plain_int(policy["workers"])
            or not 1 <= policy["workers"] <= 8
            or policy["priority"] != "BelowNormal"
            or policy["windows_affinity_mask"] != "0xFF"
            or policy["single_threaded_numeric_libraries"] is not True
            or not _plain_int(policy["lane_timeout_seconds"])
            or policy["lane_timeout_seconds"] != 10800
            or not _plain_int(policy["retries"])
            or policy["retries"] != 1):
        raise AuditError("invalid report owner policy")

    controller = report["controller_state"]
    if (not isinstance(controller, dict)
            or set(controller) != {"file_sha256", "updated_utc"}
            or not _is_hex64(controller["file_sha256"])):
        raise AuditError("invalid controller-state report schema")
    _parse_utc(controller["updated_utc"])
    star_record = report["star_interior"]
    if (not isinstance(star_record, dict)
            or set(star_record) != {"path", "file_sha256", "exact_replay"}
            or star_record["path"] != "huang_star_interior.json"
            or not _is_hex64(star_record["file_sha256"])
            or star_record["exact_replay"] is not True):
        raise AuditError("invalid star-interior report schema")

    artifact_scopes = set(SCOPE_POLICY)
    before = report["production_artifacts_before"]
    after = report["production_artifacts_after"]
    if (not isinstance(before, dict) or not isinstance(after, dict)
            or set(before) != artifact_scopes or set(after) != artifact_scopes
            or before != after
            or any(not _is_hex64(value) for value in before.values())):
        raise AuditError("invalid production artifact inventory")
    artifacts = _artifact_paths()
    current_hashes = {
        scope: file_sha256(artifact) for scope, artifact in artifacts.items()}
    if current_hashes != after:
        raise AuditError("current production artifact drift")

    if (not isinstance(report["phases"], dict)
            or set(report["phases"]) != artifact_scopes
            or not isinstance(report["batch_files"], dict)
            or set(report["batch_files"]) != artifact_scopes
            or not isinstance(report["canaries"], dict)
            or set(report["canaries"]) != artifact_scopes):
        raise AuditError("invalid report phase/canary scope set")

    production = {
        scope: _production_records(scope, artifacts[scope])
        for scope in SCOPE_POLICY}
    root = path.parent
    for scope, scope_policy in SCOPE_POLICY.items():
        canary = report["canaries"][scope]
        canary_rows = [production[scope][index]
                       for index in scope_policy["canaries"]]
        if (not isinstance(canary, dict)
                or set(canary) != {"indices", "records_sha256"}
                or canary["indices"] != list(scope_policy["canaries"])
                or canary["records_sha256"] != exact.payload_sha256(
                    canary_rows, omit=())):
            raise AuditError(f"invalid {scope} canary record")

        phase = report["phases"][scope]
        phase_schema = {
            "schedule", "replayed", "failures", "metadata_mismatches",
            "record_set_sha256", "production_record_set_sha256", "batches"}
        if (not isinstance(phase, dict) or set(phase) != phase_schema
                or phase["schedule"] != schedule_record(scope)
                or not _plain_int(phase["replayed"])
                or phase["replayed"] != scope_policy["count"]
                or not _plain_int(phase["failures"])
                or phase["failures"] != 0
                or not _plain_int(phase["metadata_mismatches"])
                or phase["metadata_mismatches"] != 0
                or not _is_hex64(phase["record_set_sha256"])
                or not _is_hex64(phase["production_record_set_sha256"])
                or phase["record_set_sha256"]
                != phase["production_record_set_sha256"]):
            raise AuditError(f"invalid {scope} report phase")
        batches = phase["batches"]
        if (not isinstance(batches, list)
                or len(batches) != scope_policy["lanes"]
                or report["batch_files"][scope] != batches):
            raise AuditError(f"invalid {scope} report batch inventory")

        aggregated = []
        seen_names = set()
        for lane, entry in enumerate(batches):
            expected_name = (
                f"{scope}-lane-{lane:03d}-of-"
                f"{scope_policy['lanes']:03d}.json")
            if (not isinstance(entry, dict)
                    or set(entry) != {
                        "name", "file_sha256", "batch_sha256"}
                    or entry["name"] != expected_name
                    or entry["name"] in seen_names
                    or not _is_hex64(entry["file_sha256"])
                    or not _is_hex64(entry["batch_sha256"])):
                raise AuditError(f"invalid {scope} batch inventory entry")
            seen_names.add(entry["name"])
            batch_path = _safe_batch_path(
                root, scope, lane, scope_policy["lanes"])
            if (not batch_path.is_file()
                    or file_sha256(batch_path) != entry["file_sha256"]):
                raise AuditError(f"missing or changed {scope} batch file")
            payload = exact.load_json(batch_path)
            if payload.get("batch_sha256") != entry["batch_sha256"]:
                raise AuditError(f"{scope} batch payload hash mismatch")
            validate_batch(
                f"{scope}-{lane}", payload, scope=scope, lane=lane,
                lanes=scope_policy["lanes"], frozen_path=DEFAULT_FROZEN,
                artifact_path=artifacts[scope], artifact_sha256=after[scope],
                auditor_sha256=report["auditor_sha256"],
            )
            aggregated.extend(payload["records"])
        aggregated.sort(key=lambda row: row["index"])
        if ([row["index"] for row in aggregated]
                != list(range(scope_policy["count"]))
                or phase["record_set_sha256"] != exact.payload_sha256(
                    aggregated, omit=())
                or phase["production_record_set_sha256"]
                != exact.payload_sha256(production[scope], omit=())):
            raise AuditError(f"{scope} aggregate record binding mismatch")

    if current:
        current_state_path = _safe_resolve(
            DEFAULT_STATE, "controller state", must_exist=True)
        controller_state = _controller_done(current_state_path)
        if (file_sha256(current_state_path) != controller["file_sha256"]
                or controller_state.get("updated_utc")
                != controller["updated_utc"]):
            raise AuditError("current controller state drift")
        import huang_star_interior as star
        star_path = _safe_resolve(
            HERE / "results" / star_record["path"],
            "star certificate", must_exist=True)
        star.verify_certificate(star_path)
        if file_sha256(star_path) != star_record["file_sha256"]:
            raise AuditError("current star certificate drift")
    return report


def verify_report(path, rerun=False, current=False) -> dict:
    """Verify under one exclusive mesh snapshot of every bound input."""
    report_path = _safe_resolve(
        path, "deep-replay report", must_exist=True)
    frozen_path = _safe_resolve(
        DEFAULT_FROZEN, "frozen manifest", must_exist=True)
    state_path = _safe_resolve(
        DEFAULT_STATE, "controller state", must_exist=True)
    frozen = _verified_frozen(frozen_path)
    artifacts = _artifact_paths()
    star_path = _safe_resolve(
        HERE / "results" / "huang_star_interior.json",
        "star certificate", must_exist=True)
    read_resources = _frozen_read_resources(
        frozen, frozen_path, state_path, artifacts, star_path)
    orchestrator = _orchestrator_module()
    singleton = orchestrator.Singleton(
        "Global\\PerceptronHuangDeepReplayV1")
    singleton.__enter__()
    joined = False
    success = False
    try:
        orchestrator = _mesh_join_and_claim(
            report_path.parent, read_resources)
        joined = True
        verified = _verify_report_unlocked(
            report_path, rerun=rerun, current=current)
        success = True
        return verified
    finally:
        if joined:
            _mesh_bye(
                orchestrator, "done" if success else "offline",
                "Huang deep-replay report verified" if success
                else "Huang deep-replay report verification stopped safely",
            )
        singleton.__exit__(None, None, None)


def _worker_command(args) -> int:
    payload = _batch_payload(
        args.scope, args.lane, args.lanes, args.frozen_manifest,
        args.artifact, args.artifact_sha256, args.auditor_sha256)
    exact = _exact_module()
    orchestrator = _orchestrator_module()
    orchestrator.mesh_command([
        "assert", "--agent", MESH_AGENT,
        "--resource", str(_safe_resolve(
            args.result_file, "isolated worker result")),
    ])
    if file_sha256(__file__) != args.auditor_sha256:
        raise AuditError("auditor source changed before worker publication")
    exact.write_json_atomic(args.result_file, payload, overwrite=False)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("run", "run-if-ready"):
        command = sub.add_parser(name)
        command.add_argument("--frozen-manifest", default=str(DEFAULT_FROZEN))
        command.add_argument("--state", default=str(DEFAULT_STATE))
        command.add_argument("--output", default=str(DEFAULT_OUTPUT))
        command.add_argument("--workers", type=int, default=8)

    worker = sub.add_parser("worker")
    worker.add_argument("--scope", choices=tuple(SCOPE_POLICY), required=True)
    worker.add_argument("--lane", type=int, required=True)
    worker.add_argument("--lanes", type=int, required=True)
    worker.add_argument("--frozen-manifest", required=True)
    worker.add_argument("--artifact", required=True)
    worker.add_argument("--artifact-sha256", required=True)
    worker.add_argument("--auditor-sha256", required=True)
    worker.add_argument("--result-file", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("path", nargs="?", default=str(DEFAULT_OUTPUT))
    verify.add_argument("--rerun-batches", action="store_true")
    verify.add_argument("--current", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "worker":
        return _worker_command(args)
    if args.command == "verify":
        report = verify_report(
            args.path, rerun=args.rerun_batches, current=args.current)
        print(report["report_sha256"])
        return 0
    try:
        run_audit(
            frozen_manifest=args.frozen_manifest,
            state_path=args.state,
            output_path=args.output,
            workers=args.workers,
        )
        return 0
    except NotReady as exc:
        if args.command == "run-if-ready":
            return 75
        raise SystemExit(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
