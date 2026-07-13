import copy
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import sys
import tempfile
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import block3a_assemble as b3a  # noqa: E402


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def utc_text(value):
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


class SyntheticBlock3aEvidence:
    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.inputs = self.root / "inputs"
        self.results = self.root / "results"
        self.evidence = self.results / "block3a_evidence"
        self.output = self.results / "block3a.log"
        self.certificate = self.results / "block3a_certificate.json"
        self.inputs.mkdir(parents=True)
        self.schedule = b3a.canonical_schedule()
        self._write_logs()
        self._write_tools()
        self._write_manifests()
        self._write_attestation_and_status()
        self._write_provenance()

    def path(self, name):
        return self.inputs / name

    @staticmethod
    def line(row):
        return (
            f"PASS {row['kind']} [{row['tau_lo']},{row['tau_hi']}] "
            "lam=n=1 val=[-1.0 +/- 1e-10] (0.0s)\n"
        )

    def _write_logs(self):
        self.prefix_raw = "".join(
            self.line(row) for row in self.schedule[:b3a.PREFIX_COUNT]
        ).encode("utf-8")
        self.local_raw = "".join(
            self.line(row)
            for row in self.schedule[b3a.PREFIX_COUNT:b3a.LOCAL_MANIFEST_COUNT]
        ).encode("utf-8")
        self.remote_raw = "".join(
            self.line(row)
            for row in reversed(self.schedule[b3a.LOCAL_MANIFEST_COUNT:])
        ).encode("utf-8")
        self.path("local_prefix.log").write_bytes(self.prefix_raw)
        self.path("block3a_local_fill.log").write_bytes(self.local_raw)
        self.path("block3a_azure_tail.log").write_bytes(self.remote_raw)

    def _write_tools(self):
        self.path("manifest_runner.py").write_bytes(
            b"# synthetic pinned manifest runner\n"
        )
        self.path("watch_local_cutover.py").write_bytes(
            b"# synthetic pinned cutover watcher\n"
        )
        self.runner_sha = b3a.file_sha256(self.path("manifest_runner.py"))
        self.watcher_sha = b3a.file_sha256(self.path("watch_local_cutover.py"))

    def _json(self, name, value):
        self.path(name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _write_manifests(self):
        common = {
            "created_utc": "2026-07-10T07:29:07+00:00",
            "schedule_total": b3a.SCHEDULE_COUNT,
            "schema": 1,
            "source_sha256": b3a.EXPECTED_PROOF_SOURCE_SHA256,
        }
        local = {
            **common,
            "description": "synthetic local schedule",
            "entries": self.schedule[:b3a.LOCAL_MANIFEST_COUNT],
        }
        remote = {
            **common,
            "description": "synthetic reverse remote tail",
            "entries": list(reversed(self.schedule[b3a.LOCAL_MANIFEST_COUNT:])),
        }
        self._json("manifest_local_under235.json", local)
        self._json("manifest_remote_tail.json", remote)
        self.local_manifest_sha = b3a.file_sha256(
            self.path("manifest_local_under235.json")
        )
        self.remote_manifest_sha = b3a.file_sha256(
            self.path("manifest_remote_tail.json")
        )

    def _write_attestation_and_status(self):
        source_rows = {}
        latest = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        for name, path in b3a._proof_source_paths().items():
            modified = dt.datetime.fromtimestamp(
                path.stat().st_mtime, tz=dt.timezone.utc
            )
            latest = max(latest, modified)
            source_rows[name] = {
                "length": path.stat().st_size,
                "mtime_utc": utc_text(modified),
                "sha256": b3a.EXPECTED_PROOF_SOURCE_SHA256[name],
            }
        created = latest + dt.timedelta(hours=1)
        captured = created + dt.timedelta(hours=2)
        observed = "".join(
            self.line(row) for row in self.schedule[:b3a.ATTESTED_PREFIX_COUNT]
        ).encode("utf-8")
        self.pid = 4242
        executable = str(pathlib.Path(sys.executable).resolve())
        attestation = {
            "capture": {
                "captured_at_utc": utc_text(captured),
                "method": "synthetic live process inspection",
                "process_alive": True,
                "worker_children": 24,
            },
            "kind": "block3a_original_process_attestation",
            "live_log_observation": {
                "fail_lines": 0,
                "length": len(observed),
                "pass_lines": b3a.ATTESTED_PREFIX_COUNT,
                "path": str(self.root / "block3a.log"),
                "sha256": sha_bytes(observed),
            },
            "process": {
                "affinity_mask": "0xFF",
                "command_line": (
                    f'"{executable}" verification/block3a_grid.py 24 '
                ),
                "creation_unix_milliseconds": int(created.timestamp() * 1000),
                "executable": executable,
                "parent_pid_at_capture": 123,
                "pid": self.pid,
                "priority_class": "BelowNormal",
                "reported_launch_workdir": str(b3a.PROOF_ROOT),
            },
            "runtime": {
                "executable_sha256": b3a.file_sha256(executable),
                "flint": b3a.EXPECTED_FLINT,
                "implementation": platform.python_implementation(),
                "python": platform.python_version(),
                "python_flint": b3a.EXPECTED_PYTHON_FLINT,
            },
            "schema_version": 1,
            "source_files": source_rows,
        }
        self._json("original_process_attestation.json", attestation)
        self.attestation_sha = b3a.file_sha256(
            self.path("original_process_attestation.json")
        )
        status = (
            f"2026-07-10 10:00:00 armed pid={self.pid} threshold=187\n"
            "2026-07-10 11:00:00 cutover triggered at 187 certified cells\n"
            "2026-07-10 11:00:01 suspended parent and 25 descendants\n"
            f"2026-07-10 11:00:01 snapshot cells=187 sha256={sha_bytes(self.prefix_raw)}\n"
            "2026-07-10 12:00:00 local fill runner exited rc=0\n"
        )
        self.path("cutover_status.log").write_bytes(status.encode("utf-8"))

    def _provenance(self, *, manifest, output, cells, preexisting,
                    completed, workers, python, host):
        return {
            "completed_this_run": completed,
            "covered_manifest_cells": cells,
            "elapsed_seconds": 1,
            "fails_this_run": 0,
            "flint": b3a.EXPECTED_FLINT,
            "host": host,
            "manifest": str(manifest),
            "manifest_cells": cells,
            "manifest_sha256": b3a.file_sha256(manifest),
            "output": str(output),
            "platform": "synthetic-platform",
            "preexisting_cells": preexisting,
            "python": python + " synthetic-build",
            "python_flint": b3a.EXPECTED_PYTHON_FLINT,
            "runner_sha256": self.runner_sha,
            "schema": 1,
            "source_sha256": b3a.EXPECTED_PROOF_SOURCE_SHA256,
            "started_utc": "2026-07-10T12:00:00+00:00",
            "state": "complete",
            "updated_utc": "2026-07-10T13:00:00+00:00",
            "workers": workers,
        }

    def _write_provenance(self):
        local = self._provenance(
            manifest=self.path("manifest_local_under235.json"),
            output=self.path("block3a_local_fill.log"),
            cells=b3a.LOCAL_MANIFEST_COUNT,
            preexisting=b3a.PREFIX_COUNT,
            completed=b3a.LOCAL_MANIFEST_COUNT - b3a.PREFIX_COUNT,
            workers=24,
            python=platform.python_version(),
            host="local-prefix-host",
        )
        remote = self._provenance(
            manifest=self.path("manifest_remote_tail.json"),
            output=self.path("block3a_azure_tail.log"),
            cells=b3a.REMOTE_TAIL_COUNT,
            preexisting=0,
            completed=b3a.REMOTE_TAIL_COUNT,
            workers=4,
            python="3.12.3",
            host="fallback-local-tail-host",
        )
        self._json("block3a_local_fill_provenance.json", local)
        self._json("block3a_azure_tail_provenance.json", remote)

    def trust_overrides(self):
        return {
            "EXPECTED_RUNNER_SHA256": self.runner_sha,
            "EXPECTED_WATCHER_SHA256": self.watcher_sha,
            "EXPECTED_PREFIX_SHA256": sha_bytes(self.prefix_raw),
            "EXPECTED_ATTESTATION_SHA256": self.attestation_sha,
            "EXPECTED_LOCAL_MANIFEST_SHA256": self.local_manifest_sha,
            "EXPECTED_REMOTE_MANIFEST_SHA256": self.remote_manifest_sha,
        }

    def assemble_kwargs(self):
        return {
            "prefix_log": self.path("local_prefix.log"),
            "attestation": self.path("original_process_attestation.json"),
            "cutover_status": self.path("cutover_status.log"),
            "local_manifest": self.path("manifest_local_under235.json"),
            "local_log": self.path("block3a_local_fill.log"),
            "local_provenance": self.path("block3a_local_fill_provenance.json"),
            "remote_manifest": self.path("manifest_remote_tail.json"),
            "remote_log": self.path("block3a_azure_tail.log"),
            "remote_provenance": self.path("block3a_azure_tail_provenance.json"),
            "runner_source": self.path("manifest_runner.py"),
            "watcher_source": self.path("watch_local_cutover.py"),
            "evidence_dir": self.evidence,
            "output_log": self.output,
            "certificate": self.certificate,
        }


class Block3aAssemblerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = SyntheticBlock3aEvidence(self.temp.name)
        self.trust = mock.patch.multiple(b3a, **self.fixture.trust_overrides())
        self.trust.start()
        self.addCleanup(self.trust.stop)
        self.boundary = {
            "precision_bits": b3a.BOUNDARY_PRECISION_BITS,
            "checks": {
                "synthetic_positive_margin": {
                    "sign": ">0",
                    "difference": {
                        "format": "arb-midrad10-v1",
                        "mid10": "2",
                        "rad10": "1",
                        "exp10": -3,
                        "digits": 60,
                    },
                }
            },
        }
        self.boundary_patch = mock.patch.object(
            b3a, "compute_boundary_pins", return_value=self.boundary
        )
        self.boundary_patch.start()
        self.addCleanup(self.boundary_patch.stop)

    def assemble(self):
        return b3a.assemble(**self.fixture.assemble_kwargs())

    def test_synthetic_end_to_end_certificate(self):
        certificate = self.assemble()
        verified = b3a.verify_certificate(self.fixture.certificate)
        self.assertEqual(verified, certificate)
        self.assertEqual(certificate["verdict"], "ALL PASS")
        self.assertEqual([x["line_count"] for x in certificate["shards"]],
                         [187, 48, 12])
        self.assertEqual(certificate["shards"][2]["role"], "reverse_tail")
        self.assertEqual(certificate["shards"][2]["runtime"]["host"],
                         "fallback-local-tail-host")
        lines = self.fixture.output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), b3a.SCHEDULE_COUNT)
        self.assertIn("[0.24,0.241]", lines[0])
        self.assertIn("[-1,-0.93]", lines[-1])

    def test_duplicate_shard_row_fails_closed(self):
        path = self.fixture.path("block3a_local_fill.log")
        path.write_bytes(path.read_bytes() + self.fixture.local_raw.splitlines(True)[0])
        with self.assertRaisesRegex(ValueError, "duplicate cell"):
            self.assemble()
        self.assertFalse(self.fixture.evidence.exists())
        self.assertFalse(self.fixture.output.exists())
        self.assertFalse(self.fixture.certificate.exists())

    def test_destination_alias_fails_before_publication(self):
        kwargs = self.fixture.assemble_kwargs()
        kwargs["output_log"] = kwargs["certificate"]
        with self.assertRaisesRegex(ValueError, "destinations alias"):
            b3a.assemble(**kwargs)
        self.assertFalse(self.fixture.evidence.exists())

    def test_existing_destination_is_never_clobbered(self):
        self.fixture.output.parent.mkdir(parents=True)
        self.fixture.output.write_bytes(b"owner data\n")
        with self.assertRaises(FileExistsError):
            self.assemble()
        self.assertEqual(self.fixture.output.read_bytes(), b"owner data\n")
        self.assertFalse(self.fixture.evidence.exists())

    def test_malformed_arb_leaf_text_fails_closed(self):
        path = self.fixture.path("block3a_local_fill.log")
        raw = path.read_bytes().replace(
            b"val=[-1.0 +/- 1e-10]", b"val=not-an-arb-ball", 1
        )
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "malformed Arb leaf"):
            self.assemble()

    def test_cutover_event_reordering_fails_closed(self):
        path = self.fixture.path("cutover_status.log")
        lines = path.read_bytes().splitlines(keepends=True)
        lines[1], lines[2] = lines[2], lines[1]
        path.write_bytes(b"".join(lines))
        with self.assertRaisesRegex(ValueError, "out of order"):
            self.assemble()

    def test_prefix_byte_chain_tamper_fails_even_with_rehashed_attestation(self):
        path = self.fixture.path("original_process_attestation.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["live_log_observation"]["sha256"] = "0" * 64
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        with mock.patch.object(
                b3a, "EXPECTED_ATTESTATION_SHA256", b3a.file_sha256(path)):
            with self.assertRaisesRegex(ValueError, "byte prefix"):
                self.assemble()

    def test_runtime_provenance_tamper_fails_closed(self):
        path = self.fixture.path("block3a_azure_tail_provenance.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["flint"] = "0.0.0"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "runtime mismatch"):
            self.assemble()

    def test_provenance_output_name_mismatch_fails_closed(self):
        path = self.fixture.path("block3a_azure_tail_provenance.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["output"] = "wrong-tail.log"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "output name mismatch"):
            self.assemble()

    def test_resumed_reverse_tail_accounting_is_valid(self):
        path = self.fixture.path("block3a_azure_tail_provenance.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["preexisting_cells"] = 5
        data["completed_this_run"] = 7
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        certificate = self.assemble()
        tail_runtime = certificate["shards"][2]["runtime"]
        self.assertEqual(tail_runtime["preexisting_cells"], 5)
        self.assertEqual(tail_runtime["completed_this_run"], 7)

    def test_manifest_schedule_tamper_fails_after_raw_repin(self):
        path = self.fixture.path("manifest_local_under235.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["entries"][0]["tau_hi"] = "0.242"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        with mock.patch.object(
                b3a, "EXPECTED_LOCAL_MANIFEST_SHA256", b3a.file_sha256(path)):
            with self.assertRaisesRegex(ValueError, "schedule mismatch"):
                self.assemble()

    def test_raw_evidence_tamper_invalidates_certificate(self):
        self.assemble()
        staged = self.fixture.evidence / "block3a_azure_tail.log"
        staged.write_bytes(staged.read_bytes().replace(b"PASS", b"FAIL", 1))
        with self.assertRaises(ValueError):
            b3a.verify_certificate(self.fixture.certificate)

    def test_rehashed_boundary_tamper_is_recomputed(self):
        self.assemble()
        data = b3a.load_certificate(self.fixture.certificate)
        changed = copy.deepcopy(data)
        changed["boundary_pins"]["checks"]["synthetic_positive_margin"][
            "difference"
        ]["mid10"] = "3"
        changed["certificate_sha256"] = b3a.payload_sha256(changed)
        self.fixture.certificate.write_bytes(
            b3a.canonical_json_bytes(changed) + b"\n"
        )
        with self.assertRaisesRegex(ValueError, "raw evidence mismatch"):
            b3a.verify_certificate(self.fixture.certificate)

    def test_certificate_payload_hash_tamper_fails(self):
        self.assemble()
        data = b3a.load_certificate(self.fixture.certificate)
        data["verdict"] = "FAIL"
        self.fixture.certificate.write_bytes(b3a.canonical_json_bytes(data) + b"\n")
        with self.assertRaises(ValueError):
            b3a.verify_certificate(self.fixture.certificate)

    def test_evidence_json_rejects_duplicate_and_nonfinite_values(self):
        path = pathlib.Path(self.temp.name) / "bad.json"
        for raw in (b'{"x":1,"x":2}\n', b'{"x":NaN}\n'):
            path.write_bytes(raw)
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    b3a.load_evidence_json(path)


if __name__ == "__main__":
    unittest.main()
