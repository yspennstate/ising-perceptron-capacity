import copy
import datetime as dt
import hashlib
import json
import pathlib
import tempfile
import unittest

import release_verification as release


class ReleaseVerificationReceiptTests(unittest.TestCase):
    def make_receipt(self, root, result=b"proof\n"):
        proof = pathlib.Path(root) / "perceptron"
        results = proof / "verification" / "results"
        results.mkdir(parents=True)
        artifact = results / "artifact.json"
        artifact.write_bytes(result)
        commands = []
        for name, arguments, marker, execution_scope in release.COMMANDS:
            output = (f"Ran {release.MINIMUM_UNIT_TESTS} tests in 0.001s\n\n"
                      if name == "unit_tests" else "") + marker + "\n"
            commands.append({
                "name": name,
                "argv": ["/pinned/python", *arguments],
                "returncode": 0,
                "elapsed_milliseconds": 1,
                "output": output,
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "required_marker": marker,
                "execution_scope": execution_scope,
                "git_head": ("a" * 40
                             if execution_scope
                             == release.IMMUTABLE_ARCHIVE_GIT_SCOPE
                             else None),
            })
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        payload = {
            "schema_version": release.SCHEMA_VERSION,
            "kind": release.KIND,
            "source_commit": "a" * 40,
            "started_at": now,
            "finished_at": now,
            "runtime": {
                "runtime_schema_version": 2,
                "host": "proof-host",
                "python": "3.12.3",
                "executable": "/pinned/python",
                "python_executable_sha256": release.EXPECTED_EXECUTABLE_SHA256,
                "python_flint_root": "/pinned/flint",
                "python_flint_tree_sha256": release.EXPECTED_FLINT_TREE_SHA256,
                "python_flint": "0.9.0",
                "flint": "3.6.0",
                "precision_bits": 50,
                "workers": 3,
            },
            "commands": commands,
            "proof_tree": release._tree_identity(proof),
            "result_sha256": {
                "verification/results/artifact.json": {
                    "sha256": hashlib.sha256(result).hexdigest(),
                    "size": len(result),
                },
            },
            "verdict": release.VERDICT,
            "attestation_scope": release.ATTESTATION_SCOPE,
        }
        payload["receipt_sha256"] = release._payload_sha256(payload)
        receipt = pathlib.Path(root) / "receipt.json"
        receipt.write_text(json.dumps(payload), encoding="utf-8")
        return proof, receipt, payload

    def test_roundtrip_binds_result_closure(self):
        with tempfile.TemporaryDirectory() as td:
            proof, receipt, payload = self.make_receipt(td)
            self.assertEqual(
                release.check_receipt(receipt, "a" * 40, proof), payload)
            (proof / "verification" / "results" / "artifact.json").write_text(
                "changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "proof tree"):
                release.check_receipt(receipt, "a" * 40, proof)
            payload["proof_tree"] = release._tree_identity(proof)
            payload["receipt_sha256"] = release._payload_sha256(payload)
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "result closure"):
                release.check_receipt(receipt, "a" * 40, proof)

    def test_rehashed_command_without_required_marker_fails(self):
        with tempfile.TemporaryDirectory() as td:
            proof, receipt, payload = self.make_receipt(td)
            changed = copy.deepcopy(payload)
            changed["commands"][2]["output"] = "not the verdict\n"
            changed["commands"][2]["output_sha256"] = hashlib.sha256(
                changed["commands"][2]["output"].encode()).hexdigest()
            changed["receipt_sha256"] = release._payload_sha256(changed)
            receipt.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verify_all"):
                release.check_receipt(receipt, "a" * 40, proof)

    def test_zero_or_missing_unit_test_inventory_fails(self):
        with tempfile.TemporaryDirectory() as td:
            proof, receipt, payload = self.make_receipt(td)
            for output in ("Ran 0 tests in 0.001s\n\nOK\n", "OK\n"):
                with self.subTest(output=output):
                    changed = copy.deepcopy(payload)
                    changed["commands"][0]["output"] = output
                    changed["commands"][0]["output_sha256"] = hashlib.sha256(
                        output.encode()).hexdigest()
                    changed["receipt_sha256"] = release._payload_sha256(changed)
                    receipt.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "unit_tests"):
                        release.check_receipt(receipt, "a" * 40, proof)

    def test_receipt_hash_and_duplicate_keys_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            proof, receipt, payload = self.make_receipt(td)
            payload["verdict"] = "PASS"
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "envelope"):
                release.check_receipt(receipt, "a" * 40, proof)
            receipt.write_text('{"schema_version":1,"schema_version":1}',
                               encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                release.check_receipt(receipt, "a" * 40, proof)

    def test_non_plain_integer_fields_and_malformed_commit_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            proof, receipt, payload = self.make_receipt(td)
            mutations = (
                ("schema", lambda value: value.__setitem__(
                    "schema_version", True)),
                ("runtime schema", lambda value: value["runtime"].__setitem__(
                    "runtime_schema_version", True)),
                ("precision", lambda value: value["runtime"].__setitem__(
                    "precision_bits", 50.0)),
                ("return code", lambda value: value["commands"][0].__setitem__(
                    "returncode", False)),
                ("elapsed", lambda value: value["commands"][0].__setitem__(
                    "elapsed_milliseconds", False)),
                ("tree count", lambda value: value["proof_tree"].__setitem__(
                    "file_count", True)),
            )
            for label, mutate in mutations:
                with self.subTest(label=label):
                    changed = copy.deepcopy(payload)
                    mutate(changed)
                    changed["receipt_sha256"] = release._payload_sha256(changed)
                    receipt.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        release.check_receipt(receipt, "a" * 40, proof)
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lowercase SHA-1"):
                release.check_receipt(receipt, "not-a-commit", proof)

    def test_worker_allowance_is_capped(self):
        with tempfile.TemporaryDirectory() as td:
            proof, receipt, payload = self.make_receipt(td)
            payload["runtime"]["workers"] = release.MAX_WORKERS + 1
            payload["receipt_sha256"] = release._payload_sha256(payload)
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime policy"):
                release.check_receipt(receipt, "a" * 40, proof)

    def test_command_execution_scopes_are_exact_and_routed(self):
        scopes = tuple(row[3] for row in release.COMMANDS)
        self.assertEqual(scopes, (
            release.IMMUTABLE_ARCHIVE_GIT_SCOPE,
            release.IMMUTABLE_ARCHIVE_SCOPE,
            release.IMMUTABLE_ARCHIVE_SCOPE,
        ))
        base = {"PYTHONDONTWRITEBYTECODE": "1"}
        archive_root = pathlib.Path("/immutable/tree/perceptron")
        with tempfile.TemporaryDirectory() as git_td:
            git_dir = pathlib.Path(git_td)
            unit_env = release._command_environment(
                base, release.IMMUTABLE_ARCHIVE_GIT_SCOPE,
                archive_root, git_dir)
            self.assertEqual(unit_env["GIT_DIR"], str(git_dir.resolve()))
        self.assertEqual(
            unit_env["GIT_WORK_TREE"], str(archive_root.resolve().parent))
        self.assertEqual(unit_env["GIT_NO_REPLACE_OBJECTS"], "1")
        self.assertEqual(unit_env["GIT_OPTIONAL_LOCKS"], "0")
        self.assertEqual(release._command_environment(
            base, release.IMMUTABLE_ARCHIVE_SCOPE,
            archive_root, git_dir), base)
        with self.assertRaisesRegex(ValueError, "execution scope"):
            release._command_environment(
                base, "untrusted", archive_root, git_dir)

        with tempfile.TemporaryDirectory() as td:
            proof, receipt, payload = self.make_receipt(td)
            payload["commands"][0]["execution_scope"] = (
                release.IMMUTABLE_ARCHIVE_SCOPE)
            payload["receipt_sha256"] = release._payload_sha256(payload)
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unit_tests"):
                release.check_receipt(receipt, "a" * 40, proof)

        with tempfile.TemporaryDirectory() as td:
            proof, receipt, payload = self.make_receipt(td)
            payload["commands"][0]["git_head"] = "b" * 40
            payload["receipt_sha256"] = release._payload_sha256(payload)
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unit_tests"):
                release.check_receipt(receipt, "a" * 40, proof)


if __name__ == "__main__":
    unittest.main()
