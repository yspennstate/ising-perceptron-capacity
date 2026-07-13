import datetime as dt
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import proof_orchestrator as po  # noqa: E402


class CanonicalStateTests(unittest.TestCase):
    def test_atomic_canonical_roundtrip_and_float_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "state.json"
            value = {"phase": "WAIT", "details": {"count": 3}}
            po.atomic_write_json(path, value)
            self.assertEqual(po.load_canonical_json(path), value)
            with self.assertRaises(TypeError):
                po.atomic_write_json(path, {"bad": 0.5})

    def test_archive_is_hash_named_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            source = root / "old.json"
            source.write_bytes(b"old evidence\n")
            digest = po.file_sha256(source)
            archived = po.archive_existing(source, root / "archive")
            self.assertFalse(source.exists())
            self.assertEqual(archived.name, f"old.json.{digest}.archive")
            self.assertEqual(archived.read_bytes(), b"old evidence\n")

    def test_atomic_publish_keeps_canonical_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            source = root / "new.json"
            destination = root / "canonical.json"
            source.write_bytes(b"new\n")
            destination.write_bytes(b"old\n")
            old_hash = po.file_sha256(destination)
            with mock.patch.object(po.os, "replace", side_effect=OSError("boom")):
                with self.assertRaisesRegex(OSError, "boom"):
                    po.atomic_publish(source, destination, root / "archive")
            self.assertEqual(destination.read_bytes(), b"old\n")
            archived = root / "archive" / (
                f"canonical.json.{old_hash}.archive")
            self.assertEqual(archived.read_bytes(), b"old\n")

    def test_archive_tree_preserves_every_entry(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            tree = root / "records"
            (tree / "nested").mkdir(parents=True)
            (tree / "a.json").write_bytes(b"a\n")
            (tree / "nested" / "b.json").write_bytes(b"b\n")
            digest = po.tree_sha256(tree)
            archived = po.archive_tree(tree, root / "archive")
            self.assertFalse(tree.exists())
            self.assertIn(digest, archived.name)
            self.assertEqual((archived / "a.json").read_bytes(), b"a\n")
            self.assertEqual(
                (archived / "nested" / "b.json").read_bytes(), b"b\n")


class FrozenSourceTests(unittest.TestCase):
    def test_freeze_verifies_exact_bytes_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            source = root / "proof.py"
            source.write_text("value = 1\n", encoding="utf-8")
            paths = {"proof.py": source}
            payload = po.frozen_payload(paths)
            manifest = root / "frozen.json"
            po.atomic_write_json(manifest, payload, overwrite=False)
            checked = po.verify_frozen_manifest(manifest, paths=paths)
            self.assertEqual(checked["source_sha256"]["proof.py"],
                             po.file_sha256(source))
            source.write_text("value = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(po.OrchestratorError, "source drift"):
                po.verify_frozen_manifest(manifest, paths=paths)

    def test_production_freeze_excludes_retired_split_run_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            payload = po.frozen_payload(recovery_dir=root / "missing-retired")
            self.assertIn("block1_gardner.py", payload["source_sha256"])
            self.assertIn("block2_near_one.py", payload["source_sha256"])
            self.assertIn("block3a_run.py", payload["source_sha256"])
            self.assertIn("block3a_singlerun.py", payload["source_sha256"])
            self.assertIn("huang_star_interior.py", payload["source_sha256"])
            self.assertIsNone(payload["recovery_root"])
            self.assertEqual(payload["recovery_sha256"], {})
            manifest = root / "frozen.json"
            po.atomic_write_json(manifest, payload, overwrite=False)
            po.verify_frozen_manifest(
                manifest, recovery_dir=root / "still-missing-retired")


class CommandPlanTests(unittest.TestCase):
    def test_plan_has_exact_governance_and_lane_policy(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            settings = po.Settings(
                run_dir=root / "run", recovery_dir=root / "recovery").normalized()
            rows = {row["phase"]: row["argv"]
                    for row in po.command_plan(settings)}
            local = rows["LOCAL_FILL_RECOVERY"]
            self.assertEqual(local[local.index("--workers") + 1], "24")
            self.assertEqual(local[local.index("--affinity") + 1],
                             "0,1,2,3,4,5,6,7")
            star = rows["HUANG_STAR_INTERIOR"]
            self.assertIn("huang_star_interior.py", star[2])
            self.assertEqual(star[-2], "--output")
            self.assertTrue(star[-1].endswith("huang_star_interior.json"))
            self.assertEqual(rows["HUANG_REGION1"][-1], "8")
            bundle = rows["HUANG_BUNDLE"]
            self.assertIn("huang_sweep_verify.py", bundle[2])
            self.assertEqual(bundle[3], "bundle")
            self.assertTrue(bundle[bundle.index("--output") + 1].endswith(
                "huang_bundle.json"))
            for phase in ("BLOCK3BC_AUX_ELL", "BLOCK3BC_AUX_K",
                          "BLOCK3BC_REPLAY_B_POS", "BLOCK3BC_REPLAY_B_NEG",
                          "BLOCK3BC_REPLAY_C"):
                argv = rows[phase]
                self.assertEqual(argv[argv.index("--workers") + 1], "8")
                self.assertEqual(argv[argv.index("--lane") + 1], "0")
                self.assertEqual(argv[argv.index("--lanes") + 1], "1")
                self.assertEqual(argv[argv.index("--timeout-seconds") + 1],
                                 "21600")
                self.assertEqual(argv[argv.index("--retries") + 1], "2")
            tail = rows["TAIL_LOCAL_FALLBACK"]
            self.assertEqual(tail[tail.index("--workers") + 1], "4")

    def test_dry_run_child_never_calls_popen(self):
        with tempfile.TemporaryDirectory() as td:
            settings = po.Settings(
                run_dir=pathlib.Path(td) / "run",
                recovery_dir=pathlib.Path(td) / "recovery").normalized()
            orchestrator = po.Orchestrator(settings, dry_run=True)
            with mock.patch.object(po.subprocess, "Popen") as popen:
                self.assertEqual(
                    orchestrator.run_child("SYNTHETIC", ["never", "run"]), 0)
                popen.assert_not_called()

    def test_mesh_claims_are_staged_and_exclusive(self):
        with tempfile.TemporaryDirectory() as td:
            settings = po.Settings(
                run_dir=pathlib.Path(td) / "run",
                recovery_dir=pathlib.Path(td) / "recovery").normalized()
            with mock.patch.object(po, "mesh_command", return_value={} ) as call:
                lease = po.MeshLease(settings, agent="test-proof-controller")
                lease.join_and_claim()
                lease.claim_recovery()
                lease.claim_remote_tail()
            commands = [row.args[0] for row in call.call_args_list]
            claims = [row for row in commands if row[0] == "claim"]
            self.assertEqual(len(claims), 3)
            self.assertIn("exclusive", claims[0])
            self.assertNotIn(str(settings.recovery_dir), claims[0])
            self.assertIn(
                str(settings.recovery_dir / "block3a_local_fill.log"),
                claims[1])
            self.assertNotIn(str(settings.recovery_dir), claims[1])
            self.assertIn(po.LOCAL_CUTOVER_RESOURCE, claims[1])
            self.assertIn(
                str(settings.recovery_dir / "block3a_azure_tail.log"),
                claims[2])
            self.assertIn(
                str(settings.recovery_dir
                    / "block3a_azure_tail_provenance.json"), claims[2])

    def test_windows_job_wrapper_waits_for_gate(self):
        encoded = po.wrapped_child_command(["proof", "arg"], pathlib.Path.cwd())[-1]
        fake_stdin = types.SimpleNamespace(buffer=io.BytesIO(b""))
        with mock.patch.object(po.sys, "stdin", fake_stdin), \
                mock.patch.object(po.subprocess, "Popen") as popen:
            self.assertEqual(po.job_child_main(encoded), 125)
            popen.assert_not_called()


class TailClearanceTests(unittest.TestCase):
    def _clearance(self, manifest, *, expires_delta=3600):
        now = dt.datetime.now(dt.timezone.utc)
        payload = {
            "schema_version": 1,
            "kind": "block3a_tail_fallback_clearance",
            "decision": "remote_tail_inactive_local_fallback_exclusive",
            "lease_id": "mesh-lease-123",
            "controller": "human-reviewed-state",
            "resource": po.TAIL_CONTROL_RESOURCE,
            "issued_utc": (now - dt.timedelta(seconds=5)).isoformat(),
            "expires_utc": (now + dt.timedelta(
                seconds=expires_delta)).isoformat(),
            "remote_manifest_sha256": po.file_sha256(manifest),
        }
        payload["clearance_sha256"] = po.payload_sha256(
            payload, "clearance_sha256")
        return payload

    def test_clearance_is_machine_neutral_and_exact_manifest_bound(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            recovery = root / "recovery"
            recovery.mkdir()
            remote = recovery / "manifest_remote_tail.json"
            remote.write_text("{}\n", encoding="utf-8")
            clearance = root / "clearance.json"
            payload = self._clearance(remote)
            po.atomic_write_json(clearance, payload)
            settings = po.Settings(
                run_dir=root / "run", recovery_dir=recovery,
                tail_clearance=clearance).normalized()
            checker = mock.Mock(return_value={})
            result = po.Orchestrator(
                settings, external_claim_checker=checker
            ).validate_tail_clearance()
            self.assertEqual(result["controller"], "human-reviewed-state")
            self.assertNotIn("host", result)
            checker.assert_called_once_with(
                "human-reviewed-state", po.TAIL_CONTROL_RESOURCE,
                "mesh-lease-123")

    def test_expired_clearance_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            recovery = root / "recovery"
            recovery.mkdir()
            remote = recovery / "manifest_remote_tail.json"
            remote.write_text("{}\n", encoding="utf-8")
            clearance = root / "clearance.json"
            payload = self._clearance(remote, expires_delta=-1)
            po.atomic_write_json(clearance, payload)
            settings = po.Settings(
                run_dir=root / "run", recovery_dir=recovery,
                tail_clearance=clearance).normalized()
            with self.assertRaisesRegex(po.OrchestratorError, "not currently"):
                po.Orchestrator(settings).validate_tail_clearance()

    def test_clearance_without_exact_live_mesh_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            recovery = root / "recovery"
            recovery.mkdir()
            remote = recovery / "manifest_remote_tail.json"
            remote.write_text("{}\n", encoding="utf-8")
            clearance = root / "clearance.json"
            po.atomic_write_json(clearance, self._clearance(remote))
            settings = po.Settings(
                run_dir=root / "run", recovery_dir=recovery,
                tail_clearance=clearance).normalized()
            checker = mock.Mock(side_effect=po.OrchestratorError("no lease"))
            with self.assertRaisesRegex(po.OrchestratorError, "no lease"):
                po.Orchestrator(
                    settings, external_claim_checker=checker
                ).validate_tail_clearance()

    def test_tail_pair_is_not_trusted_by_existence(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            recovery = root / "recovery"
            recovery.mkdir()
            (recovery / "block3a_azure_tail.log").write_text(
                "PASS forged\n", encoding="utf-8")
            (recovery / "block3a_azure_tail_provenance.json").write_text(
                "{}\n", encoding="utf-8")
            settings = po.Settings(
                run_dir=root / "run", recovery_dir=recovery).normalized()
            self.assertIsNone(po.Orchestrator(settings).tail_paths())

    def test_fallback_receipt_is_bound_to_exact_evidence_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            recovery = root / "recovery"
            run = root / "run"
            recovery.mkdir()
            run.mkdir()
            manifest = recovery / "manifest_remote_tail.json"
            manifest.write_text("{}\n", encoding="utf-8")
            paths = (run / "block3a_tail_fallback.log",
                     run / "block3a_tail_fallback_provenance.json")
            paths[0].write_bytes(b"evidence\n")
            paths[1].write_bytes(b"{}\n")
            orchestrator = po.Orchestrator(po.Settings(
                run_dir=run, recovery_dir=recovery))
            orchestrator._write_fallback_receipt(
                {"clearance_sha256": "a" * 64}, paths)
            self.assertTrue(orchestrator._fallback_receipt_valid(paths))
            paths[0].write_bytes(b"changed\n")
            self.assertFalse(orchestrator._fallback_receipt_valid(paths))


class Block3bcResumeTests(unittest.TestCase):
    def test_corrupt_immutable_aux_record_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            output = pathlib.Path(td) / "ell.json"
            records = pathlib.Path(str(output) + ".records")
            records.mkdir()
            (records / "ellp-000.json").write_text("{}\n", encoding="utf-8")
            self.assertFalse(
                po.Orchestrator._aux_resume_records_valid("ell_prime", output))


class RegionResumeTests(unittest.TestCase):
    def test_star_interior_resume_requires_exact_replay(self):
        import huang_star_interior as interior

        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "star.json"
            interior.write_certificate(path)
            self.assertTrue(po.Orchestrator._star_interior_valid(path))
            payload = json.loads(path.read_text("utf-8"))
            payload["policy"]["star_radius"] = "0.011"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(po.Orchestrator._star_interior_valid(path))

    def test_region1_resume_forwards_the_artifact_path_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "region.json"
            path.write_text("{}\n", encoding="utf-8")
            calls = []

            def validate(candidate):
                calls.append(candidate)
                return True, "ok"

            verifier = types.SimpleNamespace(
                validate_region1_manifest=validate)
            with mock.patch.dict(
                    sys.modules, {"verify_all": verifier}):
                self.assertTrue(po.Orchestrator._region1_valid(path))
            self.assertEqual(calls, [path])

            verifier = types.SimpleNamespace(
                validate_region1_manifest=lambda candidate: (_ for _ in ())
                .throw(ValueError("bad certificate")))
            with mock.patch.dict(sys.modules, {"verify_all": verifier}):
                self.assertFalse(po.Orchestrator._region1_valid(path))

    def test_huang_bundle_requires_exact_four_artifact_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "huang_bundle.json"
            path.write_text("{}\n", encoding="utf-8")
            check = mock.Mock(return_value=({}, ({}, {}, {}, [], [])))
            verifier = types.SimpleNamespace(verify_bundle=check)
            with mock.patch.dict(
                    sys.modules, {"huang_sweep_verify": verifier}):
                self.assertTrue(po.Orchestrator._huang_bundle_valid(path))
            check.assert_called_once_with(path)

            verifier = types.SimpleNamespace(
                verify_bundle=mock.Mock(side_effect=ValueError("bad pair")))
            with mock.patch.dict(
                    sys.modules, {"huang_sweep_verify": verifier}):
                self.assertFalse(po.Orchestrator._huang_bundle_valid(path))

    def test_individual_huang_artifacts_cannot_bypass_bundle_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            run = root / "run"
            recovery = root / "recovery"
            results = root / "results"
            run.mkdir()
            recovery.mkdir()
            results.mkdir()
            (results / "huang_bundle.json").write_text(
                "{}\n", encoding="utf-8")
            orchestrator = po.Orchestrator(po.Settings(
                run_dir=run, recovery_dir=recovery))
            with mock.patch.object(po, "RESULTS", results), \
                    mock.patch.object(orchestrator, "verify_sources"), \
                    mock.patch.object(
                        orchestrator, "_star_interior_valid",
                        return_value=True), \
                    mock.patch.object(
                        orchestrator, "_region1_valid", return_value=True), \
                    mock.patch.object(
                        orchestrator, "_sweep_valid", return_value=True), \
                    mock.patch.object(
                        orchestrator, "_huang_bundle_valid",
                        return_value=False), \
                    mock.patch.object(orchestrator, "run_child") as run_child, \
                    mock.patch.object(orchestrator, "set_state"):
                with self.assertRaisesRegex(
                        po.OrchestratorError, "delegation evidence"):
                    orchestrator.ensure_huang()
            self.assertEqual(run_child.call_args.args[0], "HUANG_BUNDLE")
            self.assertFalse((results / "huang_bundle.json").exists())


class WatcherWorkerCountTests(unittest.TestCase):
    def _orchestrator(self):
        root = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        return po.Orchestrator(po.Settings(
            run_dir=root / "run", recovery_dir=root / "recovery"))

    def test_one_extra_worker_is_allowed_only_during_short_handoff(self):
        orchestrator = self._orchestrator()
        orchestrator._validate_watcher_worker_count("running", 25, 100.0)
        orchestrator._validate_watcher_worker_count(
            "running", 25, 100.0 + po.LOCAL_FILL_HANDOFF_GRACE_SECONDS)
        with self.assertRaisesRegex(po.OrchestratorError, "handoff stayed"):
            orchestrator._validate_watcher_worker_count(
                "running", 25,
                100.0 + po.LOCAL_FILL_HANDOFF_GRACE_SECONDS + 0.001)

    def test_normal_count_resets_handoff_and_large_deviation_fails(self):
        orchestrator = self._orchestrator()
        orchestrator._validate_watcher_worker_count("running", 23, 100.0)
        orchestrator._validate_watcher_worker_count("running", 24, 101.0)
        self.assertIsNone(orchestrator._watcher_count_mismatch_since)
        with self.assertRaisesRegex(po.OrchestratorError, "expected 24"):
            orchestrator._validate_watcher_worker_count("running", 22, 102.0)

    def _child(self, executable, cwd, command):
        child = mock.Mock()
        child.pid = 1234
        child.exe.return_value = str(executable)
        child.cwd.return_value = str(cwd)
        child.cmdline.return_value = [str(x) for x in command]
        return child

    def test_child_roles_accept_exact_worker_and_mesh_heartbeat_only(self):
        orchestrator = self._orchestrator()
        pyw = pathlib.Path(po.pythonw_executable()).resolve()
        worker = self._child(pyw, orchestrator.settings.recovery_dir, [
            pyw, "-c",
            "from multiprocessing.spawn import spawn_main; "
            f"spawn_main(parent_pid={orchestrator.settings.watcher_pid}, "
            "pipe_handle=123)", "--multiprocessing-fork"])
        self.assertEqual(orchestrator._watcher_child_role(worker), "worker")

        python = pathlib.Path(po.console_python_executable()).resolve()
        heartbeat = self._child(
            python, orchestrator.settings.recovery_dir,
            [python, po.MESH_SCRIPT, "pulse", "--agent", po.MESH_PARENT])
        self.assertEqual(orchestrator._watcher_child_role(heartbeat), "mesh")

        heartbeat.cmdline.return_value = [
            str(python), str(po.MESH_SCRIPT), "pulse", "--agent", "wrong-agent"]
        with self.assertRaisesRegex(po.OrchestratorError, "unexpected.*command"):
            orchestrator._watcher_child_role(heartbeat)


class StatusTests(unittest.TestCase):
    def test_invalid_v2_block3a_is_never_archived_or_downgraded(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            results = root / "results"
            results.mkdir()
            (results / "block3a_certificate.json").write_text(
                "{}\n", encoding="utf-8")
            orchestrator = po.Orchestrator(po.Settings(
                run_dir=root / "run", recovery_dir=root / "recovery"))
            with mock.patch.object(po, "RESULTS", results), \
                    mock.patch(
                        "block3a_singlerun.verify_certificate",
                        side_effect=ValueError("synthetic mismatch")), \
                    mock.patch.object(orchestrator, "_archive_file") as archive, \
                    mock.patch.object(orchestrator, "run_child") as run_child:
                with self.assertRaisesRegex(
                        po.OrchestratorError, "refusing.*downgrade"):
                    orchestrator.ensure_block3a_certificate()
                archive.assert_not_called()
                run_child.assert_not_called()

    def test_missing_freeze_exits_gate_instead_of_holding_claim_forever(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            orchestrator = po.Orchestrator(po.Settings(
                run_dir=root / "run", recovery_dir=root / "recovery"))
            with self.assertRaises(po.WaitingForGate):
                orchestrator.ensure_frozen_sources()
            state = po.load_canonical_json(orchestrator.state_path)
            self.assertEqual(state["phase"], "WAIT_SOURCE_FREEZE")
            self.assertEqual(state["status"], "waiting")

    def test_status_has_no_child_creation(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            settings = po.Settings(
                run_dir=root / "run", recovery_dir=root / "recovery").normalized()
            orchestrator = po.Orchestrator(settings)
            with mock.patch.object(orchestrator, "watcher_alive",
                                   return_value=(False, [])), \
                    mock.patch.object(po.subprocess, "Popen") as popen:
                status = orchestrator.status()
                self.assertFalse(status["watcher_alive"])
                popen.assert_not_called()

    def test_once_wait_gate_does_not_launch(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            settings = po.Settings(
                run_dir=root / "run", recovery_dir=root / "recovery",
                once=True).normalized()
            orchestrator = po.Orchestrator(settings)
            with mock.patch.object(orchestrator, "watcher_alive",
                                   return_value=(True, [object()])), \
                    mock.patch.object(orchestrator, "run_child") as run_child:
                with self.assertRaises(po.WaitingForGate):
                    orchestrator.ensure_local_fill()
                run_child.assert_not_called()

    def test_run_freezes_before_observing_local_fill(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            settings = po.Settings(
                run_dir=root / "run", recovery_dir=root / "recovery")
            order = []

            class FakeMesh:
                joined = False

                def join_and_claim(self):
                    self.joined = True

                def assert_resources(self, *resources):
                    pass

                def pulse(self, force=False):
                    pass

                def bye(self, status, summary):
                    self.joined = False

            orchestrator = po.Orchestrator(settings, mesh=FakeMesh())
            with mock.patch.object(po, "apply_owner_policy"), \
                    mock.patch.object(po, "Singleton") as singleton, \
                    mock.patch.object(orchestrator, "ensure_frozen_sources",
                                      side_effect=lambda: order.append("freeze")), \
                    mock.patch.object(orchestrator,
                                      "ensure_block3a_certificate",
                                      side_effect=lambda: order.append("block3a")), \
                    mock.patch.object(orchestrator, "ensure_huang",
                                      side_effect=lambda: order.append("huang")), \
                    mock.patch.object(orchestrator, "ensure_block3bc",
                                      side_effect=lambda: order.append("block3bc")), \
                    mock.patch.object(orchestrator, "final_verify",
                                      side_effect=lambda: order.append("verify")), \
                    mock.patch.object(orchestrator, "ensure_local_fill") as local, \
                    mock.patch.object(orchestrator, "ensure_tail") as tail:
                singleton.return_value.__enter__.return_value = object()
                self.assertEqual(orchestrator.run(), 0)
            self.assertEqual(
                order, ["freeze", "block3a", "huang", "block3bc", "verify"])
            local.assert_not_called()
            tail.assert_not_called()

    def test_recovery_claim_conflict_waits_without_launching(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            settings = po.Settings(
                run_dir=root / "run", recovery_dir=root / "recovery",
                once=True).normalized()
            mesh = mock.Mock(unsafe=True)
            mesh.claim_recovery.side_effect = po.MeshConflict("live owner")
            orchestrator = po.Orchestrator(settings, mesh=mesh)
            with mock.patch.object(orchestrator, "watcher_alive",
                                   return_value=(False, [])), \
                    mock.patch.object(orchestrator, "orphan_worker_pids",
                                      return_value=[]), \
                    mock.patch.object(orchestrator, "validate_local_fill") \
                    as validate, \
                    mock.patch.object(orchestrator, "run_child") as run_child:
                with self.assertRaises(po.WaitingForGate):
                    orchestrator.ensure_local_fill()
                validate.assert_not_called()
                run_child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
