import hashlib
import pathlib
import shutil
import sys
import tempfile
import unittest

from flint import arb


HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import block3a_assemble as legacy  # noqa: E402
import block3a_run as producer  # noqa: E402
import block3a_singlerun as v2  # noqa: E402
import block3bc_exact as exact  # noqa: E402


class Block3aStructuredEvidenceTests(unittest.TestCase):
    @staticmethod
    def rows(bound=arb(-1)):
        rows = []
        runtime = exact.runtime_record(60, workers=1)
        sources = producer.source_hashes(producer.HERE)
        for target in legacy.canonical_schedule():
            rows.append({
                'schema_version': 1,
                'run_id': '0' * 32,
                'index': target['index'],
                'kind': target['kind'],
                'tau_lo': target['tau_lo'],
                'tau_hi': target['tau_hi'],
                'verdict': 'PASS',
                'recursive_calls': 1,
                'leaf_bounds': [exact.arb_packet(bound)],
                'elapsed_seconds': '0.000000',
                'worker_runtime': runtime,
                'worker_source_sha256': sources,
            })
        return rows

    @staticmethod
    def encode(rows):
        return b''.join(exact.canonical_json_bytes(row) + b'\n'
                        for row in rows)

    def test_complete_negative_packets_pass(self):
        records, leaves, upper = v2.parse_structured_log_bytes(
            self.encode(self.rows()), legacy.canonical_schedule())
        self.assertEqual(len(records), legacy.SCHEDULE_COUNT)
        self.assertEqual(leaves, legacy.SCHEDULE_COUNT)
        self.assertLess(upper, 0)

    def test_complete_fabricated_positive_log_is_rejected(self):
        rows = self.rows()
        rows[91]['leaf_bounds'] = [exact.arb_packet(arb(1))]
        with self.assertRaisesRegex(ValueError, 'nonnegative leaf upper'):
            v2.parse_structured_log_bytes(
                self.encode(rows), legacy.canonical_schedule())

    def test_source_closure_includes_producer_and_verifier(self):
        hashes = v2.source_hashes()
        self.assertEqual(hashes['block3a_run.py'],
                         exact.file_sha256(producer.__file__))
        self.assertEqual(hashes['block3a_singlerun.py'],
                         exact.file_sha256(v2.__file__))


class Block3aReceiptTests(unittest.TestCase):
    run_id = '1' * 32

    @staticmethod
    def artifact(path, root):
        raw = path.read_bytes()
        return {
            'file': path.relative_to(root).as_posix(),
            'sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw),
        }

    def build(self, root):
        root = pathlib.Path(root)
        frozen = root / 'frozen_source'
        frozen.mkdir()
        for name in producer.PROOF_SOURCE_NAMES:
            shutil.copyfile(producer.HERE / name, frozen / name)
        sources = producer.validate_frozen_tree(frozen)

        rows = Block3aStructuredEvidenceTests.rows()
        for row in rows:
            row['run_id'] = self.run_id
        (root / 'block3a.jsonl').write_bytes(
            Block3aStructuredEvidenceTests.encode(rows))
        (root / 'stdout.txt').write_bytes(b'')
        (root / 'stderr.txt').write_bytes(b'')

        runtime = exact.runtime_record(60, workers=3)
        launch = {
            'schema_version': 2,
            'kind': 'block3a_single_run_launch',
            'run_id': self.run_id,
            'created_utc': '2026-07-11T00:00:00+00:00',
            'proof_source_sha256': sources,
            'frozen_source_sha256': sources,
            'runner_source_sha256': exact.file_sha256(producer.__file__),
            'runtime': runtime,
            'workers': 3,
            'timeout_seconds': 21600,
            'command': [
                runtime['executable'], '-I', '-S', '-B', '-c',
                producer.BOOTSTRAP,
                'frozen_source', 'python_flint_site', '3',
                '--output', 'block3a.jsonl',
                '--run-id', self.run_id,
            ],
            'cwd': 'frozen_source',
            'output': 'block3a.jsonl',
            'import_policy': 'isolated-bootstrap-v3-no-site',
        }
        launch['launch_sha256'] = exact.payload_sha256(
            launch, omit=('launch_sha256',))
        launch_path = root / 'launch.json'
        exact.write_json_atomic(launch_path, launch)
        receipt = {
            'schema_version': 2,
            'kind': 'block3a_single_run_receipt',
            'run_id': self.run_id,
            'launch': self.artifact(launch_path, root),
            'finished_utc': '2026-07-11T00:10:00+00:00',
            'launcher_pid': 123,
            'exit_code': 0,
            'timed_out': False,
            'source_unchanged': True,
            'proof_source_sha256_after': sources,
            'frozen_source_sha256_after': sources,
            'artifacts': {
                key: self.artifact(root / filename, root)
                for key, filename in {
                    'log': 'block3a.jsonl', 'stdout': 'stdout.txt',
                    'stderr': 'stderr.txt'}.items()
            },
        }
        receipt['receipt_sha256'] = exact.payload_sha256(
            receipt, omit=('receipt_sha256',))
        receipt_path = root / 'receipt.json'
        exact.write_json_atomic(receipt_path, receipt)
        return receipt_path

    def repin_launch(self, root, mutate):
        root = pathlib.Path(root)
        launch_path = root / 'launch.json'
        launch = exact.load_json(launch_path)
        mutate(launch)
        launch['launch_sha256'] = exact.payload_sha256(
            launch, omit=('launch_sha256',))
        exact.write_json_atomic(launch_path, launch)
        receipt_path = root / 'receipt.json'
        receipt = exact.load_json(receipt_path)
        receipt['launch'] = self.artifact(launch_path, root)
        receipt['receipt_sha256'] = exact.payload_sha256(
            receipt, omit=('receipt_sha256',))
        exact.write_json_atomic(receipt_path, receipt)

    def test_valid_synthetic_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            path = self.build(td)
            receipt, _, records, _, _, _ = v2._validate_receipt(path)
            self.assertEqual(receipt['run_id'], self.run_id)
            self.assertEqual(len(records), legacy.SCHEDULE_COUNT)

    def test_suffix_path_impostors_and_log_substitution_fail(self):
        with tempfile.TemporaryDirectory() as td:
            path = self.build(td)
            self.repin_launch(td, lambda launch: launch.update(
                cwd='/evil/frozen_source'))
            with self.assertRaisesRegex(ValueError, 'command mismatch'):
                v2._validate_receipt(path)

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            path = self.build(root)
            shutil.copyfile(root / 'block3a.jsonl', root / 'other.jsonl')
            receipt = exact.load_json(path)
            receipt['artifacts']['log'] = self.artifact(
                root / 'other.jsonl', root)
            receipt['receipt_sha256'] = exact.payload_sha256(
                receipt, omit=('receipt_sha256',))
            exact.write_json_atomic(path, receipt)
            with self.assertRaisesRegex(ValueError, 'not launch-bound'):
                v2._validate_receipt(path)

    def test_extra_import_file_and_boolean_exit_fail(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            path = self.build(root)
            (root / 'frozen_source' / 'flint.py').write_text('raise SystemExit')
            with self.assertRaisesRegex(ValueError, 'extra entries'):
                v2._validate_receipt(path)

        with tempfile.TemporaryDirectory() as td:
            path = self.build(td)
            receipt = exact.load_json(path)
            receipt['exit_code'] = False
            receipt['receipt_sha256'] = exact.payload_sha256(
                receipt, omit=('receipt_sha256',))
            exact.write_json_atomic(path, receipt)
            with self.assertRaisesRegex(ValueError, 'finish cleanly'):
                v2._validate_receipt(path)


if __name__ == '__main__':
    unittest.main()
