import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import block3a_assemble as legacy  # noqa: E402
import block3a_singlerun as single_run  # noqa: E402
import auto_finish  # noqa: E402
import verify_all  # noqa: E402


class Block3aDispatchTests(unittest.TestCase):
    def test_auto_finisher_uses_final_v2_verifier(self):
        block3a, block3bc = auto_finish.certificate_commands()
        self.assertIn('block3a_singlerun.py', block3a)
        self.assertNotIn('block3a_assemble.py', block3a)
        self.assertIn('--certificate', block3bc)

    def test_auto_finisher_fails_once_present_certificate_is_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            results = root / 'results'
            results.mkdir()
            for name in ('block3a_certificate.json',
                         'block3bc_certificate.json'):
                (results / name).write_text('{}\n', encoding='utf-8')
            failed = types.SimpleNamespace(
                returncode=1, stdout='', stderr='runtime mismatch')
            passed = types.SimpleNamespace(
                returncode=0, stdout='PASS', stderr='')
            with mock.patch.object(auto_finish, 'HERE', str(root)), \
                    mock.patch.object(auto_finish, 'RES', str(results)), \
                    mock.patch.object(
                        auto_finish, 'LOG', str(results / 'auto_finish.log')), \
                    mock.patch.object(
                        auto_finish.subprocess, 'run',
                        side_effect=[failed, passed]), \
                    mock.patch.object(auto_finish.time, 'sleep') as sleep:
                self.assertEqual(auto_finish.main(), 1)
            sleep.assert_not_called()
            failure = (results / 'FINISH_FAILED.txt').read_text(
                encoding='utf-8')
            self.assertIn('runtime mismatch', failure)

    def test_single_run_source_hashes_include_the_verifier(self):
        hashes = single_run.source_hashes()
        self.assertEqual(
            hashes['block3a_singlerun.py'],
            legacy.file_sha256(single_run.__file__),
        )

    def test_dispatches_single_run_evidence(self):
        marker = {'certificate_sha256': 'single'}
        with mock.patch.object(
                legacy, 'load_certificate',
                return_value={'evidence_model': single_run.EVIDENCE_MODEL}), \
                mock.patch.object(
                    single_run, 'verify_certificate', return_value=marker
                ) as verifier:
            self.assertIs(
                verify_all.verify_block3a_certificate('certificate.json'),
                marker,
            )
        verifier.assert_called_once_with('certificate.json')

    def test_dispatches_legacy_attested_evidence(self):
        marker = {'certificate_sha256': 'legacy'}
        payload = {
            'policy': {'evidence_model': 'source-bound-trusted-execution'}
        }
        with mock.patch.object(
                legacy, 'load_certificate', return_value=payload), \
                mock.patch.object(
                    legacy, 'verify_certificate', return_value=marker
                ) as verifier:
            self.assertIs(
                verify_all.verify_block3a_certificate('certificate.json'),
                marker,
            )
        verifier.assert_called_once_with('certificate.json')

    def test_rejects_unknown_or_missing_evidence_mode(self):
        for payload in ({}, {'evidence_model': 'untrusted'}):
            with self.subTest(payload=payload), mock.patch.object(
                    legacy, 'load_certificate', return_value=payload):
                with self.assertRaisesRegex(
                        ValueError, 'unknown Block3a evidence model'):
                    verify_all.verify_block3a_certificate('certificate.json')

    def test_final_policy_rejects_legacy_downgrade(self):
        payload = {
            'policy': {'evidence_model': 'source-bound-trusted-execution'}
        }
        with mock.patch.object(
                legacy, 'load_certificate', return_value=payload):
            with self.assertRaisesRegex(ValueError, 'final policy'):
                verify_all.verify_block3a_certificate(
                    'certificate.json',
                    expected_model=single_run.EVIDENCE_MODEL)


if __name__ == '__main__':
    unittest.main()
