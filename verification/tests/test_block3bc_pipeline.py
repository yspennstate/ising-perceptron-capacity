"""Zero-child synthetic coverage for the complete Block3bc artifact graph."""

import contextlib
import pathlib
import sys
import tempfile
import unittest
from fractions import Fraction
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import block3bc  # noqa: E402
import block3bc_assemble as assembler  # noqa: E402
import block3bc_aux_generate as aux_generate  # noqa: E402
import block3bc_aux_verify as aux_verify  # noqa: E402
import block3bc_exact as exact  # noqa: E402


class SyntheticBlock3bcPipelineTests(unittest.TestCase):
    # The patched auxiliary primitives give K < 3.517.  This nearby exact
    # ceiling keeps the synthetic negative replay schedule to 114 cells.
    K_RUN = Fraction(18, 5)

    @staticmethod
    @contextlib.contextmanager
    def _synthetic_environment(root):
        """Replace only the few live numerical pins used while verifying."""

        def zero(*_args, **_kwargs):
            return exact.fraction_arb(0)

        def lambda_cover(*_args, **_kwargs):
            return exact.fraction_arb(Fraction(-3, 25)).union(
                exact.fraction_arb(Fraction(-3, 100)))

        def exact_a_min(*_args, **_kwargs):
            return exact.fraction_arb(Fraction(2, 3))

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                aux_verify.dsfun, 'I_third_fullbound', side_effect=zero))
            stack.enter_context(mock.patch.object(
                aux_verify.dsfun, 'lam_cell', side_effect=lambda_cover))
            stack.enter_context(mock.patch.object(
                aux_verify.dsfun, 'A_of_tau', side_effect=exact_a_min))
            stack.enter_context(mock.patch.object(
                block3bc, 'boundaries', return_value=True))
            stack.enter_context(mock.patch.object(
                assembler, 'RESULTS_DIR', str(root)))
            yield

    @staticmethod
    @contextlib.contextmanager
    def _restore_files(*paths):
        paths = tuple(pathlib.Path(path) for path in paths)
        snapshots = {
            path: path.read_bytes() if path.exists() else None for path in paths
        }
        try:
            yield
        finally:
            for path, raw in snapshots.items():
                if raw is None:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                else:
                    path.write_bytes(raw)

    @staticmethod
    def _rewrite(path, digest_field, mutate):
        path = pathlib.Path(path)
        payload = exact.load_json(path)
        mutate(payload)
        payload[digest_field] = exact.payload_sha256(
            payload, omit=(digest_field,))
        exact.write_json_atomic(path, payload)
        return payload

    @staticmethod
    def _flip_job_hash_byte(path):
        path = pathlib.Path(path)
        raw = bytearray(path.read_bytes())
        marker = b'"job_sha256":"'
        start = raw.index(marker) + len(marker)
        raw[start] = ord('1') if raw[start] != ord('1') else ord('2')
        path.write_bytes(raw)

    @staticmethod
    def _write_auxiliary_jobs(output, kind):
        output = pathlib.Path(output)
        record_dir = pathlib.Path(str(output) + '.records')
        record_dir.mkdir(parents=True, exist_ok=True)
        hashes = exact.source_hashes(aux_generate._source_paths())
        # An exact integer packet keeps the verifier's endpoint ordering
        # decidable; the synthetic A packet is only schema evidence.
        a_packet = exact.arb_packet(exact.fraction_arb(1))
        positive_packet = exact.arb_packet(exact.fraction_arb(100))
        zero_packet = exact.arb_packet(exact.fraction_arb(0))

        if kind == 'ell_prime':
            boundaries = exact.ell_boundaries()
            for index, (lo, hi) in enumerate(
                    exact.intervals_from_boundaries(boundaries)):
                job_input = {
                    'tau_lo': exact.fraction_record(lo),
                    'tau_hi': exact.fraction_record(hi),
                }
                result = {
                    'index': index,
                    'tau_lo': job_input['tau_lo'],
                    'tau_hi': job_input['tau_hi'],
                    'A': a_packet,
                    'value': positive_packet,
                    'ok': True,
                    'runtime_milliseconds': 0,
                }
                saved = aux_generate._job_record(
                    kind, result, job_input, hashes)
                exact.write_json_atomic(
                    record_dir / f'ellp-{index:03d}.json', saved,
                    overwrite=False)
            return

        if kind == 'k_grid':
            for index, lam in enumerate(exact.k_nodes()):
                job_input = {'lambda_value': exact.fraction_record(lam)}
                result = {
                    'index': index,
                    'lambda_value': job_input['lambda_value'],
                    'value': zero_packet,
                    'ok': True,
                    'runtime_milliseconds': 0,
                }
                saved = aux_generate._job_record(
                    kind, result, job_input, hashes)
                exact.write_json_atomic(
                    record_dir / f'i2-{index:03d}.json', saved,
                    overwrite=False)
            return

        raise ValueError(kind)

    @staticmethod
    def _write_replay_jobs(output, part, k_run, aux_hash):
        output = pathlib.Path(output)
        record_dir = pathlib.Path(str(output) + '.records')
        record_dir.mkdir(parents=True, exist_ok=True)
        hashes = exact.source_hashes(assembler._replay_source_paths())
        boundaries = assembler._part_boundaries(part, k_run)
        sign = '>0' if part == 'b_neg' else '<0'
        sign_packet = exact.arb_packet(
            exact.fraction_arb(1 if sign == '>0' else -1))

        for index, (lo, hi) in enumerate(
                exact.intervals_from_boundaries(boundaries)):
            job_input = {
                'tau_lo': exact.fraction_record(lo),
                'tau_hi': exact.fraction_record(hi),
                'k_run': exact.fraction_record(k_run),
            }
            result = {
                'index': index,
                'tau_lo': job_input['tau_lo'],
                'tau_hi': job_input['tau_hi'],
                'ok': True,
                'leaves': [{
                    'tau_lo': job_input['tau_lo'],
                    'tau_hi': job_input['tau_hi'],
                    'value': sign_packet,
                    'sign': sign,
                }],
                'failure': None,
                'runtime_milliseconds': 0,
            }
            saved = block3bc._replay_job_record(
                part, result, job_input, aux_hash, hashes)
            exact.write_json_atomic(
                record_dir / f'{part}-{index:04d}.json', saved,
                overwrite=False)

    def _build_bundle(self, root):
        root = pathlib.Path(root)
        aux_dir = root / 'aux'
        replay_dir = root / 'replay'
        aux_dir.mkdir()
        replay_dir.mkdir()

        ell_shard = aux_dir / 'ell.json'
        k_shard = aux_dir / 'k.json'
        self._write_auxiliary_jobs(ell_shard, 'ell_prime')
        self._write_auxiliary_jobs(k_shard, 'k_grid')
        aux_generate.generate_shard(
            'ell_prime', workers=1, output=str(ell_shard))
        aux_generate.generate_shard(
            'k_grid', workers=1, output=str(k_shard))

        manifest = aux_dir / 'manifest.json'
        manifest_payload = aux_verify.build_manifest(
            [str(ell_shard)], [str(k_shard)],
            exact.fraction_text(self.K_RUN), str(manifest))
        aux_hash = manifest_payload['manifest_sha256']

        replay_shards = {}
        for part in ('b_pos', 'b_neg', 'c'):
            shard = replay_dir / f'{part}.json'
            self._write_replay_jobs(
                shard, part, self.K_RUN, aux_hash)
            block3bc.replay_part(
                part, str(manifest), workers=1, output=str(shard))
            replay_shards[part] = shard

        certificate = root / 'certificate.json'
        assembler.assemble(
            str(manifest),
            [str(replay_shards[part])
             for part in ('b_pos', 'b_neg', 'c')],
            str(certificate))
        return {
            'ell_shard': ell_shard,
            'k_shard': k_shard,
            'manifest': manifest,
            'replay': replay_shards,
            'certificate': certificate,
            'ell_job0': pathlib.Path(str(ell_shard) + '.records') /
                        'ellp-000.json',
        }

    def test_zero_child_pipeline_and_rehash_aware_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td).resolve()
            with self._synthetic_environment(root), mock.patch.object(
                    exact.subprocess, 'Popen',
                    side_effect=AssertionError('synthetic test spawned a child')):
                paths = self._build_bundle(root)
                certificate = assembler.verify_certificate(
                    str(paths['certificate']))
                self.assertEqual(certificate['verdict'], 'ALL PASS')
                self.assertEqual(certificate['summary'], {
                    'b_pos': {'top_cells': 24, 'leaves': 24},
                    'b_neg': {'top_cells': 114, 'leaves': 114},
                    'c': {'top_cells': 16, 'leaves': 16},
                })

                with self.subTest('raw job byte is bound by the shard'):
                    with self._restore_files(paths['ell_job0']):
                        self._flip_job_hash_byte(paths['ell_job0'])
                        with self.assertRaisesRegex(
                                ValueError,
                                'missing or changed auxiliary job artifact'):
                            aux_verify.verify_shard(str(paths['ell_shard']))

                with self.subTest('rehash cannot launder stale source'):
                    with self._restore_files(paths['ell_job0']):
                        self._rewrite(
                            paths['ell_job0'], 'job_sha256',
                            lambda row: row['source_sha256'].__setitem__(
                                'dsfun.py', '0' * 64))
                        with self.assertRaisesRegex(
                                ValueError, 'stale/corrupt resume record'):
                            aux_generate.generate_shard(
                                'ell_prime', workers=1,
                                output=str(paths['ell_shard']))

                with self.subTest('rehash cannot launder runtime'):
                    with self._restore_files(paths['ell_job0']):
                        self._rewrite(
                            paths['ell_job0'], 'job_sha256',
                            lambda row: row['runtime'].__setitem__(
                                'flint', 'wrong'))
                        with self.assertRaisesRegex(
                                ValueError, 'incompatible resume runtime'):
                            aux_generate.generate_shard(
                                'ell_prime', workers=1,
                                output=str(paths['ell_shard']))

                with self.subTest('rehash cannot launder job index'):
                    with self._restore_files(paths['ell_job0']):
                        self._rewrite(
                            paths['ell_job0'], 'job_sha256',
                            lambda row: row.__setitem__('index', 1))
                        with self.assertRaisesRegex(
                                ValueError, 'stale/corrupt resume record'):
                            aux_generate.generate_shard(
                                'ell_prime', workers=1,
                                output=str(paths['ell_shard']))

                with self.subTest('old K endpoint is rejected after rehash'):
                    with self._restore_files(paths['k_shard']):
                        def old_endpoint(row):
                            row['schedule'][-1] = exact.fraction_record(
                                Fraction(-47, 2000))
                            row['schedule_sha256'] = exact.payload_sha256(
                                row['schedule'], omit=())

                        self._rewrite(
                            paths['k_shard'], 'artifact_sha256', old_endpoint)
                        with self.assertRaisesRegex(
                                ValueError, 'auxiliary shard schedule mismatch'):
                            aux_verify.verify_shard(str(paths['k_shard']))

                with self.subTest('missing schedule index is rejected'):
                    with self._restore_files(paths['k_shard']):
                        def omit_last(row):
                            row['indices'].pop()
                            row['records'].pop()
                            row['job_artifacts'].pop()

                        self._rewrite(
                            paths['k_shard'], 'artifact_sha256', omit_last)
                        with self.assertRaisesRegex(
                                ValueError, 'auxiliary lane indices mismatch'):
                            aux_verify.verify_shard(str(paths['k_shard']))

                with self.subTest('duplicate schedule index is rejected'):
                    with self._restore_files(paths['ell_shard']):
                        self._rewrite(
                            paths['ell_shard'], 'artifact_sha256',
                            lambda row: row['records'][1].__setitem__(
                                'index', 0))
                        with self.assertRaisesRegex(
                                ValueError, 'auxiliary record index mismatch'):
                            aux_verify.verify_shard(str(paths['ell_shard']))

                with self.subTest('certificate rejects missing replay schedule'):
                    with self._restore_files(paths['certificate']):
                        self._rewrite(
                            paths['certificate'], 'certificate_sha256',
                            lambda row: row['replay_artifacts'].pop())
                        with self.assertRaisesRegex(
                                ValueError, 'incomplete c replay union'):
                            assembler.verify_certificate(
                                str(paths['certificate']))

                with self.subTest('certificate rejects duplicate replay schedule'):
                    with self._restore_files(paths['certificate']):
                        def duplicate_first(row):
                            row['replay_artifacts'].append(
                                dict(row['replay_artifacts'][0]))

                        self._rewrite(
                            paths['certificate'], 'certificate_sha256',
                            duplicate_first)
                        with self.assertRaisesRegex(
                                ValueError, 'duplicate .* top cell'):
                            assembler.verify_certificate(
                                str(paths['certificate']))


if __name__ == '__main__':
    unittest.main()
