import copy
import os
import pathlib
import sys
import tempfile
import time
import unittest
from fractions import Fraction
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from flint import arb  # noqa: E402
import block3bc_exact as ex  # noqa: E402
from core import dec, set_prec  # noqa: E402


class FractionTests(unittest.TestCase):
    def test_fraction_text_parses_exactly(self):
        self.assertEqual(ex.parse_fraction_text('-0.131'), Fraction(-131, 1000))
        self.assertEqual(ex.parse_fraction_text('21/2'), Fraction(21, 2))
        self.assertEqual(ex.parse_fraction_text('0'), Fraction(0))

    def test_fraction_text_rejects_noncanonical_input(self):
        bad = (' 1/2', '1/2 ', '+1/2', '01/2', '2/4', '1/-2', '1e-3',
               '1_000/2', '', '1//2')
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError, ZeroDivisionError)):
                    ex.parse_fraction_text(value)
        for value in (True, 1.0, {'num': '1', 'den': '2'}, '1/2'):
            with self.assertRaises(TypeError):
                ex.as_fraction(value)

    def test_fraction_record_is_strict_and_reduced(self):
        value = Fraction(-7, 13)
        self.assertEqual(ex.fraction_from_record(ex.fraction_record(value)), value)
        bad = ({'num': '-07', 'den': '13'}, {'num': '-7', 'den': '013'},
               {'num': '-14', 'den': '26'}, {'num': -7, 'den': '13'},
               {'num': '-7', 'den': -13}, {'num': '1', 'den': '0'},
               {'num': '1', 'den': '2', 'extra': 'x'})
        for record in bad:
            with self.subTest(record=record):
                with self.assertRaises((TypeError, ValueError)):
                    ex.fraction_from_record(record)


class ArbPacketTests(unittest.TestCase):
    def setUp(self):
        set_prec(100)

    def test_roundtrip_contains_original(self):
        eps = dec('0.00000000000000000001')
        value = dec('0.1') / 3 + eps.union(-eps)
        packet = ex.arb_packet(value)
        self.assertTrue(ex.packet_contains(packet, value))
        lo, hi = ex.packet_fraction_endpoints(packet)
        self.assertLessEqual(lo, hi)

    def test_rejects_float_nonfinite_and_bad_packets(self):
        with self.assertRaises(TypeError):
            ex.arb_packet(0.5)
        with self.assertRaises(ValueError):
            ex.arb_packet(arb('nan'))
        packet = ex.arb_packet(dec('0.5'))
        for key, value in (('rad10', '-1'), ('mid10', '01'),
                           ('exp10', True), ('digits', 0)):
            bad = copy.deepcopy(packet)
            bad[key] = value
            with self.subTest(key=key):
                with self.assertRaises((TypeError, ValueError)):
                    ex.packet_fraction_endpoints(bad)

    def test_aux_verifier_accepts_nondyadic_packet_endpoints(self):
        import block3bc_aux_verify as verifier
        packet = ex.arb_packet(ex.fraction_arb(Fraction(2, 3)))
        lo, hi = verifier._require_packet(packet)
        self.assertTrue(lo <= hi)


class ScheduleTests(unittest.TestCase):
    def test_k_schedule(self):
        nodes = ex.k_nodes()
        self.assertEqual(len(nodes), 59)
        self.assertEqual(nodes[0], Fraction(-131, 1000))
        self.assertEqual(nodes[-1], Fraction(-53, 2500))
        self.assertTrue(all(nodes[i + 1] - nodes[i] == Fraction(549, 290000)
                            for i in range(58)))

    def test_old_right_endpoint_regression(self):
        representative_right = Fraction(-21277, 1_000_000)
        self.assertGreater(representative_right, Fraction(-47, 2000))
        self.assertLessEqual(representative_right, Fraction(-53, 2500))

    def test_other_schedules(self):
        ell = ex.ell_boundaries()
        self.assertEqual((len(ell), ell[0], ell[-1]),
                         (17, Fraction(-1, 5), Fraction(-1, 50)))
        c = ex.c_boundaries()
        self.assertEqual((len(c), c[0], c[-1]),
                         (17, Fraction(-43, 1000), Fraction(39, 500)))
        bp = ex.b_pos_boundaries()
        self.assertEqual((len(bp), bp[0], bp[-1]),
                         (25, Fraction(3, 50), Fraction(13, 50)))

    def test_bneg_floor_plus_one_rule(self):
        import block3bc_aux_generate as aux

        self.assertEqual(aux.DEFAULT_K_RUN, '21/2')
        self.assertEqual(ex.b_neg_count(Fraction(7, 11)), 21)
        self.assertEqual(ex.b_neg_count(Fraction(21, 2)), 331)
        boundaries = ex.b_neg_boundaries(Fraction(21, 2))
        self.assertEqual(
            (len(boundaries), boundaries[0], boundaries[-1]),
            (332, Fraction(-19, 100), Fraction(-3, 100)))
        self.assertTrue(all(
            boundaries[i + 1] - boundaries[i] == Fraction(4, 8275)
            for i in range(331)))

    def test_partition_and_union_are_exact(self):
        cells = [(Fraction(0), Fraction(1, 2)),
                 (Fraction(1, 2), Fraction(1))]
        self.assertTrue(ex.require_exact_partition(cells, (0, 1)))
        self.assertTrue(ex.require_exact_union(
            [(0, Fraction(3, 4)), (Fraction(1, 2), 1)], (0, 1)))
        tiny = Fraction(1, 10**100)
        bad = [(Fraction(0), Fraction(1, 2)),
               (Fraction(1, 2) + tiny, Fraction(1))]
        with self.assertRaises(ValueError):
            ex.require_exact_partition(bad, (0, 1))
        with self.assertRaises(ValueError):
            ex.require_exact_union(bad, (0, 1))
        for intervals in (cells + [cells[0]], list(reversed(cells))):
            with self.assertRaises(ValueError):
                ex.require_exact_partition(intervals, (0, 1))


class CanonicalJsonTests(unittest.TestCase):
    def test_float_and_noncanonical_json_rejected(self):
        with self.assertRaises(TypeError):
            ex.canonical_json_bytes({'x': 0.5})
        bad_raw = (b'{"x":1.2}\n', b'{"x":NaN}\n',
                   b'{"x":1,"x":2}\n', b'\xef\xbb\xbf{"x":1}\n',
                   b'{ "x": 1 }\n')
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'x.json')
            for raw in bad_raw:
                pathlib.Path(path).write_bytes(raw)
                with self.subTest(raw=raw):
                    with self.assertRaises((TypeError, ValueError)):
                        ex.load_json(path)

    def test_canonical_roundtrip_and_hash_tamper(self):
        a = {'b': 2, 'a': [1, True, None]}
        b = {'a': [1, True, None], 'b': 2}
        self.assertEqual(ex.canonical_json_bytes(a), ex.canonical_json_bytes(b))
        h = ex.payload_sha256(a, omit=())
        changed = copy.deepcopy(a)
        changed['b'] = 3
        self.assertNotEqual(h, ex.payload_sha256(changed, omit=()))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'x.json')
            ex.write_json_atomic(path, a)
            self.assertEqual(ex.load_json(path), a)

    def test_atomic_no_clobber(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'immutable.json')
            ex.write_json_atomic(path, {'value': 1}, overwrite=False)
            with self.assertRaises(FileExistsError):
                ex.write_json_atomic(path, {'value': 2}, overwrite=False)
            self.assertEqual(ex.load_json(path), {'value': 1})


class RuntimeRecordTests(unittest.TestCase):
    def test_fixed_runtime_policy_and_binary_identity(self):
        runtime = ex.runtime_record(100, workers=3)
        self.assertEqual(
            ex.validate_runtime_record(runtime, 100, workers=3), runtime)
        self.assertRegex(runtime['python_executable_sha256'], r'^[0-9a-f]{64}$')

        wrong = copy.deepcopy(runtime)
        wrong['flint'] = '0.0.0'
        with self.assertRaisesRegex(ValueError, 'proof policy'):
            ex.validate_runtime_record(wrong, 100, workers=3)

        missing = copy.deepcopy(runtime)
        del missing['python_executable_sha256']
        with self.assertRaisesRegex(ValueError, 'schema'):
            ex.validate_runtime_record(missing, 100, workers=3)

        with mock.patch.object(ex.flint, '__version__', '0.10.0'):
            with self.assertRaisesRegex(ValueError, 'current verifier'):
                ex.validate_runtime_record(runtime, 100, workers=3)

        boolean_workers = copy.deepcopy(runtime)
        boolean_workers['workers'] = True
        with self.assertRaisesRegex(ValueError, 'plain integer'):
            ex.validate_runtime_record(boolean_workers, 100, workers=1)


class IsolationRunnerTests(unittest.TestCase):
    @staticmethod
    def _pid_exists(pid):
        if os.name != 'nt':
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            return True
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [
            wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(0x00100000, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x102
        finally:
            kernel32.CloseHandle(handle)

    @staticmethod
    def _validator(key, result):
        if (not isinstance(result, dict)
                or result.get('index') != key
                or result.get('ok') is not True):
            raise ValueError('wrong synthetic result')

    def test_malformed_result_is_preserved_and_retried(self):
        code = (
            "import pathlib,sys;"
            "m=pathlib.Path(sys.argv[1]);"
            "p=pathlib.Path(sys.argv[sys.argv.index('--result-file')+1]);"
            "i=0 if m.exists() else 99;"
            "m.write_text('attempt',encoding='ascii');"
            "p.write_bytes(('{' + '\"index\":%d,\"ok\":true' % i + '}' "
            "+ chr(10)).encode('ascii'))")
        with tempfile.TemporaryDirectory() as td:
            marker = os.path.join(td, 'attempt.marker')
            specs = [(0, [sys.executable, '-B', '-c', code, marker])]
            rows = list(ex.isolated_subprocess_results(
                specs, 1, td, timeout_seconds=5, retries=1,
                result_validator=self._validator))
            self.assertEqual(rows, [(0, {'index': 0, 'ok': True})])
            evidence = list(pathlib.Path(td).glob('.worker-0-*.json'))
            errors = list(pathlib.Path(td).glob('.worker-0-*.stderr'))
            self.assertEqual(len(evidence), 1)
            self.assertEqual(ex.load_json(evidence[0]), {'index': 99, 'ok': True})
            self.assertEqual(len(errors), 1)
            self.assertIn(b'invalid child result', errors[0].read_bytes())

    def test_late_success_is_rejected_as_timeout(self):
        code = (
            "import pathlib,sys,time;"
            "time.sleep(0.05);"
            "p=pathlib.Path(sys.argv[sys.argv.index('--result-file')+1]);"
            "p.write_bytes(b'{\"index\":0,\"ok\":true}\\n')")
        with tempfile.TemporaryDirectory() as td:
            specs = [(0, [sys.executable, '-B', '-c', code])]
            with self.assertRaisesRegex(RuntimeError, 'deadline exceeded'):
                list(ex.isolated_subprocess_results(
                    specs, 1, td, timeout_seconds=0.01, retries=0,
                    result_validator=self._validator))

    def test_early_close_kills_and_reaps_sibling(self):
        slow = (
            "import os,pathlib,sys,time;"
            "start=pathlib.Path(sys.argv[1]);"
            "done=pathlib.Path(sys.argv[2]);"
            "start.write_text(str(os.getpid()),encoding='ascii');"
            "time.sleep(10);"
            "done.write_text('bad',encoding='ascii')")
        fast = (
            "import pathlib,sys,time;"
            "time.sleep(0.15);"
            "p=pathlib.Path(sys.argv[sys.argv.index('--result-file')+1]);"
            "p.write_bytes(b'{\"index\":1,\"ok\":true}\\n')")
        with tempfile.TemporaryDirectory() as td:
            started = os.path.join(td, 'slow.pid')
            completed = os.path.join(td, 'slow.completed')
            specs = [
                (0, [sys.executable, '-B', '-c', slow, started, completed]),
                (1, [sys.executable, '-B', '-c', fast]),
            ]
            generator = ex.isolated_subprocess_results(
                specs, 2, td, timeout_seconds=20, retries=0,
                result_validator=lambda key, result: None)
            self.assertEqual(next(generator)[0], 1)
            deadline = time.monotonic() + 2
            while not os.path.exists(started) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(os.path.exists(started))
            pid = int(pathlib.Path(started).read_text(encoding='ascii'))
            generator.close()
            deadline = time.monotonic() + 2
            while self._pid_exists(pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(self._pid_exists(pid))
            self.assertFalse(os.path.exists(completed))

    def test_auxiliary_python_exception_propagates(self):
        import block3bc_aux_generate as aux
        with mock.patch.object(
                aux.dsfun, 'I_second_box',
                side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                aux._k_job((0, ex.fraction_record(Fraction(-1, 10))))

    def test_shallow_auxiliary_packet_is_rejected(self):
        import block3bc_aux_generate as aux
        job_input = {'lambda_value': ex.fraction_record(Fraction(-1, 10))}
        bad = {'index': 0, 'lambda_value': job_input['lambda_value'],
               'value': None, 'ok': True, 'runtime_milliseconds': 0}
        with self.assertRaisesRegex(ValueError, 'packet'):
            aux._validate_child_result('k_grid', 0, job_input, bad)

    def test_empty_successful_replay_is_rejected(self):
        import block3bc
        expected = {
            'tau_lo': ex.fraction_record(Fraction(0)),
            'tau_hi': ex.fraction_record(Fraction(1, 10)),
            'k_run': ex.fraction_record(Fraction(21, 2)),
        }
        bad = {'index': 0, 'tau_lo': expected['tau_lo'],
               'tau_hi': expected['tau_hi'], 'ok': True, 'leaves': [],
               'failure': None, 'runtime_milliseconds': 0}
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            block3bc._validate_replay_child('b_pos', 0, expected, bad)


class ManifestAdversarialTests(unittest.TestCase):
    def test_complete_manifest_rejects_empty_artifact_list(self):
        import block3bc_aux_verify as ver

        set_prec(100)
        one = ex.arb_packet(ex.fraction_arb(1))
        ell_bounds = ex.ell_boundaries()
        ell = [dict(index=i, tau_lo=ex.fraction_record(ell_bounds[i]),
                    tau_hi=ex.fraction_record(ell_bounds[i + 1]),
                    A=one, value=one, ok=True, runtime_milliseconds=0)
               for i in range(16)]
        nodes = ex.k_nodes()
        grid = [dict(index=i, lambda_value=ex.fraction_record(nodes[i]),
                     value=one, ok=True, runtime_milliseconds=0)
                for i in range(59)]
        k_run = Fraction(21, 2)
        payload = {
            'schema_version': ex.SCHEMA_VERSION,
            'kind': 'block3bc_aux_manifest',
            'source_sha256': ex.source_hashes(ver._manifest_source_paths()),
            'runtime': ex.runtime_record(100),
            'parameter_policy': {
                'precision_bits': 100, 'I_second_inner_bits': 18,
                'I_second_outer_bits': 15, 'ell_prime_cells': 300},
            'ell_schedule': [ex.fraction_record(x) for x in ell_bounds],
            'k_schedule': [ex.fraction_record(x) for x in nodes],
            'ell_records': ell, 'k_records': grid,
            'derived': ver._derive(ell, grid, k_run),
            'k_run': ex.fraction_record(k_run), 'input_artifacts': [],
        }
        payload['manifest_sha256'] = ex.payload_sha256(payload)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'manifest.json')
            ex.write_json_atomic(path, payload)
            with self.assertRaisesRegex(ValueError, 'no input artifacts'):
                ver.verify_manifest(path, require_complete=True)

    def test_replay_resume_rejects_runtime_laundering(self):
        import block3bc

        job_input = {'tau_lo': ex.fraction_record(Fraction(0)),
                     'tau_hi': ex.fraction_record(Fraction(1, 10)),
                     'k_run': ex.fraction_record(Fraction(21, 2))}
        result = {
            'index': 0,
            'tau_lo': job_input['tau_lo'],
            'tau_hi': job_input['tau_hi'],
            'ok': True,
            'leaves': [{
                'tau_lo': job_input['tau_lo'],
                'tau_hi': job_input['tau_hi'],
                'value': ex.arb_packet(ex.fraction_arb(-1)),
                'sign': '<0',
            }],
            'failure': None,
            'runtime_milliseconds': 0,
        }
        data = block3bc._replay_job_record(
            'b_pos', result, job_input, 'a' * 64, {})
        data['runtime']['flint'] = 'wrong'
        data['job_sha256'] = ex.payload_sha256(data, omit=('job_sha256',))
        with self.assertRaisesRegex(ValueError, 'incompatible replay runtime'):
            block3bc._validate_replay_job(
                data, 'b_pos', 0, job_input, 'a' * 64, {})


class ProvenanceHardeningTests(unittest.TestCase):
    def test_confined_file_rejects_noncanonical_and_link_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            target = root / 'proof.json'
            target.write_bytes(b'proof')
            self.assertEqual(
                ex.resolve_regular_file_under(root, 'proof.json'), target)
            for relative in ('', '../proof.json', './proof.json',
                             'sub\\proof.json', 'C:/proof.json'):
                with self.subTest(relative=relative):
                    with self.assertRaises(ValueError):
                        ex.resolve_regular_file_under(root, relative)
            link = root / 'linked.json'
            try:
                os.symlink(target, link)
            except (OSError, NotImplementedError):
                return
            with self.assertRaisesRegex(ValueError, 'link/reparse'):
                ex.resolve_regular_file_under(root, 'linked.json')

    def test_isolated_command_disables_site_and_appends_bound_flint(self):
        runtime = ex.runtime_record(60, 1)
        command = ex.isolated_python_command(
            ex.__file__, runtime, ['--help'])
        self.assertEqual(command[1:5], ['-I', '-S', '-B', '-c'])
        self.assertEqual(command[5], ex.ISOLATED_BOOTSTRAP)
        self.assertIn('sys.path.insert(0,root)', command[5])
        self.assertIn('sys.path.append(site)', command[5])

    def test_aux_job_binds_source_after_and_containing_runtime(self):
        import block3bc_aux_generate as aux

        job_input = {'lambda_value': ex.fraction_record(Fraction(-1, 10))}
        result = {
            'index': 0,
            'lambda_value': job_input['lambda_value'],
            'value': ex.arb_packet(ex.fraction_arb(0)),
            'ok': True,
            'runtime_milliseconds': 0,
        }
        hashes = ex.source_hashes(aux._source_paths())
        saved = aux._job_record('k_grid', result, job_input, hashes)
        changed = copy.deepcopy(saved)
        changed['source_sha256_after']['dsfun.py'] = '0' * 64
        changed['job_sha256'] = ex.payload_sha256(
            changed, omit=('job_sha256',))
        with self.assertRaisesRegex(ValueError, 'stale/corrupt'):
            aux._validate_job_record(
                changed, 'k_grid', 0, job_input, hashes,
                require_current_runtime=False)

        foreign = ex.runtime_record(100, 2)
        foreign['host'] += '-different'
        with self.assertRaisesRegex(ValueError, 'job/shard runtime'):
            aux._validate_job_record(
                saved, 'k_grid', 0, job_input, hashes,
                require_current_runtime=False, containing_runtime=foreign)

    def test_boolean_workers_fail_before_auxiliary_work(self):
        import block3bc_aux_generate as aux

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, 'plain integer'):
                aux.generate_shard(
                    'ell_prime', workers=True,
                    output=os.path.join(td, 'ell.json'))


class SweepManifestTests(unittest.TestCase):
    def _payload(self, module, stage):
        if stage == 1:
            jobs = module.build_jobs(48)[0]
            kind = 'huang_sweep_manifest'
            policy = {'coarse': 48, 'A1MAX': repr(module.A1MAX),
                      'A2MAX': repr(module.A2MAX),
                      'MIN_SIDE': repr(module.MIN_SIDE),
                      'MAX_DEPTH': module.MAX_DEPTH}
        else:
            jobs, n1, n2 = module.build_jobs()
            kind = 'huang_sweep2_manifest'
            policy = {'n1': n1, 'n2': n2,
                      'EXCL_OLD': [repr(x) for x in module.EXCL_OLD],
                      'MIN_SIDE': repr(module.MIN_SIDE),
                      'SAFE': repr(module.SAFE)}
        schedule = [[repr(x) for x in job] for job in jobs]
        rows = [dict(index=i, cell=cell, ok=True, leaves=1, worst=None,
                     runtime_milliseconds=0)
                for i, cell in enumerate(schedule)]
        payload = {'schema_version': 1, 'kind': kind,
                   'source_sha256': ex.source_hashes(module.proof_source_paths()),
                   'runtime': ex.runtime_record(50, 1), 'policy': policy,
                   'schedule': schedule,
                   'schedule_sha256': ex.payload_sha256(schedule, omit=()),
                   'records': rows, 'total_leaves': len(rows), 'failures': 0}
        payload['artifact_sha256'] = ex.payload_sha256(
            payload, omit=('artifact_sha256',))
        return payload

    def test_legacy_sweep_summary_schemas_are_rejected(self):
        import huang_sweep
        import huang_sweep2
        import verify_all

        with tempfile.TemporaryDirectory() as td:
            for stage, module in ((1, huang_sweep), (2, huang_sweep2)):
                path = os.path.join(td, f'sweep{stage}.json')
                payload = self._payload(module, stage)
                ex.write_json_atomic(path, payload)
                ok, why, _ = verify_all.validate_sweep_manifest(
                    path, module, stage)
                self.assertFalse(ok, why)
                self.assertIn('invalid sweep certificate', why)
                payload['records'][0]['ok'] = False
                payload['artifact_sha256'] = ex.payload_sha256(
                    payload, omit=('artifact_sha256',))
                ex.write_json_atomic(path, payload)
                ok, _, _ = verify_all.validate_sweep_manifest(
                    path, module, stage)
                self.assertFalse(ok)


if __name__ == '__main__':
    unittest.main()
