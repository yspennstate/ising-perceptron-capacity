import copy
import pathlib
import sys
import tempfile
import unittest
from fractions import Fraction
from unittest import mock

from flint import arb


HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import block3bc_exact as exact  # noqa: E402
import huang_region1_verify as region1_verify  # noqa: E402
import huang_sweep as sweep1  # noqa: E402
import huang_sweep2 as sweep2  # noqa: E402
import huang_sweep_verify as verifier  # noqa: E402
import huanggrid as hg  # noqa: E402
import verify_all  # noqa: E402
from core import set_prec  # noqa: E402


class RegionIIProofTreeTests(unittest.TestCase):
    @staticmethod
    def _record(cell):
        return [exact.fraction_record(value) for value in cell]

    @staticmethod
    def _packet(value):
        return exact.arb_packet(exact.fraction_arb(Fraction(value)))

    @classmethod
    def _negative_leaf(cls, cell, value=-1):
        return {
            'kind': 'negative',
            'cell': cls._record(cell),
            'method': 'direct',
            'majorant_witness': {
                'dual_mode': 'entropy_cap',
                'tilt_s': exact.fraction_record(Fraction(0)),
            },
            'value_packet': cls._packet(value),
        }

    def test_declared_geometry_matches_certified_decimal_endpoints(self):
        self.assertEqual(sweep1._float_fraction(-1.31), Fraction('-1.31'))
        self.assertEqual(sweep1._float_fraction(0.1), Fraction('0.1'))
        self.assertEqual(sweep1._fixed_decimal(1.1e-6, 6), '0.000001')

    def test_sweep1_main_rejects_noncanonical_schedule_before_work(self):
        with mock.patch.object(sys, 'argv', ['huang_sweep.py', '1', '24']):
            with self.assertRaisesRegex(ValueError, 'coarse=48'):
                sweep1.main()

    def test_mean_value_quantization_cannot_cross_lambda_policy(self):
        with mock.patch.object(
                sweep1.nr, 'dual_of', return_value=(299.999999999, 0.0)), \
                mock.patch.object(sweep1.nr, 'G', return_value=(1.0, 0.0)):
            value, witness = sweep1.eval_cell_mv(
                0.0, 0.001, 0.0, 0.001, return_witness=True)
        self.assertIsNone(value)
        self.assertIsNone(witness)

    def test_interior_split_and_exact_child_tiling(self):
        parent = (Fraction(0), Fraction(1), Fraction(0), Fraction(1))
        split = Fraction(1, 3)
        tree = {
            'kind': 'split',
            'cell': self._record(parent),
            'axis': 1,
            'split_at': exact.fraction_record(split),
            'children': [
                self._negative_leaf((parent[0], split, parent[2], parent[3])),
                self._negative_leaf((split, parent[1], parent[2], parent[3])),
            ],
        }
        with mock.patch.object(verifier, '_replay_negative_leaf'):
            self.assertEqual(verifier._validate_tree(tree, parent, 1, []),
                             (2, 0, 0))

            gap = copy.deepcopy(tree)
            gap['children'][1]['cell'][0] = exact.fraction_record(
                Fraction(1, 2))
            with self.assertRaisesRegex(ValueError, 'geometry'):
                verifier._validate_tree(gap, parent, 1, [])

        boundary = copy.deepcopy(tree)
        boundary['split_at'] = exact.fraction_record(parent[0])
        with self.assertRaisesRegex(ValueError, 'split point'):
            verifier._validate_tree(boundary, parent, 1, [])

    def test_negative_packet_must_be_strict(self):
        cell = (Fraction(0), Fraction(1), Fraction(0), Fraction(1))
        leaf = self._negative_leaf(cell, value=0)
        with self.assertRaisesRegex(ValueError, 'strictly negative'):
            verifier._validate_tree(leaf, cell, 1, [])

    def test_direct_negative_majorant_replays_and_binds_choices(self):
        set_prec(50)
        cell = (-0.10, -0.05, -0.10, -0.05)
        value, witness = sweep1.eval_cell(*cell, return_witness=True)
        self.assertIsNotNone(value)
        self.assertLess(value, 0)
        rectangle = tuple(sweep1._float_fraction(x) for x in cell)
        tree = {
            'kind': 'negative',
            'cell': sweep1.cell_record(cell),
            'method': 'direct',
            'majorant_witness': witness,
            'value_packet': exact.arb_packet(value),
        }
        verifier._replay_negative_leaf(tree, rectangle)

        changed_choice = copy.deepcopy(tree)
        lam1 = exact.fraction_from_record(
            changed_choice['majorant_witness']['lambda1'])
        changed_choice['majorant_witness']['lambda1'] = \
            exact.fraction_record(lam1 + Fraction(1, 10 ** 6))
        with self.assertRaisesRegex(ValueError, 'does not replay'):
            verifier._replay_negative_leaf(changed_choice, rectangle)

        changed_packet = copy.deepcopy(tree)
        changed_packet['value_packet'] = self._packet(-2)
        with self.assertRaisesRegex(ValueError, 'does not replay'):
            verifier._replay_negative_leaf(changed_packet, rectangle)

        method_flip = copy.deepcopy(tree)
        method_flip['method'] = 'mean_value'
        with self.assertRaisesRegex(ValueError, 'schema mismatch'):
            verifier._replay_negative_leaf(method_flip, rectangle)

        changed_rectangle = list(rectangle)
        changed_rectangle[0] -= Fraction(1, 1000)
        with self.assertRaisesRegex(ValueError, 'does not replay'):
            verifier._replay_negative_leaf(tree, tuple(changed_rectangle))

        overprecision = copy.deepcopy(tree)
        overprecision['majorant_witness']['tilt_s'] = exact.fraction_record(
            exact.fraction_from_record(witness['tilt_s'])
            + Fraction(1, 10 ** 7))
        with self.assertRaisesRegex(ValueError, 'decimal precision'):
            verifier._replay_negative_leaf(overprecision, rectangle)

        noncanonical = copy.deepcopy(tree)
        noncanonical['majorant_witness']['lambda1'] = {
            'num': '2', 'den': '2'}
        with self.assertRaisesRegex(ValueError, 'reduced/canonical'):
            verifier._replay_negative_leaf(noncanonical, rectangle)

        missing = copy.deepcopy(tree)
        del missing['majorant_witness']['lambda2']
        with self.assertRaisesRegex(ValueError, 'schema mismatch'):
            verifier._replay_negative_leaf(missing, rectangle)

        bad_cap = copy.deepcopy(tree)
        bad_cap['majorant_witness']['dual_mode'] = 'entropy_cap'
        with self.assertRaisesRegex(ValueError, 'schema mismatch'):
            verifier._replay_negative_leaf(bad_cap, rectangle)

    def test_mean_value_negative_majorant_replays(self):
        set_prec(50)
        cell = (sweep1.A1S + 0.002, sweep1.A1S + 0.0025,
                sweep1.A2S, sweep1.A2S + 0.0005)
        value, witness = sweep1.eval_cell_mv(*cell, return_witness=True)
        self.assertIsNotNone(value)
        self.assertLess(value, 0)
        rectangle = tuple(sweep1._float_fraction(x) for x in cell)
        tree = {
            'kind': 'negative',
            'cell': sweep1.cell_record(cell),
            'method': 'mean_value',
            'majorant_witness': witness,
            'value_packet': exact.arb_packet(value),
        }
        verifier._replay_negative_leaf(tree, rectangle)

        changed = copy.deepcopy(tree)
        changed['majorant_witness']['tilt_s'] = exact.fraction_record(
            exact.fraction_from_record(witness['tilt_s'])
            + Fraction(1, 10 ** 8))
        with self.assertRaisesRegex(ValueError, 'does not replay'):
            verifier._replay_negative_leaf(changed, rectangle)

    def test_replay_rejects_a_nonnegative_majorant(self):
        from flint import arb
        from core import ALPHA, PSI

        set_prec(50)
        rectangle = (Fraction('-0.01'), Fraction('0.01'),
                     Fraction('-0.01'), Fraction('0.01'))
        a1lo, a1hi, a2lo, a2hi = (
            exact.fraction_arb(value) for value in rectangle)
        cc1, rr1 = (a1lo + a1hi) / 2, (a1hi - a1lo) / 2
        cc2, rr2 = (a2lo + a2hi) / 2, (a2hi - a2lo) / 2
        s = exact.fraction_arb(0)
        t_bound = hg.T_meanvalue(cc1, cc2, rr1, rr2, s)
        value = arb(0).union(hg.LOG2) + s * s * PSI / 2 + ALPHA * t_bound
        self.assertFalse(value < 0)
        tree = {
            'kind': 'negative',
            'cell': self._record(rectangle),
            'method': 'direct',
            'majorant_witness': {
                'dual_mode': 'entropy_cap',
                'tilt_s': exact.fraction_record(Fraction(0)),
            },
            'value_packet': exact.arb_packet(value),
        }
        with self.assertRaisesRegex(ValueError, 'not strictly negative'):
            verifier._replay_negative_leaf(tree, rectangle)

    def test_legacy_negative_leaf_without_witness_is_rejected(self):
        cell = (Fraction(0), Fraction(1), Fraction(0), Fraction(1))
        leaf = self._negative_leaf(cell)
        del leaf['majorant_witness']
        with self.assertRaisesRegex(ValueError, 'schema mismatch'):
            verifier._validate_tree(leaf, cell, 1, [])

    def test_both_stages_emit_replayable_negative_leaves(self):
        set_prec(50)
        cell = (-0.10, -0.05, -0.10, -0.05)
        rectangle = tuple(sweep1._float_fraction(x) for x in cell)
        with mock.patch.object(
                hg, 'outside_K_witness', return_value=None):
            first = sweep1.cert_cell(cell)
            second = sweep2.cert_cell(cell)
        for stage, tree in ((1, first), (2, second)):
            self.assertEqual(tree['kind'], 'negative')
            self.assertIn('majorant_witness', tree)
            self.assertEqual(
                verifier._validate_tree(tree, rectangle, stage, []),
                (1, 0, 0))

    def test_outside_witness_is_bound_to_fan_and_rectangle(self):
        from flint import ctx

        set_prec(50)
        rectangle = (Fraction(2), Fraction(3), Fraction(0), Fraction(1))
        u = exact.fraction_arb(1)
        v = exact.fraction_arb(0)
        cache_key = (0, ctx.prec, hg.GRID_N)
        old = verifier._OUTSIDE_FAN_CACHE.get(cache_key)
        try:
            support = exact.fraction_arb(1)
            verifier._OUTSIDE_FAN_CACHE[cache_key] = (u, v, support)
            box_min = hg._corner_min(
                u, v, exact.fraction_arb(2), exact.fraction_arb(3),
                exact.fraction_arb(0), exact.fraction_arb(1))
            witness = {
                'fan_index': 0,
                'sign': 1,
                'u_packet': exact.arb_packet(u),
                'v_packet': exact.arb_packet(v),
                'support_upper_packet': exact.arb_packet(support),
                'box_min_packet': exact.arb_packet(box_min),
            }
            verifier._outside_witness(witness, rectangle)

            forged = copy.deepcopy(witness)
            forged['box_min_packet'] = self._packet(3)
            with self.assertRaisesRegex(ValueError, 'canonical'):
                verifier._outside_witness(forged, rectangle)

            support = exact.fraction_arb(2)
            verifier._OUTSIDE_FAN_CACHE[cache_key] = (u, v, support)
            equality = copy.deepcopy(witness)
            equality['support_upper_packet'] = exact.arb_packet(support)
            with self.assertRaisesRegex(ValueError, 'not strict'):
                verifier._outside_witness(equality, rectangle)
        finally:
            if old is None:
                verifier._OUTSIDE_FAN_CACHE.pop(cache_key, None)
            else:
                verifier._OUTSIDE_FAN_CACHE[cache_key] = old

    def test_star_witness_is_bound_to_its_rectangle(self):
        set_prec(50)
        eps = 1e-5
        cell = (sweep2.A1S - eps, sweep2.A1S + eps,
                sweep2.A2S - eps, sweep2.A2S + eps)
        rectangle = tuple(sweep1._float_fraction(value) for value in cell)
        witness = sweep2.star_witness(*cell)
        self.assertIsNotNone(witness)
        verifier._star_witness(witness, rectangle)
        forged = copy.deepcopy(witness)
        forged['radius_max_packet'] = self._packet(0)
        with self.assertRaisesRegex(ValueError, 'canonical'):
            verifier._star_witness(forged, rectangle)

    def test_top_schedules_tile_their_declared_rectangles(self):
        def check(records, n1, n2, bounds):
            self.assertEqual(len(records), n1 * n2)
            cells = [verifier._cell(record['cell']) for record in records]
            self.assertEqual((cells[0][0], cells[-1][1],
                              cells[0][2], cells[n2 - 1][3]), bounds)
            for i in range(n1):
                for j in range(n2 - 1):
                    self.assertEqual(cells[i * n2 + j][3],
                                     cells[i * n2 + j + 1][2])
            for i in range(n1 - 1):
                for j in range(n2):
                    self.assertEqual(cells[i * n2 + j][1],
                                     cells[(i + 1) * n2 + j][0])

        records1 = sweep1.schedule_records(48)
        verifier._validate_top_schedule(records1, 1)
        check(records1, 48, len(records1) // 48,
              (Fraction(str(-sweep1.A1MAX)), Fraction(str(sweep1.A1MAX)),
               Fraction(str(-sweep1.A2MAX)), Fraction(str(sweep1.A2MAX))))
        records2 = sweep2.schedule_records()
        verifier._validate_top_schedule(records2, 2)
        check(records2, 20, 12,
              tuple(Fraction(str(value)) for value in sweep2.EXCL_OLD))

        gap = copy.deepcopy(records2)
        gap[1]['cell'][2] = exact.fraction_record(Fraction('0.447'))
        with self.assertRaisesRegex(ValueError, 'tensor grid|gap'):
            verifier._validate_top_schedule(gap, 2)

    def test_old_summary_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            for stage, module in ((1, sweep1), (2, sweep2)):
                path = pathlib.Path(td) / f'old-sweep{stage}.json'
                exact.write_json_atomic(path, {
                    'schema_version': 1,
                    'kind': f'huang_sweep{stage}_manifest',
                    'failures': 0,
                    'records': [{'ok': True}],
                })
                ok, reason, _ = verify_all.validate_sweep_manifest(
                    path, module, stage)
                self.assertFalse(ok)
                self.assertIn('invalid sweep certificate', reason)

    def test_pair_rejects_escaped_delegate_and_region1_policy_mismatch(self):
        domain = [exact.fraction_record(value) for value in
                  (Fraction(0), Fraction(1), Fraction(0), Fraction(1))]
        second = {
            'policy': {
                'EXCL_OLD': domain,
                'region1_star': sweep2.region1_star_policy(),
                'region1_sdot_model': 'model',
                'region1_sdot_coefficients': [],
            },
        }
        region1 = {
            'star': sweep2.region1_star_policy(),
            'certificate_policy': {
                'sdot_model': 'model',
                'sdot_coefficients': [],
            },
        }
        escaped = (Fraction(0), Fraction(2), Fraction(0), Fraction(1))
        with mock.patch.object(
                verifier, 'verify_certificate',
                side_effect=[({}, [escaped]), (second, [])]), \
                mock.patch.object(
                    region1_verify, 'verify_certificate', return_value=region1):
            with self.assertRaisesRegex(ValueError, 'escapes'):
                verifier.verify_pair('s1', 's2', 'r1')

        mismatch = copy.deepcopy(region1)
        mismatch['certificate_policy']['sdot_model'] = 'different'
        with mock.patch.object(
                verifier, 'verify_certificate',
                side_effect=[({}, []), (second, [])]), \
                mock.patch.object(
                    region1_verify, 'verify_certificate', return_value=mismatch):
            with self.assertRaisesRegex(ValueError, 'policy mismatch'):
                verifier.verify_pair('s1', 's2', 'r1')

    def test_canonical_bundle_binds_all_four_artifact_identities(self):
        def artifact(schema, kind, evidence, digit):
            return {
                'schema_version': schema,
                'kind': kind,
                'evidence_model': evidence,
                'run_id': digit * 32,
                'source_set_sha256': digit * 64,
                'certificate_sha256': digit * 64,
            }

        first = artifact(
            sweep1.SCHEMA_VERSION, 'huang_sweep1_certificate',
            sweep1.EVIDENCE_MODEL, '1')
        second = artifact(
            sweep2.SCHEMA_VERSION, 'huang_sweep2_certificate',
            sweep2.EVIDENCE_MODEL, '2')
        region1 = artifact(
            3, 'huang_region1_certificate',
            'exact-leaf-proof-tree-v1', '3')
        region1.update({
            'star': {
                'T_LONG': exact.fraction_record(Fraction('0.012')),
                'origin_radius': exact.fraction_record(Fraction('0.0000001')),
            },
            'records': [{
                'angular_roots': [{
                    'kind': 'certified',
                    'theta_box': exact.arb_packet(
                        arb('0').union(arb('0.016'))),
                    'radial_pieces': [{
                        'radius_box': exact.arb_packet(
                            arb('0').union(arb('0.012'))),
                    }],
                }],
            }],
        })
        star_interior = {
            'schema_version': 1,
            'kind': 'huang_star_interior_certificate',
            'source_sha256': {'huang_star_interior.py': '4' * 64},
            'certificate_sha256': '4' * 64,
            'policy': {
                'star_radius': '0.012',
                'origin_coordinate_radius': '1/10000000',
                'center_interval_diameter_factor': 2,
            },
            'bounds': {
                'required_radius': exact.arb_packet(arb('0.0122')),
                'inradius': exact.arb_packet(arb('0.015')),
            },
        }
        rectangle = (Fraction(0), Fraction(1), Fraction(0), Fraction(1))
        components = (first, second, region1, [rectangle],
                      [rectangle, rectangle])
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            artifacts = {
                'star_interior': star_interior,
                'region1': region1,
                'sweep1': first,
                'sweep2': second,
            }
            for name, filename in verifier._BUNDLE_FILES.items():
                exact.write_json_atomic(root / filename, artifacts[name])
            bundle_path = root / 'huang_bundle.json'
            with mock.patch.object(
                    verifier, 'verify_pair', return_value=components), \
                    mock.patch(
                        'huang_star_interior.verify_certificate',
                        return_value=star_interior):
                written = verifier.write_bundle(
                    bundle_path, root / 'huang_star_interior.json',
                    root / 'huang_sweep.json',
                    root / 'huang_sweep2.json',
                    root / 'huang_region1.json')
                checked, returned = verifier.verify_bundle(bundle_path)
            self.assertEqual(checked, written)
            self.assertEqual(returned, (star_interior, components))
            self.assertEqual(
                written['delegation_summary'], {
                    'sweep1_to_sweep2_rectangles': 1,
                    'sweep2_to_region1_rectangles': 2,
                })
            self.assertEqual(
                written['interior_geometry_link']['certified_angular_leaves'],
                1)
            with mock.patch.object(
                    verifier, 'verify_pair',
                    side_effect=AssertionError('must not replay')):
                cached, returned = verifier.verify_bundle(
                    bundle_path, (star_interior, components))
            self.assertEqual(cached, written)
            self.assertEqual(returned, (star_interior, components))

            alias = root / 'sweep1-alias.json'
            exact.write_json_atomic(alias, first)
            with self.assertRaisesRegex(
                    ValueError, 'canonical sibling artifacts'):
                verifier.write_bundle(
                    bundle_path, root / 'huang_star_interior.json',
                    alias, root / 'huang_sweep2.json',
                    root / 'huang_region1.json')

            changed = copy.deepcopy(first)
            changed['run_id'] = 'a' * 32
            exact.write_json_atomic(root / 'huang_sweep.json', changed)
            with self.assertRaisesRegex(
                    ValueError, 'prevalidated Huang artifact differs'):
                verifier.verify_bundle(
                    bundle_path, (star_interior, components))
            exact.write_json_atomic(root / 'huang_sweep.json', first)

            tampered = copy.deepcopy(written)
            tampered['artifacts']['sweep1']['certificate_sha256'] = 'f' * 64
            tampered['bundle_sha256'] = exact.payload_sha256(
                tampered, omit=('bundle_sha256',))
            exact.write_json_atomic(bundle_path, tampered)
            with mock.patch.object(
                    verifier, 'verify_pair', return_value=components):
                with self.assertRaisesRegex(ValueError, 'does not bind'):
                    verifier.verify_bundle(
                        bundle_path, (star_interior, components))

    def test_bundle_geometry_link_fails_closed_on_undersized_inball(self):
        theta = exact.arb_packet(arb('0').union(arb('0.016')))
        radius = exact.arb_packet(arb('0').union(arb('0.012')))
        region1 = {
            'star': {
                'T_LONG': exact.fraction_record(Fraction('0.012')),
                'origin_radius': exact.fraction_record(Fraction('0.0000001')),
            },
            'records': [{'angular_roots': [{
                'kind': 'certified',
                'theta_box': theta,
                'radial_pieces': [{'radius_box': radius}],
            }]}],
        }
        star = {
            'policy': {
                'star_radius': '0.012',
                'origin_coordinate_radius': '1/10000000',
                'center_interval_diameter_factor': 2,
            },
            'bounds': {
                'required_radius': exact.arb_packet(arb('0.0122')),
                'inradius': exact.arb_packet(arb('0.015')),
            },
        }
        link = verifier._verify_star_interior_link(star, region1)
        self.assertEqual(link['certified_angular_leaves'], 1)
        self.assertEqual(link['radial_pieces'], 1)
        star['bounds']['required_radius'] = exact.arb_packet(arb('0.011'))
        with self.assertRaisesRegex(ValueError, 'escape'):
            verifier._verify_star_interior_link(star, region1)


if __name__ == '__main__':
    unittest.main()
