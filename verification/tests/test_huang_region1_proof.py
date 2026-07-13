import copy
import os
import pathlib
import sys
import tempfile
import unittest
from fractions import Fraction


HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import block3bc_exact as exact  # noqa: E402
import huang_region1 as region1  # noqa: E402
import huang_region1_verify as verifier  # noqa: E402
import verify_all  # noqa: E402


class RegionILeafProofTests(unittest.TestCase):
    @staticmethod
    def _packet(value):
        return exact.arb_packet(exact.fraction_arb(Fraction(value)))

    def _certified_node(self):
        lo = region1._float_fraction(0.0)
        hi = region1._float_fraction(0.001)
        t0 = Fraction(0)
        t1 = region1._float_fraction(0.001)
        tmax = verifier._canonical_tmax(lo, hi)
        end = min(t1, tmax)
        theta_box = (region1._dec(float(lo), 8).union(
            region1._dec(float(hi), 8))
            + region1._ANG_PAD.union(-region1._ANG_PAD))
        radius_box = (region1._dec(float(t0), 12).union(
            region1._dec(float(end), 12))
            + region1._RAD_PAD.union(-region1._RAD_PAD))
        sdot = region1._sdot_box(theta_box.cos(), theta_box.sin())
        s_box = (region1._dec(region1.S0F, 10) + region1._ORIG
                 + radius_box * sdot)
        node = {
            'theta_lo': exact.fraction_record(lo),
            'theta_hi': exact.fraction_record(hi),
            'kind': 'certified',
            'tmax': exact.fraction_record(tmax),
            'radial_end': exact.fraction_record(end),
            'theta_box': exact.arb_packet(theta_box),
            'qB_packet': self._packet(1),
            'sdot_packet': exact.arb_packet(sdot),
            'radial_pieces': [{
                'radius_lo': exact.fraction_record(t0),
                'radius_hi': exact.fraction_record(end),
                'radius_box': exact.arb_packet(radius_box),
                's_box': exact.arb_packet(s_box),
                'curvature_packet': self._packet(-1),
            }],
        }
        return node, lo, hi, t0, t1

    def test_exact_radial_leaf_passes_and_zero_curvature_fails(self):
        node, lo, hi, t0, t1 = self._certified_node()
        self.assertEqual(
            verifier._validate_angle_node(node, lo, hi, t0, t1),
            (1, 1, 0))
        bad = copy.deepcopy(node)
        bad['radial_pieces'][0]['curvature_packet'] = self._packet(0)
        with self.assertRaisesRegex(ValueError, 'negativity'):
            verifier._validate_angle_node(bad, lo, hi, t0, t1)

    def test_radial_gap_and_false_outside_star_fail(self):
        node, lo, hi, t0, t1 = self._certified_node()
        bad = copy.deepcopy(node)
        bad['radial_pieces'][0]['radius_lo'] = exact.fraction_record(
            Fraction(1, 1000000))
        with self.assertRaisesRegex(ValueError, 'gap/overlap'):
            verifier._validate_angle_node(bad, lo, hi, t0, t1)

        outside = {
            'theta_lo': exact.fraction_record(lo),
            'theta_hi': exact.fraction_record(hi),
            'kind': 'outside_star',
            'tmax': exact.fraction_record(verifier._canonical_tmax(lo, hi)),
        }
        with self.assertRaisesRegex(ValueError, 'outside-star'):
            verifier._validate_angle_node(outside, lo, hi, t0, t1)

    def test_angular_split_uses_producer_rounded_midpoint(self):
        lo = region1._float_fraction(0.005)
        hi = region1._float_fraction(0.0087)
        midpoint = region1._float_fraction(
            0.5 * (float(lo) + float(hi)))
        self.assertNotEqual(midpoint, (lo + hi) / 2)

        def outside(left, right):
            return {
                'theta_lo': exact.fraction_record(left),
                'theta_hi': exact.fraction_record(right),
                'kind': 'outside_star',
                'tmax': exact.fraction_record(
                    verifier._canonical_tmax(left, right)),
            }

        tree = {
            'theta_lo': exact.fraction_record(lo),
            'theta_hi': exact.fraction_record(hi),
            'kind': 'split',
            'split_at': exact.fraction_record(midpoint),
            'children': [outside(lo, midpoint), outside(midpoint, hi)],
        }
        self.assertEqual(
            verifier._validate_angle_node(
                tree, lo, hi, Fraction(1), Fraction(2)),
            (0, 0, 2))

        wrong = copy.deepcopy(tree)
        rational_midpoint = (lo + hi) / 2
        wrong['split_at'] = exact.fraction_record(rational_midpoint)
        wrong['children'] = [
            outside(lo, rational_midpoint), outside(rational_midpoint, hi)]
        with self.assertRaisesRegex(ValueError, 'split point'):
            verifier._validate_angle_node(
                wrong, lo, hi, Fraction(1), Fraction(2))

    def test_edge_prefix_cover_and_convex_weights(self):
        one = exact.fraction_record(Fraction(1))
        zero = exact.fraction_record(Fraction(0))
        leaves = []
        for edge in range(4):
            for base in range(8):
                leaves.append({
                    'edge': edge,
                    'base_segment': base,
                    'path': '',
                    'polygon_certificates': [{
                        'weights': [one],
                        'lower_packet': self._packet(1),
                    }],
                })
        certificate = {
            'fan': [[zero, zero]],
            'polygons': [[[zero, zero]]],
            'initial_segments_per_edge': 8,
            'max_depth': 14,
            'leaves': leaves,
        }
        self.assertEqual(
            verifier._validate_localization(
                certificate,
                (Fraction(-1), Fraction(1), Fraction(-1), Fraction(1))),
            certificate['polygons'])
        bad = copy.deepcopy(certificate)
        bad['leaves'][0]['polygon_certificates'][0]['weights'] = [zero]
        with self.assertRaisesRegex(ValueError, 'convex weights'):
            verifier._validate_localization(bad)
        bad = copy.deepcopy(certificate)
        bad['leaves'][0]['path'] = '0'
        with self.assertRaisesRegex(ValueError, 'gap'):
            verifier._validate_localization(bad)

        bad = copy.deepcopy(certificate)
        bad['fan'][0][0] = exact.fraction_record(Fraction(2))
        with self.assertRaisesRegex(ValueError, 'escapes'):
            verifier._validate_localization(
                bad,
                (Fraction(-1), Fraction(1), Fraction(-1), Fraction(1)))
        bad = copy.deepcopy(certificate)
        bad['fan'][0][0] = exact.fraction_record(Fraction(-1))
        with self.assertRaisesRegex(ValueError, 'escapes'):
            verifier._validate_localization(
                bad,
                (Fraction(-1), Fraction(1), Fraction(-1), Fraction(1)))

    def test_full_circle_hull_uses_persisted_single_polygon_shape(self):
        region1.set_prec(50)
        values = (0.0, 1e-6, 0.0, 2 * region1.math.pi)
        job = tuple(
            verifier._fraction(region1._float_record(value))
            for value in values)
        canonical_values = tuple(verifier._float(value) for value in job)
        flat_polygon, hull = region1.xhull_of_band(*canonical_values)
        self.assertIsInstance(flat_polygon[0][0], float)
        polygons = [[
            [region1._decimal_record(f'{float(x):.12f}'),
             region1._decimal_record(f'{float(y):.12f}')]
            for x, y in flat_polygon
        ]]
        persisted_hull = dict(hull)
        persisted_hull['polygons'] = polygons
        verifier._validate_hull(job, persisted_hull, polygons)
        bad_polygons = copy.deepcopy(polygons)
        bad_polygons[0][0][0] = region1._decimal_record(
            str(float(canonical_values[1]) + 1))
        bad_hull = dict(hull)
        bad_hull['polygons'] = bad_polygons
        with self.assertRaisesRegex(
                ValueError, 'localization polygon is not canonical'):
            verifier._validate_hull(job, bad_hull, bad_polygons)

    def test_numeric_replay_rejects_a_same_sign_packet_tamper(self):
        positive_value = exact.fraction_arb(Fraction(2))
        verifier._require_packet(
            positive_value, exact.arb_packet(positive_value), 'fixture')
        positive_but_wrong = exact.arb_packet(
            exact.fraction_arb(Fraction(3)))
        with self.assertRaisesRegex(ValueError, 'numerical replay mismatch'):
            verifier._require_packet(
                positive_value, positive_but_wrong, 'same-sign tamper')

    def test_B_packets_and_root_guards_are_geometry_bound(self):
        lbox = (Fraction(1), Fraction(2), Fraction(0), Fraction(1))
        l1lo = exact.fraction_arb(lbox[0])
        l2lo = exact.fraction_arb(lbox[2])
        far = exact.fraction_arb(5)
        x = region1.SQ_PSI_B * far
        g_far = l1lo * x + l2lo * x.tanh()
        certificate = {
            'b11_packet': self._packet(2),
            'b12_packet': self._packet(0),
            'b22_packet': self._packet(2),
            'det_packet': self._packet(4),
            'root_certificate': {
                'mode': 'l2_nonnegative',
                'zq': exact.fraction_record(Fraction(0)),
                'g_zq_packet': self._packet(0),
                'condition_packet': exact.arb_packet(l2lo),
                'g_far_packet': exact.arb_packet(g_far),
            },
        }
        verifier._validate_B(certificate, lbox)

        indefinite = copy.deepcopy(certificate)
        indefinite['b12_packet'] = self._packet(3)
        with self.assertRaisesRegex(ValueError, 'definiteness'):
            verifier._validate_B(indefinite, lbox)

        wrong_mode = copy.deepcopy(certificate)
        wrong_mode['root_certificate']['mode'] = 'derivative_nonnegative'
        with self.assertRaisesRegex(ValueError, 'root mode'):
            verifier._validate_B(wrong_mode, lbox)

    def test_old_summary_schema_is_rejected_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'old.json')
            exact.write_json_atomic(path, {
                'schema_version': 2,
                'results': [{'ok': True}],
                'fails': 0,
            })
            ok, reason = verify_all.validate_region1_manifest(path)
            self.assertFalse(ok)
            self.assertIn('invalid Region-I certificate', reason)


if __name__ == '__main__':
    unittest.main()
