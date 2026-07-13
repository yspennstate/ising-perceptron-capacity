import math
import pathlib
import sys
import unittest

from flint import arb


HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import huang_region1 as region1  # noqa: E402
import block3bc_exact as exact  # noqa: E402


def _exact_float(value):
    numerator, denominator = value.as_integer_ratio()
    return arb(numerator) / arb(denominator)


def _expected_arcs(t0):
    if t0 < region1.T_CORE:
        return [(0.0, 2 * math.pi)]
    if t0 < region1.T_MID:
        radius = region1.CONE_MID
    elif t0 < region1.T_LONG:
        radius = region1.WEDGE_HALF
    else:
        return []
    return [
        (region1.W_ANG - radius, region1.W_ANG + radius),
        (region1.W_ANG + math.pi - radius,
         region1.W_ANG + math.pi + radius),
    ]


def _tmax_oracle(a, b):
    if b - a >= 2 * math.pi - 1e-12:
        return region1.T_LONG
    centers = [region1.W_ANG + k * math.pi for k in range(-10, 11)]
    if any(a <= center + region1.WEDGE_HALF
           and b >= center - region1.WEDGE_HALF for center in centers):
        return region1.T_LONG
    if any(a <= center + region1.CONE_MID
           and b >= center - region1.CONE_MID for center in centers):
        return region1.T_MID
    return region1.T_CORE


def _all_possible_angular_cells(jobs):
    roots = set()
    for _, _, a, b in jobs:
        count = max(1, int(math.ceil(
            (b - a) / (2 * math.pi / region1.N_ANG))))
        for k in range(count):
            roots.add((a + (b - a) * k / count,
                       a + (b - a) * (k + 1) / count))
    cells = set()

    def add(a, b):
        if (a, b) in cells:
            return
        cells.add((a, b))
        if b - a > 2e-3:
            midpoint = (a + b) / 2
            add(a, midpoint)
            add(midpoint, b)

    for cell in roots:
        add(*cell)
    return roots, cells


class GlobalRaySlopeTests(unittest.TestCase):
    def test_coefficients_are_exact_source_constants(self):
        self.assertEqual(region1.SDOT_MODEL, 'global-linear-v1')
        self.assertEqual(region1.SDOT_C_TEXT,
                         ('-2.448324', '0.790403'))

    def test_different_angular_cells_enclose_same_true_ray_slope(self):
        theta = 0.731
        th = arb(str(theta))
        actual = (region1.SDOT_C1 * th.cos()
                  + region1.SDOT_C2 * th.sin())
        cells = ((theta - 0.02, theta + 0.01),
                 (theta - 0.004, theta + 0.006))
        for lo, hi in cells:
            thb = arb(str(lo)).union(arb(str(hi)))
            enclosure = region1._sdot_box(thb.cos(), thb.sin())
            self.assertTrue(enclosure.contains(actual),
                            (lo, hi, enclosure, actual))

    def test_fixed_rule_tracks_the_previous_tight_heuristic(self):
        for theta in (0.0, 0.4, 1.2, 2.8, 5.5):
            proposed = (float(region1.SDOT_C1.mid()) * math.cos(theta)
                        + float(region1.SDOT_C2.mid()) * math.sin(theta))
            self.assertLess(abs(proposed - region1._sdot_of(theta)), 3e-4)


class GlobalRayCoverageTests(unittest.TestCase):
    def test_radial_chain_and_raw_angular_tiling(self):
        jobs = region1.bands()
        groups = {}
        order = []
        for t0, t1, a, b in jobs:
            key = (t0, t1)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((a, b))

        self.assertEqual(len(jobs), 1404)
        self.assertEqual(len(order), 30)
        self.assertEqual(order[0][1], region1.T_LONG)
        self.assertEqual(order[-1], (0.0, region1.T_IN))
        for current, following in zip(order, order[1:]):
            self.assertEqual(current[0], following[1])
        for lo, hi in order:
            self.assertLess(lo, hi)

        for key in order:
            got = sorted(groups[key])
            used = set()
            for expected_lo, expected_hi in _expected_arcs(key[0]):
                owned = [(index, interval)
                         for index, interval in enumerate(got)
                         if expected_lo <= interval[0]
                         and interval[1] <= expected_hi]
                self.assertTrue(owned)
                self.assertEqual(owned[0][1][0], expected_lo)
                self.assertEqual(owned[-1][1][1], expected_hi)
                for left, right in zip(owned, owned[1:]):
                    self.assertEqual(left[1][1], right[1][0])
                used.update(index for index, _ in owned)
            self.assertEqual(used, set(range(len(got))))

    def test_all_possible_adaptive_cells(self):
        roots, cells = _all_possible_angular_cells(region1.bands())
        self.assertEqual(len(roots), 319)
        self.assertEqual(len(cells), 23659)
        for a, b in cells:
            self.assertEqual(region1.T_max_over(a, b), _tmax_oracle(a, b))
            theta_box = (region1._dec(a, 8).union(region1._dec(b, 8))
                         + region1._ANG_PAD.union(-region1._ANG_PAD))
            exact_a, exact_b = _exact_float(a), _exact_float(b)
            self.assertTrue(theta_box.contains(exact_a))
            self.assertTrue(theta_box.contains(exact_b))
            slope_box = region1._sdot_box(theta_box.cos(), theta_box.sin())
            for theta in (exact_a, exact_b):
                exact_slope = (region1.SDOT_C1 * theta.cos()
                               + region1.SDOT_C2 * theta.sin())
                self.assertTrue(slope_box.contains(exact_slope))

    def test_all_sector_hull_guards_are_exactly_positive(self):
        for index, job in enumerate(region1.bands()):
            polygons, certificate = region1.xhull_of_band(*job)
            self.assertTrue(polygons, index)
            if certificate['mode'] == 'full_circle_square':
                lo, _ = exact.packet_fraction_endpoints(
                    certificate['origin_padding_packet'])
                self.assertGreater(lo, 0, index)
                continue
            self.assertEqual(certificate['mode'], 'tangent_sector_cover')
            self.assertEqual(len(polygons), len(certificate['subsectors']))
            for sector in certificate['subsectors']:
                lo, _ = exact.packet_fraction_endpoints(
                    sector['cos_half_packet'])
                self.assertGreater(lo, 0, (index, sector))
                for guard in sector['tangent_slack_packets']:
                    lo, _ = exact.packet_fraction_endpoints(guard)
                    self.assertGreaterEqual(lo, 0, (index, sector))


if __name__ == '__main__':
    unittest.main()
