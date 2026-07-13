import pathlib
import sys
import unittest
from unittest import mock

from flint import arb


HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import dsfun  # noqa: E402


class IPrimeTailTests(unittest.TestCase):
    @staticmethod
    def b32(lam):
        return (1 - lam * lam) ** arb('-1.5')

    def test_negative_threshold_uses_full_absolute_moments(self):
        lam = arb('-0.424')
        with mock.patch.object(
                dsfun, '_poly_abs_gauss_full',
                wraps=dsfun._poly_abs_gauss_full) as full, \
                mock.patch.object(
                    dsfun, '_poly_gauss_tail',
                    wraps=dsfun._poly_gauss_tail) as one_sided:
            bound = dsfun._iprime_tail(lam, self.b32(lam))
        self.assertTrue(bound > 0)
        self.assertGreater(full.call_count, 0)
        for call in one_sided.call_args_list:
            self.assertTrue(call.args[0] > 0, str(call.args[0]))

    def test_certified_narrow_domain_keeps_one_sided_path(self):
        lam = arb('-0.12')
        with mock.patch.object(
                dsfun, '_poly_abs_gauss_full',
                wraps=dsfun._poly_abs_gauss_full) as full:
            bound = dsfun._iprime_tail(lam, self.b32(lam))
        self.assertTrue(bound > 0)
        full.assert_not_called()


if __name__ == '__main__':
    unittest.main()
