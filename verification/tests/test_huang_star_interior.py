import pathlib
import sys
import tempfile
import unittest
from fractions import Fraction

from flint import arb


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import block3bc_exact as exact  # noqa: E402
import huang_star_interior as interior  # noqa: E402
import verify_all  # noqa: E402


class StarInteriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = interior.compute_certificate()

    def test_constructive_radius_covers_padded_star(self):
        self.assertEqual(
            Fraction(self.payload["policy"]["star_radius"]),
            interior.region1._float_fraction(interior.region1.T_LONG))
        radius_lo, _ = exact.packet_fraction_endpoints(
            self.payload["bounds"]["inradius"])
        _, required_hi = exact.packet_fraction_endpoints(
            self.payload["bounds"]["required_radius"])
        clearance_lo, _ = exact.packet_fraction_endpoints(
            self.payload["bounds"]["clearance"])
        self.assertGreater(radius_lo, required_hi)
        self.assertGreater(required_hi, Fraction(12095, 1000000))
        self.assertLess(required_hi, Fraction(12096, 1000000))
        self.assertGreater(clearance_lo, 0)
        self.assertGreater(radius_lo, Fraction(75195991, 5000000000))

    def test_matrix_and_determinant_have_strict_schema(self):
        self.assertEqual(len(self.payload["matrix"]), 2)
        self.assertTrue(all(len(row) == 2 for row in self.payload["matrix"]))
        det_lo, _ = exact.packet_fraction_endpoints(
            self.payload["bounds"]["determinant"])
        self.assertGreater(det_lo, 0)
        self.assertEqual(
            set(self.payload["source_sha256"]),
            {"huang_star_interior.py", "huang_region1.py",
             "block3bc_exact.py", "core.py"})

    def test_region1_padding_drift_fails_closed(self):
        original = interior.region1._E7
        try:
            interior.region1._E7 = arb("0.01")
            with self.assertRaisesRegex(ValueError, "padding constants drifted"):
                interior.compute_certificate()
        finally:
            interior.region1._E7 = original

    def test_roundtrip_and_rehashed_tamper_fail(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "star.json"
            interior.write_certificate(path)
            self.assertEqual(interior.verify_certificate(path), self.payload)
            self.assertTrue(verify_all.validate_star_interior_certificate(path)[0])
            tampered = exact.load_json(path)
            tampered["policy"]["star_radius"] = "0.011"
            tampered["certificate_sha256"] = exact.payload_sha256(
                tampered, omit=("certificate_sha256",))
            exact.write_json_atomic(path, tampered)
            with self.assertRaisesRegex(ValueError, "replay exactly"):
                interior.verify_certificate(path)
            self.assertFalse(verify_all.validate_star_interior_certificate(path)[0])


if __name__ == "__main__":
    unittest.main()
