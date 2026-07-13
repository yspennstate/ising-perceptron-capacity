import contextlib
import io
import pathlib
import tempfile
import unittest

import release_secret_scan as scanner


class ReleaseSecretScanTests(unittest.TestCase):
    def run_quiet(self, paths):
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return scanner.main(paths)

    def test_no_input_and_missing_file_fail_closed(self):
        self.assertEqual(self.run_quiet([]), 2)
        self.assertEqual(
            self.run_quiet(["definitely-missing-release-input.txt"]), 2)

    def test_binary_file_is_scanned_without_utf8_bypass(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "bad.txt"
            path.write_bytes(b"\xff\xfe")
            self.assertEqual(self.run_quiet([str(path)]), 0)
            token = ("ghp_" + "Z9" * 20).encode("ascii")
            path.write_bytes(b"\xff\xfe" + token + b"\x00")
            self.assertEqual(self.run_quiet([str(path)]), 1)

    def test_clean_and_secret_shaped_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "input.txt"
            path.write_text("ordinary proof text\n", encoding="utf-8")
            self.assertEqual(self.run_quiet([str(path)]), 0)
            token = "ghp_" + "Z9" * 20
            path.write_text(token + "\n", encoding="utf-8")
            self.assertEqual(self.run_quiet([str(path)]), 1)

    def test_findings_never_echo_secret_material(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "input.txt"
            token = "ghp_" + "Q7" * 20
            path.write_text(token + "\n", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(scanner.main([str(path)]), 1)
            self.assertNotIn(token, output.getvalue())
            self.assertNotIn(token[:14], output.getvalue())

    def test_common_private_key_headers_and_utf16_are_detected(self):
        headers = (
            "-----BEGIN " + "ENCRYPTED PRIVATE KEY-----",
            "-----BEGIN PGP " + "PRIVATE KEY BLOCK-----",
        )
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "input.bin"
            for header in headers:
                with self.subTest(header=header):
                    path.write_bytes((header + "\n").encode("ascii"))
                    self.assertEqual(self.run_quiet([str(path)]), 1)
            token = "ghp_" + "R8" * 20
            for bom, encoding in (
                    (b"\xff\xfe", "utf-16-le"),
                    (b"\xfe\xff", "utf-16-be")):
                with self.subTest(encoding=encoding):
                    path.write_bytes(bom + (token + "\n").encode(encoding))
                    self.assertEqual(self.run_quiet([str(path)]), 1)


if __name__ == "__main__":
    unittest.main()
