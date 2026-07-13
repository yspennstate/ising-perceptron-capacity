import pathlib
import unittest


FINISH = pathlib.Path(__file__).resolve().parents[2] / "notes" / "finish.ps1"
VERIFICATION = pathlib.Path(__file__).resolve().parents[1]


class ReleaseScriptTests(unittest.TestCase):
    def test_packager_has_no_publish_or_background_commands(self):
        text = FINISH.read_text(encoding="utf-8").lower()
        forbidden = (
            "git push", "git tag", "git add", "git commit",
            "start-process", "schtasks", "register-scheduledtask")
        for command in forbidden:
            with self.subTest(command=command):
                self.assertNotIn(command, text)
        self.assertIn("sourcecommit", text)
        self.assertIn("verificationreceipt", text)
        self.assertIn("verifiedreceiptpath", text)
        self.assertIn("release_verification.py", text)
        self.assertIn("receiptarguments", text)
        self.assertIn("refusing to overwrite", text)
        self.assertIn(
            "git -c core.autocrlf=false -c core.eol=lf archive", text)
        self.assertNotIn("& git archive", text)
        self.assertIn("release_secret_scan.py", text)
        self.assertIn("releasescanner @batch", text)
        self.assertIn("archivefiles[$offset..$last]", text)
        self.assertNotIn("scanextensions", text)
        self.assertIn("pdfinfo", text)
        self.assertIn("pdftotext", text)
        self.assertIn("pdftoppm", text)
        self.assertIn("immutable archive", text)
        self.assertIn("unexpected tracked verification result", text)
        self.assertNotIn("import sys,flint", text)
        self.assertNotIn("unittest discover", text)
        self.assertNotIn("compress-archive", text)
        self.assertIn("[io.filemode]::createnew", text)
        self.assertNotIn("copy-item -literalpath $validationarchive", text)
        self.assertIn("release output paths must be outside", text)
        self.assertIn("verification receipt must be external", text)
        self.assertIn("verification receipt file sha256", text)
        self.assertIn("verification receipt payload sha256", text)
        self.assertIn("ising_perceptron_external_release_handoff", text)
        self.assertIn("schema_version = 2", text)
        self.assertIn("release handoff already exists", text)
        self.assertIn("verified receipt already exists", text)
        self.assertIn("exact_validated_archive", text)
        self.assertIn("pdf sha256", text)
        self.assertIn("pdf pages", text)
        self.assertIn("release handoff", text)
        self.assertIn("handoff sha256", text)
        self.assertIn("move($packagetemporary, $packagefull)", text)
        self.assertIn("move($receipttemporary, $verifiedreceiptfull)", text)
        self.assertIn("move($handofftemporary, $handofffull)", text)
        self.assertIn("$packagetemporarycreated", text)
        self.assertIn("$receipttemporarycreated", text)
        self.assertIn("$handofftemporarycreated", text)
        self.assertIn("$receiptsnapshotstream.copyto($receiptdestinationstream)", text)
        self.assertNotIn("local_path = $receiptfull", text)
        self.assertIn("source_path_at_snapshot = $receiptfull", text)
        self.assertIn("retained_verified_path = $verifiedreceiptfull", text)
        self.assertIn("exact_checked_snapshot", text)
        source_open = text.index("$receiptsourcestream = [io.file]::open(")
        source_copy = text.index(
            "$receiptsourcestream.copyto($receiptsnapshotwriter)")
        self.assertIn("[io.fileshare]::read", text[source_open:source_copy])
        for nested_key in (
                "expectedreceiptkeys", "expectedpackagekeys",
                "expectedpdfkeys", "expectedpromotionkeys"):
            self.assertIn(nested_key, text)
        self.assertNotIn("$packagefull, [io.filemode]::createnew", text)
        self.assertNotIn("$handofffull, [io.filemode]::createnew", text)
        package_move = text.index("move($packagetemporary, $packagefull)")
        receipt_move = text.index(
            "move($receipttemporary, $verifiedreceiptfull)")
        handoff_move = text.index("move($handofftemporary, $handofffull)")
        self.assertLess(package_move, receipt_move)
        self.assertLess(receipt_move, handoff_move)
        for published, temporary, move in (
                ("$packagepublished = $true",
                 "$packagetemporarycreated = $false", package_move),
                ("$receiptpublished = $true",
                 "$receipttemporarycreated = $false", receipt_move),
                ("$handoffpublished = $true",
                 "$handofftemporarycreated = $false", handoff_move)):
            published_at = text.index(published, move)
            temporary_at = text.index(temporary, published_at)
            self.assertLess(published_at, temporary_at)
        handoff_cleanup = text.index("if ($handoffpublished)")
        receipt_cleanup = text.index("if ($receiptpublished)")
        package_cleanup = text.index("if ($packagepublished)")
        self.assertLess(handoff_cleanup, receipt_cleanup)
        self.assertLess(receipt_cleanup, package_cleanup)

    def test_box_runners_have_no_embedded_patch_continuation_markers(self):
        for name in ("run_block3bc_final.sh", "run_huang_final.sh"):
            text = (VERIFICATION / name).read_text(encoding="utf-8")
            with self.subTest(name=name):
                self.assertNotIn("+    --", text)
                self.assertNotIn("sha256sum +", text)
                self.assertIn("pressure_reasons", text)


if __name__ == "__main__":
    unittest.main()
