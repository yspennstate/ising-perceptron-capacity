#!/usr/bin/env python3
"""Fail-closed whole-file secret scan for the reproducibility package."""

from __future__ import annotations

import pathlib
import re
import sys


PATTERNS = (
    ("private key",
     r"-----BEGIN (?:[A-Z0-9][A-Z0-9 -]* )?PRIVATE KEY(?: BLOCK)?-----"),
    ("GitHub token", r"gh[pousr]_[A-Za-z0-9]{30,}|"
                     r"github_pat_[A-Za-z0-9_]{22,}"),
    ("OpenAI/Anthropic key", r"sk-(?:ant-)?[A-Za-z0-9_\-]{20,}"),
    ("AWS access key id", r"AKIA[0-9A-Z]{16}"),
    ("Telegram bot token", r"\b\d{8,10}:[A-Za-z0-9_-]{30,60}\b"),
    ("Slack token", r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    ("Google API key", r"AIza[0-9A-Za-z_\-]{35}"),
    ("RunPod/HF/Groq key", r"\b(?:rpa_|rpk_|hf_|gsk_)"
                             r"[A-Za-z0-9_\-]{16,}\b"),
    ("long API key", r"(?<![/\w.])(?=[A-Za-z0-9]*[a-z])"
                     r"(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*\d)"
                     r"[A-Za-z0-9]{40,}(?![\w.])"),
    ("plain password", r"(?i)\b(?:password|passwd|pwd)\b"
                       r"(?:\s+(?:is|was))?\s*[:=]?\s+"
                       r"(?=\S*\d)[A-Za-z0-9!@#$%^&*_\-]{6,24}\b"),
    ("2FA code", r"(?i)\bcode\b(?:\s+(?:is|was))?\s*[:=]?\s*"
                 r"\d{6,8}\b"),
    ("SSN", r"(?i)(?:\bssn\b|social security(?:\s+number)?)"
            r"(?:\s+(?:is|was|#))?\s*[:=]?\s*"
            r"\d{3}-?\d{2}-?\d{4}\b"),
    ("secret assignment",
     r"(?i)(?:api[_-]?key|secret|token|passwd|password|access[_-]?key|"
     r"credential|client[_-]?secret)['\"]?\s*[:=]\s*['\"]"
     r"[A-Za-z0-9_\-/+]{16,}['\"]"),
)

def scan_line(line: str):
    for name, pattern in PATTERNS:
        for _match in re.finditer(pattern, line):
            # Never echo even a prefix of a possible credential into a log.
            yield name


def text_views(raw: bytes):
    """Yield byte-preserving text plus BOM-marked Windows Unicode text."""
    yield "bytes", raw.decode("latin-1")
    if raw.startswith(b"\xff\xfe"):
        yield "utf-16-le", raw[2:].decode("utf-16-le", errors="replace")
    elif raw.startswith(b"\xfe\xff"):
        yield "utf-16-be", raw[2:].decode("utf-16-be", errors="replace")


def main(argv=None) -> int:
    paths = list(sys.argv[1:] if argv is None else argv)
    if not paths:
        print("SECRET SCAN ERROR: no files supplied", file=sys.stderr)
        return 2
    findings = []
    for name in paths:
        path = pathlib.Path(name)
        try:
            if not path.is_file() or path.is_symlink():
                raise OSError("input is not a regular non-symlink file")
            # Scan each supplied file's complete contents. Latin-1 is
            # reversible, so binary/non-UTF-8 files are not skipped;
            # BOM-marked UTF-16 is additionally decoded to catch ordinary
            # PowerShell redirection.
            raw = path.read_bytes()
            for view, text in text_views(raw):
                for line_number, line in enumerate(text.splitlines(), 1):
                    for kind in scan_line(line):
                        findings.append(
                            (kind, f"{path}:{line_number} [{view}]"))
        except OSError as exc:
            print(f"SECRET SCAN ERROR: cannot read {path}: {exc}",
                  file=sys.stderr)
            return 2
    if findings:
        print(f"SECRET SCAN: {len(findings)} possible secret(s); do not package")
        for kind, where in findings:
            print(f"  {kind:<22} ({where})")
        return 1
    print(f"secret scan clean: {len(paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
