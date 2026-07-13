"""Falsifiability check for the Lean skeleton layer.

Generates deliberately corrupted variants of the extracted claims and
confirms the Lean kernel REJECTS each one.  A verification layer that
cannot fail is not a check; this demonstrates the `decide` theorems are
sensitive to the data they bind.

Corruptions:
  1. definiteness: inflate a packet radius past its midpoint (b11_lo <= 0).
  2. definiteness: swell the off-diagonal so b11*b22 - max|b12|^2 <= 0.
  3. cover: drop one leaf from a localization group (gap).
  4. cover: duplicate one leaf (eraseDups length check).
  5. sign leaf: flip a claimed sign against its packet.
  6. tiling: break one adjacency equality.

Each variant is emitted as a small standalone Lean file; the expectation is
nonzero lean exit status for every one.  Exit 0 from this script means all
corruptions were correctly rejected.
"""

from __future__ import annotations

import json
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from extract_skeleton import DEFS, packet, lean_int  # noqa: E402

LEAN = os.path.expanduser(
    '~/.elan/toolchains/leanprover--lean4---v4.31.0/bin/lean.exe')


def run_lean(name, body):
    path = os.path.join(HERE, 'mutants', name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='\n', encoding='utf-8') as f:
        f.write(DEFS)
        f.write(body)
    proc = subprocess.run([LEAN, path], capture_output=True, text=True)
    return proc.returncode


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, '..', 'public', 'verification', 'results')
    r1 = json.load(open(os.path.join(results_dir, 'huang_region1.json')))
    rec = r1['records'][0]
    cert = rec['B_certificate']
    m11, r11, e11 = packet(cert['b11_packet'])
    m12, r12, e12 = packet(cert['b12_packet'])
    m22, r22, e22 = packet(cert['b22_packet'])
    p, q = e11 + e22, 2 * e12
    s = min(p, q)
    sh1, sh2 = p - s, q - s

    def bdef_body(vals):
        row = '(' + ', '.join([lean_int(v) for v in vals[:6]]
                              + [str(vals[6]), str(vals[7])]) + ')'
        return ('\ntheorem mutant : allB [' + row + '] = true := by decide\n')

    checks = []
    checks.append(('bdef_radius_swallows_mid',
                   bdef_body((m11, m11 + 1, m22, r22, m12, r12, sh1, sh2))))
    checks.append(('bdef_offdiag_dominates',
                   bdef_body((m11, r11, m22, r22, m11 * 10, r12, sh1, sh2))))

    lane = json.load(open(os.path.join(
        results_dir, 'block3bc_replay', 'b_neg.lane-0-of-1.json')))
    leaf = lane['records'][0]['leaves'][0]
    lo_n, lo_d = int(leaf['tau_lo']['num']), int(leaf['tau_lo']['den'])
    hi_n, hi_d = int(leaf['tau_hi']['num']), int(leaf['tau_hi']['den'])
    mid, rad, _ = packet(leaf['value'])
    assert leaf['sign'] == '>0'

    def leaf_body(vals, flag):
        row = '(' + ', '.join([lean_int(v) for v in vals] + [flag]) + ')'
        return '\ntheorem mutant : allLeaves [' + row + '] = true := by decide\n'

    checks.append(('leaf_sign_flipped',
                   leaf_body((lo_n, lo_d, hi_n, hi_d, mid, rad), 'false')))
    checks.append(('leaf_empty_interval',
                   leaf_body((hi_n, hi_d, lo_n, lo_d, mid, rad), 'true')))
    checks.append(('cover_gap',
                   '\ntheorem mutant : coverOk ["00", "01"] 14 = true := by decide\n'))
    checks.append(('cover_duplicate',
                   '\ntheorem mutant : coverOk ["0", "0", "1"] 14 = true := by decide\n'))
    checks.append(('cover_overlap',
                   '\ntheorem mutant : coverOk ["0", "1", "10"] 14 = true := by decide\n'))
    checks.append(('tiling_mismatch',
                   '\ntheorem mutant : allEq [(1, 2, 2, 3)] = true := by decide\n'))
    checks.append(('containment_escape',
                   '\ntheorem mutant : allLe [(3, 2, 1, 1)] = true := by decide\n'))
    checks.append(('split_order_violation',
                   '\ntheorem mutant : allLt [(1, 1, 1, 2)] = true := by decide\n'))
    checks.append(('curvature_not_negative',
                   '\ntheorem mutant : allNeg [(5, 2)] = true := by decide\n'))
    checks.append(('curvature_radius_swallows',
                   '\ntheorem mutant : allNeg [((-3), 4)] = true := by decide\n'))
    checks.append(('tilt_not_positive',
                   '\ntheorem mutant : allPos [(3, 5)] = true := by decide\n'))

    failures = 0
    for name, body in checks:
        code = run_lean(name + '.lean', body)
        verdict = 'REJECTED (good)' if code != 0 else 'ACCEPTED (BAD)'
        if code == 0:
            failures += 1
        print(f'{name}: lean exit {code} -> {verdict}')
    if failures:
        print(f'{failures} corruptions were wrongly accepted')
        return 1
    print('all corruptions correctly rejected by the Lean kernel')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
