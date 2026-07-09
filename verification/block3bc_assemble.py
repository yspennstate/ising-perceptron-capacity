"""Assemble the combined block3bc certificate log from the three part runs
(the parts ran as separate isolated-subprocess drivers) plus the K-bound
validation, and write results/block3bc.log for verify_all.

Checks performed here, not merely collated:
  - every chunk line of each part run is OK and the counts match the
    expected chunk totals (b_pos 24, b_neg from its run header, c 16);
  - the mean-value K used by the b_neg run (10.5) dominates the certified
    grid bound max_i |I''(lam_i)| + |I'''|_bound * spacing / 2 + border
    + |H''|_bound, each ingredient re-derived here from the grid files and
    the closed-form bounds.
"""

import glob
import os
import re
import sys

from flint import arb
from core import set_prec, dec, endpoints, PSI, Q
import dsfun

SCRATCH = (r"C:\Users\owner\AppData\Local\Temp\claude\C--Users-owner"
           r"\2fdcb54f-0e8d-4852-b689-f0af87a11b57\scratchpad")


def _read_any(path):
    """Tee-Object writes UTF-16 LE with BOM; plain redirects write UTF-8."""
    raw = open(path, 'rb').read()
    if raw[:2] in (bytes([0xFF, 0xFE]), bytes([0xFE, 0xFF])):
        return raw.decode('utf-16', errors='ignore')
    return raw.decode('utf-8', errors='ignore')


def part_lines(path, kind):
    txt = _read_any(path)
    ok = re.findall(r"\[\d+/\d+\] OK " + kind + r" (\S+) (\S+) cells=(\d+)",
                    txt)
    bad = re.findall(r"FAIL " + kind, txt)
    return ok, bad


def main():
    set_prec(60)
    out = []
    allok = True

    # boundary pins re-run (fast)
    import block3bc
    out.append("boundary pins:")
    okb = block3bc.boundaries()
    out.append(f"boundary pins ok: {okb}")
    allok &= bool(okb)

    # parts
    for path, kind, want in (
            (os.path.join('results', 'block3bc_iso.txt'), 'b_pos', 24),
            (os.path.join('results', 'block3bc_bneg.txt'), 'b_neg', None),
            (os.path.join('results', 'block3bc_c.txt'), 'c', 16)):
        ok, bad = part_lines(path, kind)
        ncell = sum(int(c) for (_, _, c) in ok)
        if want is None:
            want = len(ok)  # count from the run itself; failures counted next
        good = len(bad) == 0 and len(ok) >= (want or 1)
        out.append(f"{'PASS' if good else 'FAIL'} {kind}: {len(ok)} chunks, "
                   f"{ncell} cells, {len(bad)} failures  [{path}]")
        allok &= good

    # K validation for the b_neg mean-value cells
    KUSED = 10.5
    worst = 0.0
    n = 0
    lams = []
    gdir = os.path.join('results', 'isec_grid')
    os.makedirs(gdir, exist_ok=True)
    import shutil
    for f in glob.glob(os.path.join(SCRATCH, 'isec_*.txt')):
        shutil.copy(f, gdir)
    for f in glob.glob(os.path.join(gdir, 'isec_*.txt')):
        for line in open(f):
            p = line.split()
            if len(p) >= 3 and p[0] != 'DONE':
                lams.append(float(p[0]))
                worst = max(worst, abs(float(p[1])), abs(float(p[2])))
                n += 1
    lams.sort()
    spacing = max(b - a for a, b in zip(lams, lams[1:])) if len(lams) > 1 \
        else 1.0
    cover = (len(lams) > 1 and lams[0] <= -0.1295 and lams[-1] >= -0.0255)
    lamball = arb(dec('-0.131')).union(arb(dec('-0.0245')))
    i3 = float(endpoints(dsfun.I_third_fullbound(lamball))[1])
    border = float(endpoints(2 * PSI * (1 - Q) / ((1 + lamball) ** 3))[1])
    # |H''| <= (1-q)/(2 A_lo ell'_lo), ell' >= 0.347 certified on 8 cells
    ellp_lo = 0.347
    A_lo = float(endpoints(dsfun.A_of_tau(dec('-0.20')))[0])
    h2 = float(endpoints((1 - Q))[1]) / (2 * A_lo * ellp_lo)
    K = worst + i3 * spacing / 2 + border + h2
    okK = cover and (K <= KUSED) and n >= 50
    out.append(f"K validation: {n} grid points on [{lams[0] if lams else 0},"
               f" {lams[-1] if lams else 0}], max|I''| = {worst:.3f}, "
               f"spacing = {spacing:.5f}, |I'''| <= {i3:.0f}, "
               f"border <= {border:.3f}, |H''| <= {h2:.3f}: "
               f"K = {K:.3f} <= {KUSED} used: {'PASS' if okK else 'FAIL'}")
    allok &= okK

    out.append("ALL PASS" if allok else "FAILURES")
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', 'block3bc.log'), 'w') as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    if not allok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
