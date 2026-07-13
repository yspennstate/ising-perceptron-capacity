"""High-precision independent evaluation of the thinnest Region-II cells.

For the stage-2 negative leaves with the smallest certified margins (the
value-packet upper endpoints closest to zero), re-evaluate the certified
objective

    F(a) = Phi(lambda) - lambda . a + s^2 psi/2 + alpha T(a, s),
    Phi(lambda) = E log(2 cosh(lambda1 X + lambda2 tanh X)),  X ~ N(0, psi),
    T(a, s)     = E log Psi(V(z)),  z ~ N(0, 1),
    V(z) = c0 sqrt(q) z + c1 E_mills(-gamma z)/sqrt(1-q),
    c0 = -(a2/q)/D,  c1 = s - (a1/psi)/D,  D = sqrt(1 - a2^2/q),
    gamma = sqrt(q/(1-q)),

with mpmath tanh-sinh quadrature at 40 significant digits, using the leaf's
own stored rational dual (lambda1, lambda2) and tilt s, at the cell center
and all four corners.  Every evaluation must be strictly negative, and the
maximum over the sampled points must not exceed the certified enclosure's
upper endpoint (the certificate bounds the supremum over the whole cell, so
pointwise values can be smaller but never larger).

Parameters (psi, q, alpha) are taken at the midpoint of the certified
parameter rectangle; the leaf margins (~1e-4) dwarf the rectangle widths
(~1e-9), and this is a consistency check against an engine sharing nothing
with python-flint, not a replacement for the Arb certificates.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction

import mpmath as mp

mp.mp.dps = 40

PSI = (mp.mpf("2.5763513100") + mp.mpf("2.5763513224")) / 2
Q = (mp.mpf("0.56394907949") + mp.mpf("0.56394908030")) / 2
ALPHA = (mp.mpf("0.833078599") + mp.mpf("0.833078600")) / 2
GAMMA = mp.sqrt(Q / (1 - Q))
S1Q = mp.sqrt(1 - Q)
SQ_Q = mp.sqrt(Q)


def frac(record):
    return Fraction(int(record["num"]), int(record["den"]))


def packet_hi(packet):
    mid, rad, e = int(packet["mid10"]), int(packet["rad10"]), packet["exp10"]
    return (mid + rad) * (mp.mpf(10) ** e)


def gauss_phi(z):
    return mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi)


def mills(x):
    """phi(x)/Psi(x), numerically stable via erfcx for large x."""
    return mp.sqrt(2 / mp.pi) / mp.erfc(x / mp.sqrt(2)) * mp.exp(-x * x / 2)


def log_Psi(v):
    return mp.log(mp.erfc(v / mp.sqrt(2)) / 2)


def Phi(l1, l2):
    def f(z):
        x = mp.sqrt(PSI) * z
        u = l1 * x + l2 * mp.tanh(x)
        # log 2cosh(u) = |u| + log1p(exp(-2|u|))
        return (abs(u) + mp.log1p(mp.exp(-2 * abs(u)))) * gauss_phi(z)
    return mp.quad(f, [-9, -3, 0, 3, 9])


def T_of(a1, a2, s):
    D = mp.sqrt(1 - a2 * a2 / Q)
    c0 = -(a2 / Q) / D
    c1 = s - (a1 / PSI) / D

    def f(z):
        v = c0 * SQ_Q * z + c1 * mills(-GAMMA * z) / S1Q
        return log_Psi(v) * gauss_phi(z)
    return mp.quad(f, [-9, -3, 0, 3, 9])


def objective(l1, l2, s, a1, a2):
    return (Phi(l1, l2) - l1 * a1 - l2 * a2 + s * s * PSI / 2
            + ALPHA * T_of(a1, a2, s))


def collect_negative_leaves(node, out):
    if node["kind"] == "split":
        for child in node["children"]:
            collect_negative_leaves(child, out)
    elif node["kind"] == "negative":
        out.append(node)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/huang_sweep2.json"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    data = json.load(open(path))
    leaves = []
    for rec in data["records"]:
        collect_negative_leaves(rec["tree"], leaves)
    direct = [l for l in leaves if l["method"] == "direct"]
    direct.sort(key=lambda l: packet_hi(l["value_packet"]), reverse=True)
    print(f"{len(leaves)} negative leaves ({len(direct)} direct); "
          f"checking the {count} thinnest direct margins at {mp.mp.dps} dps")

    failures = 0
    for leaf in direct[:count]:
        w = leaf["majorant_witness"]
        # exact rationals through mpmath for full precision
        l1 = mp.mpf(frac(w["lambda1"]).numerator) / frac(w["lambda1"]).denominator
        l2 = mp.mpf(frac(w["lambda2"]).numerator) / frac(w["lambda2"]).denominator
        s = mp.mpf(frac(w["tilt_s"]).numerator) / frac(w["tilt_s"]).denominator
        cell = [frac(v) for v in leaf["cell"]]
        cert_hi = packet_hi(leaf["value_packet"])
        pts = [((cell[0] + cell[1]) / 2, (cell[2] + cell[3]) / 2),
               (cell[0], cell[2]), (cell[0], cell[3]),
               (cell[1], cell[2]), (cell[1], cell[3])]
        vals = []
        for a1f, a2f in pts:
            a1 = mp.mpf(a1f.numerator) / a1f.denominator
            a2 = mp.mpf(a2f.numerator) / a2f.denominator
            vals.append(objective(l1, l2, s, a1, a2))
        vmax = max(vals)
        neg = vmax < 0
        below = vmax <= cert_hi + mp.mpf("1e-12")
        verdict = "PASS" if (neg and below) else "FAIL"
        if verdict == "FAIL":
            failures += 1
        print(f"{verdict} cell [{float(cell[0]):.4f},{float(cell[1]):.4f}]x"
              f"[{float(cell[2]):.4f},{float(cell[3]):.4f}]: "
              f"max sampled {mp.nstr(vmax, 8)} vs certified sup <= "
              f"{mp.nstr(cert_hi, 8)}")

    if failures:
        print(f"{failures} thin cells disagree")
        return 1
    print("all sampled thin-cell values negative and within certified sups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
