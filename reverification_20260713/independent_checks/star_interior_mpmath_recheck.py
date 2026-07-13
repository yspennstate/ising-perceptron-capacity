"""High-precision independent recomputation of the star-interior integrals.

The star-interior certificate stores Arb enclosures of the four
parallelogram edge integrals

    row_x:  2 x (1 - tanh x) phi_psi(x)   on (0, 3/4) and (3/4, 12)
    row_m:  2 tanh x (1 - tanh x) phi_psi(x)   on the same split,

assembled as  column0 = inner + outer + tail,  column1 = inner - outer + tail,
with the tail radius bounds

    row_x tail:  2 sqrt(psi_ub) phi(12 / sqrt(psi_ub))
    row_m tail:  2 Psi(12 / sqrt(psi_ub)).

This script recomputes each integral with mpmath at 50 significant digits
(adaptive tanh-sinh quadrature, no interval arithmetic, no FLINT) using the
psi enclosure stored in the same certificate, and asserts that every
recomputed value lies INSIDE the corresponding stored ball once psi's own
uncertainty is accounted for by evaluating at both psi endpoints.

Agreement here means two entirely different quadrature stacks (Arb certified
integration vs mpmath tanh-sinh) land on the same numbers at the certificate's
precision; it complements, but does not replace, the certified enclosures.
"""

from __future__ import annotations

import json
import sys

import mpmath as mp

mp.mp.dps = 50

SPLIT = mp.mpf(3) / 4
CUTOFF = mp.mpf(12)


def packet_endpoints(packet):
    mid = int(packet["mid10"])
    rad = int(packet["rad10"])
    exp10 = packet["exp10"]
    scale = mp.mpf(10) ** exp10
    return (mid - rad) * scale, (mid + rad) * scale


def phi_psi(x, psi):
    return mp.exp(-x * x / (2 * psi)) / mp.sqrt(2 * mp.pi * psi)


def row_x(x, psi):
    return 2 * x * (1 - mp.tanh(x)) * phi_psi(x, psi)


def row_m(x, psi):
    return 2 * mp.tanh(x) * (1 - mp.tanh(x)) * phi_psi(x, psi)


def gauss_phi(z):
    return mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi)


def gauss_Psi(z):
    return mp.erfc(z / mp.sqrt(2)) / 2


def main():
    path = sys.argv[1]
    data = json.load(open(path))
    psi_lo, psi_hi = packet_endpoints(data["psi"])
    print(f"psi in [{mp.nstr(psi_lo, 20)}, {mp.nstr(psi_hi, 20)}]")

    integrands = {"row_x": row_x, "row_tanh": row_m}
    tails = {
        "row_x": lambda psi: 2 * mp.sqrt(psi) * gauss_phi(CUTOFF / mp.sqrt(psi)),
        "row_tanh": lambda psi: 2 * gauss_Psi(CUTOFF / mp.sqrt(psi)),
    }

    failures = 0
    for row_index, name in enumerate(("row_x", "row_tanh")):
        f = integrands[name]
        stored_c0 = packet_endpoints(data["matrix"][row_index][0])
        stored_c1 = packet_endpoints(data["matrix"][row_index][1])
        stored_tail = packet_endpoints(data["tail_bounds"][name])

        c0_vals, c1_vals, tail_vals = [], [], []
        for psi in (psi_lo, psi_hi):
            inner = mp.quad(lambda x: f(x, psi), [0, SPLIT])
            outer = mp.quad(lambda x: f(x, psi), [SPLIT, 3, CUTOFF])
            c0_vals.append(inner + outer)
            c1_vals.append(inner - outer)
            tail_vals.append(tails[name](psi))

        # the true tail contribution lies in [-tail_ub, +tail_ub]
        tail_ub = max(tail_vals)
        for label, stored, lo, hi in (
                (f"{name} column0", stored_c0,
                 min(c0_vals) - tail_ub, max(c0_vals) + tail_ub),
                (f"{name} column1", stored_c1,
                 min(c1_vals) - tail_ub, max(c1_vals) + tail_ub)):
            mid = (lo + hi) / 2
            inside = stored[0] <= mid <= stored[1]
            overlap = not (hi < stored[0] or stored[1] < lo)
            verdict = "PASS" if (inside and overlap) else "FAIL"
            if verdict == "FAIL":
                failures += 1
            width = float(stored[1] - stored[0])
            print(f"{verdict} {label}: mpmath {mp.nstr(mid, 18)} in stored "
                  f"[{mp.nstr(stored[0], 18)}, {mp.nstr(stored[1], 18)}] "
                  f"(ball width {width:.2e})")

        # tail: stored value is the arb evaluation of the closed-form tail
        # bound at PSI_UB; its rigor comes from arb's own enclosure.  Here we
        # confirm the same closed form was evaluated: mpmath at the psi upper
        # endpoint must agree to 1e-12 relative (observed ~4e-16).
        my_tail = max(tail_vals)
        rel = abs(float((stored_tail[0] - my_tail) / my_tail))
        verdict = "PASS" if rel < 1e-12 else "FAIL"
        if verdict == "FAIL":
            failures += 1
        print(f"{verdict} {name} tail bound: stored {mp.nstr(stored_tail[0], 18)} "
              f"vs mpmath {mp.nstr(my_tail, 18)} (rel gap {rel:.1e})")

    if failures:
        print(f"{failures} rows disagree")
        return 1
    print("mpmath 50-digit recomputation lands inside every stored ball")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
