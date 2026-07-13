"""Portable mathematical check of the Region-I star-interior certificate.

The committed verifier (huang_star_interior.verify_certificate) regenerates
the certificate and demands dict equality, which binds it to the attested
runtime (host FLINT build and Python).  This checker is the portable
complement: it re-derives the certificate's mathematical claim from the
STORED packets alone, using only integer arithmetic (fractions + isqrt).
No python-flint, no floating point.

Claim chain re-derived from the stored 2x2 matrix enclosures [lo, hi]:

    determinant = a00*a11 - a01*a10                      (interval product)
    edge_minus  = sqrt((a00-a01)^2 + (a10-a11)^2)
    edge_plus   = sqrt((a00+a01)^2 + (a10+a11)^2)
    inradius    = determinant / max(edge_minus, edge_plus)

    required = (star_radius + radial_padding) * sqrt(1 + sin(theta_width))
               + 2*sqrt(2)*origin_coordinate_radius
    theta_width = max_long_angular_cell_width + 2*angular_padding
                  + angular_interval_rounding_guard

and the certificate is accepted only if, with outward rational rounding,

    determinant_lo > 0,
    inradius_lo    > certified_radius_floor,
    inradius_lo    > required_hi.

sin(t) for the tiny rational t is enclosed by [t - t^3/6, t]; square roots
are enclosed with integer isqrt at 60 guard digits.  Every bound is exact
rational arithmetic, so this check is machine-independent.

It also asserts the stored derived-bound packets (determinant, edges,
inradius, required_radius, clearance) each overlap the independently derived
interval, catching a certificate whose summary packets disagree with its own
matrix.

Exit code 0 and a PASS line on success; any failure raises.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction

GUARD = 10 ** 60          # denominator guard for integer square roots
FLOOR = Fraction(150391982, 10 ** 10)   # certified_radius_floor 0.0150391982

POLICY_EXPECTED = {
    "origin_coordinate_radius": Fraction(1, 10 ** 7),
    "radial_padding": Fraction(1, 10 ** 12),
    "angular_padding": Fraction(1, 10 ** 8),
    "max_long_angular_cell_width": Fraction(2, 125),
    "angular_interval_rounding_guard": Fraction(1, 10 ** 10),
    "certified_radius_floor": FLOOR,
}


def canonical_int(text):
    if isinstance(text, int) and not isinstance(text, bool):
        return text
    if not isinstance(text, str):
        raise ValueError(f"integer literal must be str: {text!r}")
    value = int(text)
    if str(value) != text:
        raise ValueError(f"non-canonical integer literal: {text!r}")
    return value


def packet_endpoints(packet):
    required = {"format", "mid10", "rad10", "exp10", "digits"}
    if not isinstance(packet, dict) or set(packet) != required:
        raise ValueError("invalid Arb packet schema")
    if packet["format"] != "arb-midrad10-v1":
        raise ValueError("unknown Arb packet format")
    mid = canonical_int(packet["mid10"])
    rad = canonical_int(packet["rad10"])
    if rad < 0:
        raise ValueError("negative packet radius")
    exp10 = packet["exp10"]
    if not isinstance(exp10, int) or isinstance(exp10, bool):
        raise ValueError("packet exponent must be a plain integer")
    scale = (Fraction(10 ** exp10) if exp10 >= 0
             else Fraction(1, 10 ** -exp10))
    return (mid - rad) * scale, (mid + rad) * scale


def parse_fraction_text(text):
    if not isinstance(text, str):
        raise ValueError("fraction text must be a string")
    if "/" in text:
        num, den = text.split("/", 1)
        value = Fraction(canonical_int(num), canonical_int(den))
        if (str(value.numerator), str(value.denominator)) != (num, den):
            raise ValueError(f"fraction text not canonical: {text!r}")
        return value
    return Fraction(canonical_int(text))


# ----- exact interval arithmetic on pairs of Fractions --------------------

def iadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def isub(a, b):
    return (a[0] - b[1], a[1] - b[0])


def imul(a, b):
    products = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    return (min(products), max(products))


def isquare(a):
    lo, hi = a
    if lo <= 0 <= hi:
        return (Fraction(0), max(lo * lo, hi * hi))
    squares = (lo * lo, hi * hi)
    return (min(squares), max(squares))


def sqrt_lb(x: Fraction) -> Fraction:
    if x < 0:
        raise ValueError("sqrt of negative lower endpoint")
    n = x.numerator * x.denominator * GUARD * GUARD
    return Fraction(math.isqrt(n), x.denominator * GUARD)


def sqrt_ub(x: Fraction) -> Fraction:
    if x < 0:
        raise ValueError("sqrt of negative value")
    n = x.numerator * x.denominator * GUARD * GUARD
    root = math.isqrt(n)
    if root * root < n:
        root += 1
    return Fraction(root, x.denominator * GUARD)


def isqrt_iv(a):
    return (sqrt_lb(a[0]), sqrt_ub(a[1]))


def sin_enclosure(t: Fraction):
    """[t - t^3/6, t] encloses sin t for 0 <= t <= 1 (alternating series)."""
    if not 0 <= t <= 1:
        raise ValueError("sin enclosure valid only on [0, 1]")
    return (t - t ** 3 / 6, t)


def overlaps(mine, stored):
    return not (mine[1] < stored[0] or stored[1] < mine[0])


def check(path):
    raw = open(path, "rb").read()
    data = json.loads(raw.decode("utf-8"))
    if (data.get("kind") != "huang_star_interior_certificate"
            or data.get("schema_version") != 1):
        raise ValueError("wrong certificate kind/schema")

    policy = data["policy"]
    for key, expected in POLICY_EXPECTED.items():
        got = parse_fraction_text(policy[key])
        if got != expected:
            raise ValueError(f"policy {key} = {got} differs from {expected}")
    star_radius = parse_fraction_text(policy["star_radius"])
    if not Fraction(11, 1000) < star_radius < Fraction(13, 1000):
        raise ValueError("star radius outside its documented magnitude")

    matrix = data["matrix"]
    if (not isinstance(matrix, list) or len(matrix) != 2
            or any(len(row) != 2 for row in matrix)):
        raise ValueError("matrix must be 2x2")
    a = [[packet_endpoints(entry) for entry in row] for row in matrix]

    determinant = isub(imul(a[0][0], a[1][1]), imul(a[0][1], a[1][0]))
    edge_minus = isqrt_iv(iadd(isquare(isub(a[0][0], a[0][1])),
                               isquare(isub(a[1][0], a[1][1]))))
    edge_plus = isqrt_iv(iadd(isquare(iadd(a[0][0], a[0][1])),
                              isquare(iadd(a[1][0], a[1][1]))))
    longest_ub = max(edge_minus[1], edge_plus[1])
    longest_lb = max(edge_minus[0], edge_plus[0])
    if longest_lb <= 0:
        raise ValueError("cannot certify a positive longest edge")
    inradius = (determinant[0] / longest_ub, determinant[1] / longest_lb)

    theta_width = (POLICY_EXPECTED["max_long_angular_cell_width"]
                   + 2 * POLICY_EXPECTED["angular_padding"]
                   + POLICY_EXPECTED["angular_interval_rounding_guard"])
    sin_iv = sin_enclosure(theta_width)
    norm_iv = isqrt_iv(iadd((Fraction(1), Fraction(1)), sin_iv))
    sqrt2_iv = isqrt_iv((Fraction(2), Fraction(2)))
    base = star_radius + POLICY_EXPECTED["radial_padding"]
    required = iadd(imul((base, base), norm_iv),
                    imul(imul((Fraction(2), Fraction(2)), sqrt2_iv),
                         (POLICY_EXPECTED["origin_coordinate_radius"],) * 2))

    if not determinant[0] > 0:
        raise ValueError("determinant lower bound is not positive")
    if not inradius[0] > FLOOR:
        raise ValueError("inradius lower bound does not clear the floor")
    if not inradius[0] > required[1]:
        raise ValueError("inradius lower bound does not clear required radius")

    stored = {name: packet_endpoints(pkt)
              for name, pkt in data["bounds"].items()}
    derived = {
        "determinant": determinant,
        "edge_norm_minus": edge_minus,
        "edge_norm_plus": edge_plus,
        "longest_edge": (longest_lb, longest_ub),
        "inradius": inradius,
        "required_radius": required,
        "clearance": isub(inradius, required),
    }
    for name, mine in derived.items():
        if name not in stored:
            raise ValueError(f"stored bounds missing {name}")
        if not overlaps(mine, stored[name]):
            raise ValueError(f"stored bound {name} disagrees with matrix-derived interval")

    digest = hashlib.sha256(raw).hexdigest()
    print(f"PASS star-interior portable check: file sha256 {digest}")
    print(f"  determinant  >= {float(determinant[0]):.6e}")
    print(f"  inradius     >= {float(inradius[0]):.10f}")
    print(f"  required     <= {float(required[1]):.10f}")
    print(f"  floor           {float(FLOOR):.10f}")
    print(f"  clearance    >= {float(inradius[0] - required[1]):.3e}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate")
    args = parser.parse_args(argv)
    return check(args.certificate)


if __name__ == "__main__":
    raise SystemExit(main())
