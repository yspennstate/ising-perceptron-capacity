"""Rigorous interior-ball certificate for Huang's Region-I star.

The Region-I ray proof differentiates the moment entropy H and therefore
needs every ray point to lie in the interior of the moment body

    K = { E[(X, tanh X) Lambda] : |Lambda| <= 1 }.

This module gives a constructive certificate.  Two explicit feasible
profile perturbations map the l1 unit diamond to a parallelogram around the
maximizer a*.  A 2x2 Arb computation proves that the parallelogram contains
a Euclidean ball larger than the full star, including its origin padding.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import platform

import flint
from flint import acb, arb

import block3bc_exact as exact
import core
import huang_region1 as region1


HERE = pathlib.Path(__file__).resolve().parent
PRECISION_BITS = 100
SPLIT = arb(3) / 4
CUTOFF = arb(12)
INTEGRATION_TOLERANCE = arb(2) ** -80
CERTIFIED_RADIUS_FLOOR = arb("0.0150391982")
ORIGIN_COORDINATE_RADIUS = "0.0000001"
RADIAL_PADDING = "0.000000000001"
ANGULAR_PADDING = "0.00000001"
MAX_LONG_ANGULAR_CELL_WIDTH = "0.016"
ANGULAR_INTERVAL_ROUNDING_GUARD = "0.0000000001"


def _source_paths():
    return {
        "huang_star_interior.py": __file__,
        "huang_region1.py": region1.__file__,
        "block3bc_exact.py": exact.__file__,
        "core.py": core.__file__,
    }


def source_hashes():
    return exact.source_hashes(_source_paths())


def runtime_record():
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "python_flint": flint.__version__,
        "flint": flint.__FLINT_VERSION__,
        "precision_bits": PRECISION_BITS,
    }


def _density(x, psi):
    return (-x * x / (2 * psi)).exp() / (2 * arb.pi() * psi).sqrt()


def _body_integrals():
    """Return row-wise integrals below and above |x|=3/4.

    On x>0 the two perturbations use rho0=1-tanh(x) and rho1=rho0
    below the split, -rho0 above it.  Symmetry supplies the factor two.
    """
    psi = core.PSI

    def row_x(x, analytic):
        m = x.tanh()
        return 2 * x * (1 - m) * _density(x, psi)

    def row_m(x, analytic):
        m = x.tanh()
        return 2 * m * (1 - m) * _density(x, psi)

    out = []
    for integrand in (row_x, row_m):
        inner = core.integrate(
            integrand, arb(0), SPLIT, abs_tol=INTEGRATION_TOLERANCE)
        outer = core.integrate(
            integrand, SPLIT, CUTOFF,
            abs_tol=INTEGRATION_TOLERANCE)
        out.append((inner, outer))
    return out


def _tail_bounds():
    """Symmetric absolute error bounds for the two omitted x-tails."""
    psi_hi = core.PSI_UB
    z = CUTOFF / psi_hi.sqrt()
    row_x = 2 * psi_hi.sqrt() * core.phi(z)
    row_m = 2 * core.Psi(z)
    return row_x, row_m


def compute_matrix():
    core.set_prec(PRECISION_BITS)
    rows = _body_integrals()
    tails = _tail_bounds()
    matrix = []
    for (inner, outer), tail in zip(rows, tails):
        error = arb(0, tail)
        column_zero = inner + outer + error
        column_one = inner - outer + error
        matrix.append([column_zero, column_one])
    return matrix, tails


def _required_radius():
    # Region-I evaluates an interval box, not only geometric rays.  Relative
    # to the true center, its stored-center +/-1e-7 enclosure can be displaced
    # by 2e-7 per coordinate.  Independent cosine/sine interval hulls over the
    # widest long-ray leaf have norm at most sqrt(1 + sin(theta_width)).
    expected_origin = arb(ORIGIN_COORDINATE_RADIUS)
    expected_radial = arb(RADIAL_PADDING)
    expected_angular = arb(ANGULAR_PADDING)
    if (not abs(region1._E7 - expected_origin) < arb("1e-30")
            or not abs(region1._RAD_PAD - expected_radial) < arb("1e-30")
            or not abs(region1._ANG_PAD - expected_angular) < arb("1e-30")):
        raise ValueError("Region-I padding constants drifted from certificate policy")
    theta_width = (arb(MAX_LONG_ANGULAR_CELL_WIDTH)
                   + 2 * expected_angular
                   + arb(ANGULAR_INTERVAL_ROUNDING_GUARD))
    direction_box_norm = (1 + theta_width.sin()).sqrt()
    star_radius = exact.fraction_arb(
        region1._float_fraction(region1.T_LONG))
    return ((star_radius + expected_radial) * direction_box_norm
            + 2 * arb(2).sqrt() * expected_origin)


def compute_bounds(matrix):
    a00, a01 = matrix[0]
    a10, a11 = matrix[1]
    determinant = a00 * a11 - a01 * a10
    edge_minus = ((a00 - a01) ** 2 + (a10 - a11) ** 2).sqrt()
    edge_plus = ((a00 + a01) ** 2 + (a10 + a11) ** 2).sqrt()
    if edge_minus > edge_plus:
        longest_edge = edge_minus
    elif edge_plus > edge_minus:
        longest_edge = edge_plus
    else:
        longest_edge = edge_minus.union(edge_plus)
    inradius = determinant / longest_edge
    required = _required_radius()
    clearance = inradius - required
    if not determinant > 0:
        raise ValueError("interior perturbation matrix is not orientation preserving")
    if not inradius > CERTIFIED_RADIUS_FLOOR:
        raise ValueError("constructive moment-body inradius is too small")
    if not inradius > required:
        raise ValueError("Region-I star is not certified inside the moment body")
    return {
        "determinant": determinant,
        "edge_norm_minus": edge_minus,
        "edge_norm_plus": edge_plus,
        "longest_edge": longest_edge,
        "inradius": inradius,
        "required_radius": required,
        "clearance": clearance,
    }


def compute_certificate():
    matrix, tails = compute_matrix()
    bounds = compute_bounds(matrix)
    payload = {
        "schema_version": 1,
        "kind": "huang_star_interior_certificate",
        "policy": {
            "precision_bits": PRECISION_BITS,
            "split": "3/4",
            "cutoff": "12",
            "integration_tolerance": "2^-80",
            # Region I serializes float geometry as its exact binary ratio.
            # Preserve that same value here; str(T_LONG) would silently
            # round 0.01200000000000000025 down to the decimal 0.012.
            "star_radius": exact.fraction_text(
                region1._float_fraction(region1.T_LONG)),
            "origin_coordinate_radius": "1/10000000",
            "radial_padding": "1/1000000000000",
            "angular_padding": "1/100000000",
            "max_long_angular_cell_width": "2/125",
            "angular_interval_rounding_guard": "1/10000000000",
            "center_interval_diameter_factor": 2,
            "direction_box_norm_bound": (
                "sqrt(1+sin(max_long_angular_cell_width+"
                "2*angular_padding+angular_interval_rounding_guard))"),
            "certified_radius_floor": "75195991/5000000000",
        },
        "runtime": runtime_record(),
        "source_sha256": source_hashes(),
        "psi": exact.arb_packet(core.PSI),
        "matrix": [
            [exact.arb_packet(value) for value in row] for row in matrix],
        "tail_bounds": {
            "row_x": exact.arb_packet(tails[0]),
            "row_tanh": exact.arb_packet(tails[1]),
        },
        "bounds": {
            name: exact.arb_packet(value) for name, value in bounds.items()},
    }
    payload["certificate_sha256"] = exact.payload_sha256(
        payload, omit=("certificate_sha256",))
    return payload


def verify_certificate(path):
    observed = exact.load_json(path)
    if (not isinstance(observed, dict)
            or observed.get("schema_version") != 1
            or observed.get("kind") != "huang_star_interior_certificate"
            or observed.get("certificate_sha256") != exact.payload_sha256(
                observed, omit=("certificate_sha256",))):
        raise ValueError("invalid star-interior certificate envelope")
    expected = compute_certificate()
    if observed != expected:
        raise ValueError("star-interior certificate does not replay exactly")
    return observed


def write_certificate(path):
    payload = compute_certificate()
    exact.write_json_atomic(path, payload, overwrite=False)
    verify_certificate(path)
    return payload


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--certificate", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "generate":
        payload = write_certificate(os.path.abspath(args.output))
        print("PASS Huang star interior: "
              f"{payload['certificate_sha256']} {os.path.abspath(args.output)}")
        return 0
    payload = verify_certificate(os.path.abspath(args.certificate))
    print(f"PASS Huang star interior: {payload['certificate_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
