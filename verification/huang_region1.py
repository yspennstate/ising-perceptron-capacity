"""Region I of Huang's Condition 1.3: S_* <= 0 on a star region around the
degenerate maximizer, by banded ray-concavity certificates.

The argument
------------
Let a* = grad Phi(1,0) (at the true parameters a* = (psi(1-q), q)).  For a
unit direction v and fixed slope sdot define the affine-tilt majorant

    phi_v(t) = H(x) + s(t)^2 psi/2 + alpha T(x, s(t)),  x = a* + t v,
    s(t) = s0 + t sdot,  s0 = sqrt(1-q).

phi_v >= S_* along the ray (G = min_s <= any fixed s), and at the true fixed
point phi_v(0) = 0, phi_v'(0) = 0 (Huang's unconditional identities: Gardner
stationarity osS(1,0) = 0, grad osS(1,0) = 0, and the tilt equation
d/ds[s^2 psi/2 + alpha T(a*, s)]|_{s0} = 0).  Hence

    phi_v''(t) <= 0 on [0, T(v)]  ==>  S_* <= 0 on the segment.

Splitting phi'' = (H-part) + (T-part),

    H-part = v^T grad^2 H(x) v = -v^T [grad^2 Phi(lambda(x))]^{-1} v,
    T-part = alpha v^T d2_a T v + 2 sdot (v . alpha d_a d_s T) + sdot^2 Sss,

the T-part is enclosed by certified fixed-grid integrals over the ray piece
(no dual variable).  For the H-part, since grad^2 Phi(lambda) =
E[f f^T sech^2(lambda . f)], any rigorous localization lambda(x) in LBOX gives

    grad^2 Phi(lambda(x))  <=  B_L := E[ f f^T  max_{lambda in LBOX}
                                         sech^2(lambda . f) ],

an explicit one-dimensional integral (the max is a per-z corner selection),
whence  H-part <= -v^T B_L^{-1} v.  With LBOX = R^2 this is the global bound
B_0 = E[f f^T]; with the band localization below it is nearly sharp.

Localization WITHOUT tracking the ill-conditioned dual: lambda(x) lies in the
sublevel set  S(x) = { lambda : Phi(lambda) - lambda.x <= Phi(1,0) - (1,0).x }
(the dual value at lambda(x) is the minimum, and (1,0) is feasible).  S(x) is
convex and contains (1,0); if the rigorous check

    Phi(lambda) - Phi(1,0) - (lambda - (1,0)) . x  >  0

holds everywhere on the boundary of a candidate box LBOX (a 1-D certified
sweep over the four edges; the check is LINEAR in x, so verifying it at the
corner points of the x-region suffices), then S(x) -- hence lambda(x) -- stays
inside LBOX for every x in the region.  The genuine size of S(x) is the
Bregman ball of radius  D = [Phi(1,0) - (1,0).x] - H(x) = O(|x-a*|^2), so the
boxes shrink quadratically-in-radius toward (1,0) and B_L -> grad^2 Phi(1,0):
the H-bound converges to the sharp one exactly where sharpness is needed.

The star region is processed in log radius-bands t in [r t_k, t_k] times the
angular fan; each (band x angle cell) is one certificate:
    quadT(x-sector, s-range)  -  v^T B_L(band)^{-1} v  <  0.
Every constant is a parameter BALL over the certified Ding-Sun rectangle, so
all certificates cover the true fixed point.

Run:  python huang_region1.py [nworkers]
Writes results/huang_region1.json (including the certified star table for the
stage-2 sweep).
"""

import os
import sys
import time
import math
import uuid
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
from multiprocessing import Pool

import flint
from flint import arb, acb
import core
from core import (set_prec, dec, endpoints, PSI, Q, ALPHA,
                  z1_tail, z2_tail, gauss_tail_mass)
import huanggrid as hg
import huang_hessian as hh
import block3bc_exact as exact

# ---------------------------------------------------------------------------
# numeric policy constants (choices; rigor lives in the certificates)
# ---------------------------------------------------------------------------
A1S = 1.1234203619        # ~ a*_1 = psi(1-q)
A2S = 0.5639490799        # ~ a*_2 = q
# Center the stored tilt at the certified value.  The previous decimal was
# 3.64e-7 too high, outside the +/-1e-7 origin ball.
S0F = 0.6603415180        # ~ s0 = sqrt(1-q)
# One exact global tilt-slope rule is used on every radial annulus.  For a
# true ray direction v, s(t)=s0+t*(SDOT_C dot v) is therefore the SAME
# majorant on every band, even when the adaptive angular partitions differ.
# Joint stationarity at (a*,s0) pins value and first derivative at zero for
# any fixed C; the coefficients below are merely a tight numerical choice.
SDOT_MODEL = 'global-linear-v1'
SDOT_C_TEXT = ('-2.448324', '0.790403')
SDOT_C1, SDOT_C2 = (dec(value) for value in SDOT_C_TEXT)
W_ANG = 0.6253            # angle of the weak eigenvector w of grad^2 S_*
N_ANG = 48                # angular cells over the full circle
WEDGE_HALF = 0.16         # |omega| below which rays are long
CONE_MID = 0.45
T_LONG = 0.0120           # star radii by angular zone (sweep MIN_SIDE 1e-3:
T_MID = 0.0080            # failure ellipse ~8.2e-3 along w, ~2.2e-3 across)
T_CORE = 0.0050
T_IN = 1.0e-6             # innermost band edge; [0, T_IN] is one last band
BAND_R = 0.72             # radius-band ratio
GRID_N_RAY = 2700         # zt-grid for the T-side
ZL = 10                   # acb range; explicit tails added

SQ_PSI_B = PSI.sqrt()
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')

# The true maximizer a* = grad Phi(1,0) and tilt s0 = sqrt(1-q), at the true
# parameters of the certified rectangle, lie within 1e-7 of the stored
# decimals A1S, A2S, S0F.  Every ray origin is inflated by this ball so the
# certificates cover the true rays, on which the pinned identities hold.
_E7 = arb(dec('0.0000001'))
_ORIG = _E7.union(-_E7)
_ANG_PAD = arb(dec('0.00000001'))
_RAD_PAD = arb(dec('0.000000000001'))
_LBOX_QUANT = Decimal('0.000000000001')
_ROOT_QUANT = Decimal('0.0000000001')

SCHEMA_VERSION = 3
EVIDENCE_MODEL = 'exact-leaf-proof-tree-v1'
_WORKER_RUNTIME = None


def _float_fraction(value):
    num, den = float(value).as_integer_ratio()
    return Fraction(num, den)


def _float_record(value):
    return exact.fraction_record(_float_fraction(value))


def _decimal_fraction(text):
    return Fraction(str(text))


def _decimal_record(text):
    return exact.fraction_record(_decimal_fraction(text))


def proof_source_paths():
    import huang_np as nr
    return {
        'huang_region1.py': __file__,
        'block3bc_exact.py': exact.__file__,
        'core.py': core.__file__,
        'huanggrid.py': hg.__file__,
        'huang_hessian.py': hh.__file__,
        'huang_np.py': nr.__file__,
    }


def proof_source_hashes():
    return exact.source_hashes(proof_source_paths())


def _packet_union(values):
    if not values:
        raise ValueError('cannot packetize an empty enclosure family')
    value = values[0]
    for other in values[1:]:
        value = value.union(other)
    return exact.arb_packet(value)


def ang_dist(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def omega_of(theta):
    return min(ang_dist(theta, W_ANG), ang_dist(theta, W_ANG + math.pi))


def T_of_angle(theta):
    om = omega_of(theta)
    if om <= WEDGE_HALF:
        return T_LONG
    if om <= CONE_MID:
        return T_MID
    return T_CORE


def T_max_over(th0, th1):
    """Certified maximum of the piecewise star radius on an angle interval."""
    if th1 - th0 >= 2 * math.pi - 1e-12:
        return T_LONG
    mid = 0.5 * (th0 + th1)
    # Check all copies of the two zone centers near this interval.  Testing
    # interval intersection, rather than sampling, cannot miss a thin long
    # or middle wedge at a cell boundary.
    has_mid = False
    for k in range(-4, 5):
        c = W_ANG + k * math.pi
        if th0 <= c + WEDGE_HALF and th1 >= c - WEDGE_HALF:
            return T_LONG
        if th0 <= c + CONE_MID and th1 >= c - CONE_MID:
            has_mid = True
    return T_MID if has_mid else T_CORE


def _dec(x, nd=12):
    return arb(dec(f"{float(x):.{nd}f}"))


def _exact_decimal(x):
    """Parse a persisted decimal spelling as an exact rational Arb ball."""
    return arb(dec(str(x)))


def _quantize_lbox(values):
    """Outward-quantize a numeric lbox once and return decimal spellings.

    The resulting four strings are the certificate's actual box.  Both the
    boundary localization and B_Lambda must consume Arb values parsed from
    these same spellings; re-rounding either consumer can leave a sliver of
    the declared box unchecked.
    """
    if len(values) != 4:
        raise ValueError("lbox needs four endpoints")
    out = []
    for k, value in enumerate(values):
        rounding = ROUND_FLOOR if k in (0, 2) else ROUND_CEILING
        # Quantize around the actual binary64 proposal, not its shortened
        # display spelling.  The proposal is heuristic, but this makes the
        # relationship between that proposal and the certified box explicit.
        q = Decimal.from_float(float(value)).quantize(
            _LBOX_QUANT, rounding=rounding)
        out.append(format(q, 'f'))
    if not (Decimal(out[0]) < Decimal(out[1])
            and Decimal(out[2]) < Decimal(out[3])):
        raise ValueError(f"degenerate lbox {out}")
    return tuple(out)


def _validate_lbox(box):
    """Require a certainly ordered exact lbox in B_Lambda's domain."""
    if len(box) != 4:
        raise ValueError("lbox needs four Arb endpoints")
    l1lo, l1hi, l2lo, l2hi = box
    if not (l1hi > l1lo and l2hi > l2lo and l1lo > 0):
        raise ValueError(f"invalid exact lbox {box}")


# ---------------------------------------------------------------------------
# acb evaluators
# ---------------------------------------------------------------------------

def _XMth(z, l1, l2):
    X = acb(SQ_PSI_B) * z
    M = X.tanh()
    return X, M, acb(l1) * X + acb(l2) * M


def Phi_acb(l1, l2, tol=None):
    """Phi(lambda) = E log 2cosh(lambda . f), rigorous, with tail bound.
    l1, l2 may be balls.

    Analyticity guard: 2cosh vanishes at i(k+1/2)pi, so off the axis a ball
    of 2cosh(th) can be tightly negative-real; the principal log would then
    return a finite but WRONG enclosure for the analytic continuation.  We
    return a non-finite ball unless Re(2cosh th) is certainly positive,
    forcing the integrator to subdivide toward the axis (where 2cosh >= 2)."""
    def f(z, an):
        X, M, th = _XMth(z, l1, l2)
        c = 2 * th.cosh()
        if not (c.real > 0):
            return acb(arb(0, arb('inf')))
        return c.log() * core.c_phi(z)

    val = core.integrate(f, -ZL, ZL, abs_tol=tol)
    # |log 2cosh(th)| <= |th| + log 2 <= |l1| sqrt(psi) |z| + |l2| + log2
    LOG2 = arb(2).log()
    t = (abs(arb(l1)) * SQ_PSI_B * z1_tail(arb(ZL))
         + (abs(arb(l2)) + LOG2) * gauss_tail_mass(arb(ZL)))
    return val + t.union(-t)


def gradPhi_acb(l1, l2, tol=None):
    """(a1, a2) = grad Phi(lambda) = (E[X tanh th], E[M tanh th]); l1, l2 may
    be balls (widths propagate rigorously)."""
    def f1(z, an):
        X, M, th = _XMth(z, l1, l2)
        return X * th.tanh() * core.c_phi(z)

    def f2(z, an):
        X, M, th = _XMth(z, l1, l2)
        return M * th.tanh() * core.c_phi(z)

    a1 = core.integrate(f1, -ZL, ZL, abs_tol=tol)
    a2 = core.integrate(f2, -ZL, ZL, abs_tol=tol)
    t1 = SQ_PSI_B * z1_tail(arb(ZL))
    t2 = gauss_tail_mass(arb(ZL))
    return a1 + t1.union(-t1), a2 + t2.union(-t2)


def _sech2_corner(l1c, l2c):
    """Integrand factory: z -> sech^2(l1c X + l2c M) f_i f_j phi(z)."""
    def make(i, j):
        def f(z, an):
            X, M, th = _XMth(z, l1c, l2c)
            s2 = 1 / (th.cosh() ** 2)
            fi = X if i == 1 else M
            fj = X if j == 1 else M
            return fi * fj * s2 * core.c_phi(z)
        return f
    return make


def _ff_only():
    def make(i, j):
        def f(z, an):
            X = acb(SQ_PSI_B) * z
            M = X.tanh()
            fi = X if i == 1 else M
            fj = X if j == 1 else M
            return fi * fj * core.c_phi(z)
        return f
    return make


def B_Lambda(l1lo, l1hi, l2lo, l2hi):
    """B_L = E[f f^T max_{lambda in box} sech^2(lambda . f)] as a 2x2 ball
    matrix (b11, b12, b22).

    Corner structure: theta = l1 X + l2 M is bilinear in lambda, so its range
    over the box is spanned by the four corners.  For z > 0 (X, M > 0) the
    minimum of theta is at (l1lo, l2lo); for z < 0 the maximum is there; the
    max of sech^2 over the box is 1 where the corner range straddles 0 and
    sech^2(nearest corner value) otherwise.  With zp = the positive root of
    l1lo X + l2lo M = 0 (zp = 0 if l2lo >= 0), the pointwise max equals
    sech^2(l1lo X + l2lo M) on |z| >= zp and 1 on |z| < zp.  We enlarge
    [-zp, zp] to a rigorous outer bracket [-zq, zq] (max <= 1 always), so the
    three-piece split is sound and each piece is analytic.  The returned
    metadata records the exact split point and its Arb sign certificate."""
    _validate_lbox((l1lo, l1hi, l2lo, l2hi))
    # Corner selection (audited): theta = l1 X + l2 M is increasing in l1
    # and l2 for z > 0 and decreasing in both for z < 0, so on both tails the
    # box extreme of theta nearest zero is attained at (l1lo, l2lo), and the
    # two straddle roots coincide (g(|z|) = l1lo sqrt(psi)|z| +
    # l2lo tanh(sqrt(psi)|z|) by oddness).  Preconditions: l1lo > 0 and a
    # valid bracket g(5) > 0.  The positive-root branch searches an exact
    # decimal grid and accepts only Arb proofs of g(zq)>0 and g'(zq)>0.  The
    # two zq=0 branches instead prove global monotonicity with g(0)=0.
    l1lb, _ = endpoints(l1lo)
    l2lb, l2ub = endpoints(l2lo)
    if not (l1lb > 0):
        raise ValueError(f"B_Lambda needs l1lo > 0, got {l1lo}")

    def g_arb(z):
        z = arb(z)
        x = SQ_PSI_B * z
        return l1lo * x + l2lo * x.tanh()

    def gprime_arb(z):
        z = arb(z)
        x = SQ_PSI_B * z
        sech2 = 1 / (x.cosh() * x.cosh())
        return SQ_PSI_B * (l1lo + l2lo * sech2)

    far_z = _exact_decimal('5.000000000000')
    far_lo, _ = endpoints(g_arb(far_z))
    if not (far_lo > 0):
        raise ValueError(f"B_Lambda Arb bracket invalid: l1lo={l1lo}, "
                         f"l2lo={l2lo}, g(5)={g_arb(far_z)}")

    sum_lo, sum_hi = endpoints(l1lo + l2lo)
    if l2lb >= 0:
        # Both coefficients are nonnegative, hence g(z) >= 0 for z >= 0.
        zq = arb(0)
        split_cert = dict(
            mode='l2_nonnegative', zq=_decimal_record('0'),
            g_zq_packet=exact.arb_packet(arb(0)),
            condition_packet=exact.arb_packet(l2lo),
            g_far_packet=exact.arb_packet(g_arb(far_z)))
    elif sum_lo >= 0:
        # l2 < 0 but g'(z) >= sqrt(psi) (l1+l2) >= 0.
        zq = arb(0)
        split_cert = dict(
            mode='derivative_nonnegative', zq=_decimal_record('0'),
            g_zq_packet=exact.arb_packet(arb(0)),
            condition_packet=exact.arb_packet(l1lo + l2lo),
            g_far_packet=exact.arb_packet(g_arb(far_z)))
    else:
        if not (l2ub < 0):
            raise ValueError(f"B_Lambda cannot classify l2lo={l2lo}")
        if not (sum_hi < 0):
            raise ValueError(f"B_Lambda root branch needs l1lo+l2lo < 0, "
                             f"got {l1lo + l2lo}")
        far_gp_lo, _ = endpoints(gprime_arb(far_z))
        if not (far_gp_lo > 0):
            raise ValueError(f"B_Lambda derivative bracket invalid: "
                             f"g'(5)={gprime_arb(far_z)}")

        # Binary-search exact integer multiples of ROOT_QUANT.  An uncertain
        # Arb sign is treated as not certified and moves the lower endpoint
        # upward; only a certainly-positive g and g' can become the split.
        lo_units = 0
        hi_units = int(Decimal('5') / _ROOT_QUANT)
        while hi_units - lo_units > 1:
            mid_units = (lo_units + hi_units) // 2
            zq_s = format(Decimal(mid_units) * _ROOT_QUANT, 'f')
            zq = _exact_decimal(zq_s)
            g_zq_lo, _ = endpoints(g_arb(zq))
            gp_zq_lo, _ = endpoints(gprime_arb(zq))
            if g_zq_lo > 0 and gp_zq_lo > 0:
                hi_units = mid_units
            else:
                lo_units = mid_units
        zq_s = format(Decimal(hi_units) * _ROOT_QUANT, 'f')
        zq = _exact_decimal(zq_s)
        g_zq_lo, _ = endpoints(g_arb(zq))
        gp_zq_lo, _ = endpoints(gprime_arb(zq))
        if not (g_zq_lo > 0 and gp_zq_lo > 0):
            raise ValueError("certified B_Lambda grid split not found")
        split_cert = dict(
            mode='positive_root_outer', zq=_decimal_record(zq_s),
            g_zq_packet=exact.arb_packet(g_arb(zq)),
            gprime_zq_packet=exact.arb_packet(gprime_arb(zq)),
            g_far_packet=exact.arb_packet(g_arb(far_z)))

    mk_s = _sech2_corner(l1lo, l2lo)
    mk_f = _ff_only()
    out = []
    for (i, j) in ((1, 1), (1, 2), (2, 2)):
        if zq > 0:
            v = (core.integrate(mk_s(i, j), -ZL, -zq)
                 + core.integrate(mk_f(i, j), -zq, zq)
                 + core.integrate(mk_s(i, j), zq, ZL))
        else:
            v = core.integrate(mk_s(i, j), -ZL, ZL)
        # tails: |f_i f_j| <= psi z^2, sqrt(psi)|z|, 1
        if (i, j) == (1, 1):
            t = PSI * z2_tail(arb(ZL))
        elif (i, j) == (1, 2):
            t = SQ_PSI_B * z1_tail(arb(ZL))
        else:
            t = gauss_tail_mass(arb(ZL))
        out.append(v + t.union(-t))
    return out[0], out[1], out[2], split_cert


def Binv_form(b11, b12, b22, v1, v2):
    """v^T B^{-1} v for the 2x2 ball matrix; None if det not certainly > 0."""
    det = b11 * b22 - b12 * b12
    lo, _ = endpoints(det)
    if not (lo > 0):
        return None
    return (v1 * v1 * b22 - 2 * v1 * v2 * b12 + v2 * v2 * b11) / det


# ---------------------------------------------------------------------------
# lambda localization: sublevel edge check
# ---------------------------------------------------------------------------

def edge_check(l1lo, l1hi, l2lo, l2hi, xcorners, fan, nseg=64):
    """Multi-anchor localization.  lambda(x) lies in EVERY sublevel set
        S_k(x) = { lam : Phi(lam) - lam.x <= Phi(lhat_k) - lhat_k.x }
    (the dual value is the global minimum), hence in their intersection
    S(x).  S(x) is convex and contains the fan point lhat_{k*(x)} attaining
    min_k [Phi(lhat_k) - lhat_k.x], which lies inside the box; so if S(x)
    misses the box boundary, S(x) -- hence lambda(x) -- is inside the box.

    Boundary test: for lam on each edge, need for every x in the hull:
    exists k with  g_k(lam; x) := Phi(lam) - Phi(lhat_k) - (lam - lhat_k).x
    > 0.  The maximum is convex, so corners alone do not suffice when the
    anchor can switch.  We therefore use one common anchor or a fixed convex
    combination of anchors on each certified tangent polygon; the resulting
    affine lower bound is checked at every polygon vertex.

    Evaluation: interval evaluation of Phi over a segment ball loses the
    near-cancellation with the linear term (the valley of Phi(.) - (.).x is
    FLAT along the weak direction), so instead each segment uses the exact
    mean-value form
        g_k(lam) >= g_k(mid) - h * max_seg |dPhi/du - x_u|,
    with the derivative dPhi/du = (grad Phi)_u = a_u(lam) enclosed ONCE PER
    EDGE (its smallness near the valley is genuine, and enclosing it does
    not require tightness).  Returns a complete positive-packet proof tree,
    or ``None`` on failure."""
    _validate_lbox((l1lo, l1hi, l2lo, l2hi))
    fan_text = [[f'{float(f1):.10f}', f'{float(f2):.10f}']
                for (f1, f2) in fan]
    fanb = [(_exact_decimal(f1), _exact_decimal(f2))
            for (f1, f2) in fan_text]
    if not all(l1lo <= f1 <= l1hi and l2lo <= f2 <= l2hi
               for f1, f2 in fanb):
        raise ValueError("a localization fan anchor lies outside its lbox")
    phik = [Phi_acb(f1, f2) for (f1, f2) in fanb]
    if xcorners and isinstance(xcorners[0][0], (tuple, list)):
        polygon_text = [[[f'{float(x1):.12f}', f'{float(x2):.12f}']
                         for (x1, x2) in poly]
                        for poly in xcorners]
    else:
        polygon_text = [[[f'{float(x1):.12f}', f'{float(x2):.12f}']
                         for (x1, x2) in xcorners]]
    xpolys = [[(_exact_decimal(x1), _exact_decimal(x2))
               for (x1, x2) in poly] for poly in polygon_text]
    # The boundary margins are thin along the weak valley; a 5e-4 derivative
    # ball can drown them.  Tighten the certified quadrature tolerance.
    tol3 = arb(dec('0.000000001'))

    def seg_ok(fix, fval, u0, u1, depth, edge_index, base_index, path):
        """Certify the segment [u0, u1] of the edge, bisecting adaptively.
        Uses the per-SEGMENT derivative enclosure, which is genuinely small
        near the valley crossing (where the value margin is small too)."""
        # u0/u1/fval are exact Arb values derived from the one persisted
        # lbox.  Keep recursive midpoints in Arb so no edge is silently moved
        # by a second 10- or 12-digit decimal serialization.
        mb = (u0 + u1) / 2
        fb = fval
        sb = u0.union(u1)
        if fix == 2:
            e1, e2 = mb, fb
            a1e, a2e = gradPhi_acb(sb, fb, tol=tol3)
            deriv = a1e
        else:
            e1, e2 = fb, mb
            a1e, a2e = gradPhi_acb(fb, sb, tol=tol3)
            deriv = a2e
        phie = Phi_acb(e1, e2)
        hb = (u1 - u0) / 2
        def poly_certificate(xb):
            # Build the rigorous value intervals at each hull vertex.  The old
            # code accepted a different anchor k at each vertex; that reverses
            # the quantifiers and is not implied by convexity.  We first try
            # one common anchor, then a fixed convex combination of anchors.
            # Since max_k g_k >= sum_k w_k g_k for w_k>=0, sum w_k=1, the latter
            # remains a valid lower bound for every x in this convex polygon.
            vals = []
            for (x1, x2) in xb:
                xu = x1 if fix == 2 else x2
                du = deriv - xu
                dlo, dhi = endpoints(du)
                dmax = abs(dlo).union(abs(dhi))
                slack = hb * dmax
                vals.append([phie - pk - (e1 - f1) * x1
                             - (e2 - f2) * x2 - slack
                             for (f1, f2), pk in zip(fanb, phik)])
            for k in range(len(fanb)):
                if all(endpoints(vals[j][k])[0] > 0
                       for j in range(len(xb))):
                    weights = [Fraction(int(i == k), 1)
                               for i in range(len(fanb))]
                    return {
                        'weights': [exact.fraction_record(w) for w in weights],
                        'lower_packet': _packet_union(
                            [vals[j][k] for j in range(len(xb))]),
                    }

            # Choose weights from midpoint values only; the final decision
            # below is made with the full Arb intervals, so this numerical LP
            # is merely a sound anchor-selection heuristic.
            try:
                from decimal import Decimal, ROUND_DOWN
                from scipy.optimize import linprog
                gv = [[float(v.mid()) for v in row] for row in vals]
                nk = len(fanb)
                Aub = [[-gv[j][k] for k in range(nk)] + [1.0]
                       for j in range(len(xb))]
                sol = linprog([0.0] * nk + [-1.0], A_ub=Aub,
                              b_ub=[0.0] * len(xb),
                              A_eq=[[1.0] * nk + [0.0]], b_eq=[1.0],
                              bounds=[(0.0, None)] * nk + [(None, None)],
                              method='highs')
                if sol.success:
                    q = [Decimal(str(max(0.0, w))).quantize(
                        Decimal('0.0000000001'), rounding=ROUND_DOWN)
                         for w in sol.x[:nk]]
                    imax = max(range(nk), key=lambda i: q[i])
                    q[imax] += Decimal(1) - sum(q)
                    if any(w < 0 for w in q) or sum(q) != Decimal(1):
                        return None
                    wb = [arb(dec(format(w, 'f'))) for w in q]
                    combined = [
                        sum((wb[k] * vals[j][k] for k in range(nk)), arb(0))
                        for j in range(len(xb))]
                    if all(endpoints(value)[0] > 0 for value in combined):
                        return {
                            'weights': [
                                exact.fraction_record(Fraction(w)) for w in q],
                            'lower_packet': _packet_union(combined),
                        }
            except Exception:
                return None
            return None

        poly_certificates = [poly_certificate(xb) for xb in xpolys]
        if all(cert is not None for cert in poly_certificates):
            return [{
                'edge': edge_index,
                'base_segment': base_index,
                'path': path,
                'polygon_certificates': poly_certificates,
            }]
        if depth >= 14:
            if os.environ.get('R1_DEBUG'):
                print(f"  edge_check fail: fix={fix} fval={fval} "
                      f"seg=[{u0},{u1}]", flush=True)
            return None
        m = (u0 + u1) / 2
        left = seg_ok(
            fix, fval, u0, m, depth + 1,
            edge_index, base_index, path + '0')
        if left is None:
            return None
        right = seg_ok(
            fix, fval, m, u1, depth + 1,
            edge_index, base_index, path + '1')
        return None if right is None else left + right

    edges = [(2, l2lo, l1lo, l1hi), (2, l2hi, l1lo, l1hi),
             (1, l1lo, l2lo, l2hi), (1, l1hi, l2lo, l2hi)]
    leaves = []
    for edge_index, (fix, fval, vlo, vhi) in enumerate(edges):
        n0 = 8
        for k in range(n0):
            u0 = vlo + (vhi - vlo) * k / n0
            u1 = vlo + (vhi - vlo) * (k + 1) / n0
            proof = seg_ok(fix, fval, u0, u1, 0, edge_index, k, '')
            if proof is None:
                return None
            leaves.extend(proof)
    return {
        'fan': [[_decimal_record(x) for x in pair] for pair in fan_text],
        'polygons': [
            [[_decimal_record(x) for x in point] for point in poly]
            for poly in polygon_text],
        'initial_segments_per_edge': 8,
        'max_depth': 14,
        'leaves': leaves,
    }


# ---------------------------------------------------------------------------
# T-side quadratic form
# ---------------------------------------------------------------------------

def quadT_box(x1, x2, sb, v1, v2, sdot, gz):
    Td = hh._T_derivs(x1, x2, gz, sb)
    if Td is None:
        return None
    asT1, asT2, Sss = hh._a_s_mixed(x1, x2, sb, gz)
    return (ALPHA * (v1 * v1 * Td['d2T11'] + 2 * v1 * v2 * Td['d2T12']
                     + v2 * v2 * Td['d2T22'])
            + 2 * sdot * (v1 * asT1 + v2 * asT2)
            + sdot * sdot * Sss)


def _sdot_of(th):
    """Nonrigorous diagnostic for choosing the fixed global coefficients."""
    import huang_np as nr
    d = 1e-4
    v = (math.cos(th), math.sin(th))
    s_p = nr.G(A1S + d * v[0], A2S + d * v[1])[0]
    s_m = nr.G(A1S - d * v[0], A2S - d * v[1])[0]
    sd = (s_p - s_m) / (2 * d)
    return sd if sd == sd else 0.0


def _sdot_box(v1, v2):
    """Enclose the exact global ray slope C dot v on an angular cell."""
    return SDOT_C1 * v1 + SDOT_C2 * v2


# ---------------------------------------------------------------------------
# band machinery
# ---------------------------------------------------------------------------

def bands():
    """Jobs (t0, t1, th0, th1): radius bands from T_LONG down to 0, split
    into angular chunks when t1 is large (per-chunk dual fans keep the
    sublevel level-gaps -- hence the lambda boxes and B_L -- small)."""
    out = []
    t1 = T_LONG
    while t1 > T_IN:
        t0 = max(t1 * BAND_R, T_IN)
        if t1 > 4.5e-4:
            # chunk the allowed arcs (around +-w and, when t0 < T_MID/T_CORE,
            # the wider zones); chunk width 0.08 rad
            arcs = []
            if t0 < T_CORE:
                arcs = [(0.0, 2 * math.pi)]
            elif t0 < T_MID:
                for base in (W_ANG, W_ANG + math.pi):
                    arcs.append((base - CONE_MID, base + CONE_MID))
            else:
                for base in (W_ANG, W_ANG + math.pi):
                    arcs.append((base - WEDGE_HALF, base + WEDGE_HALF))
            for (a0, a1) in arcs:
                # narrow chunks near +-w (the lambda image spreads ~220x the
                # transverse x-extent there); coarser elsewhere
                th = a0
                while th < a1 - 1e-12:
                    om = omega_of(th + 0.008)
                    cw = 0.016 if om <= WEDGE_HALF + 0.05 else 0.05
                    out.append((t0, t1, th, min(th + cw, a1)))
                    th += cw
        else:
            out.append((t0, t1, 0.0, 2 * math.pi))
        t1 = t0
    out.append((0.0, T_IN, 0.0, 2 * math.pi))
    return out


def schedule_records():
    return [
        {'index': index, 'geometry': [_float_record(x) for x in job]}
        for index, job in enumerate(bands())
    ]


def _allowed(theta, t0):
    """Directions this band actually processes."""
    return T_of_angle(theta) > t0


def fan_and_box(t0, t1, th0, th1, pad_mult=1.0):
    """Numeric dual fan over the chunk's processed sector and a padded box
    around its hull.  The fan anchors the multi-anchor sublevel localization;
    the box is what edge_check certifies and B_Lambda consumes."""
    import huang_np as nr
    fan = [(1.0, 0.0)]
    nth = 9
    ths = [th0, th1] + [th0 + (th1 - th0) * (k + 0.5) / nth
                         for k in range(nth)]
    for th in ths:
        if not _allowed(th, t0) and (th1 - th0) < 6:
            continue
        Tt = T_of_angle(th)
        for tt in (min(t1, Tt), max(t0, 1e-7)):
            lam = nr.dual_of(A1S + tt * math.cos(th), A2S + tt * math.sin(th))
            if lam is not None and abs(lam[0] - 1) < 2 and abs(lam[1]) < 2:
                fan.append((round(lam[0], 8), round(lam[1], 8)))
    l1s = [f[0] for f in fan]
    l2s = [f[1] for f in fan]
    c1 = 0.5 * (min(l1s) + max(l1s))
    c2 = 0.5 * (min(l2s) + max(l2s))
    # pads: the sublevel sets are FLAT along e_small ~ (0.57, -0.82); the pad
    # must clear the fan's level-gap in that direction, which shrinks with
    # the chunk size.  Escalates via pad_mult on edge_check failure.
    d1 = (0.5 * (max(l1s) - min(l1s)) * 1.7 + 3e-3 + 0.35 * t1) * pad_mult
    d2 = (0.5 * (max(l2s) - min(l2s)) * 1.7 + 9e-3 + 3.0 * t1) * pad_mult
    return fan, _quantize_lbox(
        (c1 - d1, c1 + d1, c2 - d2, c2 + d2))


def xhull_of_band(t0, t1, th0, th1):
    """Convex-hull vertices of the chunk's x-sector {a* + t v(th)}.

    For a narrow chunk: the 4 sector corners plus an outer-arc bulge vertex
    (pushed past the sagitta T dth^2/8, so the hull contains the outer arc;
    the inner-arc chord dips toward a*, only enlarging the hull -- sound).
    Using true sector points avoids phantom bounding-box corners, whose
    far-off duals would inflate the sublevel level-gaps.  For wide chunks
    (full circle) fall back to the circumscribing square."""
    pad = 2e-7        # covers the origin ball (true a* vs stored decimals)
    if (th1 - th0) < 1.0:
        # Use Arb trigonometric interval evaluation to form an axis-aligned
        # rectangle containing the entire polar sector.  The former finite
        # bulge list was not a certified hull (it missed 15 of 836 sectors).
        from core import endpoints
        # Split at every radius-zone boundary before forming tangent polygons;
        # otherwise a rectangular/sector hull can introduce phantom corners.
        cuts = [th0, th1]
        for k in range(-4, 5):
            c = W_ANG + k * math.pi
            for d in (WEDGE_HALF, CONE_MID):
                if th0 < c - d < th1:
                    cuts.append(c - d)
                if th0 < c + d < th1:
                    cuts.append(c + d)
        cuts = sorted(set(cuts))
        polys = []
        subsectors = []
        for ca, cb in zip(cuts[:-1], cuts[1:]):
            nsub = max(1, int(math.ceil((cb - ca) / 0.25)))
            for j in range(nsub):
                aa = ca + (cb - ca) * j / nsub
                bb = ca + (cb - ca) * (j + 1) / nsub
                hh = (bb - aa) / 2
                mm = (aa + bb) / 2
                rr = min(t1, T_max_over(aa, bb))
                vertices = []
                tangent_slacks = []
                for rad in (t0, rr):
                    # A small outward factor survives the 15-digit persisted
                    # coordinate rounding even on the innermost 1e-6 band.
                    rb = rad / math.cos(hh) * (1 + 1e-8)
                    hhb = _dec(hh, 15)
                    tangent_slacks.append(exact.arb_packet(
                        _dec(rb, 15) * hhb.cos() - _dec(rad, 15)))
                    raw = ((rad, aa), (rb, mm), (rad, bb))
                    for rv, ang in raw:
                        thb = _dec(ang, 15)
                        rball = _dec(rv, 15)
                        dx = rball * thb.cos()
                        dy = rball * thb.sin()
                        x1lo, x1hi = endpoints(_dec(A1S, 12) + _ORIG + dx)
                        x2lo, x2hi = endpoints(_dec(A2S, 12) + _ORIG + dy)
                        xx = float(((x1lo + x1hi) / 2).mid())
                        yy = float(((x2lo + x2hi) / 2).mid())
                        for sx in (-1, 1):
                            for sy in (-1, 1):
                                vertices.append((xx + sx * 2.1e-7,
                                                 yy + sy * 2.1e-7))
                polys.append(vertices)
                subsectors.append({
                    'theta_lo': _float_record(aa),
                    'theta_hi': _float_record(bb),
                    'radius_lo': _float_record(t0),
                    'radius_hi': _float_record(rr),
                    'cos_half_packet': exact.arb_packet(_dec(hh, 15).cos()),
                    'tangent_slack_packets': tangent_slacks,
                })
        return polys, {
            'mode': 'tangent_sector_cover',
            'subsectors': subsectors,
        }
    r = t1 + pad
    return ([(A1S + s1 * r, A2S + s2 * r)
             for s1 in (-1, 1) for s2 in (-1, 1)], {
        'mode': 'full_circle_square',
        'radius_hi': _float_record(t1),
        'origin_padding_packet': exact.arb_packet(_dec(pad, 12) - _E7),
    })


def band_job(job):
    """Certify one radius band: edge-check the lambda box, build B_L, then
    certify quadT - v B_L^{-1} v < 0 on every angular cell of the fan
    (adaptive angular bisection on failure).  job = (t0, t1)."""
    set_prec(50)
    source_before = proof_source_hashes()
    worker_runtime = (_WORKER_RUNTIME if _WORKER_RUNTIME is not None else
                      exact.runtime_record(50, 1))
    if len(job) == 5:
        job_index, job = job[0], job[1:]
    else:
        job_index = None
    t0, t1, th0, th1 = job
    geometry = [_float_record(x) for x in job]
    gz = hg.get_zt_grid(9, GRID_N_RAY)
    xc, hull_certificate = xhull_of_band(t0, t1, th0, th1)
    ok_loc = False
    lbox = None
    lboxb = None
    fan = None
    used_pad_mult = None
    localization_certificate = None
    for pad_mult in (1.0, 1.6, 2.4):
        fan, lbox = fan_and_box(t0, t1, th0, th1, pad_mult)
        lboxb = tuple(_exact_decimal(x) for x in lbox)
        try:
            _validate_lbox(lboxb)
        except ValueError:
            # A numeric proposal outside B_Lambda's l1lo>0 domain cannot be
            # a certificate.  Treat it as a failed proposal, not a worker
            # crash; a valid earlier/later pad is still checked normally.
            continue
        localization_certificate = edge_check(*lboxb, xc, fan)
        if localization_certificate is not None:
            ok_loc = True
            used_pad_mult = pad_mult
            break
    if not ok_loc:
        return {
            'index': job_index,
            'geometry': geometry,
            'verdict': 'FAIL',
            'failure': 'edge_check',
            'lbox': ([_decimal_record(x) for x in lbox]
                     if lbox is not None else None),
            'worker_source_sha256_before': source_before,
            'worker_source_sha256_after': proof_source_hashes(),
            'worker_runtime': worker_runtime,
        }
    l1lo, l1hi, l2lo, l2hi = lboxb
    b11, b12, b22, split_cert = B_Lambda(
        l1lo, l1hi, l2lo, l2hi)
    detB = b11 * b22 - b12 * b12
    det_lo, _ = endpoints(detB)
    if not (det_lo > 0):
        raise ValueError('localized B matrix is not positive definite')
    hull_certificate['polygons'] = localization_certificate['polygons']

    def certify_angle(ta, tb_):
        node_geometry = {
            'theta_lo': _float_record(ta),
            'theta_hi': _float_record(tb_),
        }
        tmax = T_max_over(ta, tb_)
        if t0 >= T_max_over(ta, tb_):
            return {
                **node_geometry,
                'kind': 'outside_star',
                'tmax': _float_record(tmax),
            }
        thb = (_dec(ta, 8).union(_dec(tb_, 8))
               + _ANG_PAD.union(-_ANG_PAD))
        v1, v2 = thb.cos(), thb.sin()
        qB = Binv_form(b11, b12, b22, v1, v2)
        sdot = _sdot_box(v1, v2)
        tE = min(t1, tmax)
        ok_piece = qB is not None
        radial_pieces = []
        u = t0
        du = min(4e-4, tE - t0)
        while ok_piece and u < tE - 1e-15:
            u1 = min(u + du, tE)
            tt = (_dec(u, 12).union(_dec(u1, 12))
                  + _RAD_PAD.union(-_RAD_PAD))
            # ray origin inflated by _ORIG: the true maximizer (at the true
            # parameters) lies within 1e-7 of the stored decimals, and the
            # pinned identities hold on the TRUE ray
            x1 = _dec(A1S, 10) + _ORIG + tt * v1
            x2 = _dec(A2S, 10) + _ORIG + tt * v2
            sb = _dec(S0F, 10) + _ORIG + tt * sdot
            slo, _ = endpoints(sb)
            if not (slo > 0):
                ok_piece = False
                break
            qT = quadT_box(x1, x2, sb, v1, v2, sdot, gz)
            val = None if qT is None else qT - qB
            hi = None
            if val is not None:
                _, hi = endpoints(val)
            if val is None or not (hi < 0):
                if du > 2.5e-5:
                    du *= 0.5
                    continue
                ok_piece = False
                break
            radial_pieces.append({
                'radius_lo': _float_record(u),
                'radius_hi': _float_record(u1),
                'radius_box': exact.arb_packet(tt),
                's_box': exact.arb_packet(sb),
                'curvature_packet': exact.arb_packet(val),
            })
            u = u1
            du = min(du * 1.6, 5e-4)
        if not ok_piece:
            if (tb_ - ta) > 2e-3:
                m = 0.5 * (ta + tb_)
                return {
                    **node_geometry,
                    'kind': 'split',
                    'split_at': _float_record(m),
                    'children': [certify_angle(ta, m),
                                 certify_angle(m, tb_)],
                }
            return {
                **node_geometry,
                'kind': 'failure',
            }
        return {
            **node_geometry,
            'kind': 'certified',
            'tmax': _float_record(tmax),
            'radial_end': _float_record(tE),
            'theta_box': exact.arb_packet(thb),
            'qB_packet': exact.arb_packet(qB),
            'sdot_packet': exact.arb_packet(sdot),
            'radial_pieces': radial_pieces,
        }

    nc0 = max(1, int(math.ceil((th1 - th0) / (2 * math.pi / N_ANG))))
    angular_roots = []
    for k in range(nc0):
        ta = th0 + (th1 - th0) * k / nc0
        tb_ = th0 + (th1 - th0) * (k + 1) / nc0
        angular_roots.append(certify_angle(ta, tb_))

    def has_failure(node):
        return (node['kind'] == 'failure'
                or (node['kind'] == 'split'
                    and any(has_failure(child)
                            for child in node['children'])))

    source_after = proof_source_hashes()
    if source_after != source_before:
        raise RuntimeError('Region-I worker source changed mid-job')
    return {
        'index': job_index,
        'geometry': geometry,
        'verdict': ('FAIL' if any(has_failure(root)
                                 for root in angular_roots) else 'PASS'),
        'failure': None,
        'lbox': [_decimal_record(x) for x in lbox],
        'pad_multiplier': _decimal_record(str(used_pad_mult)),
        'hull_certificate': hull_certificate,
        'localization_certificate': localization_certificate,
        'B_certificate': {
            'b11_packet': exact.arb_packet(b11),
            'b12_packet': exact.arb_packet(b12),
            'b22_packet': exact.arb_packet(b22),
            'det_packet': exact.arb_packet(detB),
            'root_certificate': split_cert,
        },
        'angular_roots': angular_roots,
        'worker_source_sha256_before': source_before,
        'worker_source_sha256_after': source_after,
        'worker_runtime': worker_runtime,
    }


def _init():
    global _WORKER_RUNTIME
    set_prec(50)
    exact.apply_worker_policy()
    _WORKER_RUNTIME = exact.runtime_record(50, 1, fresh_flint=True)


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, os.cpu_count() - 2)
    if nw <= 0:
        raise ValueError('workers must be positive')
    jobs = bands()
    print(f"{len(jobs)} radius bands, {nw} workers", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    started = time.time()
    run_id = uuid.uuid4().hex
    source_before = proof_source_hashes()
    runtime_before = exact.runtime_record(50, nw, fresh_flint=True)
    exact.validate_runtime_record(runtime_before, 50, workers=nw)
    out = []
    fails = 0
    with Pool(nw, initializer=_init) as pool:
        indexed_jobs = [(index, *job) for index, job in enumerate(jobs)]
        for res in pool.imap_unordered(band_job, indexed_jobs):
            res['run_id'] = run_id
            if (res['worker_source_sha256_before'] != source_before
                    or res['worker_source_sha256_after'] != source_before
                    or exact.runtime_identity(res['worker_runtime'])
                    != exact.runtime_identity(runtime_before)):
                raise RuntimeError('Region-I worker attestation mismatch')
            out.append(res)
            tag = 'OK  ' if res['verdict'] == 'PASS' else 'FAIL'
            print(f"{tag} job={res['index']} "
                  f"({time.time()-started:.0f}s)", flush=True)
            if res['verdict'] != 'PASS':
                fails += 1
    out.sort(key=lambda row: row['index'])
    source_after = proof_source_hashes()
    runtime_after = exact.runtime_record(50, nw, fresh_flint=True)
    if source_after != source_before or runtime_after != runtime_before:
        raise RuntimeError('Region-I producer source/runtime changed mid-run')

    def counts(node):
        if node['kind'] == 'split':
            left = counts(node['children'][0])
            right = counts(node['children'][1])
            return tuple(a + b for a, b in zip(left, right))
        if node['kind'] == 'certified':
            return 1, len(node['radial_pieces']), 0
        if node['kind'] == 'outside_star':
            return 0, 0, 1
        return 0, 0, 0

    totals = [counts(root) for row in out for root in row.get('angular_roots', [])]
    schedule = schedule_records()
    star = {
        'W_ANG': _float_record(W_ANG),
        'WEDGE_HALF': _float_record(WEDGE_HALF),
        'CONE_MID': _float_record(CONE_MID),
        'T_LONG': _float_record(T_LONG),
        'T_MID': _float_record(T_MID),
        'T_CORE': _float_record(T_CORE),
        'T_IN': _float_record(T_IN),
        'N_ANG': N_ANG,
        'A1S': _float_record(A1S),
        'A2S': _float_record(A2S),
        'S0F': _float_record(S0F),
        'origin_radius': _decimal_record('0.0000001'),
    }
    policy = {
        'precision_bits': 50,
        'lbox_quant_digits': 12,
        'zq_quant_digits': 10,
        'angle_pad': _decimal_record('0.00000001'),
        'radius_pad': _decimal_record('0.000000000001'),
        'sdot_model': SDOT_MODEL,
        'sdot_coefficients': [_decimal_record(x) for x in SDOT_C_TEXT],
    }
    payload = {
        'schema_version': SCHEMA_VERSION,
        'kind': 'huang_region1_certificate',
        'evidence_model': EVIDENCE_MODEL,
        'run_id': run_id,
        'source_sha256': source_before,
        'source_sha256_after': source_after,
        'source_set_sha256': exact.payload_sha256(source_before, omit=()),
        'runtime': runtime_before,
        'certificate_policy': policy,
        'star': star,
        'schedule': schedule,
        'schedule_sha256': exact.payload_sha256(schedule, omit=()),
        'records': out,
        'derived_summary': {
            'jobs': len(out),
            'failures': fails,
            'certified_angular_leaves': sum(x[0] for x in totals),
            'radial_pieces': sum(x[1] for x in totals),
            'outside_star_leaves': sum(x[2] for x in totals),
        },
    }
    payload['certificate_sha256'] = exact.payload_sha256(
        payload, omit=('certificate_sha256',))
    out_path = os.path.join(RESULTS_DIR, 'huang_region1.json')
    exact.write_json_atomic(out_path, payload)
    print(f"DONE {len(jobs)} bands, {fails} fails, "
          f"{time.time()-started:.0f}s",
          flush=True)
    if fails:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
