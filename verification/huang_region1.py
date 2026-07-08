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
import json
import math
from multiprocessing import Pool

from flint import arb, acb
import core
from core import (set_prec, dec, endpoints, PSI, Q, ALPHA,
                  z1_tail, z2_tail, gauss_tail_mass)
import huanggrid as hg
import huang_hessian as hh

# ---------------------------------------------------------------------------
# numeric policy constants (choices; rigor lives in the certificates)
# ---------------------------------------------------------------------------
A1S = 1.1234203619        # ~ a*_1 = psi(1-q)
A2S = 0.5639490799        # ~ a*_2 = q
S0F = 0.6603418817        # ~ s0 = sqrt(1-q)
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

# The true maximizer a* = grad Phi(1,0) and tilt s0 = sqrt(1-q), at the true
# parameters of the certified rectangle, lie within 1e-7 of the stored
# decimals A1S, A2S, S0F (grad Phi ball ~4e-9 plus parameter sensitivity
# ~2e-8).  Every ray origin is inflated by this ball so the certificates
# cover the true rays, on which the pinned identities hold.
_E7 = arb(dec('0.0000001'))
_ORIG = _E7.union(-_E7)


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


def _dec(x, nd=12):
    return arb(dec(f"{float(x):.{nd}f}"))


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
    three-piece split is sound and each piece is analytic."""
    import mpmath
    l1lo_f, l2lo_f = float(l1lo), float(l2lo)
    # Corner selection (audited): theta = l1 X + l2 M is increasing in l1
    # and l2 for z > 0 and decreasing in both for z < 0, so on both tails the
    # box extreme of theta nearest zero is attained at (l1lo, l2lo), and the
    # two straddle roots coincide (g(|z|) = l1lo sqrt(psi)|z| +
    # l2lo tanh(sqrt(psi)|z|) by oddness).  Preconditions: l1lo > 0 and a
    # valid bisection bracket g(5) > 0.
    if not l1lo_f > 0:
        raise ValueError(f"B_Lambda needs l1lo > 0, got {l1lo_f}")
    import math as _m
    _sp = _m.sqrt(float(PSI.mid()))
    if not (l1lo_f * _sp * 5.0 + l2lo_f * _m.tanh(_sp * 5.0) > 0):
        raise ValueError(f"B_Lambda bracket invalid: l1lo={l1lo_f}, "
                         f"l2lo={l2lo_f}")
    if l2lo_f >= 0:
        zq = 0.0
    else:
        g = lambda z: l1lo_f * math.sqrt(float(PSI.mid())) * z \
            + l2lo_f * math.tanh(math.sqrt(float(PSI.mid())) * z)
        lo, hi = 1e-9, 5.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if g(mid) > 0:
                hi = mid
            else:
                lo = mid
        zq = hi * 1.02 + 1e-9      # outer pad: middle piece uses factor 1
    l1c = _dec(l1lo_f, 10)
    l2c = _dec(l2lo_f, 10)
    mk_s = _sech2_corner(l1c, l2c)
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
    return out[0], out[1], out[2]


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
    > 0.  max_k g_k is convex in x, so the hull's corner points suffice.

    Evaluation: interval evaluation of Phi over a segment ball loses the
    near-cancellation with the linear term (the valley of Phi(.) - (.).x is
    FLAT along the weak direction), so instead each segment uses the exact
    mean-value form
        g_k(lam) >= g_k(mid) - h * max_seg |dPhi/du - x_u|,
    with the derivative dPhi/du = (grad Phi)_u = a_u(lam) enclosed ONCE PER
    EDGE (its smallness near the valley is genuine, and enclosing it does
    not require tightness).  Returns bool."""
    fanb = [(_dec(f1, 10), _dec(f2, 10)) for (f1, f2) in fan]
    phik = [Phi_acb(f1, f2) for (f1, f2) in fanb]
    xb = [(_dec(x1, 12), _dec(x2, 12)) for (x1, x2) in xcorners]
    tol3 = arb(dec('0.0005'))

    def seg_ok(fix, fval, u0, u1, depth):
        """Certify the segment [u0, u1] of the edge, bisecting adaptively.
        Uses the per-SEGMENT derivative enclosure, which is genuinely small
        near the valley crossing (where the value margin is small too)."""
        mid = 0.5 * (u0 + u1)
        h = 0.5 * (u1 - u0)
        mb = _dec(mid, 12)
        fb = _dec(fval, 10)
        sb = _dec(u0, 12).union(_dec(u1, 12))
        if fix == 2:
            e1, e2 = mb, fb
            a1e, a2e = gradPhi_acb(sb, fb, tol=tol3)
            deriv = a1e
        else:
            e1, e2 = fb, mb
            a1e, a2e = gradPhi_acb(fb, sb, tol=tol3)
            deriv = a2e
        phie = Phi_acb(e1, e2)
        hb = _dec(h, 14)
        all_ok = True
        for (x1, x2) in xb:
            xu = x1 if fix == 2 else x2
            du = deriv - xu
            dlo, dhi = endpoints(du)
            dmax = max(abs(float(dlo)), abs(float(dhi)))
            slack = hb * _dec(dmax * 1.0000001, 12)
            ok = False
            for (f1, f2), pk in zip(fanb, phik):
                val = phie - pk - (e1 - f1) * x1 - (e2 - f2) * x2 - slack
                lo, _ = endpoints(val)
                if lo > 0:
                    ok = True
                    break
            if not ok:
                all_ok = False
                break
        if all_ok:
            return True
        if depth >= 14:
            if os.environ.get('R1_DEBUG'):
                print(f"  edge_check fail: fix={fix} fval={fval:.5f} "
                      f"seg=[{u0:.6f},{u1:.6f}]", flush=True)
            return False
        m = 0.5 * (u0 + u1)
        return (seg_ok(fix, fval, u0, m, depth + 1)
                and seg_ok(fix, fval, m, u1, depth + 1))

    edges = [(2, l2lo, l1lo, l1hi), (2, l2hi, l1lo, l1hi),
             (1, l1lo, l2lo, l2hi), (1, l1hi, l2lo, l2hi)]
    for (fix, fval, vlo, vhi) in edges:
        n0 = 8
        for k in range(n0):
            u0 = vlo + (vhi - vlo) * k / n0
            u1 = vlo + (vhi - vlo) * (k + 1) / n0
            if not seg_ok(fix, fval, u0, u1, 0):
                return False
    return True


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
    import huang_np as nr
    d = 1e-4
    v = (math.cos(th), math.sin(th))
    s_p = nr.G(A1S + d * v[0], A2S + d * v[1])[0]
    s_m = nr.G(A1S - d * v[0], A2S - d * v[1])[0]
    sd = (s_p - s_m) / (2 * d)
    return sd if sd == sd else 0.0


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
                    cw = 0.016 if om <= WEDGE_HALF + 0.05 else (
                        0.05 if om <= CONE_MID + 0.1 else 0.25)
                    out.append((t0, t1, th, min(th + cw, a1)))
                    th += cw
        else:
            out.append((t0, t1, 0.0, 2 * math.pi))
        t1 = t0
    out.append((0.0, T_IN, 0.0, 2 * math.pi))
    return out


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
    for k in range(nth):
        th = th0 + (th1 - th0) * (k + 0.5) / nth
        if not _allowed(th, t0) and (th1 - th0) < 6:
            continue
        Tt = T_of_angle(th)
        for tt in (min(t1, Tt) * 0.98, max(t0, 1e-7) * 0.6):
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
    return fan, (c1 - d1, c1 + d1, c2 - d2, c2 + d2)


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
        vs = []
        pts = []
        for th in (th0, th1):
            for tt in (t0, min(t1, T_of_angle(th))):
                pts.append((A1S + tt * math.cos(th), A2S + tt * math.sin(th)))
        thm = 0.5 * (th0 + th1)
        bulge = min(t1, T_of_angle(thm)) * (1 + (th1 - th0) ** 2 / 7)
        pts.append((A1S + bulge * math.cos(thm), A2S + bulge * math.sin(thm)))
        for (p1, p2) in pts:
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    vs.append((p1 + s1 * pad, p2 + s2 * pad))
        return vs
    r = t1 + pad
    return [(A1S + s1 * r, A2S + s2 * r) for s1 in (-1, 1) for s2 in (-1, 1)]


def band_job(job):
    """Certify one radius band: edge-check the lambda box, build B_L, then
    certify quadT - v B_L^{-1} v < 0 on every angular cell of the fan
    (adaptive angular bisection on failure).  job = (t0, t1)."""
    set_prec(50)
    t0, t1, th0, th1 = job
    gz = hg.get_zt_grid(9, GRID_N_RAY)
    xc = xhull_of_band(t0, t1, th0, th1)
    ok_loc = False
    lbox = None
    fan = None
    for pad_mult in (1.0, 1.6, 2.4):
        fan, lbox = fan_and_box(t0, t1, th0, th1, pad_mult)
        l1lo, l1hi, l2lo, l2hi = lbox
        if edge_check(l1lo, l1hi, l2lo, l2hi, xc, fan):
            ok_loc = True
            break
    if not ok_loc:
        return dict(band=[t0, t1, th0, th1], ok=False, why='edge_check',
                    lbox=lbox)
    l1lo, l1hi, l2lo, l2hi = lbox
    b11, b12, b22 = B_Lambda(l1lo, l1hi, l2lo, l2hi)
    # angular cells of this chunk, with adaptive bisection
    fails = []
    ncell = 0
    worst = -1e9
    nc0 = max(2, int(math.ceil((th1 - th0) / (2 * math.pi / N_ANG))))
    stack = [(th0 + (th1 - th0) * k / nc0,
              th0 + (th1 - th0) * (k + 1) / nc0) for k in range(nc0)]
    while stack:
        ta, tb_ = stack.pop()
        thc = 0.5 * (ta + tb_)
        if t0 >= T_of_angle(thc):
            continue                      # outside the star in this direction
        thb = _dec(ta, 8).union(_dec(tb_, 8))
        v1, v2 = thb.cos(), thb.sin()
        qB = Binv_form(b11, b12, b22, v1, v2)
        sdf = _sdot_of(thc)
        sdot = _dec(round(sdf, 6), 8)
        # adaptive radial walk: pieces shrink where the enclosure is wide
        # (wide s/x-balls inflate the far-tail cells of the T-integrals)
        tE = min(t1, T_of_angle(thc))
        ok_piece = qB is not None
        hi = None
        u = t0
        du = min(4e-4, tE - t0)
        while ok_piece and u < tE - 1e-15:
            u1 = min(u + du, tE)
            tt = _dec(u, 12).union(_dec(u1, 12))
            # ray origin inflated by _ORIG: the true maximizer (at the true
            # parameters) lies within 1e-7 of the stored decimals, and the
            # pinned identities hold on the TRUE ray
            x1 = _dec(A1S, 10) + _ORIG + tt * v1
            x2 = _dec(A2S, 10) + _ORIG + tt * v2
            sb = _dec(S0F, 10) + _ORIG + tt * sdot
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
            u = u1
            du = min(du * 1.6, 5e-4)
        if not ok_piece:
            if (tb_ - ta) > 2e-3:
                m = 0.5 * (ta + tb_)
                stack.append((ta, m))
                stack.append((m, tb_))
                continue
            fails.append(dict(th=[ta, tb_],
                              hi=None if hi is None else float(
                                  hi.mid() if hasattr(hi, 'mid') else hi)))
            continue
        ncell += 1
        worst = max(worst, float(hi.mid() if hasattr(hi, 'mid') else hi))
    return dict(band=[t0, t1, th0, th1], ok=not fails, cells=ncell,
                worst=worst, fails=fails[:6], nfail=len(fails),
                lbox=[l1lo, l1hi, l2lo, l2hi])


def _init():
    set_prec(50)


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, os.cpu_count() - 2)
    jobs = bands()
    print(f"{len(jobs)} radius bands, {nw} workers", flush=True)
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    out = []
    fails = 0
    with Pool(nw, initializer=_init) as pool:
        for res in pool.imap_unordered(band_job, jobs):
            out.append(res)
            tag = 'OK  ' if res.get('ok') else 'FAIL'
            print(f"{tag} band={res['band']} cells={res.get('cells')} "
                  f"worst={res.get('worst')} nfail={res.get('nfail', 0)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if not res.get('ok'):
                fails += 1
    star = dict(W_ANG=W_ANG, WEDGE_HALF=WEDGE_HALF, CONE_MID=CONE_MID,
                T_LONG=T_LONG, T_MID=T_MID, T_CORE=T_CORE, N_ANG=N_ANG,
                A1S=A1S, A2S=A2S)
    with open('results/huang_region1.json', 'w') as f:
        json.dump(dict(star=star, fails=fails, results=out), f, indent=1)
    print(f"DONE {len(jobs)} bands, {fails} fails, {time.time()-t0:.0f}s",
          flush=True)
    if fails:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
