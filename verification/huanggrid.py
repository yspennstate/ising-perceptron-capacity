"""Fast fixed-grid certified quadrature for Huang's first-moment functional,
in moment coordinates (a1, a2).

Background. Huang's Condition 1.3 asks that
    S_*(l1, l2) = inf_{s>=0}[ s^2 psi/2 + ent(l) + alpha T(a1(l), a2(l), s) ]
be <= 0 for all (l1, l2), where the profile Lam = tanh(l1 X + l2 M),
M = tanh(X), X ~ N(0, psi), and a1 = E[X Lam], a2 = E[M Lam]. Because Lam is
the max-entropy profile for its own moments, ent(l) = H(a1, a2) with
    H(a1,a2) = max{ E ent2((1+Lam)/2) : E[X Lam]=a1, E[M Lam]=a2, |Lam|<=1 }.
So sup_l S_*(l) = sup_{(a1,a2)} [ H(a1,a2) + G(a1,a2) ], G = inf_s[...],
and the sup runs over a BOUNDED rectangle: |a1| <= sqrt(psi), |a2| < sqrt(q).

Duality gives, for every dual point (b1, b2),
    H(a1, a2) <= Phi(b1, b2) - b1 a1 - b2 a2,   Phi(b) = E log 2cosh(b1 X + b2 M).
A finite dictionary of dual points thus yields a piecewise-linear certified
upper bound on H, evaluated with no per-cell integration. The only per-cell
integral is the constraint term
    T(a1, a2, s) = E_z log Psi(V),   z ~ N(0,1),
    V = ( -(a2/q) sqrt(q) z - (a1/psi) N ) / D + s N,
    D = sqrt(1 - a2^2/q),  N = E(-gamma z)/sqrt(1-q),  gamma = sqrt(q/(1-q)),
which is a one-dimensional Gaussian integral evaluated on a fixed grid whose
z-dependent data (N, N', E(-gamma z)) is precomputed once.

All quantities are Arb balls enclosing the truth; (alpha, q, psi) are the
Ding-Sun rectangle (block 1). The grid rule is first order (mean value):
    int_cell g phi  in  g(mid) m0 + g'(cell) m1p - g'(cell) m1m,
    m0 = Psi(lo) - Psi(hi),
    m1p = int_{mid}^{hi} (z-mid) phi >= 0,  m1m = int_{lo}^{mid} (mid-z) phi >= 0,
rigorous because g(z) = g(mid) + g'(xi_z)(z - mid) with g'(xi_z) in g'(cell),
and the positive/negative parts of (z - mid) are handled separately (the
remainder does NOT factor through the signed first moment m1p - m1m).
"""

from flint import arb, acb, ctx
from core import (set_prec, dec, phi, Psi, mills, ALPHA, Q, PSI, GAMMA,
                  z1_tail, z2_tail, gauss_tail_mass, ALPHA_LB, ALPHA_UB,
                  Q_LB, Q_UB, PSI_LB, PSI_UB)

# module-level parameter balls (kappa = 0)
SQ_PSI = PSI.sqrt()
SQ_Q = Q.sqrt()
S1Q = (1 - Q).sqrt()
LOG2 = arb(2).log()


# ---------------------------------------------------------------------------
# Special functions on real balls.
# ---------------------------------------------------------------------------

def E_mills(x):
    """Inverse Mills ratio E(x) = phi(x)/Psi(x) as an arb ball.

    E is strictly increasing, so for a wide ball we enclose via its (thin)
    endpoints; evaluating phi/Psi directly on a wide ball would let Psi touch
    0 and blow up. For narrow balls the direct form is tighter."""
    from core import endpoints
    lo, hi = endpoints(x)
    if hi - lo < 1:
        v = phi(x) / Psi(x)
        if v.is_finite():
            return v
    return (phi(lo) / Psi(lo)).union(phi(hi) / Psi(hi))


def Ep_mills(E, x):
    """E'(x) = E(x)(E(x) - x), given E = E(x)."""
    return E * (E - x)


def logPsi(x):
    """log Psi(x), robust for wide balls. logPsi is decreasing, so enclose a
    wide ball via its (thin) endpoints; the direct Psi(x).log() lets the ball
    of Psi values cross 0 when x is wide and large."""
    from core import endpoints
    v = Psi(x)
    if v > 0:
        w = v.log()
        if w.is_finite():
            return w
    lo, hi = endpoints(x)
    return Psi(hi).log().union(Psi(lo).log())


def log2cosh(x):
    """log(2 cosh x) = |x| + log(1 + e^{-2|x|}), stable for large |x|."""
    a = abs(x)
    return a + (-2 * a).exp().log1p()


def tanh_ball(x):
    return x.tanh()


def ent2_tanh(t):
    """ent2((1+tanh t)/2) = log(2 cosh t) - t tanh t."""
    return log2cosh(t) - t * t.tanh()


# ---------------------------------------------------------------------------
# Grid construction.
# ---------------------------------------------------------------------------

class Grid:
    """A fixed partition of [-L, L] with precomputed moment weights."""

    def __init__(self, L, n):
        self.L = arb(L)
        self.n = n
        h = 2 * self.L / n
        self.lo = []
        self.hi = []
        self.mid = []
        self.cell = []
        self.m0 = []          # int_cell phi
        self.m1c = []         # int_cell (z - mid) phi   (signed; legacy)
        self.m1p = []         # int_{mid}^{hi} (z - mid) phi   >= 0
        self.m1m = []         # int_{lo}^{mid} (mid - z) phi   >= 0
        for j in range(n):
            lo = -self.L + h * j
            hi = -self.L + h * (j + 1)
            mid = (lo + hi) / 2
            self.lo.append(lo)
            self.hi.append(hi)
            self.mid.append(mid)
            self.cell.append(lo.union(hi))
            m0 = Psi(lo) - Psi(hi)
            self.m0.append(m0)
            m1p = (phi(mid) - phi(hi)) - mid * (Psi(mid) - Psi(hi))
            m1m = mid * (Psi(lo) - Psi(mid)) - (phi(lo) - phi(mid))
            self.m1p.append(m1p)
            self.m1m.append(m1m)
            self.m1c.append(m1p - m1m)

    def integrate(self, g_mid, g_cell):
        """int_{-L}^{L} g phi dz given lists g_mid[j] = g(mid_j) (thin) and
        g_cell[j] = an enclosure of g'(xi) for ALL xi in cell j.

        Mean-value rule, done correctly: on cell j,
            g(z) = g(mid) + g'(xi_z)(z - mid),   xi_z in cell j,
        so
            int_cell g phi = g(mid) m0 + int g'(xi_z)(z - mid) phi dz
                           in g(mid) m0 + [g'] m1p - [g'] m1m,
        where m1p/m1m are the positive/negative parts of int (z - mid) phi.
        (The legacy rule [g'] * (m1p - m1m) was WRONG: g'(xi_z) varies with z
        while (z - mid) changes sign, so the remainder does NOT factor through
        the signed first moment. The flaw produced systematically too-tight
        balls -- caught by an independent mpmath cross-check of grad Phi(1,0).)
        """
        total = arb(0)
        for gm, gc, m1p, m1m, m0 in zip(g_mid, g_cell, self.m1p, self.m1m,
                                        self.m0):
            total = total + gm * m0 + gc * m1p - gc * m1m
        return total


# default grids (rebuilt if precision changes)
_GRID_CACHE = {}


import os as _os
GRID_N = int(_os.environ.get('HUANG_GRID_N', '900'))


def get_zt_grid(L=9, n=None):
    if n is None:
        n = GRID_N
    key = ('zt', L, n, ctx.prec)
    if key not in _GRID_CACHE:
        g = Grid(L, n)
        _precompute_zt(g)
        _GRID_CACHE[key] = g
    return _GRID_CACHE[key]


def get_x_grid(L=9, n=None):
    if n is None:
        n = GRID_N
    key = ('x', L, n, ctx.prec)
    if key not in _GRID_CACHE:
        g = Grid(L, n)
        _precompute_x(g)
        _GRID_CACHE[key] = g
    return _GRID_CACHE[key]


def _precompute_zt(g):
    """Box-independent data for the T integral: at midpoints (thin) and over
    cells, the values N, N' and the raw E(-gamma z), needed to form V and V'.
    N(z) = E(-gamma z)/sqrt(1-q);  N'(z) = -gamma E'(-gamma z)/sqrt(1-q)."""
    g.N_mid, g.N_cell = [], []
    g.Np_mid, g.Np_cell = [], []
    g.Ht_mid, g.Ht_cell = [], []
    for mid, cell in zip(g.mid, g.cell):
        for store_val, store_p, store_ht, zz in (
                (g.N_mid, g.Np_mid, g.Ht_mid, mid),
                (g.N_cell, g.Np_cell, g.Ht_cell, cell)):
            x = -GAMMA * zz
            E = E_mills(x)
            Ep = Ep_mills(E, x)
            store_val.append(E / S1Q)
            store_p.append(-GAMMA * Ep / S1Q)
            store_ht.append(SQ_Q * zz)


def _precompute_x(g):
    """Box-independent data for the Phi/profile integrals: X = sqrt(psi) z,
    M = tanh(X) and derivative dM/dz = sqrt(psi)(1-M^2)."""
    g.X_mid, g.X_cell = [], []
    g.M_mid, g.M_cell = [], []
    g.dM_mid, g.dM_cell = [], []
    for mid, cell in zip(g.mid, g.cell):
        for sx, sm, sdm, zz in ((g.X_mid, g.M_mid, g.dM_mid, mid),
                                 (g.X_cell, g.M_cell, g.dM_cell, cell)):
            X = SQ_PSI * zz
            M = X.tanh()
            sx.append(X)
            sm.append(M)
            sdm.append(SQ_PSI * (1 - M * M))


# ---------------------------------------------------------------------------
# Phi(b1, b2) = E log 2cosh(b1 X + b2 M): dual-dictionary log-partition.
# theta(z) = b1 X + b2 M; d theta/dz = b1 sqrt(psi)(1-M^2)... = b1 X'/...
#   X' = sqrt(psi), M' = sqrt(psi)(1-M^2).
# g(z) = log2cosh(theta); g'(z) = tanh(theta) * theta'(z).
# ---------------------------------------------------------------------------

def Phi_of(b1, b2, g=None):
    """E log 2cosh(b1 X + b2 M) as an arb ball, with tail bound.
    Tail |z|>=L: 0 <= log2cosh(theta) - |theta| <= log 2, and
    |theta| <= |b1| sqrt(psi)|z| + |b2|, so the discarded mass is bounded by
    E[ (|b1| sqrt(psi)|z| + |b2| + log2) 1{|z|>=L} ]."""
    if g is None:
        g = get_x_grid()
    b1 = arb(b1)
    b2 = arb(b2)
    gm, gc = [], []
    for Xm, Mm, dMm, Xc, Mc, dMc in zip(
            g.X_mid, g.M_mid, g.dM_mid, g.X_cell, g.M_cell, g.dM_cell):
        th_m = b1 * Xm + b2 * Mm
        gm.append(log2cosh(th_m))
        th_c = b1 * Xc + b2 * Mc
        dth_c = b1 * SQ_PSI + b2 * dMc
        gc.append(th_c.tanh() * dth_c)
    main = g.integrate(gm, gc)
    tail = (abs(b1) * SQ_PSI * z1_tail(g.L) + (abs(b2) + LOG2)
            * gauss_tail_mass(g.L))
    return main + tail.union(-tail)


# ---------------------------------------------------------------------------
# T(a1, a2, s) = E_z log Psi(V).
# V(z) = c0 * Ht(z) + c1 * N(z),  c0 = -(a2/q)/D,  c1 = s - (a1/psi)/D,
# with Ht = sqrt(q) z, so V is affine in (Ht, N).
#   g(z) = log Psi(V);  g'(z) = -E(V) V'(z),  V'(z) = c0 sqrt(q) + c1 N'(z).
# Note Ht(z) = sqrt(q) z so Ht'(z) = sqrt(q).
# Requires a2^2 < q on the ball (checked by caller / returns None).
# ---------------------------------------------------------------------------

def T_meanvalue(c1, c2, r1, r2, s, g=None):
    """Mean-value upper bound for T(a1,a2,s) over the box
    [c1-r1, c1+r1] x [c2-r2, c2+r2], with c1,c2,r1,r2,s thin arb balls:
        T(box) subset T(center) + dT/da1(box)*[-r1,r1] + dT/da2(box)*[-r2,r2].
    Center term is a tight (thin-a) grid integral; the derivative terms are
    grid integrals of  d/da_i log Psi(V) = -E(V) dV/da_i  enclosed over the
    box, multiplied by the small radii. Returns None if a2^2 >= q on the box.

    V = c0 Ht + c1V N,  c0 = -(a2/q)/D,  c1V = s - (a1/psi)/D,  D=sqrt(1-a2^2/q).
    dV/da1 = -(1/(psi D)) N
    dV/da2 = [ -1/(qD) - a2^2/(q^2 D^3) ] Ht + [ -a1 a2/(psi q D^3) ] N
    """
    if g is None:
        g = get_zt_grid()
    from core import endpoints
    a1box = (c1 - r1).union(c1 + r1)
    a2box = (c2 - r2).union(c2 + r2)
    _, a2hi = endpoints(a2box * a2box / Q)
    if not (a2hi < 1):
        return None
    # center value (thin) with z-mean-value grid rule
    Tc = T_of(c1, c2, s, g)
    if Tc is None:
        return None
    # derivative integrands over the a-box
    Db = (1 - a2box * a2box / Q).sqrt()
    c0b = -(a2box / Q) / Db
    c1b = s - (a1box / PSI) / Db
    D3 = Db * Db * Db
    dV_da1_coeffN = -(1 / (PSI * Db))
    dV_da2_coeffHt = -1 / (Q * Db) - a2box * a2box / (Q * Q * D3)
    dV_da2_coeffN = -a1box * a2box / (PSI * Q * D3)
    g1 = arb(0)
    g2 = arb(0)
    for Nc, Htc, m0 in zip(g.N_cell, g.Ht_cell, g.m0):
        V = c0b * Htc + c1b * Nc
        E = E_mills(V)
        dVda1 = dV_da1_coeffN * Nc
        dVda2 = dV_da2_coeffHt * Htc + dV_da2_coeffN * Nc
        g1 = g1 + (-E * dVda1) * m0
        g2 = g2 + (-E * dVda2) * m0
    # derivative-tail: |dT/da_i| integrand bounded; add crude tail via E(V)<=1+|V|
    # (the discarded |z|>L mass is tiny; fold a conservative constant)
    tail1 = _dT_tail(c0b, c1b, dV_da1_coeffN, arb(0), g.L)
    tail2 = _dT_tail(c0b, c1b, dV_da2_coeffN, dV_da2_coeffHt, g.L)
    g1 = g1 + tail1.union(-tail1)
    g2 = g2 + tail2.union(-tail2)
    return Tc + g1 * r1.union(-r1) + g2 * r2.union(-r2)


def _dT_tail(c0, c1, cN, cHt, L):
    """Crude bound on int_{|z|>L} E(V)|dV/da| phi dz, dV/da = cN N + cHt Ht.
    E(V) <= 1 + |V|, |V| <= |c0|sqrt(q)|z| + |c1|(1+gamma|z|)/S1Q,
    |N| <= (1+gamma|z|)/S1Q, |Ht| = sqrt(q)|z|. All moments closed form."""
    A = abs(c1) / S1Q
    B = abs(c0) * SQ_Q + abs(c1) * GAMMA / S1Q       # |V| <= A + B|z|
    dN = abs(cN) / S1Q                                # |cN N| <= dN(1+gamma|z|)
    dNg = abs(cN) * GAMMA / S1Q
    dH = abs(cHt) * SQ_Q                              # |cHt Ht| <= dH|z|
    m0 = gauss_tail_mass(L)
    m1 = z1_tail(L)
    m2 = z2_tail(L)
    # (1 + A + B|z|)(dN + dNg|z| + dH|z|) integrated over |z|>L
    p = dN + dNg * 0                                  # constant part coeff
    # expand (1+A + B|z|)(dN + (dNg+dH)|z|):
    e = dNg + dH
    c_const = (1 + A) * dN
    c_lin = (1 + A) * e + B * dN
    c_quad = B * e
    return c_const * m0 + c_lin * m1 + c_quad * m2


def T_of(a1, a2, s, g=None):
    """Certified E log Psi(V). Returns arb ball or None if a2^2 >= q."""
    if g is None:
        g = get_zt_grid()
    a2sq = a2 * a2 / Q
    from core import endpoints
    _, hi = endpoints(a2sq)
    if not (hi < 1):
        return None
    D = (1 - a2sq).sqrt()
    c0 = -(a2 / Q) / D
    c1 = s - (a1 / PSI) / D
    gm, gc = [], []
    for Nm, Npm, Htm, Nc, Npc, Htc in zip(
            g.N_mid, g.Np_mid, g.Ht_mid, g.N_cell, g.Np_cell, g.Ht_cell):
        Vm = c0 * Htm + c1 * Nm
        gm.append(logPsi(Vm))
        Vc = c0 * Htc + c1 * Nc
        E = E_mills(Vc)
        Vp = c0 * SQ_Q + c1 * Npc
        gc.append(-E * Vp)
    main = g.integrate(gm, gc)
    # tail: |V| <= A + B|z|, A = |c1|(1+gamma... )/... ; use N <= (1+gamma|z|)/S1Q
    # |V| <= |c0| sqrt(q)|z| + |c1| (1 + gamma|z|)/S1Q
    A = abs(c1) * (1 / S1Q)
    B = abs(c0) * SQ_Q + abs(c1) * GAMMA / S1Q
    # log Psi(V) in [ -(V^2/2 + log(2pi)/2 + |V| + 2Psi(|V|)), 0 ]
    quad = ((A * A / 2 + (2 * arb.pi()).log() / 2 + A + 1) * gauss_tail_mass(g.L)
            + (A * B + B) * z1_tail(g.L) + (B * B / 2) * z2_tail(g.L))
    return main + (-quad).union(arb(0))


# ---------------------------------------------------------------------------
# Assembled value and the certified upper bound over an (a1, a2) cell.
# ---------------------------------------------------------------------------

def S_moment(a1, a2, s, duals=None, gz=None, gx=None):
    """Upper bound for H(a1,a2) + s^2 psi/2 + alpha T(a1,a2,s), using the
    dual dictionary for H. Returns (value_ball, T_ball) or (None, None) if
    a2^2 >= q. duals: list of (b1, b2, Phi_ball)."""
    T = T_of(a1, a2, s, gz)
    if T is None:
        return None, None
    if duals is None:
        Hub = arb(0).union(LOG2)     # trivial 0 <= H <= log 2
    else:
        Hub = None
        for (b1, b2, Phi) in duals:
            u = Phi - b1 * a1 - b2 * a2
            Hub = u if Hub is None else _ball_min(Hub, u)
    val = Hub + s * s * PSI / 2 + ALPHA * T
    return val, T


def _ball_min(a, b):
    """Enclosure of min(a, b) for arb balls."""
    from core import endpoints
    alo, ahi = endpoints(a)
    blo, bhi = endpoints(b)
    lo = alo if (alo < blo) else blo
    hi = ahi if (ahi < bhi) else bhi
    return lo.union(hi)


# ---------------------------------------------------------------------------
# Support function of the achievable moment body K = { (E[X Lam], E[M Lam]) :
# |Lam| <= 1 } of (X, M), X ~ N(0, psi), M = tanh(X):
#     h(u, v) = sup_{|Lam|<=1} E[Lam (u X + v M)] = E |u X + v M|.
# A cell (a-box) lies OUTSIDE K iff some direction (u, v) has
#     min_{a in box} (u a1 + v a2) > h(u, v).
# We use an upper bound on h (so the test is conservative / rigorous).
# ---------------------------------------------------------------------------

def h_support_upper(u, v, g=None):
    """Upper bound on h(u,v) = E|u X + v M|, X = sqrt(psi) z.

    Mean-value per cell where f = uX+vM keeps a sign: |f| smooth with
    d|f|/dz = sign(f) f', f' = sqrt(psi)(u + v(1-M^2)); at cells where f may
    change sign, fall back to the 0th-order enclosure |f(cell)|.upper. Plus a
    tail bound |f| <= |u| sqrt(psi)|z| + |v|."""
    if g is None:
        g = get_x_grid()
    from core import endpoints
    u = arb(u)
    v = arb(v)
    total = arb(0)
    for Xm, Mm, Xc, Mc, dMc, m0, m1p, m1m in zip(
            g.X_mid, g.M_mid, g.X_cell, g.M_cell, g.dM_cell, g.m0, g.m1p,
            g.m1m):
        fc = u * Xc + v * Mc
        flo, fhi = endpoints(fc)
        if flo > 0 or fhi < 0:
            # constant sign: mean-value  |f| = sign * f
            sgn = arb(1) if flo > 0 else arb(-1)
            fm = u * Xm + v * Mm
            dfc = u * SQ_PSI + v * dMc     # f'(cell); dMc = sqrt(psi)(1-M^2)
            total = total + sgn * (fm * m0 + dfc * m1p - dfc * m1m)
        else:
            _, afhi = endpoints(abs(fc))
            total = total + afhi * m0    # |f(cell)|.upper * m0
    tail = abs(u) * SQ_PSI * z1_tail(g.L) + abs(v) * gauss_tail_mass(g.L)
    return total + tail


_HFAN = {}


def _get_hfan(ndir=360):
    """Precompute h upper bounds on a fan of directions (unit circle)."""
    key = (ndir, ctx.prec, GRID_N)
    if key not in _HFAN:
        import math
        fan = []
        for k in range(ndir):
            ang = math.pi * k / ndir     # half circle; sign symmetry covers rest
            u = arb(dec(str(round(math.cos(ang), 8))))
            v = arb(dec(str(round(math.sin(ang), 8))))
            fan.append((u, v, h_support_upper(u, v)))
        _HFAN[key] = fan
    return _HFAN[key]


def outside_K(a1lo, a1hi, a2lo, a2hi, ndir=360):
    """True if the a-box is certified to lie outside the achievable body K
    (hence contributes nothing to sup S_*). Tests a fan of directions."""
    return outside_K_witness(a1lo, a1hi, a2lo, a2hi, ndir) is not None


def outside_K_witness(a1lo, a1hi, a2lo, a2hi, ndir=360):
    """Return the exact separating support witness discarded by outside_K."""
    import block3bc_exact as exact
    from core import endpoints
    a1lo, a1hi = arb(dec(str(a1lo))), arb(dec(str(a1hi)))
    a2lo, a2hi = arb(dec(str(a2lo))), arb(dec(str(a2hi)))
    for fan_index, (u, v, hub) in enumerate(_get_hfan(ndir)):
        # min over box of u a1 + v a2, for both (u,v) and (-u,-v)
        for sign, (su, sv) in ((1, (u, v)), (-1, (-u, -v))):
            mn = _corner_min(su, sv, a1lo, a1hi, a2lo, a2hi)
            if mn > hub:
                return {
                    'fan_index': fan_index,
                    'sign': sign,
                    'u_packet': exact.arb_packet(su),
                    'v_packet': exact.arb_packet(sv),
                    'support_upper_packet': exact.arb_packet(hub),
                    'box_min_packet': exact.arb_packet(mn),
                }
    return None


def _corner_min(u, v, a1lo, a1hi, a2lo, a2hi):
    t1 = u * a1lo if (u > 0) else u * a1hi
    t2 = v * a2lo if (v > 0) else v * a2hi
    lo1, _ = _end(t1)
    lo2, _ = _end(t2)
    return lo1 + lo2


def _end(x):
    from core import endpoints
    return endpoints(x)
