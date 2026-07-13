"""The Ding-Sun limiting functions, with certified enclosures.

Notation follows their sections 8-9 (kappa = 0 throughout):

  D_H(A)   = (A^2-1)(1-m^2)^2 / (Delta+1)^2,  Delta = sqrt(A^2+m^2-A^2 m^2),
             m = tanh(H)
  ell(A)   = E[ D_{sqrt(psi) Z}(A) ] / (1-q)
  Gamma(H,D) = entropy of the pair table (1/4)[(1+m)^2+D, 1-m^2-D;
                                              1-m^2-D, (1-m)^2+D]
  frH(A)   = -2 H_star + E Gamma(sqrt(psi) Z, D_{sqrt(psi) Z}(A))
  I_s(lam) = alpha E_z int_0^infty log Psi(g - S) phi(gamma z + x)/Psi(gamma z) dx
             g = gamma c z - lam x/sqrt(1-lam^2),  c = sqrt((1-lam)/(1+lam)),
             S = E(gamma z) s / (sqrt(psi) sqrt(1-q))

Two integration backends. Real-interval Riemann sums (no analyticity needed)
for the D/Gamma family, which has sqrt branch points at A ~ 0 that defeat
complex quadrature. Nested adaptive acb.integral for the smooth log Psi
double integrals. Parameters are passed as arb balls covering their whole
ranges, so all of Ding-Sun's manual corner/envelope case analysis is
subsumed by ball arithmetic.
"""

from flint import arb, acb
from core import (rat, dec, hull, iv, phi, Psi, logPsi, mills, endpoints,
                  c_phi, c_Psi, c_logPsi, c_mills,
                  integrate, gauss_tail_mass, z1_tail, z2_tail,
                  sqrt_nonneg, sq_nonneg, sq_any, pos_part,
                  ALPHA, ALPHA_LB, ALPHA_UB, Q, PSI, GAMMA,
                  Q_LB, Q_UB, PSI_LB, PSI_UB, GAMMA_LB, GAMMA_UB,
                  gamma_of, ent2_tanh)

LOG4 = arb(4).log()


def _c_lambda_upper(lam):
    """Upper bound for c_lambda=sqrt((1-lambda)/(1+lambda)).

    c_lambda is at most one on the positive branch.  Using |lambda| there
    would needlessly inflate the tail bounds and can erase the small PG
    margins; only the negative part contributes c_lambda>1.
    """
    lo, hi = endpoints(arb(lam))
    if lo >= 0:
        return arb(1)
    al = -lo
    if hi > al:
        al = hi
    if not (1 - al > 0):
        return arb(0, arb('inf'))
    return ((1 + al) / (1 - al)).sqrt()


def _mills_range(x):
    """Tight real-ball range of the increasing inverse Mills ratio."""
    lo, hi = endpoints(arb(x))
    return mills(lo).union(mills(hi))


# ---------------------------------------------------------------------------
# Entropy helpers on real balls.
# ---------------------------------------------------------------------------

def neg_plogp(p):
    """Enclosure of -p log p for a ball p inside [0,1] that may touch 0
    (or dip slightly below from rounding). -x log x is increasing on
    [0, 1/e] with value 0 at 0."""
    if p > 0:
        return -p * p.log()
    u = arb(p.mid()) + arb(p.rad())      # upper endpoint of the ball
    if not (u > 0):
        return arb(0)                    # true p is 0
    inv_e = (-arb(1)).exp()
    if u < inv_e:
        m = -u * u.log()
    else:
        m = inv_e
    return arb(0).union(m)


def Gamma_HD(m, D):
    """Pair entropy Gamma(H, D) with m = tanh(H) and D as real balls."""
    t_pp = ((1 + m) ** 2 + D) / 4
    t_mm = ((1 - m) ** 2 + D) / 4
    t_pm = (1 - m * m - D) / 4
    return neg_plogp(t_pp) + neg_plogp(t_mm) + 2 * neg_plogp(t_pm)


def D_of(m, A):
    """D_H(A) for real balls m = tanh(H), A >= 0.

    Uses the well-conditioned identity (Delta^2 - 1 = (A^2-1)(1-m^2)):
        D_H(A) = (1 - m^2)(Delta - 1)/(Delta + 1) = (1 - m^2)(1 - 2/(Delta+1)),
    Delta = sqrt(A^2(1-m^2) + m^2) >= 1. Each of A, Delta occurs once, so
    wide A-cells no longer suffer the numerator/denominator dependency
    blow-up of the raw (A^2-1)/(Delta+1)^2 form.
    """
    from core import sq_any, sq_nonneg, pos_part
    m2 = sq_any(m)
    A2 = sq_nonneg(A)
    delta2 = pos_part(A2 * (1 - m2) + m2)
    Delta = sqrt_nonneg(delta2)
    return (1 - m2) * (1 - 2 / (Delta + 1))


# ---------------------------------------------------------------------------
# Real-interval Riemann integration of w(z) phi(z) over [0, L] (for even
# integrands, double it). w maps an arb ball to an arb ball.
# ---------------------------------------------------------------------------

def riemann_gauss(w, a, b, n):
    """Enclosure of int_a^b w(z) phi(z) dz by n interval cells."""
    a, b = arb(a), arb(b)
    h = (b - a) / n
    total = arb(0)
    for j in range(n):
        lo = a + h * j
        hi = a + h * (j + 1)
        cell = lo.union(hi)
        # int_cell phi = Psi(lo) - Psi(hi), exact up to ball rounding
        mass = Psi(lo) - Psi(hi)
        total = total + w(cell) * mass
    return total


# ---------------------------------------------------------------------------
# ell(A) and its computable sandwich; frH(A) upper bound.
# ell integrand is even in z. Tail |z| >= 9: |D| <= 1.
# ---------------------------------------------------------------------------

ZCUT = arb(9)
N_ELL = 500
N_GAM = 600
A_SAFE = None  # threshold below which the sqrt branch point forces Riemann


def _a_safe():
    global A_SAFE
    if A_SAFE is None:
        A_SAFE = dec('0.05')
    return A_SAFE


def c_D_of(m, A, an):
    """D_H(A) on acb, well-conditioned form (see D_of).
    delta2 = A^2(1-m^2) + m^2 > 0 on the axis."""
    m2 = m * m
    delta2 = A * A * (1 - m2) + m2
    Delta = delta2.sqrt(analytic=an)
    return (1 - m2) * (1 - 2 / (Delta + 1))


def c_Gamma_HD(m, D, an):
    """Pair entropy on acb with analyticity guards on the logs."""
    t_pp = ((1 + m) ** 2 + D) / 4
    t_mm = ((1 - m) ** 2 + D) / 4
    t_pm = (1 - m * m - D) / 4
    return -(t_pp * t_pp.log(analytic=an) + t_mm * t_mm.log(analytic=an)
             + 2 * t_pm * t_pm.log(analytic=an))


def ell_range(A, n=N_ELL):
    """Enclosure of ell(A) for a ball A, parameters at full range."""
    loA, hiA = endpoints(A)
    if hiA == 0:
        return ell_zero(n)
    s = PSI.sqrt()
    tail = gauss_tail_mass(ZCUT) / (1 - Q_UB)
    if A > _a_safe():
        cA = acb(A)
        body = 2 * integrate(
            lambda z, an: c_D_of((acb(s) * z).tanh(), cA, an) * c_phi(z),
            arb(0), ZCUT, pieces=[3]) / (1 - Q)
    else:
        def w(z):
            return D_of((s * z).tanh(), A)

        body = 2 * riemann_gauss(w, 0, ZCUT, n) / (1 - Q)
    return body + tail.union(-tail)


def ell_zero(n=N_ELL):
    """Exact A=0 limiting enclosure used by the final negative grid cell.

    At A=0, Delta=|tanh(sqrt(psi) z)| and
    D_H(0)=-(1-|tanh(sqrt(psi)z)|)^2.  Evaluating this limit directly avoids
    the removable 0/0 dependency in the generic Delta formula and makes the
    endpoint certificate explicit.  The omitted Gaussian tail is negative,
    so it is added as [-tail,0].
    """
    s = PSI.sqrt()
    def w(z):
        m = (s * z).tanh()
        return -sq_nonneg(1 - abs(m))
    body = 2 * riemann_gauss(w, 0, ZCUT, n) / (1 - Q)
    tail = gauss_tail_mass(ZCUT) / (1 - Q_UB)
    return body + (-tail).union(arb(0))


def H_star_lower():
    """Certified lower bound for H_star = E ent2((1+tanh(sqrt(psi) Z))/2)."""
    s = acb(PSI.sqrt())
    val = 2 * integrate(
        lambda z, an: ((2 * (s * z).cosh()).log()
                       - (s * z) * (s * z).tanh()) * c_phi(z),
        arb(0), ZCUT, pieces=[3])
    return val  # dropped tail is nonnegative


def _gamma_terms(m, D):
    t_pp = ((1 + m) ** 2 + D) / 4
    t_mm = ((1 - m) ** 2 + D) / 4
    t_pm = (1 - m * m - D) / 4
    return t_pp, t_mm, t_pm


def _dGamma_dz(m, A):
    """d/dz Gamma(sqrt(psi) z, D_{sqrt(psi) z}(A)) as a real ball, or None
    when a log argument cannot be certified positive.

    dGamma/dm = -(1+m)/2 (log t_pp + 1) + (1-m)/2 (log t_mm + 1)
                + m (log t_pm + 1)
    dGamma/dD = (1/4) log(t_pm^2/(t_pp t_mm))
    dD/dm = -2m(Delta - 1)/Delta;  dm/dz = sqrt(psi)(1 - m^2).
    """
    t_pp, t_mm, t_pm = _gamma_terms(m, D_of(m, A))
    if not (t_pp > 0 and t_mm > 0 and t_pm > 0):
        return None
    lpp, lmm, lpm = t_pp.log(), t_mm.log(), t_pm.log()
    dG_dm = (-(1 + m) / 2 * (lpp + 1) + (1 - m) / 2 * (lmm + 1)
             + m * (lpm + 1))
    dG_dD = (2 * lpm - lpp - lmm) / 4
    from core import sq_any, sq_nonneg, pos_part
    m2 = sq_any(m)
    delta2 = pos_part(sq_nonneg(A) * (1 - m2) + m2)
    Delta = sqrt_nonneg(delta2)
    if not (Delta > 0):
        return None
    dD_dm = -2 * m * (Delta - 1) / Delta
    dm_dz = PSI.sqrt() * (1 - m2)
    return (dG_dm + dG_dD * dD_dm) * dm_dz


GAM_ZCUT = arb(9) / 2


def frH_upper(A, n=N_GAM, H_lb=None):
    """Upper bound for frH(A) = -2 H_star + E Gamma(...). Even integrand.

    Gamma integral: mean-value-form Riemann on [0, GAM_ZCUT] (real interval
    arithmetic, second order accurate), plus Gamma <= log 4 on the tail.
    Falls back to a plain hull on cells where the derivative enclosure
    degenerates (only near the cutoff, where the cell mass is tiny).
    """
    s = PSI.sqrt()
    if H_lb is None:
        H_lb = H_star_lower()

    def w(z):
        m = (s * z).tanh()
        return Gamma_HD(m, D_of(m, A))

    a, b = arb(0), GAM_ZCUT
    h = (b - a) / n
    total = arb(0)
    for j in range(n):
        lo = a + h * j
        hi = a + h * (j + 1)
        cell = lo.union(hi)
        mass = Psi(lo) - Psi(hi)
        midpt = (lo + hi) / 2
        m_cell = (s * cell).tanh()
        deriv = _dGamma_dz(m_cell, A)
        if deriv is not None and deriv.is_finite():
            # f(z) = f(mid) + f'(xi_z)(z - mid): split f'(cell) into
            # midpoint and radius; the radius couples to int |z - mid| phi.
            corr_signed = phi(lo) - phi(hi) - midpt * mass
            mass_lo = Psi(lo) - Psi(midpt)
            mass_hi = Psi(midpt) - Psi(hi)
            corr_abs = (midpt * mass_lo - (phi(lo) - phi(midpt))
                        + (phi(midpt) - phi(hi)) - midpt * mass_hi)
            d_mid = arb(deriv.mid())
            d_rad = arb(deriv.rad())
            slack = d_rad * corr_abs
            total = (total + w(midpt) * mass + d_mid * corr_signed
                     + slack.union(-slack))
        else:
            total = total + w(cell) * mass
    gam = 2 * total + LOG4 * gauss_tail_mass(GAM_ZCUT)
    return -2 * H_lb + gam


# ---------------------------------------------------------------------------
# The log Psi double integrals. Negative integrand, so restriction to the
# box [-9,9] x [0,9] is always an upper bound; this is what part (a) needs.
#
# Inner integral over x for a fixed complex ball z, then adaptive over z.
# ---------------------------------------------------------------------------

XCUT = arb(9)


def _finite_shifted_gauss_moments(h, length):
    """Return int_0^length y^k phi(h+y) dy for k = 0, 1, 2.

    This is a real-ball, closed-form calculation.  In particular, it is not
    passed to ``acb.integral``: the whole point of the helper is to evaluate
    the large-logPsi part without asking the complex integrator to certify a
    neighbourhood of the logarithmic branch point.
    """
    h = arb(h)
    length = arb(length)
    hp = h + length
    ph, php = phi(h), phi(hp)
    Psh, Pshp = Psi(h), Psi(hp)

    # Raw moments in u over [h, h+length].
    r0 = Psh - Pshp
    r1 = ph - php
    r2 = h * ph + Psh - hp * php - Pshp

    # Shift u = h+y.  Natural interval evaluation is rigorous but can dip a
    # hair below zero through dependency; the mathematical moments are >= 0.
    m0 = pos_part(r0)
    m1 = pos_part(r1 - h * r0)
    m2 = pos_part(r2 - 2 * h * r1 + sq_any(h) * r0)
    return m0, m1, m2


def _ibase_mills_tail_cell(zlo, zhi, xlo, xhi, beta, t_at_xlo):
    """Upper-bound one hard cell of the base I_s integral, in closed form.

    On this cell ``t = g-S`` is certified positive at ``xlo`` and increases
    with x.  If T is a lower bound for t(z,xlo), B a lower bound for beta,
    and y=x-xlo, then t >= T+B*y.  For t>0, the elementary Mills bound gives

      log Psi(t) <= -t^2/2 - log(t) - log(sqrt(2*pi))
                 <= -(T+B*y)^2/2 - log(T) - log(sqrt(2*pi)).

    The last expression is quadratic in y, so its x integral is exactly a
    combination of the finite Gaussian moments above.  The remaining z
    dependence is enclosed on the real interval and multiplied by the exact
    Gaussian mass of the z cell (a rigorous weighted Riemann enclosure).
    """
    zlo, zhi = arb(zlo), arb(zhi)
    xlo, xhi = arb(xlo), arb(xhi)
    zcell = zlo.union(zhi)

    T, _ = endpoints(t_at_xlo)
    B, _ = endpoints(beta)
    if not (T > 0 and B > 0):
        return arb(0, arb('inf'))

    log_sqrt_2pi = (2 * arb.pi()).log() / 2
    p0 = -sq_nonneg(T) / 2 - T.log() - log_sqrt_2pi
    p1 = -T * B
    p2 = -sq_nonneg(B) / 2

    h = GAMMA * zcell
    m0, m1, m2 = _finite_shifted_gauss_moments(h + xlo, xhi - xlo)
    xint = p0 * m0 + p1 * m1 + p2 * m2
    z_mass = Psi(zlo) - Psi(zhi)
    return z_mass * xint / Psi(h)


def _iprime_hard_tail_cell(zlo, zhi, xlo, xhi, lam):
    """Absolute bound for an in-box hard wedge of the I_s' integrand.

    This uses the same inequalities as ``_iprime_tail`` but keeps the finite
    cell and integrates its quadratic x polynomial by real Gaussian moments:
    E(g-S) <= E(g) <= 1+|g| and
    |dg/dlambda| <= (2*gamma*|z|+x)(1-lambda^2)^(-3/2).
    """
    zlo, zhi = arb(zlo), arb(zhi)
    xlo, xhi = arb(xlo), arb(xhi)
    zcell = zlo.union(zhi)

    def upper(x):
        return endpoints(x)[1]

    cmax = _c_lambda_upper(lam)
    a = upper(GAMMA_UB * cmax)
    beta = upper(abs(arb(lam)) / (1 - arb(lam) * arb(lam)).sqrt())
    b32 = upper((1 - arb(lam) * arb(lam)) ** arb(-1.5))
    zmax = upper(abs(zcell))

    # y=x-xlo.  Bound (1+|g|)|dg/dlambda| by
    # (g0+g1*y)(d0+d1*y), all coefficients nonnegative.
    g0 = 1 + a * zmax + beta * xlo
    g1 = beta
    d0 = b32 * (2 * GAMMA_UB * zmax + xlo)
    d1 = b32
    p0 = g0 * d0
    p1 = g0 * d1 + g1 * d0
    p2 = g1 * d1

    h = GAMMA * zcell
    m0, m1, m2 = _finite_shifted_gauss_moments(h + xlo, xhi - xlo)
    cell = ((Psi(zlo) - Psi(zhi))
            * (p0 * m0 + p1 * m1 + p2 * m2) / Psi(h))
    ub = upper(cell)
    return arb(0) if ub < 0 else ub


def I_upper_box(lam, s_tilt, alpha=None, zlo=-6, zhi=6, xhi=None,
                inner_tol_bits=22, outer_tol_bits=18):
    """Upper bound for I_{s}(lam): the (negative) double integral restricted
    to [zlo, zhi] x [0, xhi]. lam is an arb ball; s_tilt an arb >= 0.

    The integrand is negative, so any domain restriction gives an upper
    bound; [-6, 6] x [0, 9] keeps the discarded mass far below every grid
    margin while avoiding the erfc phase-oscillation zone near |z| = 9
    that stalls complex ball evaluation.

    For lambda >= 0 this uses the original nested adaptive quadrature.  For
    lambda < 0, only the moderate-t part uses complex x quadrature; the hard
    wedge is bounded by the real closed-form Mills calculation above, and
    the outer z integral is a mean-value Gaussian-cell enclosure.  Every
    numerical error remains in the returned ball.
    """
    if alpha is None:
        alpha = ALPHA
    if xhi is None:
        xhi = XCUT
    lam_c = acb(lam)
    c_lam = ((1 - lam_c) / (1 + lam_c)).sqrt()
    root = (1 - lam_c * lam_c).sqrt()
    g_pref = acb(GAMMA) * c_lam
    s_norm = acb(s_tilt) / (acb(PSI).sqrt() * (1 - acb(Q)).sqrt())

    tol_in = arb(2) ** (-inner_tol_bits)
    tol_out = arb(2) ** (-outer_tol_bits)

    def inner_conditional(z, an, xmax):
        # fail fast on wide probe balls so the outer integrator bisects
        # cheaply instead of paying a doomed inner integration
        if z.real.rad() > 0.3 or z.imag.rad() > 0.3:
            return acb(arb(0, arb('inf')))
        gz = acb(GAMMA) * z
        Psigz = c_Psi(gz)
        if not (Psigz.real > 0):
            return acb(arb(0, arb('inf')))
        S = acb(0) if arb(s_tilt) == arb(0) else c_mills(gz) * s_norm

        def f(x, an2):
            g = g_pref * z - lam_c * x / root
            return c_logPsi(g - S, an2) * c_phi(gz + x)

        val = acb.integral(f, acb(0), acb(xmax), abs_tol=tol_in,
                           depth_limit=300, eval_limit=300000)
        return val / Psigz

    def inner(z, an, xmax):
        return inner_conditional(z, an, xmax) * c_phi(z)

    def inner_conditional_dz(z, an, xmax):
        """z derivative of the conditional moderate-x integral.

        If H(z)=J(z)/Psi(gamma*z), differentiation under the integral gives

          H' = int [-E(t)t_z + gamma(E(gamma*z)-(gamma*z+x))logPsi(t)]
                   phi(gamma*z+x) dx / Psi(gamma*z),

        with t_z=gamma*c-S'(z) and E'=E(E-u).  All arguments here remain in
        the moderate region selected by the negative-branch plan.
        """
        if z.real.rad() > 0.3 or z.imag.rad() > 0.3:
            return acb(arb(0, arb('inf')))
        gz = acb(GAMMA) * z
        Psigz = c_Psi(gz)
        if not (Psigz.real > 0):
            return acb(arb(0, arb('inf')))
        Egz = c_mills(gz)
        if arb(s_tilt) == arb(0):
            S = acb(0)
            Sz = acb(0)
        else:
            S = Egz * s_norm
            Sz = acb(GAMMA) * Egz * (Egz - gz) * s_norm
        tz = g_pref - Sz

        def df(x, an2):
            t = g_pref * z - lam_c * x / root - S
            Psit = c_Psi(t)
            if not (Psit.real > 0):
                return acb(arb(0, arb('inf')))
            Et = c_phi(t) / Psit
            Lt = c_logPsi(t, an2)
            if not Lt.is_finite():
                return acb(arb(0, arb('inf')))
            w = gz + x
            return (-Et * tz + acb(GAMMA) * (Egz - w) * Lt) * c_phi(w)

        val = acb.integral(df, acb(0), acb(xmax), abs_tol=tol_in,
                           depth_limit=300, eval_limit=300000)
        return val / Psigz

    res = acb(0)
    tail = arb(0)

    # The original path is fast and tight for lambda >= 0.  Preserve it
    # exactly, and also use it for a ball crossing zero (the negative-tail
    # split relies on beta=-lambda/sqrt(1-lambda^2) being certainly positive).
    if not (arb(lam) < 0):
        cuts = [arb(zlo), arb(-3), arb(0), arb(3), arb(zhi)]
        cuts = [c for c in cuts if (arb(zlo) <= c) and (c <= arb(zhi))]
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            res += acb.integral(
                lambda z, an: inner(z, an, xhi), acb(lo), acb(hi),
                abs_tol=tol_out, depth_limit=300, eval_limit=300000)
    else:
        # On the negative branch g-S increases with x.  Use narrow real z
        # slabs and half-unit x cutoffs.  The first cutoff where t>4 is
        # certified sends the remainder to the closed-form Mills bound; the
        # numerical piece therefore stays uniformly away from Psi(t)=0.
        # Fine real tail cells are cheap (no adaptive quadrature) and prevent the
        # interval evaluation of mills(gamma*z) and the normalized Gaussian
        # moments from losing the z dependency.  At the proof precision the
        # complete negative branch stays below the required 1e-3 radius.
        nz = 2048
        nx = 36
        tail_trigger = arb(4)
        rzlo, rzhi = arb(zlo), arb(zhi)
        rxhi = arb(xhi)
        dz = (rzhi - rzlo) / nz
        c_real = ((1 - arb(lam)) / (1 + arb(lam))).sqrt()
        beta = -arb(lam) / (1 - arb(lam) * arb(lam)).sqrt()
        s_real = (arb(s_tilt)
                  / (PSI.sqrt() * (1 - Q).sqrt()))

        plan = []
        for j in range(nz):
            zl = rzlo + dz * j
            zh = rzlo + dz * (j + 1)
            zcell = zl.union(zh)
            gz = GAMMA * zcell
            if arb(s_tilt) == arb(0):
                # Avoid the indeterminate interval product inf*0 in far
                # positive cells; mathematically S is identically zero.
                base = GAMMA * c_real * zcell
            else:
                base = GAMMA * c_real * zcell - _mills_range(gz) * s_real

            kcut = nx
            tcut = None
            for k in range(nx + 1):
                xcut = rxhi * k / nx
                candidate = base + beta * xcut
                if candidate > tail_trigger:
                    kcut = k
                    tcut = candidate
                    break
            plan.append((zl, zh, kcut, tcut))

            if kcut < nx:
                xcut = rxhi * kcut / nx
                # Tail evaluation is cheap and its interval dependency is
                # the dominant radius.  Split it once more without doubling
                # the expensive moderate-body quadratures.
                for r in range(2):
                    sl = zl + (zh - zl) * r / 2
                    sh = zl + (zh - zl) * (r + 1) / 2
                    sz = sl.union(sh)
                    sgz = GAMMA * sz
                    if arb(s_tilt) == arb(0):
                        sbase = GAMMA * c_real * sz
                    else:
                        sbase = (GAMMA * c_real * sz
                                 - _mills_range(sgz) * s_real)
                    st = sbase + beta * xcut
                    assert st > tail_trigger
                    tail += _ibase_mills_tail_cell(
                        sl, sh, xcut, rxhi, beta, st)

        # The outer complex integrator also spends heavily near z=0 because
        # every inner enclosure is treated as an analytic-function error.
        # Instead use a Gaussian-weighted mean-value enclosure on real z
        # cells.  The derivative remainder makes its width second-order; the
        # only remaining acb integrations are moderate x pieces.
        real_body = arb(0)
        for lo, hi, kcut, _ in plan:
            if kcut == 0:
                continue
            xmax = rxhi * kcut / nx
            mid = (lo + hi) / 2
            zcell = lo.union(hi)
            v = inner_conditional(acb(mid), True, xmax)
            d = inner_conditional_dz(acb(zcell), True, xmax)
            assert v.imag.contains(arb(0)) and d.imag.contains(arb(0))

            mass = Psi(lo) - Psi(hi)
            corr_signed = phi(lo) - phi(hi) - mid * mass
            mass_lo = Psi(lo) - Psi(mid)
            mass_hi = Psi(mid) - Psi(hi)
            corr_abs = (mid * mass_lo - (phi(lo) - phi(mid))
                        + (phi(mid) - phi(hi)) - mid * mass_hi)
            dmid = arb(d.real.mid())
            drad = arb(d.real.rad())
            slack = drad * corr_abs
            real_body += (v.real * mass + dmid * corr_signed
                          + slack.union(-slack))

        res = acb(real_body)

    assert res.imag.contains(arb(0))
    return alpha * (res.real + tail)


def I_upper_near1(lam, zlo=-6, zhi=6, uhi=12,
                  inner_tol_bits=22, outer_tol_bits=18):
    """Upper bound for I_0(lam) on cells hugging lam = 1, via the rescaled
    variable x = sqrt(1-lam^2) u (Ding-Sun eq. e:II.rescaled):

      I(lam) = sqrt(1-lam^2) alpha int int log Psi(c gamma z - lam u)
               phi(gamma z + sqrt(1-lam^2) u)/Psi(gamma z) phi(z) du dz.

    The u-coefficients stay O(1) as lam -> 1, so a lam ball costs little.
    Negative integrand: restricting (z, u) to [zlo, zhi] x [0, uhi] is an
    upper bound. The sqrt(1-lam^2) prefactor is kept as a ball; since the
    double integral is negative, the prefactor's lower end pairs with it.
    """
    lam_c = acb(lam)
    c_lam = ((1 - lam_c) / (1 + lam_c)).sqrt()
    root = (1 - lam_c * lam_c).sqrt()
    g_pref = acb(GAMMA) * c_lam

    tol_in = arb(2) ** (-inner_tol_bits)
    tol_out = arb(2) ** (-outer_tol_bits)

    def inner(z, an):
        if z.real.rad() > 0.3 or z.imag.rad() > 0.3:
            return acb(arb(0, arb('inf')))
        gz = acb(GAMMA) * z
        Psigz = c_Psi(gz)
        if not (Psigz.real > 0):
            return acb(arb(0, arb('inf')))

        def f(u, an2):
            return (c_logPsi(g_pref * z - lam_c * u, an2)
                    * c_phi(gz + root * u))

        val = acb.integral(f, acb(0), acb(uhi), abs_tol=tol_in,
                           depth_limit=300, eval_limit=300000)
        return val * c_phi(z) / Psigz

    res = acb(0)
    for lo, hi in [(zlo, -3), (-3, 0), (0, 3), (3, zhi)]:
        res += acb.integral(inner, acb(lo), acb(hi), abs_tol=tol_out,
                            depth_limit=300, eval_limit=300000)
    assert res.imag.contains(arb(0))
    # res.real <= 0; multiply by the root and alpha balls
    return ALPHA * (1 - lam * lam).sqrt() * res.real


def I0_lower(n=2000):
    """Lower bound for I(0) = alpha E log Psi(gamma Z).

    Even part: E log Psi(gamma Z) = E log Psi(-gamma Z). Integrate on
    [-9, 9] rigorously and bound the tail from below as in block1.
    """
    g = GAMMA

    def w(z, an):
        return c_logPsi(acb(g) * z, an)

    main = integrate(lambda z, an: w(z, an) * c_phi(z), -ZCUT, ZCUT)
    from core import z1_tail, z2_tail
    u_at = g * ZCUT
    tail_lo = -(g * g / 2 * z2_tail(ZCUT)
                + (2 * arb.pi()).log() / 2 * gauss_tail_mass(ZCUT)
                + g * z1_tail(ZCUT)
                + 2 * Psi(u_at) * gauss_tail_mass(ZCUT))
    return ALPHA * (main + tail_lo.union(arb(0)))


# ---------------------------------------------------------------------------
# P(lam) and Q(lam) upper bounds on a lambda cell (part (a) machinery).
#   P(lam) = -psi(1-q) lam/(1+lam) + I(lam) - I(0)
#   Q(lam) = 0.02 - 0.2 sqrt(psi(1-q)) c_lam - psi(1-q) lam/(1+lam)
#            + I_{0.2}(lam) - I(0)
# ---------------------------------------------------------------------------

def P_upper(lam, I0_lb):
    lin = -PSI * (1 - Q) * (1 - 1/(1 + lam))
    return lin + I_upper_box(lam, arb(0)) - I0_lb


def Q_upper(lam, I0_lb):
    s = dec('0.2')
    c_lam = ((1 - lam) / (1 + lam)).sqrt()
    lin = (s * s / 2 - s * (PSI * (1 - Q)).sqrt() * c_lam
           - PSI * (1 - Q) * (1 - 1/(1 + lam)))
    return lin + I_upper_box(lam, s) - I0_lb


def A_of_tau(tau):
    return (2 * tau.atanh()).exp()


def lam_cell(tau_lo, tau_hi, n=N_ELL):
    """Ball containing lambda(tau) = ell(A(tau)) for tau in the cell."""
    lo = ell_range(A_of_tau(tau_lo), n)
    hi = ell_range(A_of_tau(tau_hi), n)
    return lo.union(hi)


# ---------------------------------------------------------------------------
# Derivatives for the mean-value cell evaluation.
#
#   ell'(A)  = E[(D_H)'(A)]/(1-q),  (D_H)'(A) = 2A(1-m^2)^2/(Delta(Delta+1)^2)
#   frH'(A)  = -(1-q) log(A)/2 * ell'(A)          (their eq. e:dHH.dlambda)
#   I_s'(lam) = -alpha E E[E(g - S) dg/dlam],  dg/dlam = -(gamma z(1-lam)+x)
#                                                        /(1-lam^2)^{3/2}
# ---------------------------------------------------------------------------

def ell_prime(A, n=300):
    """Enclosure of ell'(A) >= 0 by hull Riemann (coarse is fine: it only
    multiplies the A-cell radius).

    Tail z >= GAM_ZCUT: (D_H)'(A) = 2A(1-m^2)^2/(Delta(Delta+1)^2) with
    Delta^2 = A^2(1-m^2) + m^2 >= min(A,1)^2, and
    (1-m^2)^2 <= 16 e^{-4 sqrt(psi) z}, so the tail mass is at most
    2A * 16 e^{-4 sqrt(psi_lb) zcut} P(|Z| >= zcut) / (min(A,1)(1-q)).
    """
    s = PSI.sqrt()

    from core import sq_any, sq_nonneg, pos_part, min_one, endpoints
    A2 = sq_nonneg(A)

    def w(z):
        m = (s * z).tanh()
        m2 = sq_any(m)
        delta2 = pos_part(A2 * (1 - m2) + m2)
        Delta = sqrt_nonneg(delta2)
        if not (Delta > 0):
            minA1 = min_one(A)
            lo, _ = endpoints(minA1)
            if not (lo > 0):
                lo = arb('1e-6')  # A > 1e-6 on every cell we use
            return arb(0).union(2 * A / lo)   # |(D_H)'| <= 2A/min(A,1)
        return 2 * A * sq_any(1 - m2) / (Delta * sq_nonneg(Delta + 1))

    body = 2 * riemann_gauss(w, 0, GAM_ZCUT, n) / (1 - Q)
    minA1 = min_one(A)
    lo, _ = endpoints(minA1)
    if not (lo > 0):
        lo = arb('1e-6')          # A > 1e-6 on every cell we use
    decay = (-4 * PSI_LB.sqrt() * GAM_ZCUT).exp()
    tail = 32 * A * decay * gauss_tail_mass(GAM_ZCUT) / (lo * (1 - Q_UB))
    return body + tail.union(-tail)


def frH_prime(A, n=300):
    return -(1 - Q) * A.log() / 2 * ell_prime(A, n)


def I_prime_box(lam, s_tilt, zlo=-6, zhi=6, xhi=None,
                inner_tol_bits=18, outer_tol_bits=15,
                depth=300, evals=300000):
    """Enclosure of d/dlam of the RESTRICTED I_s integral over the box,
    plus a rigorous bound for the derivative mass outside the box, so the
    result encloses I_s'(lam) for the FULL integral.

    Outside-the-box bound (crude, documented in notes): with
    b(lam) = 1/(1-lam^2)^{3/2},
      |integrand| <= E(g) (gamma|z| + x) b d(z,x)
    and (Lemma l:EE) E(g) <= 1 + |g| <= 1 + gamma|z| + x b'... we fold
    everything into C(z) phi-type moments:
      z >= 6:  d <= sqrt(2pi)(1+gamma z) phi(z) phi(x)
      z <= -6: d <= 2 phi(z) phi(gamma z + x)
      x >= 9, |z| <= 6: same two bounds per sign of z.
    Numerical constants are evaluated as balls below.  On lambda < 0 the
    in-box body is enclosed by real Gaussian cells, and the t>4 wedge gets a
    symmetric closed-form absolute bound; this avoids inverse-Mills poles in
    wide complex probe balls while preserving a two-sided derivative ball.
    """
    if xhi is None:
        xhi = XCUT
    lam_c = acb(lam)
    c_lam = ((1 - lam_c) / (1 + lam_c)).sqrt()
    root = (1 - lam_c * lam_c).sqrt()
    g_pref = acb(GAMMA) * c_lam
    s_norm = acb(s_tilt) / (acb(PSI).sqrt() * (1 - acb(Q)).sqrt())
    b32 = (1 - lam * lam) ** arb(-1.5)

    tol_in = arb(2) ** (-inner_tol_bits)
    tol_out = arb(2) ** (-outer_tol_bits)

    def inner_conditional(z, an, xmax=None):
        if xmax is None:
            xmax = arb(xhi)
        if z.real.rad() > 0.3 or z.imag.rad() > 0.3:
            return acb(arb(0, arb('inf')))
        gz = acb(GAMMA) * z
        Psigz = c_Psi(gz)
        if not (Psigz.real > 0):
            return acb(arb(0, arb('inf')))
        S = acb(0) if arb(s_tilt) == arb(0) else c_mills(gz) * s_norm

        def f(x, an2):
            g = g_pref * z - lam_c * x / root
            arg = g - S
            PsiA = c_Psi(arg)
            if not (PsiA.real > 0):
                return acb(arb(0, arb('inf')))
            E = c_phi(arg) / PsiA
            dg = -(gz * (1 - lam_c) + x) / (1 - lam_c * lam_c) ** arb(1.5)
            return E * dg * c_phi(gz + x)

        val = acb.integral(f, acb(0), acb(xmax), abs_tol=tol_in,
                           depth_limit=300, eval_limit=300000)
        return val / Psigz

    def inner_conditional_dz(z, an, xmax=None):
        """z-derivative of I_prime's conditional moderate-x integral H_p(z)
        = J_p(z)/Psi(gamma z).  Derived by hand and finite-difference verified
        against the scipy instrument (max rel-diff 1.2e-9):
          H_p' * Psi(gz) = int E(arg)*{dg*[(E-arg)*tz + gamma(Egz-w)] + dg_z}
                              phi(w) dx,
        with tz = g_pref - Sz, dg_z = -gamma(1-lam) b32 (constant in x)."""
        if xmax is None:
            xmax = arb(xhi)
        if z.real.rad() > 0.3 or z.imag.rad() > 0.3:
            return acb(arb(0, arb('inf')))
        gz = acb(GAMMA) * z
        Psigz = c_Psi(gz)
        if not (Psigz.real > 0):
            return acb(arb(0, arb('inf')))
        Egz = c_mills(gz)
        if arb(s_tilt) == arb(0):
            S = acb(0)
            Sz = acb(0)
        else:
            S = c_mills(gz) * s_norm
            Sz = acb(GAMMA) * Egz * (Egz - gz) * s_norm
        tz = g_pref - Sz
        dgz = -acb(GAMMA) * (1 - lam_c) / (1 - lam_c * lam_c) ** arb(1.5)

        def df(x, an2):
            g = g_pref * z - lam_c * x / root
            arg = g - S
            PsiA = c_Psi(arg)
            if not (PsiA.real > 0):
                return acb(arb(0, arb('inf')))
            E = c_phi(arg) / PsiA
            w = gz + x
            dg = -(gz * (1 - lam_c) + x) / (1 - lam_c * lam_c) ** arb(1.5)
            bracket = (E - arg) * tz + acb(GAMMA) * (Egz - w)
            return E * (dg * bracket + dgz) * c_phi(w)

        val = acb.integral(df, acb(0), acb(xmax), abs_tol=tol_in,
                           depth_limit=300, eval_limit=300000)
        return val / Psigz

    def inner(z, an):
        return inner_conditional(z, an) * c_phi(z)

    if arb(lam) < 0:
        # Negative branch: enclose the z-integral by a Gaussian-weighted
        # mean-value expansion on real z cells (second-order remainder).  The
        # moderate x-integral is done by acb.integral at the z-midpoint
        # (inner_conditional) and the t>trigger wedge goes to a closed-form
        # bound.  Mirrors I_upper_box; keeps the derivative-integrand width
        # second-order instead of the old first-order double range grid.
        nz = 2048
        nx = 36
        trigger = arb(6)
        rzlo, rzhi = arb(zlo), arb(zhi)
        rxhi = arb(xhi)
        dz = (rzhi - rzlo) / nz
        real_res = arb(0)
        hard = arb(0)
        c_real = ((1 - arb(lam)) / (1 + arb(lam))).sqrt()
        beta = -arb(lam) / (1 - arb(lam) * arb(lam)).sqrt()
        s_real = arb(s_tilt) / (PSI.sqrt() * (1 - Q).sqrt())
        plan = []
        for j in range(nz):
            lo = rzlo + dz * j
            hi = rzlo + dz * (j + 1)
            zcell = lo.union(hi)
            gz = GAMMA * zcell
            if arb(s_tilt) == arb(0):
                base = GAMMA * c_real * zcell
            else:
                base = GAMMA * c_real * zcell - _mills_range(gz) * s_real
            kcut = nx
            for kk in range(nx + 1):
                xcut = rxhi * kk / nx
                if base + beta * xcut > trigger:
                    kcut = kk
                    break
            plan.append((lo, hi, kcut))
            if kcut < nx:
                xcut = rxhi * kcut / nx
                hard += _iprime_hard_tail_cell(lo, hi, xcut, rxhi, lam)
        for lo, hi, kcut in plan:
            if kcut == 0:
                continue
            xmax = rxhi * kcut / nx
            mid = (lo + hi) / 2
            v = inner_conditional(acb(mid), True, xmax)
            d = inner_conditional_dz(acb(lo.union(hi)), True, xmax)
            assert v.imag.contains(arb(0)) and d.imag.contains(arb(0))
            mass = Psi(lo) - Psi(hi)
            corr_signed = phi(lo) - phi(hi) - mid * mass
            mass_lo = Psi(lo) - Psi(mid)
            mass_hi = Psi(mid) - Psi(hi)
            corr_abs = (mid * mass_lo - (phi(lo) - phi(mid))
                        + (phi(mid) - phi(hi)) - mid * mass_hi)
            dmid = arb(d.real.mid())
            drad = arb(d.real.rad())
            slack = drad * corr_abs
            real_res += (v.real * mass + dmid * corr_signed
                         + slack.union(-slack))
        res = acb(real_res)
    else:
        res = acb(0)
        for lo, hi in [(zlo, -3), (-3, 0), (0, 3), (3, zhi)]:
            res += acb.integral(inner, acb(lo), acb(hi), abs_tol=tol_out,
                                depth_limit=300, eval_limit=300000)
    assert res.imag.contains(arb(0))
    body = -ALPHA * res.real

    # outside-the-box derivative mass, generously bounded
    tail = _iprime_tail(lam, b32, arb(abs(zlo)), arb(xhi))
    if arb(lam) < 0:
        tail += ALPHA_UB * hard
    return body + tail.union(-tail)


def _one_sided_moments(T):
    """M_k(T) = int_T^inf t^k phi(t) dt for k = 0..3, T >= 0, as arb."""
    T = arb(T)
    M0 = Psi(T)
    M1 = phi(T)
    M2 = T * phi(T) + Psi(T)
    M3 = (T * T + 2) * phi(T)
    return M0, M1, M2, M3


def _poly_gauss_tail(T, c0, c1, c2, c3=arb(0)):
    """int_T^inf (c0 + c1 t + c2 t^2 + c3 t^3) phi(t) dt, coefficients >= 0."""
    M0, M1, M2, M3 = _one_sided_moments(T)
    return c0 * M0 + c1 * M1 + c2 * M2 + c3 * M3


def _poly_abs_gauss_full(c0, c1, c2):
    """Integral over R of c0+c1*|t|+c2*t^2 against the Gaussian density."""
    return c0 + c1 * (2 / arb.pi()).sqrt() + c2


def _iprime_tail(lam, b32, L=arb(6), XC=arb(9)):
    """Bound for the mass of the I_s' integrand outside [-6, 6] x [0, 9].

    Integrand: E(g - S) |dg/dlam| d(z, x), where (with beta = |lam|/
    sqrt(1-lam^2), b = b32 = (1-lam^2)^{-3/2},
    cmax >= c_lam, and S >= 0):

      E(g - S) <= E(g) <= 1 + |g| <= 1 + gamma*cmax*|z| + beta x
      |dg/dlam| <= (2 gamma |z| + x) b
      d(z, x)  <= 2 phi(z) phi(gamma z + x)                  for z <= 0
      d(z, x)  <= [1/Psi(1)] phi(z) phi(gamma z + x)         for 0 <= gamma z <= 1
      d(z, x)  <= sqrt(2pi)(gamma z + 1) phi(z) phi(x)       for gamma z >= 1
                  (Psi(u) >= phi(u) u/(1+u^2) and phi(gz+x)/phi(gz)
                   = e^{-x^2/2 - gz x} <= sqrt(2pi) phi(x))

    Regions: R1 = {|z| >= 6}, R2 = {|z| <= 6, x >= 9}.
    Every piece reduces to one-sided Gaussian moments; each region below is
    a closed-form bound with every substitution enlarging the integrand.
    """
    # c_lambda=sqrt((1-lambda)/(1+lambda)) multiplies the z term in g.
    # It is >1 on the negative branch; use its rigorous upper endpoint in
    # every nonnegative polynomial coefficient below.
    cmax = _c_lambda_upper(lam)
    g = GAMMA_UB * cmax
    L = arb(L)
    XC = arb(XC)
    beta = abs(lam) / (1 - lam * lam).sqrt()
    SQ2PI = (2 * arb.pi()).sqrt()
    # half-line x-moments of phi: int_0^inf x^k phi dx
    x0, x1, x2, x3 = arb(1) / 2, 1 / SQ2PI, arb(1) / 2, 2 / SQ2PI

    # --- R1a: z >= 6 (so gamma z >= 6 gamma > 1). Weight
    # sqrt(2pi)(gz+1) phi(z) phi(x); polynomial (1 + g z + beta x)(2 g z + x).
    # x-integral first: int (2gz + x)(1 + gz + beta x) phi(x) dx over [0,inf):
    #   = (1+gz)(2gz x0 + x1) + beta(2gz x1 + x2)
    # then times (gz+1) and z-tail moments up to z^3.
    # Expand in powers of z with nonnegative coefficients (g, beta >= 0):
    # (gz+1)[(1+gz)(2gz x0 + x1) + beta(2gz x1 + x2)]
    #  = (gz+1)(1+gz) 2gz x0 + (gz+1)(1+gz) x1 + beta (gz+1)(2gz x1 + x2)
    # <= (coefficients collected on z^0..z^3 with g -> GAMMA_UB):
    c3 = 2 * g ** 3 * x0
    c2 = 4 * g * g * x0 + g * g * x1 + 2 * beta * g * g * x1
    c1 = 2 * g * x0 + 2 * g * x1 + beta * (2 * g * x1 + g * x2)
    c0 = x1 + beta * x2
    r1a = SQ2PI * _poly_gauss_tail(L, c0, c1, c2, c3)

    # --- R1b: z <= -6. Weight 2 phi(z) phi(gz + x). Substitute t = gz + x
    # (t ranges over [gz, inf) contained in R), x = t + g|z|:
    # (1 + g|z| + beta(t + g|z|))(2 g|z| + t + g|z|)
    #  <= (1 + (1+beta) g|z| + beta|t|)(3 g|z| + |t|), and int over t in R
    # picks absolute moments A0 = 1, A1 = sqrt(2/pi), A2 = 1:
    a0, a1, a2 = arb(1), (2 / arb.pi()).sqrt(), arb(1)
    # expand: 3g|z| a0 + a1 + (1+beta)g|z| (3g|z| a0 + a1) + beta(3g|z| a1 + a2)
    d2 = 3 * (1 + beta) * g * g * a0
    d1 = 3 * g * a0 + (1 + beta) * g * a1 + 3 * beta * g * a1
    d0 = a1 + beta * a2
    r1b = 2 * _poly_gauss_tail(L, d0, d1, d2)

    # --- R2 (|z| <= 6, x >= 9): polynomial coefficient bounded per z-cell.
    # z <= 0: 2 phi(z) phi(gz+x); on a z-cell [zl, zh] the threshold is
    # t = gz + x >= 9 + g zl and |z| <= |zl|, so with x = t + g|z| <= t + g|zl|:
    # integrand poly <= (1 + g|zl|(1+beta) + beta t)(3 g|zl| + t).
    # Sum over 12 cells of [-6, 0], each with its own threshold: this keeps
    # the bound honest for moderate z where phi(gz + x) is genuinely tiny.
    r2_zneg = arb(0)
    ncell = 12
    for k in range(ncell):
        zl = -L + L * k / ncell          # most negative point of the cell
        zh = -L + L * (k + 1) / ncell
        zl = arb(zl)
        zh = arb(zh)
        mass = 2 * (Psi(zl) - Psi(zh))   # int_cell 2 phi(z) dz
        zmax = abs(zl)
        T = XC + g * zl                  # gz + x >= XC - g|zl| on the cell
        e0 = 1 + g * zmax * (1 + beta)
        p0 = e0 * 3 * g * zmax
        p1 = e0 + 3 * g * zmax * beta
        p2 = beta
        if T > 0:
            cell_tail = _poly_gauss_tail(T, p0, p1, p2)
        else:
            # The polynomial is in |t|.  Clamping a nonpositive T to zero
            # would omit the interval [T,0].  The actual half-line is a
            # subset of R, so full absolute Gaussian moments are a rigorous
            # (and still cheap) fallback whenever positivity is uncertain.
            cell_tail = _poly_abs_gauss_full(p0, p1, p2)
        r2_zneg += mass * cell_tail

    # z >= 0 pieces: x <= t = gz + x, so the coefficient polynomial is
    # (1 + Lg + beta t)(2Lg + t) with t-coefficients:
    g0 = (1 + L * g) * 2 * L * g
    g1 = (1 + L * g) + 2 * L * g * beta
    g2 = beta

    # 0 <= gamma z <= 1: [1/Psi(1)] phi(z) phi(gz+x), gz + x >= 9:
    r2_zmid = (1 / Psi(arb(1))) * _poly_gauss_tail(XC, g0, g1, g2)

    # gamma z >= 1: sqrt(2pi)(gz+1)phi(z)phi(x) <= sqrt(2pi)(6g+1)phi(z)phi(x):
    r2_zpos = SQ2PI * (L * g + 1) * _poly_gauss_tail(XC, g0, g1, g2)

    total = b32 * (r1a + r1b + r2_zneg + r2_zmid + r2_zpos)
    return ALPHA_UB * total


def P_prime(lam, s_tilt=None):
    """Enclosure of P'(lam) = -psi(1-q)/(1+lam)^2 + I'(lam)."""
    s = s_tilt if s_tilt is not None else arb(0)
    return -PSI * (1 - Q) / (1 + lam) ** 2 + I_prime_box(lam, s)


def PG_cell(A_lo, A_hi):
    """Upper bound for PG(lm) = H(lm) + P(lm) over one grid cell given by
    the A interval. Mean-value in A for H always; mean-value in lam for P
    unless the cell sits near lam = 1, where (1-lam^2)^{-3/2} blows up the
    derivative bound and a plain interval evaluation is tighter."""
    A_ball = A_lo.union(A_hi)
    A_mid = (A_lo + A_hi) / 2
    lam_lo = ell_range(A_lo)
    lam_hi = ell_range(A_hi)
    lam_ball = lam_lo.union(lam_hi)
    lam_mid = ell_range(A_mid)

    # H term: frH_upper already encloses H over the whole A-cell (Gamma is
    # bounded and smooth), so no mean-value refinement in A is needed.
    H_val = frH_upper(A_ball, H_lb=_hlb())

    if lam_ball < dec('0.9'):
        # interior: mean-value in lambda to beat the ~1e-3 margins
        I_mid = I_upper_box(lam_mid, arb(0))
        P_mid = -PSI * (1 - Q) * (1 - 1/(1 + lam_mid)) + I_mid - _i0lb()
        P_val = P_mid + P_prime(lam_ball) * (lam_ball - lam_mid)
    else:
        # near lambda = 1: margin is O(0.1), a direct rescaled bound suffices
        I_ball = I_upper_near1(lam_ball)
        P_val = -PSI * (1 - Q) * (1 - 1/(1 + lam_ball)) + I_ball - _i0lb()
    return H_val + P_val


S_TILT = None


def QG_cell(A_lo, A_hi):
    """Upper bound for QG(lm) = H(lm) + Q(lm) over one grid cell; the Q
    functional carries the fixed tilt s = 0.2."""
    s = dec('0.2')
    A_ball = A_lo.union(A_hi)
    # The default n=500 enclosure has radius about 0.04 near A=0, which does
    # not shrink under tau subdivision and can erase the QG margin.  This is
    # the same rigorous real Riemann bound at finer resolution.
    ell_n = 10000
    lam_lo = ell_range(A_lo, n=ell_n)
    lam_hi = ell_range(A_hi, n=ell_n)
    lam_ball = lam_lo.union(lam_hi)

    # H term: direct enclosure over the A-cell (see PG_cell).
    # On the final tau cell A_lo=0 and 0<A_hi<1.  The certified identity
    # H'(A)=-(1-q) log(A) ell'(A)/2 is nonnegative there, so the maximum is
    # attained at A_hi.  Evaluating the endpoint avoids the unnecessary
    # dependency blow-up from enclosing the removable A=0 limit in one ball.
    ep = ell_prime(A_hi) if endpoints(A_lo)[1] == 0 and A_hi < 1 else None
    if ep is not None and endpoints(ep)[0] >= 0:
        H_val = frH_upper(A_hi, H_lb=_hlb())
    else:
        H_val = frH_upper(A_ball, H_lb=_hlb())

    def Qlin(l):
        c_l = ((1 - l) / (1 + l)).sqrt()
        return (s * s / 2 - s * (PSI * (1 - Q)).sqrt() * c_l
                - PSI * (1 - Q) * (1 - 1/(1 + l)))

    # Q term (lambda in [lmin, -0.125], bounded away from +-1): evaluate at
    # a point center and use one derivative enclosure for the whole lambda
    # ball.  ``ell_range`` balls are intentionally conservative; feeding one
    # directly into nested quadrature recreates the negative-lambda slowdown.
    lam0 = arb(lam_ball.mid())
    I0 = I_upper_box(lam0, s)
    Q0 = Qlin(lam0) + I0 - _i0lb()
    Iprime = I_prime_box(lam_ball, s)
    # c'(lam) = -1/((1-lam)^{1/2} (1+lam)^{3/2})
    cp = -1 / ((1 - lam_ball).sqrt() * (1 + lam_ball) ** arb(1.5))
    Qp = (-s * (PSI * (1 - Q)).sqrt() * cp
          - PSI * (1 - Q) / (1 + lam_ball) ** 2
          + Iprime)
    Q_val = Q0 + Qp * (lam_ball - lam0)
    return H_val + Q_val


_HLB = None
_I0LB = None


def _hlb():
    global _HLB
    if _HLB is None:
        _HLB = H_star_lower()
    return _HLB


def _i0lb():
    global _I0LB
    if _I0LB is None:
        _I0LB = I0_lower()
    return _I0LB




# ---------------------------------------------------------------------------
# Second lambda-derivative of I (tilt s = 0), for the near-zero block (3c).
# ---------------------------------------------------------------------------

def I_second_box(lam, zlo=-7.5, zhi=7.5, xhi=14,
                 inner_tol_bits=18, outer_tol_bits=15):
    """Enclosure of I''(lam) at tilt s = 0 for |lam| <= 0.25.

    I(lam) = -alpha E_(z,x)[ log Psi(g) ],  g = gamma c_lam z - lam x r,
    r = (1-lam^2)^{-1/2}; with S = 0 the integrand's second derivative is
        d2/dlam2 [log Psi(g)] = -( E'(g) dg^2 + E(g) ddg ),
        dg  = -(gamma z (1-lam) + x) b32,        b32 = (1-lam^2)^{-3/2},
        ddg =  gamma z b32 - 3 lam (gamma z (1-lam) + x) b52,
        b52 = (1-lam^2)^{-5/2},   E' = E (E - g).
    Body: rigorous 2-D integral over [zlo, zhi] x [0, xhi]; the outside mass
    is bounded by _isecond_tail (closed-form Gaussian moments)."""
    if not (abs(lam) < dec('0.25')):
        return arb(0, arb('inf'))
    lam_c = acb(lam)
    c_lam = ((1 - lam_c) / (1 + lam_c)).sqrt()
    root = (1 - lam_c * lam_c).sqrt()
    g_pref = acb(GAMMA) * c_lam
    b32 = (1 - lam_c * lam_c) ** arb(-1.5)
    b52 = (1 - lam_c * lam_c) ** arb(-2.5)

    tol_in = arb(2) ** (-inner_tol_bits)
    tol_out = arb(2) ** (-outer_tol_bits)

    def inner(z, an):
        if z.real.rad() > 0.3 or z.imag.rad() > 0.3:
            return acb(arb(0, arb('inf')))
        gz = acb(GAMMA) * z
        Psigz = c_Psi(gz)
        if not (Psigz.real > 0):
            return acb(arb(0, arb('inf')))

        def f(x, an2):
            g = g_pref * z - lam_c * x / root
            PsiA = c_Psi(g)
            if not (PsiA.real > 0):
                return acb(arb(0, arb('inf')))
            E = c_phi(g) / PsiA
            Ep = E * (E - g)
            dg = -(gz * (1 - lam_c) + x) * b32
            ddg = gz * b32 - 3 * lam_c * (gz * (1 - lam_c) + x) * b52
            return (Ep * dg * dg + E * ddg) * c_phi(gz + x)

        val = acb.integral(f, acb(0), acb(xhi), abs_tol=tol_in,
                           depth_limit=300, eval_limit=300000)
        return val * c_phi(z) / Psigz

    res = acb(0)
    for lo, hi in [(zlo, -3), (-3, 0), (0, 3), (3, zhi)]:
        res += acb.integral(inner, acb(lo), acb(hi), abs_tol=tol_out,
                            depth_limit=300, eval_limit=300000)
    assert res.imag.contains(arb(0))
    body = -ALPHA * res.real
    tail = _isecond_tail(lam, arb(abs(zlo)), arb(xhi))
    return body + tail.union(-tail)


def _isecond_tail(lam, L=arb(7.5), XC=arb(14)):
    """Mass of the I'' integrand outside [-L, L] x [0, XC], |lam| <= 0.1.

    |E'(g) dg^2 + E(g) ddg| <= dg^2 + (1 + |g|) |ddg|   (0 < E' < 1,
    E <= 1 + |g|), and with beta = |lam| r <= 0.101, b32 <= 1.016,
    b52 <= 1.026 on |lam| <= 0.1:
      |g|   <= gamma|z| + beta x
      |dg|  <= (2 gamma |z| + x) b32
      |ddg| <= gamma|z| b32 + 3|lam| (2 gamma|z| + x) b52
    so the integrand is bounded by the quadratic polynomial
      P(u, x) = dg^2 + (1 + u + beta x)(u b32 + 3|lam|(2u + x) b52),
      u = gamma |z|,
    whose expansion has nonnegative coefficients.  The measure bounds and
    regions are exactly those of _iprime_tail; every piece reduces to
    one-sided Gaussian moments.  ALPHA is applied by the caller's sign
    convention (the bound is symmetric)."""
    cmax = _c_lambda_upper(lam)
    g = GAMMA_UB * cmax
    al = abs(arb(lam))
    r2 = 1 - lam * lam
    beta = al / r2.sqrt()
    b32 = r2 ** arb(-1.5)
    b52 = r2 ** arb(-2.5)
    SQ2PI = (2 * arb.pi()).sqrt()
    x0, x1, x2, x3 = arb(1) / 2, 1 / SQ2PI, arb(1) / 2, 2 / SQ2PI

    # polynomial coefficients of P(u, x) = sum p[i][j] u^i x^j  (u = gamma|z|)
    b2 = b32 * b32
    p = {}
    # dg^2 <= (2u + x)^2 b2 = (4u^2 + 4ux + x^2) b2
    p[(2, 0)] = 4 * b2
    p[(1, 1)] = 4 * b2
    p[(0, 2)] = b2
    # (1 + u + beta x)(u b32 + 6 lam' u b52 + 3 lam' x b52),  lam' = |lam|
    A_ = b32 + 6 * al * b52          # u-coefficient
    B_ = 3 * al * b52                # x-coefficient
    p[(1, 0)] = p.get((1, 0), arb(0)) + A_
    p[(0, 1)] = p.get((0, 1), arb(0)) + B_
    p[(2, 0)] = p[(2, 0)] + A_
    p[(1, 1)] = p[(1, 1)] + B_ + beta * A_
    p[(0, 2)] = p[(0, 2)] + beta * B_

    def xmom(j):
        return (x0, x1, x2, x3)[j]

    # R1a: z >= 6, weight sqrt(2pi)(u + 1) phi(z) phi(x), u = g z.
    # sum_{i,j} p_ij u^i x^j (u+1) -> z-poly of degree i+1 with g-powers.
    c = [arb(0)] * 4
    for (i, j), pij in p.items():
        m = pij * xmom(j)
        c[i + 1] += m * g ** (i + 1)
        c[i] += m * g ** i
    r1a = SQ2PI * _poly_gauss_tail(L, c[0], c[1], c[2], c[3])

    # R1b: z <= -6, weight 2 phi(z) phi(gz + x); substitute t = gz + x >= gz,
    # x = t + u (u = g|z|), and integrate |t| over R with absolute moments
    # A0 = 1, A1 = sqrt(2/pi), A2 = 1, A3 = 2 sqrt(2/pi):
    a_mom = (arb(1), (2 / arb.pi()).sqrt(), arb(1), 2 * (2 / arb.pi()).sqrt())
    c = [arb(0)] * 4
    for (i, j), pij in p.items():
        # x^j = (t + u)^j = sum_k C(j,k) u^{j-k} t^k
        from math import comb
        for k in range(j + 1):
            m = pij * comb(j, k) * a_mom[k]
            # u^{i + j - k} with u = g|z|
            c[min(i + j - k, 3)] += m * g ** (i + j - k)
    r1b = 2 * _poly_gauss_tail(L, c[0], c[1], c[2], c[3])

    # R2a: 0 <= z <= 6, x >= 9.  Here gz + x >= x so phi(gz+x) <= phi(x) and
    # the measure is <= [1/Psi(1) + sqrt(2pi)(u+1)] phi(z) phi(x), u = gz.
    # z-moments over the half line (E|z|^k <= zk below), x one-sided from 9.
    from math import comb
    z0, z1_, z2_, z3_ = arb(1), (2 / arb.pi()).sqrt(), arb(1),         2 * (2 / arb.pi()).sqrt()
    zk = (z0, z1_, z2_, z3_)
    M9 = _one_sided_moments(XC)
    wPsi = 1 / Psi(arb(1))
    r2a = arb(0)
    for (i, j), pij in p.items():
        xm = M9[j]
        # u^i (wPsi + sqrt(2pi)(u + 1)) z-moments: u = g z
        zi = ((wPsi + SQ2PI) * g ** i * zk[min(i, 3)]
              + SQ2PI * g ** (i + 1) * zk[min(i + 1, 3)])
        r2a += pij * zi * xm
    # R2b: -6 <= z <= 0, x >= 9: measure <= 2 phi(z) phi(gz + x); substitute
    # t = gz + x >= 9 - 6g > 2, x = t + u (u = g|z|): one-sided t-moments
    # from T2, u-moments over the half line.
    T2 = XC - L * g
    MT = _one_sided_moments(T2)
    r2b = arb(0)
    for (i, j), pij in p.items():
        for k in range(j + 1):
            r2b += 2 * pij * comb(j, k) * g ** (i + j - k)                 * zk[min(i + j - k, 3)] * MT[k]
    return ALPHA_UB * (r1a + r1b + r2a + r2b)


def I_second_fullbound(lam):
    """Instant closed-form bound on |I''(lam)| for |lam| <= 0.25: the
    polynomial/measure bounds of _isecond_tail applied to the WHOLE domain
    (for z >= 0, phi(gz+x) <= phi(x) and 1/Psi(gz) <= 1/Psi(1) for gz <= 1,
    Psi(u) >= phi(u) u/(1+u^2) for gz >= 1, so the measure is at most
    [1/Psi(1) + sqrt(2pi)(u+1)] phi(z) phi(x); for z <= 0 the measure is at
    most 2 phi(z) phi(t), t = gz + x, and x = t + u).  Crude but rigorous."""
    from math import comb
    cmax = _c_lambda_upper(lam)
    g = GAMMA_UB * cmax
    al = abs(arb(lam))
    r2 = 1 - lam * lam
    beta = al / r2.sqrt()
    b32 = r2 ** arb(-1.5)
    b52 = r2 ** arb(-2.5)
    SQ2PI = (2 * arb.pi()).sqrt()
    x0, x1, x2, x3 = arb(1) / 2, 1 / SQ2PI, arb(1) / 2, 2 / SQ2PI
    b2 = b32 * b32
    p = {}
    p[(2, 0)] = 4 * b2
    p[(1, 1)] = 4 * b2
    p[(0, 2)] = b2
    A_ = b32 + 6 * al * b52
    B_ = 3 * al * b52
    p[(1, 0)] = p.get((1, 0), arb(0)) + A_
    p[(0, 1)] = p.get((0, 1), arb(0)) + B_
    p[(2, 0)] = p[(2, 0)] + A_
    p[(1, 1)] = p[(1, 1)] + B_ + beta * A_
    p[(0, 2)] = p[(0, 2)] + beta * B_
    # z >= 0 (half-line z-moments zk, full half-line x-moments xm)
    z0, z1_, z2_, z3_ = arb(1) / 2, 1 / SQ2PI, arb(1) / 2, 2 / SQ2PI
    zk = (z0, z1_, z2_, z3_)
    xm = (x0, x1, x2, x3)
    wPsi = 1 / Psi(arb(1))
    rpos = arb(0)
    for (i, j), pij in p.items():
        zi = (wPsi * g ** i * zk[min(i, 3)]
              + SQ2PI * (g ** (i + 1) * zk[min(i + 1, 3)]
                         + g ** i * zk[min(i, 3)]))
        rpos += pij * zi * xm[j]
    # z <= 0: t = gz + x over R (absolute moments), u = g|z| half-line
    a_mom = (arb(1), (2 / arb.pi()).sqrt(), arb(1), 2 * (2 / arb.pi()).sqrt())
    rneg = arb(0)
    for (i, j), pij in p.items():
        for k in range(j + 1):
            rneg += 2 * pij * comb(j, k) * g ** (i + j - k) \
                * zk[min(i + j - k, 3)] * a_mom[k]
    return ALPHA_UB * (rpos + rneg)


def K_lb_neg(lam_lo, lam_hi, tol_bits=13):
    """Ding-Sun Corollary 9.3 lower envelope K_lb(lam_lo, lam_hi) for
    I'(lam) at s = 0 (their K_{s,lb,i}, i = 1,2,3: the K_{s,ub,i} displays
    with lb and ub exchanged), valid for lam_min <= lam_lo <= lam_hi <= 1
    of the same sign.  Small fixed domains keep every Mills argument
    moderate, so evaluation is fast where the direct I' evaluator ground.
    The truncation constant -1.1e-7/(1-L_ub)^{1/2} is included."""
    from core import endpoints
    ll, lu = arb(lam_lo), arb(lam_hi)
    # parameter-rectangle corners (certified in block 1)
    a_lo, a_hi = endpoints(ALPHA)
    q_lo, q_hi = endpoints(Q)
    g_lo, g_hi = endpoints(GAMMA)
    alb, aub = arb(a_lo), arb(a_hi)
    glb, gub = arb(g_lo), arb(g_hi)
    # c(lam) = sqrt((1-lam)/(1+lam)) decreasing in lam; L = lam^2
    clb = ((1 - lu) / (1 + lu)).sqrt()     # c at lam_hi = c-lower
    cub = ((1 - ll) / (1 + ll)).sqrt()     # c at lam_lo = c-upper
    # on the negative branch |lam_lo| >= |lam_hi|
    Llb = (lu * lu).union(ll * ll)
    Llo, _ = endpoints(Llb)
    Lhi_ = max(float(ll * ll), float(lu * lu))
    Llb = arb(min(float(ll * ll), float(lu * lu)))
    Lub = arb(Lhi_)
    tol = arb(2) ** (-tol_bits)

    def _E(x):
        return c_phi(x) / c_Psi(x)

    # K_{s,lb,1}: exchange lb<->ub in K_{s,ub,1}.  The integrand is a
    # single exponential of summed log-terms: the phi/Psi ratios lose
    # badly as ball quotients near the top of the z range.
    LOG2PI = (2 * arb.pi()).log()

    def f1(z, an):
        if z.real.rad() > 0.3 or z.imag.rad() > 0.3:
            return acb(arb(0, arb('inf')))
        lPz = c_logPsi(acb(glb) * z, an)
        if not lPz.is_finite():
            return acb(arb(0, arb('inf')))

        def g1(u, an2):
            arg = acb(clb) * acb(glb) * z - acb(lu) * u
            lPs = c_logPsi(arg, an2)
            if not lPs.is_finite():
                return acb(arb(0, arb('inf')))
            w = acb(gub) * z + (1 - Llb).sqrt() * u
            expo = (-arg * arg / 2 - lPs - w * w / 2 - z * z / 2 - lPz
                    - acb(LOG2PI) * 3 / 2)
            return (acb(clb) * acb(glb) * z + u) * expo.exp()
        return acb.integral(g1, acb(0), acb(8), abs_tol=tol,
                            depth_limit=200, eval_limit=200000)
    K1 = acb(0)
    for zl, zh in ((0, 2), (2, 4), (4, '5.5'), ('5.5', '6.5')):
        K1 += acb.integral(f1, acb(zl), acb(zh), abs_tol=tol,
                           depth_limit=200, eval_limit=200000)
    K1 = K1.real * alb / (1 - Llb).sqrt()

    # K_{s,lb,2}
    def f2(z, an):
        if z.real.rad() > 0.3 or z.imag.rad() > 0.3:
            return acb(arb(0, arb('inf')))
        Pz = c_Psi(acb(glb) * z)
        if not (Pz.real > 0):
            return acb(arb(0, arb('inf')))

        def g2(u, an2):
            arg = acb(clb) * acb(glb) * (1 + acb(ll) * u) * z
            Ps = c_Psi(arg)
            if not (Ps.real > 0):
                return acb(arb(0, arb('inf')))
            return z * z * (1 - u) * (c_phi(arg) / Ps) \
                * c_phi(acb(glb) * z * (1 - (1 - acb(ll)) * u)) * c_phi(z)
        v = acb.integral(g2, acb(0), acb(1), abs_tol=tol,
                         depth_limit=200, eval_limit=200000)
        return v / Pz
    K2 = acb.integral(f2, acb('-3.3'), acb(0), abs_tol=tol,
                      depth_limit=200, eval_limit=200000).real
    K2 = K2 * (-aub) * cub * gub * gub / (1 + ll)

    # K_{s,lb,3}
    SQ2PI = (2 * arb.pi()).sqrt()

    def f3(z, an):
        if z.real.rad() > 0.3 or z.imag.rad() > 0.3:
            return acb(arb(0, arb('inf')))
        Pz = c_Psi(acb(gub) * z)
        if not (Pz.real > 0):
            return acb(arb(0, arb('inf')))

        def g3(u, an2):
            arg = acb(cub) * acb(gub) * (1 + acb(lu)) * z - acb(lu) * u
            Ps = c_Psi(arg)
            if not (Ps.real > 0):
                return acb(arb(0, arb('inf')))
            expo = (-(Lub * (acb(gub) * z) ** 2
                      + u * u * (1 - Llb)
                      + z * u * acb(glb) * acb(ll)
                      * (1 - Lub).sqrt()) / 2).exp()
            return u * (c_phi(arg) / Ps) * c_phi(z) * expo
        v = acb.integral(g3, acb(0), acb(9), abs_tol=tol,
                         depth_limit=200, eval_limit=200000)
        return v / Pz
    K3 = acb.integral(f3, acb('-3.3'), acb(0), abs_tol=tol,
                      depth_limit=200, eval_limit=200000).real
    K3 = K3 * alb / (SQ2PI * (1 - Llb).sqrt())

    trunc = rat(11, 10 ** 8) / (1 - Lub).sqrt()
    return K1 + K2 + K3 - trunc


# --- generic absolute-integral bound machinery (degree-agnostic) -----------

def _pmul(p, q):
    """Product of polynomials in (|z|, x) as {(i, j): coeff >= 0}."""
    r = {}
    for (i1, j1), c1 in p.items():
        for (i2, j2), c2 in q.items():
            k = (i1 + i2, j1 + j2)
            r[k] = r.get(k, arb(0)) + c1 * c2
    return r


def _padd(*ps):
    r = {}
    for p in ps:
        for k, c in p.items():
            r[k] = r.get(k, arb(0)) + c
    return r


def _pscale(p, s):
    return {k: c * s for k, c in p.items()}


def _abs_moments(n):
    """E|t|^k, t ~ N(0,1), k = 0..n."""
    SQ2PI = (2 * arb.pi()).sqrt()
    out = [arb(1), 2 / SQ2PI]
    for k in range(2, n + 1):
        out.append((k - 1) * out[k - 2])
    return out


def _half_moments(n):
    """int_0^inf x^k phi(x) dx, k = 0..n."""
    SQ2PI = (2 * arb.pi()).sqrt()
    out = [arb(1) / 2, 1 / SQ2PI]
    for k in range(2, n + 1):
        out.append((k - 1) * out[k - 2])
    return out


def I_abs_bound(lam, poly):
    """alpha E[ P(|z|, x) ] under the I-integral measure over the WHOLE
    domain, for a polynomial P with nonnegative coefficients: measure
    d(z,x) = phi(z) phi(gamma z + x) / Psi(gamma z) dx dz, x >= 0.
      z >= 0: phi(gz+x) <= phi(x) and 1/Psi(gz) <= 1/Psi(1) + sqrt(2pi)(gz+1)
              (the two branches gz <= 1 / >= 1 of the Mills bound), so
              measure <= [1/Psi(1) + sqrt(2pi)(g z + 1)] phi(z) phi(x).
      z <= 0: 1/Psi(gz) <= 2; t = gz + x over R with x <= |t| + g|z|.
    All coefficients nonnegative, so term-by-term moment bounds apply."""
    cmax = _c_lambda_upper(lam)
    g = GAMMA_UB * cmax
    deg_z = max(i for (i, j) in poly) + 1
    deg_x = max(j for (i, j) in poly)
    zm = _half_moments(deg_z + 1)
    xm = _half_moments(deg_x)
    am = _abs_moments(deg_x)
    wPsi = 1 / Psi(arb(1))
    SQ2PI = (2 * arb.pi()).sqrt()
    total = arb(0)
    # z >= 0
    for (i, j), c in poly.items():
        base = zm[i] * wPsi + SQ2PI * (g * zm[i + 1] + zm[i])
        total += c * base * xm[j]
    # z <= 0: x^j <= (|t| + g|z|)^j, binomial with absolute t-moments
    from math import comb
    for (i, j), c in poly.items():
        s = arb(0)
        for k in range(j + 1):
            s += comb(j, k) * (g ** (j - k)) * zm[i + j - k] * am[k]
        total += 2 * c * s
    return ALPHA_UB * total


def I_third_fullbound(lam):
    """Closed-form bound on |I'''(lam)| over a lambda interval (ball):
    integrand third derivative = E''(g) dg^3 + 3 E'(g) dg ddg + E(g) dddg,
    with |E| <= 1 + |g|, |E'| <= 1, |E''| <= 3 + 4|g|, and the dg-chain
    factors expanded with nonnegative (|z|, x)-coefficients."""
    al = abs(arb(lam))
    ahi = arb(endpoints(abs(arb(lam)))[1])
    r2 = 1 - lam * lam
    b32 = (r2 ** arb(-1.5)).abs_upper() if hasattr(r2, 'abs_upper') else r2 ** arb(-1.5)
    b32 = arb(endpoints(r2 ** arb(-1.5))[1])
    b52 = arb(endpoints(r2 ** arb(-2.5))[1])
    b72 = arb(endpoints(r2 ** arb(-3.5))[1])
    g = GAMMA_UB
    beta = arb(endpoints(al / r2.sqrt())[1])
    onep = 1 + ahi
    # |g_arg| <= g_c gamma |z| + beta x <= (c <= sqrt((1+l)/(1-l)) worst) ...
    c_up = arb(endpoints(((1 + ahi) / (1 - ahi)).sqrt())[1])
    G = {(1, 0): c_up * g, (0, 1): beta}          # |g| bound
    DG = {(1, 0): g * onep * b32, (0, 1): b32}    # |dg|
    DDG = _padd({(1, 0): g * b32},
                _pscale({(1, 0): g * onep, (0, 1): arb(1)}, 3 * ahi * b52))
    DDDG = _padd(_pscale({(1, 0): g}, 6 * ahi * b52),
                 _pscale({(1, 0): g * onep, (0, 1): arb(1)},
                         3 * b52 + 15 * ahi * ahi * b72))
    ONE = {(0, 0): arb(1)}
    E2 = _padd(_pscale(ONE, arb(3)), _pscale(G, arb(4)))   # |E''|
    E0 = _padd(ONE, G)                                     # E
    DG3 = _pmul(DG, _pmul(DG, DG))
    term1 = _pmul(E2, DG3)
    term2 = _pscale(_pmul(DG, DDG), arb(3))
    term3 = _pmul(E0, DDDG)
    poly = _padd(term1, term2, term3)
    from core import endpoints as _ep
    return I_abs_bound(lam, poly)
