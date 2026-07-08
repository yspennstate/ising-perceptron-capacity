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
from core import (rat, dec, hull, iv, phi, Psi, logPsi, mills,
                  c_phi, c_Psi, c_logPsi, c_mills,
                  integrate, gauss_tail_mass, z1_tail, z2_tail,
                  ALPHA, ALPHA_LB, ALPHA_UB, Q, PSI, GAMMA,
                  Q_LB, Q_UB, PSI_LB, PSI_UB, GAMMA_LB, GAMMA_UB,
                  gamma_of, ent2_tanh)

LOG4 = arb(4).log()


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
    Delta = delta2.sqrt()
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
    Delta = delta2.sqrt()
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


def I_upper_box(lam, s_tilt, alpha=None, zlo=-6, zhi=6, xhi=None,
                inner_tol_bits=22, outer_tol_bits=18):
    """Upper bound for I_{s}(lam): the (negative) double integral restricted
    to [zlo, zhi] x [0, xhi]. lam is an arb ball; s_tilt an arb >= 0.

    The integrand is negative, so any domain restriction gives an upper
    bound; [-6, 6] x [0, 9] keeps the discarded mass far below every grid
    margin while avoiding the erfc phase-oscillation zone near |z| = 9
    that stalls complex ball evaluation.

    Nested adaptive quadrature. The inner tolerance is deliberately loose
    (its error is carried rigorously in the returned balls); both tolerances
    are absolute and far below the grid margins.
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

    def inner(z, an):
        # fail fast on wide probe balls so the outer integrator bisects
        # cheaply instead of paying a doomed inner integration
        if z.real.rad() > 0.3 or z.imag.rad() > 0.3:
            return acb(arb(0, arb('inf')))
        gz = acb(GAMMA) * z
        Psigz = c_Psi(gz)
        if not (Psigz.real > 0):
            return acb(arb(0, arb('inf')))
        S = c_mills(gz) * s_norm

        def f(x, an2):
            g = g_pref * z - lam_c * x / root
            return c_logPsi(g - S, an2) * c_phi(gz + x)

        val = acb.integral(f, acb(0), acb(xhi), abs_tol=tol_in,
                           depth_limit=2000, eval_limit=1000000)
        return val * c_phi(z) / Psigz

    res = acb(0)
    cuts = [arb(zlo), arb(-3), arb(0), arb(3), arb(zhi)]
    cuts = [c for c in cuts if (arb(zlo) <= c) and (c <= arb(zhi))]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        res += acb.integral(inner, acb(lo), acb(hi), abs_tol=tol_out,
                            depth_limit=2000, eval_limit=1000000)
    assert res.imag.contains(arb(0))
    return alpha * res.real


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
                           depth_limit=2000, eval_limit=1000000)
        return val * c_phi(z) / Psigz

    res = acb(0)
    for lo, hi in [(zlo, -3), (-3, 0), (0, 3), (3, zhi)]:
        res += acb.integral(inner, acb(lo), acb(hi), abs_tol=tol_out,
                            depth_limit=2000, eval_limit=1000000)
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
        Delta = delta2.sqrt()
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
                inner_tol_bits=18, outer_tol_bits=15):
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
    Numerical constants are evaluated as balls below.
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

    def inner(z, an):
        if z.real.rad() > 0.3 or z.imag.rad() > 0.3:
            return acb(arb(0, arb('inf')))
        gz = acb(GAMMA) * z
        Psigz = c_Psi(gz)
        if not (Psigz.real > 0):
            return acb(arb(0, arb('inf')))
        S = c_mills(gz) * s_norm

        def f(x, an2):
            g = g_pref * z - lam_c * x / root
            arg = g - S
            PsiA = c_Psi(arg)
            if not (PsiA.real > 0):
                return acb(arb(0, arb('inf')))
            E = c_phi(arg) / PsiA
            dg = -(gz * (1 - lam_c) + x) / (1 - lam_c * lam_c) ** arb(1.5)
            return E * dg * c_phi(gz + x)

        val = acb.integral(f, acb(0), acb(xhi), abs_tol=tol_in,
                           depth_limit=2000, eval_limit=1000000)
        return val * c_phi(z) / Psigz

    res = acb(0)
    for lo, hi in [(zlo, -3), (-3, 0), (0, 3), (3, zhi)]:
        res += acb.integral(inner, acb(lo), acb(hi), abs_tol=tol_out,
                            depth_limit=2000, eval_limit=1000000)
    assert res.imag.contains(arb(0))
    body = -ALPHA * res.real

    # outside-the-box derivative mass, generously bounded
    tail = _iprime_tail(lam, b32, arb(abs(zlo)), arb(xhi))
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


def _iprime_tail(lam, b32, L=arb(6), XC=arb(9)):
    """Bound for the mass of the I_s' integrand outside [-6, 6] x [0, 9].

    Integrand: E(g - S) |dg/dlam| d(z, x), where (with beta = |lam|/
    sqrt(1-lam^2), b = b32 = (1-lam^2)^{-3/2}, c = c_lam <= 1, S >= 0):

      E(g - S) <= E(g) <= 1 + |g| <= 1 + gamma|z| + beta x
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
    g = GAMMA_UB
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
        if not (T > 0):
            T = arb(0)
        e0 = 1 + g * zmax * (1 + beta)
        p0 = e0 * 3 * g * zmax
        p1 = e0 + 3 * g * zmax * beta
        p2 = beta
        r2_zneg += mass * _poly_gauss_tail(T, p0, p1, p2)

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
    A_mid = (A_lo + A_hi) / 2
    lam_lo = ell_range(A_lo)
    lam_hi = ell_range(A_hi)
    lam_ball = lam_lo.union(lam_hi)
    lam_mid = ell_range(A_mid)

    # H term: direct enclosure over the A-cell (see PG_cell).
    H_val = frH_upper(A_ball, H_lb=_hlb())

    def Qlin(l):
        c_l = ((1 - l) / (1 + l)).sqrt()
        return (s * s / 2 - s * (PSI * (1 - Q)).sqrt() * c_l
                - PSI * (1 - Q) * (1 - 1/(1 + l)))

    # Q term (lambda in [lmin, -0.125], bounded away from +-1): mean-value in
    # lambda for the tight cells near -0.125.
    I_mid = I_upper_box(lam_mid, s)
    Q_mid = Qlin(lam_mid) + I_mid - _i0lb()
    # c'(lam) = -1/((1-lam)^{1/2} (1+lam)^{3/2})
    cp = -1 / ((1 - lam_ball).sqrt() * (1 + lam_ball) ** arb(1.5))
    Qp = (-s * (PSI * (1 - Q)).sqrt() * cp
          - PSI * (1 - Q) / (1 + lam_ball) ** 2
          + I_prime_box(lam_ball, s))
    Q_val = Q_mid + Qp * (lam_ball - lam_mid)
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
    """Enclosure of I''(lam) at tilt s = 0 for |lam| <= 0.1.

    I(lam) = -alpha E_(z,x)[ log Psi(g) ],  g = gamma c_lam z - lam x r,
    r = (1-lam^2)^{-1/2}; with S = 0 the integrand's second derivative is
        d2/dlam2 [log Psi(g)] = -( E'(g) dg^2 + E(g) ddg ),
        dg  = -(gamma z (1-lam) + x) b32,        b32 = (1-lam^2)^{-3/2},
        ddg =  gamma z b32 - 3 lam (gamma z (1-lam) + x) b52,
        b52 = (1-lam^2)^{-5/2},   E' = E (E - g).
    Body: rigorous 2-D integral over [zlo, zhi] x [0, xhi]; the outside mass
    is bounded by _isecond_tail (closed-form Gaussian moments)."""
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
                           depth_limit=2000, eval_limit=1000000)
        return val * c_phi(z) / Psigz

    res = acb(0)
    for lo, hi in [(zlo, -3), (-3, 0), (0, 3), (3, zhi)]:
        res += acb.integral(inner, acb(lo), acb(hi), abs_tol=tol_out,
                            depth_limit=2000, eval_limit=1000000)
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
    g = GAMMA_UB
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
