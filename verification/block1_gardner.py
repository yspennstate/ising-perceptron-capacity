"""Rigorous verification of Ding-Sun Proposition 1.3 (their p:gardner).

The proposition defines the Krauth-Mezard threshold: on the rectangle
alpha in [ALPHA_LB, ALPHA_UB], the recursion q -> P(R(q, alpha)) is a
contraction with a unique fixed point q_*(alpha) in (Q_LB, Q_UB),
psi_*(alpha) = R(q_*, alpha) in (PSI_LB, PSI_UB), and the Gardner free energy
G_*(alpha) crosses zero inside the alpha interval.

Ding-Sun proved all of this modulo finitely many integral inequalities that
they checked with nonrigorous numerics ("computer-assisted" lemmas 7.3-7.6).
This file re-proves every one of those inequalities in interval arithmetic.

Structure of each check: rigorous integral over |z| <= L plus an explicit
analytic tail bound, evaluated with all parameters as exact rationals or
balls covering their ranges.

  P(psi)      = E tanh^2(sqrt(psi) Z)
  R(q, alpha) = (alpha/(1-q)) E E(-gamma(q) Z)^2,  gamma(q) = sqrt(q/(1-q))
  G(alpha,q,psi) = -psi(1-q)/2 + E log(2 cosh(sqrt(psi) Z))
                   + alpha E log Psi(-gamma(q) Z)

Run:  python block1_gardner.py
"""

from flint import arb, acb
from core import (set_prec, rat, dec, iv, hull,
                  ALPHA, ALPHA_LB, ALPHA_UB,
                  Q_LB, Q_LU, Q_UL, Q_UB,
                  PSI_LB, PSI_LU, PSI_UL, PSI_UB,
                  gamma_of, phi, Psi, mills,
                  c_phi, c_Psi, c_logPsi, c_mills, c_log2cosh,
                  integrate, gauss_tail_mass, z1_tail, z2_tail, report)

set_prec(150)

L = arb(10)
RESULTS = []


def check(name, value, want):
    RESULTS.append(report(name, value, want))


# ---------------------------------------------------------------------------
# P(psi) with rigorous tail: tanh^2 <= 1.
# ---------------------------------------------------------------------------

def P_of(psi):
    s = psi.sqrt()
    main = integrate(lambda z, an: (s * z).tanh() ** 2 * c_phi(z), -L, L)
    tail = gauss_tail_mass(L)
    return hull(main, main + tail)


# ---------------------------------------------------------------------------
# R(q, alpha) with rigorous tail: E(-gamma z)^2 <= (1 + gamma|z|)^2.
# ---------------------------------------------------------------------------

def R_of(q, alpha):
    g = gamma_of(q)
    main = integrate(lambda z, an: c_mills(-g * z) ** 2 * c_phi(z), -L, L)
    tail = (1 + g * g) * gauss_tail_mass(L) + 2 * g * z1_tail(L) \
        + g * g * z2_tail(L)
    body = hull(main, main + tail)
    return alpha * body / (1 - q)


# ---------------------------------------------------------------------------
# Gardner free energy at a point of the rectangle.
# E log 2cosh tail: |x| <= log 2cosh(x) <= |x| + log 2.
# E log Psi(-gamma Z) tail: for u = gamma z >= gamma L >= 11,
#   log Psi(u)  in (-u^2/2 - log(2pi)/2 + log(u/(1+u^2)), 0)
#   log Psi(-u) in (-2 Psi(u), 0)      [since log(1-t) >= -2t for t <= 1/2]
# and |log(u/(1+u^2))| <= log(2u) <= u for u >= 11; both branches are
# enclosed by  -(u^2/2 + log(2pi)/2 + u + 2 Psi(u)) <= integrand pair <= 0.
# ---------------------------------------------------------------------------

def Elog2cosh(psi):
    s = psi.sqrt()
    main = integrate(lambda z, an: c_log2cosh(s * z) * c_phi(z), -L, L)
    lo = s * z1_tail(L)
    hi = lo + arb(2).log() * gauss_tail_mass(L)
    return main + hull(lo, hi)


def ElogPsi_gamma(q):
    g = gamma_of(q)
    main = integrate(lambda z, an: c_logPsi(-g * z, an) * c_phi(z), -L, L)
    u_at_L = g * L
    tail_lo = -(g * g / 2 * z2_tail(L)
                + (2 * arb.pi()).log() / 2 * gauss_tail_mass(L)
                + g * z1_tail(L)
                + 2 * Psi(u_at_L) * gauss_tail_mass(L))
    return main + hull(tail_lo, arb(0))


def G_of(alpha, q, psi):
    return -psi * (1 - q) / 2 + Elog2cosh(psi) + alpha * ElogPsi_gamma(q)


# ---------------------------------------------------------------------------
# Lemma 7.3 (l:into): R maps the q-intervals into the psi-intervals.
# R is increasing in q and alpha (their Lemma 7.2, analytic), so checking
# the four corners suffices for the containment statements.
# ---------------------------------------------------------------------------

def lemma_into():
    check("R(q_lb, a_lb) - PSI_LB", R_of(Q_LB, ALPHA_LB) - PSI_LB, '>0')
    check("PSI_LU - R(q_lu, a_lb)", PSI_LU - R_of(Q_LU, ALPHA_LB), '>0')
    check("R(q_ul, a_ub) - PSI_UL", R_of(Q_UL, ALPHA_UB) - PSI_UL, '>0')
    check("PSI_UB - R(q_ub, a_ub)", PSI_UB - R_of(Q_UB, ALPHA_UB), '>0')


# ---------------------------------------------------------------------------
# Lemma 7.4 (l:at): contraction. P'(psi) <= 0.08 on the psi range and
# dR/dq <= 12 on the rectangle, hence sup d/dq P(R(q,alpha)) <= 0.96 < 1.
#
# P'(psi) = E[(2 - cosh(2 sqrt(psi) Z))/cosh^4(sqrt(psi) Z)]; both pieces are
# decreasing in psi (their proof, analytic), so
#   P'(psi) <= E[2/cosh^4(sqrt(PSI_LB) Z)] - E[cosh(2 sqrt(PSI_UB) Z)/cosh^4(sqrt(PSI_UB) Z)].
# First integral: integrand <= 2, tail <= 2 P(|Z|>=L); positive integrand so
# lower tail 0. Second: positive integrand, drop tail for a lower bound.
#
# dR/dq = R(q,alpha)/(1-q) + (alpha/(1-q)^2) E[E(xi) E'(xi) zeta],
# xi = -gamma(q) z, zeta = -z/sqrt(q(1-q)).  On z >= 0 the integrand is <= 0
# (E, E' > 0, zeta <= 0), so the positive part comes from z <= 0 only:
#   E[E E' zeta] <= int_{-L}^{0} E(xi) E'(xi) zeta phi dz + tail(z <= -L),
# where on z <= -L: E(xi) E'(xi) zeta <= (1 + gamma|z|) * 1 * |z|/sqrt(q(1-q)).
# All evaluated with q, alpha as full-interval balls.
# ---------------------------------------------------------------------------

def lemma_at():
    s_lb = PSI_LB.sqrt()
    s_ub = PSI_UB.sqrt()
    term1 = integrate(lambda z, an: 2 / (s_lb * z).cosh() ** 4 * c_phi(z),
                      -L, L) + 2 * gauss_tail_mass(L)
    term2 = integrate(lambda z, an: (2 * s_ub * z).cosh()
                      / (s_ub * z).cosh() ** 4 * c_phi(z), -L, L)
    p_deriv = term1 - term2
    check("P'(psi) upper bound vs 0.08", dec('0.08') - p_deriv, '>0')

    q = iv(Q_LB, Q_UB)
    alpha = ALPHA
    g = gamma_of(q)
    sq = (q * (1 - q)).sqrt()

    def integrand(z, an):
        xi = -g * z
        e = c_mills(xi)
        ep = e * (e - xi)
        zeta = -z / acb(sq)
        return e * ep * zeta * c_phi(z)

    neg_part = integrate(integrand, -L, arb(0))
    # tail: int_{z<=-L} (1+gamma|z|) |z|/sqrt(q(1-q)) phi
    #     = (1/sqrt(q(1-q))) (phi(L) + gamma (L phi(L) + Psi(L)))
    tail = (phi(L) + g * (L * phi(L) + Psi(L))) / sq
    cross = neg_part + hull(arb(0), tail)
    r_ub = R_of(Q_UB, ALPHA_UB)
    dRdq = r_ub / (1 - q) + alpha * cross / (1 - q) ** 2
    check("dR/dq upper bound vs 12", 12 - dRdq, '>0')

    total = p_deriv * dRdq
    check("contraction constant vs 0.96", dec('0.96') - total, '>0')


# ---------------------------------------------------------------------------
# Corollary 7.5 (c:q.psi.bds): sign changes of P(R(q, alpha)) - q.
# ---------------------------------------------------------------------------

def cor_fixed_point():
    check("P(R(q_lb,a_lb)) - q_lb", P_of(R_of(Q_LB, ALPHA_LB)) - Q_LB, '>0')
    check("q_lu - P(R(q_lu,a_lb))", Q_LU - P_of(R_of(Q_LU, ALPHA_LB)), '>0')
    check("P(R(q_ul,a_ub)) - q_ul", P_of(R_of(Q_UL, ALPHA_UB)) - Q_UL, '>0')
    check("q_ub - P(R(q_ub,a_ub))", Q_UB - P_of(R_of(Q_UB, ALPHA_UB)), '>0')


# ---------------------------------------------------------------------------
# Corollary 7.6 (c:gg.bds): G_*(ALPHA_UB) < 0 < G_*(ALPHA_LB).
# At alpha = ALPHA_UB: q_*(alpha) in (Q_UL, Q_UB), psi_* in (PSI_UL, PSI_UB);
# monotonicity of each term of G (their stationarity argument):
#   -psi(1-q)/2   <= -PSI_UL (1 - Q_UB)/2
#   E log 2cosh   <= value at PSI_UB
#   E log Psi     <= value at Q_UL     (decreasing in q; negative integrand)
# and symmetrically for the lower bound at ALPHA_LB with q_* in (Q_LB, Q_LU).
# ---------------------------------------------------------------------------

def cor_G_signs():
    g_hi = (-PSI_UL * (1 - Q_UB) / 2 + Elog2cosh(PSI_UB)
            + ALPHA_UB * ElogPsi_gamma(Q_UL))
    check("G_*(a_ub) < 0 (upper bound)", g_hi, '<0')
    g_lo = (-PSI_LU * (1 - Q_LB) / 2 + Elog2cosh(PSI_LB)
            + ALPHA_LB * ElogPsi_gamma(Q_LU))
    check("G_*(a_lb) > 0 (lower bound)", g_lo, '>0')


if __name__ == "__main__":
    lemma_into()
    lemma_at()
    cor_fixed_point()
    cor_G_signs()
    n_pass = sum(RESULTS)
    print(f"\n{n_pass}/{len(RESULTS)} checks passed")
    if n_pass < len(RESULTS):
        raise SystemExit(1)
