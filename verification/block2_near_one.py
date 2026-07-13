"""Rigorous verification of Ding-Sun Proposition p:HH.PP.near.lambda.one
in the CORRECTED form:  H(lm) + P(lm) < H(1) + P(1) for 0.982 <= lm < 1.

The constants printed in Ding-Sun's Lemma 8.2 are wrong: the two integrals
below evaluate to 2.678... and 3.162..., not <= 1.78 and <= 4.3 as stated
(their own verified fact 1 - ell(100) > 0.025 already contradicts the
claimed consequence 1 - ell(A) <= 1.83/A).  With the corrected constants
the same proof closes on iota = 1 - lm <= 0.018 instead of 0.02; the
remaining sliver lm in [0.98, 0.982] is covered by the part (a) value grid,
whose last cell reaches lam(0.99) = 0.98665... > 0.982 (pinned below).

Certified facts:
  (2a) the two integrals behind Lemma l:ell.A.large.A.improved:
       (2/(1-q_ub)) int sqrt(1-tanh^2(sqrt(psi_lb) z)) phi dz <= 2.679
       (2/(1-q_ub)) int |tanh(sqrt(psi_ub) z)| phi dz         <= 3.17
       (+ explicit |z| >= 9 tails), hence 1 - ell(A) <= 2.711/A for A >= 100,
       and the corollary constant (log 2.711)/2 + 1/2 <= 0.9987.
  (2b) iota(100) = 1 - ell(100) > 0.025
  (2c) the double-integral bound of Proposition p:sqrt at lambda_lb = 0.975
       (valid on [0.975, 1] and so on [0.982, 1]):
       alpha_lb sqrt(1+lambda_lb) (J_pos + J_neg) <= -0.4446,
       plus psi_ub (1-q_lb)/(2(2-0.025)) <= 0.285
  (2d) the closing inequality sqrt(i)(1.2837 + log(1/i)/2) - 0.4446 < 0 at
       i = 0.018, with the left side increasing in i (coefficient > 1), and
       the grid pin lam_lo(0.99) > 0.982.

Their remaining steps (monotonicity of the integrands in psi, the derivative
identity dH'(1-s)/ds = log(iota^{-1}(s))/2, the change of variables in
p:sqrt, and the envelope directions) are analytic; we re-derived each.
"""

from flint import arb, acb, ctx
from core import (set_prec, dec, iv, hull, phi, Psi, mills,
                  c_phi, c_Psi, c_logPsi, c_mills,
                  integrate, gauss_tail_mass, report,
                  ALPHA_LB, Q_LB, Q_UB, PSI_LB, PSI_UB, GAMMA_LB, GAMMA_UB)
import dsfun

set_prec(100)

RESULTS = []


def check(name, value, want):
    RESULTS.append(report(name, value, want))


def part_2a():
    s_lb = PSI_LB.sqrt()
    s_ub = PSI_UB.sqrt()

    def f1(z, an):
        t = (s_lb * z).tanh()
        v = 1 - t * t
        return v.sqrt(analytic=an)

    I1 = integrate(lambda z, an: f1(z, an) * c_phi(z), -arb(9), arb(9),
                   pieces=[0])
    C1 = 2 * (I1 + gauss_tail_mass(9)) / (1 - Q_UB)
    check("(2a) sqrt(1-m^2) integral vs 2.679", dec('2.679') - C1, '>0')

    # |tanh|: integrate 2 * tanh on [0, 9] to avoid the |.| kink at 0
    def f2(z, an):
        return (s_ub * z).tanh()

    I2 = 2 * integrate(lambda z, an: f2(z, an) * c_phi(z), arb(0), arb(9))
    C2 = 2 * (I2 + gauss_tail_mass(9)) / (1 - Q_UB)
    check("(2a) |tanh| integral vs 3.17", dec('3.17') - C2, '>0')

    # closing constants of the corollary (corrected)
    check("(2a) (log 2.711)/2 + 1/2 <= 0.9987",
          dec('0.9987') - (dec('2.711').log() / 2 + arb(1) / 2), '>0')
    check("(2a) 2.679/A + 3.17/A^2 <= 2.711/A at A=100",
          dec('2.711') - (dec('2.679') + dec('3.17') / 100), '>0')


def part_2b():
    ell100 = dsfun.ell_range(arb(100), n=2000)
    check("(2b) 1 - ell(100) - 0.025", 1 - ell100 - dec('0.025'), '>0')


def part_2c():
    lam_lb = dec('0.975')
    L_lb = lam_lb * lam_lb
    c_ub = ((1 - lam_lb) / (1 + lam_lb)).sqrt()
    root = (1 - L_lb).sqrt()
    tol = arb(2) ** (-60)
    check("(2c) negative-z envelope gamma_ub < psi_ub",
          PSI_UB - GAMMA_UB, '>0')

    def inner_pos(z, an):
        PsiglbZ = c_Psi(acb(GAMMA_LB) * z)
        if not (PsiglbZ.real > 0):
            return acb(arb(0, arb('inf')))

        def f(u, an2):
            return (c_logPsi(-u, an2)
                    * c_phi(acb(GAMMA_UB) * z + acb(root) * u))

        val = acb.integral(f, acb(0), acb(9), abs_tol=tol,
                           depth_limit=4000, eval_limit=4000000)
        return val * c_phi(z) / PsiglbZ

    J_pos = acb.integral(inner_pos, acb(0), acb(9), abs_tol=tol,
                         depth_limit=4000, eval_limit=4000000)

    def inner_neg(z, an):
        gz = acb(GAMMA_UB) * z
        Psigz = c_Psi(gz)
        if not (Psigz.real > 0):
            return acb(arb(0, arb('inf')))
        E = c_phi(gz) / Psigz

        def f(u, an2):
            return (c_logPsi(acb(PSI_UB) * acb(c_ub) * z - u, an2)
                    * (-(u * u) * (1 - acb(L_lb)) / 2).exp())

        val = acb.integral(f, acb(0), acb(9), abs_tol=tol,
                           depth_limit=4000, eval_limit=4000000)
        return val * E * c_phi(z)

    J_neg = acb.integral(inner_neg, acb(-9), acb(0), abs_tol=tol,
                         depth_limit=4000, eval_limit=4000000)

    J = ALPHA_LB * (1 + lam_lb).sqrt() * (J_pos.real + J_neg.real)
    check("(2c) p:sqrt double integral vs -0.4446", dec('-0.4446') - J, '>0')

    lin = PSI_UB * (1 - Q_LB) / (2 * (2 - dec('0.025')))
    check("(2c) linear term vs 0.285", dec('0.285') - lin, '>0')


def part_2d():
    # closing with the corrected constants: c = 0.9987 + 0.285 = 1.2837.
    i = dec('0.018')
    val = i.sqrt() * (dec('1.2837') + (1 / i).log() / 2) - dec('0.4446')
    check("(2d) closing inequality at iota=0.018", val, '<0')
    # increasing in iota since the coefficient exceeds 1
    check("(2d) coefficient 1.2837 > 1", dec('1.2837') - 1, '>0')
    # grid pin: the part (a) grid reaches past the new threshold 0.982
    lam99 = dsfun.ell_range(dsfun.A_of_tau(dec('0.99')))
    check("(2d) lam_lo(0.99) > 0.982", lam99 - dec('0.982'), '>0')


if __name__ == "__main__":
    part_2a()
    part_2b()
    part_2c()
    part_2d()
    n = sum(RESULTS)
    print(f"\n{n}/{len(RESULTS)} checks passed")
    if n < len(RESULTS):
        raise SystemExit(1)
