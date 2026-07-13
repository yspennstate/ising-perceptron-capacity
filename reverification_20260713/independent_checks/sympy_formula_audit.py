"""Symbolic audit of the algebraic identities transcribed into the verifiers.

Every identity below was previously checked by hand during the July 2026
audits.  This script re-derives each one with SymPy, so the formula-to-code
correspondence for the closed-form layer is machine-checked by an engine
that shares nothing with python-flint or the certificate code.

Sources of truth:
  Ding--Sun arXiv v1 (sources/dingsun_src/lbd.tex): displays (e:B.A),
  (e:D.A), (e:D.A.expand), (e:deriv.D.A), (e:D.H.intro), (e:P.H.D).
  huang_hessian.py header derivation and quadT_box coefficients.
  dsfun.py D_of / Gamma_HD.

Each check asserts a symbolic simplification to zero (or an exact
equality); any failure raises AssertionError with the offending residual.
"""

from __future__ import annotations

import sympy as sp


def _zero(expr, name, assumptions=None):
    simplified = sp.simplify(expr)
    if simplified != 0:
        simplified = sp.radsimp(sp.simplify(sp.together(simplified)))
    if simplified != 0:
        # last resort: prove zero by rewriting on the defining polynomial
        raise AssertionError(f"{name}: residual {simplified}")
    print(f"PASS {name}")


def ding_sun_D_identities():
    A, m = sp.symbols("A m", positive=True)
    # keep m in (0,1), A > 0; Delta per (e:D.A.expand) footnote
    Delta = sp.sqrt(A ** 2 + m ** 2 - (A * m) ** 2)

    # Delta definition equivalence: A^2+m^2-(Am)^2 == A^2(1-m^2)+m^2
    _zero((A ** 2 + m ** 2 - (A * m) ** 2) - (A ** 2 * (1 - m ** 2) + m ** 2),
          "Delta radicand two forms (lbd e:D.A.expand vs code)")

    # (e:B.A): A(1+m)/(Delta-m) == (Delta+m)/(A(1-m))
    lhs = A * (1 + m) / (Delta - m)
    rhs = (Delta + m) / (A * (1 - m))
    _zero(sp.together(lhs - rhs).rewrite(sp.Pow),
          "B_H(A) two closed forms (lbd e:B.A)")

    # (e:D.A): (X+1/X-2/A)/(X+1/X+2/A) - m^2 at X=B_H(A)
    X = A * (1 + m) / (Delta - m)
    D_def = (X + 1 / X - 2 / A) / (X + 1 / X + 2 / A) - m ** 2
    D_wc = (1 - m ** 2) * (Delta - 1) / (Delta + 1)      # well-conditioned
    _zero(D_def - D_wc, "D_H(A) dual-solution form equals (1-m^2)(Delta-1)/(Delta+1)")

    # (e:D.A.expand) chain and the dsfun.D_of code form
    D_code = (1 - m ** 2) * (1 - 2 / (Delta + 1))
    D_sq = (Delta - 1) ** 2 / (A ** 2 - 1)
    D_intro = (A ** 2 - 1) * (1 - m ** 2) ** 2 / (Delta + 1) ** 2
    _zero(D_code - D_wc, "dsfun.D_of code form equals (1-m^2)(Delta-1)/(Delta+1)")
    _zero(D_wc - D_sq, "(1-m^2)(Delta-1)/(Delta+1) equals (Delta-1)^2/(A^2-1)")
    _zero(D_wc - D_intro, "(e:D.A.expand) equals (e:D.H.intro) raw form")

    # (e:deriv.D.A): derivative of the code form
    dD = sp.diff(D_code, A)
    dD_ds = 2 * A * (1 - m ** 2) ** 2 / (Delta * (Delta + 1) ** 2)
    _zero(dD - dD_ds, "(D_H)'(A) matches lbd (e:deriv.D.A)")
    _zero(dD_ds - 2 * A * (Delta - 1) ** 2 / ((A ** 2 - 1) ** 2 * Delta),
          "(e:deriv.D.A) internal chain")


def pair_entropy_table():
    m, D = sp.symbols("m D", real=True)
    t_pp = ((1 + m) ** 2 + D) / 4
    t_mm = ((1 - m) ** 2 + D) / 4
    t_pm = (1 - m ** 2 - D) / 4
    _zero(t_pp + t_mm + 2 * t_pm - 1,
          "pair-entropy table probabilities sum to 1 (lbd e:P.H.D)")
    # marginals: row sums give the +/- spin marginals (1+-m)/2
    _zero(t_pp + t_pm - (1 + m) / 2, "pair-entropy + marginal")
    _zero(t_mm + t_pm - (1 - m) / 2, "pair-entropy - marginal")
    # first-moment identity: E[J K] = m^2 + D  (cf. e:D.A equivalent form)
    _zero(t_pp + t_mm - t_pm * 2 - (m ** 2 + D),
          "pair table correlation identity E[JK]=m^2+D")


def mills_ratio_derivatives():
    V = sp.Symbol("V", real=True)
    phi = sp.exp(-V ** 2 / 2) / sp.sqrt(2 * sp.pi)
    Psi = (1 - sp.erf(V / sp.sqrt(2))) / 2          # Gaussian upper tail
    E = phi / Psi
    _zero(sp.simplify(sp.diff(sp.log(Psi), V) + E),
          "d/dV log Psi = -E(V) (huang_hessian header)")
    _zero(sp.simplify(sp.diff(-E, V) + E * (E - V) - 0),
          "d/dV(-E) = ... consistency")
    # code: Ep = E*(E-V) used as -d^2/dV^2 log Psi
    _zero(sp.simplify(sp.diff(sp.log(Psi), V, 2) + E * (E - V)),
          "d2/dV2 log Psi = -E(E-V) (quadT_box Ep)")


def quadT_box_coefficients():
    a1, a2, q, psi, s, Ht, N = sp.symbols("a1 a2 q psi s Ht N", positive=True)
    D = sp.sqrt(1 - a2 ** 2 / q)
    c0 = -(a2 / q) / D
    c1 = s - (a1 / psi) / D
    V = c0 * Ht + c1 * N

    _zero(sp.diff(V, a1) - (-(1 / (psi * D)) * N),
          "dV/da1 = -N/(psi D)")

    V2_Ht = -1 / (q * D) - a2 ** 2 / (q ** 2 * D ** 3)
    V2_N = -a1 * a2 / (psi * q * D ** 3)
    _zero(sp.simplify(sp.diff(V, a2) - (V2_Ht * Ht + V2_N * N)),
          "dV/da2 coefficient split on (Ht, N)")

    _zero(sp.diff(V, a1, 2), "d2V/da1^2 = 0")
    _zero(sp.simplify(sp.diff(V, a1, a2) - (-(a2 / (psi * q * D ** 3)) * N)),
          "d2V/da1 da2 = -a2 N/(psi q D^3)")

    dInvD = a2 / (q * D ** 3)
    dInvD3 = 3 * a2 / (q * D ** 5)
    _zero(sp.simplify(sp.diff(1 / D, a2) - dInvD), "d(1/D)/da2")
    _zero(sp.simplify(sp.diff(1 / D ** 3, a2) - dInvD3), "d(1/D^3)/da2")

    dV2_Ht = -(1 / q) * dInvD - (1 / q ** 2) * (2 * a2 / D ** 3
                                                + a2 ** 2 * dInvD3)
    dV2_N = -(a1 / (psi * q)) * (1 / D ** 3 + a2 * dInvD3)
    _zero(sp.simplify(sp.diff(V, a2, 2) - (dV2_Ht * Ht + dV2_N * N)),
          "d2V/da2^2 coefficient split (dV2_Ht, dV2_N)")

    # D itself: dD/da2 = -a2/(q D) (comment line in quadT_box)
    _zero(sp.simplify(sp.diff(D, a2) - (-a2 / (q * D))), "dD/da2 = -a2/(qD)")


def ds_I_s_change_of_variables():
    # rationalize the domain: c = sqrt((1-lam)/(1+lam)) > 0 parameterizes
    # lam in (-1,1) via lam = (1-c^2)/(1+c^2), with sqrt(1-lam^2) = 2c/(1+c^2)
    z, x, gamma, c = sp.symbols("z x gamma c", positive=True)
    lam = (1 - c ** 2) / (1 + c ** 2)
    root = 2 * c / (1 + c ** 2)
    _zero(sp.simplify(root ** 2 - (1 - lam ** 2)),
          "c-parameterization: root^2 = 1 - lambda^2")
    _zero(sp.simplify(c ** 2 - (1 - lam) / (1 + lam)),
          "c-parameterization: c^2 = (1-lambda)/(1+lambda)")
    nu = x + gamma * z
    # dsfun.I_upper_box argument (nu = xi + x substitution):
    code_arg = gamma * c * z - lam * x / root
    ds_arg = (gamma * z - lam * nu) / root
    _zero(sp.simplify(sp.together(code_arg - ds_arg)),
          "I_s integrand argument: code form equals lbd (e:II.repeat) "
          "under nu = x + gamma z")

    # conditional-density normalization behind the /Psi(gamma z) factor
    u, w = sp.symbols("u w", real=True)
    phi = sp.exp(-u ** 2 / 2) / sp.sqrt(2 * sp.pi)
    tail = sp.integrate(phi, (u, w, sp.oo))
    Psi_w = (1 - sp.erf(w / sp.sqrt(2))) / 2
    _zero(sp.simplify((tail - Psi_w).rewrite(sp.erf)),
          "conditional nu-density normalizer equals Psi(gamma z)")


def ds_PP_pair_consistency():
    lam, psi, q, I_lam, I_0 = sp.symbols("lambda psi q I_lam I_0",
                                         positive=True)
    P_star = psi * (1 - q) / 2 + I_0     # the anchor identity
    form1 = (-P_star + psi * (1 - q) * (1 - lam) / (2 * (1 + lam)) + I_lam)
    form2 = -psi * (1 - q) * lam / (1 + lam) + I_lam - I_0
    _zero(sp.simplify(form1 - form2),
          "lbd (e:PP.pair.repeat) two printed forms agree iff "
          "P_star = psi(1-q)/2 + I(0)")


def moment_reduction_lagrange():
    # Lemma "moment": pointwise Lagrangian ent2((1+L)/2) + (l1 X + l2 M) L is
    # maximized at L = tanh(l1 X + l2 M); the stationarity condition is
    # d/dL ent2((1+L)/2) = -atanh(L).
    L = sp.Symbol("Lambda")
    p = (1 + L) / 2
    ent2 = -p * sp.log(p) - (1 - p) * sp.log(1 - p)
    _zero(sp.simplify(sp.diff(ent2, L) + sp.atanh(L).rewrite(sp.log)),
          "moment-reduction Lagrange stationarity: d/dL ent2((1+L)/2) = -atanh L")
    # and V_s display of sec:huang-moments equals the quadT_box coefficients
    a1, a2, q, psi, s, Z, N = sp.symbols("a1 a2 q psi s Z N", positive=True)
    D = sp.sqrt(1 - a2 ** 2 / q)
    paper_V = (-(a2 / q) * sp.sqrt(q) * Z - (a1 / psi) * N) / D + s * N
    code_V = (-(a2 / q) / D) * (sp.sqrt(q) * Z) + (s - (a1 / psi) / D) * N
    _zero(sp.simplify(paper_V - code_V),
          "V_s display (sec:huang-moments) equals quadT_box c0 Ht + c1 N")


def ds_P_prime_shape():
    lam, psi, q = sp.symbols("lambda psi q", positive=True)
    # the non-integral term of P: -psi(1-q)/(2)*(...)  -- the audited claim
    # is only the elementary derivative d/dlam [ psi(1-q)/(1+lam) ]
    _zero(sp.diff(psi * (1 - q) / (1 + lam), lam)
          + psi * (1 - q) / (1 + lam) ** 2,
          "d/dlam psi(1-q)/(1+lam) = -psi(1-q)/(1+lam)^2 (P' non-integral term)")


def main():
    ding_sun_D_identities()
    pair_entropy_table()
    mills_ratio_derivatives()
    quadT_box_coefficients()
    ds_I_s_change_of_variables()
    ds_PP_pair_consistency()
    moment_reduction_lagrange()
    ds_P_prime_shape()
    print("ALL SYMBOLIC IDENTITIES PASS")


if __name__ == "__main__":
    main()
