"""Region I of Huang's Condition 1.3: the degenerate maximizer.

Certify the frozen-tilt functional
    osS(l1, l2) = s0^2 psi/2 + ent(l1,l2) + alpha T(a1(l), a2(l), s0),
    s0 = sqrt(1 - q0),
has negative-definite Hessian on an explicit box around (1, 0). Since
(unconditionally, Huang Lemma "sS-zero") osS(1,0) = 0 and grad osS(1,0) = 0,
a negative-definite Hessian on the box gives osS(l) <= 0 there, and
S_*(l) <= osS(l) <= 0 with equality only at (1,0).

Everything is a fixed-grid integral (huanggrid primitives), so the Hessian
over a whole l-box is cheap and the robust monotone logPsi/E_mills keep it
finite. Formulas (chain rule; derivation in notes/huang_condition_plan.md):

  f1 = X, f2 = M;  d_i u = f_i,  d_i Lam = (1-Lam^2) f_i
  d_i ent = -E[u (1-Lam^2) f_i]
  d2_ij ent = -E[(1-Lam^2) f_i f_j] + 2 E[u Lam (1-Lam^2) f_i f_j]
  d_i a1 = E[X (1-Lam^2) f_i],  d_i a2 = E[M (1-Lam^2) f_i]
  d2_ij a1 = -2 E[X Lam (1-Lam^2) f_i f_j],  a2 with leading M
  V = c0 Ht + c1 N,  c0 = -(a2/q)/D,  c1 = s0 - (a1/psi)/D,  D=sqrt(1-a2^2/q)
  dV/da1 = -(1/(psi D)) N
  dV/da2 = [-1/(qD) - a2^2/(q^2 D^3)] Ht + [-a1 a2/(psi q D^3)] N
  P_i = dV/da1 . d_i a1 + dV/da2 . d_i a2   (coefficient split on Ht, N)
  Hess osS = d2 ent + alpha ( sum_i dT/da_i d2_ii-part ... )
We assemble via the moment chain rule directly:
  d2_ij [T(a(l))] = sum_k dT/da_k d2_ij a_k
                    + sum_{k,l} d2T/da_k da_l  d_i a_k d_j a_l,
with dT/da_k, d2T/da_k da_l the T-derivatives below (zt-grid integrals).
"""

from flint import arb
from core import (set_prec, dec, PSI, Q, GAMMA, ALPHA, endpoints)
import huanggrid as hg

SQ_PSI = hg.SQ_PSI
SQ_Q = hg.SQ_Q
S1Q = hg.S1Q
S0 = (1 - Q).sqrt()


# ---------------------------------------------------------------------------
# x-space: ent second derivatives and a1, a2 first/second derivatives, over
# an l-box. Uses the x-grid, mean-value in z, l as balls.
# ---------------------------------------------------------------------------

def _x_quant(l1, l2, gx=None):
    """Return dict of enclosures over the l-box:
        da1[i], da2[i]  (i=1,2);  d2ent[ij], d2a1[ij], d2a2[ij] (ij in 11,12,22).
    l1, l2 are arb balls (the box)."""
    if gx is None:
        gx = hg.get_x_grid()
    f = {1: (lambda X, M: X), 2: (lambda X, M: M)}
    out = {('da1', 1): arb(0), ('da1', 2): arb(0),
           ('da2', 1): arb(0), ('da2', 2): arb(0)}
    for ij in ((1, 1), (1, 2), (2, 2)):
        out[('d2ent', ij)] = arb(0)
        out[('d2a1', ij)] = arb(0)
        out[('d2a2', ij)] = arb(0)
    # 0th-order interval rule: accumulate value*m0 with the CELL (interval)
    # X, M so the enclosure is valid over the whole cell. (A previous version
    # used the midpoint values here with no remainder term -- not a valid
    # enclosure; caught in the soundness audit alongside the integrate() bug.)
    for (Xm, Mm, dMm, Xc, Mc, dMc, m0, m1c) in zip(
            gx.X_mid, gx.M_mid, gx.dM_mid, gx.X_cell, gx.M_cell, gx.dM_cell,
            gx.m0, gx.m1c):
        for (Xz, Mz, w) in ((Xc, Mc, m0),):
            u = l1 * Xz + l2 * Mz
            Lam = u.tanh()
            om = 1 - Lam * Lam        # 1 - Lam^2
            for i in (1, 2):
                fi = f[i](Xz, Mz)
                out[('da1', i)] = out[('da1', i)] + Xz * om * fi * w
                out[('da2', i)] = out[('da2', i)] + Mz * om * fi * w
            for ij in ((1, 1), (1, 2), (2, 2)):
                i, j = ij
                fifj = f[i](Xz, Mz) * f[j](Xz, Mz)
                out[('d2ent', ij)] = out[('d2ent', ij)] + \
                    (-om + 2 * u * Lam * om) * fifj * w
                out[('d2a1', ij)] = out[('d2a1', ij)] + \
                    (-2 * Xz * Lam * om) * fifj * w
                out[('d2a2', ij)] = out[('d2a2', ij)] + \
                    (-2 * Mz * Lam * om) * fifj * w
    return out


def _a1_a2(l1, l2, gx=None):
    """a1, a2 at (thin) l via the x-grid (z-mean-value)."""
    if gx is None:
        gx = hg.get_x_grid()
    a1m, a1c = [], []
    a2m, a2c = [], []
    for (Xm, Mm, dMm, Xc, Mc, dMc) in zip(
            gx.X_mid, gx.M_mid, gx.dM_mid, gx.X_cell, gx.M_cell, gx.dM_cell):
        um = l1 * Xm + l2 * Mm
        Lm = um.tanh()
        a1m.append(Xm * Lm)
        a2m.append(Mm * Lm)
        uc = l1 * Xc + l2 * Mc
        Lc = uc.tanh()
        dLc = (1 - Lc * Lc)
        # d/dz (X tanh u) = X' tanh u + X (1-Lam^2) u',  u' = l1 X' + l2 M'
        upc = l1 * SQ_PSI + l2 * dMc
        a1c.append(SQ_PSI * Lc + Xc * dLc * upc)
        a2c.append(dMc * Lc + Mc * dLc * upc)
    a1 = gx.integrate(a1m, a1c) + _a_tail()
    a2 = gx.integrate(a2m, a2c) + _a_tail()
    return a1, a2


def _a_tail():
    return arb(0).union(hg.gauss_tail_mass(arb(9)) * 0 + hg.z1_tail(arb(9)) * SQ_PSI) \
        .union(-(hg.z1_tail(arb(9)) * SQ_PSI))


# ---------------------------------------------------------------------------
# zt-space: T first and second derivatives wrt (a1, a2), over the a-box.
#   dT/da_i = E[-E(V) dV/da_i]
#   d2T/da_i da_j = E[ E'(V) (dV/da_i)(dV/da_j) - E(V) d2V/da_i da_j ]
#     ( d/da log Psi(V) = -E(V) V_a ;  d^2 = -E'(V)... wait sign:
#       d/dV log Psi = -E(V);  d^2/dV^2 log Psi = -E'(V).
#       d2/da_i da_j logPsi(V) = -E'(V) V_i V_j - E(V) V_ij. )
#   so d2T/da_i da_j = E[ -E'(V) V_i V_j - E(V) V_ij ].
# ---------------------------------------------------------------------------

def _T_derivs(a1, a2, gz=None, s=None):
    """dT/da1, dT/da2, d2T/da1^2, d2T/da1da2, d2T/da2^2 over the a-box, at
    tilt s (an arb ball; may be an interval covering the optimal-s range)."""
    if gz is None:
        gz = hg.get_zt_grid()
    if s is None:
        s = S0
    a2sq = a2 * a2 / Q
    _, hi = endpoints(a2sq)
    if not (hi < 1):
        return None
    D = (1 - a2sq).sqrt()
    D3 = D * D * D
    c0 = -(a2 / Q) / D
    c1 = s - (a1 / PSI) / D
    # V_i coefficients on (Ht, N):
    V1_N = -(1 / (PSI * D))                      # dV/da1 = V1_N * N
    V2_Ht = -1 / (Q * D) - a2 * a2 / (Q * Q * D3)
    V2_N = -a1 * a2 / (PSI * Q * D3)
    # V_ij coefficients (second derivs of V):
    # d2V/da1^2 = 0 (c1 linear in a1, D indep of a1) -> V11 = 0
    # d2V/da1 da2: d/da2 of (-(1/(psi D)) N) = (1/(psi D^2)) dD/da2 N,
    #   dD/da2 = -a2/(qD); so = -(a2/(psi q D^3)) N
    V12_N = -a2 / (PSI * Q * D3)
    # d2V/da2^2: differentiate V2_Ht, V2_N wrt a2 (tedious); assemble:
    # dD/da2 = -a2/(qD); d(1/D)/da2 = a2/(qD^3); d(1/D^3)/da2 = 3a2/(qD^5)
    dInvD = a2 / (Q * D3)
    dInvD3 = 3 * a2 / (Q * D3 * D * D)
    # V2_Ht = -(1/q)(1/D) - (a2^2/q^2)(1/D^3)
    dV2_Ht = -(1 / Q) * dInvD - (1 / (Q * Q)) * (2 * a2 / D3 + a2 * a2 * dInvD3)
    # V2_N = -(a1/(psi q)) a2 / D^3 = -(a1/(psi q)) a2 (1/D^3)
    dV2_N = -(a1 / (PSI * Q)) * (1 / D3 + a2 * dInvD3)
    SQq = SQ_Q
    res = {'dT1': arb(0), 'dT2': arb(0),
           'd2T11': arb(0), 'd2T12': arb(0), 'd2T22': arb(0)}
    keys = ('dT1', 'dT2', 'd2T11', 'd2T12', 'd2T22')

    def integrand_and_deriv(Ht, N, Np, want_deriv):
        """Return (5 kernel values) and, if want_deriv, (5 kernel z-derivs).
        Ht, N are z-values; Np = dN/dz; dHt/dz = sqrt(q)."""
        V = c0 * Ht + c1 * N
        E = hg.E_mills(V)
        Ep = E * (E - V)
        V1 = V1_N * N
        V2 = V2_Ht * Ht + V2_N * N
        V12 = V12_N * N
        V22 = dV2_Ht * Ht + dV2_N * N
        vals = (-E * V1,
                -E * V2,
                -Ep * V1 * V1,
                -Ep * V1 * V2 - E * V12,
                -Ep * V2 * V2 - E * V22)
        if not want_deriv:
            return vals, None
        Vp = c0 * SQq + c1 * Np           # dV/dz
        Epp = E * ((2 * E - V) * (E - V) - 1)   # E''(V)
        dE = Ep * Vp                      # d/dz E(V)
        dEp = Epp * Vp                    # d/dz E'(V)
        V1p = V1_N * Np
        V2p = V2_Ht * SQq + V2_N * Np
        V12p = V12_N * Np
        V22p = dV2_Ht * SQq + dV2_N * Np
        d = (-(dE * V1 + E * V1p),
             -(dE * V2 + E * V2p),
             -(dEp * V1 * V1 + Ep * 2 * V1 * V1p),
             -(dEp * V1 * V2 + Ep * (V1p * V2 + V1 * V2p) + dE * V12 + E * V12p),
             -(dEp * V2 * V2 + Ep * 2 * V2 * V2p + dE * V22 + E * V22p))
        return vals, d

    for (Nm, Npm, Htm, Nc, Npc, Htc, m0, m1p, m1m) in zip(
            gz.N_mid, gz.Np_mid, gz.Ht_mid, gz.N_cell, gz.Np_cell,
            gz.Ht_cell, gz.m0, gz.m1p, gz.m1m):
        vmid, _ = integrand_and_deriv(Htm, Nm, Npm, False)
        _, dcell = integrand_and_deriv(Htc, Nc, Npc, True)
        for k, gm, gd in zip(keys, vmid, dcell):
            res[k] = res[k] + gm * m0 + gd * m1p - gd * m1m
    # |z| >= L tails: E <= 1+|V|, E' in (0,1), N <= (1+gamma|z|)/sqrt(1-q);
    # every kernel is bounded by a product of two linear forms in |z|.
    lE = _lin_E(c0, c1)
    lV1 = _lin_of(arb(0), V1_N)
    lV2 = _lin_of(V2_Ht, V2_N)
    lV12 = _lin_of(arb(0), V12_N)
    lV22 = _lin_of(dV2_Ht, dV2_N)
    L = gz.L
    res['dT1'] = res['dT1'] + _pm(_ltail(_lmul(lE, lV1), L))
    res['dT2'] = res['dT2'] + _pm(_ltail(_lmul(lE, lV2), L))
    res['d2T11'] = res['d2T11'] + _pm(_ltail(_lmul(lV1, lV1), L))
    res['d2T12'] = res['d2T12'] + _pm(_ltail(_lmul(lV1, lV2), L)
                                      + _ltail(_lmul(lE, lV12), L))
    res['d2T22'] = res['d2T22'] + _pm(_ltail(_lmul(lV2, lV2), L)
                                      + _ltail(_lmul(lE, lV22), L))
    return res


def _absup(x):
    """Upper bound on |x| as an arb."""
    lo, hi = endpoints(x)
    m = abs(lo).union(abs(hi))
    _, u = endpoints(m)
    return u


def _lin_of(cHt, cN):
    """(p0, p1) with |cHt*Ht + cN*N| <= p0 + p1|z| on the tail:
    Ht = sqrt(q) z, N <= (1+gamma|z|)/sqrt(1-q)."""
    n1 = 1 / (1 - Q).sqrt()
    p0 = _absup(cN) * n1
    p1 = _absup(cHt) * Q.sqrt() + _absup(cN) * n1 * GAMMA
    return p0, p1


def _lin_N():
    n1 = 1 / (1 - Q).sqrt()
    return n1, n1 * GAMMA


def _lin_E(c0, c1):
    """E(V) <= 1 + |V| <= e0 + e1 |z|."""
    v0, v1 = _lin_of(c0, c1)
    return 1 + v0, v1


def _lmul(p, q):
    """Degree-2 coefficients of (p0+p1|z|)(q0+q1|z|)."""
    return (p[0] * q[0], p[0] * q[1] + p[1] * q[0], p[1] * q[1])


def _ltail(c, L):
    """int_{|z|>=L} (c0 + c1|z| + c2 z^2) phi(z) dz."""
    return (c[0] * hg.gauss_tail_mass(L) + c[1] * hg.z1_tail(L)
            + c[2] * hg.z2_tail(L))


def _pm(t):
    return t.union(-t)


def _Eprime(E, x):
    """E'(x) = E(x)(E(x)-x), enclosed. E' in (0,1)."""
    v = E * (E - x)
    lo, hi = endpoints(v)
    # clamp to (0,1] rigorously known bound
    lo2 = lo if lo > 0 else arb(0)
    hi2 = hi if hi < 1 else arb(1)
    return lo2.union(hi2)


# ---------------------------------------------------------------------------
# Hessian of osS over the l-box.
# ---------------------------------------------------------------------------

def hessian_box(l1, l2, gx=None, gz=None, s=None):
    """(H11, H12, H22) enclosures of nabla^2 [S(.,s)] over the l-box, at tilt
    s (arb ball or interval). With s an interval covering the optimal-tilt
    range over the box, negative-definiteness certifies nabla^2 S_* < 0 there
    by the envelope inequality nabla^2 S_* <= nabla^2 S(.,s*(.)). Returns None
    if a2^2 >= q on the box."""
    if gx is None:
        gx = hg.get_x_grid()
    if gz is None:
        gz = hg.get_zt_grid()
    xq = _x_quant(l1, l2, gx)
    a1, a2 = _a1_a2(l1, l2, gx)
    Td = _T_derivs(a1, a2, gz, s)
    if Td is None:
        return None
    da1 = {1: xq[('da1', 1)], 2: xq[('da1', 2)]}
    da2 = {1: xq[('da2', 1)], 2: xq[('da2', 2)]}
    H = {}
    for ij in ((1, 1), (1, 2), (2, 2)):
        i, j = ij
        d2a1 = xq[('d2a1', ij)]
        d2a2 = xq[('d2a2', ij)]
        # d2[T(a(l))]_ij = dT1 d2a1 + dT2 d2a2
        #   + d2T11 da1_i da1_j + d2T12 (da1_i da2_j + da2_i da1_j) + d2T22 da2_i da2_j
        d2T = (Td['dT1'] * d2a1 + Td['dT2'] * d2a2
               + Td['d2T11'] * da1[i] * da1[j]
               + Td['d2T12'] * (da1[i] * da2[j] + da2[i] * da1[j])
               + Td['d2T22'] * da2[i] * da2[j])
        H[ij] = xq[('d2ent', ij)] + ALPHA * d2T
    return H[(1, 1)], H[(1, 2)], H[(2, 2)]


def _s_derivs(a1, a2, s, da1, da2, gz=None):
    """At (thin) a, s: return
        Ss  = d/ds S       = psi s - alpha E[E(V) N]
        Sss = d^2/ds^2 S   = psi - alpha E[E'(V) N^2]
        Ssl = (d^2/ds dl1 S, d^2/ds dl2 S)
            = -alpha E[E'(V) N (V1 da1_i + V2 da2_i)]
    with V1 = dV/da1, V2 = dV/da2. z-mean-value rule.
    (Sign: d_s T = -E[E(V) N], so d^2_s T = -E[E'(V) N^2]; the earlier
    "+ alpha E[E' N^2]" here and in _a_s_mixed was a sign error, caught by a
    finite-difference cross-check: true Sss(a*, s0) = 0.4988, not 4.65.)"""
    if gz is None:
        gz = hg.get_zt_grid()
    a2sq = a2 * a2 / Q
    _, _hi = endpoints(a2sq)
    if not (_hi < 1):
        return None
    D = (1 - a2sq).sqrt()
    D3 = D * D * D
    c0 = -(a2 / Q) / D
    c1 = s - (a1 / PSI) / D
    V1_N = -(1 / (PSI * D))
    V2_Ht = -1 / (Q * D) - a2 * a2 / (Q * Q * D3)
    V2_N = -a1 * a2 / (PSI * Q * D3)

    def pieces(Ht, N, Np, want_d):
        V = c0 * Ht + c1 * N
        E = hg.E_mills(V)
        Ep = E * (E - V)
        V1 = V1_N * N
        V2 = V2_Ht * Ht + V2_N * N
        dl1V = V1 * da1[1] + V2 * da2[1]
        dl2V = V1 * da1[2] + V2 * da2[2]
        vals = (E * N, Ep * N * N, Ep * N * dl1V, Ep * N * dl2V)
        if not want_d:
            return vals
        Vp = c0 * SQ_Q + c1 * Np
        Epp = E * ((2 * E - V) * (E - V) - 1)
        dE = Ep * Vp
        dEp = Epp * Vp
        V1p = V1_N * Np
        V2p = V2_Ht * SQ_Q + V2_N * Np
        dl1Vp = V1p * da1[1] + V2p * da2[1]
        dl2Vp = V1p * da1[2] + V2p * da2[2]
        d = (dE * N + E * Np,
             dEp * N * N + Ep * 2 * N * Np,
             dEp * N * dl1V + Ep * (Np * dl1V + N * dl1Vp),
             dEp * N * dl2V + Ep * (Np * dl2V + N * dl2Vp))
        return vals, d

    I = [arb(0), arb(0), arb(0), arb(0)]
    for (Nm, Npm, Htm, Nc, Npc, Htc, m0, m1p, m1m) in zip(
            gz.N_mid, gz.Np_mid, gz.Ht_mid, gz.N_cell, gz.Np_cell,
            gz.Ht_cell, gz.m0, gz.m1p, gz.m1m):
        vm = pieces(Htm, Nm, Npm, False)
        _, dc = pieces(Htc, Nc, Npc, True)
        for k in range(4):
            I[k] = I[k] + vm[k] * m0 + dc[k] * m1p - dc[k] * m1m
    # |z| >= L tails (E <= 1+|V|, E' in (0,1); dl_iV = V1 da1_i + V2 da2_i)
    lE = _lin_E(c0, c1)
    lN = _lin_N()
    lV1 = _lin_of(arb(0), V1_N)
    lV2 = _lin_of(V2_Ht, V2_N)
    L = gz.L
    I[0] = I[0] + _pm(_ltail(_lmul(lE, lN), L))
    I[1] = I[1] + _pm(_ltail(_lmul(lN, lN), L))
    for k, i in ((2, 1), (3, 2)):
        d0 = (lV1[0] * _absup(da1[i]) + lV2[0] * _absup(da2[i]),
              lV1[1] * _absup(da1[i]) + lV2[1] * _absup(da2[i]))
        I[k] = I[k] + _pm(_ltail(_lmul(lN, d0), L))
    EEN, EpN2, EpNl1, EpNl2 = I
    Ss = PSI * s - ALPHA * EEN
    Sss = PSI - ALPHA * EpN2
    Ssl1 = -ALPHA * EpNl1
    Ssl2 = -ALPHA * EpNl2
    return Ss, Sss, Ssl1, Ssl2


def true_hessian(l1, l2, s, gx=None, gz=None):
    """Enclosure of nabla^2 S_*(l1,l2) at a (thin) point, via the envelope
    identity  nabla^2 S_* = nabla^2_l S(l,s*) - (d_s nabla_l S) (d_s nabla_l S)^T / d^2_s S,
    evaluated at s (which should enclose the optimal tilt s*(l), where d_s S=0).
    Returns (H11, H12, H22, Ss) with Ss = d_s S (should contain 0 at s=s*)."""
    if gx is None:
        gx = hg.get_x_grid()
    if gz is None:
        gz = hg.get_zt_grid()
    F = hessian_box(l1, l2, gx, gz, s=s)      # frozen Hessian nabla^2_l S(l,s)
    if F is None:
        return None
    F11, F12, F22 = F
    xq = _x_quant(l1, l2, gx)
    da1 = {1: xq[('da1', 1)], 2: xq[('da1', 2)]}
    da2 = {1: xq[('da2', 1)], 2: xq[('da2', 2)]}
    a1, a2 = _a1_a2(l1, l2, gx)
    Ss, Sss, Ssl1, Ssl2 = _s_derivs(a1, a2, s, da1, da2, gz)
    H11 = F11 - Ssl1 * Ssl1 / Sss
    H12 = F12 - Ssl1 * Ssl2 / Sss
    H22 = F22 - Ssl2 * Ssl2 / Sss
    return H11, H12, H22, Ss


def _Phi_hessian(l1, l2, gx=None):
    """nabla^2 Phi(l) = [[da1/dl1, da1/dl2],[da2/dl1, da2/dl2]] (the tilted
    covariance of (X,M)), over the l-box, via _x_quant. Symmetric PD."""
    if gx is None:
        gx = hg.get_x_grid()
    xq = _x_quant(l1, l2, gx)
    return (xq[('da1', 1)], xq[('da1', 2)], xq[('da2', 2)])  # P11, P12, P22


def _inv2(a, b, c):
    """Inverse of symmetric [[a,b],[b,c]] -> (i11,i12,i22)."""
    det = a * c - b * b
    return c / det, -b / det, a / det


def _a_s_mixed(a1, a2, s, gz=None):
    """d_a1 d_s T, d_a2 d_s T, and d^2_s S = psi - alpha E[E'(V)N^2], over the
    a-box at tilt s. d_ai d_s T = -E[E'(V) (dV/da_i) N]."""
    if gz is None:
        gz = hg.get_zt_grid()
    a2sq = a2 * a2 / Q
    _, _hi = endpoints(a2sq)
    if not (_hi < 1):
        return None
    D = (1 - a2sq).sqrt()
    D3 = D * D * D
    c0 = -(a2 / Q) / D
    c1 = s - (a1 / PSI) / D
    V1_N = -(1 / (PSI * D))
    V2_Ht = -1 / (Q * D) - a2 * a2 / (Q * Q * D3)
    V2_N = -a1 * a2 / (PSI * Q * D3)

    def pieces(Ht, N, Np, want):
        V = c0 * Ht + c1 * N
        E = hg.E_mills(V)
        Ep = E * (E - V)
        V1 = V1_N * N
        V2 = V2_Ht * Ht + V2_N * N
        vals = (Ep * V1 * N, Ep * V2 * N, Ep * N * N)
        if not want:
            return vals
        Vp = c0 * SQ_Q + c1 * Np
        Epp = E * ((2 * E - V) * (E - V) - 1)
        dEp = Epp * Vp
        V1p = V1_N * Np
        V2p = V2_Ht * SQ_Q + V2_N * Np
        d = (dEp * V1 * N + Ep * (V1p * N + V1 * Np),
             dEp * V2 * N + Ep * (V2p * N + V2 * Np),
             dEp * N * N + Ep * 2 * N * Np)
        return d

    I = [arb(0), arb(0), arb(0)]
    for (Nm, Npm, Htm, Nc, Npc, Htc, m0, m1p, m1m) in zip(
            gz.N_mid, gz.Np_mid, gz.Ht_mid, gz.N_cell, gz.Np_cell,
            gz.Ht_cell, gz.m0, gz.m1p, gz.m1m):
        vm = pieces(Htm, Nm, Npm, False)
        dc = pieces(Htc, Nc, Npc, True)
        for k in range(3):
            I[k] = I[k] + vm[k] * m0 + dc[k] * m1p - dc[k] * m1m
    # |z| >= L tails: E' in (0,1), so kernels bounded by |V_i| N and N^2
    lV1 = _lin_of(arb(0), V1_N)
    lV2 = _lin_of(V2_Ht, V2_N)
    lN = _lin_N()
    L = gz.L
    I[0] = I[0] + _pm(_ltail(_lmul(lV1, lN), L))
    I[1] = I[1] + _pm(_ltail(_lmul(lV2, lN), L))
    I[2] = I[2] + _pm(_ltail(_lmul(lN, lN), L))
    asT1 = -ALPHA * I[0]
    asT2 = -ALPHA * I[1]
    Sss = PSI - ALPHA * I[2]      # d^2_s T = -E[E' N^2]  (sign fixed)
    return asT1, asT2, Sss


def true_hessian_a(a1, a2, l1, l2, s, gx=None, gz=None):
    """Enclosure of nabla^2 (H+G)(a) = nabla^2 S_* in the MOMENT coordinates a,
    over the a-box, given the enclosing l-box (dual image) and tilt s (thin
    ~ s*(a)). Here the eigenvalues are O(-100), so modest wrapping is tolerated.

      nabla^2 H = -[nabla^2 Phi(l)]^{-1}
      nabla^2 G = alpha nabla^2_a T(a,s) - (alpha d_a d_s T)(...)^T / d^2_s S.
    Returns (H11, H12, H22)."""
    if gx is None:
        gx = hg.get_x_grid()
    if gz is None:
        gz = hg.get_zt_grid()
    P11, P12, P22 = _Phi_hessian(l1, l2, gx)
    iH11, iH12, iH22 = _inv2(P11, P12, P22)   # [nabla^2 Phi]^{-1}
    H_H11, H_H12, H_H22 = -iH11, -iH12, -iH22  # nabla^2 H
    Td = _T_derivs(a1, a2, gz, s)
    if Td is None:
        return None
    asT1, asT2, Sss = _a_s_mixed(a1, a2, s, gz)
    # nabla^2 G = alpha d2T - (asT1,asT2)(asT1,asT2)^T / Sss
    G11 = ALPHA * Td['d2T11'] - asT1 * asT1 / Sss
    G12 = ALPHA * Td['d2T12'] - asT1 * asT2 / Sss
    G22 = ALPHA * Td['d2T22'] - asT2 * asT2 / Sss
    return H_H11 + G11, H_H12 + G12, H_H22 + G22


if __name__ == "__main__":
    set_prec(60)
    import time
    # point check at (1,0): should match Huang M11~-0.0459, M12~-0.0262, M22~-0.0205
    t0 = time.time()
    H = hessian_box(arb(1), arb(0))
    print(f"point (1,0): H11={H[0]}  H12={H[1]}  H22={H[2]}  ({time.time()-t0:.1f}s)")
    print(f"  Huang: M11<=-0.045408, M12 in[-0.026567,-0.025685], M22<=-0.020490")
