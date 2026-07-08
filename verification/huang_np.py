"""Vectorized (numpy) nonrigorous companion: fast S_*, moment map, optimal
tilt, lambda<->(a1,a2). Guides cell tilts and dual points; not a certificate.
"""

import numpy as np
from scipy.special import erfc, log_ndtr

ALPHA = 0.8330785995
Q = 0.5639490799
PSI = 2.5763513162
GAMMA = np.sqrt(Q / (1 - Q))
SQPSI = np.sqrt(PSI)
SQQ = np.sqrt(Q)
S1Q = np.sqrt(1 - Q)

NZ = 4001
ZL = 9.0
_z = np.linspace(-ZL, ZL, NZ)
_dz = _z[1] - _z[0]
# Simpson weights * phi
_w = np.ones(NZ)
_w[1:-1:2] = 4
_w[2:-1:2] = 2
_w *= _dz / 3 * np.exp(-_z**2 / 2) / np.sqrt(2 * np.pi)
_M = np.tanh(SQPSI * _z)
_X = SQPSI * _z


def _Psi(x):
    return 0.5 * erfc(x / np.sqrt(2))


def _logPsi(x):
    return log_ndtr(-x)   # log Psi(x) = log Phi(-x)


def _mills(x):
    return np.exp(-x * x / 2) / np.sqrt(2 * np.pi) / _Psi(x)


_N = _mills(-GAMMA * _z) / S1Q


def profile(l1, l2):
    u = l1 * _X + l2 * _M
    Lam = np.tanh(u)
    au = np.abs(u)
    ent = np.sum(_w * (au + np.log1p(np.exp(-2 * au)) - u * Lam))
    a1 = np.sum(_w * _X * Lam)
    a2 = np.sum(_w * _M * Lam)
    return ent, a1, a2


def Phi(b1, b2):
    th = b1 * _X + b2 * _M
    ath = np.abs(th)
    return np.sum(_w * (ath + np.log1p(np.exp(-2 * ath))))


def T(a1, a2, s):
    d2 = 1 - a2 * a2 / Q
    if d2 <= 0:
        return np.nan
    D = np.sqrt(d2)
    c0 = -(a2 / Q) / D
    c1 = s - (a1 / PSI) / D
    V = c0 * SQQ * _z + c1 * _N
    return np.sum(_w * _logPsi(V))


def G(a1, a2):
    """inf_s [ s^2 psi/2 + alpha T ]."""
    ss = np.linspace(0.0, 4.0, 200)
    vals = [s * s * PSI / 2 + ALPHA * T(a1, a2, s) for s in ss]
    k = int(np.nanargmin(vals))
    # local golden refine
    lo, hi = ss[max(0, k - 1)], ss[min(len(ss) - 1, k + 1)]
    g = (np.sqrt(5) - 1) / 2
    f = lambda s: s * s * PSI / 2 + ALPHA * T(a1, a2, s)
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc; c = b - g * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd; d = a + g * (b - a); fd = f(d)
    sm = (a + b) / 2
    return sm, f(sm)


def _dual_val_grad_hess(b1, b2, a1, a2):
    """Psi = Phi(b) - b.a, its grad (a(b)-a) and Hessian Cov(X,M | tilt)."""
    th = b1 * _X + b2 * _M
    ath = np.abs(th)
    Phiv = np.sum(_w * (ath + np.log1p(np.exp(-2 * ath))))
    tanth = np.tanh(th)
    A1 = np.sum(_w * _X * tanth)
    A2 = np.sum(_w * _M * tanth)
    s = 1 - tanth * tanth
    H11 = np.sum(_w * _X * _X * s)
    H12 = np.sum(_w * _X * _M * s)
    H22 = np.sum(_w * _M * _M * s)
    val = Phiv - b1 * a1 - b2 * a2
    return val, np.array([A1 - a1, A2 - a2]), np.array([[H11, H12], [H12, H22]])


def dual_of(a1, a2, l0=(1.0, 0.0)):
    """Dual (b1,b2) with a(b)=(a1,a2), via damped Newton on the convex dual
    objective with the analytic (PD) Hessian. Returns (b1,b2) or None."""
    b = np.array([float(l0[0]), float(l0[1])])
    val, g, H = _dual_val_grad_hess(b[0], b[1], a1, a2)
    for _ in range(80):
        if abs(g[0]) + abs(g[1]) < 1e-12:
            return float(b[0]), float(b[1])
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[0, 1]
        if det < 1e-16:
            break
        step = np.array([(H[1, 1] * g[0] - H[0, 1] * g[1]) / det,
                         (-H[0, 1] * g[0] + H[0, 0] * g[1]) / det])
        t = 1.0
        while t > 1e-6:
            nb = b - t * step
            if abs(nb[0]) < 300 and abs(nb[1]) < 300:
                nval, ng, nH = _dual_val_grad_hess(nb[0], nb[1], a1, a2)
                if nval <= val - 1e-4 * t * (g[0] * step[0] + g[1] * step[1]):
                    break
            t *= 0.5
        else:
            break
        b, val, g, H = nb, nval, ng, nH
    _, A1, A2 = profile(b[0], b[1])
    if abs(A1 - a1) + abs(A2 - a2) < 1e-8:
        return float(b[0]), float(b[1])
    return None


def S_star(a1, a2):
    lam = dual_of(a1, a2)
    if lam is None:
        return None, None
    ent, _, _ = profile(*lam)
    sm, g = G(a1, a2)
    return ent + g, (lam, sm)
