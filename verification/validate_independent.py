"""Independent spot validation of the certified quantities.

Everything here is recomputed from the defining formulas using mpmath and
scipy only -- none of the interval-arithmetic machinery is imported -- so a
formula-level error in the certificates (which ball arithmetic cannot catch)
would show up as a disagreement.  The checks are floating point and support,
rather than replace, the certificates.

Checks:
  1. the fixed-point identities at (1,0): grad Phi(1,0) = (psi(1-q), q),
     the tilt stationarity at s0, and Phi(1,0) - a1* = H(a*);
  2. S_* = H + G < 0 at sample interior points of the moment body, with H
     from an independent Newton solve of grad Phi(lambda) = a (agreeing with
     the sweep verdicts);
  3. the ray majorant phi_v'' < 0 at sample (t, theta) in the star (agreeing
     with the Region I verdicts), by second differences of the majorant;
  4. PG' and PG'' at sample lambda on the Ding-Sun middle interval (agreeing
     with the block3bc verdicts), by differences of PG = H_DS + P;
  5. the corrected near-one integrals of Ding-Sun Lemma 8.2.

Run:  python validate_independent.py     (several minutes)
"""

import mpmath as mp

mp.mp.dps = 25

PSI = mp.mpf('2.5763513162')
Q = mp.mpf('0.563949080')
ALPHA = mp.mpf('0.8330785996')
GAMMA = mp.sqrt(Q / (1 - Q))
SQPSI = mp.sqrt(PSI)
S0 = mp.sqrt(1 - Q)
A1S = PSI * (1 - Q)
A2S = Q

FAILS = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'BAD '} {name} {detail}")
    if not ok:
        FAILS.append(name)


def E_mills(x):
    return mp.npdf(x) / mp.ncdf(-x)


# --- the Huang-side objects ------------------------------------------------

def gauss(f):
    return mp.quad(lambda z: f(z) * mp.npdf(z), [-9, -1, 0, 1, 9])


def Phi(l1, l2):
    return gauss(lambda z: mp.log(2 * mp.cosh(l1 * SQPSI * z
                                              + l2 * mp.tanh(SQPSI * z))))


def gradPhi(l1, l2):
    a1 = gauss(lambda z: SQPSI * z * mp.tanh(l1 * SQPSI * z
                                             + l2 * mp.tanh(SQPSI * z)))
    a2 = gauss(lambda z: mp.tanh(SQPSI * z) * mp.tanh(l1 * SQPSI * z
                                                      + l2 * mp.tanh(SQPSI * z)))
    return a1, a2


def dual_of_ct(a1, a2, nsteps=14):
    """Continuation from a* along the segment (robust for far points)."""
    l = (mp.mpf(1), mp.mpf(0))
    for k in range(1, nsteps + 1):
        t = mp.mpf(k) / nsteps
        p1 = A1S + t * (a1 - A1S)
        p2 = A2S + t * (a2 - A2S)
        l = dual_of(p1, p2, l)
    return l


def dual_of(a1, a2, l0=(1.0, 0.0), steps=60):
    l1, l2 = mp.mpf(l0[0]), mp.mpf(l0[1])
    for _ in range(steps):
        g1, g2 = gradPhi(l1, l2)
        r1, r2 = g1 - a1, g2 - a2
        if abs(r1) + abs(r2) < mp.mpf('1e-18'):
            break
        h = mp.mpf('1e-6')
        j11 = (gradPhi(l1 + h, l2)[0] - g1) / h
        j12 = (gradPhi(l1, l2 + h)[0] - g1) / h
        j21 = (gradPhi(l1 + h, l2)[1] - g2) / h
        j22 = (gradPhi(l1, l2 + h)[1] - g2) / h
        det = j11 * j22 - j12 * j21
        dl1 = (j22 * r1 - j12 * r2) / det
        dl2 = (-j21 * r1 + j11 * r2) / det
        step = mp.mpf(1)
        n = abs(dl1) + abs(dl2)
        if n > 1:
            step = 1 / n          # damping for large steps
        l1, l2 = l1 - step * dl1, l2 - step * dl2
    return l1, l2


def H_of(a1, a2, l0=None):
    l1, l2 = dual_of_ct(a1, a2) if l0 is None else dual_of(a1, a2, l0)
    # verify the solve
    g1, g2 = gradPhi(l1, l2)
    assert abs(g1 - a1) + abs(g2 - a2) < mp.mpf('1e-10'), "dual not solved"
    return Phi(l1, l2) - l1 * a1 - l2 * a2, (l1, l2)


def T_of(a1, a2, s):
    D = mp.sqrt(1 - a2 * a2 / Q)

    def f(z):
        N = E_mills(-GAMMA * z) / mp.sqrt(1 - Q)
        V = -(a2 / Q) * mp.sqrt(Q) * z / D - (a1 / PSI) * N / D + s * N
        return mp.log(mp.ncdf(-V))
    return gauss(f)


def G_of(a1, a2):
    """min over s >= 0 (the minimizer can sit at the boundary s = 0, where
    Newton fails; fall back to a scan with golden refinement)."""
    f = lambda s: s * s * PSI / 2 + ALPHA * T_of(a1, a2, s)
    s = mp.mpf(S0)
    try:
        for _ in range(40):
            h = mp.mpf('1e-6')
            d = (f(s + h) - f(s - h)) / (2 * h)
            d2 = (f(s + h) - 2 * f(s) + f(s - h)) / (h * h)
            if abs(d) < mp.mpf('1e-15'):
                break
            if d2 <= 0:
                raise ZeroDivisionError
            s = s - d / d2
        if s < 0:
            raise ZeroDivisionError
    except ZeroDivisionError:
        ss = [mp.mpf(k) / 100 for k in range(0, 301)]
        vs = [f(x) for x in ss]
        k = vs.index(min(vs))
        lo = ss[max(0, k - 1)]
        hi = ss[min(len(ss) - 1, k + 1)]
        g = (mp.sqrt(5) - 1) / 2
        a, b = lo, hi
        c, d_ = b - g * (b - a), a + g * (b - a)
        fc, fd = f(c), f(d_)
        for _ in range(60):
            if fc < fd:
                b, d_, fd = d_, c, fc
                c = b - g * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d_, fd
                d_ = a + g * (b - a)
                fd = f(d_)
        s = (a + b) / 2
    return f(s), s


def majorant(t, th, sdot):
    v1, v2 = mp.cos(th), mp.sin(th)
    x1, x2 = A1S + t * v1, A2S + t * v2
    s = S0 + t * sdot
    Hv, lam = H_of(x1, x2)
    return Hv + s * s * PSI / 2 + ALPHA * T_of(x1, x2, s)


# --- the Ding-Sun-side objects ---------------------------------------------

def ell_of(A):
    """lambda = ell(A) = E[D_H(A)]/(1-q), Ding-Sun eq. (168), with their
    well-conditioned D_H = (1-m^2)(1 - 2/(Delta+1))."""
    def DH(z):
        m = mp.tanh(SQPSI * z)
        Delta = mp.sqrt(A * A * (1 - m * m) + m * m)
        return (1 - m * m) * (1 - 2 / (Delta + 1))
    return gauss(DH) / (1 - Q)


def I_of(lam):
    c = mp.sqrt((1 - lam) / (1 + lam))
    r = mp.sqrt(1 - lam * lam)

    def inner(z):
        gz = GAMMA * z

        def f(x):
            g = GAMMA * c * z - lam * x / r
            return mp.log(mp.ncdf(-g)) * mp.npdf(gz + x)
        v = mp.quad(f, [0, 3, 9])
        return v * mp.npdf(z) / mp.ncdf(-gz)
    return ALPHA * mp.quad(inner, [-7, -2, 0, 2, 7])


def dPG(lam, A):
    """PG'(lam) = -(1-q) log(A)/2 + [-psi(1-q)/(1+lam)^2 + I'(lam)],
    I' by central difference of the independent I_of."""
    h = mp.mpf('2e-4')
    Ip = (I_of(lam + h) - I_of(lam - h)) / (2 * h)
    return -(1 - Q) * mp.log(A) / 2 - PSI * (1 - Q) / (1 + lam) ** 2 + Ip


def A_of_lam(lam):
    """Invert ell numerically (ell is increasing; bisection is robust)."""
    lo, hi = mp.mpf('1e-6'), mp.mpf(50)
    for _ in range(200):
        mid = (lo + hi) / 2
        if ell_of(mid) < lam:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    # 1. fixed-point identities
    a1, a2 = gradPhi(1, 0)
    check("grad Phi(1,0) = (psi(1-q), q)",
          abs(a1 - A1S) < 1e-9 and abs(a2 - A2S) < 1e-9,
          f"({mp.nstr(a1, 12)}, {mp.nstr(a2, 12)})")
    h = mp.mpf('1e-6')
    f = lambda s: s * s * PSI / 2 + ALPHA * T_of(A1S, A2S, s)
    ds = (f(S0 + h) - f(S0 - h)) / (2 * h)
    check("tilt stationarity at s0", abs(ds) < 1e-8, f"dS/ds = {mp.nstr(ds, 3)}")
    Sstar0 = Phi(1, 0) - A1S + f(S0)
    check("S_*(a*) = 0", abs(Sstar0) < 1e-9, f"= {mp.nstr(Sstar0, 3)}")

    # 2. bulk sample points: S_* < 0 (matching the sweep verdicts)
    for (p1, p2) in [(0.8, 0.40), (1.0, 0.50), (0.6, 0.30), (1.05, 0.58),
                     (0.3, 0.15), (-0.5, -0.3)]:
        try:
            if p1 < 0:
                # grad Phi is odd, so lambda(-a) = -lambda(a): seed there
                lp = dual_of_ct(mp.mpf(-p1), mp.mpf(-p2))
                Hv, lam = H_of(mp.mpf(p1), mp.mpf(p2),
                               l0=(-lp[0], -lp[1]))
            else:
                Hv, lam = H_of(mp.mpf(p1), mp.mpf(p2))
            Gv, _ = G_of(mp.mpf(p1), mp.mpf(p2))
            check(f"S_*({p1},{p2}) < 0", Hv + Gv < 0,
                  f"= {mp.nstr(Hv + Gv, 6)}")
        except Exception as e:
            check(f"S_*({p1},{p2}) < 0", False, f"error {e}")

    # 3. ray majorant concavity at sample star points (slope from the
    # numeric argmin-s drift along the ray, as in the certificates)
    def sdot_num(th, d=mp.mpf('2e-4')):
        v1, v2 = mp.cos(th), mp.sin(th)
        sp = G_of(A1S + d * v1, A2S + d * v2)[1]
        sm = G_of(A1S - d * v1, A2S - d * v2)[1]
        return (sp - sm) / (2 * d)

    for (t, th) in [(0.006, 0.6253), (0.010, 0.6253),
                    (0.004, 0.6253 + 3.1416), (0.003, 2.2)]:
        sdot = sdot_num(mp.mpf(th))
        h = mp.mpf('4e-4')
        pp = (majorant(t + h, th, sdot) - 2 * majorant(t, th, sdot)
              + majorant(t - h, th, sdot)) / (h * h)
        check(f"phi''(t={t}, th={th}) < 0", pp < 0, f"= {mp.nstr(pp, 4)}")

    # 4. DS middle interval: dPG signs at sample lambda
    for (lam, sign) in [(0.08, -1), (0.15, -1), (-0.06, 1), (-0.10, 1)]:
        A = A_of_lam(mp.mpf(lam))
        d = dPG(mp.mpf(lam), A)
        check(f"sign PG'({lam}) = {sign}", mp.sign(d) == sign,
              f"= {mp.nstr(d, 4)} (A = {mp.nstr(A, 6)})")

    # 5. corrected Lemma 8.2 integrals
    I1 = gauss(lambda z: mp.sqrt(1 - mp.tanh(SQPSI * z) ** 2))
    I2 = gauss(lambda z: abs(mp.tanh(SQPSI * z)))
    c1 = 2 * I1 / (1 - Q)
    c2 = 2 * I2 / (1 - Q)
    check("Lemma 8.2 first integral in (2.67, 2.679)", 2.67 < c1 < 2.679,
          f"= {mp.nstr(c1, 8)}")
    check("Lemma 8.2 second integral in (3.15, 3.17)", 3.15 < c2 < 3.17,
          f"= {mp.nstr(c2, 8)}")

    print(f"\n{'ALL AGREE' if not FAILS else 'DISAGREEMENTS: ' + str(FAILS)}")
    if FAILS:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
