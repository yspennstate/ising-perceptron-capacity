"""Certified numerics shared by the perceptron capacity verification.

Everything here returns arb/acb balls that rigorously enclose the true value.
Parameter uncertainty (the Ding-Sun rectangle for alpha_*, q_0, psi_0) is
propagated by passing arb balls that contain the whole interval.

Conventions. arb comparisons (a < b, a > b) are True only when certain; we
rely on this for all verdicts. Decimal constants are built from integer
ratios so no float rounding enters. Gaussian measure is phi(z) dz; Psi is
the upper tail, E(x) = phi(x)/Psi(x) the inverse Mills ratio.
"""

from flint import arb, acb, ctx

DIGITS_GUARD = 30


def set_prec(bits):
    ctx.prec = bits


set_prec(120)


def rat(p, q=1):
    """Exact rational p/q as a tight arb ball."""
    return arb(p) / arb(q)


def dec(s):
    """Decimal string as an exact rational arb (e.g. dec('0.833078599'))."""
    s = s.strip()
    neg = s.startswith('-')
    if neg:
        s = s[1:]
    if '.' in s:
        a, b = s.split('.')
        v = rat(int(a + b), 10 ** len(b))
    else:
        v = rat(int(s))
    return -v if neg else v


def hull(a, b):
    """arb ball containing both a and b."""
    return a.union(b)


def endpoints(x):
    """Conservative lower and upper endpoints of a ball as arbs."""
    m, r = arb(x.mid()), arb(x.rad())
    return m - r, m + r


def pos_part(x):
    """Enclosure of max(0, x). (The identity (x+|x|)/2 is NOT valid in ball
    arithmetic for balls spanning zero.)"""
    lo, hi = endpoints(x)
    if lo > 0:
        return x
    if hi < 0:                    # certainly nonpositive
        return arb(0)
    return arb(0).union(hi)      # covers the uncertain-endpoint corner too


def sq_nonneg(x):
    """Tight enclosure of x^2 for a ball with x >= 0 (or clamped at 0).
    Plain ball multiplication x*x loses the dependency and can produce
    sign-spanning squares for wide balls."""
    lo, hi = endpoints(x)
    if not (lo > 0):
        lo = arb(0)
    return (lo * lo).union(hi * hi)


def sq_any(x):
    """Tight enclosure of x^2 for any real ball."""
    return sq_nonneg(abs(x))


def sqrt_nonneg(x):
    """Enclose sqrt(x) when the mathematical interval x is nonnegative.

    Outward rounding can make a ball touching zero have a tiny negative
    lower endpoint; calling ``arb.sqrt`` on that representation returns NaN.
    Clamp only that lower endpoint before taking endpoint roots, preserving a
    sound enclosure while keeping the zero endpoint usable.
    """
    lo, hi = endpoints(x)
    if hi < 0:
        return arb(0, arb('inf'))
    if lo < 0:
        lo = arb(0)
    return lo.sqrt().union(hi.sqrt())


def min_one(x):
    """Enclosure of min(x, 1)."""
    lo, hi = endpoints(x)
    lo1 = arb(1) if (lo > 1) else lo      # keep lo unless certainly > 1
    hi1 = arb(1) if not (hi < 1) else hi  # cap at 1 when possibly >= 1
    return lo1.union(hi1)


def iv(lo, hi):
    """arb ball for the closed interval [lo, hi], endpoints arb or str."""
    if isinstance(lo, str):
        lo = dec(lo)
    if isinstance(hi, str):
        hi = dec(hi)
    return lo.union(hi)


# ---------------------------------------------------------------------------
# The Ding-Sun parameter rectangle (their eq. (7.1); quoted by Huang App. B).
# ---------------------------------------------------------------------------

ALPHA = iv('0.833078599', '0.833078600')
ALPHA_LB = dec('0.833078599')
ALPHA_UB = dec('0.833078600')

Q_LB = dec('0.56394907949')
Q_LU = dec('0.56394907950')
Q_UL = dec('0.56394908029')
Q_UB = dec('0.56394908030')
Q = iv(Q_LB, Q_UB)

PSI_LB = dec('2.5763513100')
PSI_LU = dec('2.5763513103')
PSI_UL = dec('2.5763513221')
PSI_UB = dec('2.5763513224')
PSI = iv(PSI_LB, PSI_UB)


def gamma_of(q):
    """gamma(q) = sqrt(q/(1-q)) (Ding-Sun eq. (7.2))."""
    return (q / (1 - q)).sqrt()


GAMMA_LB = gamma_of(Q_LB)
GAMMA_UB = gamma_of(Q_UB)
GAMMA = iv(GAMMA_LB, GAMMA_UB)

SQRT2PI = (2 * arb.pi()).sqrt()
LOG2 = arb(2).log()


# ---------------------------------------------------------------------------
# Special functions on arb (real balls).
# ---------------------------------------------------------------------------

def phi(x):
    return (-x * x / 2).exp() / SQRT2PI


def Psi(x):
    """Gaussian upper tail P(Z >= x) = erfc(x/sqrt(2))/2."""
    return (x / arb(2).sqrt()).erfc() / 2


def logPsi(x):
    return Psi(x).log()


def mills(x):
    """E(x) = phi(x)/Psi(x). Fine for |x| up to ~1e3 at this precision."""
    return phi(x) / Psi(x)


def ent2(p):
    """Binary entropy in nats; p an arb ball inside (0,1)."""
    return -(p * p.log() + (1 - p) * (1 - p).log())


def ent2_tanh(h):
    """ent2((1+tanh h)/2) = log(2 cosh h) - h tanh h, stable for large |h|.

    log(2 cosh h) = |h| + log(1+exp(-2|h|)).
    """
    a = abs(h)
    return a + (-2 * a).exp().log1p() - h * h.tanh()


# ---------------------------------------------------------------------------
# Special functions on acb (for integrands; guard analyticity).
# ---------------------------------------------------------------------------

def _nonanalytic():
    return acb(arb(0, arb('inf')))


def c_phi(z):
    return (-z * z / 2).exp() / SQRT2PI


def c_Psi(z):
    return (z / acb(2).sqrt()).erfc() / 2


def c_logPsi(z, analytic):
    return c_Psi(z).log(analytic=analytic)


def c_mills(z):
    return c_phi(z) / c_Psi(z)


def c_log2cosh(z):
    """log(2 cosh z) with an analyticity guard: 2cosh vanishes at
    i(k+1/2)pi, so off the axis a ball of 2cosh(z) can be tightly
    negative-real and the principal log would give a finite but wrong
    enclosure of the analytic continuation.  Require Re(2cosh z) > 0."""
    c = 2 * z.cosh()
    if not (c.real > 0):
        return _nonanalytic()
    return c.log()


def c_ent2(p, analytic):
    """Binary entropy of an acb ball p; requires Re p in (0,1) to be analytic."""
    ok = (p.real > 0) and ((1 - p).real > 0)
    if not ok:
        return _nonanalytic()
    return -(p * p.log() + (1 - p) * (1 - p).log())


# ---------------------------------------------------------------------------
# Rigorous integration helpers.
# ---------------------------------------------------------------------------

def integrate(f, a, b, abs_tol=None, pieces=None):
    """Rigorous integral of f over [a, b] via arb's adaptive integrator.

    f: callable (acb z, bool analytic) -> acb, holomorphic on the integration
    tube wherever it returns a finite ball with analytic=True.
    Returns the real part as arb (all our integrands are real on the axis;
    the imaginary enclosure is checked to contain 0).

    abs_tol defaults to 2^-(2 prec/3): loose enough that guarded integrands
    (log Psi etc.) converge, tight enough for every margin in this project.
    pieces: optional list of interior breakpoints for badly scaled ranges.
    depth/eval limits are raised; the defaults abort on guarded integrands.
    """
    tol = abs_tol if abs_tol is not None else arb(2) ** (-(2 * ctx.prec) // 3)
    pts = [arb(a)] + [arb(p) for p in (pieces or [])] + [arb(b)]
    total = acb(0)
    for lo, hi in zip(pts[:-1], pts[1:]):
        total += acb.integral(f, acb(lo), acb(hi), abs_tol=tol,
                              depth_limit=4000, eval_limit=4000000)
    assert total.imag.contains(arb(0)), "integral drifted off the real axis"
    return total.real


def gauss_integral(w, L):
    """Rigorous int_{-L}^{L} w(z) phi(z) dz; w: (acb, analytic) -> acb."""
    return integrate(lambda z, an: w(z, an) * c_phi(z), -arb(L), arb(L))


def gauss_tail_mass(L):
    """P(|Z| >= L) as arb upper bound (exact: 2 Psi(L)); requires L >= 0."""
    L = arb(L)
    assert L >= 0
    return 2 * Psi(L)


def z2_tail(L):
    """int_{|z|>=L} z^2 phi(z) dz = 2(L phi(L) + Psi(L))."""
    L = arb(L)
    return 2 * (L * phi(L) + Psi(L))


def z1_tail(L):
    """int_{|z|>=L} |z| phi(z) dz = 2 phi(L)."""
    return 2 * phi(arb(L))


def report(name, value, want):
    """Print one verified inequality; return bool verdict.

    want: '<0', '>0', or ('in', lo, hi).
    """
    if want == '<0':
        ok = value < 0
    elif want == '>0':
        ok = value > 0
    else:
        _, lo, hi = want
        ok = (value > lo) and (value < hi)
    print(f"{'PASS' if ok else 'FAIL'}  {name} = {value}  [{want}]")
    return bool(ok)
