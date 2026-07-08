# Review pass: core.py and huanggrid.py

Scope: the special functions (real and complex), the adaptive integration
wrapper, the tail formulas, the exact-rational constructors, the fixed-grid
quadrature rule and every accumulation site using it, the Phi and T
evaluators, the robust wide-ball evaluators, the support function and the
outside-K test, and the S_moment upper bound. Every formula was re-derived
from the definitions.

## Defects found

1. core.py, `c_log2cosh`: no analyticity guard. 2cosh(z) vanishes at
   i(k+1/2)pi; for a complex ball where 2cosh is tightly negative-real the
   principal log returns a finite enclosure that is wrong for the analytic
   continuation, so a quadrature using it off the axis could return an
   invalid ball. Latent at the time of review (no caller in these files);
   fixed by requiring Re(2cosh z) certainly positive, and the same guard
   was added to the Phi evaluator in huang_region1.py, which has the same
   pattern and is on the live path.
2. core.py, `pos_part` and `min_one`: in an ulp-scale corner case (an
   endpoint ball straddling the threshold) the returned ball could exclude
   part of the true range. Margins in this project are many orders larger;
   fixed regardless.
3. core.py, tail formulas (`gauss_tail_mass`, `z1_tail`, `z2_tail`): valid
   only for L >= 0, previously unguarded (every call site passes L >= 9).
   A domain assertion was added.
4. huanggrid.py: the legacy signed first moment `m1c` remained stored on
   the Grid; documented as legacy to prevent reuse of the old (invalid)
   rule.

## Verified with no defects

The real special functions (phi, Psi, mills, ent2, logPsi and the
log 2cosh identities); the complex evaluators other than c_log2cosh
(analyticity guards correct; c_mills sound wherever finite); the
integrate() wrapper (returning the real part is sound for integrands real
on the axis); the closed-form tails at L >= 0; the exact-rational
constructors; the corrected grid rule (m0, m1p, m1m closed forms
re-derived; the positive/negative-part split gives a valid enclosure; a
consistent exact partition lies inside all cell balls, so the telescoped
sum encloses the integral); Phi_of (derivative and tail); T_of and
T_meanvalue (the dV/da formulas re-derived and matched; the derivative
tail bounds verified, including E(x) <= 1 + |x| and the logPsi lower
bound); the monotone-endpoint evaluators (directions correct); the support
function and outside_K (conservative in the right direction; exclusion
only when every point of the box strictly exceeds the support upper
bound); S_moment (one-sided upper bound structure sound; only its upper
endpoint is used).

Conclusion: the certification path exercised by these two files is sound
as written; the four findings were latent or ulp-scale and were fixed.
