# Review pass: huang_hessian.py

Scope: the T-derivative integrals (first and second derivatives in the
moment coordinates and in the tilt), the moment map and the tilted
covariance, and the mean-value remainder assembly. Every formula was
re-derived by hand and the running code was cross-checked against
finite-difference ground truth at the fixed point (all signs and values
reproduce to about four digits: d2T11 = -0.8616, d2T12 = 3.0309,
d2T22 = 1.3733, alpha d2T/da1ds = 1.2212, alpha d2T/da2ds = -0.3942,
Sss = 0.4988, dT/da1 = 1.2003, a(1,0) = (1.1234, 0.5639), and the tilt
stationarity holds at s0).

A note on conventions: d^2/dV^2 log Psi = -E'(V), so the second-derivative
kernels carry -E' (the code is correct; the opposite sign reproduces the
wrong ground truth).

## Defects found

1. Missing |z| >= L tails in `_T_derivs`, `_s_derivs`, `_a_s_mixed` and
   `_x_quant`: the grid sums ran over [-9, 9] with no tail term, so the
   returned balls were not formally enclosures of the integrals over the
   line. Measured size about 1.5e-16 against working radii of 0.01-0.15;
   immaterial to every verdict, but a genuine rigor gap. Fixed by adding
   closed-form tail bounds (products of linear forms in |z| against
   Gaussian moments).
2. `_inv2` divides by a determinant enclosure with no positivity guard; at
   the default grid the determinant ball spans zero and the routine
   returns NaN (which fails every comparison, hence fails safe). The
   routine sits on a legacy path not used by the certification drivers.
3. Unguarded division by the Sss enclosure in the legacy `true_hessian`
   routines (same status: fails safe, unused by the drivers).
4. `_s_derivs` and `_a_s_mixed` relied on a caller having already checked
   a2^2 < q on the same box; local guards were added.

## Verified with no defects

The dV/da coefficient formulas (first and second derivatives, including
the tedious d^2 V/da2^2 assembly); all five kernel values and their
z-derivative (mean-value remainder) formulas; the signs of the mixed and
second tilt derivatives (after the previously found sign fix); the moment
map integrands, z-derivatives and tail (which covers both components);
the tilted covariance; the 0th-order interval rule in `_x_quant` (cell
enclosures, not midpoints); and every mean-value assembly site (the
corrected positive/negative-part rule throughout).

Conclusion: the certification path (the T-derivative and mixed-derivative
integrals used by the ray certificates and the mean-value sweep cells) is
sound; the findings were either immaterial in size or on unused legacy
paths, and all were fixed.
