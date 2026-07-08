# Audit record

The certificates in this repository were audited in several independent
passes before the final runs. This directory preserves the reports. The
reviews were carried out by independent AI reviewer sessions, each given
only the source files and the defining mathematics and instructed to
re-derive every formula and hunt for soundness defects (wrong signs,
invalid enclosures, missing tails, unsound comparisons, coverage gaps);
the findings were then fixed and every affected certificate re-run. Two
further checks are independent of the review passes: finite-difference
validation of every ingredient of the ray second derivative against a
floating-point implementation, and `../validate_independent.py`, which
recomputes sample certified quantities from the defining formulas in
mpmath with no shared code.

## Findings that led to fixes (all fixed, all runs re-done afterwards)

1. Quadrature remainder rule. The fixed-grid mean-value rule multiplied
   the derivative enclosure by the signed first moment
   int_cell (z-mid) phi ~ 0; the remainder does not factor through the
   signed moment (the enclosed derivative varies while (z-mid) changes
   sign), so the balls were systematically too tight. Caught by a
   30-digit independent quadrature of grad Phi(1,0) falling outside the
   returned ball. Fixed by treating the positive and negative parts of
   (z-mid) separately.
2. Sign of the second tilt derivative: psi - alpha E[E'N^2], not plus.
   Caught by finite-difference validation (the true value at the fixed
   point is 0.4988, not 4.65).
3. Analyticity guard on the complex log of 2cosh. Off the real axis a
   ball of 2cosh can be tightly negative-real and the principal log then
   returns a finite but wrong enclosure of the analytic continuation;
   the evaluators now return a non-finite ball unless Re(2cosh) is
   certainly positive, forcing subdivision toward the axis.
4. Missing |z| >= L tails in the derivative integrals (order 1e-16
   against working radii of 1e-2; added for completeness).
5. Ray origin. The certificates must cover rays from the true maximizer,
   which lies within about 2e-8 of the stored decimals; all ray origins,
   hull corners and the tilt origin are inflated by a 1e-7 ball.
6. Star-radius bookkeeping between Region I and the stage-2 sweep:
   angular cells can straddle the radius-zone boundaries, so the sweep's
   skip test uses zone radii shrunk by the certified cell widths.
7. A coverage sliver between the stage-1 exclusion box and the stage-2
   domain (cells straddling the box edge at minimal size were claimed by
   neither); stage 2 now sweeps the box enlarged by twice the stage-1
   minimal cell size.
8. Containment of the moment body in the sweep rectangle is now itself a
   certified check (support function values at the axis directions).
9. Endpoint corner cases in two clamp helpers (`pos_part`, `min_one`)
   and a domain assertion on the Gaussian tail formulas.
10. The B_Lambda corner-selection preconditions were derived, added as
    runtime guards, and validated post hoc against all 836 localization
    boxes of the completed Region I run (no violations).
11. Nested rigorous integrals could recurse to a C-stack overflow and
    kill worker processes silently; recursion depth is now capped (Arb
    then returns wider, still rigorous, enclosures) and the Ding-Sun
    middle-interval driver runs each chunk in an isolated subprocess.

## Findings about the published literature

The constants printed in Ding-Sun Lemma 8.2 (1.78 and 4.3) do not match
the integrals they bound (2.678... and 3.162...); their own verified fact
1 - ell(100) > 0.025 already contradicts the printed consequence. With
corrected constants the same proof closes on [0.982, 1) instead of
[0.98, 1), and the part (a) grid reaches lambda(0.99) = 0.98665..., so
the covering is unchanged. The double integral of their Proposition 8.4
is -0.4447..., not -0.45. Details in the paper.

## Reports

- `01_core_quadrature.md` - review of `core.py` and `huanggrid.py`.
- `02_derivative_integrals.md` - review of `huang_hessian.py` against
  finite-difference ground truth.
- `03_remaining_scopes.md` - closure of the remaining review scopes
  (Region I geometry, the sweeps and their assembly, the Ding-Sun
  evaluators), whose reviewer sessions were interrupted; the questions
  they were assigned are answered point by point.
