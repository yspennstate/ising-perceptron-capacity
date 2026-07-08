# Review pass: Region I geometry, the sweeps and their assembly, the
# Ding-Sun evaluators

The reviewer sessions assigned to these scopes were interrupted before
reporting; the questions they were assigned were closed point by point
afterwards. The items and their resolutions:

1. Ray origin (Region I). The pinned identities hold at the true fixed
   point, which lies within about 2e-8 of the stored decimal origin, so
   the certificates must cover rays from the true point. Resolved by
   inflating every ray origin, hull corner and the tilt origin by a 1e-7
   ball and re-running Region I in full.

2. B_Lambda corner selection. For the per-z maximum of sech^2 over the
   lambda box, theta = l1 X + l2 M is monotone in both coordinates
   pointwise in z (increasing for z > 0, decreasing for z < 0), so the
   box extreme nearest zero is attained at (l1lo, l2lo) on both tails for
   any box, and the two straddle roots coincide by oddness. The genuine
   preconditions are l1lo > 0 and a valid root bracket; both were added
   as runtime guards and validated post hoc against all 836 localization
   boxes of the completed run (no violations). The root bisection
   maintains the bracket invariant and its outward pad errs toward the
   factor-1 middle piece, which is always an upper bound.

3. Localization anchors (edge check). Each sublevel set contains the
   anchor attaining the minimum, and the anchors are numeric duals of
   sector points lying inside the candidate box; the boundary test is
   convex in x, so corner points of the sector hull suffice. The hull is
   the sector's own vertex set plus an outer-arc bulge vertex beyond the
   sagitta, with corners padded for the origin ball.

4. Star bookkeeping between Region I and the stage-2 sweep. Angular
   cells can straddle the radius-zone boundaries, so a direction's
   certified radius can be the smaller zone's. The sweep's skip test uses
   zone radii shrunk by the certified adjacent cell widths (0.016-radian
   chunks split at least once near the wedge boundary, 0.05-radian
   chunks near the middle boundary), plus a safety margin covering the
   origin ball.

5. Coverage assembly across the sweeps. Stage-1 cells at minimal size
   straddling the exclusion box were claimed by neither stage; stage 2
   now sweeps the box enlarged by twice the stage-1 minimal cell size.
   Containment of the moment body in the sweep rectangle is a certified
   check (support function at the axis directions). The kappa-ladder
   near the body boundary is sound for any fixed direction and scale
   (convex duality holds for any fixed dual vector), and the mean-value
   cell form encloses the maximum over the cell with the gradient
   enclosed over the cell and radii inflated for the rounded center.

6. Ding-Sun evaluators. The derivative identities used by the
   middle-interval certificates (H' = -(1-q) log A / 2,
   H'' = -(1-q)/(2 A ell'(A)), P' and P'' from the border term and the
   certified I' and I'') were re-derived from their definitions; the
   I'' integrand (dg and ddg) was derived symbolically and the closed
   Gaussian-moment tail bounds were checked term by term, with every
   substitution enlarging the integrand. I' and I'' were validated
   against independent numeric derivatives. The tail boxes were enlarged
   so the outside-mass bounds sit far below the certified margins, and
   recursion depth in the nested integrals is capped (wider but rigorous
   results at the cap; previously worker processes could die on C-stack
   overflow).

7. The near-one chain and the value grids follow the Ding-Sun paper
   structure with the corrected constants; the internal consistency of
   the corrected chain (the corollary constant, the closing coefficient
   and its monotonicity, and the grid pin covering the shortened
   interval) is part of block2's certified checks.

Independent of all of the above, `../validate_independent.py` recomputes
sample certified quantities from the defining formulas in mpmath with no
shared code: the fixed-point identities, bulk sample values, ray-majorant
concavity at sample star points, derivative signs on the middle interval,
and the corrected near-one integrals.
