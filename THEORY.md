# Theory and certificate map

## Claim and scope

This repository presents a computer-assisted proof of the Krauth--Mezard
capacity conjecture for the zero-margin Ising perceptron.

Let $M_N$ be the largest number of independent random half-space constraints
that can be satisfied by some vector in $\{-1,+1\}^N$. For Gaussian disorder,
the claimed theorem is

$$
\frac{M_N}{N}\xrightarrow{\mathbb P}\alpha_\star,
\qquad
\alpha_\star\in[0.833078599,\,0.833078600].
$$

Nakajima--Sun's universality theorem transfers the limiting threshold to iid
mean-zero, unit-variance subgaussian disorder, including the original
Bernoulli $\pm1$ model.

The result is specifically for $\kappa=0$. It does not establish the capacity
for nonzero margins, the general positive-temperature free energy, or the full
TAP conjecture.

The certificates and verifiers have passed the repository's source-bound
checks and a separate internal audit. They have not yet been independently
reproduced or peer reviewed. The result should therefore be described as a
claimed computer-assisted solution pending external specialist verification.

## The two published reductions

The proof closes two numerical hypotheses left by earlier analytic work.

Ding--Sun reduce the lower bound to their full Condition 1.2. For their
one-variable rate function $\mathscr S_\star$, on its natural domain
$[\lambda_{\min},1]$, the condition has three clauses:

$$
\mathscr S_\star(\lambda)<0
\quad\text{for }\lambda\notin\{0,1\},
$$

$$
\mathscr S_\star''(0)<0,
\qquad
\lim_{\lambda\uparrow1}\mathscr S_\star'(\lambda)>0.
$$

Under this condition, Ding--Sun obtain the lower bound with positive
probability for every $\alpha<\alpha_\star$. The sharp-threshold result
upgrades this to probability tending to one.

Huang reduces the matching upper bound to Condition 1.3:

$$
\mathcal S_\star(\lambda_1,\lambda_2)\le0
\qquad
\text{for every }(\lambda_1,\lambda_2)\in\mathbb R^2.
$$

Huang rigorously verified the other numerical conditions needed by his
theorem at $\kappa=0$, but left this global two-variable inequality as a
hypothesis.

## Ding--Sun Condition 1.2

Write

$$
\mathscr S_\star=\mathcal H+\mathcal P+\mathcal A,
\qquad
\mathrm{PG}:=\mathcal H+\mathcal P,
$$

where Ding--Sun's auxiliary optimization satisfies

$$
\mathcal A\le0,\qquad
\mathcal A(0)=\mathcal A'(0)=0,\qquad
\mathcal A''(0)\le0.
$$

The certificate covers $[\lambda_{\min},1]$ by overlapping pieces.

- The parameter block independently re-establishes the narrow intervals for
  $\alpha_\star,q_\star,\psi_\star$. It checks the fixed-point rectangle,
  contraction, corner signs, and Gardner free-energy signs using certified
  Gaussian integration.
- The value-grid block proves strict negativity on
  $[\lambda_{\min},-0.125]\cup[0.2,0.982]$. It contains 247 top-level cells
  and 263 final recursive leaves.
- The derivative block proves $\mathrm{PG}'(\lambda)>0$ on
  $[-0.125,-0.03]$ and $\mathrm{PG}'(\lambda)<0$ on $[0.05,0.2]$.
- The central block proves $\mathrm{PG}''(\lambda)<0$ on $[-0.03,0.05]$.
  Since $\mathrm{PG}(0)=\mathrm{PG}'(0)=0$, this gives strict negativity away
  from zero throughout the central interval.
- The near-one block proves strict negativity on $[0.982,1)$ using a corrected
  endpoint estimate. Its interval overlaps the final value-grid cell, so
  there is no uncovered sliver.

The derivative and central computations use the exact identities for
$\mathcal H'$ and $\mathcal H''$, certified integral bounds for the derivatives
of $\mathcal P$, and explicit Gaussian-moment tail estimates. The thin
negative-branch derivative margins are handled by a mean-value argument
supported by a certified 59-point $I''$ grid and a closed bound for $I'''$.

These pieces prove the global-sign clause. The central curvature together with
$\mathcal A''(0)\le0$ gives $\mathscr S_\star''(0)<0$. Ding--Sun's analytic
endpoint calculation gives the stronger statement

$$
\mathscr S_\star'(\lambda)\longrightarrow+\infty
\qquad(\lambda\uparrow1),
$$

which supplies the third clause.

## Huang Condition 1.3

### Compact moment coordinates

Set $X\sim N(0,\psi_\star)$, $M=\tanh X$, and

$$
\Lambda_{\lambda_1,\lambda_2}(X)
=\tanh(\lambda_1X+\lambda_2M).
$$

The functional depends on $\Lambda$ through the two moments

$$
a_1=\mathbb E[X\Lambda],
\qquad
a_2=\mathbb E[M\Lambda].
$$

All feasible moment pairs form the compact convex body

$$
K=\left\{
\bigl(\mathbb E[X\Lambda],\mathbb E[M\Lambda]\bigr):
|\Lambda|\le1
\right\}.
$$

Its support function is $h(u,v)=\mathbb E|uX+vM|$. Let $H(a)$ be the
maximum binary entropy among profiles with moments $a$, and let $G(a)$ be the
remaining constraint term after minimizing over the scalar tilt $s$. Entropy
duality gives

$$
\sup_{\lambda_1,\lambda_2}\mathcal S_\star(\lambda_1,\lambda_2)
\le
\sup_{a\in K}\bigl(H(a)+G(a)\bigr).
$$

Thus the unbounded $(\lambda_1,\lambda_2)$-plane is replaced by a bounded
domain. For every fixed dual vector $b$,

$$
H(a)\le\Phi(b)-b\cdot a,
\qquad
\Phi(b)=\mathbb E\log\!\bigl(2\cosh(b_1X+b_2M)\bigr).
$$

For every fixed tilt $s$, the corresponding constraint expression is an upper
bound on $G(a)$. Candidate duals and tilts may therefore be chosen
numerically, rounded to rationals, and then treated as fixed data by the
rigorous verifier.

### Region II: strict negativity away from the maximizer

Region II uses two adaptive rectangle sweeps. For each retained cell, the
verifier reconstructs an Arb enclosure of

$$
\Phi(b)-b\cdot a
+\frac12s^2\psi_\star
+\alpha_\star\,\mathcal T(a,s),
$$

where $\mathcal T$ is the one-dimensional Gaussian constraint integral. A
strictly negative upper endpoint proves $H+G<0$ throughout the cell.

Cells disjoint from $K$ are rejected using certified support-function
witnesses. Cells near the distinguished point are delegated first to the
finer sweep and then to Region I. The bundle verifier checks the entire
delegation chain, so a cell cannot disappear between stages.

The first sweep contains 1,200 top-level cells and 1,822 verified leaves. The
second contains 240 top-level cells and 1,499 verified leaves.

### Region I: ray concavity at the zero

The distinguished moment is

$$
a^\star=(\psi_\star(1-q_\star),q_\star),
$$

the image of $(\lambda_1,\lambda_2)=(1,0)$. Here
$H(a^\star)+G(a^\star)=0$, so strict box evaluation cannot work.

A constructive certificate first proves that a Euclidean ball of radius
greater than $0.0150391982$ around $a^\star$ lies inside $K$. This ensures
that the entropy dual is finite and differentiable throughout the full
Region-I star.

The proof then works along rays $a^\star+tv$. A single rational slope rule
defines a tilt $s(t)$ consistently along every radial band. For each band, the
certificate:

1. localizes the entropy dual inside a bounded rectangle $L$;
2. constructs a positive-definite Loewner majorant
   $$
   \widehat B_L\succeq
   \mathbb E\!\left[
   ff^\top\max_{b\in L}\operatorname{sech}^2(b\cdot f)
   \right];
   $$
3. uses order-reversing matrix inversion to bound the entropy curvature; and
4. proves that the nonentropy curvature is no larger than the matching inverse
   quadratic form.

Consequently the one-dimensional ray majorant is concave. Its value and first
derivative vanish at $t=0$, so it is nonpositive along the entire certified
ray prefix.

The Region-I artifact contains 1,404 band jobs, 4,542 angular leaves, and
16,543 radial pieces. The full verifier reconstructs the localization
witnesses, matrix entries, determinants, inverse forms, and curvature packets
rather than trusting stored signs.

## Code and certificate map

| Mathematical role | Programs | Canonical evidence |
|---|---|---|
| Ding--Sun parameter rectangle | `block1_gardner.py`, `core.py` | Recomputed directly by `verify_all.py` |
| Ding--Sun near-one estimate | `block2_near_one.py`, `dsfun.py` | Recomputed directly by `verify_all.py` |
| Ding--Sun value grids | `block3a_run.py`, `block3a_singlerun.py`, `block3a_assemble.py` | `results/block3a_certificate.json` and `results/run_root/` |
| Ding--Sun derivative and central blocks | `block3bc_exact.py`, `block3bc_aux_verify.py`, `block3bc_assemble.py` | `results/block3bc_certificate.json`, `results/block3bc_aux/`, and `results/block3bc_replay/` |
| Huang moment-body interior | `huang_star_interior.py` | `results/huang_star_interior.json` |
| Huang Region I | `huang_region1.py`, `huang_region1_verify.py` | `results/huang_region1.json` |
| Huang Region II | `huang_sweep.py`, `huang_sweep2.py`, `huang_sweep_verify.py` | `results/huang_sweep.json`, `results/huang_sweep2.json` |
| Huang delegation closure | `huang_sweep_verify.py` | `results/huang_bundle.json` |
| Theorem-level assembly | `verify_all.py` | Nine required PASS rows and `ALL CERTIFICATES PASS` |
| Manuscript inventory consistency | `update_paper_counts.py` | Counts checked against canonical JSON |

Paths in the final column are relative to `verification/`.

## What remains analytic

Arb proves the stated numerical enclosures once the formulas and reductions
are correct. The moment-coordinate reduction, entropy duality,
support-function description of $K$, ray-concavity implication, localization
lemma, Loewner-order argument, Ding--Sun interval decomposition, endpoint
assembly, and final use of the published probabilistic theorems are
mathematical arguments given in the manuscript.

The certificate system is designed to minimize trust in stored output, but it
cannot replace scrutiny of that formula-to-code correspondence. Independent
reproduction and specialist review remain the essential next checks.
