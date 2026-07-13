# Reverification, 2026-07-13 afternoon

This directory records an afternoon-long independent reverification of the
published proof tree (public `ising-perceptron-capacity`, commit
`0221e98c6b473c064960938022d310c6de35389e`, tag `v1.0` peeling to
`bd9a14fa5c092a69a362d2fc6e6cbf7874def1c1`), performed after the release.
Everything below was run from fresh clones of the public repository; nothing
in the proof tree was modified. The new material is additive: a Lean 4
kernel-checked skeleton of the exact-arithmetic certificate layer, portable
and high-precision rechecks of the star-interior certificate, a symbolic
audit of the closed-form identities, falsifiability batteries for both the
new and the published verifiers, and a finite-size satisfiability scan.

## Full replays

Pinned runtime (the attested environment: Python 3.12.3, python-flint 0.9.0,
FLINT 3.6.0, on the same Linux host as the release runs, fresh clone at
`v1.0`): 90/90 public unit tests,
`update_paper_counts.py --check` clean, and `verify_all.py` ending in
`ALL CERTIFICATES PASS` with the full 1,404/1,404 Region-I numerical replay,
fails=0. Log: `logs/box_rerun.log`.

Portability runtime (Windows, Python 3.14.3, python-flint 0.9.0 with its own
FLINT build; complete run in `logs/local_rerun.log`): 90/90 unit tests and
every mathematical replay pass on this second, unattested runtime - the
Block-1 rectangle rows with enclosures digit-identical to the attested run,
the full 1,404/1,404 Region-I numerical replay with zero failures, both
Region-II sweep replays (1,822 and 1,499 leaves), the near-one block, and
block3bc with the recorded certificate hash. The three failing rows are a
single family: `huang_star_interior.verify_certificate` regenerates its
certificate and compares for dict equality, which binds it to the attested
runtime (the payload embeds the runtime record, and a different FLINT build
produces different, equally valid, enclosure radii); the Block-3a evidence
binding is runtime-bound the same way; and the Region-II delegation-chain
row fails only as a cascade, because its bundle check consumes the
star-interior replay. All three pass on the attested runtime. The two
portable checkers below were written to close the star-interior gap.
After the final push, a fresh clone of this private repository was
consumer-tested on the portability runtime: the committed extractor
regenerates all 376 Lean files bit-identical to the committed set, the
portable checkers and the falsifiability batteries pass from the clone,
and the complete private test suite runs 140/140.

## Lean 4 kernel-checked skeleton (`lean_skeleton/`)

`extract_skeleton.py` reads the canonical certificates and integer-clears
every exact-rational claim that the Python verifiers check with
`fractions.Fraction`, emitting self-contained Lean 4 files (no mathlib, no
imports) whose theorems are decided by the Lean kernel's bignum arithmetic:

- Region I, all 1,404 bands: the serialized Loewner-majorant packets prove
  positive definiteness (`b11_lo > 0`, `b22_lo > 0`,
  `b11_lo*b22_lo > max(|b12_lo|,|b12_hi|)^2`), from packet endpoints
  `(mid - rad) * 10^exp10`.
- Region I, all 1,404 bands x 32 localization groups (51,706 leaves): the
  binary leaf paths form complete, prefix-free, duplicate-free covers of
  depth 14 - the combinatorial no-gaps property behind the localization
  certificate.
- Block 3b/3c replay lanes (b_neg 578, b_pos 124, c 185 leaves): every leaf
  packet proves its claimed sign, and 1,629 cross-multiplied equalities
  verify that leaves tile each job interval and jobs tile each lane schedule
  exactly.
- Block 3a (the Ding--Sun value grid): all 263 recursive leaf packets prove
  strict negativity, the recursion-count invariant holds for every row, the
  247 cells tile exactly two contiguous tau runs (245 adjacency equalities),
  and the mirror reproduces the certificate's max_leaf_upper exactly. The
  pinned spans knit the one-dimensional cover together with strict overlaps
  in exact rationals: grid runs tau in [-1, -9/50] and [6/25, 99/100],
  b_neg [-19/100, -3/100], c [-43/1000, 39/500], b_pos [3/50, 13/50], so
  every junction overlaps (-19/100 < -9/50, -43/1000 < -3/100,
  3/50 < 39/500, 6/25 < 13/50) exactly as the interval decomposition in the
  manuscript requires.
- Region II, both sweeps: every one of the 1,881 binary splits partitions
  its parent rectangle exactly (15,048 coordinate equalities and 3,762
  strict orderings on the split points), all 2,210 negative-leaf packets
  (873 in stage 1, 1,337 in stage 2) prove strict negativity, and 5,920
  containment rows place all 1,440 root cells inside their stage domains and
  all 40 delegated rectangles inside `EXCL_OLD`.
- Region I angular and radial trees: the 2,198 angular splits partition
  their intervals exactly and the roots chain across each band's sector; all
  4,542 certified leaves have radial chains that tile `[t0, radial_end]`
  gaplessly with `radial_end = min(t1, tmax)` bound by explicit order rows;
  every one of the 16,543 radial-piece curvature packets proves strict
  negativity (the ray-concavity facts), and all tilt and `qB` packets prove
  strict positivity - 110,506 integer facts in this family.

The same extraction is mirrored fail-closed in Python `Fraction` arithmetic
inside the generator, so an extraction bug cannot silently weaken the Lean
statements. All 376 generated files check under Lean 4.31.0
(`logs/lean_run.log`). `mutation_test.py` corrupts each claim class and
confirms the kernel rejects all thirteen corruption classes
(`logs/lean_mutation.log`).

Scope: this is not a formalization of the analytic mathematics (Gaussian
integration, entropy duality, the ray-concavity implication, the published
probabilistic theorems). It kernel-checks the exact combinatorial and
algebraic layer of the certificate chain - covers, tilings, orderings, and
the sign and definiteness facts carried by the stored packet endpoints - in
a second proof system that shares nothing with python-flint. That the
stored enclosures really contain the transcendental quantities they claim
remains the domain of the certified Arb arithmetic and its replays.

## Portable star-interior checks (`independent_checks/`)

`star_interior_portable_check.py` re-derives the star-interior claim chain
from the stored packets alone, in pure integer arithmetic (fractions plus
isqrt guard digits; rational sin/sqrt envelopes): determinant, edge norms,
inradius, required radius, and the final inequalities, with outward
rounding. On the canonical certificate it certifies inradius >= 0.0150391983
against required <= 0.0120959020 and the floor 0.0150391982
(`logs/star_portable.log`). A four-way corruption battery (matrix shift
exceeding the slack, sign flip, summary-packet tamper, policy tamper) is
rejected with the correct diagnosis in each case.

`star_interior_mpmath_recheck.py` recomputes the four parallelogram edge
integrals with mpmath tanh-sinh quadrature at 50 digits, at both endpoints
of the stored psi enclosure: every value lands inside the corresponding
stored Arb ball (widths ~5e-10), and the closed-form tail bounds agree to
4e-16 relative (`logs/star_mpmath.log`).

`sweep_thin_cells_mpmath.py` re-evaluates the certified Region-II objective
`Phi(lambda) - lambda.a + s^2 psi/2 + alpha T(a,s)` from first principles
(the T kernel rebuilt from `V = c0 sqrt(q) z + c1 E(-gamma z)/sqrt(1-q)`)
with mpmath at 40 digits, at the center and corners of the ten
direct-method cells with the thinnest certified margins in each sweep stage,
using each leaf's own stored rational dual and tilt: every sampled value is
strictly negative and lies below the certified supremum enclosure
(`logs/sweep_thin_cells.log`).

`delegation_portable_check.py` re-establishes the Region-II delegation-chain
geometry in exact rational arithmetic, portably: the 1,200 stage-1 root
cells are disjoint and tile the full `[-A1MAX, A1MAX] x [-A2MAX, A2MAX]`
domain with zero uncovered area; all 40 stage-1 delegated rectangles lie
inside the stage-2 domain `EXCL_OLD`; the 240 stage-2 root cells tile
`EXCL_OLD` exactly; and the sweep-2 policy's Region-I star fields and sdot
model/coefficients equal the Region-I certificate's records
(`logs/delegation_portable.log`). The delegate counts (40 and 42) equal the
paper inventory. Dropping a root cell or shrinking `EXCL_OLD` is rejected
with the correct diagnosis. The one bundle condition left to the attested
runtime is the star-interior norm link, whose few-parts-in-1e5 margin needs
certified trigonometric enclosures; it passes there as part of
`verify_all.py`.

## Symbolic audit (`independent_checks/sympy_formula_audit.py`)

SymPy re-derives 29 closed-form identities that the verifiers transcribe
from the sources, anchored to the Ding--Sun arXiv v1 displays (e:B.A),
(e:D.A), (e:D.A.expand), (e:deriv.D.A), (e:D.H.intro), (e:P.H.D),
(e:II.repeat), (e:PP.pair.repeat) and to the `huang_hessian.py` derivation:
the well-conditioned `D_H(A)` chain used by `dsfun.D_of`, the pair-entropy
table and its marginals, the Mills-ratio log-derivatives, every `quadT_box`
Hessian coefficient, the `I_s` integrand change of variables (`nu = x +
gamma z`, rationalized through `lambda = (1-c^2)/(1+c^2)`) together with its
conditional-density normalizer, and the equivalence of the two printed forms
of (e:PP.pair.repeat), which holds exactly when `P_star = psi(1-q)/2 +
I(0)`. All 29 pass (`logs/sympy_audit.log`); a deliberately wrong identity
is rejected with a nonzero residual.

## Verifier falsifiability battery

`independent_checks/verifier_mutation_battery.py` runs the published
verifiers on pristine and corrupted canonical certificates, with corrupted
copies re-serialized by the repo's own canonical writer. On the attested
runtime all pristine certificates are accepted and every corruption is
rejected (`logs/mutation_battery.log`). The rejections occur at the
certificates' self-hash identity chain, which is the designed tamper
evidence: stored-data corruption cannot reach the mathematical layer without
rebuilding the whole hash tree. Falsifiability of the mathematical layer
itself is demonstrated separately by the Lean kernel battery (eight
corruption classes rejected by arithmetic, no hashes involved) and by the
portable star-interior checker, which rejects slack-exceeding corruption at
the inequality level.

## Finite-size satisfiability scan

`independent_checks/capacity_enumeration.py` estimates
`P(exists J in {-1,+1}^N : GJ >= 0)` by exact, budget-censored frontier
search (validated 40/40 against brute-force enumeration at N=12) for
N = 20, 24, 28, 32 across alpha in [0.70, 1.00] (26 cells; one N = 32
instance hit the node budget and is reported as censored). The
satisfiability probability decays through the neighborhood of the certified
alpha* = 0.833078599...0.833078600, with the p = 1/2 crossing at effective
ratio M/N of roughly 0.85-0.89 for N <= 28 and roughly 0.82-0.87 at N = 32,
above alpha* and moving toward it as N grows, as the known finite-size
corrections require; results in `logs/capacity_scan.jsonl`. This is a
consistency check, not evidence at the precision of the certificates.

## Manuscript spot-checks

A targeted line-pass of the manuscript's hardest quantitative paragraphs
against the certificate artifacts, using only numbers measured independently
this session. The star-interior display: the printed matrix enclosure
contains all four mpmath-recomputed integrals; the printed
`det A >= 0.003059813871082545`, ball radius `> 0.01503919829455`, required
radius `< 0.012096`, and clearance `> 0.002943` agree digit-for-digit with
the fraction-derived bounds of the portable checker. The stage-2 bisection
"to side 10^-3" is stage 2's own `MIN_SIDE = 1/1000` (stage 1 uses 1/500).
The polar-matching constants all derive from the Region-I star policy:
effective radii (0.012, 0.008, 0.005) are (T_LONG, T_MID, T_CORE); the cone
thresholds 0.1479995 and 0.4199995 are WEDGE_HALF - 0.012 - 5e-7 and
CONE_MID - 0.030 - 5e-7 for WEDGE_HALF = 0.16, CONE_MID = 0.45; and the
angular shrinkages 0.012 and 0.030 exceed the band chunk half-widths
0.016/2 and 0.05/2 used by the Region-I schedule. The mean-value machinery
matches its artifacts: 59 k-grid records and 16 ell-prime records in the
block3bc aux manifest, with `K = 21/2`, `I_third_bound`, and
`max_abs_I_second` present as the paper describes. The interval
decomposition of the lower bound matches the exact-rational spans pinned in
the Lean manifest, including the strict overlaps between the value grids,
the derivative lanes, the central lane, and the near-one block.

## Provenance

Public/private tree identity was re-established with today's fresh clones:
all 503 files common to `public:verification/` and
`private:perceptron/verification/` are byte-identical (SHA-256 per file);
the 14 private-only files are exactly the curated-out release and
orchestration machinery and its tests, per the release policy.

The Lean extraction consumed `results/huang_region1.json` with SHA-256
`f3cd141614b07276...` (the exact canonical certificate file bound by the
release receipts) and the three block3bc replay lanes; input hashes, output
hashes, and counts are pinned in `lean_skeleton/generated/manifest.json`.
All runs used fresh clones made today from the public GitHub repository.
This bundle lives at the repository root, outside `perceptron/`, because the
release receipts bind the exact byte tree of that subtree; nothing inside
`perceptron/` or `perceptron_release/` was modified.
