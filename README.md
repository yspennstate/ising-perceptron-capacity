# The Ising perceptron capacity: a source-bound interval verification

This repository contains the manuscript, proof programs, mathematical tests,
canonical certificates, and raw record trees for the outstanding numerical
hypotheses in the Ding--Sun lower bound and Huang upper bound for the Ising
perceptron capacity. All theorem-level comparisons use Arb ball arithmetic
through `python-flint`.

## Status and scope

The manuscript presents a new computer-assisted proof of the 1989
Krauth--Mezard capacity conjecture at `kappa = 0`:

```text
M_N / N -> alpha_*,    alpha_* in [0.833078599, 0.833078600].
```

The included certificates pass their source-bound verifiers and have received
a separate second-system internal audit. The proof has not yet been
independently reproduced or peer reviewed. The result does not
address nonzero margin or the general free-energy/TAP problem.

The paper is available as [paper/main.pdf](paper/main.pdf), with source in
[paper/main.tex](paper/main.tex). A proof-architecture and certificate
dependency guide is available in [THEORY.md](THEORY.md).

## AI contributions

Most of the work was done by two AI systems. Fable 5 (Anthropic) developed the
main mathematical architecture, wrote the bulk of the proof code, and drafted
the initial manuscript. Codex (OpenAI) performed a separate internal re-audit;
applied and validated the confirmed Region-I full-circle normalization fix;
validated and landed the repaired Block3a and Huang closures; completed the
fresh Block3b/c and final closure; hardened the source-bound verification
boundary; assembled the reproducibility artifacts; and audited the final
manuscript and public-release readiness. The author supervised the runs,
reviewed the mathematics, and accepts responsibility for the result.

## What is certified

- The Ding--Sun parameter rectangle and corrected near-one estimate.
- The full Ding--Sun Condition 1.2: the 247-cell value grid, positive and
  negative derivative ranges, strict curvature at `lambda = 0`, and the
  analytic one-sided derivative condition at `lambda = 1`. The middle
  interval is replayed at `K = 21/2`.
- Huang Condition 1.3 after compactification into the moment body: two bulk
  sweeps, a constructive interior neighborhood of the fixed point, and a
  gapless Region-I ray-concavity cover.
- The Region-I verifier reconstructs every fixed-weight localization witness,
  localized covariance majorant, inverse quadratic form, and curvature packet
  from frozen producer bytes. Stored signs alone are not accepted.

The helper `verification/huang_np.py` only proposes rational duals, tilts,
anchors, and boxes. Every accepted inequality is recomputed in ball
arithmetic, and every certificate is bound to producer-source and
arithmetic-core hashes.

## Exact runtime

Verification requires:

- Python 3.12.3
- python-flint 0.9.0
- FLINT 3.6.0

`numpy` and `scipy` are needed by proposal and generation code, but do not
replace any Arb proof comparison.

Byte-identical certificate regeneration requires this attested runtime. The
unit tests and source-bound certificate verifiers are otherwise portable; the
complete private suite was also reproduced on Windows with 140/140 tests
passing.

The curated public test discovery reports 90 mathematical tests. The private
140-test suite adds 50 tests for release, orchestration, and machine-specific
dispatch tools that are not part of this public proof tree. This difference is
intentional and does not remove any theorem verifier, certificate, or raw
proof record.

## Release attestation

The final Stage-A run and Stage-B package bind private source commit
`ef57bc257000dd8cd63ec02e644e33cd49133858`:

- Stage-A receipt file SHA-256:
  `b0e6be09918b8334fae2596fcb216dbbd09495d3023da082a39fa0f951a7bf4b`
- Stage-A receipt payload SHA-256:
  `56b1ce54bb574b3542779c4b18c7acddaa0fb6af1634f174088f4769d421b334`
- Stage-B exact validated archive SHA-256:
  `fecadcd2f09a0ed0cef6ba8d2dcb4f3996bc51f591fbbb73b8906dba6ad33ad5`
- Stage-B handoff SHA-256:
  `8c1d155c70267d2572c67a1003ab89c99fde255451c108ef13ad1ba1294bb58f`
- Committed 13-page PDF SHA-256:
  `1c4d17912c48e3b3c6300ea08378d9d20fd9a2053fa7e74e6bbe9307a3818fd1`

The receipt binds the full private `perceptron/` subtree: 561 files and
90,350,421 bytes, including working source papers and internal release notes
that are not republished here for licensing and operational-provenance
reasons. This public repository contains the complete theorem programs,
mathematical tests, canonical certificates, and raw proof records; the
canonical certificates also bind their producer and arithmetic-core bytes.
The receipt and handoff themselves retain machine-local provenance and are
therefore kept in the private release archive; the hashes above are their
public identifiers. The receipt detects run mixups and accidental corruption;
it is not a cryptographic signature or independent reproduction.

## Verify the included closure

From `verification/`, using the pinned Python executable:

```sh
export HUANG_GRID_N=2700
export HUANG_REGION1_REPLAY_WORKERS=3
/path/to/pinned/python -B -m unittest discover -s tests -p 'test_*.py'
/path/to/pinned/python -B update_paper_counts.py --check
/path/to/pinned/python -B verify_all.py
```

A successful theorem-level replay prints nine `PASS` rows followed by
`ALL CERTIFICATES PASS`. `verify_all.py` numerically replays all 1,404
Region-I jobs; the shorter structural verifier is only a diagnostic.

The Block3a source-bound verifier can also be run directly:

```sh
/path/to/pinned/python -B block3a_singlerun.py verify \
  results/block3a_certificate.json
```

To regenerate and replay Region I directly, choose a worker count appropriate
for the host:

```sh
/path/to/pinned/python -B huang_region1.py 3
/path/to/pinned/python -B huang_region1_verify.py \
  --full-replay --workers 3 results/huang_region1.json
```

The canonical certificates and their complete raw record trees are committed
under `verification/results/`.

## Prior conditional results

- Jian Ding and Nike Sun, *Capacity lower bound for the Ising perceptron*,
  Probability Theory and Related Fields 193 (2025), 627--715,
  [journal version](https://doi.org/10.1007/s00440-025-01364-x) and
  [arXiv:1809.07742](https://arxiv.org/abs/1809.07742).
- Brice Huang, *Capacity threshold for the Ising perceptron*,
  [arXiv:2404.18902](https://arxiv.org/abs/2404.18902).
- Dylan J. Altschuler and Konstantin Tikhomirov, *A note on the capacity of the
  binary perceptron*, [arXiv:2401.15092](https://arxiv.org/abs/2401.15092).
- Shuta Nakajima and Nike Sun, *Sharp threshold sequence and universality for
  Ising perceptron models*,
  [arXiv:2204.03469](https://arxiv.org/abs/2204.03469).

## Layout

```text
THEORY.md              proof architecture and code/certificate dependency map
paper/                  manuscript source and rendered PDF
verification/           proof producers, independent verifiers, and tests
verification/results/   canonical certificates and raw record trees
```

## License

Repository-authored material is released under the [MIT License](LICENSE).
The cited papers are not redistributed here; use the bibliography and links in
the manuscript to obtain them from their publishers or arXiv.
