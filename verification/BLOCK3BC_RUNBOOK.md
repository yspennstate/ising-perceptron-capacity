# Exact Block 3b/c replay

This is the production sequence for the repaired negative-lambda proof. Old
text logs are not evidence. The canonical certificate must replay fresh JSON
jobs from one byte-identical source tree and one attested arithmetic runtime.

## Safety and source gates

1. Read `/srv/aiwork/AI_BOX_RULES.md` and the current compute-health file.
   Do not launch while the box is in genuine pressure or trading distress.
2. Work only in the claimed proof lane and use the `aiwork` identity that owns
   it. Do not modify trading services or another proof lane.
3. Freeze source bytes before the first job. Any later edit to
   `block3bc_aux_generate.py`, `block3bc_aux_verify.py`, `block3bc.py`,
   `block3bc_assemble.py`, `block3bc_exact.py`, `core.py`, or `dsfun.py`
   invalidates the affected artifacts.
4. Use Python 3.12.3, python-flint 0.9.0, and FLINT 3.6.0 throughout.
5. Run at nice 15 with idle I/O and at most three proof workers on Azure.
6. A shard is its JSON file plus the adjacent `.records` directory. Preserve
   both; the summary JSON is not independently sufficient.

## Canonical all-box run

Run from the frozen lane. For the current lane:

```bash
D=/srv/aiwork/codex-proof-finisher-20260712
PY=/srv/aiwork/perceptron-verify-claude/venv/bin/python
cd "$D"
mkdir -p results/block3bc_aux results/block3bc_replay
```

Generate the 16 ell-prime jobs and 59 K-grid jobs:

```bash
nice -n 15 ionice -c3 "$PY" -B block3bc_aux_generate.py ell-prime \
  --lane 0 --lanes 1 --workers 3 --timeout-seconds 21600 --retries 2 \
  --output results/block3bc_aux/ell_prime.lane-0-of-1.json

nice -n 15 ionice -c3 "$PY" -B block3bc_aux_generate.py k-grid \
  --lane 0 --lanes 1 --workers 3 --timeout-seconds 21600 --retries 2 \
  --output results/block3bc_aux/k_grid.lane-0-of-1.json
```

Build and verify the auxiliary manifest at the authoritative
`K_run = 21/2 = 10.5`:

```bash
"$PY" -B block3bc_aux_generate.py finalize \
  --ell-shard results/block3bc_aux/ell_prime.lane-0-of-1.json \
  --k-shard results/block3bc_aux/k_grid.lane-0-of-1.json \
  --k-run 21/2 --output results/block3bc_aux/manifest.json
"$PY" -B block3bc_aux_verify.py results/block3bc_aux/manifest.json
```

Replay all three proof parts. The manifest determines the exact schedules:

```bash
for part in b_pos b_neg c; do
  nice -n 15 ionice -c3 "$PY" -B block3bc.py replay \
    --part "$part" --aux-manifest results/block3bc_aux/manifest.json \
    --lane 0 --lanes 1 --workers 3 \
    --timeout-seconds 21600 --retries 2 \
    --output "results/block3bc_replay/$part.lane-0-of-1.json"
done
```

Expected top-cell counts are 24 `b_pos`, 331 `b_neg`, and 16 `c`. The
`b_neg` partition is exactly `[-19/100,-3/100]` in 331 cells of width
`4/8275`.

## Assembly and exact replay

```bash
"$PY" -B block3bc_assemble.py \
  --aux-manifest results/block3bc_aux/manifest.json \
  --shard results/block3bc_replay/b_pos.lane-0-of-1.json \
  --shard results/block3bc_replay/b_neg.lane-0-of-1.json \
  --shard results/block3bc_replay/c.lane-0-of-1.json \
  --output results/block3bc_certificate.json

"$PY" -B block3bc_assemble.py --check-only \
  --certificate results/block3bc_certificate.json
```

Do not claim completion unless the certificate reports `ALL PASS`,
`k_run = 21/2`, all 24/331/16 top cells, and exact replay succeeds after
rereading all 446 auxiliary/replay job JSON files and their `.records` trees.
Final theorem completion additionally requires the five-artifact Huang bundle,
Block3a v2 replay, and nine PASS rows from `verify_all.py`.
