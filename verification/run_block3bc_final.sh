#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-/srv/aiwork/perceptron-verify-claude/venv/bin/python}"
WORKERS="${WORKERS:-3}"
RESULTS="$HERE/results"
AUX="$RESULTS/block3bc_aux"
REPLAY="$RESULTS/block3bc_replay"
LOG="$RESULTS/block3bc_final_pipeline.log"
LOCK="$RESULTS/.block3bc_final.lock"
SOURCE_LIST="$RESULTS/block3bc_final_source.sha256"

mkdir -p "$AUX" "$REPLAY"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another Block3bc final pipeline owns $LOCK" >&2
    exit 73
fi
exec > >(tee -a "$LOG") 2>&1

echo "START $(date -Is) workers=$WORKERS python=$PY"

check_health() {
    "$PY" -I -B - <<'PY'
import json
import os
import time
from pathlib import Path

path = Path("/var/local/compute_health.json")
age = time.time() - path.stat().st_mtime
if age > 120:
    raise SystemExit(f"compute health is stale ({age:.1f}s old)")
health = json.loads(path.read_text())
reasons = health.get("reasons") or []
pressure_reasons = [reason for reason in reasons
                    if not str(reason).startswith("trading_unit_not_active:")]
load1 = float(health.get("load1", 1e9))
psi_cpu = float(health.get("psi_cpu_some_avg60", 1e9))
memory = int(health.get("mem_avail_mb", 0))
if (pressure_reasons or psi_cpu >= 35
        or load1 >= 0.7 * (os.cpu_count() or 1) or memory <= 2000):
    raise SystemExit("compute pressure gate failed: " + repr(health))
print("HEALTH pressure-safe", "age_s", round(age, 1),
      "availability_alerts", reasons,
      "load1", health.get("load1"),
      "psi", health.get("psi_cpu_some_avg60"),
      health.get("psi_mem_some_avg60"), health.get("psi_io_some_avg60"))
PY
}

wait_for_health() {
    until check_health; do
        echo "WAIT pressure gate $(date -Is); retrying in 30s"
        sleep 30
    done
}

SOURCE_FILES=(
    block3bc_aux_generate.py
    block3bc_aux_verify.py
    block3bc.py
    block3bc_assemble.py
    block3bc_exact.py
    core.py
    dsfun.py
)

freeze_sources() {
    sha256sum "${SOURCE_FILES[@]}" > "$SOURCE_LIST"
    sha256sum -c "$SOURCE_LIST"
}

phase() {
    local name="$1"
    shift
    echo "PHASE $name START $(date -Is)"
    wait_for_health
    sha256sum -c "$SOURCE_LIST"
    nice -n 15 ionice -c3 "$@"
    sha256sum -c "$SOURCE_LIST"
    echo "PHASE $name PASS $(date -Is)"
}

wait_for_health
"$PY" -I -B - <<'PY'
import flint
import sys

assert sys.version_info[:3] == (3, 12, 3), sys.version
assert flint.__version__ == "0.9.0", flint.__version__
assert flint.__FLINT_VERSION__ == "3.6.0", flint.__FLINT_VERSION__
print("RUNTIME", sys.version.split()[0], flint.__version__,
      flint.__FLINT_VERSION__)
PY
freeze_sources

phase AUX_ELL "$PY" -B block3bc_aux_generate.py ell-prime --lane 0 --lanes 1 --workers "$WORKERS" --timeout-seconds 21600 --retries 2 --output "$AUX/ell_prime.lane-0-of-1.json"

phase AUX_K "$PY" -B block3bc_aux_generate.py k-grid --lane 0 --lanes 1 --workers "$WORKERS" --timeout-seconds 21600 --retries 2 --output "$AUX/k_grid.lane-0-of-1.json"

phase AUX_FINALIZE "$PY" -B block3bc_aux_generate.py finalize --ell-shard "$AUX/ell_prime.lane-0-of-1.json" --k-shard "$AUX/k_grid.lane-0-of-1.json" --k-run 21/2 --output "$AUX/manifest.json"

phase AUX_VERIFY "$PY" -B block3bc_aux_verify.py "$AUX/manifest.json"

for part in b_pos b_neg c; do
    phase "REPLAY_${part}" "$PY" -B block3bc.py replay --part "$part" --aux-manifest "$AUX/manifest.json" --lane 0 --lanes 1 --workers "$WORKERS" --timeout-seconds 21600 --retries 2 --output "$REPLAY/$part.lane-0-of-1.json"
done

phase ASSEMBLE "$PY" -B block3bc_assemble.py --aux-manifest "$AUX/manifest.json" --shard "$REPLAY/b_pos.lane-0-of-1.json" --shard "$REPLAY/b_neg.lane-0-of-1.json" --shard "$REPLAY/c.lane-0-of-1.json" --output "$RESULTS/block3bc_certificate.json"

phase VERIFY "$PY" -B block3bc_assemble.py --check-only --certificate "$RESULTS/block3bc_certificate.json"

{
    echo "ALL PASS"
    echo "completed=$(date -Is)"
    sha256sum "$RESULTS/block3bc_certificate.json"
} > "$RESULTS/BLOCK3BC_FINAL_ALL_PASS"
echo "ALL PASS $(date -Is)"
