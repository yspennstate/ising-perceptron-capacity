#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-/srv/aiwork/perceptron-verify-claude/venv/bin/python}"
WORKERS="${WORKERS:-3}"
RESULTS="$HERE/results"
LOG="$RESULTS/huang_final_pipeline.log"
LOCK="$RESULTS/.huang_final.lock"
SOURCE_LIST="$RESULTS/huang_final_source.sha256"

export HUANG_GRID_N=2700
export HUANG_REGION1_REPLAY_WORKERS="$WORKERS"

mkdir -p "$RESULTS"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another Huang final pipeline owns $LOCK" >&2
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
    block1_gardner.py
    block2_near_one.py
    block3a_grid.py
    block3a_run.py
    block3a_singlerun.py
    block3a_assemble.py
    block3bc.py
    block3bc_assemble.py
    block3bc_aux_generate.py
    block3bc_aux_verify.py
    block3bc_exact.py
    core.py
    dsfun.py
    huang_hessian.py
    huang_np.py
    huang_region1.py
    huang_region1_verify.py
    huang_star_interior.py
    huang_sweep.py
    huang_sweep2.py
    huang_sweep_verify.py
    huanggrid.py
    verify_all.py
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

phase TESTS "$PY" -B -m unittest discover -s tests -p 'test_*.py'
phase BLOCK3A_VERIFY "$PY" -B block3a_singlerun.py verify "$RESULTS/block3a_certificate.json"
phase REGION1_STRUCTURE "$PY" -B huang_region1_verify.py "$RESULTS/huang_region1.json"

if [[ -f "$RESULTS/huang_star_interior.json" ]]; then
    phase STAR_INTERIOR_VERIFY "$PY" -B huang_star_interior.py verify --certificate "$RESULTS/huang_star_interior.json"
else
    phase STAR_INTERIOR_GENERATE "$PY" -B huang_star_interior.py generate --output "$RESULTS/huang_star_interior.json"
fi

if [[ -f "$RESULTS/huang_sweep.json" ]]; then
    phase SWEEP1_VERIFY "$PY" -B -c 'import huang_sweep_verify as v; v.verify_certificate("results/huang_sweep.json", 1)'
else
    phase SWEEP1_GENERATE "$PY" -B huang_sweep.py "$WORKERS" 48
fi

if [[ -f "$RESULTS/huang_sweep2.json" ]]; then
    phase SWEEP2_VERIFY "$PY" -B -c 'import huang_sweep_verify as v; v.verify_certificate("results/huang_sweep2.json", 2)'
else
    phase SWEEP2_GENERATE "$PY" -B huang_sweep2.py "$WORKERS"
fi

phase BUNDLE_BUILD "$PY" -B huang_sweep_verify.py bundle --output "$RESULTS/huang_bundle.json" --star-interior "$RESULTS/huang_star_interior.json" --region1 "$RESULTS/huang_region1.json" --sweep1 "$RESULTS/huang_sweep.json" --sweep2 "$RESULTS/huang_sweep2.json"
phase BUNDLE_VERIFY "$PY" -B huang_sweep_verify.py verify "$RESULTS/huang_bundle.json"
if [[ ! -f "$RESULTS/block3bc_certificate.json" ]]; then
    echo "HUANG ARTIFACTS COMPLETE; Block3bc certificate is still pending"
    exit 75
fi
phase BLOCK3BC_VERIFY "$PY" -B block3bc_assemble.py --check-only --certificate "$RESULTS/block3bc_certificate.json"

# verify_all performs the independent, packet-by-packet Region-I numerical
# replay in addition to the other eight theorem outcomes.
phase VERIFY_ALL "$PY" -B verify_all.py

{
    echo "ALL CERTIFICATES PASS"
    echo "completed=$(date -Is)"
    sha256sum "$RESULTS/huang_star_interior.json" "$RESULTS/huang_region1.json" "$RESULTS/huang_sweep.json" "$RESULTS/huang_sweep2.json" "$RESULTS/huang_bundle.json" "$RESULTS/block3a_certificate.json" "$RESULTS/block3bc_certificate.json"
} > "$RESULTS/FINAL_ALL_PASS"
echo "ALL CERTIFICATES PASS $(date -Is)"
