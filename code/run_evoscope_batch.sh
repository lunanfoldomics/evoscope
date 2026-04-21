#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_evoscope_batch.sh [NUM_RUNS] [EPOCHS] [WIDTH] [HEIGHT] [OUTDIR]
#
# Example:
#   bash run_evoscope_batch.sh 100 150 60 40 runs

NUM_RUNS="${1:-100}"
EPOCHS="${2:-150}"
WIDTH="${3:-60}"
HEIGHT="${4:-40}"
OUTDIR="${5:-runs}"

# Absolute path to evoscope.py
EVOSCOPE_PY="$(pwd)/evoscope.py"

if [[ ! -f "$EVOSCOPE_PY" ]]; then
    echo "ERROR: evoscope.py not found in $(pwd)"
    exit 1
fi

mkdir -p "$OUTDIR"

for i in $(seq 1 "$NUM_RUNS"); do
    RUN_ID=$(printf "run_%03d" "$i")
    RUN_DIR="${OUTDIR}/${RUN_ID}"
    mkdir -p "$RUN_DIR"

    # seed progression; change if you prefer a different scheme
    SEED=$((24 + i))

    echo "============================================================"
    echo "Running ${RUN_ID}"
    echo "  seed   = ${SEED}"
    echo "  epochs = ${EPOCHS}"
    echo "  width  = ${WIDTH}"
    echo "  height = ${HEIGHT}"
    echo "  outdir = ${RUN_DIR}"
    echo "============================================================"

    # Run INSIDE the run directory so that snapshots/ and csv files are created there
    (
        cd "$RUN_DIR"
        python "$EVOSCOPE_PY" \
            --width "$WIDTH" \
            --height "$HEIGHT" \
            --seed "$SEED" \
            --epochs "$EPOCHS" \
            --plot n
    )

    # Basic checks
    if [[ -f "${RUN_DIR}/global_genes.csv" ]]; then
        echo "  [OK] global_genes.csv"
    else
        echo "  [WARN] global_genes.csv not found in ${RUN_DIR}"
    fi

    if [[ -f "${RUN_DIR}/cluster_genes.csv" ]]; then
        echo "  [OK] cluster_genes.csv"
    else
        echo "  [WARN] cluster_genes.csv not found in ${RUN_DIR}"
    fi

    if [[ -d "${RUN_DIR}/snapshots" ]]; then
        N_SNAPS=$(find "${RUN_DIR}/snapshots" -type f | wc -l | awk '{print $1}')
        echo "  [OK] snapshots/ present (${N_SNAPS} files)"
    else
        echo "  [WARN] snapshots/ not found in ${RUN_DIR}"
    fi
done

echo
echo "Done. All runs written under: ${OUTDIR}"