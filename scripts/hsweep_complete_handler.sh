#!/bin/bash
# Handler called when an h-sweep run completes
# Usage: hsweep_complete_handler.sh <h_value> <seed>

H_VALUE=$1
SEED=$2

if [ -z "$H_VALUE" ] || [ -z "$SEED" ]; then
    echo "Usage: $0 <h_value> <seed>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="./logs"
PROGRESS_LOG="${LOG_DIR}/hsweep_progress.log"
RUN_LOG="./logs/hsweep-h${H_VALUE}-seed${SEED}.log"

mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Collect final metrics from run log
FINAL_METRICS="no_log_found"
if [ -f "$RUN_LOG" ]; then
    FINAL_METRICS=$(grep -oP "val/success_rate['\"]?:\s*\K[0-9.]+" "$RUN_LOG" | tail -1)
    if [ -z "$FINAL_METRICS" ]; then
        FINAL_METRICS=$(grep -i "success" "$RUN_LOG" | tail -3 | tr '\n' ' ')
    fi
    if [ -z "$FINAL_METRICS" ]; then
        FINAL_METRICS="metrics_not_parsed"
    fi
fi

# Log completion
echo "[${TIMESTAMP}] COMPLETED h=${H_VALUE} seed=${SEED} | metrics: ${FINAL_METRICS}" >> "$PROGRESS_LOG"
echo "Logged completion: h=${H_VALUE} seed=${SEED}"

# Launch next run and capture output
echo "Calling auto_next_hsweep.sh ${H_VALUE} ${SEED}..."
NEXT_OUTPUT=$(bash "${SCRIPT_DIR}/auto_next_hsweep.sh" "${H_VALUE}" "${SEED}" 2>&1)
echo "$NEXT_OUTPUT"
echo "[${TIMESTAMP}] ${NEXT_OUTPUT}" >> "$PROGRESS_LOG"
