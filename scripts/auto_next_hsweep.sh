#!/bin/bash
# Auto-advance h-sweep to next run in sequence
# Usage: auto_next_hsweep.sh <completed_h> <completed_seed>

COMPLETED_H=$1
COMPLETED_SEED=$2

if [ -z "$COMPLETED_H" ] || [ -z "$COMPLETED_SEED" ]; then
    echo "Usage: $0 <completed_h_value> <completed_seed>"
    exit 1
fi

SEQUENCE="2:1 2:2 10:1 5:1 20:1 full:1 2:3 5:2 5:3 10:2 10:3 20:2 20:3 full:2 full:3"

FOUND=0
NEXT_H=""
NEXT_SEED=""

for entry in $SEQUENCE; do
    if [ $FOUND -eq 1 ]; then
        NEXT_H="${entry%%:*}"
        NEXT_SEED="${entry##*:}"
        break
    fi
    if [ "$entry" = "${COMPLETED_H}:${COMPLETED_SEED}" ]; then
        FOUND=1
    fi
done

if [ $FOUND -eq 0 ]; then
    echo "ERROR: ${COMPLETED_H}:${COMPLETED_SEED} not found in sequence"
    exit 1
fi

if [ -z "$NEXT_H" ]; then
    echo "ALL DONE - h-sweep sequence complete"
    exit 0
fi

echo "Advancing: ${COMPLETED_H}:${COMPLETED_SEED} -> ${NEXT_H}:${NEXT_SEED}"

TMUX_SESSION="hsweep_h${NEXT_H}_s${NEXT_SEED}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Enable per-sample advantage logging for h=2 seed 2
if [ "$NEXT_H" = "2" ] && [ "$NEXT_SEED" = "2" ]; then
    export LOG_PER_SAMPLE_ADV=1
    export ADV_LOG_PATH="./logs/hsweep_h2_seed2/advantage_log.jsonl"
    mkdir -p ./logs/hsweep_h2_seed2
fi
tmux new-session -d -s "$TMUX_SESSION" \
    "LOG_PER_SAMPLE_ADV=${LOG_PER_SAMPLE_ADV:-0} ADV_LOG_PATH=${ADV_LOG_PATH:-} bash ${SCRIPT_DIR}/launch_hsweep.sh ${NEXT_H} ${NEXT_SEED}"

echo "Started tmux session: $TMUX_SESSION (h=${NEXT_H}, seed=${NEXT_SEED})"
