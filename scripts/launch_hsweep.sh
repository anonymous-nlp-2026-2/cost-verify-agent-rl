#!/bin/bash
# H-sweep launch script for context window ablation
# Usage: ./launch_hsweep.sh <h_value> <seed>
# h_value: 2, 5, 10, 20, full
# seed: 1, 2, 3, 4, 5

H_VALUE=$1
SEED=$2

if [ -z "$H_VALUE" ] || [ -z "$SEED" ]; then
    echo "Usage: $0 <h_value: 2|5|10|20|full> <seed: 1-5>"
    exit 1
fi

# Validate h_value
case "$H_VALUE" in
    2|5|10|20|full) ;;
    *) echo "Error: h_value must be one of: 2, 5, 10, 20, full"; exit 1 ;;
esac

# Validate seed
if [ "$SEED" -lt 1 ] || [ "$SEED" -gt 5 ]; then
    echo "Error: seed must be 1-5"
    exit 1
fi

# Compute train seed (non-overlapping ranges: seed*10000)
TRAIN_SEED=$((SEED * 10000))

# Environment setup
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export TMPDIR=/data/tmp
export WANDB_DIR=/data/wandb
export WANDB_TMPDIR=/data/tmp

cd ./RAGEN

# Config and experiment name
CONFIG="_alfworld_hsweep_h${H_VALUE}"
EXP_NAME="hsweep-h${H_VALUE}-seed${SEED}"

# Log path
LOG_PATH="./logs/${EXP_NAME}.log"
mkdir -p "$(dirname "$LOG_PATH")"

echo "=== H-Sweep Launch ==="
echo "  h_value: $H_VALUE"
echo "  seed: $SEED (train_seed=$TRAIN_SEED)"
echo "  config: $CONFIG"
echo "  experiment: $EXP_NAME"
echo "  log: $LOG_PATH"
echo "======================"

# Launch with seed and experiment_name override
python train.py --config-name "$CONFIG" \
    seed.train=$TRAIN_SEED \
    trainer.experiment_name="$EXP_NAME" \
    2>&1 | tee "$LOG_PATH"

# Auto-chain: trigger next h-sweep run after training completes
echo "[$(date)] Training complete for h=$H_VALUE seed=$SEED, triggering auto_next..."
bash "$(dirname "$0")/auto_next_hsweep.sh" "$H_VALUE" "$SEED"
