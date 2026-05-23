#!/bin/bash
# Gamma-sweep launch script for discount factor ablation
# Usage: ./launch_gsweep.sh <gamma_label> <seed>
# gamma_label: 050, 080, 095, 100

GAMMA_LABEL=$1
SEED=$2

if [ -z "$GAMMA_LABEL" ] || [ -z "$SEED" ]; then
    echo "Usage: $0 <gamma_label: 050|080|095|100> <seed: 1-5>"
    exit 1
fi

case "$GAMMA_LABEL" in
    050|080|095|100) ;;
    *) echo "Error: gamma_label must be one of: 050, 080, 095, 100"; exit 1 ;;
esac

if [ "$SEED" -lt 1 ] || [ "$SEED" -gt 5 ]; then
    echo "Error: seed must be 1-5"
    exit 1
fi

TRAIN_SEED=$((SEED * 10000))

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export TMPDIR=/data/tmp
export WANDB_DIR=/data/wandb
export WANDB_TMPDIR=/data/tmp

cd ./RAGEN

CONFIG="_alfworld_gamma_sweep_g${GAMMA_LABEL}_h5"
EXP_NAME="gsweep-g${GAMMA_LABEL}-h5-seed${SEED}"

LOG_PATH="./logs/${EXP_NAME}.log"
mkdir -p "$(dirname "$LOG_PATH")"

echo "=== Gamma-Sweep Launch ==="
echo "  gamma_label: $GAMMA_LABEL"
echo "  seed: $SEED (train_seed=$TRAIN_SEED)"
echo "  config: $CONFIG"
echo "  experiment: $EXP_NAME"
echo "  log: $LOG_PATH"
echo "=========================="

python train.py --config-name "$CONFIG" \
    seed.train=$TRAIN_SEED \
    trainer.experiment_name="$EXP_NAME" \
    2>&1 | tee "$LOG_PATH"

echo "[$(date)] Training complete for gamma=$GAMMA_LABEL seed=$SEED"
