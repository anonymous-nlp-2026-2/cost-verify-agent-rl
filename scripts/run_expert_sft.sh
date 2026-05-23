#!/bin/bash
# Expert SFT pipeline: data prep → training
set -euo pipefail

CONDA_BASE="/root/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate base

cd .

EXPERT_INPUT="/data/expert_trajectories/expert_trajectories.jsonl"
SFT_DIR="/data/expert_sft"
CKPT_DIR="./checkpoints/exp017_expert_sft"
LOG_FILE="./logs/exp017_expert_sft.log"

# ── Pre-flight: check GPU availability ──
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
if [ "${GPU_MEM:-0}" -gt 1000 ]; then
    echo "[ERROR] GPU 0 has ${GPU_MEM}MiB in use. Aborting."
    exit 1
fi

# ── Pre-flight: disk check ──
df -h /data | tail -1

# ── Step 1: Check data ──
LINES=$(wc -l < "$EXPERT_INPUT")
echo "[INFO] Expert trajectories: $LINES lines"

# ── Step 2: Prepare data ──
echo "[INFO] Running data preparation..."
python scripts/prepare_expert_sft_data.py \
    --input "$EXPERT_INPUT" \
    --output-dir "$SFT_DIR" \
    --min-turns 2 \
    --max-turns 40 \
    --val-frac 0.05

TRAIN_DATA="$SFT_DIR/expert_sft_train.jsonl"
TRAIN_LINES=$(wc -l < "$TRAIN_DATA")
echo "[INFO] Training data: $TRAIN_LINES examples"

# ── Step 3: Train ──
echo "[INFO] Starting SFT training on GPU 0..."
mkdir -p "$(dirname "$LOG_FILE")"
CUDA_VISIBLE_DEVICES=0 python scripts/sft_train.py \
    --data_path "$TRAIN_DATA" \
    --output_dir "$CKPT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --max_seq_length 4096 \
    2>&1 | tee "$LOG_FILE"

echo "[INFO] Done. Checkpoint at $CKPT_DIR/final/"
