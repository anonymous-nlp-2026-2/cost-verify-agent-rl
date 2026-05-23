#!/bin/bash
# exp019_expert_sft_v2: second-stage SFT from exp008_sft on format-fixed expert data.
# Usage: bash scripts/run_exp019.sh
set -euo pipefail

CONDA_BASE="/root/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate base

cd .

export HF_HOME="/data/.hf_cache"

CKPT_BASE="./checkpoints/exp008_sft/final"
TRAIN_DATA="/data/expert_sft_v2/expert_sft_train.jsonl"
VAL_DATA="/data/expert_sft_v2/expert_sft_val.jsonl"
OUTPUT_DIR="./checkpoints/exp019_expert_sft_v2"
LOG_FILE="./logs/exp019_expert_sft_v2.log"

# ── Pre-flight checks (C002) ──
echo "[PRE-FLIGHT] Checking base checkpoint..."
if [ ! -f "$CKPT_BASE/model.safetensors" ]; then
    echo "[ERROR] Base checkpoint not found at $CKPT_BASE"
    exit 1
fi
echo "[OK] Checkpoint exists: $(du -sh $CKPT_BASE | cut -f1)"

echo "[PRE-FLIGHT] Checking training data..."
if [ ! -f "$TRAIN_DATA" ]; then
    echo "[ERROR] Training data not found at $TRAIN_DATA"
    exit 1
fi
TRAIN_LINES=$(wc -l < "$TRAIN_DATA")
echo "[OK] Training data: $TRAIN_LINES examples"

echo "[PRE-FLIGHT] Checking disk..."
df -h /data | tail -1

echo "[PRE-FLIGHT] Checking GPU 0..."
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
if [ "${GPU_MEM:-0}" -gt 1000 ]; then
    echo "[ERROR] GPU 0 has ${GPU_MEM}MiB in use. Aborting."
    exit 1
fi
echo "[OK] GPU 0 free (${GPU_MEM}MiB used)"

echo "[PRE-FLIGHT] Config: model=$CKPT_BASE lr=5e-6 epochs=2 batch=4 grad_accum=4 max_seq=4096"

# ── Train ──
mkdir -p "$(dirname "$LOG_FILE")"
CUDA_VISIBLE_DEVICES=0 python scripts/sft_train_v2.py \
    --model_path "$CKPT_BASE" \
    --data_path "$TRAIN_DATA" \
    --val_path "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --max_seq_length 4096 \
    2>&1 | tee "$LOG_FILE"

echo "[INFO] Done. Checkpoint at $OUTPUT_DIR/final/"
