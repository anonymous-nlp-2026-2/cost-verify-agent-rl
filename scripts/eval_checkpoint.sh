#!/bin/bash
# eval_checkpoint.sh — Quick 16-episode val eval for exp027 early checkpoints.
# Usage:
#   bash scripts/eval_checkpoint.sh ./checkpoints/exp027/global_step_5
#   bash scripts/eval_checkpoint.sh ./checkpoints/exp027/global_step_5 [GPU_ID]
#
# Output: JSON file in /data/eval_results/ with success_rate, action_is_valid_rate, etc.
# MVP threshold: val success_rate >= 20% (SFT baseline exp021: 6.25%)
#
# GPU note: Training uses cuda:0,1 with vLLM gpu_memory_utilization=0.4.
#   Default eval uses gpu_memory_utilization=0.35 on a single GPU, which should coexist.
#   If OOM, either: (1) wait for training val interval, or (2) lower --gpu_memory_utilization.

set -euo pipefail

CKPT_PATH="${1:?Usage: $0 <checkpoint_path> [gpu_id]}"
GPU_ID="${2:-0}"

# Environment setup
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export ALFWORLD_DATA=/data/alfworld
export TMPDIR=/data/tmp/ && mkdir -p "$TMPDIR"

# Pre-flight: check checkpoint exists
if [ ! -d "$CKPT_PATH" ]; then
    echo "[ERROR] Checkpoint dir not found: $CKPT_PATH"
    exit 1
fi

# Pre-flight: check GPU memory
echo "[PRE-FLIGHT] GPU ${GPU_ID} status:"
nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free --format=csv,noheader -i "$GPU_ID"

cd .

python scripts/eval_checkpoint.py \
    --checkpoint_path "$CKPT_PATH" \
    --num_episodes 16 \
    --temperature 0.4 \
    --max_steps 50 \
    --gpu "$GPU_ID" \
    --gpu_memory_utilization 0.35 \
    --output_dir /data/eval_results/

echo ""
echo "=== MVP CHECK: val success_rate >= 20% means PASS ==="
