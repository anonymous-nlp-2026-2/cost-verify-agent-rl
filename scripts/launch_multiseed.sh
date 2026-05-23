#!/bin/bash
set -euo pipefail

# Multi-seed launcher for exp029 (episode-level GRPO, full context window)
# Usage: bash launch_multiseed.sh <seed> [gpu_ids]
# Example: bash launch_multiseed.sh 2 "0,1"

SEED=${1:?Usage: launch_multiseed.sh <seed> [gpu_ids]}
GPU_IDS=${2:-"0,1"}

echo "=== [PRE-FLIGHT] exp029 multi-seed: seed=${SEED}, GPUs=${GPU_IDS} ==="

# --- Model ---
MODEL="./checkpoints/qwen3_exp021_ragen_sft_v4"
if [ ! -f "$MODEL/config.json" ]; then
    echo "[ERROR] Model not found: $MODEL"
    exit 1
fi
echo "[OK] Model: $MODEL"

# --- Disk ---
echo "[PRE-FLIGHT] Disk space:"
df -h /data/ | tail -1
AVAIL=$(df --output=avail /data/ | tail -1 | tr -d ' ')
if [ "$AVAIL" -lt 52428800 ]; then
    echo "[ERROR] Less than 50GB available on /data/"
    exit 1
fi
echo "[OK] Disk space sufficient"

# --- GPU ---
echo "[PRE-FLIGHT] GPU status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
echo ""

# --- Config file ---
CONFIG_FILE="./RAGEN/config/_alfworld_exp029_episode_level.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config not found: $CONFIG_FILE"
    exit 1
fi
echo "[OK] Config file exists"

# --- Environment ---
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export TMPDIR=/data/tmp/ && mkdir -p $TMPDIR
export ALFWORLD_DATA=/data/alfworld
export HF_HOME=/data/.hf_cache
export RAY_memory_monitor_refresh_ms=0
export RAY_SYSTEM_MEMORY=210000000000
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export WANDB_PROJECT=cost-verify-agent-rl
export WANDB_NAME="exp029_multiseed_s${SEED}"
export WANDB_MODE=online
export WANDB_DIR=/data/wandb && mkdir -p $WANDB_DIR

# --- Output directory ---
CKPT_DIR="./checkpoints/exp029_multiseed/seed${SEED}/"
mkdir -p "$CKPT_DIR"
mkdir -p ~/runs/cost-verify-agent-rl/

# --- Clean /tmp ---
rm -rf /tmp/fast_downward_* /tmp/ray/* /tmp/tmp* 2>/dev/null || true
echo "[OK] Cleaned /tmp"

echo "=== [PRE-FLIGHT] All checks passed. Starting training (seed=${SEED})... ==="

cd ./RAGEN
python train.py --config-name _alfworld_exp029_episode_level \
    seed.train=${SEED} \
    seed.val=${SEED} \
    system.CUDA_VISIBLE_DEVICES="\"${GPU_IDS}\"" \
    trainer.experiment_name="exp029-multiseed-s${SEED}" \
    trainer.local_log_dir="\"${CKPT_DIR}\"" \
    trainer.default_local_dir="\"${CKPT_DIR}\"" \
    2>&1 | tee ~/runs/cost-verify-agent-rl/exp029_multiseed_s${SEED}.log

echo "EXIT_CODE=$?"
