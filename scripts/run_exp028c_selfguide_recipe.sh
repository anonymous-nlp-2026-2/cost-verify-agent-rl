#!/bin/bash
set -euo pipefail

# exp028c: Self-Guide Recipe Alignment

echo "=== [PRE-FLIGHT] exp028c: selfguide recipe ==="

# --- Model ---
MODEL="./checkpoints/qwen3_exp021_ragen_sft_v4"
if [ ! -f "$MODEL/config.json" ]; then
    echo "[ERROR] Model not found: $MODEL"
    exit 1
fi
echo "[OK] Model: SFT checkpoint (qwen3_exp021_ragen_sft_v4)"

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
CONFIG_FILE="./RAGEN/config/_alfworld_exp028c_selfguide_recipe.yaml"
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
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=cost-verify-agent-rl
export WANDB_NAME=exp028c_selfguide_recipe
export WANDB_MODE=online

# --- Output directory ---
mkdir -p ./checkpoints/exp028c/
mkdir -p ~/runs/cost-verify-agent-rl/

echo "=== [PRE-FLIGHT] All checks passed. Starting training... ==="

cd ./RAGEN
python train.py --config-name _alfworld_exp028c_selfguide_recipe 2>&1 | tee ~/runs/cost-verify-agent-rl/exp028c_selfguide_recipe.log

echo "EXIT_CODE=$?"
