#!/bin/bash
set -euo pipefail

echo "=== [PRE-FLIGHT] exp026: step-independent GRPO from SFT v4 ==="

# --- Checkpoint ---
CKPT="./checkpoints/qwen3_exp021_ragen_sft_v4"
if [ ! -f "$CKPT/model.safetensors" ]; then
    echo "[ERROR] Checkpoint not found: $CKPT/model.safetensors"
    exit 1
fi
echo "[OK] Checkpoint: $(du -sh $CKPT | cut -f1)"

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
CONFIG_FILE="./RAGEN/config/_alfworld_exp026_step_independent_grpo.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config not found: $CONFIG_FILE"
    exit 1
fi
echo "[OK] Config file exists"

# --- Environment ---
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export TMPDIR=/data/tmp/ && mkdir -p $TMPDIR
export ALFWORLD_DATA=/data/alfworld
export RAY_memory_monitor_refresh_ms=0
export RAY_SYSTEM_MEMORY=210000000000
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1

# --- Output directory ---
mkdir -p ./checkpoints/exp026/

echo "=== [PRE-FLIGHT] All checks passed. Starting training... ==="

cd ./RAGEN
python train.py --config-name _alfworld_exp026_step_independent_grpo

echo "EXIT_CODE=$?"
