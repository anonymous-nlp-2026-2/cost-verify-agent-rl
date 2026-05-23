#!/bin/bash
set -euo pipefail

# exp028_prereq: Zero-shot eval — 验证"SFT 是毒药"假设
# 用 Qwen3-1.7B 原始权重跑 ALFWorld eval（val_only 模式）
# 对比 SFT checkpoint (exp027 step_0 val: success=6.25%)

echo "=== [PRE-FLIGHT] exp028_prereq: Zero-shot eval ==="

# --- Model ---
MODEL="./models/Qwen3-1.7B"
if [ ! -f "$MODEL/config.json" ]; then
    echo "[ERROR] Model not found: $MODEL"
    exit 1
fi
echo "[OK] Model: Qwen3-1.7B"

# --- Disk ---
echo "[PRE-FLIGHT] Disk space:"
df -h /data/ | tail -1
AVAIL=$(df --output=avail /data/ | tail -1 | tr -d ' ')
if [ "$AVAIL" -lt 10485760 ]; then
    echo "[ERROR] Less than 10GB available on /data/"
    exit 1
fi
echo "[OK] Disk space sufficient"

# --- GPU ---
echo "[PRE-FLIGHT] GPU status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
echo ""

# --- Config file ---
CONFIG_FILE="./RAGEN/config/_alfworld_exp028_prereq_zeroshot.yaml"
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

# --- W&B ---
export WANDB_PROJECT=cost-verify-agent-rl
export WANDB_NAME=exp028_prereq_zeroshot

# --- Output directory ---
mkdir -p ./checkpoints/exp028_prereq/

echo "=== [PRE-FLIGHT] All checks passed. Starting zero-shot eval... ==="

cd ./RAGEN
python train.py --config-name _alfworld_exp028_prereq_zeroshot 2>&1 | tee ~/runs/cost-verify-agent-rl/exp028_prereq_zeroshot.log

echo "EXIT_CODE=$?"
