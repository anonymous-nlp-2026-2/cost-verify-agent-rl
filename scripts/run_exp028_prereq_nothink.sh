#!/bin/bash
set -euo pipefail

echo "=== [PRE-FLIGHT] exp028_prereq_nothink: Zero-shot eval (thinking=OFF) ==="

# --- Environment (activate first so python3 is available) ---
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export TMPDIR=/data/tmp/ && mkdir -p $TMPDIR
export ALFWORLD_DATA=/data/alfworld
export RAY_memory_monitor_refresh_ms=0
export RAY_SYSTEM_MEMORY=210000000000
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1

# --- Model ---
MODEL="./models/Qwen3-1.7B"
if [ ! -f "$MODEL/config.json" ]; then
    echo "[ERROR] Model not found: $MODEL"
    exit 1
fi
echo "[OK] Model: Qwen3-1.7B"

# --- Verify thinking is OFF ---
echo "[PRE-FLIGHT] Checking tokenizer chat template..."
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL')
out = tok.apply_chat_template([{'role':'user','content':'test'}], add_generation_prompt=True, tokenize=False)
assert '<think>\n\n</think>' in out, 'ERROR: thinking not disabled in template!'
print('[OK] Thinking mode disabled (empty <think> block confirmed)')
"

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
CONFIG_FILE="./RAGEN/config/_alfworld_exp028_prereq_nothink.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config not found: $CONFIG_FILE"
    exit 1
fi
echo "[OK] Config file exists"

# --- W&B ---
export WANDB_PROJECT=cost-verify-agent-rl
export WANDB_NAME=exp028_prereq_nothink

# --- Output directory ---
mkdir -p ./checkpoints/exp028_prereq_nothink/

echo "=== [PRE-FLIGHT] All checks passed. Starting zero-shot eval (thinking=OFF)... ==="

cd ./RAGEN
python train.py --config-name _alfworld_exp028_prereq_nothink 2>&1 | tee ~/runs/cost-verify-agent-rl/exp028_prereq_nothink.log

echo "EXIT_CODE=$?"
