#!/bin/bash
set -euo pipefail

echo "[PRE-FLIGHT] exp020 GRPO from exp019b SFT"

# 检查 checkpoint
CKPT="./checkpoints/exp019b_expert_sft_v3/final"
if [ ! -f "$CKPT/model.safetensors" ]; then
    echo "[ERROR] Checkpoint not found: $CKPT/model.safetensors"
    exit 1
fi
echo "[OK] Checkpoint: $(du -sh $CKPT | cut -f1)"

# 磁盘检查
echo "[PRE-FLIGHT] Disk space:"
df -h /data/ | tail -1
AVAIL=$(df --output=avail /data/ | tail -1 | tr -d ' ')
if [ "$AVAIL" -lt 52428800 ]; then
    echo "[ERROR] Less than 50GB available on /data/"
    exit 1
fi
echo "[OK] Disk space sufficient"

# GPU 检查
echo "[PRE-FLIGHT] GPU status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
echo ""

# 环境
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export TMPDIR=/data/tmp/ && mkdir -p $TMPDIR
export ALFWORLD_DATA=/data/alfworld
export RAY_memory_monitor_refresh_ms=0
export RAY_SYSTEM_MEMORY=210000000000
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1

# 输出目录
mkdir -p ./checkpoints/exp020/

cd ./RAGEN
python train.py --config-name _alfworld_exp018_sft_grpo \
  model_path=./checkpoints/exp019b_expert_sft_v3/final \
  trainer.experiment_name=exp020-alfworld-sft-grpo \
  trainer.local_log_dir=./checkpoints/exp020/ \
  trainer.default_local_dir=./checkpoints/exp020/ \
  es_manager.val.env_groups=4 \
  'es_manager.val.env_configs.n_groups=[4]'

echo "EXIT_CODE=$?"
