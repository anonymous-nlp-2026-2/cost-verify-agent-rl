#!/bin/bash
set -euo pipefail

echo "=== [DRY-RUN] exp026: step-independent GRPO (2 steps) ==="

# --- Checkpoint ---
CKPT="./checkpoints/qwen3_exp021_ragen_sft_v4"
if [ ! -f "$CKPT/model.safetensors" ]; then
    echo "[ERROR] Checkpoint not found: $CKPT/model.safetensors"
    exit 1
fi
echo "[OK] Checkpoint exists"

# --- GPU ---
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

# --- Environment ---
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export TMPDIR=/data/tmp/ && mkdir -p $TMPDIR
export ALFWORLD_DATA=/data/alfworld
export RAY_memory_monitor_refresh_ms=0
export RAY_SYSTEM_MEMORY=210000000000
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1

mkdir -p ./checkpoints/exp026_dryrun/

echo "=== Starting dry-run... ==="

cd ./RAGEN
python train.py --config-name _alfworld_exp026_step_independent_grpo \
  trainer.total_training_steps=2 \
  trainer.val_before_train=True \
  trainer.test_freq=1 \
  trainer.save_freq=100 \
  trainer.local_log_dir=./checkpoints/exp026_dryrun/ \
  trainer.default_local_dir=./checkpoints/exp026_dryrun/ \
  trainer.experiment_name=exp026-dryrun \
  "trainer.logger=['console']" \
  es_manager.train.env_groups=2 \
  'es_manager.train.env_configs.n_groups=[2]' \
  es_manager.val.env_groups=4 \
  'es_manager.val.env_configs.n_groups=[4]'

echo "DRY_RUN_EXIT_CODE=$?"
