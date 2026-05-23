#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export TMPDIR=/data/tmp/ && mkdir -p $TMPDIR
export ALFWORLD_DATA=/data/alfworld
export RAY_memory_monitor_refresh_ms=0
export RAY_SYSTEM_MEMORY=210000000000
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1
cd ./RAGEN
set +e
python train.py --config-name _alfworld_exp018_sft_grpo \
  trainer.experiment_name=exp018-alfworld-sft-grpo \
  trainer.local_log_dir=./checkpoints/exp018/ \
  trainer.default_local_dir=./checkpoints/exp018/ \
  es_manager.val.env_groups=4 \
  'es_manager.val.env_configs.n_groups=[4]'
EXIT_CODE=$?
set -e
RAY_LOG_BACKUP="/data/logs/exp018_ray_$(date +%s)"
mkdir -p "$RAY_LOG_BACKUP"
cp -r /tmp/ray/session_latest/logs/ "$RAY_LOG_BACKUP" 2>/dev/null || true
exit $EXIT_CODE
