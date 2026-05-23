#!/bin/bash
set -euo pipefail

echo "=== exp013f 启动 $(date) ==="

# 环境
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

# 目录和临时文件
export TMPDIR=/data/tmp/ && mkdir -p $TMPDIR
export ALFWORLD_DATA=/data/alfworld

# Ray 修复: 禁用 MemoryMonitor（阈值 224GB > cgroup 220GB）
export RAY_memory_monitor_refresh_ms=0

# Ray 修复: 限制可见内存为 210GB（cgroup 220GB - 10GB 安全余量）
export RAY_SYSTEM_MEMORY=210000000000

# 完整堆栈
export HYDRA_FULL_ERROR=1

# GPU
export CUDA_VISIBLE_DEVICES=0,1

cd ./RAGEN

echo "=== 环境变量 ==="
echo "TMPDIR=$TMPDIR"
echo "RAY_memory_monitor_refresh_ms=$RAY_memory_monitor_refresh_ms"
echo "RAY_SYSTEM_MEMORY=$RAY_SYSTEM_MEMORY"
echo "HYDRA_FULL_ERROR=$HYDRA_FULL_ERROR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 训练（前台，允许失败以便执行 log 备份）
set +e
python train.py --config-name _alfworld_exp013b_dense_reward \
  trainer.experiment_name=exp013f-alfworld-dense-reward \
  trainer.local_log_dir=./checkpoints/exp013f/ \
  trainer.default_local_dir=./checkpoints/exp013f/ \
  es_manager.val.env_groups=4 \
  'es_manager.val.env_configs.n_groups=[4]'
EXIT_CODE=$?
set -e

# 崩溃后自动保存 Ray logs（在 ray stop 之前！）
echo "=== Train exit code: $EXIT_CODE ==="
RAY_LOG_BACKUP="/data/logs/exp013f_ray_$(date +%s)"
mkdir -p "$RAY_LOG_BACKUP"
cp -r /tmp/ray/session_latest/logs/ "$RAY_LOG_BACKUP" 2>/dev/null && \
  echo "Ray logs saved to $RAY_LOG_BACKUP" || \
  echo "No Ray logs found to save"

exit $EXIT_CODE
