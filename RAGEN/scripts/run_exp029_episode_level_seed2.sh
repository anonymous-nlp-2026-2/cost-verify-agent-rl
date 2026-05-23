#!/bin/bash
# exp029 episode-level training — seed 2
# Clean /tmp to prevent disk full
rm -rf /tmp/fast_downward_* /tmp/ray/* /tmp/tmp* 2>/dev/null
echo "Cleaned /tmp, available space: $(df -h /tmp | tail -1 | awk '{print $4}')"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd ./RAGEN
export WANDB_PROJECT=cost-verify-agent-rl
python train.py --config-name _alfworld_exp029_episode_level_seed2
