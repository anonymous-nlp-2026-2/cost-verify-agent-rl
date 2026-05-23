#!/bin/bash
# exp029 re-eval on valid_unseen (134 tasks, OOD) — D064 alignment
# Checkpoint: global_step_15 (best, 28.1% val success on old split)
# Expected: val_before_train gives OOD success rate, then exit after 1 training step
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd ./RAGEN
ALFWORLD_DATA=/data/alfworld CUDA_VISIBLE_DEVICES=0,1 python train.py --config-name _alfworld_exp029_reeval_valid_unseen
