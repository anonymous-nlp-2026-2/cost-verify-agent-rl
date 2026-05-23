#!/bin/bash
# Evaluate Qwen2.5-7B-Instruct on Sokoban with 128 trajectories
#
# Quick usage:
#   bash scripts/eval_qwen_7b_sokoban.sh
#
# Custom trajectories:
#   bash scripts/eval_qwen_7b_sokoban.sh 256
#
# Custom GPU:
#   bash scripts/eval_qwen_7b_sokoban.sh 128 1

set -e

# Configuration
NUM_TRAJECTORIES=${1:-128}
GPU_ID=${2:-"0"}

# Fixed configuration for Qwen 2.5 7B Instruct
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME="qwen-2.5-7b"

# Calculate groups (use 16 per group as standard)
GROUP_SIZE=16
ENV_GROUPS=$((NUM_TRAJECTORIES / GROUP_SIZE))
if [ $((NUM_TRAJECTORIES % GROUP_SIZE)) -ne 0 ]; then
    ENV_GROUPS=$(( (NUM_TRAJECTORIES + GROUP_SIZE - 1) / GROUP_SIZE ))
fi
ACTUAL_TRAJECTORIES=$((ENV_GROUPS * GROUP_SIZE))

# Output configuration
OUTPUT_DIR="outputs"
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}-sokoban-${ACTUAL_TRAJECTORIES}.jsonl"

mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Model:         ${MODEL_PATH}"
echo "Environment:   Sokoban"
echo "Trajectories:  ${ACTUAL_TRAJECTORIES}"
echo "Output:        ${OUTPUT_FILE}"
echo "=========================================="

python -m ragen.llm_agent.agent_proxy \
    --config-name eval \
    system.CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    model_path="${MODEL_PATH}" \
    es_manager.val.env_groups=${ENV_GROUPS} \
    es_manager.val.group_size=${GROUP_SIZE} \
    es_manager.val.env_configs.tags=["CoordSokoban"] \
    es_manager.val.env_configs.n_groups=[${ENV_GROUPS}] \
    output.dir="${OUTPUT_DIR}" \
    output.filename="${MODEL_NAME}-sokoban-${ACTUAL_TRAJECTORIES}.jsonl" \
    output.format=jsonl \
    output.append_timestamp=false

echo ""
echo "Done! Saved to: ${OUTPUT_FILE}"
