#!/bin/bash
# Evaluate Qwen model on Sokoban environment
#
# Usage:
#   bash scripts/eval_qwen_sokoban.sh <model_version> [num_trajectories] [gpu_id]
#
# Examples:
#   bash scripts/eval_qwen_sokoban.sh 3B 128 0
#   bash scripts/eval_qwen_sokoban.sh 7B 256 1
#   bash scripts/eval_qwen_sokoban.sh 14B 128 0,1

set -e

# Parse arguments
MODEL_VERSION=${1:-"14B"}
NUM_TRAJECTORIES=${2:-128}
GPU_ID=${3:-"0"}

# Calculate env_groups and group_size
# Strategy: Use group_size=16 (common default), calculate groups accordingly
GROUP_SIZE=16
ENV_GROUPS=$((NUM_TRAJECTORIES / GROUP_SIZE))

if [ $((NUM_TRAJECTORIES % GROUP_SIZE)) -ne 0 ]; then
    echo "Warning: NUM_TRAJECTORIES ($NUM_TRAJECTORIES) is not divisible by GROUP_SIZE ($GROUP_SIZE)"
    echo "Rounding up to $((ENV_GROUPS * GROUP_SIZE)) trajectories"
    ENV_GROUPS=$(( (NUM_TRAJECTORIES + GROUP_SIZE - 1) / GROUP_SIZE ))
fi

ACTUAL_TRAJECTORIES=$((ENV_GROUPS * GROUP_SIZE))

# Model configuration
MODEL_PATH="Qwen/Qwen2.5-${MODEL_VERSION}-Instruct"
OUTPUT_DIR="outputs"
OUTPUT_FILE="${OUTPUT_DIR}/qwen-${MODEL_VERSION}-sokoban-${ACTUAL_TRAJECTORIES}.jsonl"

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "RAGEN Evaluation Configuration"
echo "=========================================="
echo "Model:              ${MODEL_PATH}"
echo "Environment:        Sokoban (CoordSokoban)"
echo "Trajectories:       ${ACTUAL_TRAJECTORIES} (${ENV_GROUPS} groups × ${GROUP_SIZE} size)"
echo "GPU:                ${GPU_ID}"
echo "Output:             ${OUTPUT_FILE}"
echo "=========================================="
echo ""

# Run evaluation
python -m ragen.llm_agent.agent_proxy \
    --config-name eval \
    system.CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    model_path="${MODEL_PATH}" \
    es_manager.val.env_groups=${ENV_GROUPS} \
    es_manager.val.group_size=${GROUP_SIZE} \
    es_manager.val.env_configs.tags=["CoordSokoban"] \
    es_manager.val.env_configs.n_groups=[${ENV_GROUPS}] \
    output.dir="${OUTPUT_DIR}" \
    output.filename="qwen-${MODEL_VERSION}-sokoban-${ACTUAL_TRAJECTORIES}.jsonl" \
    output.format=jsonl \
    output.append_timestamp=false

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_FILE}"
echo "=========================================="
