#!/bin/bash
# Batch evaluation script for multiple Qwen models on Sokoban
#
# Usage:
#   bash scripts/eval_batch.sh

set -e

OUTPUT_DIR="outputs"
mkdir -p ${OUTPUT_DIR}

# Configuration
NUM_TRAJECTORIES=128
GROUP_SIZE=16
ENV_GROUPS=$((NUM_TRAJECTORIES / GROUP_SIZE))  # 8 groups
GPU_ID="0"

echo "=========================================="
echo "Batch Evaluation on Sokoban"
echo "Trajectories per model: ${NUM_TRAJECTORIES}"
echo "=========================================="
echo ""

# List of models to evaluate
MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct:qwen-2.5-0.5b"
    "Qwen/Qwen2.5-1.5B-Instruct:qwen-2.5-1.5b"
    "Qwen/Qwen2.5-3B-Instruct:qwen-2.5-3b"
    "Qwen/Qwen2.5-7B-Instruct:qwen-2.5-7b"
    "Qwen/Qwen2.5-14B-Instruct:qwen-2.5-14b"
)

for model_config in "${MODELS[@]}"; do
    IFS=':' read -r model_path model_name <<< "$model_config"

    output_file="${OUTPUT_DIR}/${model_name}-sokoban-${NUM_TRAJECTORIES}.jsonl"

    echo ">>> Evaluating: ${model_name}"
    echo "    Path: ${model_path}"
    echo "    Output: ${output_file}"
    echo ""

    python -m ragen.llm_agent.agent_proxy \
        --config-name eval \
        system.CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        model_path="${model_path}" \
        es_manager.val.env_groups=${ENV_GROUPS} \
        es_manager.val.group_size=${GROUP_SIZE} \
        es_manager.val.env_configs.tags=["CoordSokoban"] \
        es_manager.val.env_configs.n_groups=[${ENV_GROUPS}] \
        output.dir="${OUTPUT_DIR}" \
        output.filename="${model_name}-sokoban-${NUM_TRAJECTORIES}.jsonl" \
        output.format=jsonl \
        output.append_timestamp=false

    echo ""
    echo "✓ Completed: ${model_name}"
    echo "=========================================="
    echo ""
done

echo "All evaluations completed!"
echo "Results in: ${OUTPUT_DIR}/"
ls -lh ${OUTPUT_DIR}/*.jsonl
