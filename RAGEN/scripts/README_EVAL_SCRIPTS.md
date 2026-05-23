# Evaluation Scripts

Quick reference for running evaluations on Sokoban environment.

## Quick Start

### Evaluate Qwen 2.5 7B (Default)

```bash
# 128 trajectories, GPU 0
bash scripts/eval_qwen_7b_sokoban.sh

# 256 trajectories
bash scripts/eval_qwen_7b_sokoban.sh 256

# Custom GPU
bash scripts/eval_qwen_7b_sokoban.sh 128 1
```

**Output:** `outputs/qwen-2.5-7b-sokoban-128.jsonl`

### Evaluate Any Qwen Model

```bash
# General usage
bash scripts/eval_qwen_sokoban.sh <version> [num_traj] [gpu]

# Examples
bash scripts/eval_qwen_sokoban.sh 3B 128 0
bash scripts/eval_qwen_sokoban.sh 7B 256 1
bash scripts/eval_qwen_sokoban.sh 14B 128 0,1  # Multi-GPU
```

**Output:** `outputs/qwen-<version>-sokoban-<num>.jsonl`

### Batch Evaluation (All Models)

```bash
bash scripts/eval_batch.sh
```

Evaluates all Qwen models (0.5B, 1.5B, 3B, 7B, 14B) sequentially.

**Output:** Multiple files in `outputs/`

## Scripts Overview

### `eval_qwen_7b_sokoban.sh`
- **Purpose:** Quick eval for Qwen 2.5 7B Instruct
- **Args:** `[num_trajectories] [gpu_id]`
- **Default:** 128 trajectories on GPU 0
- **Output:** `outputs/qwen-2.5-7b-sokoban-{N}.jsonl`

### `eval_qwen_sokoban.sh`
- **Purpose:** Flexible eval for any Qwen model
- **Args:** `<model_version> [num_trajectories] [gpu_id]`
- **Versions:** 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
- **Output:** `outputs/qwen-{version}-sokoban-{N}.jsonl`

### `eval_batch.sh`
- **Purpose:** Evaluate multiple models in sequence
- **Edit:** Modify `MODELS` array to customize model list
- **Output:** One JSONL per model in `outputs/`

## Output Format

All scripts output OpenAI-compatible JSONL:

```json
{
  "custom_id": "traj_0",
  "messages": [
    {"role": "user", "content": "Grid state..."},
    {"role": "assistant", "content": "<think>...</think><ans>up</ans>"},
    {"role": "user", "content": "New grid... (reward: 0.0)"}
  ],
  "metadata": {
    "env_id": 0,
    "success": true,
    "total_reward": 1.0,
    "num_turns": 5
  }
}
```

## Common Configurations

### Trajectories
- **128:** Good for quick eval (8 groups × 16)
- **256:** Standard eval (16 groups × 16)
- **512:** Thorough eval (32 groups × 16)

The script auto-rounds to nearest multiple of 16.

### GPU Settings
- Single GPU: `0` or `1`
- Multi-GPU: `0,1` or `0,1,2,3`

### Model Sizes
| Model | VRAM | Recommended GPU |
|-------|------|-----------------|
| 0.5B  | ~2GB | Any |
| 1.5B  | ~4GB | RTX 3090 |
| 3B    | ~8GB | RTX 3090 |
| 7B    | ~16GB | A100 40GB |
| 14B   | ~32GB | A100 80GB |

## Custom Evaluation

For full control, use the Python command directly:

```bash
python -m ragen.llm_agent.agent_proxy \
    --config-name eval \
    model_path="Qwen/Qwen2.5-7B-Instruct" \
    system.CUDA_VISIBLE_DEVICES="0" \
    es_manager.val.env_groups=8 \
    es_manager.val.group_size=16 \
    output.dir="outputs" \
    output.filename="custom-name.jsonl" \
    output.format=jsonl \
    output.append_timestamp=false
```

## Troubleshooting

**Out of Memory:**
```bash
# Reduce context length in eval config
python -m ragen.llm_agent.agent_proxy --config-name eval \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.response_length=128
```

**Model not found:**
- Ensure model is downloaded or accessible via HuggingFace
- Check path format: `Qwen/Qwen2.5-{size}B-Instruct`

**Slow evaluation:**
- Use fewer trajectories for testing: `bash scripts/eval_qwen_7b_sokoban.sh 32`
- Enable greedy decoding: add `actor_rollout_ref.rollout.val_kwargs.temperature=0`
