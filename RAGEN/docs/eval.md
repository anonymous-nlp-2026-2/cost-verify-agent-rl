# RAGEN Evaluation Guide

This guide explains how to evaluate trained RAGEN models and configure output formats.

## Quick Start

Evaluate a model using the default configuration:

```bash
python -m ragen.llm_agent.agent_proxy --config-name eval
```

Or use a specific config:

```bash
python -m ragen.llm_agent.agent_proxy --config-name _2_sokoban
```

## Configuration File

Evaluation settings are configured in `config/eval.yaml`. Key sections:

### Model Configuration

```yaml
model_path: Qwen/Qwen2.5-3B-Instruct

lora:
  rank: 0        # Set to 0 to disable LoRA; set to > 0 for LoRA-finetuned models
  alpha: 64
  target_modules: all-linear
```

### Rollout Settings

```yaml
actor_rollout_ref:
  rollout:
    max_model_len: 3600        # Max context length
    response_length: 400       # Max tokens per response
    val_kwargs:
      do_sample: True          # Enable sampling
      temperature: 0.5         # Sampling temperature
      top_p: 1.0              # Nucleus sampling
      top_k: -1               # Top-k sampling (-1 = disabled)
```

### Agent Proxy Settings

```yaml
agent_proxy:
  context_window_mode: "full"  # "full" | "limited_multi_turn" | "single_turn"
  max_context_window: -1       # Number of previous turns to retain (-1 = unlimited)
  max_turn: 5                  # Maximum interaction turns
  enable_think: True           # Enable <think>...</think> reasoning
```

**Context Window Modes:**
- `full`: Keep all previous turns in context
- `limited_multi_turn`: Keep only the last `max_context_window` turns
- `single_turn`: Only current state, no history

### Environment Settings

```yaml
es_manager:
  val:
    env_groups: 32           # Number of environment groups
    group_size: 16           # Environments per group (total = groups × size)
    env_configs:
      tags: ["CoordSokoban"]  # Environment type(s)
      n_groups: [32]          # Groups per environment type
```

**Available environment tags** are defined in `config/envs.yaml` under `custom_envs`.

### Output Configuration

```yaml
output:
  dir: results/eval              # Output directory
  filename: val_rollouts.pkl     # Output filename
  format: pkl                    # pkl | jsonl
  append_timestamp: true         # Add timestamp to filename
  save_jsonl_backup: false       # Save JSONL backup when format=pkl
  save_pkl_backup: false         # Save PKL backup when format=jsonl
  keep_batch_keys: null          # Filter batch keys (null = keep all)
  keep_non_tensor_keys: null     # Filter non-tensor keys (null = keep all)
  keep_meta_info: true           # Include metadata
```

## Output Formats

### PKL Format (Default)

Binary format containing the full `DataProto` object with tensors, metadata, and trajectories.

```yaml
output:
  format: pkl
  filename: val_rollouts.pkl
```

**Visualization:**
```bash
python scripts/visualize.py --rollout_path results/eval/
```

### JSONL Format (OpenAI-Compatible)

Human-readable JSONL where each line is a trajectory in OpenAI message format.

```yaml
output:
  format: jsonl
  filename: trajectories.jsonl
```

**JSONL structure:**
```json
{
  "custom_id": "traj_0",
  "messages": [
    {"role": "user", "content": "Initial state..."},
    {"role": "assistant", "content": "<think>...</think><ans>action</ans>"},
    {"role": "user", "content": "Next state... (reward: 1.0)"},
    ...
  ],
  "metadata": {
    "env_id": 0,
    "group_id": 0,
    "success": true,
    "total_reward": 5.0,
    "num_turns": 3,
    "entropy": 2.45,
    "n_tokens": 128
  }
}
```

### Dual Output

Save both formats simultaneously:

```yaml
output:
  format: pkl
  save_jsonl_backup: true  # Also save JSONL
```

Or:

```yaml
output:
  format: jsonl
  save_pkl_backup: true    # Also save PKL
```

## Converting Existing PKL Files

Convert existing PKL rollouts to JSONL:

```bash
python scripts/convert_to_jsonl.py \
  --input results/eval/val_rollouts_20260413_123456.pkl \
  --output trajectories.jsonl
```

Auto-generate output filename:

```bash
python scripts/convert_to_jsonl.py --input results/eval/val_rollouts_*.pkl
# Creates: val_rollouts_*.jsonl in the same directory
```

## Advanced Usage

### Override Config from Command Line

```bash
python -m ragen.llm_agent.agent_proxy --config-name eval \
  model_path=path/to/checkpoint \
  actor_rollout_ref.rollout.temperature=0.7 \
  output.format=jsonl \
  es_manager.val.env_groups=64
```

### Custom Evaluation Seeds

Control randomness for reproducibility:

```yaml
seed:
  val: 123  # Validation seed
```

### GPU Configuration

```yaml
system:
  CUDA_VISIBLE_DEVICES: "0"  # GPU device(s)

actor_rollout_ref:
  rollout:
    tensor_model_parallel_size: 1  # Number of GPUs for tensor parallelism
    gpu_memory_utilization: 0.9    # Max GPU memory fraction
```

### Filtering Output Data

Reduce file size by filtering keys:

```yaml
output:
  keep_batch_keys: ["rm_scores", "responses"]  # Only keep these tensor keys
  keep_non_tensor_keys: ["history", "metrics"] # Only keep these non-tensor keys
```

Set to `null` to keep all keys.

## Metrics

After evaluation, metrics are displayed in the terminal:

```
rollout rewards: 0.85
metrics:
  CoordSokoban/success: 0.78
  CoordSokoban/num_actions: 4.2
  CoordSokoban/pass@16: 0.92
```

**Common metrics:**
- `{env}/success`: Success rate (0-1)
- `{env}/num_actions`: Average actions per trajectory
- `{env}/pass@k`: At least one success in group of k rollouts
- `episodic_return`: Cumulative reward

## Troubleshooting

**Out of memory:**
```yaml
actor_rollout_ref:
  rollout:
    max_model_len: 2048          # Reduce context length
    response_length: 128         # Reduce response length
    gpu_memory_utilization: 0.7  # Lower memory usage
```

**Evaluation too slow:**
- Reduce `es_manager.val.env_groups` or `group_size`
- Use `temperature: 0` for greedy decoding (faster)
- Enable `enforce_eager: False` for compiled mode (if compatible)

**JSONL parsing errors:**
- Ensure `history` data is serializable
- Check for special characters in state/response strings
- Use `save_pkl_backup: true` to preserve original data

## Related Documentation

- [Main README](../README.md) - General RAGEN overview
- [Rollout Filtering Guide](guide_rollout_filtering.md) - Training-time filtering
- [V1 README](readme_v1.md) - Legacy evaluation instructions
- [WebShop Evaluation](experiment_webshop_release.md) - WebShop-specific setup
