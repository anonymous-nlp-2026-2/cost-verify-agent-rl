"""Smoke test for step_independent context mode."""
import sys
sys.path.insert(0, "./RAGEN")

import copy
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoTokenizer

config_dict = {
    "agent_proxy": {
        "context_window_mode": "step_independent",
        "max_context_window": -1,
        "step_independent_history_length": 3,
        "action_sep": "||",
        "max_actions_per_turn": 2,
        "enable_think": True,
        "use_turn_scores": False,
        "max_prompt_length": None,
        "reward_normalization": {
            "grouping": "state",
            "method": "identity",
        },
    },
    "enable_response_mask": False,
    "es_manager": {
        "train": {
            "env_groups": 1,
            "group_size": 2,
            "env_configs": {
                "tags": ["TestEnv"],
                "n_groups": [1],
            },
        },
    },
    "custom_envs": {
        "TestEnv": {
            "env_type": "frozen_lake",
            "max_actions_per_traj": 10,
            "max_tokens": 256,
            "env_config": None,
        },
    },
    "actor_rollout_ref": {
        "rollout": {
            "response_length": 256,
            "max_model_len": 4096,
        },
    },
}

config = OmegaConf.create(config_dict)
tokenizer = AutoTokenizer.from_pretrained("/data/.hf_cache/Qwen/Qwen2___5-1___5B")

def make_env_outputs():
    history = []
    for i in range(5):
        turn = {
            "state": f"You are at position {i}. Admissible actions: [go north, go south]",
            "anchor_obs": f"You are at position {i}. Admissible actions: [go north, go south]",
            "llm_response": f"<think>I should go north.</think> <answer>go north</answer>",
            "reward": 0.0 if i < 4 else 1.0,
            "actions_left": 10 - i,
        }
        history.append(turn)
    history.append({
        "state": "You reached the goal!",
        "anchor_obs": "You reached the goal!",
        "actions_left": 5,
    })
    return [
        {"env_id": 0, "group_id": 0, "tag": "TestEnv", "history": copy.deepcopy(history), "penalty": 0, "uid": "test-0"},
        {"env_id": 1, "group_id": 0, "tag": "TestEnv", "history": copy.deepcopy(history), "penalty": 0, "uid": "test-1"},
    ]

from ragen.llm_agent.ctx_manager import ContextManager

ctx = ContextManager(config=config, tokenizer=tokenizer)

# ============ Test 1: Training samples ============
print("=" * 60)
print("TEST 1: step_independent training samples")
print("=" * 60)

env_outputs = make_env_outputs()
result = ctx.get_lm_inputs(env_outputs, prepare_for_update=True)

n_samples = result.batch["input_ids"].shape[0]
seq_len = result.batch["input_ids"].shape[1]
print(f"  Samples: {n_samples} (expected 10 = 2 eps * 5 turns)")
assert n_samples == 10, f"Expected 10, got {n_samples}"

assert "anchor_obs" in result.non_tensor_batch, "anchor_obs missing"
assert len(result.non_tensor_batch["anchor_obs"]) == n_samples
print(f"  anchor_obs[0]: {result.non_tensor_batch['anchor_obs'][0][:50]}...")

loss_mask = result.batch["loss_mask"]
print(f"  Seq len: {seq_len}, Avg loss tokens: {loss_mask.sum(dim=-1).float().mean().item():.1f}")
print("  PASS\n")

# ============ Test 2: Inference samples ============
print("=" * 60)
print("TEST 2: step_independent inference samples")
print("=" * 60)

result_infer = ctx.get_lm_inputs(make_env_outputs(), prepare_for_update=False)
n_infer = result_infer.batch["input_ids"].shape[0]
print(f"  Inference samples: {n_infer} (expected 2)")
assert n_infer == 2, f"Expected 2, got {n_infer}"

first_prompt = tokenizer.decode(result_infer.batch["input_ids"][0], skip_special_tokens=True)
print(f"  Prompt tokens: {result_infer.batch['input_ids'].shape[1]}")
print(f"  Prompt tail: ...{first_prompt[-150:]}")
print("  PASS\n")

# ============ Test 3: Bounded history ============
print("=" * 60)
print("TEST 3: Bounded history (history_length=3)")
print("=" * 60)

sample_4_text = tokenizer.decode(result.batch["input_ids"][4], skip_special_tokens=True)
position_mentions = sample_4_text.count("position")
print(f"  Turn 5 has {position_mentions} 'position' mentions (bounded: should be <= 4)")
print("  PASS\n")

# ============ Test 4: vs single_turn ============
print("=" * 60)
print("TEST 4: step_independent vs single_turn")
print("=" * 60)

config_st = copy.deepcopy(config)
config_st.agent_proxy.context_window_mode = "single_turn"
config_st.agent_proxy.max_context_window = 3
ctx_st = ContextManager(config=config_st, tokenizer=tokenizer)
result_st = ctx_st.get_lm_inputs(make_env_outputs(), prepare_for_update=True)

print(f"  step_independent: {n_samples} samples, single_turn: {result_st.batch['input_ids'].shape[0]}")
assert result_st.batch['input_ids'].shape[0] == n_samples

has_anchor_si = "anchor_obs" in result.non_tensor_batch
has_anchor_st = "anchor_obs" in result_st.non_tensor_batch
print(f"  step_independent anchor_obs: {has_anchor_si}, single_turn: {has_anchor_st}")
assert has_anchor_si and not has_anchor_st
print("  PASS\n")

# ============ Test 5: vs full mode ============
print("=" * 60)
print("TEST 5: step_independent vs full mode")
print("=" * 60)

config_full = copy.deepcopy(config)
config_full.agent_proxy.context_window_mode = "full"
ctx_full = ContextManager(config=config_full, tokenizer=tokenizer)
result_full = ctx_full.get_lm_inputs(make_env_outputs(), prepare_for_update=True)

full_samples = result_full.batch["input_ids"].shape[0]
full_seq_len = result_full.batch["input_ids"].shape[1]
print(f"  full: {full_samples} samples, seq_len={full_seq_len}")
print(f"  step_independent: {n_samples} samples, seq_len={seq_len}")
assert n_samples > full_samples, "step_independent should have more samples"
assert seq_len < full_seq_len, "step_independent should have shorter sequences"
print(f"  {n_samples/full_samples:.0f}x more samples, {full_seq_len/seq_len:.1f}x shorter seqs")
print("  PASS\n")



# ============ Test 6: non_tensor_batch step-level fields ============
print("=" * 60)
print("TEST 6: non_tensor_batch contains is_terminal, is_action_valid, reward")
print("=" * 60)

env_outputs_t6 = make_env_outputs()
result_t6 = ctx.get_lm_inputs(env_outputs_t6, prepare_for_update=True)

assert 'is_terminal' in result_t6.non_tensor_batch, "is_terminal missing from non_tensor_batch"
assert 'is_action_valid' in result_t6.non_tensor_batch, "is_action_valid missing from non_tensor_batch"
assert 'reward' in result_t6.non_tensor_batch, "reward missing from non_tensor_batch"

n = result_t6.batch["input_ids"].shape[0]
assert len(result_t6.non_tensor_batch["is_terminal"]) == n, f"is_terminal length mismatch: {len(result_t6.non_tensor_batch['is_terminal'])} vs {n}"
assert len(result_t6.non_tensor_batch["is_action_valid"]) == n
assert len(result_t6.non_tensor_batch["reward"]) == n

# Each episode has 5 turns -> is_terminal should be True only for the last turn of each episode
is_terminal = result_t6.non_tensor_batch["is_terminal"]
# 2 episodes * 5 turns = 10 samples; turns 4 and 9 are terminal
for i in range(n):
    if (i + 1) % 5 == 0:  # last turn of each episode
        assert is_terminal[i] == 1.0, f"Step {i} should be terminal, got {is_terminal[i]}"
    else:
        assert is_terminal[i] == 0.0, f"Step {i} should NOT be terminal, got {is_terminal[i]}"

# is_action_valid should default to 1.0 (test data has no is_action_valid field)
is_valid = result_t6.non_tensor_batch["is_action_valid"]
for i in range(n):
    assert is_valid[i] == 1.0, f"is_action_valid[{i}] should be 1.0 (default), got {is_valid[i]}"

# reward should be consistent across turns of the same episode
rewards = result_t6.non_tensor_batch["reward"]
ep1_reward = rewards[0]
for i in range(5):
    assert rewards[i] == ep1_reward, f"reward[{i}] should match ep1 reward {ep1_reward}, got {rewards[i]}"
ep2_reward = rewards[5]
for i in range(5, 10):
    assert rewards[i] == ep2_reward, f"reward[{i}] should match ep2 reward {ep2_reward}, got {rewards[i]}"

print(f"  is_terminal: {list(is_terminal)}")
print(f"  is_action_valid: {list(is_valid[:3])}... (all 1.0)")
print(f"  reward: ep1={ep1_reward}, ep2={ep2_reward}")
print("  PASS\n")

print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
