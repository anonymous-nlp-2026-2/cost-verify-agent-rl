"""Unit tests for step-level reward assignment and advantage computation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RAGEN'))

import numpy as np
import torch
from ragen.trainer.step_rewards import (
    assign_step_rewards,
    compute_step_discounted_returns,
    compute_episode_returns,
)
from ragen.trainer.core_algos import compute_grpo_step_advantage


def test_assign_step_rewards_success_episode():
    """5-step episode, success (reward=10), step 2 invalid action."""
    episode_rewards = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    episode_ids = np.array(["ep1", "ep1", "ep1", "ep1", "ep1"])
    is_action_valid = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
    is_terminal = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

    rewards = assign_step_rewards(
        episode_rewards, episode_ids, is_action_valid, is_terminal,
        invalid_action_penalty_coef=0.1,
    )

    assert rewards[0] == 0.0, f"Step 0 (valid, non-terminal): {rewards[0]}"
    assert rewards[1] == -0.1, f"Step 1 (invalid, non-terminal): {rewards[1]}"
    assert rewards[2] == 0.0, f"Step 2 (valid, non-terminal): {rewards[2]}"
    assert rewards[4] == 10.0, f"Step 4 (valid, terminal): {rewards[4]}"
    print("PASS: test_assign_step_rewards_success_episode")


def test_assign_step_rewards_fail_episode():
    """3-step episode, failure (reward=0), step 1 invalid."""
    episode_rewards = np.array([0.0, 0.0, 0.0])
    episode_ids = np.array(["ep2", "ep2", "ep2"])
    is_action_valid = np.array([1.0, 0.0, 1.0])
    is_terminal = np.array([0.0, 0.0, 1.0])

    rewards = assign_step_rewards(
        episode_rewards, episode_ids, is_action_valid, is_terminal,
        invalid_action_penalty_coef=0.1,
    )

    assert rewards[0] == 0.0
    assert rewards[1] == -0.1
    assert rewards[2] == 0.0  # terminal but episode reward = 0
    print("PASS: test_assign_step_rewards_fail_episode")


def test_discounted_returns_simple():
    """5-step success episode, gamma=0.95."""
    step_rewards = np.array([0.0, -0.1, 0.0, 0.0, 10.0], dtype=np.float32)
    episode_ids = np.array(["ep1", "ep1", "ep1", "ep1", "ep1"])
    gamma = 0.95

    returns = compute_step_discounted_returns(step_rewards, episode_ids, gamma)

    # Manual backward:
    # R4 = 10.0
    # R3 = 0 + 0.95 * 10.0 = 9.5
    # R2 = 0 + 0.95 * 9.5 = 9.025
    # R1 = -0.1 + 0.95 * 9.025 = 8.47375
    # R0 = 0 + 0.95 * 8.47375 = 8.0500625

    assert abs(returns[4].item() - 10.0) < 1e-4, f"R4={returns[4]}"
    assert abs(returns[3].item() - 9.5) < 1e-4, f"R3={returns[3]}"
    assert abs(returns[2].item() - 9.025) < 1e-4, f"R2={returns[2]}"
    assert abs(returns[1].item() - 8.47375) < 1e-4, f"R1={returns[1]}"
    assert abs(returns[0].item() - 8.0500625) < 1e-3, f"R0={returns[0]}"
    print("PASS: test_discounted_returns_simple")


def test_discounted_returns_multi_episode():
    """Two episodes interleaved in batch (but ordered within each)."""
    step_rewards = np.array([0.0, 10.0, 0.0, 0.0], dtype=np.float32)
    episode_ids = np.array(["ep1", "ep1", "ep2", "ep2"])
    gamma = 0.95

    returns = compute_step_discounted_returns(step_rewards, episode_ids, gamma)

    # ep1: R1=10.0, R0=0+0.95*10=9.5
    # ep2: R3=0.0, R2=0+0.95*0=0.0
    assert abs(returns[0].item() - 9.5) < 1e-4
    assert abs(returns[1].item() - 10.0) < 1e-4
    assert abs(returns[2].item() - 0.0) < 1e-4
    assert abs(returns[3].item() - 0.0) < 1e-4
    print("PASS: test_discounted_returns_multi_episode")


def test_episode_returns():
    """All steps in an episode share the same episode-level return."""
    step_rewards = np.array([0.0, -0.1, 10.0], dtype=np.float32)
    episode_ids = np.array(["ep1", "ep1", "ep1"])
    gamma = 0.95

    ep_returns = compute_episode_returns(step_rewards, episode_ids, gamma)

    # Episode return from step 0:
    # R = 0 + 0.95*(-0.1 + 0.95*10) = 0.95*(-0.1 + 9.5) = 0.95*9.4 = 8.93
    expected = 0 + 0.95 * (-0.1 + 0.95 * 10.0)
    for i in range(3):
        assert abs(ep_returns[i] - expected) < 1e-4, f"ep_returns[{i}]={ep_returns[i]}, expected={expected}"
    print("PASS: test_episode_returns")


def test_grpo_step_advantage_all_same():
    """All episodes with same return -> advantages should be ~0."""
    step_returns = torch.tensor([5.0, 5.0, 5.0, 5.0])
    response_mask = torch.ones(4, 10)
    index = np.array(["task1", "task1", "task1", "task1"])

    advantages, returns = compute_grpo_step_advantage(
        step_returns, response_mask, index
    )

    assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-5), \
        f"All-same returns should give zero advantage, got {advantages}"
    print("PASS: test_grpo_step_advantage_all_same")


def test_grpo_step_advantage_varied():
    """Two tasks, each with two rollouts of different returns."""
    step_returns = torch.tensor([10.0, 0.0, 8.0, 2.0])
    response_mask = torch.ones(4, 5)
    index = np.array(["task1", "task1", "task2", "task2"])

    advantages, returns = compute_grpo_step_advantage(
        step_returns, response_mask, index, norm_adv_by_std=True
    )

    # task1: mean=5, std=7.071; (10-5)/7.071 ≈ 0.707, (0-5)/7.071 ≈ -0.707
    # task2: mean=5, std=4.243; (8-5)/4.243 ≈ 0.707, (2-5)/4.243 ≈ -0.707
    assert advantages[0, 0].item() > 0, "Higher return should have positive advantage"
    assert advantages[1, 0].item() < 0, "Lower return should have negative advantage"
    assert advantages[2, 0].item() > 0
    assert advantages[3, 0].item() < 0
    print("PASS: test_grpo_step_advantage_varied")


def test_grpo_step_advantage_single_rollout():
    """Single rollout per task -> advantage = 0, std = 1."""
    step_returns = torch.tensor([7.5])
    response_mask = torch.ones(1, 5)
    index = np.array(["task1"])

    advantages, returns = compute_grpo_step_advantage(
        step_returns, response_mask, index
    )

    # mean=0, std=1 for single sample
    expected = 7.5  # (7.5 - 0) / (1 + eps)
    assert abs(advantages[0, 0].item() - expected) < 0.01
    print("PASS: test_grpo_step_advantage_single_rollout")


def test_end_to_end_step_pipeline():
    """Full pipeline: assign rewards -> discounted returns -> GRPO advantage."""
    # Two 3-step episodes for the same task, one success, one failure
    episode_rewards = np.array([10., 10., 10., 0., 0., 0.], dtype=np.float32)
    episode_ids = np.array(["ep1", "ep1", "ep1", "ep2", "ep2", "ep2"])
    is_action_valid = np.array([1., 1., 1., 1., 0., 1.], dtype=np.float32)
    is_terminal = np.array([0., 0., 1., 0., 0., 1.], dtype=np.float32)
    uid = np.array(["task1"] * 6)
    gamma = 0.95

    step_rews = assign_step_rewards(
        episode_rewards, episode_ids, is_action_valid, is_terminal,
        invalid_action_penalty_coef=0.1,
    )
    # ep1: [0, 0, 10], ep2: [0, -0.1, 0]
    assert step_rews[2] == 10.0
    assert step_rews[4] == -0.1

    step_rets = compute_step_discounted_returns(step_rews, episode_ids, gamma)
    # ep1 returns should be positive (discounted 10.0)
    # ep2 returns should be near zero or slightly negative
    assert step_rets[0] > 0
    assert step_rets[3] < step_rets[0]

    response_mask = torch.ones(6, 8)
    advantages, returns = compute_grpo_step_advantage(
        step_rets, response_mask, uid
    )

    # Success episode steps should generally have positive advantage
    # Failure episode steps should have negative advantage
    ep1_adv_mean = advantages[:3, 0].mean().item()
    ep2_adv_mean = advantages[3:, 0].mean().item()
    assert ep1_adv_mean > 0, f"Success episode advantage should be positive: {ep1_adv_mean}"
    assert ep2_adv_mean < 0, f"Failure episode advantage should be negative: {ep2_adv_mean}"
    print("PASS: test_end_to_end_step_pipeline")


if __name__ == "__main__":
    test_assign_step_rewards_success_episode()
    test_assign_step_rewards_fail_episode()
    test_discounted_returns_simple()
    test_discounted_returns_multi_episode()
    test_episode_returns()
    test_grpo_step_advantage_all_same()
    test_grpo_step_advantage_varied()
    test_grpo_step_advantage_single_rollout()
    test_end_to_end_step_pipeline()
    print("\n=== ALL TESTS PASSED ===")
