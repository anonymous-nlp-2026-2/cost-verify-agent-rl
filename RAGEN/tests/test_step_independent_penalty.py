"""
Verify that invalid_action_penalty works in the step_independent + GRPO path.

Tests:
1. assign_step_rewards applies penalty to invalid actions
2. step_returns differ between trajectories with different invalid counts
3. penalty=0 produces same results as all-valid
4. GRPO advantage has non-zero variance in all-fail groups with different invalid counts
"""
import numpy as np
import torch
import sys
sys.path.insert(0, './RAGEN')

from ragen.trainer.step_rewards import (
    assign_step_rewards,
    compute_step_discounted_returns,
)
from ragen.trainer.core_algos import compute_grpo_step_advantage


def test_penalty_applied_to_invalid_actions():
    """Invalid actions get -penalty_coef in step_rewards."""
    episode_rewards = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    episode_ids = np.array([0, 0, 0, 0, 0])
    is_action_valid = np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    is_terminal = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    step_rews = assign_step_rewards(
        episode_rewards=episode_rewards,
        episode_ids=episode_ids,
        is_action_valid=is_action_valid,
        is_terminal=is_terminal,
        invalid_action_penalty_coef=0.1,
    )

    assert step_rews[0] == 0.0, f"Valid action should have 0 reward, got {step_rews[0]}"
    assert step_rews[1] == -0.1, f"Invalid action should have -0.1, got {step_rews[1]}"
    assert step_rews[2] == 0.0
    assert step_rews[3] == -0.1
    assert step_rews[4] == -0.1, f"Terminal + invalid should have -0.1 + 0.0 = -0.1, got {step_rews[4]}"
    print("PASS: test_penalty_applied_to_invalid_actions")


def test_step_returns_differ_by_invalid_count():
    """Two episodes (same group, both fail) with different invalid counts produce different step_returns."""
    # Episode 0: 3 steps, 1 invalid action
    # Episode 1: 3 steps, 3 invalid actions
    episode_rewards = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    episode_ids = np.array([0, 0, 0, 1, 1, 1])
    is_action_valid = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    is_terminal = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)

    step_rews = assign_step_rewards(
        episode_rewards=episode_rewards,
        episode_ids=episode_ids,
        is_action_valid=is_action_valid,
        is_terminal=is_terminal,
        invalid_action_penalty_coef=0.1,
    )

    step_returns = compute_step_discounted_returns(
        step_rewards=step_rews,
        episode_ids=episode_ids,
        gamma=0.95,
    )

    # Episode 0's return should be > Episode 1's return (fewer penalties)
    ep0_return = step_returns[0].item()
    ep1_return = step_returns[3].item()
    assert ep0_return > ep1_return, (
        f"Episode with fewer invalid actions should have higher return: "
        f"ep0={ep0_return:.4f}, ep1={ep1_return:.4f}"
    )
    print(f"PASS: test_step_returns_differ_by_invalid_count (ep0={ep0_return:.4f} > ep1={ep1_return:.4f})")


def test_zero_penalty_equals_no_penalty():
    """penalty_coef=0 produces same step_rewards as all-valid."""
    episode_rewards = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    episode_ids = np.array([0, 0, 0])
    is_action_valid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    is_terminal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    step_rews_zero = assign_step_rewards(
        episode_rewards=episode_rewards,
        episode_ids=episode_ids,
        is_action_valid=is_action_valid,
        is_terminal=is_terminal,
        invalid_action_penalty_coef=0.0,
    )

    step_rews_valid = assign_step_rewards(
        episode_rewards=episode_rewards,
        episode_ids=episode_ids,
        is_action_valid=np.ones(3, dtype=np.float32),
        is_terminal=is_terminal,
        invalid_action_penalty_coef=0.1,
    )

    np.testing.assert_array_equal(step_rews_zero, step_rews_valid)
    print("PASS: test_zero_penalty_equals_no_penalty")


def test_grpo_advantage_variance_in_all_fail_group():
    """
    All-fail group: 4 trajectories with different invalid action counts.
    GRPO advantage should have non-zero variance (gradient signal exists).
    """
    # 4 episodes in same group, all fail (reward=0)
    # Ep0: 3 steps, 0 invalid  Ep1: 3 steps, 1 invalid
    # Ep2: 3 steps, 2 invalid  Ep3: 3 steps, 3 invalid
    n_eps = 4
    steps_per_ep = 3
    n = n_eps * steps_per_ep

    episode_rewards = np.zeros(n, dtype=np.float32)
    episode_ids = np.repeat(np.arange(n_eps), steps_per_ep)
    is_terminal = np.array([0, 0, 1] * n_eps, dtype=np.float32)
    is_action_valid = np.array([
        1.0, 1.0, 1.0,   # ep0: 0 invalid
        1.0, 1.0, 0.0,   # ep1: 1 invalid
        1.0, 0.0, 0.0,   # ep2: 2 invalid
        0.0, 0.0, 0.0,   # ep3: 3 invalid
    ], dtype=np.float32)

    step_rews = assign_step_rewards(
        episode_rewards=episode_rewards,
        episode_ids=episode_ids,
        is_action_valid=is_action_valid,
        is_terminal=is_terminal,
        invalid_action_penalty_coef=0.1,
    )

    step_returns = compute_step_discounted_returns(
        step_rewards=step_rews,
        episode_ids=episode_ids,
        gamma=0.95,
    )

    # All episodes share same uid (same GRPO group)
    uid = np.array(["group0"] * n)
    response_mask = torch.ones(n, 10)

    advantages, returns = compute_grpo_step_advantage(
        step_returns=step_returns,
        response_mask=response_mask,
        index=uid,
        norm_adv_by_std=True,
    )

    # Check that step_returns differ across episodes
    ep_returns = [step_returns[i * steps_per_ep].item() for i in range(n_eps)]
    assert len(set(round(r, 6) for r in ep_returns)) > 1, (
        f"Step returns should differ across episodes: {ep_returns}"
    )

    # Check advantage variance is non-zero
    adv_values = advantages[response_mask.bool()].detach()
    adv_var = adv_values.var().item()
    assert adv_var > 1e-8, f"Advantage variance should be non-zero, got {adv_var}"
    print(f"PASS: test_grpo_advantage_variance_in_all_fail_group")
    print(f"  ep_returns: {[f'{r:.4f}' for r in ep_returns]}")
    print(f"  advantage variance: {adv_var:.6f}")


def test_penalty_with_success_episode():
    """Success episode (reward=10) + penalty should have net positive return."""
    episode_rewards = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    episode_ids = np.array([0, 0, 0])
    is_action_valid = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    is_terminal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    step_rews = assign_step_rewards(
        episode_rewards=episode_rewards,
        episode_ids=episode_ids,
        is_action_valid=is_action_valid,
        is_terminal=is_terminal,
        invalid_action_penalty_coef=0.1,
    )

    step_returns = compute_step_discounted_returns(
        step_rewards=step_rews,
        episode_ids=episode_ids,
        gamma=0.95,
    )

    # First step return should be positive (10 * gamma^2 - 0.1 - 0.1*gamma)
    assert step_returns[0].item() > 0, f"Success episode return should be positive: {step_returns[0].item()}"
    # Terminal step should be exactly 10.0 (valid action at terminal)
    assert step_rews[2] == 10.0, f"Terminal valid action reward: {step_rews[2]}"
    print(f"PASS: test_penalty_with_success_episode (first_step_return={step_returns[0].item():.4f})")


if __name__ == "__main__":
    test_penalty_applied_to_invalid_actions()
    test_step_returns_differ_by_invalid_count()
    test_zero_penalty_equals_no_penalty()
    test_grpo_advantage_variance_in_all_fail_group()
    test_penalty_with_success_episode()
    print("\n=== ALL TESTS PASSED ===")
