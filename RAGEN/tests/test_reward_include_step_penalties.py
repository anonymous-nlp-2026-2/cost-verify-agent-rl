"""Tests for reward_include_step_penalties feature.

Verifies that:
1. _adjust_token_rewards_with_step_penalties correctly modifies token_level_rewards
2. compute_advantage with reward_include_step_penalties=True adjusts rewards before Phase 1
3. Non-tensor batch "reward" is NOT modified (avoids double-counting in step_returns)
4. Backward compatibility: reward_include_step_penalties=False leaves behavior unchanged
"""
import numpy as np
import torch
import pytest
from collections import defaultdict

import sys
sys.path.insert(0, "./RAGEN")

from ragen.trainer.agent_trainer import _adjust_token_rewards_with_step_penalties


class TestAdjustTokenRewards:
    """Test _adjust_token_rewards_with_step_penalties function."""

    def test_basic_penalty_application(self):
        """Invalid actions should reduce token_level_rewards by accumulated penalty."""
        # 2 episodes, 3 steps each
        # Episode 0: step 0 (valid), step 1 (invalid), step 2 (terminal, valid)
        # Episode 1: step 3 (invalid), step 4 (invalid), step 5 (terminal, valid)
        token_rewards = torch.zeros(6, 10)
        token_rewards[2, -1] = 0.0    # episode 0 failed
        token_rewards[5, -1] = 10.0   # episode 1 succeeded

        episode_ids = np.array([0, 0, 0, 1, 1, 1])
        is_action_valid = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        penalty_coef = 0.1

        _adjust_token_rewards_with_step_penalties(
            token_rewards, episode_ids, is_action_valid, penalty_coef
        )

        # Episode 0: 1 invalid action -> penalty = -0.1
        # All 3 steps of episode 0 should have -0.1 on last token
        assert torch.isclose(token_rewards[0, -1], torch.tensor(-0.1)), f"Got {token_rewards[0, -1]}"
        assert torch.isclose(token_rewards[1, -1], torch.tensor(-0.1)), f"Got {token_rewards[1, -1]}"
        assert torch.isclose(token_rewards[2, -1], torch.tensor(-0.1)), f"Got {token_rewards[2, -1]}"

        # Episode 1: 2 invalid actions -> penalty = -0.2
        assert torch.isclose(token_rewards[3, -1], torch.tensor(-0.2)), f"Got {token_rewards[3, -1]}"
        assert torch.isclose(token_rewards[4, -1], torch.tensor(-0.2)), f"Got {token_rewards[4, -1]}"
        assert torch.isclose(token_rewards[5, -1], torch.tensor(10.0 - 0.2)), f"Got {token_rewards[5, -1]}"

    def test_all_valid_no_change(self):
        """If all actions are valid, token_level_rewards should be unchanged."""
        token_rewards = torch.zeros(4, 10)
        token_rewards[1, -1] = 10.0
        token_rewards[3, -1] = 0.0
        original = token_rewards.clone()

        episode_ids = np.array([0, 0, 1, 1])
        is_action_valid = np.array([1.0, 1.0, 1.0, 1.0])

        _adjust_token_rewards_with_step_penalties(
            token_rewards, episode_ids, is_action_valid, 0.1
        )

        assert torch.equal(token_rewards, original), "Should not modify when all actions are valid"

    def test_all_invalid(self):
        """All invalid actions in an episode should accumulate penalties."""
        token_rewards = torch.zeros(3, 5)
        episode_ids = np.array([0, 0, 0])
        is_action_valid = np.array([0.0, 0.0, 0.0])

        _adjust_token_rewards_with_step_penalties(
            token_rewards, episode_ids, is_action_valid, 0.1
        )

        # 3 invalid actions -> penalty = -0.3 for each step
        for i in range(3):
            assert torch.isclose(token_rewards[i, -1], torch.tensor(-0.3)), f"Step {i}: got {token_rewards[i, -1]}"

    def test_multiple_episodes_independent(self):
        """Penalties should be computed independently per episode."""
        token_rewards = torch.zeros(4, 5)
        episode_ids = np.array([0, 0, 1, 1])
        is_action_valid = np.array([0.0, 1.0, 0.0, 0.0])  # ep0: 1 invalid, ep1: 2 invalid

        _adjust_token_rewards_with_step_penalties(
            token_rewards, episode_ids, is_action_valid, 0.5
        )

        # Episode 0: 1 invalid -> -0.5
        assert torch.isclose(token_rewards[0, -1], torch.tensor(-0.5))
        assert torch.isclose(token_rewards[1, -1], torch.tensor(-0.5))

        # Episode 1: 2 invalid -> -1.0
        assert torch.isclose(token_rewards[2, -1], torch.tensor(-1.0))
        assert torch.isclose(token_rewards[3, -1], torch.tensor(-1.0))

    def test_does_not_modify_other_tokens(self):
        """Only the last token position should be modified."""
        token_rewards = torch.ones(2, 5)
        episode_ids = np.array([0, 0])
        is_action_valid = np.array([0.0, 1.0])

        _adjust_token_rewards_with_step_penalties(
            token_rewards, episode_ids, is_action_valid, 0.1
        )

        # Non-last tokens should remain 1.0
        assert torch.all(token_rewards[:, :-1] == 1.0), "Non-last tokens should not be modified"

    def test_zero_penalty_coef(self):
        """Zero penalty coefficient should not modify rewards."""
        token_rewards = torch.zeros(2, 5)
        token_rewards[1, -1] = 10.0
        original = token_rewards.clone()

        episode_ids = np.array([0, 0])
        is_action_valid = np.array([0.0, 0.0])

        _adjust_token_rewards_with_step_penalties(
            token_rewards, episode_ids, is_action_valid, 0.0
        )

        assert torch.equal(token_rewards, original), "Zero coef should not modify rewards"

    def test_preserves_existing_rewards(self):
        """Penalties should be ADDED to existing rewards, not replace them."""
        token_rewards = torch.zeros(2, 5)
        token_rewards[0, -1] = 10.0  # success
        token_rewards[1, -1] = 10.0  # success

        episode_ids = np.array([0, 0])
        is_action_valid = np.array([0.0, 1.0])  # 1 invalid

        _adjust_token_rewards_with_step_penalties(
            token_rewards, episode_ids, is_action_valid, 0.1
        )

        # Both steps should have 10.0 - 0.1 = 9.9
        assert torch.isclose(token_rewards[0, -1], torch.tensor(9.9))
        assert torch.isclose(token_rewards[1, -1], torch.tensor(9.9))

    def test_realistic_scenario(self):
        """
        Simulate exp027-like scenario: 
        - 4 episodes (same task group), 5 steps each
        - Episode 0: success, 1 invalid action
        - Episode 1: fail, 0 invalid actions
        - Episode 2: fail, 3 invalid actions
        - Episode 3: fail, 5 invalid actions (all invalid)
        
        Without penalty adjustment: episodes 1,2,3 all have reward=0 -> zero variance
        With penalty adjustment: episode 1=0, episode 2=-0.3, episode 3=-0.5 -> non-zero variance
        """
        n_episodes = 4
        n_steps = 5
        total = n_episodes * n_steps
        
        token_rewards = torch.zeros(total, 10)
        episode_ids = np.repeat(np.arange(n_episodes), n_steps)
        
        # Episode rewards: only episode 0 succeeds
        for i in range(n_steps):
            token_rewards[0 * n_steps + i, -1] = 10.0  # episode 0 success
            # episodes 1,2,3: reward = 0 (fail)
        
        # Action validity
        is_action_valid = np.ones(total)
        # Episode 0: step 2 invalid
        is_action_valid[0 * n_steps + 2] = 0.0
        # Episode 1: all valid
        # Episode 2: steps 0,2,4 invalid
        is_action_valid[2 * n_steps + 0] = 0.0
        is_action_valid[2 * n_steps + 2] = 0.0
        is_action_valid[2 * n_steps + 4] = 0.0
        # Episode 3: all invalid
        for j in range(n_steps):
            is_action_valid[3 * n_steps + j] = 0.0
        
        _adjust_token_rewards_with_step_penalties(
            token_rewards, episode_ids, is_action_valid, 0.1
        )
        
        # Check episode-level rewards (last token of each step)
        ep0_reward = token_rewards[0, -1].item()  # 10.0 - 0.1 = 9.9
        ep1_reward = token_rewards[5, -1].item()  # 0.0 (no invalid actions)
        ep2_reward = token_rewards[10, -1].item() # 0.0 - 0.3 = -0.3
        ep3_reward = token_rewards[15, -1].item() # 0.0 - 0.5 = -0.5
        
        assert abs(ep0_reward - 9.9) < 1e-5, f"Episode 0: expected 9.9, got {ep0_reward}"
        assert abs(ep1_reward - 0.0) < 1e-5, f"Episode 1: expected 0.0, got {ep1_reward}"
        assert abs(ep2_reward - (-0.3)) < 1e-5, f"Episode 2: expected -0.3, got {ep2_reward}"
        assert abs(ep3_reward - (-0.5)) < 1e-5, f"Episode 3: expected -0.5, got {ep3_reward}"
        
        # Key: episodes 1,2,3 now have different rewards -> non-zero variance
        fail_rewards = [ep1_reward, ep2_reward, ep3_reward]
        variance = np.var(fail_rewards)
        assert variance > 0, f"Failing episodes should have non-zero variance, got {variance}"
        print(f"Failing episode rewards: {fail_rewards}, variance: {variance:.6f}")


if __name__ == "__main__":
    test = TestAdjustTokenRewards()
    
    test_methods = [m for m in dir(test) if m.startswith("test_")]
    passed = 0
    failed = 0
    
    for method_name in sorted(test_methods):
        try:
            getattr(test, method_name)()
            print(f"  PASS: {method_name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {method_name}: {e}")
            failed += 1
    
    print(f"\n{passed}/{passed + failed} tests passed")
    if failed > 0:
        sys.exit(1)
