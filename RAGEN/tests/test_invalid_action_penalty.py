"""
Unit tests for invalid_action_penalty mechanism.
Verifies: step_rewards correctly penalizes invalid actions,
and all-fail groups with different invalid action counts produce different rewards.
"""
import numpy as np
import pytest
from ragen.trainer.step_rewards import (
    assign_step_rewards,
    compute_step_discounted_returns,
    compute_episode_returns,
)


def test_invalid_action_penalty_basic():
    """Invalid actions get -0.1 penalty, valid actions get 0."""
    episode_rewards = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    episode_ids = np.array([0, 0, 0])
    is_action_valid = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    is_terminal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    step_rews = assign_step_rewards(
        episode_rewards=episode_rewards,
        episode_ids=episode_ids,
        is_action_valid=is_action_valid,
        is_terminal=is_terminal,
        invalid_action_penalty_coef=0.1,
    )
    assert step_rews[0] == 0.0, "Valid non-terminal: 0"
    assert step_rews[1] == pytest.approx(-0.1), "Invalid non-terminal: -0.1"
    assert step_rews[2] == 0.0, "Valid terminal (reward=0): 0"


def test_invalid_action_penalty_terminal():
    """Terminal step with invalid action gets both penalty and episode reward."""
    episode_rewards = np.array([10.0], dtype=np.float32)
    episode_ids = np.array([0])
    is_action_valid = np.array([0.0], dtype=np.float32)
    is_terminal = np.array([1.0], dtype=np.float32)

    step_rews = assign_step_rewards(
        episode_rewards=episode_rewards,
        episode_ids=episode_ids,
        is_action_valid=is_action_valid,
        is_terminal=is_terminal,
        invalid_action_penalty_coef=0.1,
    )
    assert step_rews[0] == pytest.approx(9.9), "Terminal invalid: reward - penalty"


def test_all_fail_group_variance():
    """
    Core test: In an all-fail GRPO group, trajectories with more invalid
    actions should have lower episode returns, creating variance for gradients.
    
    Scenario: 3 trajectories, all fail (reward=0), but:
    - Traj 0: 5 steps, 0 invalid → return = 0
    - Traj 1: 5 steps, 3 invalid → return < 0
    - Traj 2: 5 steps, 5 invalid → return << 0
    """
    n_steps = 5
    n_trajs = 3
    total = n_steps * n_trajs

    episode_rewards = np.zeros(total, dtype=np.float32)
    episode_ids = np.repeat([0, 1, 2], n_steps)
    is_terminal = np.zeros(total, dtype=np.float32)
    is_terminal[4] = 1.0   # end of traj 0
    is_terminal[9] = 1.0   # end of traj 1
    is_terminal[14] = 1.0  # end of traj 2

    # Traj 0: all valid
    # Traj 1: steps 5,6,7 invalid (3 of 5)
    # Traj 2: all invalid
    is_action_valid = np.ones(total, dtype=np.float32)
    is_action_valid[5:8] = 0.0   # traj 1: 3 invalid
    is_action_valid[10:15] = 0.0  # traj 2: all invalid

    step_rews = assign_step_rewards(
        episode_rewards=episode_rewards,
        episode_ids=episode_ids,
        is_action_valid=is_action_valid,
        is_terminal=is_terminal,
        invalid_action_penalty_coef=0.1,
    )

    ep_returns = compute_episode_returns(step_rews, episode_ids, gamma=0.95)

    traj0_return = ep_returns[0]
    traj1_return = ep_returns[5]
    traj2_return = ep_returns[10]

    assert traj0_return == pytest.approx(0.0), "All valid, all fail → 0"
    assert traj1_return < 0, "3 invalid actions → negative return"
    assert traj2_return < traj1_return, "5 invalid < 3 invalid"

    # Verify variance exists (non-zero → gradient signal)
    returns_per_traj = [traj0_return, traj1_return, traj2_return]
    assert np.std(returns_per_traj) > 0, "Variance must exist for GRPO gradient"


def test_penalty_coef_zero_disables():
    """When coef=0, no penalty is applied regardless of validity."""
    episode_rewards = np.array([0.0, 0.0], dtype=np.float32)
    episode_ids = np.array([0, 0])
    is_action_valid = np.array([0.0, 0.0], dtype=np.float32)
    is_terminal = np.array([0.0, 1.0], dtype=np.float32)

    step_rews = assign_step_rewards(
        episode_rewards=episode_rewards,
        episode_ids=episode_ids,
        is_action_valid=is_action_valid,
        is_terminal=is_terminal,
        invalid_action_penalty_coef=0.0,
    )
    assert step_rews[0] == 0.0
    assert step_rews[1] == 0.0


def test_success_plus_penalty():
    """Success reward (10) + invalid action penalties sum correctly."""
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

    ep_returns = compute_episode_returns(step_rews, episode_ids, gamma=0.95)
    # Step rewards: [-0.1, -0.1, 10.0]
    # Discounted from step 0: -0.1 + 0.95*(-0.1) + 0.95^2*(10.0)
    expected = -0.1 + 0.95 * (-0.1) + 0.95**2 * 10.0
    assert ep_returns[0] == pytest.approx(expected, rel=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
