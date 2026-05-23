"""
Step-level reward assignment and discounted return computation.

Phase 1: Standard GRPO + discounted returns + invalid action penalty.
Phase 2 (future): GiGPO step-level advantage with anchor state grouping.
"""

import numpy as np
import torch
from collections import defaultdict
from typing import Optional


def assign_step_rewards(
    episode_rewards: np.ndarray,
    episode_ids: np.ndarray,
    is_action_valid: np.ndarray,
    is_terminal: np.ndarray,
    invalid_action_penalty_coef: float = 0.1,
) -> np.ndarray:
    """
    Assign per-step rewards from episode-level rewards.

    Intermediate steps: -penalty if action invalid, else 0.
    Terminal step: episode_reward (+ invalid penalty if applicable).

    Args:
        episode_rewards: (batch_size,) episode reward for each sample's episode
        episode_ids: (batch_size,) episode identifier
        is_action_valid: (batch_size,) 1.0 if valid, 0.0 if invalid
        is_terminal: (batch_size,) 1.0 if last step of episode
        invalid_action_penalty_coef: penalty magnitude for invalid actions

    Returns:
        step_rewards: (batch_size,) per-step rewards
    """
    batch_size = len(episode_rewards)
    step_rewards = np.zeros(batch_size, dtype=np.float32)

    for i in range(batch_size):
        if not is_action_valid[i]:
            step_rewards[i] -= invalid_action_penalty_coef
        if is_terminal[i]:
            step_rewards[i] += float(episode_rewards[i])

    return step_rewards


def compute_step_discounted_returns(
    step_rewards: np.ndarray,
    episode_ids: np.ndarray,
    gamma: float = 0.95,
) -> torch.Tensor:
    """
    Backward discounted returns per step within each episode.
    R_t = r_t + gamma * R_{t+1}

    Assumes samples within each episode are ordered by step index
    (ascending) in the batch.

    Args:
        step_rewards: (batch_size,) per-step reward
        episode_ids: (batch_size,) episode identifier
        gamma: discount factor

    Returns:
        (batch_size,) tensor of discounted returns
    """
    returns = np.zeros_like(step_rewards, dtype=np.float32)

    episode_to_indices = defaultdict(list)
    for i, eid in enumerate(episode_ids):
        episode_to_indices[eid].append(i)

    for eid, indices in episode_to_indices.items():
        running_return = 0.0
        for idx in reversed(indices):
            running_return = step_rewards[idx] + gamma * running_return
            returns[idx] = running_return

    return torch.tensor(returns, dtype=torch.float32)


def compute_episode_returns(
    step_rewards: np.ndarray,
    episode_ids: np.ndarray,
    gamma: float = 0.95,
) -> np.ndarray:
    """
    Compute a single discounted return per episode (from step 0).
    Used for GRPO group normalization at the episode level.

    Returns:
        (batch_size,) array where all steps in an episode share the same
        episode-level return (the return from step 0).
    """
    episode_to_indices = defaultdict(list)
    for i, eid in enumerate(episode_ids):
        episode_to_indices[eid].append(i)

    episode_return = np.zeros(len(step_rewards), dtype=np.float32)
    for eid, indices in episode_to_indices.items():
        running = 0.0
        for idx in reversed(indices):
            running = step_rewards[idx] + gamma * running
        for idx in indices:
            episode_return[idx] = running

    return episode_return
