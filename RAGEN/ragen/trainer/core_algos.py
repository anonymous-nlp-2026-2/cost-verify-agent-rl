from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage as _compute_grpo_outcome_advantage,
    compute_reinforce_plus_plus_outcome_advantage,
    compute_reinforce_plus_plus_baseline_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_value_loss,
)


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    episode_ids: Optional[np.ndarray] = None,
    return_group_std: bool = False,
):
    """
    Compute advantage for GRPO with episode-level deduplication support.

    When episode_ids is provided (for single_turn/limited_multi_turn mode), each (index, episode_id) pair
    only contributes once to mean/std calculation, avoiding bias from different turn counts.

    Args:
        return_group_std: If True, also returns the per-sample group std tensor for soft reweighting.

    Returns:
        If return_group_std=False: (advantages, returns)
        If return_group_std=True: (advantages, returns, group_std) where group_std is shape (batch_size,)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        # Use seen_pairs to deduplicate when episode_ids is provided
        seen_pairs = set()
        for i in range(bsz):
            if episode_ids is not None:
                pair = (index[i], episode_ids[i])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

        # Collect per-sample group std for soft reweighting
        if return_group_std:
            group_std = torch.zeros(bsz, device=scores.device)
            for i in range(bsz):
                group_std[i] = id2std[index[i]]
            return scores, scores, group_std

    return scores, scores

# supported by Kangrui Wang
def compute_bi_level_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor, 
        loss_mask: torch.Tensor,
        gamma: float,
        lam: float,
        high_level_gamma: float
    ):
    """Modified GAE calculation that compute two level of advantage and return:
    high level: per-turn wise
    low level: token wise
    there're two level of MDP, where high level is the agentic MDP and low level is the token MDP
    Args:
        token_level_rewards: `(torch.Tensor)` (multi-turn reward, per turn reward is given at eos token for each response token sequence)
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)`
            discounted factor used in RL for token rewards
        high_level_gamma: `(float)`
            discounted factor used in RL for per-turn reward
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        token_level_rewards = token_level_rewards.float()
        reward_mask = token_level_rewards.bool()
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        updated_reward = token_level_rewards.clone()
        
        for b in range(batch_size):
            # First, calculate high level advantage and return for eos token of each turn using high level gamma
            eos_positions=reward_mask[b].nonzero(as_tuple=True)[0]
            lastgaelam = 0.0
            for i in range(len(eos_positions) - 1, -1, -1):
                curr_pos = eos_positions[i]
                
                # Get the next value
                if i < len(eos_positions) - 1:
                    # Next valid position
                    next_pos = eos_positions[i + 1]
                    nextvalue = values[b, next_pos]
                    
                else:
                    # Last valid position
                    nextvalue = 0.0
                
                # Calculate delta using the next valid token
                delta = updated_reward[b, curr_pos] + high_level_gamma * nextvalue - values[b, curr_pos]
                
                # Update advantage estimate
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
            
            for i, pos in enumerate(eos_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]
                updated_reward[b, pos] = advantages[b, pos] + values[b, pos]
            
            # Then, calculate low level advantage and return for each token using gamma, assume the reward for the sequence now is the return at eos token
            lastgaelam = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                if curr_pos not in eos_positions:
                    # Next valid position
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                else:
                    # Last valid position
                    nextvalue = 0.0
                    lastgaelam = 0.0
                delta = updated_reward[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
                returns[b, curr_pos] = lastgaelam + values[b, curr_pos]

        advantages = verl_F.masked_whiten(advantages, loss_mask)
    
    return advantages, returns


# set up unittest
if __name__ == "__main__":
    token_level_rewards = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    loss_mask = torch.ones(1, 10)
    advantages, returns = compute_bi_level_gae_advantage_return(token_level_rewards, values, loss_mask, 1, 1, 0.95)
    print(advantages)
    print(returns)

def compute_grpo_step_advantage(
    step_returns: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std: bool = True,
):
    """
    GRPO advantage for step-level samples.

    Each sample is one step of an episode. Groups are defined by `index`
    (task/prompt id). Within each group, advantage = (return - mean) / (std + eps),
    then broadcast to all response tokens via response_mask.

    Args:
        step_returns: (batch_size,) discounted return per step
        response_mask: (batch_size, response_length)
        index: (batch_size,) task/prompt group identifier
        epsilon: numerical stability
        norm_adv_by_std: divide by group std (True = standard GRPO)

    Returns:
        (advantages, returns) both shape (batch_size, response_length)
    """
    scores = step_returns.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            else:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)

        for i in range(bsz):
            if norm_adv_by_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        advantages = scores.unsqueeze(-1) * response_mask

    return advantages, advantages


# ---------------------------------------------------------- #
# GiGPO: Group-in-Group Policy Optimization
# Adapted from verl-agent (https://github.com/langfengQ/verl-agent)
# Paper: https://arxiv.org/abs/2505.10978
# ---------------------------------------------------------- #

import uuid
from difflib import SequenceMatcher
from typing import List, Dict, Any


def to_hashable(x):
    """Convert an object into a hashable type for anchor state grouping."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """Check whether two text observations are similar enough for grouping."""
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO.")
    return SequenceMatcher(None, a, b).ratio() >= threshold


def build_step_group(
    anchor_obs: np.ndarray,
    index: np.ndarray,
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
) -> np.ndarray:
    """
    Group steps by anchor state within each task/prompt group.

    For each unique task index, steps with the same anchor observation
    get the same group UID. Returns an array of string UIDs.

    Args:
        anchor_obs: (batch_size,) raw env observations at each step
        index: (batch_size,) task/prompt group identifier
        enable_similarity: use fuzzy matching instead of exact match
        similarity_thresh: threshold for fuzzy matching

    Returns:
        step_group_uids: (batch_size,) string UIDs for step groups
    """
    bsz = len(anchor_obs)
    step_group_uids = np.empty(bsz, dtype=object)
    group_sizes = []

    unique_indices = np.unique(index)
    for idx in unique_indices:
        locs = np.where(index == idx)[0]
        obs_group = anchor_obs[locs]

        if not enable_similarity:
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(locs[i])

            for obs_key, original_indices in clusters.items():
                uid = str(uuid.uuid4())
                group_sizes.append(len(original_indices))
                for orig_idx in original_indices:
                    step_group_uids[orig_idx] = uid
        else:
            # Fuzzy matching: O(n*k) where k = number of clusters
            clusters: List[Dict[str, Any]] = []
            for obs, loc in zip(obs_group, locs):
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})

            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_sizes.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

    if np.any(step_group_uids == None):
        missing = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign step group UIDs at indices: {missing}")

    avg_size = np.mean(group_sizes) if group_sizes else 0.0
    return step_group_uids, avg_size


def gigpo_episode_norm_reward(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    episode_ids: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std: bool = True,
) -> torch.Tensor:
    """
    Episode-level advantage with per-trajectory deduplication (Eq. 3 in GiGPO paper).

    Each (index, episode_id) pair contributes only once to mean/std,
    avoiding bias from different step counts across trajectories.

    Returns:
        advantages: (batch_size, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)
    response_length = response_mask.shape[-1]

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            pair = (index[i], episode_ids[i])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            else:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = scores_tensor.mean()
                id2std[idx] = scores_tensor.std()

        for i in range(bsz):
            if norm_adv_by_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        advantages = scores.unsqueeze(-1).expand(-1, response_length) * response_mask

    return advantages


def gigpo_step_norm_reward(
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_uids: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std: bool = True,
) -> torch.Tensor:
    """
    Step-level advantage within anchor state groups (Eq. 7 in GiGPO paper).

    Groups are defined by step_group_uids from build_step_group().

    Returns:
        step_advantages: (batch_size, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[step_group_uids[i]].append(scores[i])

        for uid in id2score:
            if len(id2score[uid]) == 1:
                id2mean[uid] = scores[0].new_tensor(0.0)
                id2std[uid] = scores[0].new_tensor(1.0)
            else:
                stacked = torch.stack(id2score[uid])
                id2mean[uid] = stacked.mean()
                id2std[uid] = stacked.std()

        for i in range(bsz):
            if norm_adv_by_std:
                scores[i] = (scores[i] - id2mean[step_group_uids[i]]) / (id2std[step_group_uids[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[step_group_uids[i]]

        step_advantages = scores.unsqueeze(-1).expand(-1, response_length) * response_mask

    return step_advantages


def compute_gigpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_returns: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    episode_ids: np.ndarray,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    norm_adv_by_std: bool = True,
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
):
    """
    Compute GiGPO advantage: episode-level + step-level (Eq. 8 in paper).

    Combines:
    - Episode-level advantage (GRPO-style, with trajectory deduplication)
    - Step-level advantage (anchor state grouping + normalization)

    Args:
        token_level_rewards: (bs, response_length) token rewards
        step_returns: (bs,) discounted returns per step
        response_mask: (bs, response_length) mask
        anchor_obs: (bs,) raw env observation at each step
        index: (bs,) task/prompt group identifier
        episode_ids: (bs,) trajectory identifier
        step_advantage_w: weight ω for step-level advantage
        norm_adv_by_std: normalize by std (True) or just subtract mean
        enable_similarity: fuzzy anchor matching
        similarity_thresh: threshold if enable_similarity

    Returns:
        (advantages, advantages, metrics_dict) where metrics_dict has
        step_group_avg_size and step_advantage_magnitude for logging
    """
    # Eq. 3: Episode-level advantage
    episode_advantages = gigpo_episode_norm_reward(
        token_level_rewards, response_mask, index, episode_ids,
        epsilon=epsilon, norm_adv_by_std=norm_adv_by_std,
    )

    # Eq. 6: Anchor state grouping
    step_group_uids, step_group_avg_size = build_step_group(
        anchor_obs, index,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )

    # Eq. 7: Step-level advantage
    step_advantages = gigpo_step_norm_reward(
        step_returns, response_mask, step_group_uids,
        epsilon=epsilon, norm_adv_by_std=norm_adv_by_std,
    )

    # Eq. 8: Combined advantage
    advantages = episode_advantages + step_advantage_w * step_advantages

    metrics = {
        "gigpo/step_group_avg_size": step_group_avg_size,
        "gigpo/step_advantage_magnitude": step_advantages.abs().mean().item(),
        "gigpo/episode_advantage_magnitude": episode_advantages.abs().mean().item(),
    }

    return advantages, advantages, metrics
