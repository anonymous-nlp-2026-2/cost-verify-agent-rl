from ragen.env.base import BaseEnvConfig
from dataclasses import dataclass


@dataclass
class AlfredEnvConfig(BaseEnvConfig):
    """Configuration for text world AlfredEnv.

    Matches verl-agent AlfWorld configuration:
    - Sparse reward: score * won (1.0 on success)
    - Supports train/eval_in_distribution/eval_out_of_distribution splits
    """
    config_file: str = "./ragen/env/alfworld/alfworld_config.yaml"
    score: float = 10.0  # Reward on success (matching verl-agent)
    valid_action_bonus: float = 0.0
    success_gated_bonus: bool = False
    render_mode: str = "text"
    eval_dataset: str = "eval_in_distribution"  # 'eval_in_distribution' or 'eval_out_of_distribution'
