# ScienceWorld environment configuration for RAGEN.
# Mirrors AlfredEnvConfig pattern: sparse reward at episode end, mode-based splits.
from ragen.env.base import BaseEnvConfig
from dataclasses import dataclass, field
from typing import List


@dataclass
class ScienceWorldEnvConfig(BaseEnvConfig):
    task_names: str = "boil"  # comma-separated task names, e.g. "boil,melt,freeze"
    score: float = 10.0  # reward on full completion (100/100 score)
    max_steps: int = 100
    simplification_str: str = ""  # ScienceWorld simplification flags
    max_valid_actions: int = 50  # cap on admissible actions shown in observation
    eval_split: str = "dev"  # "dev" or "test" for val/test mode
