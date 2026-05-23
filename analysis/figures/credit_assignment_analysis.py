"""
Advantage Sign Misattribution Analysis (plan_004)

Visualizes two mechanisms of credit assignment failure in step-independent GRPO:
1. Sign misattribution (PRIMARY): advantage SIGN is determined by episode outcome
   (success/failure), not window-local action quality. Good actions in failed episodes
   are incorrectly penalized; bad actions in successful episodes are incorrectly reinforced.
2. γ^k magnitude decay (SECONDARY): advantage magnitude of window at position k = γ^k * base
   → for h=2 in 50-step episodes, earliest window gets γ^24 ≈ 29% signal

Data source: advantage_log.jsonl from h=2 seed 2 (LOG_PER_SAMPLE_ADV=1)

Planned figures:
- Fig A: γ^k magnitude decay curve (secondary mechanism)
- Fig B: within-episode advantage variance (near-zero confirms uniform sign assignment)
- Fig C: Sign misattribution quadrant visualization (primary mechanism)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ADV_LOG_PATH = Path("./logs/hsweep_h2_seed2/advantage_log.jsonl")


def load_advantage_data(path=ADV_LOG_PATH):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def plot_advantage_vs_position(records):
    """Fig A: γ^k magnitude decay visualization.
    
    NOTE: applies to step_independent mode's compute_step_discounted_returns(),
    not GRPO algorithm in general. This shows magnitude attenuation;
    sign misattribution is the more fundamental issue.
    """
    positions = [r["window_position"] for r in records if r["window_position"] >= 0]
    advantages = [r["advantage"] for r in records if r["window_position"] >= 0]
    
    plt.figure(figsize=(8, 5))
    plt.scatter(positions, advantages, alpha=0.3, s=10)
    
    # Overlay theoretical gamma^k curve
    k = np.arange(max(positions) + 1)
    gamma = 0.95
    theoretical = (gamma ** k) * 10  # unnormalized
    # Note: actual advantages are group-normalized, so scale for visual comparison
    
    plt.xlabel("Window Position (steps from start)")
    plt.ylabel("Advantage (group-normalized)")
    plt.title("Magnitude Decay: γ^k Attenuation vs Window Position (h=2)")
    plt.savefig(str(ADV_LOG_PATH.parent / "credit_decay_scatter.pdf"), bbox_inches="tight")
    plt.close()


def compute_within_episode_variance(records):
    """Fig B: Within-episode variance of advantages.
    
    Near-zero variance confirms that all windows in an episode receive the same
    advantage sign (determined by episode outcome, not action quality).
    """
    from collections import defaultdict
    episodes = defaultdict(list)
    for r in records:
        if r["episode_id"] >= 0:
            episodes[r["episode_id"]].append(r["advantage"])
    
    variances = {eid: np.var(advs) for eid, advs in episodes.items() if len(advs) > 1}
    return variances


def plot_sign_misattribution(records):
    """Fig C: Sign misattribution visualization.
    
    Shows: for each window, color by (episode_success, action_quality).
    Quadrants:
    - Good action + Success episode → correctly reinforced (green)
    - Good action + Failed episode → incorrectly penalized (RED - sign error)
    - Bad action + Success episode → incorrectly reinforced (RED - sign error)
    - Bad action + Failed episode → correctly penalized (green)
    """
    pass  # Will implement after advantage_log.jsonl available


if __name__ == "__main__":
    if ADV_LOG_PATH.exists():
        records = load_advantage_data()
        print(f"Loaded {len(records)} records")
        plot_advantage_vs_position(records)
        variances = compute_within_episode_variance(records)
        print(f"Mean within-episode variance: {np.mean(list(variances.values())):.6f}")
    else:
        print(f"Data not yet available: {ADV_LOG_PATH}")
        print("Run h=2 seed 2 with LOG_PER_SAMPLE_ADV=1 first.")
