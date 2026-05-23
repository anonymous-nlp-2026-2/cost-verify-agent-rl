# Failure Mode Unification: Step-Independent Credit Assignment as Root Cause

## Summary

All four negative GRPO experiments (exp013f, exp020b, exp026, exp027) exhibit distinct surface symptoms (reward gaming, policy collapse, catastrophic forgetting) yet share a single root cause: step-independent optimization suffers from Advantage Sign Misattribution: advantage SIGN is determined by episode outcome (success/failure), not window-local action quality. Good actions in failed episodes are incorrectly penalized; bad actions in successful episodes are incorrectly reinforced. γ^k magnitude decay further attenuates early windows. The attempted fixes (bonus ratio tuning, KL constraints, LR reduction, entropy bonuses) addressed symptoms without resolving the structural advantage sign misattribution (direction error from episode-level outcome + γ^k magnitude attenuation). Episode-level training (exp029) with the same SFT checkpoint achieved 28.125% at step 15 (single seed, N=32 episodes), confirming the diagnosis.

## Unification Table

| Experiment | Surface Symptom | Attempted Fix | Actual Root Cause | Key Evidence |
|------------|----------------|---------------|-------------------|--------------|
| exp013f | Reward gaming: action_is_valid 60%→97.5%, task success 0% | Reduce bonus ratio (1:67→1:1000), success-gated reward | Step-independent optimization maximizes per-step bonus without requiring task completion | In step-independent mode, per-window bonus (0.3) is the only causally linked reward signal; episode success (10.0) is γ-attenuated across windows (position k gets γ^k weight), reducing early windows to noise; failed episodes assign incorrect advantage SIGN to all windows (good actions penalized, bad actions reinforced based on episode outcome rather than action quality) → model optimizes window-local bonus |
| exp020b | Policy collapse: success 20%→4.69%, entropy collapse 61%, pass@16 75%→25% | Increase KL constraint, fix response_mask, adjust LR | Per-step GRPO destroys inter-step correlations learned by SFT | Entropy 0.107→0.042; model collapsed to "safe" per-step actions, losing sequential planning ability |
| exp026 | Catastrophic collapse: train success peaks 32.81% (step 2) then drops to 0% (step 4) | exp027: increase env_groups (4→8), lower LR (1e-6→5e-7), lower KL, add entropy bonus | Advantage sign misattribution: advantage SIGN determined by episode outcome, not window-local action quality. Good actions in failed episodes are incorrectly penalized; bad actions in successful episodes are incorrectly reinforced. Additionally, γ^k magnitude decay for early windows (γ^24≈29% for h=2 in 50-step episodes) | Model learned locally valid actions (action_is_valid ~72%) but lost task completion; exp029 (episode-level) reached 28.125% stable |
| exp027 | Same collapse pattern: peak 35.94% (step 1) then decay to 10.16% (step 2-3), Ray OOM crash | Recipe gap analysis → concluded mini-batch/LR/steps mismatch | Still step-independent (h=5); larger rollouts delayed but did not prevent collapse | D045 recipe gap analysis was incomplete: the "62pp gap" was not about parameter tuning but about fundamental credit assignment mode |

## Diagnostic Timeline

1. **exp013f** (early): Observed reward gaming. Hypothesized bonus ratio was too large. Proposed ratio reduction and success-gating as fixes.
2. **exp020b** (mid): Observed policy collapse from SFT baseline. Hypothesized KL/LR misconfiguration. Attempted constraint tuning.
3. **exp026** (late-mid): Observed catastrophic collapse after initial improvement. Hypothesized insufficient rollout diversity and aggressive updates.
4. **exp027** (late): Applied conventional recipe fixes (more data, lower LR, entropy bonus). Same collapse pattern persisted, plus infrastructure crash.
5. **D045 recipe gap analysis**: Systematically compared our hyperparameters against published GRPO recipes. Concluded a "62pp gap" from parameter mismatch. This diagnosis was incorrect.
6. **Insight**: Recognized that all failures shared a common structure: the model could learn locally optimal per-step behavior but could not maintain or improve multi-step coherence. The distinguishing variable was not hyperparameters but the granularity of credit assignment.
7. **exp029 (episode-level)**: Same SFT checkpoint, same task, episode-level training. Achieved 28.125% at step 15 (single seed, N=32 episodes), confirming that the credit assignment mode was the true bottleneck.

## Implications

- **Community value**: Multi-turn agent training with GRPO is increasingly popular, but published recipes assume single-turn or short-horizon settings. This unification demonstrates that step-independent GRPO suffers from Advantage Sign Misattribution in long-horizon tasks (avg ~50 steps): advantage sign is determined by episode outcome rather than window-local action quality, causing good actions in failed episodes to be penalized and bad actions in successful episodes to be reinforced. γ^k magnitude decay further attenuates early windows. Practitioners attempting multi-turn agent RL should default to episode-level credit assignment rather than iterating on per-step hyperparameters.
- **Paper placement**: This table belongs in the Experiments section as an analysis subsection (e.g., "Analysis: Unifying Failure Modes"). It serves as evidence that episode-level training is not merely one design choice among equals but a necessary condition for multi-turn success. The diagnostic timeline supports the narrative arc: surface-level fixes failed repeatedly until the structural root cause was identified.

## Counter-evidence: Independent Failure Modes

- **exp021 (data quality)**: Failed due to insufficient evaluation episodes (4-episode eval showed 25% but 16-episode eval showed 6.25%). This is a measurement/data quality issue unrelated to step-independent credit assignment.
- **exp028_prereq (zero-shot diagnostic)**: A diagnostic experiment testing base model capability without RL training. Not a training failure and thus outside the scope of this analysis.
- **Scope of the claim**: Step-independent credit assignment is the dominant failure mode for GRPO in multi-turn ALFWorld (4/4 training failures), but it is not a universal explanation for all negative results. Infrastructure issues (Ray OOM), evaluation methodology (insufficient episodes), and data quality remain independent failure sources. The unification claim is specific: when GRPO training itself fails to improve task success in long-horizon settings, Advantage Sign Misattribution (sign error from episode-level outcome + γ^k magnitude decay) is the first mechanism to examine.
