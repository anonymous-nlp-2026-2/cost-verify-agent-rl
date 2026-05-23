# Why LLM Agent Training Defaults to Step-Independent Mode

## Summary

LLM agent RL frameworks (RAGEN, verl/verl-agent, OpenRLHF, SkyRL-Agent, Verlog) default to step-independent mode primarily for three reinforcing reasons: (1) backward compatibility with single-turn RLHF infrastructure, (2) context window and GPU memory constraints, and (3) computational efficiency through uniform-length batching. The design was not a deliberate algorithmic choice based on multi-turn RL theory, but an engineering path-of-least-resistance that inherited assumptions from single-turn RLHF and happened to sidestep context/memory bottlenecks. The community has only recently begun to recognize this as a problem rather than a feature.

## Evidence by Framework

### verl / verl-agent

**The most explicit evidence of step-independent-by-design.**

verl decomposes each agent execution into prompt-response pairs via an "Adapter" and associates them with reward signals as "Triplet" objects. The final scalar reward from the last triplet is propagated to all preceding triplets following an "identical assignment strategy." The documentation explicitly states: "each triplet receives an identical reward signal and can be **independently optimized as a valid RLHF trajectory** within the VERL framework" (Microsoft Agent-Lightning docs, https://microsoft.github.io/agent-lightning/latest/algorithm-zoo/verl/).

verl-agent further proposes a "step-independent multi-turn rollout mechanism" that "allows for fully customizable per-step input structures, history management, and memory modules." The stated rationale: "keeping the context length almost constant over time and making it highly scalable for long-horizon scenarios" (https://github.com/langfengQ/verl-agent).

**Root cause revealed:** The design explicitly prioritizes compatibility with existing RLHF training loops. Multi-turn trajectories are forced into the (prompt, response, reward) triplet format that single-turn RLHF expects. This is an engineering convenience, not an algorithmic recommendation.

### RAGEN (StarPO)

RAGEN's StarPO framework supports both trajectory-level and per-step modes. The default critic-free regime assigns a scalar reward to each trajectory and computes normalized advantage across all tokens. However, when instantiated with PPO (which requires a critic), it falls back to token-level updates where a critic estimates token-level value and advantages (https://arxiv.org/html/2504.20073v2).

**Key finding:** RAGEN explicitly identifies that "vanilla adaptations from single-turn methods like PPO and GRPO achieve early gains in agent settings but often collapse." This confirms the problem exists but acknowledges it as the status quo that their work tries to fix, not something they introduced.

StarPO-S is their stabilized variant addressing this, suggesting the default (unstabilized) mode was inherited from prior practice.

### OpenRLHF

OpenRLHF's architecture uses a "unified token-in-token-out pipeline" that "decouples execution mode (single-turn/multi-turn) from the RL algorithm." While this sounds clean, the practical implication is that multi-turn is an extension layered on top of a fundamentally single-turn infrastructure.

Multiple GitHub issues document OOM problems even for small models:
- Issue #519: Llama3-8B OOM on 8xH100 during PPO training
- Issue #698: OOM on 80GB A100 with batch_size=1, seq_len=128
- Issue #1139: OOM on 4xH100 with 150M parameter models

These OOM issues with short sequences explain why the framework defaults to shorter, per-step training samples rather than full episode contexts.

### SkyRL-Agent

SkyRL-Agent explicitly "decomposes each trajectory into three stage jobs" and records "each LLM invocation as an individual transition tuple containing input tokens, output tokens, and their corresponding log probabilities" (https://arxiv.org/html/2511.16108v1).

**Rationale:** Efficiency. Their async dispatcher achieves 1.55x speedup by processing per-step transitions independently. Full-episode processing would create variable-length batches that waste GPU compute.

### Verlog (CMU)

Verlog most explicitly articulates the rationale: "To handle extremely long episodes, [the framework] treats each turn as an independent training sample. This eliminates the need to encode the entire trajectory into a single context window" (https://blog.ml.cmu.edu/2025/09/15/verlog-a-multi-turn-rl-framework-for-llm-agents/).

They further show that "the LLM agent achieves higher performance when using a shorter memory length during fine-tuning, with memory lengths of one and two consistently yielding the highest performance." This finding may have reinforced the community belief that step-independence is not just expedient but also optimal, though it likely reflects overfitting to short-horizon credit assignment rather than a true advantage.

### SUPO (ByteDance/Stanford/CMU)

SUPO explicitly diagnoses the problem: "context length of the underlying LLM fundamentally restricts the horizon of RL training, preventing agents from tackling tasks whose solution requires more tool calls than can fit into a single context window" (https://www.alphaxiv.org/overview/2510.06727v1).

Their solution (learned summarization to compress history) implicitly confirms that the default practice is to truncate or ignore history, i.e., step-independent mode is the fallback when context overflows.

## Root Causes (ranked by evidence strength)

### 1. Single-Turn RLHF Infrastructure Inheritance (Primary Cause)

**Evidence:** verl's explicit "Triplet" decomposition; OpenRLHF's "unified token-in-token-out pipeline" designed for single-turn first; GRPO/PPO algorithms designed for (prompt, response) pairs.

**Mechanism:** The entire LLM post-training ecosystem was built for single-turn RLHF (InstructGPT, ChatGPT-style). When researchers extended these frameworks to multi-turn agents, the path of least resistance was to decompose multi-turn trajectories into independent single-turn samples. verl makes this most explicit: each step becomes a "valid RLHF trajectory" that plugs directly into existing training code.

**Why it persists:** Rewriting the entire training pipeline (loss computation, batching, gradient accumulation, distributed training) for trajectory-level optimization is a massive engineering effort. Per-step decomposition "just works" with existing infrastructure.

### 2. Context Window Hard Constraint

**Evidence:** SUPO paper ("context length fundamentally restricts the horizon"); Verlog ("eliminates the need to encode the entire trajectory"); verl-agent ("keeping context length almost constant").

**Mechanism:** A 10-step agent episode with environment observations can easily reach 30K-50K tokens. Models with 4K-8K context (common during training) physically cannot fit full episodes. Even with 32K context models, training with full episodes means each sample consumes enormous GPU memory for attention computation (quadratic in sequence length).

**Quantitative evidence from SUPO:** Using 4K working context with summarization matches or beats 32K full-context training (47.7% vs 43.0% on CodeGym), suggesting the community may have empirically observed that full-context training doesn't help and concluded step-independence is fine.

### 3. GPU Memory / OOM Constraints

**Evidence:** OpenRLHF issues #519, #698, #1139 showing OOM even with small models and short sequences; the prevalence of gradient checkpointing, offloading, and micro-batching workarounds.

**Mechanism:** Full-episode training requires:
- Storing all token activations for backpropagation (O(L^2) for attention)
- KV cache during generation proportional to full episode length
- Critic/value network forward pass on full episode

Per-step decomposition reduces each training sample to a few hundred tokens, making memory manageable. This is not just about context window limits but about practical trainability.

### 4. Batching Efficiency

**Evidence:** SkyRL-Agent's 1.55x speedup from per-step decomposition; verl-agent's "constant context length" design.

**Mechanism:** Multi-turn episodes have highly variable lengths (some tasks solve in 2 steps, others need 20). Batching variable-length full episodes wastes compute through padding. Per-step samples are more uniform in length, enabling efficient GPU utilization. vLLM and other inference engines are optimized for uniform-length batch processing.

### 5. GRPO/PPO Algorithm Design Assumption

**Evidence:** ArCHer paper ("token-level methods face extremely long horizon, leading to numerical instabilities and slow convergence"); RC-GRPO paper (group-normalized advantage becomes "uninformative" with low within-group reward variation in multi-turn).

**Mechanism:** GRPO computes advantages by comparing responses within a group. In multi-turn settings with sparse terminal rewards, all steps in an episode get the same reward, causing advantage sign misattribution: the advantage sign is determined by episode outcome (success/failure) rather than window-local action quality. Good actions in failed episodes are incorrectly penalized; bad actions in successful episodes are incorrectly reinforced. The algorithm was designed for single-turn where each response has its own distinct reward. Using it per-step is a workaround for this fundamental mismatch, not a principled choice.

### 6. Empirical Misinterpretation

**Evidence:** Verlog finding that "memory length of 1-2 yields highest performance"; early RAGEN results showing per-step can work in simple environments.

**Mechanism:** On short-horizon tasks or tasks with strong per-step signals, step-independent training can appear to work fine or even outperform trajectory-level training (due to lower variance). This creates survivorship bias: researchers who tried it on easy tasks concluded it was sufficient, and never tested on tasks where cross-step credit assignment matters.

## Implications for Our Paper

### Framing the Contribution

Our paper can position the finding as: **The community's default of step-independent training is not a deliberate algorithmic choice but an engineering accident inherited from single-turn RLHF infrastructure, reinforced by context/memory constraints, and sustained by insufficient evaluation on tasks requiring multi-step credit assignment.**

### Specific Arguments to Make

1. **Infrastructure lock-in:** verl's "Triplet" decomposition and OpenRLHF's "unified pipeline" show that multi-turn was retrofitted onto single-turn code. The frameworks chose compatibility over correctness.

2. **Conflation of scalability with optimality:** verl-agent and Verlog explicitly chose step-independence for scalability (constant context length, efficient batching). But scalability != training effectiveness. Our results show this trade-off is catastrophic for tasks requiring temporal credit assignment.

3. **No framework warns users:** None of the frameworks document that step-independent mode may fail for multi-turn tasks. It's the silent default, not a flagged trade-off.

4. **The "it works on benchmarks" trap:** Most agent RL papers evaluate on tasks solvable in 2-5 steps (ALFWorld, WebShop, simple tool-use). On these, step-independent is adequate. Our contribution is showing failure on tasks requiring longer-horizon reasoning.

### Citation Targets

- verl-agent (GiGPO paper, arXiv:2505.10978) for explicit step-independent design rationale
- SUPO (arXiv:2510.06727) for context window as fundamental bottleneck
- ArCHer (arXiv:2402.19446) for theoretical framing of token-level vs utterance-level challenges
- Verlog (NeurIPS 2025) for "each turn as independent training sample" design
- RC-GRPO (arXiv:2602.03025) for GRPO failure modes in multi-turn

## Sources

- verl-agent: https://github.com/langfengQ/verl-agent
- VERL/Agent-Lightning docs: https://microsoft.github.io/agent-lightning/latest/algorithm-zoo/verl/
- RAGEN/StarPO: https://arxiv.org/html/2504.20073v2, https://github.com/RAGEN-AI/RAGEN
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
- SkyRL-Agent: https://arxiv.org/html/2511.16108v1
- Verlog: https://blog.ml.cmu.edu/2025/09/15/verlog-a-multi-turn-rl-framework-for-llm-agents/
- SUPO: https://www.alphaxiv.org/overview/2510.06727v1
- ArCHer: https://arxiv.org/abs/2402.19446
- RC-GRPO: https://arxiv.org/html/2602.03025
- GiGPO: https://arxiv.org/abs/2505.10978
