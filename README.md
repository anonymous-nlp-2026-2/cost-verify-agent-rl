# Sign Misattribution in Step-Independent Agent RL

Code and data for **"Sign Misattribution in Step-Independent Agent RL: Dual Mechanism Diagnosis and Episode-Level Correction"** (EMNLP 2026 submission).

## Abstract

Reinforcement learning (RL) for multi-turn large language model (LLM) agents requires credit assignment across long interaction trajectories. The dominant training paradigm decomposes these trajectories into fixed-size windows optimized independently via Group Relative Policy Optimization (GRPO), yet this design choice has received little scrutiny. We identify a structural credit assignment pathology driven by two distinct mechanisms: *policy information loss*, which truncates cross-window context and imposes a constant performance ceiling, and *advantage sign misattribution*, which causes a substantial fraction of gradient updates for locally correct actions to carry incorrect sign. A partial factorial experiment isolates each mechanism: removing sign misattribution converts catastrophic collapse into bounded learning; additionally removing information loss enables out-of-distribution (OOD) generalization. Episode-level GRPO, which treats the full trajectory as a single optimization sample, structurally eliminates both failure modes but remains seed-dependent: 17.7% mean OOD success versus 6.72% SFT baseline, though only 36% of seeds achieve significant generalization. Cross-scale verification at 8B parameters and cross-environment validation on ScienceWorld provide preliminary evidence that the pathology extends beyond our primary setting.

## Repository Structure

```
├── RAGEN/                  # Core framework (extended from RAGEN v0.1)
│   ├── ragen/
│   │   ├── biavr/          # Action canonicalization and reward verification
│   │   ├── trainer/        # Agent trainer with episode-level GRPO support
│   │   ├── env/            # Environment wrappers (ALFWorld, ScienceWorld, etc.)
│   │   └── llm_agent/      # LLM agent with context management
│   ├── verl/               # Distributed RL training backend
│   └── config/             # Training & evaluation configs
├── scripts/                # Experiment launch scripts
├── analysis/               # Credit assignment analysis and documentation
├── figures/                # Figure generation scripts for reproducibility
│   └── paper/              # Paper figure generation (theoretical curves, training plots)
├── tests/                  # Unit tests
├── eval_ood_seed*.py       # OOD evaluation scripts (134-task protocol)
├── eval_per_task_type.py   # Per-task-type evaluation breakdown
├── eval_scienceworld*.py   # ScienceWorld cross-environment evaluation
└── analyze_training.py     # Training log analysis
```

## Setup

### Requirements
- Python 3.10+
- PyTorch 2.x with CUDA support
- vLLM for efficient inference during rollout

### Installation
```bash
cd RAGEN
pip install -e .
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

## Reproducing Experiments

### 1. SFT Baseline (Expert Trajectories)
```bash
bash scripts/run_expert_sft.sh
```

### 2. Cell A: Step-Independent GRPO (h=2)
```bash
bash scripts/run_exp026_step_independent_grpo.sh
```

### 3. Cell D: Episode-Level GRPO
```bash
bash scripts/run_exp029_episode_level.sh
```

### 4. Window-Size Sweep
```bash
bash scripts/launch_hsweep.sh
```

### 5. Multi-Seed Evaluation
```bash
bash scripts/launch_multiseed.sh
```

### 6. OOD Evaluation (134-task protocol)
```bash
python scripts/eval_checkpoint.py \
  --config RAGEN/config/_alfworld_8b_det_eval_full.yaml \
  --checkpoint <path>
```

### 7. Credit Assignment Analysis
```bash
python scripts/analyze_advantage_log.py --log_dir <advantage_log_dir>
```

## Key Results

| Condition | Mechanisms | Best Step | OOD (%) |
|-----------|-----------|-----------|---------|
| SFT baseline | --- | --- | 6.72 |
| Cell A (h=2, step) | M1+M2 | 5 | — |
| Cell B (h=2, episode) | M1 only | 5 | 5.97 |
| Cell D (full, episode) | neither | 15 | **17.16** |

- **M1**: Policy information loss (truncated cross-window context)
- **M2**: Advantage sign misattribution (gradient sign determined by episode outcome, not action quality)

Episode-level GRPO across 11 seeds: 27% collapse, 36% stagnate, **36% generalize** (mean OOD: 17.7%).

## Figure Reproduction

```bash
# Theoretical sign probability curve (Figure 3)
python figures/paper/gen_fig_theoretical_sign_probability.py

# Training curves (Figure 1)
python figures/paper/gen_fig6_training_curves.py
```

## License

MIT
