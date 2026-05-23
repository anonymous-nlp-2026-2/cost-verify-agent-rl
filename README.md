# Cost-Verify Agent RL

Code for "Sign Misattribution in Step-Independent Agent RL: Dual Mechanism Diagnosis and Episode-Level Correction"

## Overview

This repository implements episode-level GRPO training for agentic reinforcement learning, addressing the **sign misattribution problem** in step-independent advantage estimation. We provide:

- Diagnostic tools for analyzing credit assignment failures in multi-step agent RL
- Episode-level advantage correction that resolves sign misattribution
- Full training pipeline (SFT → GRPO) for ALFWorld interactive environments
- Comprehensive evaluation scripts with per-task-type analysis

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

## Usage

### SFT Training (Expert Trajectories)
```bash
bash scripts/run_expert_sft.sh
```

### GRPO Training — Step-Independent Baseline
```bash
bash scripts/run_exp026_step_independent_grpo.sh
```

### GRPO Training — Episode-Level (Ours)
```bash
bash scripts/run_exp029_episode_level.sh
```

### Evaluation
```bash
python scripts/eval_checkpoint.py --config RAGEN/config/_alfworld_8b_det_eval_full.yaml --checkpoint <path>
```

## Project Structure
```
RAGEN/                  # Core framework (based on RAGEN)
├── ragen/
│   ├── biavr/          # Binary Individual Advantage Verification & Reward
│   ├── trainer/        # Agent trainer with episode-level GRPO
│   ├── env/            # Environment wrappers (ALFWorld, etc.)
│   └── llm_agent/      # LLM agent with context management
├── verl/               # Distributed RL training backend
└── config/             # Training & evaluation configs
scripts/                # Experiment launch scripts
analysis/               # Credit assignment analysis tools
figures/                # Generated figures
tests/                  # Unit tests
```

## Key Components

- **Episode-Level GRPO** (`RAGEN/ragen/trainer/`): Corrects advantage sign misattribution by computing advantages at the episode level rather than per-step.
- **BIAVR** (`RAGEN/ragen/biavr/`): Binary Individual Advantage Verification mechanism for reward assignment.
- **Gradient Analysis** (`RAGEN/gradient_analysis/`): Tools for diagnosing gradient sign conflicts in step-independent training.

## License
MIT
