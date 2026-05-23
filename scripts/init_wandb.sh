#!/usr/bin/env bash
set -euo pipefail

# W&B initialization for cost-verify-agent-rl
# Source this before training: source scripts/init_wandb.sh

# Reject disabled mode
if [ "${WANDB_MODE:-}" = "disabled" ]; then
    echo "ERROR: WANDB_MODE=disabled is set. Phase B requires W&B logging."
    echo "Unset it: unset WANDB_MODE"
    exit 1
fi

# Check API key
if [ -z "${WANDB_API_KEY:-}" ]; then
    if [ ! -f "$HOME/.netrc" ] || ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
        echo "ERROR: WANDB_API_KEY not set and no .netrc credentials found."
        echo "Run: wandb login"
        exit 1
    fi
fi

# Project config
export WANDB_PROJECT="${WANDB_PROJECT:-cost-verify-agent-rl}"

# Entity (uncomment and set if needed)
# export WANDB_ENTITY="${WANDB_ENTITY:-your-team}"

echo "W&B initialized: project=${WANDB_PROJECT}"
