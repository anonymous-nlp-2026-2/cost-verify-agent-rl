#!/bin/bash
# FSDP → HF checkpoint converter for Phase B
# Usage: ./convert_fsdp_to_hf.sh <fsdp_checkpoint_dir> <output_hf_dir>
# Example: ./convert_fsdp_to_hf.sh ./checkpoints/exp012_grpo/global_step_30 ./models/phase_a_hf

set -euo pipefail

FSDP_DIR=${1:?Usage: $0 <fsdp_dir> <hf_output_dir>}
HF_DIR=${2:?Usage: $0 <fsdp_dir> <hf_output_dir>}

source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

echo "Converting FSDP checkpoint: $FSDP_DIR → $HF_DIR"
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$FSDP_DIR" \
    --target_dir "$HF_DIR" \
    --trust-remote-code

echo "Verifying output..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('$HF_DIR', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('$HF_DIR', trust_remote_code=True)
print(f'Model loaded: {model.config.architectures}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M')
print('Conversion verified OK')
"
