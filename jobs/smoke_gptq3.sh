#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -N IFH_smoke3
# Quick kernel-bug verification: re-run the 3-bit smoke test with the plain
# PyTorch backend. Expected if the Triton-kernel diagnosis is right: coherent
# (possibly degraded) text that VARIES across the three prompts — instead of
# the identical garbage seen with the default backend.

set -e
STORE="/store01/yshi4/jzheng7"
export HF_HOME="$STORE/hf_cache"
source ~/.bashrc 2>/dev/null || true
conda activate IFEval
cd "$STORE/Instruction-Following-Attention-Head-Sensitive"

python src/quantize_gptq.py --smoke-only --bits 3 --backend TORCH \
  --out "$STORE/models/qwen2.5-7b-gptq3-c4-g128"
