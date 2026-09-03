#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W19
#$ -t 1-3
# W19 (diagnostic, minutes each): pin down the MICROSTRUCTURE of the
# Llama-3.2-3B 158x perplexity explosion before it goes in the paper.
# Hypothesis: a few catastrophic positions (tails miscalibrated, argmax
# intact) -> generation survives. Prediction: median NLL near-normal, ppl
# collapses back once the worst <1% tokens are excluded, top-1 agreement
# barely moves. Control: the true collapse (Llama-8B gptq3) should look
# different (broad degradation).
#  1: l32 fp16   2: l32 gptq3   3: llama-8b gptq3 (collapse control)
#   qsub jobs/w19_ppl_diag.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

case "$SGE_TASK_ID" in
  1) python src/ppl_forensics.py --model meta-llama/Llama-3.2-3B-Instruct \
       --tag pplf_l32_fp16 --out runs/ppl_forensics.csv ;;
  2) python src/ppl_forensics.py --model "$STORE/models/llama3.2-3b-v2gptq3-none" \
       --tag pplf_l32_gptq3 --out runs/ppl_forensics.csv ;;
  3) python src/ppl_forensics.py --model "$STORE/models/llama3.1-8b-v2gptq3-none" \
       --tag pplf_llama8_gptq3 --out runs/ppl_forensics.csv ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W19] ✅ task $SGE_TASK_ID done"
