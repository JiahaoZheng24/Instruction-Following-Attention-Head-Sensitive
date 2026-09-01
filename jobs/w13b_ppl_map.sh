#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W13b
#$ -t 1-6
# W13b: complete the PPL/MMLU detection map. The "quantize -> check PPL ->
# collapse detected" decision rule needs BOTH sides: collapse arms show high
# PPL (llama 59.2; q14 pending in W13), graceful arms must show low PPL —
# currently only qwen7 (9.7) covers the graceful side. Pure evals.
#  1: mistral-7b fp16          2: mistral-7b gptq3 none
#  3: qwen2.5-32b fp16         4: qwen2.5-32b gptq3 none
#  5: mistral-24b fp16         6: mistral-24b gptq3 none
#   qsub jobs/w13b_ppl_map.sh

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

gen () {  # $1 model, $2 tag
  python src/eval_general.py --model "$1" --tag "$2" \
    --scores-csv "runs/general_$2.csv"
}

case "$SGE_TASK_ID" in
  1) gen mistralai/Mistral-7B-Instruct-v0.3               gen_mistral_fp16 ;;
  2) gen "$STORE/models/mistral-7b-v2gptq3-none"          gen_mistral_none ;;
  3) gen Qwen/Qwen2.5-32B-Instruct                        gen_q32_fp16 ;;
  4) gen "$STORE/models/qwen2.5-32b-v2gptq3-none"         gen_q32_none ;;
  5) gen mistralai/Mistral-Small-24B-Instruct-2501        gen_m24_fp16 ;;
  6) gen "$STORE/models/mistral-24b-v2gptq3-none"         gen_m24_none ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W13b] ✅ task $SGE_TASK_ID done"
