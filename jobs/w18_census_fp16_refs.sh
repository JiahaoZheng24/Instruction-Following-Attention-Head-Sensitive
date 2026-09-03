#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W18
#$ -t 1-5
# W18: fp16 PPL/MMLU references for the census middle tier — required to
# interpret two W17 anomalies before anything enters the paper:
#  - Llama-3.2-3B quantized PPL = 1749 (collapse-grade!) yet mild generation
#    damage -> REVERSE dissociation if its fp16 PPL is normal; protocol
#    artifact if fp16 is also huge.
#  - Yi-9B quantized MMLU = 0.23 (below chance) -> real knowledge collapse if
#    fp16 is ~0.7; harness/tokenizer artifact if fp16 is also ~0.25.
#   qsub jobs/w18_census_fp16_refs.sh

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
  1) python src/eval_general.py --model meta-llama/Llama-3.2-3B-Instruct \
       --tag gen_l32_fp16 --scores-csv runs/general_gen_l32_fp16.csv ;;
  2) python src/eval_general.py --model meta-llama/Meta-Llama-3-8B-Instruct \
       --tag gen_l3_fp16 --scores-csv runs/general_gen_l3_fp16.csv ;;
  3) python src/eval_general.py --model 01-ai/Yi-1.5-9B-Chat \
       --tag gen_yi_fp16 --scores-csv runs/general_gen_yi_fp16.csv ;;
  4) python src/eval_general.py --model mistralai/Mistral-Nemo-Instruct-2407 \
       --tag gen_nemo_fp16 --scores-csv runs/general_gen_nemo_fp16.csv ;;
  5) python src/eval_general.py --model Qwen/Qwen2.5-3B-Instruct \
       --tag gen_q3_fp16 --scores-csv runs/general_gen_q3_fp16.csv ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W18] ✅ task $SGE_TASK_ID done"
