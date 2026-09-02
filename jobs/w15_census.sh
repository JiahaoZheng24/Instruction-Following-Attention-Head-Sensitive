#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W15
#$ -t 1-6
# W15: regime CENSUS — answer "only two collapse instances?" with a survey.
# Six more models, each: fp16 IFEval baseline + GPTQ-3bit unprotected + IFEval.
# No salience, no protection arms — pure prevalence statement (k of 12 models
# collapse at 3-bit). If a new collapse appears -> follow up with salience+tacq
# for a third rescue instance.
# Model notes: all llama-style module layouts (our pipeline's assumption);
# task 3 (Meta-Llama-3-8B) is a separate gated repo from 3.1 — if it 401s,
# accept the license on HF or skip it.
#   qsub jobs/w15_census.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
FULL="data/ifeval_input_data.jsonl"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

census () {  # $1 hf model id, $2 short tag, $3 local ckpt name
  python src/diagnose_heads.py ablate --model "$1" --prompts "$FULL" \
    --tag "cens_$2_fp16" --batch 16
  python src/score_ifeval.py \
    --responses "runs/$(basename $1)/cens_$2_fp16/responses.jsonl" \
    --input-data "$FULL" --tag "cens_$2_fp16" \
    --scores-csv "runs/scores_cens_$2_fp16.csv"
  CKPT="$STORE/models/$3-v2gptq3-none"
  python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 \
    --protect none --out "$CKPT"
  python src/diagnose_heads.py ablate --model "$CKPT" --prompts "$FULL" \
    --tag "cens_$2_gptq3" --batch 16
  python src/score_ifeval.py \
    --responses "runs/$(basename $CKPT)/cens_$2_gptq3/responses.jsonl" \
    --input-data "$FULL" --tag "cens_$2_gptq3" \
    --scores-csv "runs/scores_cens_$2_gptq3.csv"
}

case "$SGE_TASK_ID" in
  1) census meta-llama/Llama-3.2-3B-Instruct        l32_3b   llama3.2-3b ;;
  2) census Qwen/Qwen2.5-3B-Instruct                q25_3b   qwen2.5-3b ;;
  3) census meta-llama/Meta-Llama-3-8B-Instruct     l3_8b    llama3-8b ;;
  4) census 01-ai/Yi-1.5-9B-Chat                    yi_9b    yi1.5-9b ;;
  5) census mistralai/Mistral-Nemo-Instruct-2407    nemo_12b mistral-nemo-12b ;;
  6) census allenai/OLMo-2-1124-7B-Instruct         olmo_7b  olmo2-7b ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W15] ✅ task $SGE_TASK_ID done"
