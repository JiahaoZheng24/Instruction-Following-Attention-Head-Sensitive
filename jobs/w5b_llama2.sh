#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W5b
#$ -t 1-6
# W5b: complete the Llama collapse-regime picture.
#  1-2: does GEOMETRY alone (g64) rescue the collapse, or is weight IDENTITY
#       required? (the two-regime story's key discriminator)
#  3:   llama 4-bit baseline (expected graceful — the regime boundary)
#  4-6: MMLU/PPL for fp16 / v2l_none / v2l_tacq (is the rescue capability-general?)
#   qsub jobs/w5b_llama2.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
FULL="data/ifeval_input_data.jsonl"
BUDGET=40200000
SAL_DIR="$STORE/salience/llama31-8b-if"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IFH_OFFLOAD_DIR="${TMPDIR:-/tmp}/gptq_offload_${JOB_ID}_${SGE_TASK_ID}"
cd "$BASE_DIR"

run_ifeval () {  # $1 = ckpt dir/id, $2 = tag
  python src/diagnose_heads.py ablate --model "$1" --prompts "$FULL" --tag "$2" --batch 16
  python src/score_ifeval.py --responses "runs/$(basename $1)/$2/responses.jsonl" \
    --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}

case "$SGE_TASK_ID" in
  1) CKPT="$STORE/models/llama3.1-8b-v2gptq3g64-none"
     python src/quantize_protected.py --model "$MODEL_ID" --bits 3 --group-size 64 \
       --out "$CKPT" --protect none
     run_ifeval "$CKPT" v2lg64_none ;;
  2) CKPT="$STORE/models/llama3.1-8b-v2gptq3g64-tacq"
     python src/quantize_protected.py --model "$MODEL_ID" --bits 3 --group-size 64 \
       --out "$CKPT" --protect tacq --salience-dir "$SAL_DIR" --budget-params $BUDGET
     run_ifeval "$CKPT" v2lg64_tacq ;;
  3) CKPT="$STORE/models/llama3.1-8b-gptq4-c4-g128"
     python src/quantize_gptq.py --model "$MODEL_ID" --bits 4 --group-size 128 \
       --calib c4 --out "$CKPT"
     run_ifeval "$CKPT" qbase_llama_gptq4 ;;
  4) python src/eval_general.py --model "$MODEL_ID" --tag gen_llama_fp16 \
       --batch 8 --scores-csv runs/general_gen_llama_fp16.csv ;;
  5) python src/eval_general.py --model "$STORE/models/llama3.1-8b-v2gptq3-none" \
       --tag gen_v2l_none --batch 8 --scores-csv runs/general_gen_v2l_none.csv ;;
  6) python src/eval_general.py --model "$STORE/models/llama3.1-8b-v2gptq3-tacq" \
       --tag gen_v2l_tacq --batch 8 --scores-csv runs/general_gen_v2l_tacq.csv ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W5b] ✅ task $SGE_TASK_ID done"
