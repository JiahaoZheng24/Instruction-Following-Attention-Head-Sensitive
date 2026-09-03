#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W17
#$ -t 1-8
# W17: paper-grade completeness batch (the last one).
#  1: qwen14b RTN3 + tacq@1e6  — completes the 2x2 (correction x protection)
#     on collapse instance #2 (currently Llama-only, p=0.03 borderline)
#  2: qwen14b critical-set anatomy @1e6 — predicts LOW channel concentration
#     (would EXPLAIN why column-protection fails on 14B vs Llama's 69.7%)
#  3: llama tacq@40.2M with calib-seed 1 — quantization-seed replicate of the
#     +0.493 rescue (none already has a two-pipeline replicate; tacq doesn't)
#  4-8: PPL+MMLU on the five middle-tier census checkpoints — fills the
#     likelihood-vs-generation detection map from 8 to 13 points
#   qsub jobs/w17_paper_grade.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
LLAMA="meta-llama/Llama-3.1-8B-Instruct"
Q14="Qwen/Qwen2.5-14B-Instruct"
FULL="data/ifeval_input_data.jsonl"
SAL14="$STORE/salience/qwen25-14b-if"
SAL_L="$STORE/salience/llama31-8b-if"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

run_ifeval () {
  python src/diagnose_heads.py ablate --model "$1" --prompts "$FULL" --tag "$2" --batch 16
  python src/score_ifeval.py --responses "runs/$(basename $1)/$2/responses.jsonl" \
    --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}

case "$SGE_TASK_ID" in
  1) CKPT="$STORE/models/qwen2.5-14b-rtn3-tacq1e6"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --rtn --protect tacq --salience-dir "$SAL14" --budget-params 1000000 \
       --out "$CKPT"
     run_ifeval "$CKPT" rtn3q14_tacq1e6 ;;
  2) python src/critical_anatomy.py --salience-dir "$SAL14" \
       --budget-params 1000000 --model "$Q14" \
       --out runs/critical_anatomy_q14.csv ;;
  3) CKPT="$STORE/models/llama3.1-8b-v2gptq3-tacq_cs1"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --protect tacq --salience-dir "$SAL_L" --budget-params 40200000 \
       --calib-seed 1 --out "$CKPT"
     run_ifeval "$CKPT" v2l_tacq_cs1 ;;
  4) python src/eval_general.py --model "$STORE/models/llama3.2-3b-v2gptq3-none" \
       --tag gen_l32_none --scores-csv runs/general_gen_l32_none.csv ;;
  5) python src/eval_general.py --model "$STORE/models/llama3-8b-v2gptq3-none" \
       --tag gen_l3_none --scores-csv runs/general_gen_l3_none.csv ;;
  6) python src/eval_general.py --model "$STORE/models/yi1.5-9b-v2gptq3-none" \
       --tag gen_yi_none --scores-csv runs/general_gen_yi_none.csv ;;
  7) python src/eval_general.py --model "$STORE/models/mistral-nemo-12b-v2gptq3-none" \
       --tag gen_nemo_none --scores-csv runs/general_gen_nemo_none.csv ;;
  8) python src/eval_general.py --model "$STORE/models/qwen2.5-3b-v2gptq3-none" \
       --tag gen_q3_none --scores-csv runs/general_gen_q3_none.csv ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W17] ✅ task $SGE_TASK_ID done"
