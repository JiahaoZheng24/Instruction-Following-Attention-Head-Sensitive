#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W7_crit
#$ -t 1-8
# W7: pin down the critical set. Method-generality (RTN), task-generality
# (c4-conditioned salience), fp16 criticality (zero probe), and the rescue
# window's lower bound (2-bit budget escalation).
#  1: llama RTN-3bit none        (does collapse survive a different quantizer?)
#  2: llama RTN-3bit tacq@1e5    (does the SAME critical set rescue RTN?)
#  3: qwen  RTN-3bit none        (graceful regime under RTN, reference)
#  4: llama c4-conditioned salience  -> $STORE/salience/llama31-8b-c4
#  5: qwen  c4-conditioned salience  -> $STORE/salience/qwen25-7b-c4
#  6: llama fp16 zero-probe @1e5 (prune the critical set at fp16: collapse?)
#  7: qwen 2-bit tacq @2% budget  (rescue window lower bound)
#  8: qwen 2-bit tacq @5% budget
# Then: qsub -hold_jid IFH_W7_crit jobs/w7b_anatomy.sh
#   qsub jobs/w7_critical.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
LLAMA="meta-llama/Llama-3.1-8B-Instruct"
QWEN="Qwen/Qwen2.5-7B-Instruct"
FULL="data/ifeval_input_data.jsonl"
SAL_L="$STORE/salience/llama31-8b-if"
SAL_Q="$STORE/salience/qwen25-7b-if"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

run_ifeval () {  # $1 ckpt, $2 tag
  python src/diagnose_heads.py ablate --model "$1" --prompts "$FULL" --tag "$2" --batch 16
  python src/score_ifeval.py --responses "runs/$(basename $1)/$2/responses.jsonl" \
    --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}

case "$SGE_TASK_ID" in
  1) CKPT="$STORE/models/llama3.1-8b-rtn3-none"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --rtn --protect none --out "$CKPT"
     run_ifeval "$CKPT" rtn3l_none ;;
  2) CKPT="$STORE/models/llama3.1-8b-rtn3-tacq1e5"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --rtn --protect tacq --salience-dir "$SAL_L" --budget-params 100000 --out "$CKPT"
     run_ifeval "$CKPT" rtn3l_tacq1e5 ;;
  3) CKPT="$STORE/models/qwen2.5-7b-rtn3-none"
     python src/quantize_protected.py --model "$QWEN" --bits 3 --group-size 128 \
       --rtn --protect none --out "$CKPT"
     run_ifeval "$CKPT" rtn3q_none ;;
  4) mkdir -p "$STORE/salience/llama31-8b-c4"
     python src/tacq_salience.py --model "$LLAMA" --calib-kind c4 --bits 3 \
       --out-dir "$STORE/salience/llama31-8b-c4" ;;
  5) mkdir -p "$STORE/salience/qwen25-7b-c4"
     python src/tacq_salience.py --model "$QWEN" --calib-kind c4 --bits 3 \
       --out-dir "$STORE/salience/qwen25-7b-c4" ;;
  6) python src/zero_probe.py --model "$LLAMA" --salience-dir "$SAL_L" \
       --budget-params 100000 --prompts "$FULL" --tag zero1e5_llama
     python src/score_ifeval.py \
       --responses "runs/Llama-3.1-8B-Instruct/zero1e5_llama/responses.jsonl" \
       --input-data "$FULL" --tag zero1e5_llama --scores-csv runs/scores_zero1e5_llama.csv ;;
  7) CKPT="$STORE/models/qwen2.5-7b-v2gptq2-b2_tacq2pct"
     python src/quantize_protected.py --model "$QWEN" --bits 2 --group-size 128 \
       --protect tacq --salience-dir "$SAL_Q" --budget-params 152000000 --out "$CKPT"
     run_ifeval "$CKPT" v2_b2_tacq2pct ;;
  8) CKPT="$STORE/models/qwen2.5-7b-v2gptq2-b2_tacq5pct"
     python src/quantize_protected.py --model "$QWEN" --bits 2 --group-size 128 \
       --protect tacq --salience-dir "$SAL_Q" --budget-params 380000000 --out "$CKPT"
     run_ifeval "$CKPT" v2_b2_tacq5pct ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W7] ✅ task $SGE_TASK_ID done"
