#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W8_final
#$ -t 1-12
# W8 (final experimental batch): mechanism closure + generalization armor.
#  Mechanism:
#   1: llama gptq3 rescued by the C4-conditioned mask @1e5 (Jaccard puzzle)
#   2: llama gptq3 COLUMN-structured protection @40M (deployable-method demo)
#   3: llama gptq3 without desc_act (compensation-order probe)
#   4: weight displacement |W_q-W_fp| on critical vs random, GPTQ vs RTN
#  Within-model regime contrast (does the law follow regime, not model?):
#   5: llama gptq4 in-loop none (graceful reference on Llama)
#   6: llama gptq4 tacq@40.2M   (prediction: ~useless, like Qwen@3bit)
#   7: llama gptq4 randw@40.2M  (prediction: ~= tacq)
#  Third model + regime predictor:
#   8: mistral gptq3 none  9: mistral rtn3 none  10: mistral IF-salience
#   11: regime markers for all three models (one forward each)
#   12: llama gptq4 heads32 (head-null third model-regime instance)
#   qsub jobs/w8_final.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
LLAMA="meta-llama/Llama-3.1-8B-Instruct"
QWEN="Qwen/Qwen2.5-7B-Instruct"
MISTRAL="mistralai/Mistral-7B-Instruct-v0.3"
FULL="data/ifeval_input_data.jsonl"
SAL_L="$STORE/salience/llama31-8b-if"
SAL_LC4="$STORE/salience/llama31-8b-c4"
BUDGET_L=40200000

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

run_ifeval () {  # $1 ckpt, $2 tag
  python src/diagnose_heads.py ablate --model "$1" --prompts "$FULL" --tag "$2" --batch 16
  python src/score_ifeval.py --responses "runs/$(basename $1)/$2/responses.jsonl" \
    --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}

case "$SGE_TASK_ID" in
  1) CKPT="$STORE/models/llama3.1-8b-v2gptq3-tacqc4"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --protect tacq --salience-dir "$SAL_LC4" --budget-params 100000 --out "$CKPT"
     run_ifeval "$CKPT" v2l_tacqc4_1e5 ;;
  2) CKPT="$STORE/models/llama3.1-8b-v2gptq3-cols40m"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --protect cols --salience-dir "$SAL_L" --budget-params 40000000 --out "$CKPT"
     run_ifeval "$CKPT" v2l_cols40m ;;
  3) CKPT="$STORE/models/llama3.1-8b-v2gptq3-noactorder"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --no-actorder --protect none --out "$CKPT"
     run_ifeval "$CKPT" v2l_noactorder ;;
  4) python src/weight_displacement.py --fp-model "$LLAMA" \
       --ckpt-gptq "$STORE/models/llama3.1-8b-v2gptq3-none" \
       --ckpt-rtn "$STORE/models/llama3.1-8b-rtn3-none" \
       --salience-dir "$SAL_L" --budget-params 100000 \
       --out runs/weight_displacement_llama.csv ;;
  5) CKPT="$STORE/models/llama3.1-8b-v2gptq4-none"
     python src/quantize_protected.py --model "$LLAMA" --bits 4 --group-size 128 \
       --protect none --out "$CKPT"
     run_ifeval "$CKPT" v2l4_none ;;
  6) CKPT="$STORE/models/llama3.1-8b-v2gptq4-tacq"
     python src/quantize_protected.py --model "$LLAMA" --bits 4 --group-size 128 \
       --protect tacq --salience-dir "$SAL_L" --budget-params $BUDGET_L --out "$CKPT"
     run_ifeval "$CKPT" v2l4_tacq ;;
  7) CKPT="$STORE/models/llama3.1-8b-v2gptq4-randw"
     python src/quantize_protected.py --model "$LLAMA" --bits 4 --group-size 128 \
       --protect randw --budget-params $BUDGET_L --seed 0 --out "$CKPT"
     run_ifeval "$CKPT" v2l4_randw ;;
  8) CKPT="$STORE/models/mistral-7b-v2gptq3-none"
     python src/quantize_protected.py --model "$MISTRAL" --bits 3 --group-size 128 \
       --protect none --out "$CKPT"
     run_ifeval "$CKPT" v2m_none ;;
  9) CKPT="$STORE/models/mistral-7b-rtn3-none"
     python src/quantize_protected.py --model "$MISTRAL" --bits 3 --group-size 128 \
       --rtn --protect none --out "$CKPT"
     run_ifeval "$CKPT" rtn3m_none ;;
  10) mkdir -p "$STORE/salience/mistral7b-if"
      python src/tacq_salience.py --model "$MISTRAL" \
        --calib-file data/calib_prompts.jsonl --bits 3 \
        --out-dir "$STORE/salience/mistral7b-if" ;;
  11) python src/regime_marker.py --model "$QWEN"    --out runs/regime_markers.csv
      python src/regime_marker.py --model "$LLAMA"   --out runs/regime_markers.csv
      python src/regime_marker.py --model "$MISTRAL" --out runs/regime_markers.csv ;;
  12) CKPT="$STORE/models/llama3.1-8b-v2gptq4-heads32"
      python src/quantize_protected.py --model "$LLAMA" --bits 4 --group-size 128 \
        --protect heads --topk-from runs/dev_ranking_llama3.csv --k 32 --kv --projs all \
        --out "$CKPT"
      run_ifeval "$CKPT" v2l4_heads32 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W8] ✅ task $SGE_TASK_ID done"
