#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W13
#$ -t 1-12
# W13 (rigor closeout — audit-driven): the controls and replicates a hostile
# reviewer would demand. After this: hard freeze for writing.
#  Mechanism discriminators (missing controls):
#   1: POST-HOC restore of critical@1e5 on the COLLAPSED llama ckpt
#      (rescues => corrupted-weights story; fails => compensation-propagation)
#   2: zero-probe RANDOM control @1e5 on llama fp16 (calibrates "lethal")
#  Robustness of the second collapse:
#   3: qwen14b gptq3 none, calib-seed 1 (llama collapse has a 2-pipeline
#      replication; q14 currently rests on ONE run)
#  Predictors must cover the models the non-monotonicity claim lives on:
#   4: regime markers Q14/Q32/M24 + salience concentration incl. qwen14
#  Detection claim on collapse #2:
#   5-7: MMLU+PPL on q14 fp16 / none / tacq
#  Critical-set-scale universality + structure at 14B:
#   8: q14 tacq@1e5   9: q14 tacq@1e6   10: q14 cols@70M
#  Capability generality at 14B + Multi-IF reference patch:
#   11: GSM8K q14 fp16+none+tacq   12: Multi-IF qwen v2none
#   qsub jobs/w13_rigor.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
LLAMA="meta-llama/Llama-3.1-8B-Instruct"
Q14="Qwen/Qwen2.5-14B-Instruct"
Q32="Qwen/Qwen2.5-32B-Instruct"
M24="mistralai/Mistral-Small-24B-Instruct-2501"
FULL="data/ifeval_input_data.jsonl"
SAL_L="$STORE/salience/llama31-8b-if"
SAL14="$STORE/salience/qwen25-14b-if"

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
  1) python src/posthoc_scatter.py --quant-ckpt "$STORE/models/llama3.1-8b-v2gptq3-none" \
       --model "$LLAMA" --salience-dir "$SAL_L" --budget-params 100000 \
       --prompts "$FULL" --tag ph_crit_llama
     python src/score_ifeval.py \
       --responses "runs/llama3.1-8b-v2gptq3-none/ph_crit_llama/responses.jsonl" \
       --input-data "$FULL" --tag ph_crit_llama --scores-csv runs/scores_ph_crit_llama.csv ;;
  2) python src/zero_probe.py --model "$LLAMA" --salience-dir "$SAL_L" \
       --budget-params 100000 --random --seed 0 --prompts "$FULL" --tag zero_rand_llama
     python src/score_ifeval.py \
       --responses "runs/Llama-3.1-8B-Instruct/zero_rand_llama/responses.jsonl" \
       --input-data "$FULL" --tag zero_rand_llama --scores-csv runs/scores_zero_rand_llama.csv ;;
  3) CKPT="$STORE/models/qwen2.5-14b-v2gptq3-none_cs1"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --protect none --calib-seed 1 --out "$CKPT"
     run_ifeval "$CKPT" v2q14_none_cs1 ;;
  4) python src/regime_marker.py --model "$Q14" --out runs/regime_markers.csv
     python src/regime_marker.py --model "$Q32" --out runs/regime_markers.csv
     python src/regime_marker.py --model "$M24" --out runs/regime_markers.csv
     python src/salience_concentration.py \
       --dirs qwen14="$SAL14" \
       --out runs/salience_concentration_14b.csv ;;
  5) python src/eval_general.py --model "$Q14" --tag gen_q14_fp16 \
       --scores-csv runs/general_gen_q14_fp16.csv ;;
  6) python src/eval_general.py --model "$STORE/models/qwen2.5-14b-v2gptq3-none" \
       --tag gen_q14_none --scores-csv runs/general_gen_q14_none.csv ;;
  7) python src/eval_general.py --model "$STORE/models/qwen2.5-14b-v2gptq3-tacq" \
       --tag gen_q14_tacq --scores-csv runs/general_gen_q14_tacq.csv ;;
  8) CKPT="$STORE/models/qwen2.5-14b-v2gptq3-tacq1e5"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --protect tacq --salience-dir "$SAL14" --budget-params 100000 --out "$CKPT"
     run_ifeval "$CKPT" v2q14_tacq1e5 ;;
  9) CKPT="$STORE/models/qwen2.5-14b-v2gptq3-tacq1e6"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --protect tacq --salience-dir "$SAL14" --budget-params 1000000 --out "$CKPT"
     run_ifeval "$CKPT" v2q14_tacq1e6 ;;
  10) CKPT="$STORE/models/qwen2.5-14b-v2gptq3-cols70m"
      python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
        --protect cols --salience-dir "$SAL14" --budget-params 70000000 --out "$CKPT"
      run_ifeval "$CKPT" v2q14_cols70m ;;
  11) python src/gsm8k_eval.py --model "$Q14" --tag gsm_q14_fp16 --batch 16 \
        --scores-csv runs/scores_gsm8k.csv
      python src/gsm8k_eval.py --model "$STORE/models/qwen2.5-14b-v2gptq3-none" \
        --tag gsm_q14_none --batch 16 --scores-csv runs/scores_gsm8k.csv
      python src/gsm8k_eval.py --model "$STORE/models/qwen2.5-14b-v2gptq3-tacq" \
        --tag gsm_q14_tacq --batch 16 --scores-csv runs/scores_gsm8k.csv ;;
  12) python src/multi_if.py --model "$STORE/models/qwen2.5-7b-v2gptq3-none" \
        --tag mif_qwen_v2none --batch 8 --scores-csv runs/scores_mif_qwen_v2none.csv ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W13] ✅ task $SGE_TASK_ID done"
