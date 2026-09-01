#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W12
#$ -t 1-5
# W12: anatomy of the SURPRISE 14B collapse (qwen14b@3bit avg4 0.41 vs fp16
# 0.82, while 7B AND 32B are graceful -> regime is non-monotonic in scale).
# Mirrors the Llama collapse anatomy, plus one comparability patch.
#  1: qwen14b gptq3 tacq@70M   (rerun after the torch.quantile>16M fix —
#     does the rescue generalize across models AND scale?)
#  2: qwen14b rtn3 none        (is the 14B collapse GPTQ-specific, like Llama?)
#  3: qwen14b rot+gptq3 none   (does rotation rescue 14B like it rescued Llama?)
#  4: qwen14b gptq3 randw@70M  (criterion control: random scattered weights)
#  5: GSM8K on qwen2.5-7b-v2gptq3-none (v2-vs-v2 reference; the W11 qwen
#     reference was the packed gptq3 checkpoint)
# NOTE: sync the fixed src/common.py + quantize_protected.py etc. BEFORE qsub.
#   qsub jobs/w12_qwen14b_anatomy.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
Q14="Qwen/Qwen2.5-14B-Instruct"
FULL="data/ifeval_input_data.jsonl"
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
  1) CKPT="$STORE/models/qwen2.5-14b-v2gptq3-tacq"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --protect tacq --salience-dir "$SAL14" --budget-params 70000000 --out "$CKPT"
     run_ifeval "$CKPT" v2q14_tacq ;;
  2) CKPT="$STORE/models/qwen2.5-14b-rtn3-none"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --rtn --protect none --out "$CKPT"
     run_ifeval "$CKPT" rtn3q14_none ;;
  3) CKPT="$STORE/models/qwen2.5-14b-rot-gptq3-none"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --rotate --rotate-seed 0 --protect none --out "$CKPT"
     run_ifeval "$CKPT" v2q14_rot_none ;;
  4) CKPT="$STORE/models/qwen2.5-14b-v2gptq3-randw"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --protect randw --budget-params 70000000 --seed 0 --out "$CKPT"
     run_ifeval "$CKPT" v2q14_randw ;;
  5) python src/gsm8k_eval.py --model "$STORE/models/qwen2.5-7b-v2gptq3-none" \
       --tag gsm_qwen_v2none --batch 16 --scores-csv runs/scores_gsm8k.csv ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W12] ✅ task $SGE_TASK_ID done"
