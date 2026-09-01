#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W10a
#$ -t 1-9
# W10a (UNFREEZE for ICLR): rotation arms + scale arms, independent tasks.
#  Rotation (does the collapse regime survive QuaRot-style R1?):
#   1: llama R1-rotated, UNquantized -> IFEval must reproduce fp16 (sanity arm)
#   2: llama rot+gptq3 none   <- the money arm: does collapse survive rotation?
#   3: qwen  rot+gptq3 none   (does graceful stay graceful?)
#   4: llama rotated-basis IF salience (feeds W10b tacq arm)
#  Scale (does the regime law survive within-family scaling?):
#   5: qwen2.5-14b fp16 IFEval baseline
#   6: qwen2.5-14b IF salience (feeds W10b)
#   7: qwen2.5-14b gptq3 none
#   8: qwen2.5-32b fp16 IFEval baseline
#   9: qwen2.5-32b gptq3 none
#   qsub jobs/w10a_rotation_scale.sh && qsub jobs/w10b_rot_tacq.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
LLAMA="meta-llama/Llama-3.1-8B-Instruct"
QWEN="Qwen/Qwen2.5-7B-Instruct"
Q14="Qwen/Qwen2.5-14B-Instruct"
Q32="Qwen/Qwen2.5-32B-Instruct"
FULL="data/ifeval_input_data.jsonl"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

run_ifeval () {  # $1 ckpt, $2 tag, $3 batch
  python src/diagnose_heads.py ablate --model "$1" --prompts "$FULL" \
    --tag "$2" --batch "${3:-16}"
  python src/score_ifeval.py --responses "runs/$(basename $1)/$2/responses.jsonl" \
    --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}

case "$SGE_TASK_ID" in
  1) CKPT="$STORE/models/llama3.1-8b-rotfp"
     python src/rotate_model.py --model "$LLAMA" --seed 0 --out "$CKPT"
     run_ifeval "$CKPT" rotfp_llama ;;
  2) CKPT="$STORE/models/llama3.1-8b-rot-gptq3-none"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --rotate --rotate-seed 0 --protect none --out "$CKPT"
     run_ifeval "$CKPT" v2l_rot_none ;;
  3) CKPT="$STORE/models/qwen2.5-7b-rot-gptq3-none"
     python src/quantize_protected.py --model "$QWEN" --bits 3 --group-size 128 \
       --rotate --rotate-seed 0 --protect none --out "$CKPT"
     run_ifeval "$CKPT" v2q_rot_none ;;
  4) mkdir -p "$STORE/salience/llama31-8b-if-rot"
     python src/tacq_salience.py --model "$LLAMA" --rotate --rotate-seed 0 \
       --calib-file data/calib_prompts.jsonl --bits 3 \
       --out-dir "$STORE/salience/llama31-8b-if-rot" ;;
  5) python src/diagnose_heads.py ablate --model "$Q14" --prompts "$FULL" \
       --tag qbase_qwen14_fp16 --batch 16
     python src/score_ifeval.py \
       --responses "runs/Qwen2.5-14B-Instruct/qbase_qwen14_fp16/responses.jsonl" \
       --input-data "$FULL" --tag qbase_qwen14_fp16 \
       --scores-csv runs/scores_qbase_qwen14_fp16.csv ;;
  6) mkdir -p "$STORE/salience/qwen25-14b-if"
     python src/tacq_salience.py --model "$Q14" \
       --calib-file data/calib_prompts.jsonl --bits 3 \
       --out-dir "$STORE/salience/qwen25-14b-if" ;;
  7) CKPT="$STORE/models/qwen2.5-14b-v2gptq3-none"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --protect none --out "$CKPT"
     run_ifeval "$CKPT" v2q14_none ;;
  8) python src/diagnose_heads.py ablate --model "$Q32" --prompts "$FULL" \
       --tag qbase_qwen32_fp16 --batch 8
     python src/score_ifeval.py \
       --responses "runs/Qwen2.5-32B-Instruct/qbase_qwen32_fp16/responses.jsonl" \
       --input-data "$FULL" --tag qbase_qwen32_fp16 \
       --scores-csv runs/scores_qbase_qwen32_fp16.csv ;;
  9) CKPT="$STORE/models/qwen2.5-32b-v2gptq3-none"
     python src/quantize_protected.py --model "$Q32" --bits 3 --group-size 128 \
       --protect none --out "$CKPT"
     run_ifeval "$CKPT" v2q32_none 8 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W10a] ✅ task $SGE_TASK_ID done"
