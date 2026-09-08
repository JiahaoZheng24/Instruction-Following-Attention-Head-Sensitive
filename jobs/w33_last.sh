#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=10:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W33
#$ -t 1-8
# W33: LAST batch before the writing freeze.
#  1-3  calibration-seed replicates of the three headline cures (single-seed
#       so far; the Q14 collapse arm had +-10 seed variance).
#  4-5  close the lambda-tuner question on Llama: IFEval at rho=0.5 (E_d ratio
#       already 0.84 there; rho=1 IFEval is .152) and stats at rho=1.
#  6-8  Mistral-7B forensics: rho>=0.5 WITH act-order gives byte garbage
#       (.17) while rho=5 without act-order is fine (.49) and rho=1e6 with
#       act-order is RTN-like (.41); layer objectives see nothing. Keep two
#       checkpoints (rho 0.05 and 0.5), diff them module by module, run the
#       divergence curve, and test rho=0.5 without act-order.
#   qsub jobs/w33_last.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
mkdir -p runs/stats

fq () {  # $1 model $2 ckpt $3 tag $4.. flags   (deleted after eval)
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}
keep () {  # same, checkpoint KEPT
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
}

case "$SGE_TASK_ID" in
  1) fq "$LLAMA" llama3.1-8b-v2gptq3-damp5_cs1   v2l_damp5_cs1   --percdamp 5 --calib-seed 1 ;;
  2) fq "$LLAMA" llama3.1-8b-v2gptq3-calchat_cs1 v2l_calchat_cs1 --calib ultrachat --calib-seed 1 ;;
  3) fq "$Q14"   qwen2.5-14b-v2gptq3-calchat_cs1 v2q14_calchat_cs1 --calib ultrachat --calib-seed 1 ;;
  4) fq "$LLAMA" llama3.1-8b-v2gptq3-damp0p5     v2l_damp0p5     --percdamp 0.5 ;;
  5) $T python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 --protect none \
       --percdamp 1 --stats-dir runs/stats/llama31-8b-damp1 --stats-chat-n 64 --no-save \
       --out "$STORE/models/_unused_l_damp1" ;;
  6) keep "$M7" mistral-7b-v2gptq3-none-keep v2m_none_keep
     python src/divergence.py --fp16 "$M7" --quant "$STORE/models/mistral-7b-v2gptq3-none-keep" \
       --prompts "$FULL" --n 32 --out runs/div_m_none.csv ;;
  7) keep "$M7" mistral-7b-v2gptq3-damp0p5-keep v2m_damp0p5_keep --percdamp 0.5
     python src/divergence.py --fp16 "$M7" --quant "$STORE/models/mistral-7b-v2gptq3-damp0p5-keep" \
       --prompts "$FULL" --n 32 --out runs/div_m_damp0p5.csv
     # diff against the rho=0.05 checkpoint if task 6 has finished; harmless otherwise
     [ -d "$STORE/models/mistral-7b-v2gptq3-none-keep" ] && python src/ckpt_diff.py \
       --a "$STORE/models/mistral-7b-v2gptq3-none-keep" --b "$STORE/models/mistral-7b-v2gptq3-damp0p5-keep" \
       --out runs/ckpt_diff_mistral_damp0p5.csv || echo "rho=0.05 keep-checkpoint not there yet; run ckpt_diff by hand" ;;
  8) fq "$M7" mistral-7b-v2gptq3-damp0p5-noao v2m_damp0p5_noao --percdamp 0.5 --no-actorder ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W33] done task $SGE_TASK_ID"
