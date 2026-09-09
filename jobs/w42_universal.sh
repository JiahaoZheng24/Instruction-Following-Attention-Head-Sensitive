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
#$ -N IFH_W42
#$ -t 1-15
# W42: from "two collapses" to a 17-model law + a mechanism-derived repair.
#  1-9   position statistics (no checkpoint, ~30-60 min each) on the 9 census
#        models not yet measured -> detector strength for all 17 models.
#        (done: Llama-3.1-8B, Q14, Q7, Nemo, L3.2-3B, gemma-2-9b, Mistral-v0.2,
#        Mistral-v0.3 at rho=0.5 only -> task 9 adds rho=0.05.)
#  10-12 REPAIR THE OBJECTIVE: token-normalised Hessian (every calibration
#        token weighs the same, so BOS cannot dominate the sink-forming
#        down_proj) on Llama and Q14, frozen c4 calibration otherwise;
#        BOS-exclusion ablation (drop positions 0-1) on Llama.
#        Per-position divergence on the two token-norm arms.
#  13-14 IS IT THE BIT-WIDTH OR THE ERROR ENERGY? Llama 3-bit g32 (smaller
#        error than the collapsing g128) and 4-bit per-channel (g=-1, larger
#        error than the graceful 4-bit g128), each with per-position divergence.
#        Prediction: collapse follows the error at the sink-forming module,
#        not the nominal bit-width.
#  15    Llama 3-bit g128 + token-norm + rho=0.05, calib seed 1 (replicate of 10)
#   qsub jobs/w42_universal.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
mkdir -p runs/stats

posstats () {  # $1 model $2 stats-dir $3.. flags   (no checkpoint)
  local model="$1" sd="$2"; shift 2
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none \
     --stats-dir "$sd" --stats-chat-n 64 --no-save "$@" --out "$STORE/models/_unused_posstats"
}
fq () {  # $1 model $2 ckpt $3 tag $4 divergence-out-or-"-" $5 bits $6 group $7.. flags   (deleted after eval)
  local model="$1" ckpt="$2" tag="$3" div="$4" bits="$5" grp="$6"; shift 6
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits "$bits" --group-size "$grp" --protect none "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  if [ "$div" != "-" ]; then
    $T python src/divergence.py --fp16 "$model" --quant "$CKPT" --prompts "$FULL" --n 32 --per-position --out "$div"
  fi
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1)  posstats meta-llama/Llama-3.2-1B-Instruct        runs/stats/llama32-1b-pos ;;
  2)  posstats google/gemma-2-2b-it                    runs/stats/gemma2-2b-pos ;;
  3)  posstats tiiuae/Falcon3-7B-Instruct              runs/stats/falcon3-7b-pos ;;
  4)  posstats HuggingFaceTB/SmolLM2-1.7B-Instruct     runs/stats/smollm2-1.7b-pos ;;
  5)  posstats Qwen/Qwen2.5-3B-Instruct                runs/stats/qwen25-3b-pos ;;
  6)  posstats Qwen/Qwen2.5-32B-Instruct               runs/stats/qwen25-32b-pos ;;
  7)  posstats mistralai/Mistral-Small-24B-Instruct-2501 runs/stats/mistral-24b-pos ;;
  8)  posstats meta-llama/Meta-Llama-3-8B-Instruct     runs/stats/llama3-8b-pos ;;
  9)  posstats "$M7"                                   runs/stats/mistral-7b-pos ;;
  10) fq "$LLAMA" llama3.1-8b-v2gptq3-tnorm    v2l_tnorm    runs/div_l_tnorm.csv   3 128 --hess-token-norm ;;
  11) fq "$Q14"   qwen2.5-14b-v2gptq3-tnorm    v2q14_tnorm  runs/div_q14_tnorm.csv 3 128 --hess-token-norm ;;
  12) fq "$LLAMA" llama3.1-8b-v2gptq3-dropbos  v2l_dropbos  -                      3 128 --hess-drop-pos 2 ;;
  13) fq "$LLAMA" llama3.1-8b-v2gptq3-g32      v2l_g32      runs/div_l_g32.csv     3 32 ;;
  14) fq "$LLAMA" llama3.1-8b-v2gptq4-gm1      v2l4_gm1     runs/div_l4_gm1.csv    4 -1 ;;
  15) fq "$LLAMA" llama3.1-8b-v2gptq3-tnorm_cs1 v2l_tnorm_cs1 -                    3 128 --hess-token-norm --calib-seed 1 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W42] done task $SGE_TASK_ID"
