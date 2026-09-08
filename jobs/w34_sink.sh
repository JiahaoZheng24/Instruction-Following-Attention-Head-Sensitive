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
#$ -N IFH_W34
#$ -t 1-10
# W34: WHERE the collapse lives — early-layer attention at the sink position?
# W33 facts: Mistral rho=0.5 residual stream explodes at layer 2 (rel_err 117,
# cos_mean 0.08) while the LAST prompt token stays intact (cos_last 0.76) and
# every layer-wise objective looks normal; Llama rho=1 has E_d(GPTQ)<E_d(RTN)
# in all 224 modules yet IFEval .152. Layer-wise proxies are blind because
# the damage is (hypothesis) at a few positions — BOS / attention sink — and
# is amplified nonlinearly downstream. Two tests:
#  1-4  per-position divergence (positions 0..7 vs rest) on Mistral rho=0.5,
#       Llama frozen (rho=0.05), Llama rho=1 (re-created, deleted), Llama rho=2.
#  5-10 structured protection of EARLY-LAYER ATTENTION (whole modules fp16):
#       does keeping layers 0-2 q/k/v/o (~0.8-1% of params) cure Llama at
#       rho=0.05, Mistral at rho=0.5, and does keeping Q14's single 257x
#       module (layer 4 down_proj) cure Q14? Controls: same budget on layers
#       14-16 (mid) for Llama and Mistral.
#   qsub jobs/w34_sink.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
EARLY="0-2:q_proj,k_proj,v_proj,o_proj"
MID="14-16:q_proj,k_proj,v_proj,o_proj"

fq () {  # $1 model $2 ckpt $3 tag $4.. flags   (deleted after eval)
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}
divpos () {  # $1 fp16 $2 ckpt $3 out
  $T python src/divergence.py --fp16 "$1" --quant "$2" --prompts "$FULL" --n 32 --per-position --out "$3"
}

case "$SGE_TASK_ID" in
  1) divpos "$M7" "$STORE/models/mistral-7b-v2gptq3-damp0p5-keep" runs/div_m_damp0p5.csv
     divpos "$M7" "$STORE/models/mistral-7b-v2gptq3-none-keep"    runs/div_m_none.csv ;;
  2) divpos "$LLAMA" "$STORE/models/llama3.1-8b-v2gptq3-none"  runs/div_l_none.csv
     divpos "$LLAMA" "$STORE/models/llama3.1-8b-v2gptq3-damp5" runs/div_l_damp5.csv ;;
  3) CKPT="$STORE/models/llama3.1-8b-v2gptq3-damp1-tmp"
     $T python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 --protect none --percdamp 1 --out "$CKPT"
     divpos "$LLAMA" "$CKPT" runs/div_l_damp1.csv; rm -rf "$CKPT" ;;
  4) CKPT="$STORE/models/llama3.1-8b-v2gptq3-damp2-tmp"
     $T python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 --protect none --percdamp 2 --out "$CKPT"
     divpos "$LLAMA" "$CKPT" runs/div_l_damp2.csv; rm -rf "$CKPT" ;;
  5)  fq "$LLAMA" llama3.1-8b-v2gptq3-early v2l_early --protect modules --modules "$EARLY" ;;
  6)  fq "$LLAMA" llama3.1-8b-v2gptq3-mid   v2l_mid   --protect modules --modules "$MID" ;;
  7)  fq "$M7" mistral-7b-v2gptq3-damp0p5-early v2m_damp0p5_early --protect modules --modules "$EARLY" --percdamp 0.5 ;;
  8)  fq "$M7" mistral-7b-v2gptq3-damp0p5-mid   v2m_damp0p5_mid   --protect modules --modules "$MID"   --percdamp 0.5 ;;
  9)  fq "$Q14" qwen2.5-14b-v2gptq3-l4down v2q14_l4down --protect modules --modules "4:down_proj" ;;
  10) fq "$Q14" qwen2.5-14b-v2gptq3-early  v2q14_early  --protect modules --modules "$EARLY" ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W34] done task $SGE_TASK_ID"
