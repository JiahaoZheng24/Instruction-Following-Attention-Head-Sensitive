#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=12:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W58
#$ -t 1-41
# W58: CENSUS UNDER THE WINDOW PROTOCOLS + non-OBS slice + mechanism signatures.
# W56c: the two collapses hold under per-document short calibration (our frozen
# protocol) but NOT under c4 128x2048 windows, while pile 128x2048 windows
# collapse Q14 (.291) and hurt Llama (.402). The 32-model census (2 collapses)
# was run under the short protocol only. Pre-registered in RESULTS 9.10am:
#   1-16  GPTQ 3-bit, c4win (literature budget), 16 models  -> 0 collapses
#  17-32  GPTQ 3-bit, pile 128x2048, same 16 models           -> <=3 collapses, in the
#                                                                top half of the detector ranking
#  33     GPTQ 4-bit pile, Llama                              -> >= .70 (4-bit immune under pile too)
#  34-37  AutoRound 3-bit, our short c4, Q7 / Hermes-3 / Nemo / Qwen3-14B
#                                                             -> follows GPTQ's short-c4 outcome
#  38-41  per-position divergence signatures (no IFEval): AutoRound short-c4 Q14 & Llama
#         (same early-layer template-position signature as GPTQ), GPTQ c4win Llama
#         (signature below the 1.4 line), GPTQ pile Llama (at/above the line)
# RTN baselines exist for every model (protocol-independent). Checkpoints deleted.
#   qsub jobs/w58_window_census.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 10h"

MODELS=(
  "Qwen/Qwen2.5-7B-Instruct|q7"
  "Qwen/Qwen2.5-32B-Instruct|q32"
  "Qwen/Qwen2.5-3B-Instruct|q3"
  "mistralai/Mistral-Nemo-Instruct-2407|nemo"
  "mistralai/Mistral-7B-Instruct-v0.3|m7"
  "google/gemma-2-9b-it|g29"
  "meta-llama/Llama-3-8B-Instruct|l3"
  "meta-llama/Llama-3.2-3B-Instruct|l32"
  "tiiuae/Falcon3-7B-Instruct|f3"
  "HuggingFaceTB/SmolLM2-1.7B-Instruct|sm"
  "NousResearch/Hermes-3-Llama-3.1-8B|hermes3"
  "mistralai/Ministral-8B-Instruct-2410|ministral"
  "Qwen/Qwen3-14B|qwen3_14b"
  "allenai/Llama-3.1-Tulu-3-8B|tulu3"
  "Qwen/Qwen3-8B|qwen3_8b"
  "ibm-granite/granite-3.1-8b-instruct|granite"
)
in_ar_env () {
  conda activate IFEval_ar || { echo "env IFEval_ar missing: run bash jobs/w56_setup.sh"; exit 4; }
  "$@"; local rc=$?
  conda activate "$CONDA_ENV"
  return $rc
}
finish () {    # $1 ckpt  $2 tag
  run_ifeval "$1" "$2"
  cp "$1/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$1")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$1"
}
gq () {   # $1 model $2 ckpt $3 tag $4 calib $5 bits
  CK="$STORE/models/$2"
  $T python src/quantize_protected.py --model "$1" --bits "$5" --group-size 128 --protect none --calib "$4" --out "$CK"
  finish "$CK" "$3"
}
ar () {   # $1 model $2 ckpt $3 tag
  CK="$STORE/models/$2"
  in_ar_env $T python src/quantize_autoround.py --model "$1" --bits 3 --group-size 128 --calib c4 --out "$CK"
  finish "$CK" "$3"
}
div_only () {   # $1 model $2 ckpt $3 div-out  (checkpoint already built into $2; deleted after)
  $T python src/divergence.py --fp16 "$1" --quant "$2" --prompts "$FULL" --n 32 --per-position --out "$3"
  cp "$2/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$2")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$2"
}

id=$SGE_TASK_ID
if [ "$id" -le 32 ]; then
  if [ "$id" -le 16 ]; then calib=c4win; idx=$((id - 1)); else calib=pile; idx=$((id - 17)); fi
  entry="${MODELS[$idx]}"; model="${entry%%|*}"; s="${entry##*|}"
  gq "$model" "win-${s}-v2gptq3-${calib}" "win_${s}_${calib}" "$calib" 3
elif [ "$id" -eq 33 ]; then
  gq "$LLAMA" llama3.1-8b-v2gptq4-pile v2l4_pile pile 4
elif [ "$id" -le 37 ]; then
  case $id in
    34) ar "$Q7"                                   qwen2.5-7b-ar3       m_q7_ar3 ;;
    35) ar NousResearch/Hermes-3-Llama-3.1-8B      hermes3-ar3          m_hermes3_ar3 ;;
    36) ar mistralai/Mistral-Nemo-Instruct-2407    nemo-ar3             m_nemo_ar3 ;;
    37) ar Qwen/Qwen3-14B                          qwen3-14b-ar3        m_qwen3_14b_ar3 ;;
  esac
else
  case $id in
    38) CK="$STORE/models/qwen2.5-14b-ar3-div"
        in_ar_env $T python src/quantize_autoround.py --model "$Q14" --bits 3 --group-size 128 --calib c4 --out "$CK"
        div_only "$Q14" "$CK" runs/div_q14_ar3.csv ;;
    39) CK="$STORE/models/llama3.1-8b-ar3-div"
        in_ar_env $T python src/quantize_autoround.py --model "$LLAMA" --bits 3 --group-size 128 --calib c4 --out "$CK"
        div_only "$LLAMA" "$CK" runs/div_l_ar3.csv ;;
    40) CK="$STORE/models/llama3.1-8b-v2gptq3-c4win-div"
        $T python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 --protect none --calib c4win --out "$CK"
        div_only "$LLAMA" "$CK" runs/div_l_c4win.csv ;;
    41) CK="$STORE/models/llama3.1-8b-v2gptq3-pile-div"
        $T python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 --protect none --calib pile --out "$CK"
        div_only "$LLAMA" "$CK" runs/div_l_pile.csv ;;
    *) echo "bad task id"; exit 1 ;;
  esac
fi
echo "[W58] done task $SGE_TASK_ID"
