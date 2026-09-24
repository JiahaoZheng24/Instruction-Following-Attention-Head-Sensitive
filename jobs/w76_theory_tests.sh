#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=08:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W76
#$ -t 1-32
# W76 (2026-09-17). Direct tests of Theorem 1 / Corollary 2 on the real Hessian
# and the real GPTQ output. Pre-registered in RESULTS 9.10bo; design in
# paper/SELF_REVIEW_2026-09-17.md section 5. No checkpoint survives the job.
#   1-17   E2+E3  short-document c4: kappa, rank-one share of the compensation
#                 along x0 / v at the sink down_proj and two controls, 17 sink models
#                 -> runs/theory/f_<s>_c4.json + INDEX_theory.csv
#   18-25  E2     protocol sweep: Llama {c4win, pile, wikitext}, Q14 {c4win, pile,
#                 wikitext}, Mistral-v0.3 {c4, c4win}
#   26-28  E1     per-token readout on the real GPTQ3 checkpoint (prompt + 128
#                 fp16 tokens, 16 prompts): isotropic overlap, whitened overlap
#                 v^T x_t, row readout delta_r^T x_t -> runs/sink/f_<s>_readout.csv
#                 (+ per-token runs/sink/f_<s>_readout_tokens.csv)
#   29-32  E4     position-matched injection in the RTN3 Llama background:
#                 beta {4,8} x positions {2-7 template, 20-25 ordinary} -> IFEval
#   qsub jobs/w76_theory_tests.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 7h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
mkdir -p runs/theory runs/sink runs/protocols

SINK=(                                  # 17 census models with a sink (W74 list minus gemma, plus Llama / Q14)
  "$LLAMA|l"
  "$Q14|q14"
  "$Q7|q7"
  "Qwen/Qwen2.5-32B-Instruct|q32"
  "Qwen/Qwen2.5-3B-Instruct|q3"
  "mistralai/Mistral-Nemo-Instruct-2407|nemo"
  "$M7|m7"
  "meta-llama/Meta-Llama-3-8B-Instruct|l3"
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
PROTO=(                                 # model|short|calib
  "$LLAMA|l|c4win" "$LLAMA|l|pile" "$LLAMA|l|wikitext"
  "$Q14|q14|c4win" "$Q14|q14|pile" "$Q14|q14|wikitext"
  "$M7|m7|c4" "$M7|m7|c4win"
)
READOUT=(                               # model|short|calib|sink layer
  "$LLAMA|l|c4|1"
  "$Q14|q14|c4|4"
  "$M7|m7|c4win|1"
)
E4=( "4|2-7" "4|20-25" "8|2-7" "8|20-25" )   # beta|positions

theory () {   # $1 model $2 tag $3 calib [$4 save-whiten]
  if [ -n "${4:-}" ]; then
    $T python src/theory_tests.py --model "$1" --calib "$3" --tag "$2" --save-whiten "$4"
  else
    $T python src/theory_tests.py --model "$1" --calib "$3" --tag "$2"
  fi
}
score () {   # $1 model-id-or-ckpt  $2 tag
  $T python src/score_ifeval.py --responses "runs/$(basename "$1")/$2/responses.jsonl" \
     --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}

id=$SGE_TASK_ID
if [ "$id" -le 17 ]; then
  entry="${SINK[$((id - 1))]}"; model="${entry%%|*}"; s="${entry##*|}"
  theory "$model" "f_${s}_c4" c4
elif [ "$id" -le 25 ]; then
  entry="${PROTO[$((id - 18))]}"; IFS='|' read -r model s calib <<< "$entry"
  theory "$model" "f_${s}_${calib}" "$calib"
elif [ "$id" -le 28 ]; then
  entry="${READOUT[$((id - 26))]}"; IFS='|' read -r model s calib L <<< "$entry"
  WH="runs/theory/f_${s}_${calib}_whiten.pt"
  theory "$model" "f_${s}_${calib}_ro" "$calib" "$WH"
  CK="$STORE/models/f-${s}-gptq3-${calib}-readout"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none \
     --calib "$calib" --out "$CK"
  $T python src/inject_sink.py artifact --model "$model" --quant "$CK" --prompts "$FULL" --n 16 \
     --layer "$L" --continuation 128 --whiten "$WH" \
     --out "runs/sink/f_${s}_readout.csv" --out-tokens "runs/sink/f_${s}_readout_tokens.csv"
  rm -rf "$CK"
else
  entry="${E4[$((id - 29))]}"; IFS='|' read -r beta pos <<< "$entry"
  CK="$STORE/models/f-l-rtn3-inj-${id}"
  $T python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 --protect none \
     --calib c4 --rtn --out "$CK"
  tag="f_inj_rtn3_b${beta}_p${pos//-/_}"
  $T python src/inject_sink.py inject --model "$CK" --prompts "$FULL" --layer 1 --beta "$beta" \
     --pos "$pos" --tag "$tag" --batch 16
  score "$CK" "$tag"
  rm -rf "$CK"
fi
echo "[W76] done task $SGE_TASK_ID"
