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
#$ -N IFH_W79
#$ -t 1-41
# W79 (2026-09-18). Re-runs and follow-ups of W76-W78, pre-registered in RESULTS 9.10br.
#   1-17   W76 E2/E3 re-run for all 17 sink models (the BOS-free control Hessian is now
#          accumulated directly; the subtracted one lost positive-definiteness on Llama-family
#          models and gave NaN controls on Qwen). Overwrites runs/theory/f_<s>_c4.json.
#   18-25  W76 protocol sweep re-run (same reason).
#   26     W76 E1 Llama readout (failed with task 1).
#   27-32  W77 activation patching re-run (device fix): Llama template/ordinary/bos/all, Q14 template/ordinary.
#   33     W78 Mistral c4win second seed (HF stale file handle).
#   34-38  collapse frequency: Llama short-c4 seeds 2,3,4; Q14 short-c4 seeds 2,3.
#   39-41  position weight where it can be told apart (Mistral c4win: tnorm alone fails, G alone cures):
#          tnorm + first 8 x10 | x100 ; corrected objective + first 8 x0.01.
#   qsub jobs/w79_reruns.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 7h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
GWTN="--hess-grad-weight --hess-token-norm"
mkdir -p runs/theory runs/sink runs/protocols

SINK=(
  "$LLAMA|l" "$Q14|q14" "$Q7|q7" "Qwen/Qwen2.5-32B-Instruct|q32" "Qwen/Qwen2.5-3B-Instruct|q3"
  "mistralai/Mistral-Nemo-Instruct-2407|nemo" "$M7|m7" "meta-llama/Meta-Llama-3-8B-Instruct|l3"
  "meta-llama/Llama-3.2-3B-Instruct|l32" "tiiuae/Falcon3-7B-Instruct|f3" "HuggingFaceTB/SmolLM2-1.7B-Instruct|sm"
  "NousResearch/Hermes-3-Llama-3.1-8B|hermes3" "mistralai/Ministral-8B-Instruct-2410|ministral"
  "Qwen/Qwen3-14B|qwen3_14b" "allenai/Llama-3.1-Tulu-3-8B|tulu3" "Qwen/Qwen3-8B|qwen3_8b"
  "ibm-granite/granite-3.1-8b-instruct|granite"
)
PROTO=( "$LLAMA|l|c4win" "$LLAMA|l|pile" "$LLAMA|l|wikitext" "$Q14|q14|c4win" "$Q14|q14|pile" "$Q14|q14|wikitext" "$M7|m7|c4" "$M7|m7|c4win" )
PATCH=( "$LLAMA|l|1|template" "$LLAMA|l|1|ordinary" "$LLAMA|l|1|bos" "$LLAMA|l|1|all" "$Q14|q14|4|template" "$Q14|q14|4|ordinary" )

theory () {   # $1 model $2 tag $3 calib [$4 whiten]
  if [ -n "${4:-}" ]; then $T python src/theory_tests.py --model "$1" --calib "$3" --tag "$2" --save-whiten "$4"
  else $T python src/theory_tests.py --model "$1" --calib "$3" --tag "$2"; fi
}
score () { $T python src/score_ifeval.py --responses "runs/$(basename "$1")/$2/responses.jsonl" --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"; }
qz () {   # $1 model $2 ckpt $3 tag $4.. flags
  CK="$STORE/models/$2"; m="$1"; tag="$3"; shift 3
  t0=$SECONDS
  $T python src/quantize_protected.py --model "$m" --bits 3 --group-size 128 --protect none --out "$CK" "$@"
  echo "[W79-timing] tag=$tag quantise_seconds=$((SECONDS - t0))"
  run_ifeval "$CK" "$tag"
  cp "$CK/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CK")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CK"
}
patch_arm () {   # $1 model $2 short $3 layer $4 select
  CK="$STORE/models/f-$2-gptq3-c4-patch-$4"
  $T python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 --protect none --calib c4 --out "$CK"
  tag="f_patch_$2_$4"
  $T python src/inject_sink.py patch --model "$CK" --ref "$1" --prompts "$FULL" --layer "$3" --select "$4" --tag "$tag" --batch 16
  score "$CK" "$tag"
  rm -rf "$CK"
}

id=$SGE_TASK_ID
if [ "$id" -le 17 ]; then
  entry="${SINK[$((id - 1))]}"; model="${entry%%|*}"; s="${entry##*|}"
  if [ "$s" = "l" ] || [ "$s" = "q14" ]; then theory "$model" "f_${s}_c4" c4 "runs/theory/f_${s}_c4_whiten.pt"; else theory "$model" "f_${s}_c4" c4; fi
elif [ "$id" -le 25 ]; then
  entry="${PROTO[$((id - 18))]}"; IFS='|' read -r model s calib <<< "$entry"
  if [ "$s" = "m7" ] && [ "$calib" = "c4win" ]; then theory "$model" "f_${s}_${calib}" "$calib" "runs/theory/f_m7_c4win_whiten.pt"; else theory "$model" "f_${s}_${calib}" "$calib"; fi
elif [ "$id" -eq 26 ]; then
  WH="runs/theory/f_l_c4_whiten.pt"
  theory "$LLAMA" f_l_c4_ro c4 "$WH"
  CK="$STORE/models/f-l-gptq3-c4-readout"
  $T python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 --protect none --calib c4 --out "$CK"
  $T python src/inject_sink.py artifact --model "$LLAMA" --quant "$CK" --prompts "$FULL" --n 16 --layer 1 --continuation 128 \
     --whiten "$WH" --out runs/sink/f_l_readout.csv --out-tokens runs/sink/f_l_readout_tokens.csv
  rm -rf "$CK"
elif [ "$id" -le 32 ]; then
  entry="${PATCH[$((id - 27))]}"; IFS='|' read -r model s L sel <<< "$entry"
  patch_arm "$model" "$s" "$L" "$sel"
elif [ "$id" -eq 33 ]; then
  qz "$M7" f-m7-gptq3-c4win-cs1 f_m7_c4win_cs1 --calib c4win --calib-seed 1
elif [ "$id" -le 36 ]; then
  sd=$((id - 32))     # 34->2, 35->3, 36->4
  qz "$LLAMA" "f-l-gptq3-none-cs${sd}" "f_l_none_cs${sd}" --calib c4 --calib-seed "$sd"
elif [ "$id" -le 38 ]; then
  sd=$((id - 35))     # 37->2, 38->3
  qz "$Q14" "f-q14-gptq3-none-cs${sd}" "f_q14_none_cs${sd}" --calib c4 --calib-seed "$sd"
elif [ "$id" -eq 39 ]; then
  qz "$M7" f-m7-gptq3-c4win-tnorm-pos8x10  f_m7_c4win_tnorm_pos8x10  --calib c4win --hess-token-norm --hess-pos-weight 8:10
elif [ "$id" -eq 40 ]; then
  qz "$M7" f-m7-gptq3-c4win-tnorm-pos8x100 f_m7_c4win_tnorm_pos8x100 --calib c4win --hess-token-norm --hess-pos-weight 8:100
else
  qz "$M7" f-m7-gptq3-c4win-gwtn-pos8x0p01 f_m7_c4win_gwtn_pos8x0p01 --calib c4win $GWTN --hess-pos-weight 8:0.01
fi
echo "[W79] done task $SGE_TASK_ID"
