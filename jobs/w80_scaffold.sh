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
#$ -N IFH_W80
#$ -t 1-6
# W80 (2026-09-18). Activation patching with the WHOLE chat scaffold, pre-registered in RESULTS 9.10bs.
# W79: restoring Llama's 8 special/newline positions does not cure (.144) but the whole prompt does (.664);
# Llama-3.1's template carries a default system block ("Cutting Knowledge Date ...") that the special-token
# selector classed as ordinary. scaffold = every prompt token outside the user's text; content = same count
# from the user's text.
#   1  Llama  scaffold      2  Llama  content      3  Q14  scaffold      4  Q14  content
#   5-6  short-document protocol with a NON-C4 corpus (128 Pile documents tokenised separately,
#        the library's sampling shape with only the corpus changed): Llama, Q14 -> IFEval
#   qsub jobs/w80_scaffold.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 7h"

score () { $T python src/score_ifeval.py --responses "runs/$(basename "$1")/$2/responses.jsonl" --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"; }
patch_arm () {   # $1 model $2 short $3 layer $4 select
  CK="$STORE/models/f-$2-gptq3-c4-patch-$4"
  $T python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 --protect none --calib c4 --out "$CK"
  tag="f_patch_$2_$4"
  $T python src/inject_sink.py patch --model "$CK" --ref "$1" --prompts "$FULL" --layer "$3" --select "$4" --tag "$tag" --batch 16
  score "$CK" "$tag"
  rm -rf "$CK"
}
qz () {   # $1 model $2 ckpt $3 tag $4.. flags
  CK="$STORE/models/$2"; m="$1"; tag="$3"; shift 3
  $T python src/quantize_protected.py --model "$m" --bits 3 --group-size 128 --protect none --out "$CK" "$@"
  run_ifeval "$CK" "$tag"
  cp "$CK/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CK")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CK"
}
mkdir -p runs/protocols
case "$SGE_TASK_ID" in
  1) patch_arm "$LLAMA" l 1 scaffold ;;
  2) patch_arm "$LLAMA" l 1 content ;;
  3) patch_arm "$Q14" q14 4 scaffold ;;
  4) patch_arm "$Q14" q14 4 content ;;
  5) qz "$LLAMA" f-l-gptq3-pileshort  f_l_pileshort  --calib pileshort ;;
  6) qz "$Q14"   f-q14-gptq3-pileshort f_q14_pileshort --calib pileshort ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W80] done task $SGE_TASK_ID"
