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
#$ -N IFH_W52
#$ -t 1-4
# W52: the most surgical test of the mechanism. W51: in blocks 0-1 the only
# matrices where GPTQ's template-token error exceeds RTN's are the two
# down_proj, and inside them the error sits in a handful of output rows —
# the super-weight rows (layer 1: 788, 1384, 4062; layer 0: 1430 / 1268) —
# whose near-zero weights on the BOS channels GPTQ pushes to the edge of the
# quantisation range (3-bit and 4-bit alike, max|Q| 6.8 vs max|W| 0.6).
# Every other matrix has GPTQ ~ RTN there. Prediction: keeping just those 5
# rows (72k weights, 0.0009% of the model) at fp16 removes the pathology.
#  1  Llama 3-bit, fp16 the 5 rows (L0: 1430,1268; L1: 788,1384,4062): IFEval + per-position divergence
#  2  Llama 4-bit, same rows: does the remaining 4-bit gap (.745 vs .768) close?
#  3  Llama 3-bit, only the 3 layer-1 rows
#  4  Qwen2.5-14B default stats re-run with the new columns (which rows / input
#     channels carry the template error at its layer-4 down_proj; super weight
#     is at layer-3 down_proj row 458 col 1526)
#   qsub jobs/w52_swrows.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
mkdir -p runs/stats runs/coords

mkcoords () {  # $1 out-file $2 spec "layer:row,row;layer:row"   (all 14336 columns of each row)
  python - "$1" "$2" <<'PYEOF'
import sys, csv
out, spec = sys.argv[1], sys.argv[2]
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["layer", "proj", "row", "col"])
    for part in spec.split(";"):
        layer, rows = part.split(":")
        for r in rows.split(","):
            for c in range(14336):
                w.writerow([int(layer), "down_proj", int(r), c])
print("coords ->", out)
PYEOF
}

fq () {  # $1 ckpt $2 tag $3 div-out-or-"-" $4 bits $5 coords-file
  local ckpt="$1" tag="$2" div="$3" bits="$4" cf="$5"
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$LLAMA" --bits "$bits" --group-size 128 --protect coords --coords-file "$cf" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  if [ "$div" != "-" ]; then
    $T python src/divergence.py --fp16 "$LLAMA" --quant "$CKPT" --prompts "$FULL" --n 32 --per-position --out "$div"
  fi
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1) mkcoords runs/coords/llama_swrows_L0L1.csv "0:1430,1268;1:788,1384,4062"
     fq llama3.1-8b-v2gptq3-swrows5  v2l_swrows5  runs/div_l_swrows5.csv 3 runs/coords/llama_swrows_L0L1.csv ;;
  2) mkcoords runs/coords/llama_swrows_L0L1.csv "0:1430,1268;1:788,1384,4062"
     fq llama3.1-8b-v2gptq4-swrows5  v2l4_swrows5 - 4 runs/coords/llama_swrows_L0L1.csv ;;
  3) mkcoords runs/coords/llama_swrows_L1.csv "1:788,1384,4062"
     fq llama3.1-8b-v2gptq3-swrows3  v2l_swrows3  - 3 runs/coords/llama_swrows_L1.csv ;;
  4) $T python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 --protect none \
       --stats-dir runs/stats/trade-q14-3b-g128 --stats-chat-n 64 --no-save --out "$STORE/models/_unused_theory" ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W52] done task $SGE_TASK_ID"
