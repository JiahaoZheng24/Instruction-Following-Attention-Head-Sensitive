#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=06:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W75
#$ -t 1-4
# W75 (2026-09-16): four cells the W72 tables still lack.
#   1  Meta-Llama-3-8B-Instruct  RTN3 IFEval   (no RTN3 reference existed for this census model)
#   2  Llama-3.2-3B-Instruct     RTN3 IFEval   (same)
#   3  Llama-3.1-8B  GPTQ3 with ONLY layer-1 down_proj re-rounded (the single-matrix excision; W72a had blocks 0-1)
#   4  Llama-3.1-8B  GPTQ3 with the five super-weight rows at fp16 (W72a task 51 failed: coords file missing on the cluster)
#   qsub jobs/w75_gaps2.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 5h"
mkdir -p runs/coords runs/protocols

qz () {   # $1 model $2 ckpt $3 tag $4.. flags
  CK="$STORE/models/$2"; m="$1"; tag="$3"; shift 3
  $T python src/quantize_protected.py --model "$m" --bits 3 --group-size 128 --protect none --calib c4 --out "$CK" "$@"
  run_ifeval "$CK" "$tag"
  cp "$CK/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CK")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CK"
}
mkcoords () {  # $1 out-file $2 spec "layer:row,row;layer:row"  (all 14336 columns of each row)
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

case "$SGE_TASK_ID" in
  1) qz meta-llama/Meta-Llama-3-8B-Instruct f-l3-rtn3  f_l3_rtn3  --rtn ;;
  2) qz meta-llama/Llama-3.2-3B-Instruct    f-l32-rtn3 f_l32_rtn3 --rtn ;;
  3) qz "$LLAMA" f-l-gptq3-rtn1down f_l_rtn1down --rtn-modules "1:down_proj" ;;
  4) mkcoords runs/coords/llama_swrows_L0L1.csv "0:1430,1268;1:788,1384,4062"
     qz "$LLAMA" f-l-gptq3-swrows5 f_l_swrows5 --protect coords --coords-file runs/coords/llama_swrows_L0L1.csv ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W75] done task $SGE_TASK_ID"
