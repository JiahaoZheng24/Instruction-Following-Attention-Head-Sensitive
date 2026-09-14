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
#$ -N IFH_W50
#$ -t 1-8
# W50: the theory table, second pass. W49 showed (Llama, layer-1 down_proj,
# template token <|end_header_id|>):
#   - GPTQ's linear error there is BIT-INDEPENDENT (3-bit 1.6-1.8, 4-bit 1.5-1.7)
#     while RTN's halves (0.17 -> 0.08): the compensation displacement does not
#     scale with the rounding step -> hypothesis: weights in the super-weight's
#     group are annihilated at every bit-width and their compensation is fixed.
#   - within one bit-width the down_proj template error orders the arms
#     (1.57 > 1.34 | 1.20 > 1.0 > 0.27) with the collapse boundary between
#     1.34 and 1.20; across bit-widths it does not (4-bit 1.68, no collapse).
#   - a two-factor product, down_proj template error x template error of the
#     other 13 block-0/1 matrices, separates all 9 arms (threshold 8.6-9.5).
# Pre-registered tests:
#  1  default 3-bit g128 stats re-run with the new columns (absolute errors,
#     output-channel concentration of the template error)
#  2  3-bit PER-CHANNEL: product = 27.7 -> collapse predicted; IFEval + per-position divergence
#  3  c4chat per-position divergence (product 2.9 -> no collapse; layer-2 error at the
#     template token predicted small although the down_proj linear error is 1.68)
#  4  4-bit g32 stats (finer 4-bit: product should stay well below threshold)
#  5  3-bit g64 stats (known collapse .145: product should exceed threshold)
#  6  4-bit stats at damping 0.01 (is there any 4-bit recipe that crosses?)
#  7  3-bit g128 WITHOUT act-order, stats: does the bit-independent template
#     error at layer-1 down_proj disappear when the super-activation input
#     columns are no longer grouped with the giant-weight columns?
#     (annihilation-by-grouping hypothesis; IFEval of this arm is .156)
#  8  same arm, per-position divergence
#   qsub jobs/w50_theory2.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
mkdir -p runs/stats

st () {  # $1 stats-dir $2 bits $3 group $4.. flags
  local sd="$1" bits="$2" grp="$3"; shift 3
  $T python src/quantize_protected.py --model "$LLAMA" --bits "$bits" --group-size "$grp" --protect none \
     --stats-dir "$sd" --stats-chat-n 64 --no-save "$@" --out "$STORE/models/_unused_theory"
}
fqd () {  # $1 ckpt $2 tag $3 div-out $4 bits $5 group $6.. flags   (IFEval + per-position divergence, deleted)
  local ckpt="$1" tag="$2" div="$3" bits="$4" grp="$5"; shift 5
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$LLAMA" --bits "$bits" --group-size "$grp" --protect none "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  $T python src/divergence.py --fp16 "$LLAMA" --quant "$CKPT" --prompts "$FULL" --n 32 --per-position --out "$div"
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1) st runs/stats/theory-l-3b-g128 3 128 ;;
  2) fqd llama3.1-8b-v2gptq3-gm1     v2l_gm1     runs/div_l_gm1.csv    3 -1 ;;
  3) fqd llama3.1-8b-v2gptq3-c4chat-tmp v2l_c4chat_div runs/div_l_c4chat.csv 3 128 --calib c4chat ;;
  4) st runs/stats/theory-l-4b-g32   4 32 ;;
  5) st runs/stats/theory-l-3b-g64   3 64 ;;
  6) st runs/stats/theory-l-4b-damp0p01 4 128 --percdamp 0.01 ;;
  7) st runs/stats/theory-l-3b-noao 3 128 --no-actorder ;;
  8) fqd llama3.1-8b-v2gptq3-noao-tmp v2l_noao_div runs/div_l_noao.csv 3 128 --no-actorder ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W50] done task $SGE_TASK_ID"
