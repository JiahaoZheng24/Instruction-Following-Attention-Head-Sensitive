#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m ae
#$ -q long
#$ -l h_rt=1:00:00
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_CLEANUP
# Delete every downloaded model from the HF cache EXCEPT the keep list, once the
# batches it is held on have finished. No GPU needed. Submit AFTER W63/W64 so the
# hold applies (SGE holds on job NAMES; a name that is not in the queue is ignored):
#   qsub -hold_jid IFH_W63,IFH_W64 jobs/w6x_cleanup.sh
# Or run it directly on the front end once qstat is empty:
#   bash jobs/w6x_cleanup.sh            # deletes
#   bash jobs/w6x_cleanup.sh --dry-run  # only lists what would go
#   add --ckpts to also sweep /store01/.../models (off by default; kept checkpoints live there)
#   add --glob 'PAT' to sweep only the checkpoints matching PAT, for leftovers of a batch whose
#   tasks died before their own rm -rf, e.g.  bash jobs/w6x_cleanup.sh --dry-run --glob '*84-*'
HF_HOME="${HF_HOME:-/store01/yshi4/jzheng7/hf_cache}"
HUB="$HF_HOME/hub"
KEEP="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct"
DRY=0; CK=0; GLOB=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --ckpts)   CK=1 ;;
    --glob)    GLOB="$2"; shift ;;
  esac
  shift
done

echo "[cleanup] $(date)  before:"; df -h "$HF_HOME" | tail -1
for d in "$HUB"/models--*; do
  [ -d "$d" ] || continue
  name="${d##*/models--}"; name="${name//--/\/}"
  keep=0; for k in $KEEP; do [ "$name" = "$k" ] && keep=1; done
  size=$(du -sh "$d" 2>/dev/null | cut -f1)
  if [ $keep = 1 ]; then echo "keep $size  $name"; continue; fi
  if [ $DRY = 1 ]; then echo "would rm $size  $name"; else echo "rm $size  $name"; rm -rf "$d"; fi
done
# stray fake-quant checkpoints: only with --ckpts (some checkpoints under models/
# were kept on purpose, e.g. W33's Mistral trigger-2 weights; check the list first)
[ $CK = 1 ] && for d in /store01/yshi4/jzheng7/models/*; do
  [ -d "$d" ] || continue
  case "$(basename "$d")" in _unused_theory) continue ;; esac
  if [ $DRY = 1 ]; then echo "would rm ckpt $(du -sh "$d" | cut -f1)  $d"; else echo "rm ckpt $(du -sh "$d" | cut -f1)  $d"; rm -rf "$d"; fi
done
# targeted sweep: only the checkpoints whose basename matches --glob
if [ -n "$GLOB" ]; then
  echo "[cleanup] checkpoints matching '$GLOB':"
  n=0
  for d in /store01/yshi4/jzheng7/models/*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    case "$b" in
      $GLOB) n=$((n + 1))
             if [ $DRY = 1 ]; then echo "would rm ckpt $(du -sh "$d" | cut -f1)  $b"
             else echo "rm ckpt $(du -sh "$d" | cut -f1)  $b"; rm -rf "$d"; fi ;;
    esac
  done
  [ $n = 0 ] && echo "  (none: every task deleted its own checkpoint)"
fi
echo "[cleanup] after:"; df -h "$HF_HOME" | tail -1
