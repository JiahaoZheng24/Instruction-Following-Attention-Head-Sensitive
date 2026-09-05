#!/bin/bash
# Non-destructive checkpoint housekeeping (run on the login node, no GPU):
#   1. copy every checkpoint's PROTECT_PROTOCOL.json / ROTATE_PROTOCOL.json
#      into runs/protocols/<ckpt>_<file>  (reproducibility record, as before)
#   2. print size + KEEP/DELETABLE for every dir under $STORE/models
#   3. write the deletable list to runs/ckpt_deletable.txt
# Nothing is deleted. To delete, review the list and run the printed command.
#   bash jobs/archive_and_list_ckpts.sh
set -e
STORE="/store01/yshi4/jzheng7"
cd "$STORE/Instruction-Following-Attention-Head-Sensitive"
mkdir -p runs/protocols

# Checkpoints still needed for follow-up evals / rebuttal. Everything else has
# all of its results (scores, responses, general evals) recorded under runs/.
KEEP='^(qwen2\.5-7b-v2gptq3-(none|tacq|heads32|randw)|llama3\.1-8b-v2gptq3-(none|tacq|tacq1e5|damp5|calinst|calwiki|tacq_sx|tacq1e5_sx|hmag1e5|hmag40m)|qwen2\.5-14b-v2gptq3-(none|tacq|tacq1e6|damp5|calinst|tacq1e5_sx|tacq1e5_sx_cs1|hmag1e6)|llama3\.1-8b-awq3(asym)?-none|qwen2\.5-14b-awq3(asym)?-none|llama3\.1-8b-rtn3-none|.*gptq2.*)$'

: > runs/ckpt_deletable.txt
total_del=0
printf "%-48s %8s  %s\n" "checkpoint" "size" "verdict"
for d in "$STORE"/models/*/; do
  n=$(basename "$d")
  for f in PROTECT_PROTOCOL.json ROTATE_PROTOCOL.json quantize_config.json; do
    [ -f "$d/$f" ] && cp -n "$d/$f" "runs/protocols/${n}_$f" 2>/dev/null || true
  done
  sz=$(du -sh "$d" 2>/dev/null | cut -f1)
  if [[ "$n" =~ $KEEP ]]; then v="KEEP"; else v="deletable"; echo "$d" >> runs/ckpt_deletable.txt; fi
  printf "%-48s %8s  %s\n" "$n" "$sz" "$v"
done
echo
echo "protocol JSONs archived: $(ls runs/protocols | wc -l) files in runs/protocols/"
echo "deletable dirs: $(wc -l < runs/ckpt_deletable.txt)  (list: runs/ckpt_deletable.txt)"
echo "total deletable size:"; xargs -d '\n' du -shc < runs/ckpt_deletable.txt 2>/dev/null | tail -1
echo
echo "To delete after reviewing the list:"
echo "  xargs -d '\\n' rm -rf < runs/ckpt_deletable.txt"
