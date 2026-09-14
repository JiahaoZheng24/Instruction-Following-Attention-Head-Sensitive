#!/bin/bash
# Pack the results of one batch into a single archive for hand-copy.
#   bash jobs/pack_batch.sh w51_tradeclip.sh          -> archives/w51_tradeclip_results.tgz
#   bash jobs/pack_batch.sh w51_tradeclip.sh w50_theory2.sh   (several batches, one archive)
# Collects every file under runs/ and logs/ that is NEWER than the (oldest)
# named job script, i.e. everything those batches produced. On the laptop:
#   tar xzf W51_results.tgz -C /d/Jiahao_Zheng/Instruction-Following-Attention-Head-Sensitive
# (paths inside the archive are repo-relative, so it drops straight into runs/ and logs/).
set -e
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "usage: bash jobs/pack_batch.sh <job-script> [more job scripts]"; exit 2; }
mkdir -p archives
ref=""
for j in "$@"; do
  f="jobs/$(basename "$j")"
  [ -f "$f" ] || { echo "no such job script: $f"; exit 2; }
  if [ -z "$ref" ] || [ "$f" -ot "$ref" ]; then ref="$f"; fi
done
name="$(basename "$1" .sh)"
out="archives/${name}_results.tgz"
list="$(mktemp)"
find runs logs -type f -newer "$ref" 2>/dev/null | sort > "$list"
n=$(wc -l < "$list")
[ "$n" -gt 0 ] || { echo "nothing newer than $ref under runs/ or logs/"; rm -f "$list"; exit 1; }
tar czf "$out" -T "$list"
echo "packed $n files newer than $ref -> $out ($(du -h "$out" | cut -f1))"
echo "  scores:   $(grep -c 'runs/scores_' "$list")   stats dirs: $(grep -o 'runs/stats/[^/]*' "$list" | sort -u | wc -l)   logs: $(grep -c '^logs/' "$list")"
rm -f "$list"
