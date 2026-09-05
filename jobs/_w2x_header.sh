# shared preamble for W20+ jobs (sourced from the job scripts, not executed)
set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
LLAMA="meta-llama/Llama-3.1-8B-Instruct"
Q7="Qwen/Qwen2.5-7B-Instruct"
Q14="Qwen/Qwen2.5-14B-Instruct"
FULL="data/ifeval_input_data.jsonl"
SAL_L="$STORE/salience/llama31-8b-if"
SAL14="$STORE/salience/qwen25-14b-if"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

# Orphan guard: on qdel (SIGTERM/SIGKILL to the job shell), on timeout, or on
# normal exit, kill every child of this shell so no python/CUDA process can
# outlive the job and hold the card. SGE's own kill uses the job's group id,
# so this is belt-and-braces; it costs nothing when there is nothing to kill.
ifh_cleanup () {
  pkill -TERM -P $$ 2>/dev/null || true
  sleep 3
  pkill -KILL -P $$ 2>/dev/null || true
  [ -n "${IFH_OFFLOAD_DIR:-}" ] && rm -rf "$IFH_OFFLOAD_DIR" 2>/dev/null || true
}
trap ifh_cleanup TERM INT HUP EXIT

run_ifeval () {   # $1 ckpt-or-hf-id  $2 tag   ($T = optional timeout prefix set by the job)
  ${T:-} python src/diagnose_heads.py ablate --model "$1" --prompts "$FULL" --tag "$2" --batch 16
  ${T:-} python src/score_ifeval.py --responses "runs/$(basename "$1")/$2/responses.jsonl" \
    --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}
