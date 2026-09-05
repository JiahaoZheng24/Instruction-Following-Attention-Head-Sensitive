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

run_ifeval () {   # $1 ckpt-or-hf-id  $2 tag   ($T = optional timeout prefix set by the job)
  ${T:-} python src/diagnose_heads.py ablate --model "$1" --prompts "$FULL" --tag "$2" --batch 16
  ${T:-} python src/score_ifeval.py --responses "runs/$(basename "$1")/$2/responses.jsonl" \
    --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}
