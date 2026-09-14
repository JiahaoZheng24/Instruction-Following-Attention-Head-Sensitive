#!/bin/bash
# Delete downloaded model weights from the HF cache to free /store01.
#   bash jobs/hf_cache_rm.sh --list                      # show cached models and sizes
#   bash jobs/hf_cache_rm.sh Qwen/Qwen3-14B allenai/Llama-3.1-Tulu-3-8B   # delete these
#   bash jobs/hf_cache_rm.sh --blind                      # delete all 19 W53 blind-test models
HF_HOME="${HF_HOME:-/store01/yshi4/jzheng7/hf_cache}"
HUB="$HF_HOME/hub"
BLIND="allenai/Llama-3.1-Tulu-3-8B NousResearch/Hermes-3-Llama-3.1-8B deepseek-ai/DeepSeek-R1-Distill-Llama-8B NousResearch/Hermes-2-Pro-Llama-3-8B Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-0.5B-Instruct deepseek-ai/DeepSeek-R1-Distill-Qwen-14B Qwen/Qwen2-7B-Instruct ibm-granite/granite-3.1-8b-instruct HuggingFaceTB/SmolLM3-3B google/gemma-3-1b-it mistralai/Ministral-8B-Instruct-2410 CohereLabs/aya-expanse-8b tiiuae/Falcon3-10B-Instruct meta-llama/Llama-2-7b-chat-hf mistralai/Mistral-7B-Instruct-v0.1"
case "${1:-}" in
  --list) du -sh "$HUB"/models--* 2>/dev/null | sort -h; exit 0 ;;
  --blind) set -- $BLIND ;;
  "") echo "usage: bash jobs/hf_cache_rm.sh --list | --blind | <org/model> ..."; exit 2 ;;
esac
for m in "$@"; do
  d="$HUB/models--${m//\//--}"
  if [ -d "$d" ]; then echo "rm $(du -sh "$d" | cut -f1)  $d"; rm -rf "$d"; else echo "not cached: $m"; fi
done
df -h "$HF_HOME" | tail -1
