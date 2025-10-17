
# IF-Attention Analysis (FP16 vs Quantized)

This mini-repo gives you **end-to-end code and a reproducible workflow** to:
1) Select *instruction-constrained* IFEval prompts (e.g., "use two words", "start with", "JSON").
2) Dump **per-layer, per-head attentions** targeted at *instruction tokens* for **FP16** and **quantized** models (GPTQ/AWQ).
3) Compute deltas/indices (e.g., **Instruction Sensitivity Index, ISI**) and correlate with **IFEval score drops**.
4) Visualize heatmaps and summary plots.

It assumes your environment can load your local models (no internet). Model identifiers can be either HF IDs or local folders.

> Your uploaded IFEval outputs referenced below:  
> - `samples_ifeval_2025-10-09T13-58-12.528717.jsonl`  
> - `results_2025-10-09T13-58-12.528717.json`

---

## Quickstart

```bash
# 0) (Optional) create env
conda create -n ifattn python=3.10 -y && conda activate ifattn

# 1) Install deps (GPU recommended)
pip install -U torch transformers accelerate einops matplotlib numpy scipy pandas tqdm pyyaml

# If you will load GPTQ or AWQ directly:
pip install -U auto-gptq optimum autoawq  # (whichever you use; some envs use autoawq, some awq)
# In some stacks: pip install -U awq

# 2) Edit config
vim config.yaml
# - set paths/ids for FP16, GPTQ, AWQ
# - (optional) add keyword rules

# 3) Select instruction-constrained prompts from your IFEval samples
python select_prompts.py   --samples_jsonl "/mnt/data/samples_ifeval_2025-10-09T13-58-12.528717.jsonl"   --out_jsonl "artifacts/selected_prompts.jsonl"   --top_k 10

# 4) Dump attentions for each model (FP16, GPTQ, AWQ). Example:
python dump_attn.py --model_id "Qwen/Qwen2.5-7B-Instruct" --run_tag "fp16"   --prompts_jsonl "artifacts/selected_prompts.jsonl" --out_dir "artifacts"

python dump_attn.py --model_id "/path/to/your/gptq_model" --run_tag "gptq4"   --prompts_jsonl "artifacts/selected_prompts.jsonl" --out_dir "artifacts"

python dump_attn.py --model_id "/path/to/your/awq_model" --run_tag "awq4"   --prompts_jsonl "artifacts/selected_prompts.jsonl" --out_dir "artifacts"

# 5) Parse IFEval results (your JSON files) and compute average 4-score metric
python parse_ifeval.py   --results_json "/mnt/data/results_2025-10-09T13-58-12.528717.json"   --out_csv "artifacts/ifeval_scores.csv"

# (Repeat step 5 for your GPTQ/AWQ results json files and put them in the same CSV or separate CSVs)

# 6) Compute instruction sensitivity metrics + correlations
python compute_metrics.py   --attn_glob "artifacts/attn_*.npz"   --ifeval_csv "artifacts/ifeval_scores_all_models.csv"   --out_dir "artifacts"

# 7) Visualize
python viz.py --attn_glob "artifacts/attn_*.npz" --out_dir "artifacts"
```

---

## Files

- `config.yaml` – model IDs/paths, token rules, prompt filters.
- `select_prompts.py` – choose instruction-constrained prompts from IFEval samples.
- `dump_attn.py` – run model with `output_attentions=True`, extract per-head attentions on *instruction tokens*, save to `npz`.
- `parse_ifeval.py` – read `results_*.json` (lm-eval-harness) and compute the 4-metric average.
- `compute_metrics.py` – compute **A_head**, **ΔA**, **ISI**, aggregate; correlate with IFEval drops.
- `viz.py` – heatmaps & summary plots.

---

## Notes

- For robust token-to-char mapping, we rely on *fast tokenizers* (`return_offsets_mapping=True`).
- If your GPTQ/AWQ integration replaces attention modules, `output_attentions=True` may still work; if not, see the hooks fallback in `dump_attn.py` (register_forward_hook).
- Keywords in `config.yaml` define instruction segments; adjust to your IFEval subset.
- ISI definition (default):
  - Per-head attention to instruction tokens: average over selected tokens and over prompt positions.
  - ΔA per head: `A_fp16 - A_quant` (positive means quantization reduced instruction focus).
  - ISI: mean of positive ΔA across heads (normalized by number of heads).

Good luck – this pipeline is modular so you can swap models/benchmarks as needed.
