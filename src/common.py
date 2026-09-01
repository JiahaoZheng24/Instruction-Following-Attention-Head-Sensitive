"""Shared utilities: model loading, head geometry (GQA-aware), ablation hooks.

Design decisions (fixed before experiments, do not tune post-hoc):
- Head unit = QUERY head. For GQA models (Qwen2.5: 28 q-heads / 4 kv-groups),
  ablation and later weight protection operate on q-head slices:
  o_proj input slice [h*d : (h+1)*d]; W_Q rows per head; W_O cols per head;
  W_K/W_V are handled per kv-group (each q-head maps to group h // (H // H_kv)).
- Ablation = mean ablation: replace the head's o_proj input slice with its mean
  vector computed on a held-out calibration set (never the eval prompts).
"""
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _is_gptq_dir(path: str) -> bool:
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "quantize_config.json"))
        or os.path.exists(os.path.join(path, "quant_config.json")))


def load_model(model_id: str = DEFAULT_MODEL, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if _is_gptq_dir(model_id):
        # GPTQ checkpoints: fancy low-bit kernels are NOT trustworthy (the
        # Triton 3-bit dequant kernel produced input-independent garbage,
        # W0 smoke test 2026-08). Default to the plain-PyTorch backend —
        # slower but numerically correct; override via IFH_GPTQ_BACKEND.
        from gptqmodel import BACKEND, GPTQModel
        backend = BACKEND[os.environ.get("IFH_GPTQ_BACKEND", "TORCH")]
        gm = GPTQModel.load(model_id, backend=backend)
        model = gm.model  # underlying transformers model (hooks need .model.layers)
        model.eval()
        print(f"[load] GPTQ checkpoint via gptqmodel backend={backend}")
        return model, tok
    # No output_attentions needed anywhere in the new pipeline -> flash/sdpa OK.
    # transformers 5.x renamed torch_dtype -> dtype; support both.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
    model.eval()
    return model, tok


class HeadGeom:
    def __init__(self, model):
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = getattr(cfg, "num_key_value_heads", self.n_heads)
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // self.n_heads)
        self.group = self.n_heads // self.n_kv

    def kv_group(self, h: int) -> int:
        return h // self.group

    def slice(self, h: int):
        return slice(h * self.head_dim, (h + 1) * self.head_dim)


def o_proj(model, layer: int):
    return model.model.layers[layer].self_attn.o_proj


# ---------------------------------------------------------------- hooks

class MeanCapture:
    """Accumulates per-head mean of o_proj input (and mean |x| salience)."""

    def __init__(self, model, geom: HeadGeom):
        self.geom = geom
        self.sum = torch.zeros(geom.n_layers, geom.n_heads, geom.head_dim, dtype=torch.float64)
        self.abs_sum = torch.zeros(geom.n_layers, geom.n_heads, dtype=torch.float64)
        self.count = 0
        self.handles = [
            o_proj(model, l).register_forward_pre_hook(self._make(l))
            for l in range(geom.n_layers)
        ]

    def _make(self, layer):
        def hook(_mod, args):
            x = args[0].detach()  # [B, T, H*d]
            b, t, _ = x.shape
            xs = x.reshape(b * t, self.geom.n_heads, self.geom.head_dim).to(torch.float64).cpu()
            self.sum[layer] += xs.sum(dim=0)
            self.abs_sum[layer] += xs.abs().mean(dim=-1).sum(dim=0)
            if layer == 0:
                self.count += b * t
        return hook

    def finalize(self):
        for h in self.handles:
            h.remove()
        means = (self.sum / max(self.count, 1)).to(torch.float32)      # [L, H, d]
        act_sal = (self.abs_sum / max(self.count, 1)).to(torch.float32)  # [L, H]
        return means, act_sal


class HeadAblator:
    """Mean-ablates a set of (layer, head) pairs during generation."""

    def __init__(self, model, geom: HeadGeom, means: torch.Tensor):
        self.model, self.geom = model, geom
        self.means = means  # [L, H, d] on CPU float32
        self.active: dict[int, list[int]] = {}
        self.handles = []

    def set_heads(self, heads: list[tuple[int, int]]):
        self.clear()
        by_layer: dict[int, list[int]] = {}
        for l, h in heads:
            by_layer.setdefault(l, []).append(h)
        self.active = by_layer
        for l, hs in by_layer.items():
            mod = o_proj(self.model, l)
            vecs = {h: self.means[l, h] for h in hs}

            def make(layer_heads=hs, layer_vecs=vecs):
                def hook(mod_, args):
                    x = args[0]
                    for h in layer_heads:
                        v = layer_vecs[h].to(device=x.device, dtype=x.dtype)
                        x[..., h * self.geom.head_dim:(h + 1) * self.geom.head_dim] = v
                    return (x,) + tuple(args[1:])
                return hook

            self.handles.append(mod.register_forward_pre_hook(make()))

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles, self.active = [], {}


def safe_quantile(t: torch.Tensor, q: float) -> float:
    """torch.quantile with a hard 2^24-element limit worked for 7-8B salience
    samples but crashes at 14B+. Deterministically subsample to 10M first —
    quantile estimation error at that sample size is negligible for our
    thresholds."""
    t = t.float().flatten()
    if t.numel() > 10_000_000:
        g = torch.Generator().manual_seed(0)
        t = t[torch.randint(0, t.numel(), (10_000_000,), generator=g)]
    return torch.quantile(t, q).item()


# ---------------------------------------------------------------- io

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_heads(spec: str) -> list[tuple[int, int]]:
    """'12:5,17:20' -> [(12,5), (17,20)]"""
    out = []
    for item in spec.split(","):
        l, h = item.strip().split(":")
        out.append((int(l), int(h)))
    return out
