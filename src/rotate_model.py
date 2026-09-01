"""QuaRot-style R1 rotation: fold RMSNorms, then rotate the residual stream
with a fixed random orthogonal matrix Q, fused offline into the weights.

Purpose (W10, unfreeze batch): rotation-based outlier mitigation
(QuaRot/SpinQuant) is the modern answer to quantization outliers. Question:
is the collapse regime (Llama@3bit) a property of the legacy weight basis
that rotation removes, or does it survive incoherence processing?

Scope: R1 only (residual-stream rotation), fused offline — the rotation that
matters for weight-only quantization. R3/R4 (online rotations for KV/act
quant) are out of scope. Q is a random orthogonal matrix (QR of a Gaussian,
float64) rather than a Hadamard: same incoherence effect (QuIP-style), works
for any hidden size (Qwen 3584 is not a Hadamard-friendly power of two), and
runtime cost is irrelevant for fake-quant research checkpoints.

Exact function preservation in exact arithmetic: RMSNorm with unit weight
commutes with any orthogonal Q (it only depends on the vector norm), RoPE /
softmax / SiLU all act downstream of the projections in unrotated head or
intermediate space, and input-side biases (Qwen q/k/v) live after the
projection. Stored in bf16, so expect rounding-level logit diffs only —
checked and printed by `logit_probe`.

CLI (save a rotated-but-unquantized checkpoint — the fp16-equivalence arm):
  python src/rotate_model.py --model meta-llama/Llama-3.1-8B-Instruct \
      --seed 0 --out $STORE/models/llama3.1-8b-rotfp
"""
import argparse
import json
import os

import torch
import torch.nn as nn

from common import load_model


def build_q(dim: int, seed: int) -> torch.Tensor:
    """Deterministic random orthogonal [dim, dim] in float64 (CPU)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(dim, dim, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # unique QR
    return q


def _fold_norm_into(norm, linears):
    """W <- W diag(g); norm weight <- 1. Input-side scaling absorbed."""
    g = norm.weight.data.double()
    for lin in linears:
        W = lin.weight.data
        lin.weight.data = (W.double() * g.to(W.device).unsqueeze(0)).to(W.dtype)
    norm.weight.data = torch.ones_like(norm.weight.data)


def _rot_in(lin, Q):
    """Input side (reads rotated residual): W <- W @ Q."""
    W = lin.weight.data
    lin.weight.data = (W.double() @ Q.to(W.device)).to(W.dtype)


def _rot_out(lin, Q):
    """Output side (writes rotated residual): W <- Q^T @ W."""
    assert lin.bias is None, "output-side bias would need rotating too"
    W = lin.weight.data
    lin.weight.data = (Q.to(W.device).T @ W.double()).to(W.dtype)


@torch.no_grad()
def fold_and_rotate(model, seed: int = 0):
    cfg = model.config
    if getattr(cfg, "tie_word_embeddings", False):
        # norm folding into lm_head breaks the tie — clone first
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone())
        cfg.tie_word_embeddings = False
    Q = build_q(cfg.hidden_size, seed)
    emb = model.model.embed_tokens
    emb.weight.data = (emb.weight.data.double()
                       @ Q.to(emb.weight.device)).to(emb.weight.dtype)
    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        _fold_norm_into(layer.input_layernorm, [a.q_proj, a.k_proj, a.v_proj])
        _fold_norm_into(layer.post_attention_layernorm, [m.gate_proj, m.up_proj])
        for lin in (a.q_proj, a.k_proj, a.v_proj, m.gate_proj, m.up_proj):
            _rot_in(lin, Q)
        _rot_out(a.o_proj, Q)
        _rot_out(m.down_proj, Q)
    _fold_norm_into(model.model.norm, [model.lm_head])
    _rot_in(model.lm_head, Q)
    print(f"[rotate] R1 fused: dim={cfg.hidden_size} seed={seed} "
          f"Q checksum={float(Q.abs().sum()):.6f}", flush=True)
    return model


@torch.no_grad()
def logit_probe(model, tok,
                text="The quick brown fox jumps over the lazy dog. In 1969,"):
    ids = tok(text, return_tensors="pt").to(model.device)
    return model(**ids).logits[0].float().cpu()  # [T, vocab]


def probe_report(before: torch.Tensor, after: torch.Tensor,
                 min_top1: float = 0.9) -> str:
    diff = (before - after).abs().max().item()
    top1 = (before.argmax(-1) == after.argmax(-1)).float().mean().item()
    rep = f"max|dlogit|={diff:.4f} top1-agreement={top1:.3f}"
    if top1 < min_top1:
        raise RuntimeError(
            f"rotation broke the model ({rep}) — aborting before quantization")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, tok = load_model(args.model)
    ref = logit_probe(model, tok)
    fold_and_rotate(model, seed=args.seed)
    rep = probe_report(ref, logit_probe(model, tok))
    print(f"[rotate] sanity (bf16 rounding only expected): {rep}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "ROTATE_PROTOCOL.json"), "w") as f:
        json.dump({"model": args.model, "rotation": "r1_random_orthogonal",
                   "seed": args.seed, "quantized": False, "sanity": rep},
                  f, indent=2)
    print(f"[rotate] saved -> {args.out}")


if __name__ == "__main__":
    main()
