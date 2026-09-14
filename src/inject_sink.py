"""W66-W68: sink-leakage tolerance — the third gate, measured without quantization.

The GPTQ lesion (W50–W52) writes a spurious component along the BOS / sink
direction into the residual stream at the chat-template positions right after
the sink-forming layer. Whether that becomes a collapse depends on how much the
rest of the network amplifies it (W57: decided in layers 2–4). This tool
injects exactly that perturbation into a model (fp16 or a quantized
checkpoint), with a dose knob, so the tolerance can be measured directly:

    h_L[p] <- h_L[p] + beta * ||h_L[p]|| * u_L
    u_L = mean unit BOS hidden state after layer L (the massive-activation direction)
    p   = template positions lo..hi of the prompt (prefill), and/or (W68) every
          token whose id is in a token set (chat-template specials, newlines),
          prefill AND decode — the lesion is a weight defect that fires on such
          tokens throughout generation, not a fixed prompt perturbation.

Modes
  measure   fp16 only: per layer, BOS / template / rest hidden norms and the
            attention mass later tokens put on BOS, on positions 2-7, on the rest.
  inject    generate the IFEval responses with the perturbation (then score with
            src/score_ifeval.py exactly like a quantized checkpoint).
  artifact  fp16 vs a quantized checkpoint: cosine of the actual quantization
            error at the template positions after layer L with u_L, relative
            magnitude, and the same at ordinary positions as control. With
            --continuation N the prompts are extended by the fp16 model's own
            greedy response and the error is reported per token class
            (template / newline / ordinary) for prompt and response separately.
"""
import argparse
import csv
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEFAULT_MODEL, load_model, read_jsonl, write_jsonl  # noqa: E402

MAX_NEW_TOKENS = 1280


def chat(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)


def parse_pos(s):
    lo, _, hi = s.partition("-")
    return int(lo), int(hi or lo)


def token_set(tok, spec):
    """W68: ids of the tokens the lesion fires on during generation.
    'template' = every special token of the tokenizer (chat-template markers);
    'newline'  = every vocab entry that decodes to newlines only (optionally with spaces)."""
    ids = set()
    for name in [x for x in spec.split(",") if x]:
        if name == "template":
            ids |= set(int(i) for i in tok.all_special_ids)
        elif name == "newline":
            for i in range(len(tok)):
                t = tok.decode([i])
                if t and "\n" in t and t.strip(" \n\r\t") == "":
                    ids.add(i)
        else:
            raise ValueError(name)
    return ids


@torch.no_grad()
def hidden_after(model, tok, texts, layers, max_len=1024, dp_layer=None):
    """dict layer -> list of [T, d] float CPU tensors (output of that layer).
    If dp_layer is given, also returns the INPUT of layers[dp_layer].mlp.down_proj
    per text ([T, d_ff]) under the key ('dpin', dp_layer) — the x_t of the
    derivation (W68: ||e_t|| should scale with |x_0^T x_t|)."""
    out = {L: [] for L in layers}
    cap = []
    h = None
    if dp_layer is not None:
        out[("dpin", dp_layer)] = []
        h = model.model.layers[dp_layer].mlp.down_proj.register_forward_pre_hook(
            lambda _m, a: cap.append(a[0][0].detach().float().cpu()))
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        o = model(**ids, output_hidden_states=True, use_cache=False)
        for L in layers:
            out[L].append(o.hidden_states[L + 1][0].float().cpu())
        if dp_layer is not None:
            out[("dpin", dp_layer)].append(cap[-1])
            cap.clear()
    if h is not None:
        h.remove()
    return out


def bos_direction(hs):
    u = torch.stack([h[0] / h[0].norm().clamp(min=1e-6) for h in hs]).mean(0)
    return u / u.norm()


# ------------------------------------------------------------------ measure
@torch.no_grad()
def mode_measure(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto",
                                                     attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto",
                                                     attn_implementation="eager")
    model.eval()
    prompts = read_jsonl(args.prompts)[: args.n]
    lo, hi = parse_pos(args.pos)
    nL = model.config.num_hidden_layers
    acc = {L: {"bos_norm": 0.0, "tpl_norm": 0.0, "rest_norm": 0.0, "att_bos": 0.0, "att_p1": 0.0,
               "att_tpl": 0.0, "att_rest": 0.0} for L in range(nL)}
    n = 0
    for ex in prompts:
        ids = tok(chat(tok, ex["prompt"]), return_tensors="pt", truncation=True,
                  max_length=args.max_len).to(model.device)
        T = ids["input_ids"].shape[1]
        if T <= hi + 2:
            continue
        o = model(**ids, output_hidden_states=True, output_attentions=True, use_cache=False)
        for L in range(nL):
            h = o.hidden_states[L + 1][0].float()
            a = acc[L]
            a["bos_norm"] += float(h[0].norm())
            a["tpl_norm"] += float(h[lo:hi + 1].norm(dim=1).mean())
            a["rest_norm"] += float(h[hi + 1:].norm(dim=1).mean())
            att = o.attentions[L][0].float()          # [heads, T, T]
            q = att[:, hi + 1:, :]                    # queries after the template
            a["att_bos"] += float(q[:, :, 0].mean())
            a["att_p1"] += float(q[:, :, 1].mean())
            a["att_tpl"] += float(q[:, :, lo:hi + 1].sum(-1).mean())
            a["att_rest"] += float(q[:, :, hi + 1:].sum(-1).mean())
        n += 1
        del o
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "bos_norm", "tpl_norm", "rest_norm", "bos_over_rest",
                    "att_bos", "att_p1", "att_tpl", "att_rest"])
        for L in range(nL):
            a = {k: v / max(n, 1) for k, v in acc[L].items()}
            w.writerow([L, f"{a['bos_norm']:.4g}", f"{a['tpl_norm']:.4g}", f"{a['rest_norm']:.4g}",
                        f"{a['bos_norm'] / max(a['rest_norm'], 1e-6):.4g}",
                        f"{a['att_bos']:.4f}", f"{a['att_p1']:.4f}", f"{a['att_tpl']:.4f}", f"{a['att_rest']:.4f}"])
    best = max(range(nL), key=lambda L: acc[L]["bos_norm"] / max(acc[L]["rest_norm"], 1e-6))
    print(f"[measure] {args.model}: {n} prompts; sink-forming layer by BOS/rest norm = L{best} "
          f"({acc[best]['bos_norm'] / n:.1f} vs {acc[best]['rest_norm'] / n:.2f}); "
          f"att_tpl at L{best + 1}..L{best + 3} = "
          + ", ".join(f"{acc[L]['att_tpl'] / n:.3f}" for L in range(best + 1, min(best + 4, nL)))
          + f" -> {args.out}")


# ------------------------------------------------------------------- inject
class Injector:
    """Adds beta*||h[p]||*u to the residual stream after `layer` at (a) the
    template positions lo..hi of the prompt (prefill only) and/or (b) every
    position, prefill AND decode, whose token id is in `token_ids`."""

    def __init__(self, model, layer, u, beta, lo, hi, token_ids=None, use_pos=True):
        self.u = u.to(model.device, torch.float32)
        self.beta, self.lo, self.hi = beta, lo, hi
        self.token_ids = None if not token_ids else torch.tensor(sorted(token_ids), device=model.device)
        self.use_pos = use_pos
        self.mask, self.ids = None, None
        self.h_top = model.register_forward_pre_hook(self._catch_mask, with_kwargs=True)
        self.h_layer = model.model.layers[layer].register_forward_hook(self._inject)
        self.n_injected = 0
        self.n_injected_tok = 0

    def _catch_mask(self, _m, args, kw):
        self.mask = kw.get("attention_mask")
        ids = kw.get("input_ids")
        if ids is None and args:
            ids = args[0]
        self.ids = ids

    def _inject(self, _m, _a, out):
        h = out[0] if isinstance(out, tuple) else out
        B, T, _ = h.shape
        sel = torch.zeros(B, T, dtype=torch.bool, device=h.device)
        if self.use_pos and T > 1:
            for b in range(B):
                if self.mask is not None and self.mask.shape[0] == B and self.mask.shape[1] == T:
                    nz = torch.nonzero(self.mask[b], as_tuple=False)
                    start = int(nz[0]) if nz.numel() else 0
                else:
                    start = 0
                sel[b, start + self.lo: min(start + self.hi + 1, T)] = True
        if (self.token_ids is not None and self.ids is not None
                and self.ids.shape[0] == B and self.ids.shape[-1] == T):
            tsel = torch.isin(self.ids.to(h.device), self.token_ids)
            self.n_injected_tok += int(tsel.sum())
            sel |= tsel
        if not bool(sel.any()):
            return out
        h = h.clone()
        v = h[sel].float()
        h[sel] = (v + self.beta * v.norm(dim=1, keepdim=True) * self.u).to(h.dtype)
        self.n_injected += int(sel.sum())
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def remove(self):
        self.h_top.remove()
        self.h_layer.remove()


@torch.no_grad()
def mode_inject(args):
    model, tok = load_model(args.model)
    prompts = read_jsonl(args.prompts)
    lo, hi = parse_pos(args.pos)
    texts_u = [chat(tok, ex["prompt"]) for ex in prompts[: args.n]]
    hs = hidden_after(model, tok, texts_u, [args.layer], args.max_len)[args.layer]
    u = bos_direction(hs)
    tpl = torch.stack([h[lo:hi + 1].norm(dim=1).mean() for h in hs]).mean()
    print(f"[inject] {args.model} L{args.layer} beta={args.beta}: BOS norm {torch.stack([h[0].norm() for h in hs]).mean():.1f}, "
          f"template norm {tpl:.2f}, top |u| channels {u.abs().topk(3).indices.tolist()} "
          f"(share {float((u.abs().topk(3).values ** 2).sum()):.2f})", flush=True)
    tids = token_set(tok, args.tokens) if args.tokens else None
    if tids:
        print(f"[inject] token mode: {len(tids)} token ids ({args.tokens}); prompt positions "
              f"{'NOT' if args.no_pos else 'also'} injected", flush=True)
    inj = Injector(model, args.layer, u, args.beta, lo, hi, token_ids=tids, use_pos=not args.no_pos)
    out_rows = []
    for i in range(0, len(prompts), args.batch):
        batch = prompts[i:i + args.batch]
        texts = [chat(tok, ex["prompt"]) for ex in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id)
        for ex, seq in zip(batch, gen):
            out_rows.append({"prompt": ex["prompt"],
                             "response": tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)})
        print(f"[inject:{args.tag}] {min(i + args.batch, len(prompts))}/{len(prompts)}", flush=True)
    inj.remove()
    run_dir = os.path.join("runs", os.path.basename(args.model), args.tag)
    os.makedirs(run_dir, exist_ok=True)
    write_jsonl(os.path.join(run_dir, "responses.jsonl"), out_rows)
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(f"model={args.model}\ninject layer={args.layer} beta={args.beta} pos={args.pos} "
                f"tokens={args.tokens or '-'} n_injected={inj.n_injected} n_injected_tok={inj.n_injected_tok}\n")
    print(f"[inject] {len(out_rows)} responses -> {run_dir}/responses.jsonl "
          f"(injected {inj.n_injected} positions, {inj.n_injected_tok} by token id)")


# ----------------------------------------------------------------- artifact
@torch.no_grad()
def mode_artifact(args):
    lo, hi = parse_pos(args.pos)
    model, tok = load_model(args.model)
    prompts = read_jsonl(args.prompts)[: args.n]
    texts = [chat(tok, ex["prompt"]) for ex in prompts]
    L = args.layer
    prompt_len, tids_tpl, tids_nl = None, None, None
    if args.continuation > 0:      # W68: extend each prompt by the fp16 model's own greedy response
        new_texts = []
        for t in texts:
            enc = tok(t, return_tensors="pt").to(model.device)
            gen = model.generate(**enc, do_sample=False, max_new_tokens=args.continuation,
                                 pad_token_id=tok.pad_token_id)
            new_texts.append(t + tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=False))
        prompt_len = [tok(t, return_tensors="pt")["input_ids"].shape[1] for t in texts]
        texts = new_texts
        tids_tpl = token_set(tok, "template")
        tids_nl = token_set(tok, "newline")
    both = hidden_after(model, tok, texts, [L], args.max_len, dp_layer=L)
    h_fp, x_dp = both[L], both[("dpin", L)]
    u = bos_direction(h_fp)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    qm, _ = load_model(args.quant)
    h_q = hidden_after(qm, tok, texts, [L], args.max_len)[L]
    rows = []
    for name, sl in [("bos", slice(0, 1)), ("tpl", slice(lo, hi + 1)), ("rest", slice(hi + 1, hi + 9))]:
        cos, rel, share = [], [], []
        for a, b in zip(h_fp, h_q):
            if b.shape[0] != a.shape[0] or a.shape[0] < hi + 9:
                continue
            d = (b[sl] - a[sl])
            dn = d.norm(dim=1).clamp(min=1e-6)
            proj = d @ u
            cos.append((proj / dn).mean())
            rel.append((dn / a[sl].norm(dim=1).clamp(min=1e-6)).mean())
            share.append(((proj ** 2) / (dn ** 2)).mean())
        rows.append({"class": name, "cos_with_bos_dir": float(torch.stack(cos).mean()),
                     "rel_err": float(torch.stack(rel).mean()),
                     "energy_share_on_bos_dir": float(torch.stack(share).mean())})
    for p in range(lo, hi + 1):
        cos, rel = [], []
        for a, b in zip(h_fp, h_q):
            if b.shape[0] != a.shape[0] or a.shape[0] <= p:
                continue
            d = b[p] - a[p]
            cos.append(float(d @ u / d.norm().clamp(min=1e-6)))
            rel.append(float(d.norm() / a[p].norm().clamp(min=1e-6)))
        rows.append({"class": f"pos{p}", "cos_with_bos_dir": sum(cos) / len(cos),
                     "rel_err": sum(rel) / len(rel), "energy_share_on_bos_dir": float("nan")})
    if args.continuation > 0:      # per token class, prompt part vs response part
        cls = {}
        for a, b, t, pl in zip(h_fp, h_q, texts, prompt_len):
            ids = tok(t, return_tensors="pt", truncation=True, max_length=args.max_len)["input_ids"][0]
            if b.shape[0] != a.shape[0] or a.shape[0] != ids.shape[0]:
                continue
            for p in range(a.shape[0]):
                part = "prompt" if p < pl else "resp"
                kind = ("template" if int(ids[p]) in tids_tpl
                        else ("newline" if int(ids[p]) in tids_nl else "ordinary"))
                d = b[p] - a[p]
                cls.setdefault(f"{part}_{kind}", []).append(
                    (float(d @ u / d.norm().clamp(min=1e-6)), float(d.norm() / a[p].norm().clamp(min=1e-6))))
        for k in sorted(cls):
            v = cls[k]
            rows.append({"class": f"{k}(n={len(v)})", "cos_with_bos_dir": sum(x[0] for x in v) / len(v),
                         "rel_err": sum(x[1] for x in v) / len(v), "energy_share_on_bos_dir": float("nan")})
        # W68 derivation test: ||e_t|| (residual error after layer L) vs |x_0^T x_t| at the down_proj input.
        ov, err, kinds = [], [], []
        for a, b, x, t, pl in zip(h_fp, h_q, x_dp, texts, prompt_len):
            ids = tok(t, return_tensors="pt", truncation=True, max_length=args.max_len)["input_ids"][0]
            if b.shape[0] != a.shape[0] or x.shape[0] != a.shape[0] or a.shape[0] != ids.shape[0]:
                continue
            x0 = x[0]
            for p in range(1, a.shape[0]):
                ov.append(float((x0 @ x[p]).abs() / x0.norm()))
                err.append(float((b[p] - a[p]).norm()))
                kinds.append("template" if int(ids[p]) in tids_tpl else ("newline" if int(ids[p]) in tids_nl else "ordinary"))
        if len(ov) > 10:
            o_t, e_t = torch.tensor(ov), torch.tensor(err)
            pear = float(torch.corrcoef(torch.stack([o_t, e_t]))[0, 1])
            lo_t, le_t = o_t.clamp(min=1e-9).log(), e_t.clamp(min=1e-9).log()
            pear_log = float(torch.corrcoef(torch.stack([lo_t, le_t]))[0, 1])
            qs = torch.quantile(o_t, torch.tensor([0.2, 0.4, 0.6, 0.8]))
            bins = torch.bucketize(o_t, qs)
            for q in range(5):
                m = bins == q
                kk = [kinds[i] for i in torch.nonzero(m).flatten().tolist()]
                frac_tpl = (sum(k != "ordinary" for k in kk) / max(len(kk), 1))
                rows.append({"class": f"overlap_q{q + 1}(n={int(m.sum())},tpl_frac={frac_tpl:.2f})",
                             "cos_with_bos_dir": float("nan"), "rel_err": float(e_t[m].mean()),
                             "energy_share_on_bos_dir": float(o_t[m].mean())})
            rows.append({"class": f"pearson(|x0.x_t|,||e_t||) linear={pear:.3f} log={pear_log:.3f} n={len(ov)}",
                         "cos_with_bos_dir": pear, "rel_err": pear_log, "energy_share_on_bos_dir": float("nan")})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "cos_with_bos_dir", "rel_err", "energy_share_on_bos_dir"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        print(f"[artifact] {r['class']:22s} cos={r['cos_with_bos_dir']:.3f} rel={r['rel_err']:.3f} "
              f"share={r['energy_share_on_bos_dir']:.3f}")
    print(f"[artifact] -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["measure", "inject", "artifact"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quant", help="artifact: quantized checkpoint")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--layer", type=int, default=1, help="inject/artifact: residual stream AFTER this layer")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--pos", default="2-7", help="template positions (inclusive) to perturb / analyse")
    ap.add_argument("--tag", default="inj")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out")
    ap.add_argument("--tokens", default="",
                    help="inject: comma list of 'template','newline' -> perturb every such token, prefill and decode")
    ap.add_argument("--no-pos", action="store_true", help="inject: do not perturb the prompt positions --pos")
    ap.add_argument("--continuation", type=int, default=0,
                    help="artifact: extend prompts by N greedy fp16 tokens; report error by token class, prompt vs response")
    args = ap.parse_args()
    {"measure": mode_measure, "inject": mode_inject, "artifact": mode_artifact}[args.mode](args)


if __name__ == "__main__":
    main()
