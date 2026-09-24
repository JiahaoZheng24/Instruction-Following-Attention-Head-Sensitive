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
    W_dp = model.model.layers[L].mlp.down_proj.weight.data.float().cpu()   # W76: for the row readout
    del model
    gc.collect()
    torch.cuda.empty_cache()
    qm, _ = load_model(args.quant)
    h_q = hidden_after(qm, tok, texts, [L], args.max_len)[L]
    delta_dp = W_dp - qm.model.layers[L].mlp.down_proj.weight.data.float().cpu()   # W - Q of the lesion matrix
    whiten = torch.load(args.whiten, map_location="cpu") if args.whiten else None
    if whiten is not None:
        assert int(whiten["layer"]) == L, f"--whiten file is for layer {whiten['layer']}, artifact runs layer {L}"
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
        # W76 (E1): the same against the WHITENED overlap |v^T x_t|, v = M^{-1} x_0 from the calibration
        # Hessian (Corollary 2 of the collaborator's draft; needs --whiten), and against the ROW READOUT
        # |delta_r^T x_t| of the lesion row r (the error the quantised matrix itself writes into the sink
        # output channel at this token); plus the share of ||e_t||^2 that sits in channel r.
        r_top = int(delta_dp.pow(2).sum(1).argmax())
        if whiten is not None and "toprow_C" in whiten:
            r_top = int(whiten["toprow_C"])
        d_r = delta_dp[r_top]
        v_w = whiten["v"].float() if whiten is not None else None
        stats = {"iso": [], "white": [], "row": []}
        err, err_rshare, kinds, parts, tokstr = [], [], [], [], []
        for a, b, x, t, pl in zip(h_fp, h_q, x_dp, texts, prompt_len):
            ids = tok(t, return_tensors="pt", truncation=True, max_length=args.max_len)["input_ids"][0]
            if b.shape[0] != a.shape[0] or x.shape[0] != a.shape[0] or a.shape[0] != ids.shape[0]:
                continue
            x0 = x[0]
            for p in range(1, a.shape[0]):
                e = b[p] - a[p]
                stats["iso"].append(float((x0 @ x[p]).abs() / x0.norm()))
                stats["white"].append(float((v_w @ x[p]).abs() / v_w.norm()) if v_w is not None else float("nan"))
                stats["row"].append(float((d_r @ x[p]).abs()))
                err.append(float(e.norm()))
                err_rshare.append(float(e[r_top] ** 2 / e.norm().clamp(min=1e-9) ** 2))
                kinds.append("template" if int(ids[p]) in tids_tpl else ("newline" if int(ids[p]) in tids_nl else "ordinary"))
                parts.append("prompt" if p < pl else "resp")
                tokstr.append(tok.decode([int(ids[p])]).replace("\n", "\\n"))
        if len(err) > 10:
            e_t = torch.tensor(err)
            for name, ov in stats.items():
                o_t = torch.tensor(ov)
                if not torch.isfinite(o_t).all():
                    continue
                pear = float(torch.corrcoef(torch.stack([o_t, e_t]))[0, 1])
                lo_t, le_t = o_t.clamp(min=1e-9).log(), e_t.clamp(min=1e-9).log()
                pear_log = float(torch.corrcoef(torch.stack([lo_t, le_t]))[0, 1])
                rk = lambda z: torch.argsort(torch.argsort(z)).float()   # noqa: E731  Spearman
                spear = float(torch.corrcoef(torch.stack([rk(o_t), rk(e_t)]))[0, 1])
                qs = torch.quantile(o_t, torch.tensor([0.2, 0.4, 0.6, 0.8]))
                bins = torch.bucketize(o_t, qs)
                n_tpl = sum(k != "ordinary" for k in kinds)
                for q in range(5):
                    m = bins == q
                    kk = [kinds[i] for i in torch.nonzero(m).flatten().tolist()]
                    n_here = sum(k != "ordinary" for k in kk)
                    frac_tpl = n_here / max(len(kk), 1)          # share of this quintile that is template/newline
                    recall_tpl = n_here / max(n_tpl, 1)          # share of ALL template/newline tokens in this quintile
                    rows.append({"class": f"{name}_q{q + 1}(n={int(m.sum())},tpl_frac={frac_tpl:.2f},tpl_recall={recall_tpl:.2f})",
                                 "cos_with_bos_dir": float("nan"), "rel_err": float(e_t[m].mean()),
                                 "energy_share_on_bos_dir": float(o_t[m].mean())})
                rows.append({"class": f"{name}_pearson linear={pear:.3f} log={pear_log:.3f} spearman={spear:.3f} n={len(err)}",
                             "cos_with_bos_dir": pear, "rel_err": pear_log, "energy_share_on_bos_dir": spear})
            # share of the hidden-state error energy in the lesion row's channel, by token class
            rs = torch.tensor(err_rshare)
            for kind in ("template", "newline", "ordinary"):
                m = torch.tensor([k == kind for k in kinds])
                if m.any():
                    rows.append({"class": f"rowshare_{kind}(row={r_top},n={int(m.sum())})", "cos_with_bos_dir": float("nan"),
                                 "rel_err": float(e_t[m].mean()), "energy_share_on_bos_dir": float(rs[m].mean())})
            if args.out_tokens:
                os.makedirs(os.path.dirname(args.out_tokens) or ".", exist_ok=True)
                with open(args.out_tokens, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["part", "kind", "token", "ov_iso", "ov_white", "ov_row", "err", "err_rowshare"])
                    for i in range(len(err)):
                        w.writerow([parts[i], kinds[i], tokstr[i], f"{stats['iso'][i]:.5g}", f"{stats['white'][i]:.5g}",
                                    f"{stats['row'][i]:.5g}", f"{err[i]:.5g}", f"{err_rshare[i]:.4f}"])
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


# -------------------------------------------------------------- patch (W77)
class Patcher:
    """Activation patching. During prefill (T > 1) the residual stream after `layer`
    at the selected positions of each prompt is replaced by the full-precision
    model's values; decode steps are untouched. Batches are left-padded, so the
    stored positions are offset by the first non-pad index of each row."""

    def __init__(self, model, layer, store):
        self.store = store              # prompt index -> (positions LongTensor, vectors [n, d] fp32)
        self.batch_ids = None
        self.mask = None
        self.n_patched = 0
        self.h_top = model.register_forward_pre_hook(self._catch, with_kwargs=True)
        self.h_layer = model.model.layers[layer].register_forward_hook(self._patch)

    def _catch(self, _m, args, kw):
        self.mask = kw.get("attention_mask")

    def _patch(self, _m, _a, out):
        h = out[0] if isinstance(out, tuple) else out
        B, T, _ = h.shape
        if T == 1 or self.batch_ids is None:
            return out
        h = h.clone()
        for b, idx in enumerate(self.batch_ids):
            pos, vec = self.store[idx]
            if self.mask is not None and self.mask.shape[0] == B and self.mask.shape[1] == T:
                nz = torch.nonzero(self.mask[b], as_tuple=False)
                start = int(nz[0]) if nz.numel() else 0
            else:
                start = 0
            p = pos.cpu() + start
            ok = p < T
            sel = p[ok].tolist()
            if not sel:
                continue
            h[b, sel] = vec[ok].to(device=h.device, dtype=h.dtype)
            self.n_patched += len(sel)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def remove(self):
        self.h_top.remove()
        self.h_layer.remove()


@torch.no_grad()
def mode_patch(args):
    """--model = quantised checkpoint, --ref = full-precision model; --select which prompt positions
    to restore: template (special + newline tokens), ordinary (same count, seeded, non-template),
    bos (position 0), all (whole prompt), or lo-hi."""
    ref, tok = load_model(args.ref)
    prompts = read_jsonl(args.prompts)
    texts = [chat(tok, ex["prompt"]) for ex in prompts]
    tids = token_set(tok, "template,newline")
    gen = torch.Generator().manual_seed(args.seed)
    store, counts = {}, []
    for i, t in enumerate(texts):
        ex_prompt = prompts[i]["prompt"]
        ids = tok(t, return_tensors="pt", truncation=True, max_length=2048).to(ref.device)
        T = ids["input_ids"].shape[1]
        is_tpl = torch.tensor([int(x) in tids for x in ids["input_ids"][0].tolist()])
        is_tpl[0] = False                                   # BOS handled separately
        # W80: the whole chat scaffold = every prompt token outside the user's own text (special tokens,
        # newlines, role names and the template's default system block), located by character offsets.
        enc_off = tok(t, return_offsets_mapping=True, truncation=True, max_length=2048)
        c0 = t.find(ex_prompt)
        c1 = c0 + len(ex_prompt)
        is_scaf = torch.tensor([not (c0 <= a_ and b_ <= c1) for a_, b_ in enc_off["offset_mapping"]])
        if is_scaf.shape[0] != T:
            is_scaf = torch.zeros(T, dtype=torch.bool)
        is_scaf[0] = False
        if args.select == "template":
            pos = torch.nonzero(is_tpl).flatten()
        elif args.select == "ordinary":
            cand = torch.nonzero(~is_tpl).flatten()
            cand = cand[cand > 0]
            n = int(is_tpl.sum())
            pos = cand[torch.randperm(len(cand), generator=gen)[:n]].sort().values
        elif args.select == "scaffold":
            pos = torch.nonzero(is_scaf).flatten()
        elif args.select == "content":          # same count as the scaffold, drawn from the user's text
            cand = torch.nonzero(~is_scaf).flatten()
            cand = cand[cand > 0]
            n = int(is_scaf.sum())
            pos = cand[torch.randperm(len(cand), generator=gen)[:n]].sort().values
        elif args.select == "bos":
            pos = torch.tensor([0])
        elif args.select == "all":
            pos = torch.arange(T)
        else:
            lo, hi = parse_pos(args.select)
            pos = torch.arange(lo, min(hi + 1, T))
        o = ref(**ids, output_hidden_states=True, use_cache=False)
        h = o.hidden_states[args.layer + 1][0].float().cpu()
        store[i] = (pos, h[pos].clone())
        counts.append(len(pos))
        del o
    print(f"[patch] {args.select}: {sum(counts) / len(counts):.1f} positions per prompt restored to fp16 "
          f"(layer {args.layer} output, prefill only)", flush=True)
    del ref
    gc.collect()
    torch.cuda.empty_cache()
    qm, _ = load_model(args.model)
    pt = Patcher(qm, args.layer, store)
    out_rows = []
    for i in range(0, len(prompts), args.batch):
        batch = prompts[i:i + args.batch]
        pt.batch_ids = list(range(i, i + len(batch)))
        enc = tok([chat(tok, ex["prompt"]) for ex in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(qm.device)
        g = qm.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id)
        for ex, seq in zip(batch, g):
            out_rows.append({"prompt": ex["prompt"],
                             "response": tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)})
        print(f"[patch:{args.tag}] {min(i + args.batch, len(prompts))}/{len(prompts)}", flush=True)
    pt.remove()
    run_dir = os.path.join("runs", os.path.basename(args.model), args.tag)
    os.makedirs(run_dir, exist_ok=True)
    write_jsonl(os.path.join(run_dir, "responses.jsonl"), out_rows)
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(f"model={args.model}\nref={args.ref}\npatch layer={args.layer} select={args.select} "
                f"mean_positions={sum(counts) / len(counts):.2f} n_patched={pt.n_patched}\n")
    print(f"[patch] {len(out_rows)} responses -> {run_dir}/responses.jsonl (patched {pt.n_patched} positions)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["measure", "inject", "artifact", "patch"])
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
    ap.add_argument("--whiten", help="artifact (W76): file from src/theory_tests.py --save-whiten (x0, v, lesion row)")
    ap.add_argument("--ref", help="patch (W77): full-precision model whose hidden states are restored")
    ap.add_argument("--select", default="template",
                    help="patch (W77/W80): template | ordinary | bos | all | scaffold | content | lo-hi (positions of the prompt to restore)")
    ap.add_argument("--seed", type=int, default=0, help="patch: seed for the ordinary-position control")
    ap.add_argument("--out-tokens", help="artifact (W76): per-token csv (overlap statistics, error, row share)")
    args = ap.parse_args()
    {"measure": mode_measure, "inject": mode_inject, "artifact": mode_artifact, "patch": mode_patch}[args.mode](args)


if __name__ == "__main__":
    main()
