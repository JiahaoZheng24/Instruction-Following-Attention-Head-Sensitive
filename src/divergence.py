"""Where does a quantized model leave the fp16 trajectory? Per-layer hidden
state agreement between an fp16 model and a (fake-quant) checkpoint on
chat-formatted prompts. Complements the mechanism log: the log says which
layers SEND the most compensation; this says at which layer the forward
pass actually diverges (and whether that is the same layer).

Per layer l (residual stream after block l, teacher-forced on the prompt):
  cos_mean   mean over tokens of cos(h_fp[l], h_q[l])
  cos_last   cos at the last prompt token (the one that starts generation)
  rel_err    ||h_q - h_fp|| / ||h_fp||   (token-mean)
plus top1_agree = fraction of prompt positions where argmax logits agree.

  python src/divergence.py --fp16 meta-llama/Llama-3.1-8B-Instruct \
      --quant $STORE/models/llama3.1-8b-v2gptq3-none \
      --prompts data/ifeval_input_data.jsonl --n 32 --out runs/div_l_none.csv
"""
import argparse
import csv
import gc
import json

import torch

from common import load_model


def chat(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def collect(model_id, tok_src, texts, max_len):
    model, tok = load_model(model_id)
    if tok_src is not None:
        tok = tok_src
    hs, logits = [], []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        out = model(**ids, output_hidden_states=True)
        hs.append([h[0].float().cpu() for h in out.hidden_states])   # L+1 x [T, d]
        logits.append(out.logits[0].argmax(-1).cpu())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return hs, logits, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-position", action="store_true",
                    help="also write <out>.pos.csv: rel_err at positions 0..7 and mean of the rest, "
                         "per layer (is the damage at the BOS / attention-sink position?)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8")][: args.n]
    _, tok0 = None, None
    from transformers import AutoTokenizer
    tok0 = AutoTokenizer.from_pretrained(args.fp16)
    texts = [chat(tok0, r.get("prompt") or r["text"]) for r in rows]

    ids0 = tok0(texts[0], truncation=True, max_length=args.max_len)["input_ids"]
    print(f"[divergence] prompt-0 tokens at positions 0..9: {tok0.convert_ids_to_tokens(ids0[:10])}")
    hs_fp, lg_fp, _ = collect(args.fp16, tok0, texts, args.max_len)
    hs_q, lg_q, _ = collect(args.quant, tok0, texts, args.max_len)

    n_layers = len(hs_fp[0])
    acc = [{"cos": [], "cos_last": [], "rel": []} for _ in range(n_layers)]
    agree = []
    for a, b, la, lb in zip(hs_fp, hs_q, lg_fp, lg_q):
        agree.append(float((la == lb).float().mean()))
        for l in range(n_layers):
            x, y = a[l], b[l]
            cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
            acc[l]["cos"].append(float(cos.mean()))
            acc[l]["cos_last"].append(float(cos[-1]))
            acc[l]["rel"].append(float(((x - y).norm(dim=-1) / x.norm(dim=-1).clamp(min=1e-6)).mean()))

    if args.per_position:
        P = 8
        with open(args.out.replace(".csv", "") + ".pos.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer"] + [f"rel_err_pos{i}" for i in range(P)] + ["rel_err_rest"]
                       + [f"norm_ratio_pos{i}" for i in range(P)] + ["norm_ratio_rest"]
                       + [f"fp16_norm_pos{i}" for i in range(P)] + ["fp16_norm_rest"])
            for l in range(n_layers):
                per = [[] for _ in range(P)]; nrp = [[] for _ in range(P)]; fnp = [[] for _ in range(P)]
                rest = []; nrr = []; fnr = []
                for a, b in zip(hs_fp, hs_q):
                    x, y = a[l], b[l]
                    xn = x.norm(dim=-1)
                    re = (x - y).norm(dim=-1) / xn.clamp(min=1e-6)
                    nr = y.norm(dim=-1) / xn.clamp(min=1e-6)
                    for i in range(min(P, x.shape[0])):
                        per[i].append(float(re[i])); nrp[i].append(float(nr[i])); fnp[i].append(float(xn[i]))
                    if x.shape[0] > P:
                        rest.append(float(re[P:].mean())); nrr.append(float(nr[P:].mean())); fnr.append(float(xn[P:].mean()))
                m = lambda v, d=4: round(sum(v) / len(v), d) if v else ""  # noqa: E731
                w.writerow([l] + [m(v) for v in per] + [m(rest)]
                           + [m(v, 3) for v in nrp] + [m(nrr, 3)]
                           + [m(v, 1) for v in fnp] + [m(fnr, 1)])
        print(f"[divergence] per-position -> {args.out.replace('.csv', '')}.pos.csv")
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "cos_mean", "cos_last", "rel_err", "top1_agree_prompt"])
        for l in range(n_layers):
            m = lambda k: sum(acc[l][k]) / len(acc[l][k])  # noqa: E731
            w.writerow([l, round(m("cos"), 5), round(m("cos_last"), 5), round(m("rel"), 5),
                        round(sum(agree) / len(agree), 4) if l == n_layers - 1 else ""])
    print(f"[divergence] {args.quant}: top1 agreement on prompts = {sum(agree) / len(agree):.3f}")
    print(f"[divergence] -> {args.out}")


if __name__ == "__main__":
    main()
