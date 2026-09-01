"""General-capability evals for the capability-contrast experiment: does
protection rescue knowledge (MMLU) while failing to rescue IF (our IFEval
result)? Plus wikitext-2 perplexity (the proxy metric the outlier-protection
literature optimizes).

No lm-eval dependency (keeps the pinned stack intact). MMLU protocol =
standard 5-shot, harness-style prompt, answer scored by comparing the model's
next-token logits over " A"/" B"/" C"/" D" (single forward per question,
left padding). PPL protocol = wikitext-2-raw test, non-overlapping 2048
windows, token-weighted mean NLL.

  python src/eval_general.py --model <hf-id-or-ckpt-dir> --tag v2_tacq \
      --scores-csv runs/general_v2_tacq.csv [--mmlu-limit N]
"""
import argparse
import csv
import os

import torch

from common import load_model

CHOICES = ["A", "B", "C", "D"]


def fmt_subject(s):
    return s.replace("_", " ")


def fmt_example(row, with_answer=True):
    s = row["question"].strip() + "\n"
    for letter, choice in zip(CHOICES, row["choices"]):
        s += f"{letter}. {choice}\n"
    s += "Answer:"
    if with_answer:
        s += f" {CHOICES[row['answer']]}\n\n"
    return s


@torch.no_grad()
def eval_mmlu(model, tok, batch_size=8, limit=None):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all")
    dev, test = ds["dev"], ds["test"]
    few = {}
    for row in dev:
        few.setdefault(row["subject"], []).append(row)

    letter_ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0]
                  for c in CHOICES]
    rows = list(test)
    if limit:
        rows = rows[:limit]

    prompts, answers = [], []
    for row in rows:
        head = (f"The following are multiple choice questions (with answers) "
                f"about {fmt_subject(row['subject'])}.\n\n")
        shots = "".join(fmt_example(r) for r in few[row["subject"]][:5])
        prompts.append(head + shots + fmt_example(row, with_answer=False))
        answers.append(row["answer"])

    correct = 0
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True
                  ).to(model.device)
        logits = model(**enc).logits[:, -1, :]           # left padding
        pred = logits[:, letter_ids].argmax(dim=-1)
        correct += sum(int(p == a) for p, a in
                       zip(pred.tolist(), answers[i:i + batch_size]))
        if (i // batch_size) % 50 == 0:
            done = min(i + batch_size, len(prompts))
            print(f"[mmlu] {done}/{len(prompts)} acc={correct / done:.4f}", flush=True)
    return correct / len(prompts)


@torch.no_grad()
def eval_ppl(model, tok, seqlen=2048):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt")["input_ids"][0]
    nll, n_tok = 0.0, 0
    for i in range(0, (ids.numel() // seqlen) * seqlen, seqlen):
        chunk = ids[i:i + seqlen].unsqueeze(0).to(model.device)
        out = model(chunk, labels=chunk)
        nll += out.loss.float().item() * (chunk.numel() - 1)
        n_tok += chunk.numel() - 1
    return float(torch.exp(torch.tensor(nll / n_tok)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--mmlu-limit", type=int, help="debug: cap #questions")
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--scores-csv", required=True)
    args = ap.parse_args()

    model, tok = load_model(args.model)
    acc = eval_mmlu(model, tok, args.batch, args.mmlu_limit)
    print(f"[mmlu] {args.tag}: {acc:.4f}")
    ppl = None
    if not args.skip_ppl:
        ppl = eval_ppl(model, tok)
        print(f"[ppl] {args.tag}: {ppl:.3f}")

    os.makedirs(os.path.dirname(args.scores_csv) or ".", exist_ok=True)
    new = not os.path.exists(args.scores_csv)
    with open(args.scores_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "mmlu_acc", "wikitext2_ppl"])
        w.writerow([args.tag, round(acc, 4), round(ppl, 3) if ppl else ""])
    print(f"[general] appended -> {args.scores_csv}")


if __name__ == "__main__":
    main()
