"""GSM8K (test, 1319 problems) for instruct models: zero-shot CoT via chat
template, greedy decoding, answer = last number after '####' or, failing
that, the last number in the response.

Purpose (W11): capability-generality evidence for the regime law. IFEval /
Multi-IF (instruction following), MMLU (knowledge) and PPL already replicate
the regime picture; GSM8K adds reasoning — same checkpoints, no new
quantization.

  python src/gsm8k_eval.py --model <ckpt-or-id> --tag gsm_llama_tacq \
      --scores-csv runs/scores_gsm8k.csv
"""
import argparse
import csv
import os
import re

import torch

from common import load_model, write_jsonl

PROMPT = ("{question}\n\nSolve step by step, then give the final numeric "
          "answer on the last line in the form '#### <answer>'.")
NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract(text: str):
    m = re.search(r"####\s*(-?\$?[\d,]*\.?\d+)", text)
    cand = m.group(1) if m else (NUM.findall(text)[-1] if NUM.findall(text) else None)
    if cand is None:
        return None
    try:
        return float(cand.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, help="debug cap")
    ap.add_argument("--scores-csv", required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = list(ds)[: args.limit] if args.limit else list(ds)
    print(f"[gsm8k] {len(rows)} problems")

    model, tok = load_model(args.model)
    records, correct, parsed = [], 0, 0
    with torch.no_grad():
        for b in range(0, len(rows), args.batch):
            chunk = rows[b:b + args.batch]
            texts = [tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(question=r["question"])}],
                tokenize=False, add_generation_prompt=True) for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(model.device)
            gen = model.generate(**enc, do_sample=False, max_new_tokens=512,
                                 pad_token_id=tok.pad_token_id)
            for r, seq in zip(chunk, gen):
                resp = tok.decode(seq[enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
                gold = float(r["answer"].split("####")[-1].strip()
                             .replace(",", ""))
                pred = extract(resp)
                ok = pred is not None and abs(pred - gold) < 1e-4
                correct += int(ok)
                parsed += int(pred is not None)
                records.append({"question": r["question"], "gold": gold,
                                "pred": pred, "ok": ok, "response": resp})
            print(f"[gsm8k:{args.tag}] {min(b + args.batch, len(rows))}/{len(rows)} "
                  f"acc so far={correct / max(len(records), 1):.4f}", flush=True)

    run_dir = os.path.join("runs", os.path.basename(args.model.rstrip("/")),
                           f"gsm8k_{args.tag}")
    os.makedirs(run_dir, exist_ok=True)
    write_jsonl(os.path.join(run_dir, "responses.jsonl"), records)

    row = {"tag": args.tag, "n": len(records),
           "accuracy": round(correct / max(len(records), 1), 4),
           "parse_rate": round(parsed / max(len(records), 1), 4)}
    new = not os.path.exists(args.scores_csv)
    with open(args.scores_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[gsm8k] {row}")


if __name__ == "__main__":
    main()
