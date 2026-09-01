"""Multi-IF (facebook/Multi-IF) runner: multi-turn instruction following.

English subset, up to 3 turns. Each turn: append user prompt to the running
conversation, generate, then check THAT turn's instruction list against THAT
turn's response with the official IFEval instruction registry (vendored,
same taxonomy). Reports per-turn prompt-strict and instruction-strict
accuracy. Instructions whose id is missing from the vendored registry are
skipped and counted (logged) — Multi-IF extends IFEval slightly.

  python src/multi_if.py --model <ckpt-or-id> --tag mif_v2l_tacq \
      --scores-csv runs/scores_mif_v2l_tacq.csv
"""
import argparse
import csv
import json
import os
import sys

import torch

from common import load_model, write_jsonl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party"))
from instruction_following_eval import instructions_registry  # noqa: E402

MAX_NEW_TOKENS = 1024


def parse_maybe_json(x):
    if x is None:
        return None
    if isinstance(x, (list, dict)):
        return x
    try:
        return json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return None


def check_turn(prompt, response, iids, kwargs_list, skipped):
    """Strict per-instruction check via the official registry."""
    flags = []
    for iid, kw in zip(iids, kwargs_list):
        if isinstance(kw, str):
            kw = parse_maybe_json(kw) or {}
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        cls = instructions_registry.INSTRUCTION_DICT.get(iid)
        if cls is None:
            skipped[iid] = skipped.get(iid, 0) + 1
            continue
        try:
            instr = cls(iid)
            instr.build_description(**kw)
            args = instr.get_instruction_args()
            if args and "prompt" in args:
                instr.build_description(prompt=prompt)
            flags.append(bool(response.strip()) and instr.check_following(response))
        except Exception as e:
            skipped[f"{iid}!err"] = skipped.get(f"{iid}!err", 0) + 1
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, help="debug cap")
    ap.add_argument("--scores-csv", required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("facebook/Multi-IF", split="train")
    rows = [r for r in ds if str(r.get("language", "")).lower() in ("english", "en")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[multi-if] {len(rows)} English examples")

    model, tok = load_model(args.model)
    convs = [[] for _ in rows]           # running chat history per example
    records = []
    stats = {t: {"p_ok": 0, "p_n": 0, "i_ok": 0, "i_n": 0} for t in (1, 2, 3)}
    skipped: dict[str, int] = {}

    with torch.no_grad():
        for t in (1, 2, 3):
            active = [i for i, r in enumerate(rows) if r.get(f"turn_{t}_prompt")]
            if not active:
                continue
            for b in range(0, len(active), args.batch):
                idxs = active[b:b + args.batch]
                texts = []
                for i in idxs:
                    convs[i].append({"role": "user", "content": rows[i][f"turn_{t}_prompt"]})
                    texts.append(tok.apply_chat_template(
                        convs[i], tokenize=False, add_generation_prompt=True))
                enc = tok(texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=4096).to(model.device)
                gen = model.generate(**enc, do_sample=False,
                                     max_new_tokens=MAX_NEW_TOKENS,
                                     pad_token_id=tok.pad_token_id)
                for i, seq in zip(idxs, gen):
                    resp = tok.decode(seq[enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
                    convs[i].append({"role": "assistant", "content": resp})
                    iids = parse_maybe_json(rows[i][f"turn_{t}_instruction_id_list"]) or []
                    kws = parse_maybe_json(rows[i][f"turn_{t}_kwargs"]) or [{}] * len(iids)
                    flags = check_turn(rows[i][f"turn_{t}_prompt"], resp, iids, kws, skipped)
                    if flags:
                        stats[t]["p_ok"] += int(all(flags))
                        stats[t]["p_n"] += 1
                        stats[t]["i_ok"] += sum(flags)
                        stats[t]["i_n"] += len(flags)
                    records.append({"key": rows[i]["key"], "turn": t,
                                    "response": resp, "follow": flags})
                done = min(b + args.batch, len(active))
                print(f"[multi-if:{args.tag}] turn {t}: {done}/{len(active)}", flush=True)

    run_dir = os.path.join("runs", os.path.basename(args.model.rstrip("/")),
                           f"multiif_{args.tag}")
    os.makedirs(run_dir, exist_ok=True)
    write_jsonl(os.path.join(run_dir, "responses.jsonl"), records)

    row = {"tag": args.tag}
    for t in (1, 2, 3):
        s = stats[t]
        row[f"t{t}_prompt_strict"] = round(s["p_ok"] / max(s["p_n"], 1), 4)
        row[f"t{t}_inst_strict"] = round(s["i_ok"] / max(s["i_n"], 1), 4)
    row["skipped_instructions"] = sum(skipped.values())
    new = not os.path.exists(args.scores_csv)
    with open(args.scores_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[multi-if] {row}")
    if skipped:
        print(f"[multi-if] skipped ids: {skipped}")


if __name__ == "__main__":
    main()
