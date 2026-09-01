"""Score a responses.jsonl with the official IFEval evaluator (vendored in
third_party/) and append the four standard metrics to runs/scores.csv.

Example:
  python src/score_ifeval.py --responses runs/Qwen2.5-7B-Instruct/baseline/responses.jsonl \
      --tag baseline
"""
import argparse
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def acc(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    prompt_acc = sum(r["follow_all_instructions"] for r in rows) / len(rows)
    flags = [b for r in rows for b in r["follow_instruction_list"]]
    return prompt_acc, sum(flags) / len(flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--input-data", default=os.path.join(ROOT, "data", "ifeval_input_data.jsonl"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scores-csv", default=os.path.join(ROOT, "runs", "scores.csv"))
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.responses)), "eval")
    os.makedirs(out_dir, exist_ok=True)
    r = subprocess.run(
        [sys.executable, "-m", "instruction_following_eval.evaluation_main",
         f"--input_data={os.path.abspath(args.input_data)}",
         f"--input_response_data={os.path.abspath(args.responses)}",
         f"--output_dir={out_dir}"],
        cwd=os.path.join(ROOT, "third_party"),
        env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "third_party")},
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"evaluator failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")

    ps, is_ = acc(os.path.join(out_dir, "eval_results_strict.jsonl"))
    pl, il = acc(os.path.join(out_dir, "eval_results_loose.jsonl"))
    row = {"tag": args.tag, "prompt_strict": round(ps, 4), "inst_strict": round(is_, 4),
           "prompt_loose": round(pl, 4), "inst_loose": round(il, 4),
           "avg4": round((ps + is_ + pl + il) / 4, 4)}
    os.makedirs(os.path.dirname(args.scores_csv), exist_ok=True)
    exists = os.path.exists(args.scores_csv)
    with open(args.scores_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
