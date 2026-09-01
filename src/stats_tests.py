"""Paired prompt-level bootstrap for IFEval contrasts (stdlib only).

Both arms scored the same 541 prompts -> resample prompt indices with
replacement (paired), recompute the avg4 estimator on each resample for both
arms, report the diff CI and a sign-flip p-value. avg4 = mean(prompt_strict,
inst_strict, prompt_loose, inst_loose), with inst_* aggregated over
instructions exactly as in score_ifeval.

  python src/stats_tests.py --a runs/<ckpt>/<tagA> --b runs/<ckpt>/<tagB> \
      [--n 10000] [--label "tacq vs none"]
"""
import argparse
import json
import os
import random


def load_arm(run_dir):
    rows = {}
    for kind in ("strict", "loose"):
        p = os.path.join(run_dir, "eval", f"eval_results_{kind}.jsonl")
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.setdefault(r["prompt"], {})[kind] = (
                    bool(r["follow_all_instructions"]),
                    [bool(x) for x in r["follow_instruction_list"]])
    return rows


def avg4(sample):
    ps = pi = ls = li = 0
    n_p = n_i = 0
    for s in sample:
        (sa, sf), (la, lf) = s["strict"], s["loose"]
        ps += sa
        ls += la
        pi += sum(sf)
        li += sum(lf)
        n_p += 1
        n_i += len(sf)
    return (ps / n_p + pi / n_i + ls / n_p + li / n_i) / 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run dir of arm A")
    ap.add_argument("--b", required=True, help="run dir of arm B (reference)")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    A, B = load_arm(args.a), load_arm(args.b)
    keys = sorted(set(A) & set(B))
    assert len(keys) > 500, f"only {len(keys)} shared prompts"
    pa = [A[k] for k in keys]
    pb = [B[k] for k in keys]
    point = avg4(pa) - avg4(pb)

    rng = random.Random(0)
    diffs = []
    n = len(keys)
    for _ in range(args.n):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(avg4([pa[i] for i in idx]) - avg4([pb[i] for i in idx]))
    diffs.sort()
    lo, hi = diffs[int(0.025 * args.n)], diffs[int(0.975 * args.n)]
    p_boot = 2 * min(sum(d <= 0 for d in diffs), sum(d >= 0 for d in diffs)) / args.n
    print(f"{args.label or (os.path.basename(args.a) + ' vs ' + os.path.basename(args.b))}: "
          f"diff={point * 100:+.2f}pt  95%CI=[{lo * 100:+.2f}, {hi * 100:+.2f}]  "
          f"p={min(p_boot, 1.0):.4f}  (n={n} prompts)")


if __name__ == "__main__":
    main()
