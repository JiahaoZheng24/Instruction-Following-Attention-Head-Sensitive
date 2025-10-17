
import argparse, json, os
import pandas as pd

def compute_four_avg(res):
    r = res["results"]["ifeval"]
    vals = [
        r["prompt_level_strict_acc,none"],
        r["inst_level_strict_acc,none"],
        r["prompt_level_loose_acc,none"],
        r["inst_level_loose_acc,none"],
    ]
    return sum(vals) / 4.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--tag", default=None, help="model tag in the output csv, e.g., fp16/gptq4")
    args = ap.parse_args()

    res = json.load(open(args.results_json, "r", encoding="utf-8"))
    avg4 = compute_four_avg(res)
    model_name = res.get("configs", {}).get("ifeval", {}).get("metadata", {}).get("pretrained") or "unknown"
    tag = args.tag or model_name

    row = {
        "tag": tag,
        "model_name": model_name,
        "prompt_level_strict": res["results"]["ifeval"]["prompt_level_strict_acc,none"],
        "inst_level_strict": res["results"]["ifeval"]["inst_level_strict_acc,none"],
        "prompt_level_loose": res["results"]["ifeval"]["prompt_level_loose_acc,none"],
        "inst_level_loose": res["results"]["ifeval"]["inst_level_loose_acc,none"],
        "avg4": avg4,
    }
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    if os.path.exists(args.out_csv):
        df = pd.read_csv(args.out_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(args.out_csv, index=False)
    print(df.tail())

if __name__ == "__main__":
    main()
