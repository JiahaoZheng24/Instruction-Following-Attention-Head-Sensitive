
import argparse, json, re, sys, os
from tqdm import tqdm

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            yield json.loads(line)

def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--min_len", type=int, default=30)
    ap.add_argument("--max_len", type=int, default=1200)
    ap.add_argument("--keywords", nargs="*", default=None,
                    help="If set, override default keyword list.")
    args = ap.parse_args()

    # default keywords (can be overridden)
    default_keywords = [
        "用", "two words", "开头", "start with", "JSON", "格式", "only output", "no explanation", "regex", "正则"
    ]
    keywords = set(args.keywords if args.keywords else default_keywords)

    selected = []
    for ex in load_jsonl(args.samples_jsonl):
        p = ex.get("prompt") or ex.get("text") or ""
        if not p or not (args.min_len <= len(p) <= args.max_len):
            continue
        low = p.lower()
        if any(k in low for k in keywords):
            selected.append({"key": ex.get("key"), "prompt": p})
        if len(selected) >= args.top_k:
            break

    # Fallback：如果一个都没选到，就直接拿前 top_k 条
    if not selected:
        print("[WARN] No prompts matched keywords. Falling back to first top_k samples.")
        cnt = 0
        for ex in load_jsonl(args.samples_jsonl):
            p = ex.get("prompt") or ex.get("text") or ""
            if not p: continue
            selected.append({"key": ex.get("key"), "prompt": p})
            cnt += 1
            if cnt >= args.top_k:
                break

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    save_jsonl(args.out_jsonl, selected)
    print(f"Selected {len(selected)} prompts -> {args.out_jsonl}")

if __name__ == "__main__":
    main()
