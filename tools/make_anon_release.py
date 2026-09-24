"""Build the anonymized code-and-data release for the ICLR submission.

  python tools/make_anon_release.py            -> ICLR/anon_release/  and  ICLR/anon_release.zip

What goes in (everything the paper's reproducibility statement promises):
  src/                       quantization loop (gptq_core.py), quantizer front end (quantize_protected.py), the
                             GPTQModel end-to-end scripts, evaluation (IFEval, Multi-IF, GSM8K, MMLU/PPL),
                             transplants, divergence, mechanism statistics, gradient weights, theory tests, index builder
  jobs/                      every SGE job script; their header comments are the batch-by-batch record, including
                             the predictions written before each batch ran (collected into PREREGISTRATION.md)
  data/                      IFEval prompts and the chat-formatted calibration prompts
  results/                   INDEX_scores.csv (one row per evaluated arm), INDEX_stats.csv, INDEX_theory.csv,
                             sides/, sink/, divergence/, stats/*/stats.csv (send-mass tensors left out; 700 MB)
                             (per-tag score files and per-checkpoint protocol dumps are summarised by the indexes)
  ICLR/figure/code/          figure generators;  ICLR/ICLR_quantization/tables/make_*.py  table generators
What stays out: checkpoints, HF cache, logs, archives, notes, slides, drafts, anything with names or paths.
Every text file is scrubbed of user names, cluster paths, e-mail addresses and institution names; the script
fails if a hit survives, so add patterns below rather than releasing a file that names someone.
"""
import glob
import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'ICLR', 'anon_release')
ZIP = OUT + '.zip'

SCRUB = [  # (pattern, replacement); order matters
    (r'/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive', '$REPO'),
    (r'/store01/yshi4/jzheng7', '$STORE'),
    (r'/users/jzheng7/anaconda3', '$CONDA'),
    (r'jzheng7@nd\.edu', 'author@example.org'),
    (r'jzheng7', 'user'),
    (r'yshi4', 'lab'),
    (r'zzheng3_Lab', 'lab_queue'),
    (r'gpu@@lab_queue', 'gpu_queue'),
    (r'crc\.nd\.edu', 'cluster.example.org'),
    (r'vaststore01', 'store'),
    (r'Notre Dame', 'the institution'),
    (r'Jiahao|Zheng|jiahao', 'author'),
    (r'Instruction-Following-Attention-Head-Sensitive', 'repo'),   # the GitHub repository name is searchable
    (r'/store01', '$STORE'),
    (r'JiahaoZheng24', 'account'),
]
FORBIDDEN = re.compile(r'jzheng7|yshi4|nd\.edu|Jiahao|Zheng|Notre Dame|zzheng3|Roy\b|Instruction-Following-Attention|store01|/users/|C:\\\\Users', re.I)

INCLUDE = [
    ('src/*.py', 'src'),
    ('jobs/*.sh', 'jobs'),
    ('data/*.jsonl', 'data'),
    ('runs/INDEX_scores.csv', 'results'),
    ('runs/INDEX_stats.csv', 'results'),
    ('runs/theory/INDEX_theory.csv', 'results'),
    ('runs/sides/*.csv', 'results/sides'),
    ('runs/div_*.csv', 'results/divergence'),
    ('runs/sink/*.csv', 'results/sink'),
    ('ICLR/figure/code/*.py', 'figure_code'),
    ('ICLR/figure/code/README.md', 'figure_code'),
    ('ICLR/ICLR_quantization/tables/make_*.py', 'table_code'),
    ('requirements*.txt', '.'),
    ('environment*.yml', '.'),
]
TEXT_EXT = {'.py', '.sh', '.md', '.csv', '.json', '.jsonl', '.txt', '.yml', '.yaml', '.env', '.example'}


def scrub(text):
    for pat, rep in SCRUB:
        text = re.sub(pat, rep, text)
    return text


def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ext = os.path.splitext(src)[1].lower()
    if ext in TEXT_EXT:
        raw = open(src, encoding='utf-8', errors='replace').read()
        if ext == '.sh':
            raw = '\n'.join(ln for ln in raw.split('\n') if not re.match(r'#\$ -(M|m)\b', ln))
        clean = scrub(raw)
        hit = FORBIDDEN.search(clean)
        if hit:
            raise SystemExit(f'identifying string survives in {src}: {hit.group(0)!r} - add a pattern to SCRUB')
        open(dst, 'w', encoding='utf-8', newline='\n').write(clean)
    else:
        shutil.copy2(src, dst)


def prereg_log(job_dir):
    """The comment header of every job script, in file order: what each batch did and, where written, what it predicted."""
    parts = ['# Pre-registration log\n\nThe header comment of every job script, in the order the batches were written. '
             'From the diagnosis onward (W7x and later) each header states the prediction before the batch ran; '
             'the refuted ones are collected in Appendix C of the paper.\n']
    for f in sorted(glob.glob(os.path.join(job_dir, '*.sh')), key=lambda p: os.path.basename(p)):
        lines = open(f, encoding='utf-8').read().split('\n')
        head = []
        for ln in lines[1:]:
            if ln.startswith('#$'):
                continue
            if ln.startswith('#'):
                head.append(ln[1:].strip())
            elif ln.strip() == '':
                continue
            else:
                break
        if head:
            parts.append(f'\n## {os.path.basename(f)}\n\n' + '\n'.join(head) + '\n')
    return '\n'.join(parts)


README = """# One Token Owns the Calibration Hessian - code and run index

Anonymous release for review. Everything here was produced by the authors; the run index maps every number in
the paper to a tag, and the job scripts record how each tag was produced.

## Layout
- `src/` - the quantization loop (`gptq_core.py`: GPTQ/OBS with act-order, natural or group-aware column order,
  token-normalized and gradient-weighted Hessians, mechanism statistics), the quantizer front end
  (`quantize_protected.py`), GPTQModel end-to-end runs (`quantize_gptqmodel_default.py`, `quantize_gptqmodel_ablate.py`),
  evaluation (`diagnose_heads.py ablate` + `score_ifeval.py` for IFEval, `multi_if.py`, `gsm8k_eval.py`, `eval_general.py`),
  transplants (`transplant.py`), per-position divergence (`divergence.py`), sink injection and activation patching
  (`inject_sink.py`, `zero_probe.py`), gradient weights (`grad_weights.py`), theory statistics (`theory_tests.py`),
  and the index builder (`build_index.py`).
- `jobs/` - one SGE array script per batch (W0-W92); `PREREGISTRATION.md` collects their headers.
- `results/INDEX_scores.csv` - one row per evaluated arm (tag, IFEval components, MMLU, PPL, GSM8K); the tag names
  the job script and task that produced it. `results/INDEX_stats.csv` - one row per statistics directory;
  `results/stats/<dir>/stats.csv` - per-matrix mechanism statistics; `results/INDEX_theory.csv` - the Theorem 1
  quantities per model; `results/sides/` - A_BOS / G_BOS per model; `results/sink/`, `results/divergence/` - the
  per-position and per-token-class errors behind Figures 1 and 4.
- `figure_code/`, `table_code/` - generators of the paper's figures and tables from these files.

## Reproducing one collapse, its excision and its repair (one GPU, Llama-3.1-8B-Instruct, 3 bits)
```
export HF_HOME=...; export STORE=...        # see jobs/_w2x_header.sh
python src/quantize_protected.py --model meta-llama/Llama-3.1-8B-Instruct --bits 3 --group-size 128 \\
    --protect none --calib c4 --out $STORE/models/llama_gptq3            # collapse: IFEval 0.155
python src/quantize_protected.py ... --rtn --out $STORE/models/llama_rtn3                        # RTN 0.565
python src/quantize_protected.py ... --rtn-modules model.layers.1.mlp.down_proj --out .../llama_excised   # 0.622
python src/quantize_protected.py ... --hess-grad-weight --hess-token-norm --salience-dir <g_t dir> \\
    --out .../llama_corrected                                                                     # 0.638
python src/diagnose_heads.py ablate --model <ckpt> --prompts data/ifeval_input_data.jsonl --tag <tag> --batch 16
python src/score_ifeval.py --responses runs/<ckpt>/<tag>/responses.jsonl --input-data data/ifeval_input_data.jsonl --tag <tag>
```
`jobs/w72a_fixed_core.sh` runs every main-text arm; `jobs/w90_library_default.sh`, `w91_library_ablation.sh` and
`w92_gar_order.sh` are the GPTQModel end-to-end and column-order runs of Appendix B.

Environment: Python 3.10, torch 2.9, transformers 4.57, gptqmodel 5.6.12 (TORCH backend), datasets; one H200 per task.
"""


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    n = 0
    for pattern, dest in INCLUDE:
        for src in sorted(glob.glob(os.path.join(ROOT, pattern))):
            if os.path.isdir(src):
                continue
            dst = os.path.join(OUT, dest, os.path.basename(src))
            copy_file(src, dst)
            n += 1
    for st in sorted(glob.glob(os.path.join(ROOT, 'runs', 'stats', '*', ''))):
        name = os.path.basename(st.rstrip('/\\'))
        for fn in ('stats.csv', 'STATS_PROTOCOL.json'):
            f = os.path.join(st, fn)
            if os.path.exists(f):
                copy_file(f, os.path.join(OUT, 'results', 'stats', name, fn))
                n += 1
    open(os.path.join(OUT, 'PREREGISTRATION.md'), 'w', encoding='utf-8', newline='\n').write(scrub(prereg_log(os.path.join(OUT, 'jobs'))))
    open(os.path.join(OUT, 'README.md'), 'w', encoding='utf-8', newline='\n').write(README)
    # final sweep over everything written
    for f in glob.glob(os.path.join(OUT, '**', '*'), recursive=True):
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in TEXT_EXT:
            hit = FORBIDDEN.search(open(f, encoding='utf-8', errors='replace').read())
            if hit:
                raise SystemExit(f'identifying string in release: {f}: {hit.group(0)!r}')
    if os.path.exists(ZIP):
        os.remove(ZIP)
    with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(os.path.join(OUT, '**', '*'), recursive=True):
            if os.path.isfile(f):
                z.write(f, os.path.relpath(f, OUT))
    print(f'{n} files + stats dirs -> {OUT}  ({os.path.getsize(ZIP) / 1e6:.1f} MB zipped)')


if __name__ == '__main__':
    sys.exit(main())
