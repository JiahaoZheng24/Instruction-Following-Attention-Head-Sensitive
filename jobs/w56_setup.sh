#!/bin/bash
# W56 one-time setup: AutoRound and HQQ live in a CLONE of the IFEval env so
# their pip dependencies cannot touch the evaluation environment. Run once on
# a login node (no GPU needed), before qsub jobs/w56_methods.sh:
#   bash jobs/w56_setup.sh
set -e
source ~/.bashrc 2>/dev/null || true
if ! conda env list | grep -qE "^IFEval_ar\s"; then
  conda create -y -n IFEval_ar --clone IFEval
fi
conda activate IFEval_ar
pip install -q auto-round hqq
python - <<'EOF'
import auto_round, hqq, torch, transformers
print("auto-round", auto_round.__version__, "| hqq", getattr(hqq, "__version__", "?"),
      "| torch", torch.__version__, "| transformers", transformers.__version__)
EOF
echo "[w56_setup] ok — now: qsub jobs/w56_methods.sh"
