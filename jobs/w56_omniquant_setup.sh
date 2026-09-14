#!/bin/bash
# W56 OPTIONAL: OmniQuant (ICLR'24; learnable clipping + equivalent transform,
# block-wise token-averaged objective, NO OBS compensation). Prediction: no
# collapse. The official repo pins old transformers, so it gets its own env.
# Try this on a login node; if it does not install within ~1 h, drop it and
# the paper states OmniQuant as an inference. Run:
#   bash jobs/w56_omniquant_setup.sh            # install + smoke test
# then, on a GPU node (interactive qrsh or a copy of w55's header):
#   conda activate omniquant && cd $STORE/OmniQuant && \
#   python main.py --model meta-llama/Llama-3.1-8B-Instruct --epochs 20 --output_dir ./log/llama31-8b-w3a16g128 \
#     --wbits 3 --abits 16 --group_size 128 --lwc --net llama-7b --save_dir $STORE/models/llama3.1-8b-omni3 --nsamples 128
#   (then run_ifeval "$STORE/models/llama3.1-8b-omni3" m_l_omni3 as in w56_methods.sh)
set -e
STORE="/store01/yshi4/jzheng7"
source ~/.bashrc 2>/dev/null || true
cd "$STORE"
[ -d OmniQuant ] || git clone https://github.com/OpenGVLab/OmniQuant.git
cd OmniQuant
if ! conda env list | grep -qE "^omniquant\s"; then
  conda create -y -n omniquant python=3.10
fi
conda activate omniquant
pip install -q torch --index-url https://download.pytorch.org/whl/cu121
pip install -q -r requirements.txt 2>/dev/null || pip install -q transformers==4.36.2 accelerate datasets sentencepiece protobuf
python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "transformers", transformers.__version__)
from transformers import AutoConfig
AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")   # rope_scaling 'llama3' needs transformers >= 4.43
print("Llama-3.1 config loads: OK")
EOF
echo "[omniquant] installed. Llama-3.1 needs transformers>=4.43 while the repo targets 4.36; if main.py"
echo "            fails on rope_scaling, this method is out (write as inference)."
