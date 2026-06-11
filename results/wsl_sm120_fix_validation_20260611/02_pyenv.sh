#!/bin/bash
# Python venv + torch cu130. Run as default user (jetha).
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"

cd ~
if [ ! -d ~/sm120env ]; then
    python3 -m venv ~/sm120env
fi
source ~/sm120env/bin/activate
pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu130
python - <<'EOF'
import torch, json
cap = torch.cuda.get_device_capability()
info = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "capability": cap,
}
print(json.dumps(info))
assert cap == (12, 0), f"expected (12,0), got {cap}"
major = int(torch.version.cuda.split(".")[0])
assert major >= 13, f"expected cuda>=13, got {torch.version.cuda}"
print("TORCH_VERIFY_OK")
EOF
