#!/usr/bin/env bash
# E3 vLLM/FlashInfer setup for native sm_120 Vast boxes.
#
# Required env:
#   VLLM_WHEEL_URL  GitHub release asset URL for the E3 sm120a vLLM wheel.
# Optional env:
#   FLASHINFER_REF  jethac/flashinfer ref; defaults to the shared E3 ref.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

: "${VLLM_WHEEL_URL:?set VLLM_WHEEL_URL to the E3 sm120a wheel release asset}"
FLASHINFER_REF="${FLASHINFER_REF:-1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a}"
WHEEL="/root/$(basename "${VLLM_WHEEL_URL}" | sed 's/%2B/+/g')"

echo "=== apt ==="
apt-get update -q >/dev/null 2>&1
apt-get install -y -q python3.12-venv python3.12-dev build-essential git wget ca-certificates >/dev/null 2>&1

echo "=== venv + torch ==="
python3 -m venv /root/v
/root/v/bin/pip install -q -U pip >/dev/null 2>&1
/root/v/bin/pip install -q torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -1
/root/v/bin/pip install -q torchvision --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -1
/root/v/bin/pip install -q ninja transformers pyarrow accelerate huggingface_hub 2>&1 | tail -1

echo "=== download + install vLLM wheel ==="
wget -q -O "${WHEEL}" "${VLLM_WHEEL_URL}"
ls -la "${WHEEL}" | awk '{print "wheel size:",$5}'
/root/v/bin/pip install -q "${WHEEL}" 2>&1 | tail -2

echo "=== flashinfer source ==="
cd /root
rm -rf /root/flashinfer
git clone -q https://github.com/jethac/flashinfer
git -C flashinfer fetch -q origin "${FLASHINFER_REF}" || git -C flashinfer fetch -q --all
git -C flashinfer checkout -q "${FLASHINFER_REF}"
git -C flashinfer submodule update --init --recursive --depth 1 >/dev/null 2>&1
mkdir -p flashinfer/flashinfer/data
ln -sfn ../../csrc flashinfer/flashinfer/data/csrc
ln -sfn ../../include flashinfer/flashinfer/data/include
ln -sfn ../../3rdparty/cutlass flashinfer/flashinfer/data/cutlass
ln -sfn ../../3rdparty/cccl flashinfer/flashinfer/data/cccl
ln -sfn ../../3rdparty/spdlog flashinfer/flashinfer/data/spdlog

echo "=== verify ==="
PYTHONPATH=/root/flashinfer /root/v/bin/python - <<'PY'
import importlib.metadata
import torch
import vllm
import flashinfer

print("OK vllm", vllm.__version__)
print("OK torch", torch.__version__, torch.version.cuda)
print("OK flashinfer", getattr(flashinfer, "__file__", None))
print("vllm_dist", importlib.metadata.version("vllm"))
PY
echo DONE_E3_SETUP
