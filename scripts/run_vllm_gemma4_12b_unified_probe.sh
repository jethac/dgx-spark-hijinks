#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 VLLM_SRC HF_CACHE RESULTS_DIR RUN_ID [IMAGE]" >&2
  exit 2
fi

VLLM_SRC=$1
HF_CACHE=$2
RESULTS_DIR=$3
RUN_ID=$4
IMAGE=${5:-vllm/vllm-openai:latest-cu130}
COMMIT=${VLLM_PRECOMPILED_WHEEL_COMMIT:-da1daf40bf18e5eaae04f26a80a537c8168a8bc2}
TRANSFORMERS_SPEC=${TRANSFORMERS_SPEC:-}
REMOVE_FLASHINFER_JIT_CACHE=${REMOVE_FLASHINFER_JIT_CACHE:-1}

mkdir -p "${RESULTS_DIR}"
docker rm -f "${RUN_ID}" >/dev/null 2>&1 || true

docker run -d --gpus all --ipc=host \
  --name "${RUN_ID}" \
  -p 8000:8000 \
  -v "${VLLM_SRC}:/vllm-src" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${RESULTS_DIR}:/results" \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "
set -euo pipefail
apt-get update -qq >/results/${RUN_ID}_apt_update.log 2>&1
apt-get install -y -qq git >/results/${RUN_ID}_apt_git.log 2>&1
git config --global --add safe.directory /vllm-src
python3 -m pip install -q setuptools-rust >/results/${RUN_ID}_pip_bootstrap.log 2>&1
cd /vllm-src
VLLM_USE_PRECOMPILED=1 VLLM_MAIN_CUDA_VERSION=13.0 VLLM_PRECOMPILED_WHEEL_COMMIT=${COMMIT} \
  python3 -m pip install --no-build-isolation -e . >/results/${RUN_ID}_editable_install.log 2>&1
if [[ \"${REMOVE_FLASHINFER_JIT_CACHE}\" == \"1\" ]]; then
  python3 -m pip uninstall -y flashinfer-jit-cache >/results/${RUN_ID}_flashinfer_jit_cache_uninstall.log 2>&1 || true
  find /usr/local/lib/python3.12/dist-packages -maxdepth 1 \
    \( -name 'flashinfer_jit_cache' -o -name 'flashinfer_jit_cache-*.dist-info' \) \
    -print -exec rm -rf {} + >>/results/${RUN_ID}_flashinfer_jit_cache_uninstall.log 2>&1 || true
fi
if [[ -n \"${TRANSFORMERS_SPEC}\" ]]; then
  python3 -m pip install \"${TRANSFORMERS_SPEC}\" >/results/${RUN_ID}_transformers_install.log 2>&1
fi
python3 - <<'PY' >/results/${RUN_ID}_import_probe.txt 2>&1
import json
import torch
import transformers
import vllm
import vllm.model_executor.models.registry as registry

print(json.dumps({
    \"vllm_file\": vllm.__file__,
    \"vllm_version\": vllm.__version__,
    \"torch\": torch.__version__,
    \"cuda\": torch.version.cuda,
    \"compute_capability\": torch.cuda.get_device_capability(0),
    \"gpu\": torch.cuda.get_device_name(0),
    \"transformers\": transformers.__version__,
    \"has_gemma4_unified\": \"Gemma4UnifiedForConditionalGeneration\" in str(registry.__dict__),
}, sort_keys=True))
PY
exec vllm serve google/gemma-4-12B-it \
  --served-model-name gemma4-12b-it \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 4096 \
  --host 0.0.0.0 \
  --port 8000
"
