#!/usr/bin/env bash
# Block E2/E3 server launcher (claude_blockE23). Adapted from the proven
# claude_blockB_results/run_blockB.sh recipe: vllm/vllm-openai:latest-cu130,
# editable source install (python-only change), HF cache mount, transformers
# pin, --gpu-memory-utilization 0.72, --rm + memory caps.
#
# Usage: run_blockE23_server.sh <run_id> [extra_env]
#   run_blockE23_server.sh claude_blockE3a            # BEFORE row (default)
#   run_blockE23_server.sh claude_blockE2 VLLM_FLASHINFER_VOSPLIT=1   # AFTER
set -euo pipefail
RUN_ID=$1
EXTRA_ENV=${2:-}

VLLM_SRC=/home/jethac/spark_tmp/vllm-022-ad2337814-clone
HF_CACHE=/home/jethac/.cache/huggingface
RESULTS_DIR=/home/jethac/spark_tmp/claude_blockE23_results
IMAGE=vllm/vllm-openai:latest-cu130
# merge-base of ad2337814 with vllm-project/main (precompiled wheel exists,
# cu130 index verified)
COMMIT=4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa
TRANSFORMERS_SPEC="git+https://github.com/huggingface/transformers.git@effde20942e3f82a1b97449f60b3a48c5ff96145"
MODEL=google/gemma-4-E4B-it

mkdir -p "${RESULTS_DIR}"

ENV_PREFIX=""
if [ -n "${EXTRA_ENV}" ]; then
  ENV_PREFIX="export ${EXTRA_ENV};"
fi

docker run -d --rm --gpus all --ipc=host --memory 100g --memory-swap 100g \
  --name "${RUN_ID}" \
  -p 8000:8000 \
  -v "${VLLM_SRC}:/vllm-src" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${RESULTS_DIR}:/work" \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "
set -euo pipefail
apt-get update -qq >/work/${RUN_ID}_apt_update.log 2>&1
apt-get install -y -qq git >/work/${RUN_ID}_apt_git.log 2>&1
git config --global --add safe.directory /vllm-src
python3 -m pip install -q setuptools-rust >/work/${RUN_ID}_pip_bootstrap.log 2>&1
cd /vllm-src
VLLM_USE_PRECOMPILED=1 VLLM_MAIN_CUDA_VERSION=13.0 VLLM_PRECOMPILED_WHEEL_COMMIT=${COMMIT} \
  python3 -m pip install --no-build-isolation -e . >/work/${RUN_ID}_editable_install.log 2>&1
python3 -m pip uninstall -y flashinfer-jit-cache >/work/${RUN_ID}_flashinfer_jit_cache_uninstall.log 2>&1 || true
find /usr/local/lib/python3.12/dist-packages -maxdepth 1 \
  \( -name flashinfer_jit_cache -o -name flashinfer_jit_cache-*.dist-info \) \
  -print -exec rm -rf {} + >>/work/${RUN_ID}_flashinfer_jit_cache_uninstall.log 2>&1 || true
python3 -m pip install \"${TRANSFORMERS_SPEC}\" >/work/${RUN_ID}_transformers_install.log 2>&1
${ENV_PREFIX}
python3 - <<PY >/work/${RUN_ID}_import_probe.txt 2>&1
import json, os
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
    \"has_gemma4_cond_gen\": \"Gemma4ForConditionalGeneration\" in str(registry.__dict__),
    \"VLLM_FLASHINFER_VOSPLIT\": os.environ.get(\"VLLM_FLASHINFER_VOSPLIT\"),
}, sort_keys=True))
PY
exec vllm serve ${MODEL} \
  --served-model-name gemma4-e4b-it \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.72 \
  --max-num-batched-tokens 4096 \
  --host 0.0.0.0 \
  --port 8000 \
  >/work/${RUN_ID}_server.log 2>&1
"
