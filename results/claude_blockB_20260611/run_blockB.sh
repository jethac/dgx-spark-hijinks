#!/usr/bin/env bash
# Adapted from scripts/run_vllm_gemma4_12b_unified_probe.sh for the shared-box
# protocol: --rm + memory caps + gpu-memory-utilization 0.72 (was 0.80),
# container name claude_blockB. Everything else follows the script.
set -euo pipefail
VLLM_SRC=/home/jethac/spark-validation/vllm-gemma4-unified-da1daf4-clone
HF_CACHE=/home/jethac/.cache/huggingface
RESULTS_DIR=/home/jethac/spark_tmp/claude_blockB_results
RUN_ID=claude_blockB
IMAGE=vllm/vllm-openai:latest-cu130
COMMIT=da1daf40bf18e5eaae04f26a80a537c8168a8bc2
TRANSFORMERS_SPEC="git+https://github.com/huggingface/transformers.git@effde20942e3f82a1b97449f60b3a48c5ff96145"

mkdir -p "${RESULTS_DIR}"

docker run -d --rm --gpus all --ipc=host --memory 100g --memory-swap 100g \
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
python3 -m pip uninstall -y flashinfer-jit-cache >/results/${RUN_ID}_flashinfer_jit_cache_uninstall.log 2>&1 || true
find /usr/local/lib/python3.12/dist-packages -maxdepth 1 \
  \( -name flashinfer_jit_cache -o -name flashinfer_jit_cache-*.dist-info \) \
  -print -exec rm -rf {} + >>/results/${RUN_ID}_flashinfer_jit_cache_uninstall.log 2>&1 || true
python3 -m pip install \"${TRANSFORMERS_SPEC}\" >/results/${RUN_ID}_transformers_install.log 2>&1
python3 - <<PY >/results/${RUN_ID}_import_probe.txt 2>&1
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
  --gpu-memory-utilization 0.72 \
  --max-num-batched-tokens 4096 \
  --host 0.0.0.0 \
  --port 8000
"
