#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/build_vllm_qwen_nvfp4kv_image.sh IMAGE_TAG

Build a derived Qwen vLLM image for the NVFP4-KV serving proof. This starts
from the clean jethac Qwen image that already contains the vLLM SM12x NVFP4-KV
routing patch and native SM121a FA2 binary, then replaces FlashInfer with the
jethac FA2 NVFP4-KV scale-stride patch.

Environment:
  BASE_IMAGE=jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass
  FLASHINFER_REPO=https://github.com/jethac/flashinfer.git
  FLASHINFER_REF=e152cf4da4ab2a9d093b7d9d4b499198b0211c61
  RESULTS_DIR=results
  RUN_ID=vllm_qwen_nvfp4kv_image_build_YYYYMMDDTHHMMSS
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

IMAGE_TAG=$1
BASE_IMAGE=${BASE_IMAGE:-jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass}
FLASHINFER_REPO=${FLASHINFER_REPO:-https://github.com/jethac/flashinfer.git}
FLASHINFER_REF=${FLASHINFER_REF:-e152cf4da4ab2a9d093b7d9d4b499198b0211c61}
RESULTS_DIR=${RESULTS_DIR:-results}
RUN_ID=${RUN_ID:-vllm_qwen_nvfp4kv_image_build_$(date -u +%Y%m%dT%H%M%SZ)}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_PATH="${REPO_ROOT}/${RESULTS_DIR}/${RUN_ID}.log"
mkdir -p "${REPO_ROOT}/${RESULTS_DIR}"
DOCKERFILE=$(mktemp)
BUILD_CONTEXT=$(mktemp -d)
trap 'rm -f "${DOCKERFILE}"; rm -rf "${BUILD_CONTEXT}"' EXIT

cat >"${DOCKERFILE}" <<'EOF'
ARG BASE_IMAGE=jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass
FROM ${BASE_IMAGE}

ARG FLASHINFER_REPO=https://github.com/jethac/flashinfer.git
ARG FLASHINFER_REF=e152cf4da4ab2a9d093b7d9d4b499198b0211c61

LABEL org.opencontainers.image.source="https://github.com/jethac/flashinfer" \
      org.opencontainers.image.revision="${FLASHINFER_REF}" \
      spark.nvfp4_kv_flashinfer="true"

RUN python3 -m pip install --no-cache-dir \
      'apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2' \
      'setuptools>=77' 'packaging>=24'

RUN git clone "${FLASHINFER_REPO}" /opt/jethac-flashinfer \
 && cd /opt/jethac-flashinfer \
 && git checkout "${FLASHINFER_REF}" \
 && git submodule update --init 3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog

RUN python3 -m pip uninstall -y flashinfer-python flashinfer-jit-cache flashinfer-cubin || true \
 && find /usr/local/lib/python3.12/dist-packages -maxdepth 1 \
      \( -name 'flashinfer' -o -name 'flashinfer-*.dist-info' \
         -o -name 'flashinfer_python-*.dist-info' \
         -o -name 'flashinfer_jit_cache' -o -name 'flashinfer_jit_cache-*.dist-info' \
         -o -name 'flashinfer_cubin' -o -name 'flashinfer_cubin-*.dist-info' \) \
      -print -exec rm -rf {} +

RUN cd /opt/jethac-flashinfer \
 && python3 -m pip install --no-cache-dir --no-build-isolation --no-deps -e . -v

RUN python3 - <<'PY'
from pathlib import Path
import flashinfer
import vllm

root = Path(flashinfer.__file__).resolve().parent
source_hits = []
for path in root.rglob("*.py"):
    text = path.read_text(errors="ignore")
    if "maybe_k_cache_sf_stride_page" in text or "FLASHINFER_PAGED_V_SF_DESWIZZLE" in text:
        source_hits.append(str(path))

backend = Path(vllm.__file__).resolve().parent / "v1/attention/backends/flashinfer.py"
backend_text = backend.read_text(errors="ignore")

print("flashinfer", getattr(flashinfer, "__version__", None), flashinfer.__file__)
print("vllm", getattr(vllm, "__version__", None), vllm.__file__)
print("flashinfer_source_hits", source_hits[:10])
assert source_hits, "patched FlashInfer stride/deswizzle source markers were not found"
assert "use_fa2_nvfp4_kv" in backend_text, "vLLM NVFP4-KV FA2 routing marker missing"
assert "FLASHINFER_PAGED_V_SF_DESWIZZLE" in backend_text, "vLLM deswizzle flag marker missing"
PY
EOF

docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg FLASHINFER_REPO="${FLASHINFER_REPO}" \
  --build-arg FLASHINFER_REF="${FLASHINFER_REF}" \
  -t "${IMAGE_TAG}" \
  -f "${DOCKERFILE}" "${BUILD_CONTEXT}" 2>&1 | tee "${LOG_PATH}"

echo "wrote ${LOG_PATH}"
