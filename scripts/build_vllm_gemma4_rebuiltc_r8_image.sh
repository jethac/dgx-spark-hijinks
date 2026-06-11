#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/build_vllm_gemma4_rebuiltc_r8_image.sh IMAGE_TAG

Build the Gemma 4 rebuilt-C vLLM image from a clean local jethac/vllm
checkout. This is intentionally stricter than the earlier r7 builder: the
vLLM source is copied from VLLM_SRC only after checking that it is clean and
at EXPECTED_VLLM_REF.

Environment:
  VLLM_SRC=/home/jethac/spark_tmp/vllm-022-ad2337814-clone
  EXPECTED_VLLM_REF=e08a6f3ae7557d87553f1892d2ecc822f2187957
  BASE_IMAGE=ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2
  FLASHINFER_REPO=https://github.com/jethac/flashinfer.git
  FLASHINFER_REF=fb7d62ea45f19cb61f19057a93519c17b6e257f3
  IMAGE_GENERATION=r8
  MAX_JOBS=3
  NVCC_THREADS=1
  RESULTS_DIR=/home/jethac/dgx-spark-hijinks/results
  RUN_ID=vllm_gemma4_rebuiltc_image_build_YYYYMMDDTHHMMJST_r8
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

IMAGE_TAG=$1
VLLM_SRC=${VLLM_SRC:-/home/jethac/spark_tmp/vllm-022-ad2337814-clone}
EXPECTED_VLLM_REF=${EXPECTED_VLLM_REF:-e08a6f3ae7557d87553f1892d2ecc822f2187957}
BASE_IMAGE=${BASE_IMAGE:-ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2}
FLASHINFER_REPO=${FLASHINFER_REPO:-https://github.com/jethac/flashinfer.git}
FLASHINFER_REF=${FLASHINFER_REF:-fb7d62ea45f19cb61f19057a93519c17b6e257f3}
IMAGE_GENERATION=${IMAGE_GENERATION:-r8}
MAX_JOBS=${MAX_JOBS:-3}
NVCC_THREADS=${NVCC_THREADS:-1}
RUN_ID=${RUN_ID:-vllm_gemma4_rebuiltc_image_build_$(TZ=Asia/Tokyo date +%Y%m%dT%H%MJST)_${IMAGE_GENERATION}}
RESULTS_DIR=${RESULTS_DIR:-/home/jethac/dgx-spark-hijinks/results}
LOG_PATH=${RESULTS_DIR}/${RUN_ID}.log
SUMMARY_PATH=${RESULTS_DIR}/${RUN_ID}_summary.md

DOCKERFILE=$(mktemp)
BUILD_CONTEXT=$(mktemp -d)
trap 'rm -f "${DOCKERFILE}"; rm -rf "${BUILD_CONTEXT}"' EXIT
mkdir -p "${RESULTS_DIR}"

test -d "${VLLM_SRC}/vllm"
actual_ref=$(git -C "${VLLM_SRC}" rev-parse HEAD)
actual_short_ref=${actual_ref:0:9}
if [[ "${actual_ref}" != "${EXPECTED_VLLM_REF}" ]]; then
  echo "wrong vLLM ref: got ${actual_ref}, expected ${EXPECTED_VLLM_REF}" >&2
  exit 1
fi
if [[ -n "$(git -C "${VLLM_SRC}" status --porcelain=v1)" ]]; then
  echo "vLLM source tree is dirty; refusing to build" >&2
  git -C "${VLLM_SRC}" status -sb >&2
  exit 1
fi

mkdir -p "${BUILD_CONTEXT}/vllm-src"
tar -C "${VLLM_SRC}" --exclude=.git -cf - . | tar -C "${BUILD_CONTEXT}/vllm-src" -xf -
printf '%s\n' "${actual_ref}" > "${BUILD_CONTEXT}/vllm-src/.spark_build_source_rev"

cat >"${DOCKERFILE}" <<'EOF'
ARG BASE_IMAGE=ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2
FROM ${BASE_IMAGE}

ARG BASE_IMAGE=ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2
ARG VLLM_REF=e08a6f3ae7557d87553f1892d2ecc822f2187957
ARG VLLM_SHORT_REF=e08a6f3ae
ARG FLASHINFER_REPO=https://github.com/jethac/flashinfer.git
ARG FLASHINFER_REF=fb7d62ea45f19cb61f19057a93519c17b6e257f3
ARG IMAGE_GENERATION=r8
ARG MAX_JOBS=3
ARG NVCC_THREADS=1

LABEL org.opencontainers.image.source="https://github.com/jethac/vllm" \
      org.opencontainers.image.revision="${VLLM_REF}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      spark.vllm_rebuilt_c="true" \
      spark.flashinfer_ref="${FLASHINFER_REF}" \
      spark.cuda_arch="sm_121a" \
      spark.image_generation="${IMAGE_GENERATION}"

RUN test -x /usr/local/cuda/bin/nvcc
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates cuda-nvrtc-dev-13-0 libcusparse-dev-13-0 \
      libcublas-dev-13-0 libcusolver-dev-13-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
      'cmake>=3.26.1' ninja 'packaging>=24.2' \
      'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8.0' \
      'setuptools-rust>=1.9.0' wheel jinja2 pyelftools \
      'apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2' \
      'nvidia-cutlass-dsl[cu13]>=4.5.2'

RUN python3 -m pip install --no-cache-dir --no-deps \
      'humming-kernels[cu13]==0.1.4'

COPY vllm-src/ /opt/jethac-vllm/
RUN test "$(cat /opt/jethac-vllm/.spark_build_source_rev)" = "${VLLM_REF}"

WORKDIR /opt/jethac-vllm

ENV VLLM_MAIN_CUDA_VERSION=13.0 \
    VLLM_VERSION_OVERRIDE=0.1.dev1+g${VLLM_SHORT_REF} \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.dev1+g${VLLM_SHORT_REF} \
    VLLM_TARGET_DEVICE=cuda \
    CMAKE_CUDA_ARCHITECTURES=121a-real \
    TORCH_CUDA_ARCH_LIST=12.1a \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}

RUN python3 -m pip uninstall -y vllm || true
RUN unset VLLM_USE_PRECOMPILED VLLM_PRECOMPILED_SKIP_FLASH_ATTN VLLM_PRECOMPILED_WHEEL_COMMIT VLLM_PRECOMPILED_WHEEL_LOCATION \
 && python3 -m pip install --no-cache-dir --no-build-isolation --no-deps -e . -v

RUN git clone "${FLASHINFER_REPO}" /opt/jethac-flashinfer \
 && cd /opt/jethac-flashinfer \
 && git checkout "${FLASHINFER_REF}" \
 && git submodule update --init 3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog

RUN python3 -m pip uninstall -y flashinfer-python flashinfer-jit-cache flashinfer-cubin || true \
 && { test ! -d /usr/local/lib/python3.12/dist-packages || find /usr/local/lib/python3.12/dist-packages -maxdepth 1  \
      \( -name 'flashinfer' -o -name 'flashinfer-*.dist-info' \
         -o -name 'flashinfer_python-*.dist-info' \
         -o -name 'flashinfer_jit_cache' -o -name 'flashinfer_jit_cache-*.dist-info' \
         -o -name 'flashinfer_cubin' -o -name 'flashinfer_cubin-*.dist-info' \) \
      -print -exec rm -rf {} +; }  \
 && cd /opt/jethac-flashinfer \
 && python3 -m pip install --no-cache-dir --no-build-isolation --no-deps -e . -v

RUN python3 - <<'PY'
from pathlib import Path
import importlib.metadata as md
import torch
import vllm
import flashinfer
import humming

pkg = Path(vllm.__file__).resolve().parent
print("vllm", getattr(vllm, "__version__", None), vllm.__file__)
print("flashinfer", getattr(flashinfer, "__version__", None), flashinfer.__file__)
print("torch", torch.__version__, torch.version.cuda)
print("humming", getattr(humming, "__file__", None), md.version("humming-kernels"))
print("arch_list", torch.cuda.get_arch_list() if torch.cuda.is_available() else [])
for rel in [
    "_C.abi3.so",
    "_C_stable_libtorch.abi3.so",
    "_moe_C.abi3.so",
    "vllm_flash_attn/_vllm_fa2_C.abi3.so",
]:
    path = pkg / rel
    print(rel, path, "exists=", path.exists(), "size=", path.stat().st_size if path.exists() else 0)
    assert path.exists(), f"missing {rel}"
PY

RUN VLLM_C="$(python3 -c 'from pathlib import Path; import vllm; print(Path(vllm.__file__).resolve().parent / "_C.abi3.so")')" \
 && /usr/local/cuda/bin/cuobjdump -lelf "$VLLM_C" | tee /tmp/vllm_C_cuobjdump.txt \
 && grep -E 'sm_121|sm_120|compute_121|compute_120' /tmp/vllm_C_cuobjdump.txt || true
EOF

{
  echo "# vLLM Gemma4 Rebuilt-C Image Build ${IMAGE_GENERATION}"
  echo
  echo "- image: \`${IMAGE_TAG}\`"
  echo "- base image: \`${BASE_IMAGE}\`"
  echo "- vLLM source: \`${VLLM_SRC}\`"
  echo "- vLLM ref: \`${actual_ref}\`"
  echo "- FlashInfer ref: \`${FLASHINFER_REF}\`"
  echo "- image generation: \`${IMAGE_GENERATION}\`"
  echo "- MAX_JOBS: \`${MAX_JOBS}\`"
  echo "- NVCC_THREADS: \`${NVCC_THREADS}\`"
  echo "- log: \`${LOG_PATH}\`"
  echo "- started JST: \`$(TZ=Asia/Tokyo date --iso-8601=seconds)\`"
  echo
  echo "Status: running"
} >"${SUMMARY_PATH}"

if docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg VLLM_REF="${actual_ref}" \
  --build-arg VLLM_SHORT_REF="${actual_short_ref}" \
  --build-arg FLASHINFER_REPO="${FLASHINFER_REPO}" \
  --build-arg FLASHINFER_REF="${FLASHINFER_REF}" \
  --build-arg IMAGE_GENERATION="${IMAGE_GENERATION}" \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  --build-arg NVCC_THREADS="${NVCC_THREADS}" \
  -t "${IMAGE_TAG}" \
  -f "${DOCKERFILE}" "${BUILD_CONTEXT}" 2>&1 | tee "${LOG_PATH}"; then
  IMAGE_ID=$(docker image inspect --format '{{.Id}}' "${IMAGE_TAG}")
  IMAGE_SIZE=$(docker image inspect --format '{{.Size}}' "${IMAGE_TAG}")
  STATUS=built
else
  STATUS=failed
  IMAGE_ID=none
  IMAGE_SIZE=0
fi

{
  echo "# vLLM Gemma4 Rebuilt-C Image Build ${IMAGE_GENERATION}"
  echo
  echo "- image: \`${IMAGE_TAG}\`"
  echo "- image id: \`${IMAGE_ID}\`"
  echo "- image size bytes: \`${IMAGE_SIZE}\`"
  echo "- base image: \`${BASE_IMAGE}\`"
  echo "- vLLM source: \`${VLLM_SRC}\`"
  echo "- vLLM ref: \`${actual_ref}\`"
  echo "- FlashInfer ref: \`${FLASHINFER_REF}\`"
  echo "- image generation: \`${IMAGE_GENERATION}\`"
  echo "- MAX_JOBS: \`${MAX_JOBS}\`"
  echo "- NVCC_THREADS: \`${NVCC_THREADS}\`"
  echo "- log: \`${LOG_PATH}\`"
  echo "- finished JST: \`$(TZ=Asia/Tokyo date --iso-8601=seconds)\`"
  echo
  echo "Status: ${STATUS}"
} >"${SUMMARY_PATH}"

echo "wrote ${LOG_PATH}"
echo "wrote ${SUMMARY_PATH}"
test "${STATUS}" = built
