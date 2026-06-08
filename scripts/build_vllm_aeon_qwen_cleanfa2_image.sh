#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/build_vllm_aeon_qwen_cleanfa2_image.sh IMAGE_TAG

Build a derived jethac/vLLM Qwen3.6 image from AEON's known-good base while
avoiding AEON's restored FA2 binary. The build installs the jethac vLLM fork
with precompiled non-FA extensions, skips bundled FA2/FA3 extraction, then
builds the FA2 component against the container's actual Torch/CUDA ABI.

Environment:
  BASE_IMAGE=ghcr.io/aeon-7/vllm-spark-omni-q36:v2
  VLLM_REF=a919d635d
  VLLM_VERSION_OVERRIDE=0.1.dev1+g${VLLM_REF}
  VLLM_PRECOMPILED_WHEEL_COMMIT=4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa
  CMAKE_CUDA_ARCHITECTURES=121a-real
  REQUIRE_NATIVE_FA2_ARCH=1
  MAX_JOBS=4
  NVCC_THREADS=1
  RESULTS_DIR=results
  RUN_ID=vllm_cleanfa2_build_YYYYMMDDTHHMMSS
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

IMAGE_TAG=$1
BASE_IMAGE=${BASE_IMAGE:-ghcr.io/aeon-7/vllm-spark-omni-q36:v2}
VLLM_REF=${VLLM_REF:-a919d635d}
VLLM_VERSION_OVERRIDE=${VLLM_VERSION_OVERRIDE:-0.1.dev1+g${VLLM_REF}}
VLLM_PRECOMPILED_WHEEL_COMMIT=${VLLM_PRECOMPILED_WHEEL_COMMIT:-4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa}
CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES:-121a-real}
REQUIRE_NATIVE_FA2_ARCH=${REQUIRE_NATIVE_FA2_ARCH:-1}
MAX_JOBS=${MAX_JOBS:-4}
NVCC_THREADS=${NVCC_THREADS:-1}
RESULTS_DIR=${RESULTS_DIR:-results}
RUN_ID=${RUN_ID:-vllm_cleanfa2_build_$(date -u +%Y%m%dT%H%M%SZ)}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VLLM_ROOT="${REPO_ROOT}/third_party/vllm"
VLLM_FLASH_ATTN_ROOT="${REPO_ROOT}/third_party/vllm-flash-attention"
LOG_PATH="${REPO_ROOT}/${RESULTS_DIR}/${RUN_ID}.log"
mkdir -p "${REPO_ROOT}/${RESULTS_DIR}"
DOCKERFILE=$(mktemp)
BUILD_CONTEXT=$(mktemp -d)
trap 'rm -f "${DOCKERFILE}"; rm -rf "${BUILD_CONTEXT}"' EXIT

test -d "${VLLM_ROOT}/vllm"
test -d "${VLLM_FLASH_ATTN_ROOT}/csrc/flash_attn"
mkdir -p "${BUILD_CONTEXT}/vllm-src" "${BUILD_CONTEXT}/vllm-flash-attn-src"
tar -C "${VLLM_ROOT}" --exclude=.git -cf - . | tar -C "${BUILD_CONTEXT}/vllm-src" -xf -
tar -C "${VLLM_FLASH_ATTN_ROOT}" --exclude=.git -cf - . | tar -C "${BUILD_CONTEXT}/vllm-flash-attn-src" -xf -

cat >"${DOCKERFILE}" <<'EOF'
ARG BASE_IMAGE=ghcr.io/aeon-7/vllm-spark-omni-q36:v2
FROM ${BASE_IMAGE}

ARG BASE_IMAGE=ghcr.io/aeon-7/vllm-spark-omni-q36:v2
ARG VLLM_REF=a919d635d
ARG VLLM_VERSION_OVERRIDE=0.1.dev1+ga919d635d
ARG VLLM_PRECOMPILED_WHEEL_COMMIT=4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa
ARG CMAKE_CUDA_ARCHITECTURES=121a-real
ARG REQUIRE_NATIVE_FA2_ARCH=1
ARG MAX_JOBS=4
ARG NVCC_THREADS=1

LABEL org.opencontainers.image.source="https://github.com/jethac/vllm" \
      org.opencontainers.image.revision="${VLLM_REF}" \
      aeon.base="${BASE_IMAGE}" \
      spark.clean_fa2="true"

RUN python3 -m pip install --no-cache-dir --no-deps \
      'compressed-tensors==0.17.0' \
      pyelftools \
      'humming-kernels[cu13]==0.1.4'

RUN python3 -m pip install --no-cache-dir \
      'cmake>=3.26.1' ninja 'packaging>=24.2' \
      'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8.0' \
      'setuptools-rust>=1.9.0' wheel jinja2

COPY vllm-src/ /opt/jethac-vllm/
COPY vllm-flash-attn-src/ /opt/jethac-vllm-flash-attn/
WORKDIR /opt/jethac-vllm

ENV VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1 \
    VLLM_MAIN_CUDA_VERSION=13.0 \
    VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE}" \
    VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT}" \
    VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX=1 \
    VLLM_FLASH_ATTN_SRC_DIR=/opt/jethac-vllm-flash-attn \
    REQUIRE_NATIVE_FA2_ARCH="${REQUIRE_NATIVE_FA2_ARCH}" \
    MAX_JOBS="${MAX_JOBS}" \
    NVCC_THREADS="${NVCC_THREADS}"

RUN python3 -m pip install --no-cache-dir --no-build-isolation --no-deps -e . -v

RUN test ! -e /opt/jethac-vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so

RUN test -x /usr/local/cuda/bin/nvcc

RUN cmake -S . -B /tmp/vllm-fa2-build -G Ninja \
      -DVLLM_PYTHON_EXECUTABLE="$(command -v python3)" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DVLLM_TARGET_DEVICE=cuda \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
      -DFETCHCONTENT_BASE_DIR=/opt/jethac-vllm/.deps \
      2>&1 | tee /tmp/vllm-fa2-configure.log \
 && if [ "${REQUIRE_NATIVE_FA2_ARCH}" = "1" ]; then \
      grep -Eq 'FA2_ARCHS:.*12\.1a?|FA2_ARCHS:.*121a?' /tmp/vllm-fa2-configure.log || \
        (echo "FA2 configure did not select a native SM121/SM121a arch; refusing to build a non-native FA2 binary." >&2; exit 1); \
    fi \
 && cmake --build /tmp/vllm-fa2-build --target _vllm_fa2_C -j "${MAX_JOBS}" \
 && cmake --install /tmp/vllm-fa2-build \
      --prefix /tmp/vllm-fa2-install \
      --component _vllm_fa2_C \
 && test -s /tmp/vllm-fa2-install/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so \
 && cp /tmp/vllm-fa2-install/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so \
      /opt/jethac-vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so

RUN python3 - <<'PY'
import torch
import vllm
import vllm.vllm_flash_attn._vllm_fa2_C as fa2_ext

print("vllm", vllm.__version__, vllm.__file__)
print("torch", torch.__version__, torch.version.cuda)
print("arch_list", torch.cuda.get_arch_list() if torch.cuda.is_available() else [])
print("fa2", fa2_ext.__file__)
PY
EOF

docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg VLLM_REF="${VLLM_REF}" \
  --build-arg VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE}" \
  --build-arg VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT}" \
  --build-arg CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
  --build-arg REQUIRE_NATIVE_FA2_ARCH="${REQUIRE_NATIVE_FA2_ARCH}" \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  --build-arg NVCC_THREADS="${NVCC_THREADS}" \
  -t "${IMAGE_TAG}" \
  -f "${DOCKERFILE}" "${BUILD_CONTEXT}" 2>&1 | tee "${LOG_PATH}"

echo "wrote ${LOG_PATH}"
