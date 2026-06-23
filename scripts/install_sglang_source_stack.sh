#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/work}
FLASHINFER_INSTALL_LOG=${FLASHINFER_INSTALL_LOG:-}
SGLANG_INSTALL_LOG=${SGLANG_INSTALL_LOG:-}

log_flashinfer() {
  if [[ -n "${FLASHINFER_INSTALL_LOG}" ]]; then
    "$@" >>"${FLASHINFER_INSTALL_LOG}" 2>&1
  else
    "$@"
  fi
}

log_sglang() {
  if [[ -n "${SGLANG_INSTALL_LOG}" ]]; then
    "$@" >>"${SGLANG_INSTALL_LOG}" 2>&1
  else
    "$@"
  fi
}

mkdir -p "$(dirname "${FLASHINFER_INSTALL_LOG:-/tmp/flashinfer-install.log}")"
mkdir -p "$(dirname "${SGLANG_INSTALL_LOG:-/tmp/sglang-install.log}")"

# GPU-less build containers have no libcuda.so.1 (the driver lib), so importing
# the freshly-compiled sgl_kernel/flashinfer extensions during verification
# fails to dlopen it. Expose the CUDA driver STUB (ships in -devel images) under
# the .1 soname and put it on the loader path FOR THIS PROCESS ONLY -- do NOT
# bake stubs into the image ENV, or it would shadow the real driver the
# nvidia-container-runtime injects at runtime on the GPU box.
CUDA_STUB_DIR="${CUDA_HOME:-/usr/local/cuda}/lib64/stubs"
if [[ -e "${CUDA_STUB_DIR}/libcuda.so" ]]; then
  ln -sf "${CUDA_STUB_DIR}/libcuda.so" "${CUDA_STUB_DIR}/libcuda.so.1"
  export LD_LIBRARY_PATH="${CUDA_STUB_DIR}:${LD_LIBRARY_PATH:-}"
fi

# flashinfer nested submodules (cutlass/cccl/spdlog) are required for runtime
# JIT. When this script runs against a real git clone we init them here; when it
# runs inside a Docker image whose context was rsync'd WITHOUT .git, there is no
# repo to init -- the content must already be present (pre-populated on the
# build runner before the rsync). Guard so the content-only tree does not fatal,
# and fail loudly if the content is genuinely missing.
if git -C "${REPO_ROOT}/third_party/flashinfer" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "${REPO_ROOT}/third_party/flashinfer" submodule update --init \
    3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog
else
  echo "flashinfer .git absent (content-only tree); trusting pre-populated 3rdparty submodules"
  for d in 3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog; do
    if [[ -z "$(ls -A "${REPO_ROOT}/third_party/flashinfer/${d}" 2>/dev/null)" ]]; then
      echo "FATAL: ${REPO_ROOT}/third_party/flashinfer/${d} is empty and no .git to init it" >&2
      exit 3
    fi
  done
fi

python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache \
  sglang-kernel || true
rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer \
       /usr/local/lib/python3.12/dist-packages/flashinfer_python-*.dist-info \
       /usr/local/lib/python3.12/dist-packages/flashinfer_cubin* \
       /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache* \
       /usr/local/lib/python3.12/dist-packages/sgl_kernel \
       /usr/local/lib/python3.12/dist-packages/sglang_kernel-*.dist-info \
       /root/.cache/flashinfer || true

# Build tooling: let the lightweight backends (scikit-build-core etc.) resolve
# their own deps (pathspec/packaging/pyproject-metadata) -- with
# --no-build-isolation there is no other source for them. Keep --no-deps only on
# the heavy CUDA package so it does not drag in a conflicting cuda/torch tree.
log_flashinfer python3 -m pip install --upgrade --no-deps \
  "nvidia-cutlass-dsl[cu13]>=4.5.0"
log_flashinfer python3 -m pip install --upgrade \
  scikit-build-core ninja cmake wheel setuptools-scm pathspec pyproject-metadata
# The rsync'd tree has no .git, so setuptools-scm cannot derive flashinfer's
# version -- pin it (scoped to this install) to match the c3dae30f nightly tag.
log_flashinfer env SETUPTOOLS_SCM_PRETEND_VERSION=0.6.13 \
  python3 -m pip install --no-deps --no-build-isolation -e \
  "${REPO_ROOT}/third_party/flashinfer" -v

pushd "${REPO_ROOT}/third_party/sglang/sgl-kernel" >/dev/null
log_sglang env \
  SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0.dev0" \
  CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}" \
  MAX_JOBS="${MAX_JOBS:-2}" \
  CMAKE_ARGS="${CMAKE_ARGS:--DSGL_KERNEL_COMPILE_THREADS=1 -DENABLE_BELOW_SM90=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DSGL_KERNEL_BUILD_SM90=OFF -DSGL_KERNEL_BUILD_SM100=ON -DSGL_KERNEL_ENABLE_FA3=OFF -DSGL_KERNEL_ENABLE_FLASHMLA=OFF -DSGL_KERNEL_ENABLE_SPATIAL=OFF}" \
  python3 -m pip install --no-deps --no-build-isolation --force-reinstall -v .
popd >/dev/null

log_sglang env SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0.dev0" \
  python3 -m pip install --no-deps --no-build-isolation -e \
  "${REPO_ROOT}/third_party/sglang/python" -v

python3 - <<'PY'
import importlib.metadata as md
import sgl_kernel
import torch

import flashinfer

print("torch", torch.__version__, torch.version.cuda)
print("capability", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
print("flashinfer", getattr(flashinfer, "__version__", None), getattr(flashinfer, "__file__", None))
print("flashinfer_python", md.version("flashinfer_python"))
print("sglang_kernel", md.version("sglang-kernel"))
print("sglang", md.version("sglang"))
print("sgl_kernel", getattr(sgl_kernel, "__file__", None))
print("common_ops", getattr(getattr(sgl_kernel, "common_ops", None), "__file__", None))
PY
