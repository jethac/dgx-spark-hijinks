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

if git -C "${REPO_ROOT}/third_party/flashinfer" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "${REPO_ROOT}/third_party/flashinfer" submodule update --init \
    3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog
else
  for required_dir in 3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog; do
    if [[ ! -d "${REPO_ROOT}/third_party/flashinfer/${required_dir}" ]]; then
      echo "missing FlashInfer dependency directory: ${required_dir}" >&2
      echo "provide a recursive checkout or run git submodule update before packaging the source tree" >&2
      exit 1
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

log_flashinfer python3 -m pip install --upgrade --no-deps \
  "nvidia-cutlass-dsl[cu13]>=4.5.0" scikit-build-core ninja cmake wheel \
  packaging pathspec pyproject-metadata
log_flashinfer python3 -m pip install --no-deps --no-build-isolation -e \
  "${REPO_ROOT}/third_party/flashinfer" -v

pushd "${REPO_ROOT}/third_party/sglang/sgl-kernel" >/dev/null
log_sglang env \
  CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}" \
  MAX_JOBS="${MAX_JOBS:-2}" \
  CMAKE_ARGS="${CMAKE_ARGS:--DSGL_KERNEL_COMPILE_THREADS=1 -DENABLE_BELOW_SM90=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DSGL_KERNEL_BUILD_SM90=OFF -DSGL_KERNEL_BUILD_SM100=ON -DSGL_KERNEL_ENABLE_FA3=OFF -DSGL_KERNEL_ENABLE_FLASHMLA=OFF -DSGL_KERNEL_ENABLE_SPATIAL=OFF}" \
  python3 -m pip install --no-deps --no-build-isolation --force-reinstall -v .
popd >/dev/null

log_sglang python3 -m pip install --no-deps --no-build-isolation -e \
  "${REPO_ROOT}/third_party/sglang/python" -v

python3 - <<'PY'
import importlib.util
import importlib.metadata as md
from pathlib import Path

import flashinfer
import torch

print("torch", torch.__version__, torch.version.cuda)
print("capability", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
print("flashinfer", getattr(flashinfer, "__version__", None), getattr(flashinfer, "__file__", None))
print("flashinfer_python", md.version("flashinfer_python"))
print("sglang_kernel", md.version("sglang-kernel"))
print("sglang", md.version("sglang"))
spec = importlib.util.find_spec("sgl_kernel")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("sgl_kernel package not installed")
sgl_kernel_dir = Path(next(iter(spec.submodule_search_locations)))
common_ops = sorted(sgl_kernel_dir.glob("sm*/common_ops*.so"))
if not common_ops:
    raise SystemExit(f"sgl_kernel common_ops library missing under {sgl_kernel_dir}")
print("sgl_kernel_dir", sgl_kernel_dir)
print("common_ops_files", [str(path) for path in common_ops])
PY
