#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/jethac/spark_tmp/dgx-spark-hijinks-022}"
IMAGE="${IMAGE:-sglang-source-stack-c3dae30f-e631a13fd:latest}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/sglang_fp4_kv_writer_roundtrip_$(date +%Y%m%dT%H%M%SJST)}"
CACHE_DIR="${CACHE_DIR:-/tmp/flashinfer-cache-sglang-writer-roundtrip-$(date +%Y%m%dT%H%M%S)}"

mkdir -p "${OUT_DIR}"

set +e
docker run --rm --gpus all --memory=16g --memory-swap=16g --ipc=host \
  -w /work \
  -v "${REPO_ROOT}:/work" \
  -v "${REPO_ROOT}/third_party/flashinfer:/flashinfer-src" \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CACHE_DIR="${CACHE_DIR}" \
  -e FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a" \
  -e PYTHONPATH=/work/third_party/sglang/python:/tmp/flashinfer-python-path \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    python - <<'"'"'PY'"'"'
import os
import shutil
import site

for sp in site.getsitepackages():
    for name in ("flashinfer", "flashinfer_python", "flashinfer_cubin"):
        path = os.path.join(sp, name)
        if os.path.exists(path) or os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)

os.makedirs("/tmp/flashinfer-python-path", exist_ok=True)
link = "/tmp/flashinfer-python-path/flashinfer"
if os.path.lexists(link):
    os.unlink(link)
os.symlink("/flashinfer-src/flashinfer", link)
PY
    python /work/scripts/sglang_fp4_kv_writer_roundtrip_probe.py \
      --head-dim 256 \
      --num-qo-heads 32 \
      --num-kv-heads 16 \
      --kv-len 384 \
      --qo-len 16 \
      --window-left 255
  ' > "${OUT_DIR}/container.stdout" 2> "${OUT_DIR}/run.log"
status=$?
set -e

awk 'found || /^\{/ { found = 1; print }' "${OUT_DIR}/container.stdout" > "${OUT_DIR}/output.json"
cat "${OUT_DIR}/output.json"
echo "wrote ${OUT_DIR}" >&2
exit "${status}"
