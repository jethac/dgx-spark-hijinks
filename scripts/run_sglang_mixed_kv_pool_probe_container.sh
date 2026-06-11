#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 WORKDIR [IMAGE]" >&2
  exit 2
fi

WORKDIR=$1
IMAGE=${2:-nvcr.io/nvidia/sglang:26.05-py3}
SCRIPT=${SCRIPT:-scripts/sglang_mixed_kv_pool_probe.py}
CONTAINER=${CONTAINER:-sglang-mixed-kv-pool-probe}
FLASHINFER_EXTRA_CUDAFLAGS=${FLASHINFER_EXTRA_CUDAFLAGS:-"-gencode=arch=compute_121a,code=sm_121a"}

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

docker run --rm --name "${CONTAINER}" \
  --gpus all --ipc=host --network=host \
  --memory=16g --memory-swap=16g \
  -v "${WORKDIR}:/work" \
  -w /work \
  --entrypoint bash \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
  -e FLASHINFER_MIXED_KV_SCRIPT="${SCRIPT}" \
  -e FLASHINFER_MIXED_KV_PAGE_SIZE="${FLASHINFER_MIXED_KV_PAGE_SIZE:-16}" \
  -e FLASHINFER_EXTRA_CUDAFLAGS="${FLASHINFER_EXTRA_CUDAFLAGS}" \
  "${IMAGE}" \
  -lc '
set -euo pipefail
python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache >/tmp/fi_uninstall.log 2>&1 || true
python3 -m pip install -q "apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2"
rm -rf /tmp/flashinfer-python-path /tmp/flashinfer-mixed-kv-cache
mkdir -p /tmp/flashinfer-python-path /tmp/flashinfer-mixed-kv-cache
ln -s /work/third_party/flashinfer/flashinfer /tmp/flashinfer-python-path/flashinfer
export PYTHONPATH="/work/third_party/sglang/python:/tmp/flashinfer-python-path:${PYTHONPATH:-}"
export FLASHINFER_CACHE_DIR=/tmp/flashinfer-mixed-kv-cache
export FLASHINFER_EXTRA_CUDAFLAGS="${FLASHINFER_EXTRA_CUDAFLAGS}"
export TORCH_CUDA_ARCH_LIST=12.1a
python3 "${FLASHINFER_MIXED_KV_SCRIPT}"
'
