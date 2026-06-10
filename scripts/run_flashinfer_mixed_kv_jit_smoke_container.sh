#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 WORKDIR FLASHINFER_SRC [IMAGE] [SCRIPT]" >&2
  exit 2
fi

WORKDIR=$1
FLASHINFER_SRC=$2
IMAGE=${3:-jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass}
SCRIPT=${4:-scripts/flashinfer_mixed_kv_jit_smoke.py}

docker run --rm --name flashinfer-mixed-kv-jit-smoke \
  --gpus all --ipc=host --network=host \
  --memory=16g --memory-swap=16g \
  -v "${WORKDIR}:/work" \
  -v "${FLASHINFER_SRC}:/flashinfer-src" \
  -w /work \
  --entrypoint bash \
  -e FLASHINFER_MIXED_KV_SCRIPT="${SCRIPT}" \
  "${IMAGE}" \
  -lc '
set -euo pipefail
python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache >/tmp/fi_uninstall.log 2>&1 || true
python3 -m pip install -q "apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2"
rm -rf /tmp/flashinfer-python-path /tmp/flashinfer-mixed-kv-cache
mkdir -p /tmp/flashinfer-python-path /tmp/flashinfer-mixed-kv-cache
ln -s /flashinfer-src/flashinfer /tmp/flashinfer-python-path/flashinfer
export PYTHONPATH="/tmp/flashinfer-python-path:${PYTHONPATH:-}"
export FLASHINFER_CACHE_DIR=/tmp/flashinfer-mixed-kv-cache
export FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1 -gencode=arch=compute_121a,code=sm_121a"
export TORCH_CUDA_ARCH_LIST=12.1a
python3 "${FLASHINFER_MIXED_KV_SCRIPT}"
'
