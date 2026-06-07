#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 FLASHINFER_SRC WORKDIR OUTPUT_DIR RUN_PREFIX [IMAGE]" >&2
  exit 2
fi

FLASHINFER_SRC=$1
WORKDIR=$2
OUTPUT_DIR=$3
RUN_PREFIX=$4
IMAGE=${5:-vllm/vllm-openai:latest-cu130}

docker run --rm --gpus all --ipc=host \
  -v "${WORKDIR}:/work" \
  -v "${FLASHINFER_SRC}:/flashinfer-src" \
  -w /work \
  --entrypoint bash \
  "${IMAGE}" \
  -lc '
set -euo pipefail
python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache >/tmp/flashinfer_uninstall.log 2>&1 || true
python3 - <<'PY'
import shutil
import site
from pathlib import Path

patterns = [
    "flashinfer",
    "flashinfer-*.dist-info",
    "flashinfer_cubin",
    "flashinfer_cubin-*.dist-info",
    "flashinfer_jit_cache",
    "flashinfer_jit_cache-*.dist-info",
]
for root in site.getsitepackages():
    base = Path(root)
    for pattern in patterns:
        for path in base.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
PY
python3 -m pip install -q --upgrade \
  "nvidia-cutlass-dsl[cu13]>=4.5.0" \
  "apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2"
rm -rf /tmp/flashinfer-python-path
mkdir -p /tmp/flashinfer-python-path
ln -s /flashinfer-src/flashinfer /tmp/flashinfer-python-path/flashinfer
export PYTHONPATH="/tmp/flashinfer-python-path:${PYTHONPATH:-}"

python3 scripts/flashinfer_mm_fp4_microbench.py \
  --phase after \
  --run-id "'"${RUN_PREFIX}"'-dense-decode" \
  --container "'"${IMAGE}"'-patched-flashinfer-source" \
  --preset dense_decode \
  --iterations 30 \
  --warmup 5 \
  --output "'"${OUTPUT_DIR}"'/'"${RUN_PREFIX}"'_dense_decode.json"

python3 scripts/flashinfer_mm_fp4_microbench.py \
  --phase after \
  --run-id "'"${RUN_PREFIX}"'-moe-expert" \
  --container "'"${IMAGE}"'-patched-flashinfer-source" \
  --preset moe_expert \
  --iterations 30 \
  --warmup 5 \
  --output "'"${OUTPUT_DIR}"'/'"${RUN_PREFIX}"'_moe_expert.json"
'
