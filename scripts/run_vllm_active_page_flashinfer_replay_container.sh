#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass}
REPO_ROOT=${REPO_ROOT:-/work}
HOST_REPO_ROOT=${HOST_REPO_ROOT:-$(pwd)}
FLASHINFER_SRC=${FLASHINFER_SRC:-/flashinfer-src}
DUMP=${DUMP:?set DUMP to a spark-active-page-prefill-dump .pt file}
OUT=${OUT:-results/vllm_active_page_flashinfer_replay.json}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$(dirname "${OUT}")"

docker run --rm --gpus all --ipc=host --network=host \
  -v "${HOST_REPO_ROOT}:${REPO_ROOT}" \
  -v "${HOST_REPO_ROOT}/third_party/flashinfer:${FLASHINFER_SRC}" \
  -w "${REPO_ROOT}" \
  "${IMAGE}" \
  bash -lc "
    set -euo pipefail
    python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache >/tmp/active_page_pip_uninstall.log 2>&1 || true
    python3 -m pip install -q --upgrade \
      'nvidia-cutlass-dsl[cu13]>=4.5.0' \
      'apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2'
    rm -rf /tmp/flashinfer-python-path
    mkdir -p /tmp/flashinfer-python-path
    ln -s ${FLASHINFER_SRC}/flashinfer /tmp/flashinfer-python-path/flashinfer
    export PYTHONPATH=/tmp/flashinfer-python-path:\${PYTHONPATH:-}
    export FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1'
    export FLASHINFER_JIT_DEBUG=1
    python3 scripts/vllm_active_page_flashinfer_replay.py \
      ${DUMP} \
      --flashinfer-source-root ${FLASHINFER_SRC} \
      --output ${OUT} \
      --deswizzle-reference \
      ${EXTRA_ARGS}
  "
