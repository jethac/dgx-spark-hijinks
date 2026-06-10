#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass}
REPO_ROOT=${REPO_ROOT:-/work}
HOST_REPO_ROOT=${HOST_REPO_ROOT:-$(pwd)}
FLASHINFER_SRC=${FLASHINFER_SRC:-/flashinfer-src}
OUT_DIR=${OUT_DIR:-results/flashinfer_nvfp4_page_deswizzle_matrix_20260610TmanualJST}

mkdir -p "${OUT_DIR}"

run_case() {
  local page_size=$1
  local deswizzle=$2
  local out="${OUT_DIR}/page${page_size}_deswizzle_${deswizzle}.json"
  local extra_args=()
  local extra_flags=""

  if [[ "${deswizzle}" == "on" ]]; then
    extra_flags="-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1"
  else
    extra_args+=(--no-deswizzle-flag)
  fi

  docker run --rm --gpus all --ipc=host --network=host \
    --memory=100g --memory-swap=100g \
    -v "${HOST_REPO_ROOT}:${REPO_ROOT}" \
    -v "${HOST_REPO_ROOT}/third_party/flashinfer:${FLASHINFER_SRC}" \
    -w "${REPO_ROOT}" \
    -e FLASHINFER_EXTRA_CUDAFLAGS="${extra_flags}" \
    "${IMAGE}" \
    bash -lc "
      set -euo pipefail
      python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache >/tmp/flashinfer_uninstall.log 2>&1 || true
      python3 -m pip install -q --upgrade \
        'nvidia-cutlass-dsl[cu13]>=4.5.0' \
        'apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2'
      rm -rf /tmp/flashinfer-python-path /tmp/flashinfer-jit-cache
      mkdir -p /tmp/flashinfer-python-path /tmp/flashinfer-jit-cache
      ln -s ${FLASHINFER_SRC}/flashinfer /tmp/flashinfer-python-path/flashinfer
      export PYTHONPATH=/tmp/flashinfer-python-path:\${PYTHONPATH:-}
      export FLASHINFER_JIT_CACHE_DIR=/tmp/flashinfer-jit-cache
      export FLASHINFER_JIT_DEBUG=1
      set +e
      python3 scripts/flashinfer_nvfp4_kv_probe.py \
        --flashinfer-source-root ${FLASHINFER_SRC} \
        --output ${out} \
        --layouts NHD \
        --kv-container tuple \
        --v-scale-layout linear \
        --head-dim 128 \
        --num-kv-heads 2 \
        --num-qo-heads 4 \
        --page-size ${page_size} \
        --kv-len 96 \
        --qo-len 32 \
        --k-global-scale 1.0 \
        --v-global-scale 1.0 \
        --signed-values \
        ${extra_args[*]}
      probe_status=\$?
      set -e
      if [[ ! -s ${out} ]]; then
        exit \${probe_status}
      fi
      exit 0
    "
}

for page_size in 1 16; do
  for deswizzle in off on; do
    run_case "${page_size}" "${deswizzle}"
  done
done

MATRIX_OUT_DIR="${OUT_DIR}" python3 - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["MATRIX_OUT_DIR"])
rows = []
for path in sorted(out_dir.glob("page*_deswizzle_*.json")):
    data = json.loads(path.read_text())
    page_size = data["metadata"]["page_size"]
    flags = data["environment"].get("flashinfer_extra_cudaflags", "")
    deswizzle = "on" if "FLASHINFER_PAGED_V_SF_DESWIZZLE" in flags else "off"
    for item in data["results"]:
        rows.append(
            {
                "file": path.name,
                "page_size": page_size,
                "deswizzle": deswizzle,
                "operation": item["operation"],
                "layout": item["layout"],
                "passed": item.get("passed", False),
                "cosine": item.get("cosine"),
                "max_abs": item.get("max_abs"),
                "error": item.get("error"),
            }
        )
summary = {"schema": "flashinfer-nvfp4-page-deswizzle-matrix/v1", "rows": rows}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
