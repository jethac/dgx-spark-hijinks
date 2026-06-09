#!/usr/bin/env bash
set -uo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/sglang:26.05-py3}
RUNTIME_IMAGE=${RUNTIME_IMAGE:-}
PREPARE_RUST_IMAGE=${PREPARE_RUST_IMAGE:-1}
PREPARE_SOURCE_STACK_IMAGE=${PREPARE_SOURCE_STACK_IMAGE:-0}
SOURCE_STACK_IMAGE=${SOURCE_STACK_IMAGE:-}
INSTALL_SOURCE_STACK_PER_CASE=${INSTALL_SOURCE_STACK_PER_CASE:-1}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
PORT=${PORT:-30000}
RUN_ID=${RUN_ID:-sglang_qwen_fp4kv_dense_cache_$(date -u +%Y%m%dT%H%M%SZ)}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results}
HF_CACHE=${HF_CACHE:-${HOME}/.cache/huggingface}
CASES=${CASES:-default}
TRACE_LAYERS=${TRACE_LAYERS:-0,1,7,13,20,27}
TRACE_VALUES=${TRACE_VALUES:-64}
TRACE_LOC_LIMIT=${TRACE_LOC_LIMIT:-128}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-360}

mkdir -p "${RESULTS_DIR}"

if [[ -z "${RUNTIME_IMAGE}" ]]; then
  RUNTIME_IMAGE="${IMAGE}"
fi

image_tag=$(
  printf '%s' "${RUN_ID}" |
    tr '[:upper:]' '[:lower:]' |
    tr -c 'a-z0-9_.-' '-'
)

if [[ "${PREPARE_SOURCE_STACK_IMAGE}" == "1" ]]; then
  if [[ -n "${SOURCE_STACK_IMAGE}" ]]; then
    RUNTIME_IMAGE="${SOURCE_STACK_IMAGE}"
  else
    RUNTIME_IMAGE="sglang-source-stack-${image_tag}"
  fi
  OUTPUT_IMAGE="${RUNTIME_IMAGE}" \
    IMAGE="${IMAGE}" \
    PREPARE_RUST_IMAGE="${PREPARE_RUST_IMAGE}" \
    RUN_ID="${RUN_ID}_source_stack" \
    REPO_ROOT="${REPO_ROOT}" \
    RESULTS_DIR="${RESULTS_DIR}" \
    HF_CACHE="${HF_CACHE}" \
    bash "${REPO_ROOT}/scripts/prepare_sglang_source_stack_image.sh"
  INSTALL_SOURCE_STACK_PER_CASE=0
elif [[ "${PREPARE_RUST_IMAGE}" == "1" ]]; then
  RUNTIME_IMAGE="sglang-dense-cache-${image_tag}-rust"
  if ! docker build -t "${RUNTIME_IMAGE}" - <<EOF
FROM ${IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler && rm -rf /var/lib/apt/lists/*
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:\${PATH}"
EOF
  then
    exit 1
  fi
fi

COMMON_ENVS=(
  -e SGLANG_FP4_KV_TRACE_DENSE_CACHE=1
  -e SGLANG_FP4_KV_TRACE_RADIX=1
  -e SGLANG_FP4_KV_TRACE_BACKEND=1
  -e SGLANG_FP4_KV_TRACE_LAYERS="${TRACE_LAYERS}"
  -e SGLANG_FP4_KV_TRACE_VALUES="${TRACE_VALUES}"
  -e SGLANG_FP4_KV_TRACE_LOC_LIMIT="${TRACE_LOC_LIMIT}"
)

case_envs() {
  local name="$1"
  case "${name}" in
    default)
      ;;
    full_paged)
      printf '%s\n' SGLANG_FLASHINFER_USE_PAGED=1
      ;;
    force_miss)
      printf '%s\n' SGLANG_RADIX_FORCE_MISS=1
      ;;
    force_miss_full_paged)
      printf '%s\n' SGLANG_RADIX_FORCE_MISS=1 SGLANG_FLASHINFER_USE_PAGED=1
      ;;
    *)
      echo "unknown case: ${name}" >&2
      return 1
      ;;
  esac
}

run_case() {
  local name="$1"
  local container="${RUN_ID}_${name}"
  local out="results/${RUN_ID}_${name}.json"
  local server_log="results/${RUN_ID}_${name}_server.log"
  local trace_compare="results/${RUN_ID}_${name}_dense_cache_compare.json"
  local inspect="results/${RUN_ID}_${name}_container_inspect.json"
  local flashinfer_install_log="results/${RUN_ID}_${name}_flashinfer_editable_install.log"
  local install_log="results/${RUN_ID}_${name}_editable_install.log"
  local cid_file="results/${RUN_ID}_${name}_container_id.txt"

  echo "==== CASE ${name} ===="
  docker rm -f "${container}" >/dev/null 2>&1 || true

  local extra_envs=()
  while IFS= read -r envpair; do
    [[ -z "${envpair}" ]] && continue
    extra_envs+=(-e "${envpair}")
  done < <(case_envs "${name}")

  local cid
  cid=$(
    docker run -d --name "${container}" --gpus all --ipc=host --network=host \
      -v "${REPO_ROOT}:/work" \
      -v "${HF_CACHE}:/root/.cache/huggingface" \
      -w /work \
      "${COMMON_ENVS[@]}" "${extra_envs[@]}" \
      -e SGLANG_DENSE_CACHE_INSTALL_SOURCE_STACK="${INSTALL_SOURCE_STACK_PER_CASE}" \
      "${RUNTIME_IMAGE}" \
      bash -lc "set -euo pipefail; git config --global --add safe.directory /work; git config --global --add safe.directory /work/third_party/flashinfer; git config --global --add safe.directory /work/third_party/sglang; if [[ \"\${SGLANG_DENSE_CACHE_INSTALL_SOURCE_STACK}\" == \"1\" ]]; then REPO_ROOT=/work FLASHINFER_INSTALL_LOG=/work/${flashinfer_install_log} SGLANG_INSTALL_LOG=/work/${install_log} bash /work/scripts/install_sglang_source_stack.sh >> /work/${install_log} 2>&1; else echo \"using prepared source-stack image ${RUNTIME_IMAGE}\" > /work/${install_log}; : > /work/${flashinfer_install_log}; fi; python3 -c 'import flashinfer, importlib.metadata as md, sgl_kernel; print(\"flashinfer\", getattr(flashinfer, \"__version__\", None), getattr(flashinfer, \"__file__\", None)); print(\"flashinfer_python\", md.version(\"flashinfer_python\")); print(\"sglang_kernel\", md.version(\"sglang-kernel\")); print(\"sglang\", md.version(\"sglang\")); print(\"sgl_kernel\", getattr(sgl_kernel, \"__file__\", None)); print(\"common_ops\", getattr(getattr(sgl_kernel, \"common_ops\", None), \"__file__\", None))' >> /work/${install_log} 2>&1; exec python3 -m sglang.launch_server --model-path ${MODEL} --attention-backend flashinfer --kv-cache-dtype fp4_e2m1 --page-size 1 --mem-fraction-static 0.40 --disable-cuda-graph --disable-piecewise-cuda-graph --host 0.0.0.0 --port ${PORT}"
  )
  echo "${cid}" >"${REPO_ROOT}/${cid_file}"

  local ready=0
  local attempts=$((READY_TIMEOUT_S / 2))
  if (( attempts < 1 )); then
    attempts=1
  fi
  for _ in $(seq 1 "${attempts}"); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${container}"; then
      break
    fi
    sleep 2
  done

  if [[ "${ready}" != "1" ]]; then
    echo "server not ready for ${name}" >&2
    docker logs "${container}" >"${REPO_ROOT}/${server_log}" 2>&1 || true
    docker inspect "${container}" >"${REPO_ROOT}/${inspect}" 2>/dev/null || true
    docker rm -f "${container}" >/dev/null 2>&1 || true
    return 1
  fi

  docker exec "${container}" python3 /work/scripts/sglang_fp4_request_order_probe.py \
    --url "http://127.0.0.1:${PORT}" \
    --model "${MODEL}" \
    --model-path "${MODEL}" \
    --case medium_decode \
    --run-id "${RUN_ID}_${name}" \
    --output "/work/${out}" \
    --max-new-tokens 1 \
    --top-logprobs-num 20 \
    --timeout 180
  local rc=$?

  docker logs "${container}" >"${REPO_ROOT}/${server_log}" 2>&1 || true
  docker inspect "${container}" >"${REPO_ROOT}/${inspect}" 2>/dev/null || true
  docker rm -f "${container}" >/dev/null 2>&1 || true

  if [[ -f "${REPO_ROOT}/${out}" && -f "${REPO_ROOT}/${server_log}" ]]; then
    python3 "${REPO_ROOT}/scripts/sglang_dense_cache_trace_compare.py" \
      --request-json "${REPO_ROOT}/${out}" \
      --server-log "${REPO_ROOT}/${server_log}" \
      --output "${REPO_ROOT}/${trace_compare}" || rc=1
  fi
  return "${rc}"
}

status=0
for name in ${CASES}; do
  run_case "${name}" || status=1
done

python3 - <<PY
import json
from pathlib import Path

run = "${RUN_ID}"
case_names = "${CASES}".split()
base = Path("${RESULTS_DIR}")
summary = {
    "schema": "sglang-fp4-kv-dense-cache-trace-summary/v1",
    "run_id": run,
    "runtime_image": "${RUNTIME_IMAGE}",
    "cases": {},
}
for name in case_names:
    path = base / f"{run}_{name}.json"
    server_log = base / f"{run}_{name}_server.log"
    trace_compare = base / f"{run}_{name}_dense_cache_compare.json"
    if not path.exists():
        summary["cases"][name] = {
            "ok": False,
            "missing": str(path),
            "server_log": str(server_log),
            "trace_compare": str(trace_compare),
        }
        continue
    obj = json.loads(path.read_text())
    trace_compare_ok = None
    if trace_compare.exists():
        try:
            trace_compare_ok = json.loads(trace_compare.read_text()).get("ok")
        except Exception as exc:
            trace_compare_ok = f"error: {exc!r}"
    rows = []
    for row in obj.get("rows", []):
        row_summary = {"name": row.get("name"), "requests": []}
        for req in row.get("requests") or []:
            req_summary = req.get("summary") or {}
            first = req_summary.get("first_token") or {}
            if req.get("endpoint") == "native_generate":
                cached = req_summary.get("cached_tokens")
            else:
                cached = (req_summary.get("meta_info") or {}).get("cached_tokens")
            row_summary["requests"].append(
                {
                    "endpoint": req.get("endpoint"),
                    "rid": req.get("rid"),
                    "cached_tokens": cached,
                    "token": first.get("token"),
                    "token_id": first.get("token_id"),
                    "logprob": first.get("logprob"),
                }
            )
        rows.append(row_summary)
    summary["cases"][name] = {
        "ok": trace_compare_ok is not False,
        "server_log": str(server_log),
        "trace_compare": str(trace_compare),
        "trace_compare_ok": trace_compare_ok,
        "token_summary": obj.get("token_summary"),
        "rows": rows,
    }
out = base / f"{run}_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

exit "${status}"
