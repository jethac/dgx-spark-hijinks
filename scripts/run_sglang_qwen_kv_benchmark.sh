#!/usr/bin/env bash
set -uo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/sglang:26.05-py3}
RUNTIME_IMAGE=${RUNTIME_IMAGE:-sglang-source-stack-c3dae30f-e631a13fd}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-${MODEL}}
PORT=${PORT:-30000}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
PAGE_SIZE=${PAGE_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.40}
MIXED_KV=${MIXED_KV:-0}
DISABLE_GRAPHS=${DISABLE_GRAPHS:-1}
ENABLE_FP4_CUDA_GRAPH=${ENABLE_FP4_CUDA_GRAPH:-0}
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-0}
INSTALL_SOURCE_STACK_PER_CASE=${INSTALL_SOURCE_STACK_PER_CASE:-0}
RUN_ID=${RUN_ID:-sglang_qwen_kv_${KV_CACHE_DTYPE}_$(date -u +%Y%m%dT%H%M%SZ)}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results}
HF_CACHE=${HF_CACHE:-${HOME}/.cache/huggingface}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-360}
GB10_DOCKER_MEMORY=${GB10_DOCKER_MEMORY:-100g}
GB10_DOCKER_MEMORY_SWAP=${GB10_DOCKER_MEMORY_SWAP:-100g}
BENCHMARK_CASES=${BENCHMARK_CASES:-short_decode medium_decode long_prefill}
BENCHMARK_TIMEOUT=${BENCHMARK_TIMEOUT:-300}
EXTRA_SERVER_ENVS=${EXTRA_SERVER_ENVS:-}

mkdir -p "${RESULTS_DIR}"

container="${RUN_ID}"
server_log="results/${RUN_ID}_server.log"
benchmark_json="results/${RUN_ID}_openai_benchmark.json"
manifest_json="results/${RUN_ID}_row_manifest.json"
inspect_json="results/${RUN_ID}_container_inspect.json"
install_log="results/${RUN_ID}_editable_install.log"
flashinfer_install_log="results/${RUN_ID}_flashinfer_editable_install.log"
cid_file="results/${RUN_ID}_container_id.txt"

docker rm -f "${container}" >/dev/null 2>&1 || true

extra_envs=(
  -e PYTHONPATH="/work/third_party/sglang/python:${PYTHONPATH:-}"
  -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
)
if [[ "${MIXED_KV}" == "1" ]]; then
  extra_envs+=(-e SGLANG_FP4_KV_MIXED_KV=1)
fi
if [[ "${ENABLE_FP4_CUDA_GRAPH}" == "1" ]]; then
  extra_envs+=(-e SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=1)
fi
for envpair in ${EXTRA_SERVER_ENVS}; do
  extra_envs+=(-e "${envpair}")
done

launch_args=(
  --model-path "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --attention-backend flashinfer
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --page-size "${PAGE_SIZE}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --host 0.0.0.0
  --port "${PORT}"
)
if [[ "${DISABLE_GRAPHS}" == "1" ]]; then
  launch_args+=(--disable-cuda-graph --disable-piecewise-cuda-graph)
fi
if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
  launch_args+=(--disable-radix-cache)
fi

printf 'run_id=%s\nkv_cache_dtype=%s\nmixed_kv=%s\nmem_fraction_static=%s\ndisable_graphs=%s\nenable_fp4_cuda_graph=%s\ndisable_radix_cache=%s\n' \
  "${RUN_ID}" "${KV_CACHE_DTYPE}" "${MIXED_KV}" "${MEM_FRACTION_STATIC}" \
  "${DISABLE_GRAPHS}" "${ENABLE_FP4_CUDA_GRAPH}" "${DISABLE_RADIX_CACHE}"

cid=$(
  docker run -d --name "${container}" --gpus all --ipc=host --network=host \
    --memory "${GB10_DOCKER_MEMORY}" --memory-swap "${GB10_DOCKER_MEMORY_SWAP}" \
    -v "${REPO_ROOT}:/work" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -w /work \
    "${extra_envs[@]}" \
    -e SGLANG_BENCH_INSTALL_SOURCE_STACK="${INSTALL_SOURCE_STACK_PER_CASE}" \
    "${RUNTIME_IMAGE}" \
    bash -lc "set -euo pipefail; git config --global --add safe.directory /work; git config --global --add safe.directory /work/third_party/flashinfer; git config --global --add safe.directory /work/third_party/sglang; if [[ \"\${SGLANG_BENCH_INSTALL_SOURCE_STACK}\" == \"1\" ]]; then REPO_ROOT=/work FLASHINFER_INSTALL_LOG=/work/${flashinfer_install_log} SGLANG_INSTALL_LOG=/work/${install_log} bash /work/scripts/install_sglang_source_stack.sh >> /work/${install_log} 2>&1; else echo \"using prepared source-stack image ${RUNTIME_IMAGE}\" > /work/${install_log}; : > /work/${flashinfer_install_log}; fi; python3 -c 'import flashinfer, importlib.metadata as md, sgl_kernel; print(\"flashinfer\", getattr(flashinfer, \"__version__\", None), getattr(flashinfer, \"__file__\", None)); print(\"flashinfer_python\", md.version(\"flashinfer_python\")); print(\"sglang_kernel\", md.version(\"sglang-kernel\")); print(\"sglang\", md.version(\"sglang\")); print(\"sgl_kernel\", getattr(sgl_kernel, \"__file__\", None)); print(\"common_ops\", getattr(getattr(sgl_kernel, \"common_ops\", None), \"__file__\", None))' >> /work/${install_log} 2>&1; exec python3 -m sglang.launch_server ${launch_args[*]}"
)
echo "${cid}" >"${REPO_ROOT}/${cid_file}"

ready=0
attempts=$((READY_TIMEOUT_S / 2))
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

status=0
if [[ "${ready}" != "1" ]]; then
  echo "server not ready for ${RUN_ID}" >&2
  status=1
else
  case_args=()
  for case_name in ${BENCHMARK_CASES}; do
    case_args+=(--case "${case_name}")
  done
  docker exec "${container}" python3 /work/scripts/openai_serving_benchmark.py \
    --url "http://127.0.0.1:${PORT}" \
    --model "${SERVED_MODEL_NAME}" \
    --backend sglang \
    --phase exploratory \
    --run-id "${RUN_ID}" \
    "${case_args[@]}" \
    --timeout "${BENCHMARK_TIMEOUT}" \
    --output "/work/${benchmark_json}"
  status=$?
fi

docker logs "${container}" >"${REPO_ROOT}/${server_log}" 2>&1 || true
docker inspect "${container}" >"${REPO_ROOT}/${inspect_json}" 2>/dev/null || true
docker rm -f "${container}" >/dev/null 2>&1 || true

python3 - <<PY
import json
import re
from pathlib import Path

repo = Path("${REPO_ROOT}")
server_log = repo / "${server_log}"
benchmark = repo / "${benchmark_json}"
manifest = repo / "${manifest_json}"
inspect = repo / "${inspect_json}"
install_log = repo / "${install_log}"
kv_line = None
max_total_num_tokens = None
if server_log.exists():
    for line in server_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "KV Cache is allocated." in line:
            kv_line = line.strip()
        if "max_total_num_tokens=" in line:
            match = re.search(r"max_total_num_tokens=(\\d+)", line)
            if match:
                max_total_num_tokens = int(match.group(1))
kv_tokens = None
k_size_gb = None
v_size_gb = None
if kv_line:
    match = re.search(r"#tokens: (\\d+)", kv_line)
    if match:
        kv_tokens = int(match.group(1))
    match = re.search(r"K size: ([0-9.]+) GB", kv_line)
    if match:
        k_size_gb = float(match.group(1))
    match = re.search(r"V size: ([0-9.]+) GB", kv_line)
    if match:
        v_size_gb = float(match.group(1))
bench_obj = None
if benchmark.exists():
    try:
        bench_obj = json.loads(benchmark.read_text(encoding="utf-8"))
    except Exception as exc:
        bench_obj = {"parse_error": repr(exc)}
manifest_obj = {
    "schema": "sglang-qwen-kv-serving-row/v1",
    "run_id": "${RUN_ID}",
    "runtime_image": "${RUNTIME_IMAGE}",
    "model": "${MODEL}",
    "served_model_name": "${SERVED_MODEL_NAME}",
    "kv_cache_dtype": "${KV_CACHE_DTYPE}",
    "mixed_kv": "${MIXED_KV}" == "1",
    "page_size": int("${PAGE_SIZE}"),
    "mem_fraction_static": float("${MEM_FRACTION_STATIC}"),
    "graphs_disabled": "${DISABLE_GRAPHS}" == "1",
    "enable_fp4_cuda_graph": "${ENABLE_FP4_CUDA_GRAPH}" == "1",
    "radix_cache_disabled": "${DISABLE_RADIX_CACHE}" == "1",
    "extra_server_envs": "${EXTRA_SERVER_ENVS}".split(),
    "docker_memory": "${GB10_DOCKER_MEMORY}",
    "docker_memory_swap": "${GB10_DOCKER_MEMORY_SWAP}",
    "ready": "${ready}" == "1",
    "runner_status": int("${status}"),
    "kv_line": kv_line,
    "kv_tokens": kv_tokens,
    "max_total_num_tokens": max_total_num_tokens,
    "k_size_gb": k_size_gb,
    "v_size_gb": v_size_gb,
    "artifacts": {
        "server_log": str(server_log),
        "openai_benchmark": str(benchmark),
        "container_inspect": str(inspect),
        "editable_install_log": str(install_log),
    },
    "benchmark_ok": bool(bench_obj and bench_obj.get("ok")),
    "benchmark_cases": (bench_obj or {}).get("cases"),
}
manifest.write_text(json.dumps(manifest_obj, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(manifest_obj, indent=2, sort_keys=True))
PY

exit "${status}"
