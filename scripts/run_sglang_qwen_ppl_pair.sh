#!/usr/bin/env bash
set -euo pipefail

STAMP=${STAMP:-$(date +%Y%m%dT%H%MJST)}
CTX_LIST=${CTX_LIST:-512}
RUN=${RUN:-sglang_qwen_mixedkv_ppl_${STAMP}}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results}
CORPUS=${CORPUS:-${RESULTS_DIR}/${RUN}_corpus.md}
CORPUS_MANIFEST=${CORPUS_MANIFEST:-${RESULTS_DIR}/${RUN}_corpus_manifest.json}
RUNTIME_IMAGE=${RUNTIME_IMAGE:-sglang-source-stack-c3dae30f-e631a13fd}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
PORT=${PORT:-30000}
PAGE_SIZE=${PAGE_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.40}
HF_CACHE=${HF_CACHE:-${HOME}/.cache/huggingface}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-900}
GB10_DOCKER_MEMORY=${GB10_DOCKER_MEMORY:-100g}
GB10_DOCKER_MEMORY_SWAP=${GB10_DOCKER_MEMORY_SWAP:-100g}
PPL_TIMEOUT=${PPL_TIMEOUT:-1800}
DISABLE_GRAPHS=${DISABLE_GRAPHS:-0}
ENABLE_FP4_CUDA_GRAPH=${ENABLE_FP4_CUDA_GRAPH:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1}
REUSE_PREFIX_LEN=${REUSE_PREFIX_LEN:-0}
LOGPROB_START_LEN=${LOGPROB_START_LEN:-${REUSE_PREFIX_LEN}}
INCLUDE_TOKEN_LOGPROBS=${INCLUDE_TOKEN_LOGPROBS:-0}
EXTRA_SERVER_ENVS=${EXTRA_SERVER_ENVS:-}

mkdir -p "${RESULTS_DIR}"

if [[ ! -f "${CORPUS}" ]]; then
  python3 "${REPO_ROOT}/scripts/build_ppl_corpus.py" \
    --repo-root "${REPO_ROOT}" \
    --output "${CORPUS}" \
    --manifest "${CORPUS_MANIFEST}" \
    --max-chars "${CORPUS_MAX_CHARS:-250000}"
fi

ctx_args=()
for ctx in ${CTX_LIST}; do
  ctx_args+=(--ctx "${ctx}")
done

wait_ready() {
  local name=$1
  local attempts=$((READY_TIMEOUT_S / 2))
  if (( attempts < 1 )); then
    attempts=1
  fi
  for _ in $(seq 1 "${attempts}"); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${name}"; then
      return 1
    fi
    sleep 2
  done
  return 1
}

run_one() {
  local label=$1
  local kv_dtype=$2
  local mixed=$3
  local name="${RUN}_${label}"
  local server_log="${RESULTS_DIR}/${RUN}_${label}_server.log"
  local before_log="${RESULTS_DIR}/${RUN}_${label}_server_before_ppl.log"
  local inspect_json="${RESULTS_DIR}/${RUN}_${label}_container_inspect.json"
  local install_log="${RESULTS_DIR}/${RUN}_${label}_install.log"
  local ppl_json="${RESULTS_DIR}/${RUN}_${label}_ppl.json"
  local stdout_json="${RESULTS_DIR}/${RUN}_${label}_ppl_stdout.json"
  local stderr_log="${RESULTS_DIR}/${RUN}_${label}_ppl_stderr.log"

  docker rm -f "${name}" >/dev/null 2>&1 || true

  extra_envs=(
    -e PYTHONPATH="/work/third_party/sglang/python:${PYTHONPATH:-}"
    -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
  )
  if [[ "${mixed}" == "1" ]]; then
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
    --served-model-name "${MODEL}"
    --attention-backend flashinfer
    --kv-cache-dtype "${kv_dtype}"
    --page-size "${PAGE_SIZE}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --host 0.0.0.0
    --port "${PORT}"
  )
  if [[ "${DISABLE_GRAPHS}" == "1" ]]; then
    launch_args+=(--disable-cuda-graph --disable-piecewise-cuda-graph)
  fi
  logprob_detail_args=()
  if [[ "${INCLUDE_TOKEN_LOGPROBS}" == "1" ]]; then
    logprob_detail_args+=(--include-token-logprobs)
  fi

  docker run -d --name "${name}" --gpus all --ipc=host --network=host \
    --memory "${GB10_DOCKER_MEMORY}" --memory-swap "${GB10_DOCKER_MEMORY_SWAP}" \
    -v "${REPO_ROOT}:/work" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -w /work \
    "${extra_envs[@]}" \
    "${RUNTIME_IMAGE}" \
    bash -lc "set -euo pipefail; git config --global --add safe.directory /work; git config --global --add safe.directory /work/third_party/flashinfer; git config --global --add safe.directory /work/third_party/sglang; python3 -c 'import flashinfer, importlib.metadata as md, sgl_kernel; print(\"flashinfer\", getattr(flashinfer, \"__version__\", None), getattr(flashinfer, \"__file__\", None)); print(\"flashinfer_python\", md.version(\"flashinfer_python\")); print(\"sglang_kernel\", md.version(\"sglang-kernel\")); print(\"sglang\", md.version(\"sglang\")); print(\"sgl_kernel\", getattr(sgl_kernel, \"__file__\", None)); print(\"common_ops\", getattr(getattr(sgl_kernel, \"common_ops\", None), \"__file__\", None))' > /work/results/${RUN}_${label}_install.log 2>&1; exec python3 -m sglang.launch_server ${launch_args[*]}" \
    >/dev/null

  if ! wait_ready "${name}"; then
    docker logs "${name}" > "${server_log}" 2>&1 || true
    docker inspect "${name}" > "${inspect_json}" 2>/dev/null || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
    echo "server not ready for ${label}" >&2
    return 2
  fi

  docker logs "${name}" > "${before_log}" 2>&1 || true
  docker exec "${name}" python3 /work/scripts/sglang_prompt_ppl_sweep.py \
    --url "http://127.0.0.1:${PORT}" \
    --tokenizer "${MODEL}" \
    --text-file "/work/results/$(basename "${CORPUS}")" \
    "${ctx_args[@]}" \
    --run-id "${RUN}_${label}" \
    --kv-cache-dtype "${kv_dtype}" \
    --runtime-ref "SGLang ${RUNTIME_IMAGE}; mixed_kv=${mixed}; guarded piecewise graph for mixed cached-prefix prefill" \
    --container-image "${RUNTIME_IMAGE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --reuse-prefix-len "${REUSE_PREFIX_LEN}" \
    --logprob-start-len "${LOGPROB_START_LEN}" \
    --timeout "${PPL_TIMEOUT}" \
    "${logprob_detail_args[@]}" \
    --output "/work/results/${RUN}_${label}_ppl.json" \
    > "${stdout_json}" \
    2> "${stderr_log}"

  docker logs "${name}" > "${server_log}" 2>&1 || true
  docker inspect "${name}" > "${inspect_json}" 2>/dev/null || true
  docker rm -f "${name}" >/dev/null 2>&1 || true

  if [[ ! -s "${ppl_json}" ]]; then
    echo "missing PPL output ${ppl_json}" >&2
    return 3
  fi
}

run_one fp8 fp8_e4m3 0
run_one mixed fp4_e2m1 1

python3 "${REPO_ROOT}/scripts/sglang_prompt_ppl_sweep.py" \
  --compare-fp8 "${RESULTS_DIR}/${RUN}_fp8_ppl.json" \
  --compare-candidate "${RESULTS_DIR}/${RUN}_mixed_ppl.json" \
  --output "${RESULTS_DIR}/${RUN}_compare.json"

python3 - <<PY
import json
import re
from pathlib import Path

results = Path("${RESULTS_DIR}")
run = "${RUN}"
manifest = {
    "schema": "sglang-qwen-ppl-pair/v1",
    "run_id": run,
    "runtime_image": "${RUNTIME_IMAGE}",
    "model": "${MODEL}",
    "ctx_list": [int(x) for x in "${CTX_LIST}".split()],
    "reuse_prefix_len": int("${REUSE_PREFIX_LEN}"),
    "logprob_start_len": int("${LOGPROB_START_LEN}"),
    "page_size": int("${PAGE_SIZE}"),
    "mem_fraction_static": float("${MEM_FRACTION_STATIC}"),
    "graphs_disabled": "${DISABLE_GRAPHS}" == "1",
    "enable_fp4_cuda_graph": "${ENABLE_FP4_CUDA_GRAPH}" == "1",
    "docker_memory": "${GB10_DOCKER_MEMORY}",
    "docker_memory_swap": "${GB10_DOCKER_MEMORY_SWAP}",
    "include_token_logprobs": "${INCLUDE_TOKEN_LOGPROBS}" == "1",
    "extra_server_envs": "${EXTRA_SERVER_ENVS}".split(),
    "corpus": "${CORPUS}",
    "corpus_manifest": "${CORPUS_MANIFEST}",
    "artifacts": {
        "fp8_ppl": str(results / f"{run}_fp8_ppl.json"),
        "mixed_ppl": str(results / f"{run}_mixed_ppl.json"),
        "compare": str(results / f"{run}_compare.json"),
        "fp8_server_log": str(results / f"{run}_fp8_server.log"),
        "mixed_server_log": str(results / f"{run}_mixed_server.log"),
    },
}
for label in ("fp8", "mixed"):
    log_path = results / f"{run}_{label}_server.log"
    kv_line = None
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "KV Cache is allocated." in line:
                kv_line = line.strip()
    manifest[f"{label}_kv_line"] = kv_line
    if kv_line:
        match = re.search(r"#tokens: (\\d+)", kv_line)
        manifest[f"{label}_kv_tokens"] = int(match.group(1)) if match else None
compare = json.loads((results / f"{run}_compare.json").read_text(encoding="utf-8"))
manifest["compare_ok"] = bool(compare.get("ok"))
manifest["rows"] = compare.get("rows")
(results / f"{run}_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
