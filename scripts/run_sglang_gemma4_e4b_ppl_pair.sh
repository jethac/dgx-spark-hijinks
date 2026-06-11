#!/usr/bin/env bash
set -euo pipefail

STAMP=${STAMP:-$(date +%Y%m%dT%H%MJST)}
RUN=${RUN:-sglang_gemma4_e4b_fullnvfp4_ppl_${STAMP}}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results}
CORPUS=${CORPUS:-${RESULTS_DIR}/${RUN}_corpus.md}
CORPUS_MANIFEST=${CORPUS_MANIFEST:-${RESULTS_DIR}/${RUN}_corpus_manifest.json}
IMAGE=${IMAGE:-sglang-source-stack-c3dae30f-e631a13fd:latest}
MODEL=${MODEL:-google/gemma-4-E4B-it}
PORT=${PORT:-30000}
PAGE_SIZE=${PAGE_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.40}
HF_CACHE=${HF_CACHE:-${HOME}/.cache/huggingface}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-900}
GB10_DOCKER_MEMORY=${GB10_DOCKER_MEMORY:-100g}
GB10_DOCKER_MEMORY_SWAP=${GB10_DOCKER_MEMORY_SWAP:-100g}
PPL_TIMEOUT=${PPL_TIMEOUT:-1800}
CTX_LIST=${CTX_LIST:-512}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1}
REUSE_PREFIX_LEN=${REUSE_PREFIX_LEN:-256}
LOGPROB_START_LEN=${LOGPROB_START_LEN:-${REUSE_PREFIX_LEN}}
BASELINE_LABEL=${BASELINE_LABEL:-bf16}
BASELINE_KV_DTYPE=${BASELINE_KV_DTYPE:-auto}
BASELINE_MIXED_KV=${BASELINE_MIXED_KV:-0}
CANDIDATE_LABEL=${CANDIDATE_LABEL:-fullnvfp4}
CANDIDATE_KV_DTYPE=${CANDIDATE_KV_DTYPE:-fp4_e2m1}
CANDIDATE_MIXED_KV=${CANDIDATE_MIXED_KV:-0}

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
    if curl -fsS "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then
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

  env_args=(
    -e PYTHONPATH=/work/python_sitecustomize:/work/third_party/sglang/python:/tmp/flashinfer-python-path
    -e SPARK_FLASHINFER_SOURCE_ROOT=/work/third_party/flashinfer
    -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
    -e SGLANG_FLASHINFER_VOSPLIT=1
    -e SGLANG_GEMMA4_TRACE_GEOMETRY=1
    -e SGLANG_GEMMA_KV_GEOMETRY=1
    -e SGLANG_FP4_KV_MIXED_KV="${mixed}"
    -e SGLANG_FP4_KV_TRACE_MODULE=0
    -e HF_TOKEN="${HF_TOKEN:-}"
  )

  launch_args=(
    --model-path "${MODEL}"
    --served-model-name "${MODEL}"
    --dtype bfloat16
    --attention-backend flashinfer
    --page-size "${PAGE_SIZE}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --disable-cuda-graph
    --disable-piecewise-cuda-graph
    --host 0.0.0.0
    --port "${PORT}"
  )
  if [[ "${kv_dtype}" != "auto" ]]; then
    launch_args+=(--kv-cache-dtype "${kv_dtype}")
  fi

  docker run -d --name "${name}" --gpus all --ipc=host --network=host \
    --memory "${GB10_DOCKER_MEMORY}" --memory-swap "${GB10_DOCKER_MEMORY_SWAP}" \
    -v "${REPO_ROOT}:/work" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -w /work \
    "${env_args[@]}" \
    "${IMAGE}" \
    bash -lc "set -euo pipefail; git config --global --add safe.directory /work; git config --global --add safe.directory /work/third_party/flashinfer; git config --global --add safe.directory /work/third_party/sglang; python3 - <<'PY' > /work/results/${RUN}_${label}_install.log 2>&1
import hashlib
import importlib.metadata as md
import pathlib
import sgl_kernel
import flashinfer
import flashinfer.jit.env as jit_env

def md5(path):
    p = pathlib.Path(path)
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else 'missing'

print('flashinfer', getattr(flashinfer, '__version__', None), getattr(flashinfer, '__file__', None), flush=True)
print('flashinfer_python', md.version('flashinfer_python'), flush=True)
print('sglang_kernel', md.version('sglang-kernel'), flush=True)
print('sglang', md.version('sglang'), flush=True)
print('sgl_kernel', getattr(sgl_kernel, '__file__', None), flush=True)
print('common_ops', getattr(getattr(sgl_kernel, 'common_ops', None), '__file__', None), flush=True)
print('binary_md5', md5(getattr(getattr(sgl_kernel, 'common_ops', None), '__file__', '')), flush=True)
print('flashinfer_data', jit_env.FLASHINFER_DATA, flush=True)
print('flashinfer_csrc', jit_env.FLASHINFER_CSRC_DIR, flush=True)
print('flashinfer_include', jit_env.FLASHINFER_INCLUDE_DIR, flush=True)
PY
exec python3 -m sglang.launch_server ${launch_args[*]}" \
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
    --runtime-ref "SGLang ${IMAGE}; Gemma4 E4B VO-split; mixed_kv=${mixed}; graphs disabled" \
    --container-image "${IMAGE}" \
    --scope "SGLang Gemma 4 E4B supplied-token PPL; ${BASELINE_LABEL} versus ${CANDIDATE_LABEL}; sequential servers; graphs disabled" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --reuse-prefix-len "${REUSE_PREFIX_LEN}" \
    --logprob-start-len "${LOGPROB_START_LEN}" \
    --timeout "${PPL_TIMEOUT}" \
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

run_one "${BASELINE_LABEL}" "${BASELINE_KV_DTYPE}" "${BASELINE_MIXED_KV}"
run_one "${CANDIDATE_LABEL}" "${CANDIDATE_KV_DTYPE}" "${CANDIDATE_MIXED_KV}"

python3 "${REPO_ROOT}/scripts/sglang_prompt_ppl_sweep.py" \
  --compare-fp8 "${RESULTS_DIR}/${RUN}_${BASELINE_LABEL}_ppl.json" \
  --compare-candidate "${RESULTS_DIR}/${RUN}_${CANDIDATE_LABEL}_ppl.json" \
  --output "${RESULTS_DIR}/${RUN}_compare.json"

python3 - <<PY
import json
import re
from pathlib import Path

results = Path("${RESULTS_DIR}")
run = "${RUN}"
baseline_label = "${BASELINE_LABEL}"
candidate_label = "${CANDIDATE_LABEL}"

manifest = {
    "schema": "sglang-gemma4-e4b-ppl-pair/v1",
    "run_id": run,
    "runtime_image": "${IMAGE}",
    "model": "${MODEL}",
    "ctx_list": [int(x) for x in "${CTX_LIST}".split()],
    "reuse_prefix_len": int("${REUSE_PREFIX_LEN}"),
    "logprob_start_len": int("${LOGPROB_START_LEN}"),
    "page_size": int("${PAGE_SIZE}"),
    "mem_fraction_static": float("${MEM_FRACTION_STATIC}"),
    "graphs_disabled": True,
    "docker_memory": "${GB10_DOCKER_MEMORY}",
    "docker_memory_swap": "${GB10_DOCKER_MEMORY_SWAP}",
    "corpus": "${CORPUS}",
    "corpus_manifest": "${CORPUS_MANIFEST}",
    "baseline": {"label": baseline_label, "kv_cache_dtype": "${BASELINE_KV_DTYPE}", "mixed_kv": "${BASELINE_MIXED_KV}" == "1"},
    "candidate": {"label": candidate_label, "kv_cache_dtype": "${CANDIDATE_KV_DTYPE}", "mixed_kv": "${CANDIDATE_MIXED_KV}" == "1"},
    "artifacts": {
        "baseline_ppl": str(results / f"{run}_{baseline_label}_ppl.json"),
        "candidate_ppl": str(results / f"{run}_{candidate_label}_ppl.json"),
        "compare": str(results / f"{run}_compare.json"),
        "baseline_server_log": str(results / f"{run}_{baseline_label}_server.log"),
        "candidate_server_log": str(results / f"{run}_{candidate_label}_server.log"),
    },
}
for label in (baseline_label, candidate_label):
    log_path = results / f"{run}_{label}_server.log"
    kv_lines = []
    max_tokens = None
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "KV Cache is allocated." in line or "SWAKVPool mem usage:" in line or "max_total_num_tokens=" in line:
                kv_lines.append(line.strip())
            if "max_total_num_tokens=" in line:
                match = re.search(r"max_total_num_tokens=(\\d+)", line)
                if match:
                    max_tokens = int(match.group(1))
    manifest[f"{label}_kv_lines"] = kv_lines[:8]
    manifest[f"{label}_max_total_num_tokens"] = max_tokens
compare = json.loads((results / f"{run}_compare.json").read_text(encoding="utf-8"))
manifest["compare_ok"] = bool(compare.get("ok"))
manifest["rows"] = compare.get("rows")
(results / f"{run}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
