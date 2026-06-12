#!/usr/bin/env bash
# SGLang Gemma 4 AR ladder packet.
#
# Runs one or more Gemma 4 autoregressive sizes through sequential bf16/auto-KV
# and full-NVFP4 K+V servers. The baked source-stack image keeps its editable
# SGLang/FlashInfer sources under /work, so this script mounts the hijinks repo
# at /hijinks and never overlays /work.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live}"
IMAGE="${IMAGE:-ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack:epoch2-gemma4-tf511-12fca91}"
IMAGE_DIGEST="${IMAGE_DIGEST:-ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:bf24438b302c96e457b8a59f8a8dbaf109fab08013554be81e6957d4fb0f1a70}"
MODELS="${MODELS:-google/gemma-4-12B-it google/gemma-4-26B-A4B-it google/gemma-4-31B-it}"
PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
PAGE_SIZE="${PAGE_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1800}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-1800}"
PPL_TIMEOUT_S="${PPL_TIMEOUT_S:-1800}"
GB10_DOCKER_MEMORY="${GB10_DOCKER_MEMORY:-100g}"
GB10_DOCKER_MEMORY_SWAP="${GB10_DOCKER_MEMORY_SWAP:-100g}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
RUN_ID="${RUN_ID:-sglang_gemma4_ar_ladder_$(TZ=Asia/Tokyo date +%Y%m%dT%H%M%SJST)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/${RUN_ID}}"
CORPUS="${CORPUS:-${OUT_DIR}/ppl_corpus.md}"
CORPUS_MANIFEST="${CORPUS_MANIFEST:-${OUT_DIR}/ppl_corpus_manifest.json}"
CTX_LIST="${CTX_LIST:-512}"
REUSE_PREFIX_LEN="${REUSE_PREFIX_LEN:-256}"
LOGPROB_START_LEN="${LOGPROB_START_LEN:-${REUSE_PREFIX_LEN}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
CLAUDE_MARKER="${CLAUDE_MARKER:-/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN}"

if [[ -e "${CLAUDE_MARKER}" ]]; then
  echo "CLAUDE_WINDOW_OPEN present; yielding" >&2
  exit 99
fi
if [[ "$(docker ps -q | wc -l)" != "0" ]]; then
  echo "docker is not empty; yielding" >&2
  docker ps >&2
  exit 99
fi

cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

if [[ ! -f "${CORPUS}" ]]; then
  python3 scripts/build_ppl_corpus.py \
    --repo-root "${REPO_ROOT}" \
    --output "${CORPUS}" \
    --manifest "${CORPUS_MANIFEST}" \
    --max-chars "${CORPUS_MAX_CHARS:-250000}"
fi

slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's#[^a-z0-9]+#-#g; s#^-+##; s#-+$##'
}

wait_ready() {
  local name="$1"
  local attempts=$((READY_TIMEOUT_S / 5))
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
    sleep 5
  done
  return 1
}

capture_logs() {
  local name="$1"
  local out="$2"
  docker logs "${name}" >"${out}.tmp" 2>&1 && mv "${out}.tmp" "${out}" || rm -f "${out}.tmp"
}

run_one() {
  local model="$1"
  local label="$2"
  local kv_dtype="$3"
  local mixed_kv="$4"
  local model_slug
  model_slug="$(slugify "${model}")"
  local name="${RUN_ID}_${model_slug}_${label}"
  name="$(printf '%.120s' "${name}")"
  local model_dir="${OUT_DIR}/${model_slug}"
  local server_log="${model_dir}/${label}_server.log"
  local install_log="${model_dir}/${label}_provenance.log"
  local chat1="${model_dir}/${label}_chat_1.json"
  local chat2="${model_dir}/${label}_chat_2.json"
  local ppl_json="${model_dir}/${label}_ppl.json"
  local ppl_stdout="${model_dir}/${label}_ppl_stdout.json"
  local ppl_stderr="${model_dir}/${label}_ppl_stderr.log"
  local inspect_json="${model_dir}/${label}_container_inspect.json"
  local flashinfer_cache_dir="/tmp/flashinfer-cache-${name}"

  mkdir -p "${model_dir}"
  docker rm -f "${name}" >/dev/null 2>&1 || true

  local kv_args=()
  if [[ "${kv_dtype}" != "auto" ]]; then
    kv_args+=(--kv-cache-dtype "${kv_dtype}")
  fi

  {
    echo "run_id=${RUN_ID}"
    echo "model=${model}"
    echo "label=${label}"
    echo "kv_cache_dtype=${kv_dtype}"
    echo "mixed_kv=${mixed_kv}"
    echo "image=${IMAGE}"
    echo "image_digest=${IMAGE_DIGEST}"
    echo "mem_fraction_static=${MEM_FRACTION_STATIC}"
    echo "page_size=${PAGE_SIZE}"
    echo "context_length=${CONTEXT_LENGTH}"
    echo "started_at=$(TZ=Asia/Tokyo date -Is)"
    free -h
  } >"${model_dir}/${label}_preflight.log"

  docker run -d --name "${name}" --gpus all --ipc=host --network=host \
    --memory "${GB10_DOCKER_MEMORY}" --memory-swap "${GB10_DOCKER_MEMORY_SWAP}" \
    -w /hijinks \
    -v "${REPO_ROOT}:/hijinks" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -e TORCH_CUDA_ARCH_LIST=12.1a \
    -e FLASHINFER_CACHE_DIR="${flashinfer_cache_dir}" \
    -e FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a" \
    -e FLASHINFER_PREFILL_DEBUG_ONCE=1 \
    -e SGLANG_FLASHINFER_VOSPLIT=1 \
    -e SGLANG_GEMMA4_TRACE_GEOMETRY=1 \
    -e SGLANG_GEMMA_KV_GEOMETRY=1 \
    -e SGLANG_FP4_KV_MIXED_KV="${mixed_kv}" \
    -e SGLANG_FP4_KV_TRACE_MODULE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    "${IMAGE}" \
    bash -lc '
      set -euo pipefail
      rm -rf /root/.cache/flashinfer "${FLASHINFER_CACHE_DIR}"
      python3 - <<'"'"'PY'"'"'
import hashlib
import importlib
import importlib.metadata as md
from pathlib import Path

import flashinfer
import sglang
import transformers
from flashinfer.jit import env as jit_env

print("transformers", transformers.__version__, flush=True)
print("sglang", md.version("sglang"), sglang.__file__, flush=True)
print("flashinfer", getattr(flashinfer, "__version__", None), flashinfer.__file__, flush=True)
print("flashinfer_python", md.version("flashinfer_python"), flush=True)
print("sglang_kernel", md.version("sglang-kernel"), flush=True)
for name in ("sgl_kernel", "sgl_kernel.common_ops"):
    try:
        mod = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        print(f"binary_missing {name} {exc}", flush=True)
        continue
    path = Path(getattr(mod, "__file__", "")).resolve()
    if path.is_file():
        print(f"binary_md5 {name} {path} {hashlib.md5(path.read_bytes()).hexdigest()}", flush=True)
print("flashinfer_data", jit_env.FLASHINFER_DATA, flush=True)
print("flashinfer_csrc", jit_env.FLASHINFER_CSRC_DIR, flush=True)
print("flashinfer_include", jit_env.FLASHINFER_INCLUDE_DIR, flush=True)
print("flashinfer_cutlass", jit_env.CUTLASS_INCLUDE_DIRS, flush=True)
print("flashinfer_cccl", jit_env.CCCL_INCLUDE_DIRS, flush=True)
print("flashinfer_spdlog", jit_env.SPDLOG_INCLUDE_DIR, flush=True)
PY
      exec python3 -m sglang.launch_server "$@"
    ' -- \
      --model-path "${model}" \
      --served-model-name "${model}" \
      --trust-remote-code \
      --dtype bfloat16 \
      --attention-backend flashinfer \
      --page-size "${PAGE_SIZE}" \
      --context-length "${CONTEXT_LENGTH}" \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --disable-cuda-graph \
      --disable-piecewise-cuda-graph \
      --host 0.0.0.0 \
      --port "${PORT}" \
      "${kv_args[@]}" \
      >"${model_dir}/${label}_container_id.txt"

  if ! wait_ready "${name}"; then
    capture_logs "${name}" "${server_log}" || true
    docker inspect "${name}" >"${inspect_json}" 2>/dev/null || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
    echo "server_not_ready" >"${model_dir}/${label}_status.txt"
    return 1
  fi

  capture_logs "${name}" "${install_log}" || true

  for idx in 1 2; do
    if ! curl -sS --max-time "${REQUEST_TIMEOUT_S}" "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "'"${model}"'",
        "messages": [{"role": "user", "content": "Answer with only the capital city of Japan."}],
        "temperature": 0,
        "max_tokens": 16
      }' >"${model_dir}/${label}_chat_${idx}.json"; then
      capture_logs "${name}" "${server_log}" || true
      docker inspect "${name}" >"${inspect_json}" 2>/dev/null || true
      docker rm -f "${name}" >/dev/null 2>&1 || true
      echo "chat_request_failed" >"${model_dir}/${label}_status.txt"
      return 1
    fi
  done

  ctx_args=()
  for ctx in ${CTX_LIST}; do
    ctx_args+=(--ctx "${ctx}")
  done

  if ! docker exec "${name}" python3 /hijinks/scripts/sglang_prompt_ppl_sweep.py \
    --url "http://127.0.0.1:${PORT}" \
    --tokenizer "${model}" \
    --text-file "/hijinks/results/${RUN_ID}/ppl_corpus.md" \
    "${ctx_args[@]}" \
    --run-id "${RUN_ID}_${model_slug}_${label}" \
    --kv-cache-dtype "${kv_dtype}" \
    --runtime-ref "SGLang ${IMAGE}; Gemma4 AR VO-split; ${label}; graphs disabled" \
    --container-image "${IMAGE}" \
    --scope "SGLang Gemma 4 AR ladder supplied-token PPL; sequential bf16/full-NVFP4 servers; graphs disabled" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --reuse-prefix-len "${REUSE_PREFIX_LEN}" \
    --logprob-start-len "${LOGPROB_START_LEN}" \
    --timeout "${PPL_TIMEOUT_S}" \
    --output "/hijinks/results/${RUN_ID}/${model_slug}/${label}_ppl.json" \
    >"${ppl_stdout}" \
    2>"${ppl_stderr}"; then
    capture_logs "${name}" "${server_log}" || true
    docker inspect "${name}" >"${inspect_json}" 2>/dev/null || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
    echo "ppl_request_failed" >"${model_dir}/${label}_status.txt"
    return 1
  fi

  capture_logs "${name}" "${server_log}" || true
  docker inspect "${name}" >"${inspect_json}" 2>/dev/null || true
  docker rm -f "${name}" >/dev/null 2>&1 || true

  python3 - <<PY
import json
from pathlib import Path
model_dir = Path("${model_dir}")
def load(name):
    return json.loads((model_dir / name).read_text(encoding="utf-8"))
chat1 = load("${label}_chat_1.json")
chat2 = load("${label}_chat_2.json")
ppl = load("${label}_ppl.json")
content1 = chat1["choices"][0]["message"].get("content", "")
content2 = chat2["choices"][0]["message"].get("content", "")
summary = {
    "label": "${label}",
    "model": "${model}",
    "kv_cache_dtype": "${kv_dtype}",
    "mixed_kv": "${mixed_kv}" == "1",
    "chat_1": content1,
    "chat_2": content2,
    "chat_bitwise_equal": chat1 == chat2,
    "chat_content_equal": content1 == content2,
    "chat_tokyo": "tokyo" in content1.lower(),
    "ppl_ok": bool(ppl.get("ok")),
    "ppl_rows": ppl.get("results") or ppl.get("rows"),
}
(model_dir / "${label}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

{
  echo "run_id=${RUN_ID}"
  echo "repo_root=${REPO_ROOT}"
  echo "image=${IMAGE}"
  echo "image_digest=${IMAGE_DIGEST}"
  echo "models=${MODELS}"
  echo "mem_fraction_static=${MEM_FRACTION_STATIC}"
  echo "page_size=${PAGE_SIZE}"
  echo "context_length=${CONTEXT_LENGTH}"
  echo "ctx_list=${CTX_LIST}"
  echo "started_at=$(TZ=Asia/Tokyo date -Is)"
  git rev-parse HEAD
  docker image inspect "${IMAGE}" --format '{{json .RepoDigests}}' 2>/dev/null || true
  free -h
} | tee "${OUT_DIR}/preflight.log"

overall_status=0
for model in ${MODELS}; do
  model_slug="$(slugify "${model}")"
  mkdir -p "${OUT_DIR}/${model_slug}"
  if ! run_one "${model}" "bf16" "auto" "0"; then
    overall_status=1
    break
  fi
  if ! run_one "${model}" "fullnvfp4" "fp4_e2m1" "0"; then
    overall_status=1
    break
  fi
  python3 scripts/sglang_prompt_ppl_sweep.py \
    --compare-fp8 "${OUT_DIR}/${model_slug}/bf16_ppl.json" \
    --compare-candidate "${OUT_DIR}/${model_slug}/fullnvfp4_ppl.json" \
    --output "${OUT_DIR}/${model_slug}/compare.json" || overall_status=1
done

python3 - <<PY
import json
from pathlib import Path
out = Path("${OUT_DIR}")
models = "${MODELS}".split()
rows = []
for model in models:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in model).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    model_dir = out / slug
    row = {"model": model, "dir": str(model_dir)}
    for label in ("bf16", "fullnvfp4"):
        path = model_dir / f"{label}_summary.json"
        row[label] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    compare = model_dir / "compare.json"
    row["compare"] = json.loads(compare.read_text(encoding="utf-8")) if compare.exists() else None
    rows.append(row)
manifest = {
    "schema": "sglang-gemma4-ar-ladder-pair/v1",
    "run_id": "${RUN_ID}",
    "image": "${IMAGE}",
    "image_digest": "${IMAGE_DIGEST}",
    "scope": "SGLang Gemma 4 AR ladder; sequential bf16/auto and full-NVFP4 K+V; graphs disabled; one server at a time",
    "models": models,
    "rows": rows,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

docker ps >"${OUT_DIR}/docker_ps_after.txt" 2>&1 || true
free -h >"${OUT_DIR}/free_after.txt" 2>&1 || true

exit "${overall_status}"
