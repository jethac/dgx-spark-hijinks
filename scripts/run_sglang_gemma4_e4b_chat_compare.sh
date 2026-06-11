#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live}"
IMAGE="${IMAGE:-sglang-source-stack-c3dae30f-e631a13fd:latest}"
MODEL="${MODEL:-google/gemma-4-E4B-it}"
PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.40}"
GB10_DOCKER_MEMORY="${GB10_DOCKER_MEMORY:-100g}"
GB10_DOCKER_MEMORY_SWAP="${GB10_DOCKER_MEMORY_SWAP:-100g}"
RUN_ID="${RUN_ID:-sglang_gemma4_e4b_chat_compare_$(date +%Y%m%dT%H%M%SJST)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/${RUN_ID}}"
FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-/tmp/flashinfer-cache-${RUN_ID}}"
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

{
  echo "run_id=${RUN_ID}"
  echo "repo_root=${REPO_ROOT}"
  echo "image=${IMAGE}"
  echo "model=${MODEL}"
  echo "parent_commit=$(git rev-parse HEAD)"
  echo "sglang_commit=$(git -C third_party/sglang rev-parse HEAD)"
  echo "flashinfer_commit=$(git -C third_party/flashinfer rev-parse HEAD)"
  free -h
} | tee "${OUT_DIR}/preflight.log"

container="${RUN_ID}"
cid_file="${OUT_DIR}/container_id.txt"
server_log="${OUT_DIR}/server.log"

capture_docker_logs() {
  local tmp_log="${server_log}.tmp"
  if docker logs "${container}" >"${tmp_log}" 2>&1; then
    mv "${tmp_log}" "${server_log}"
  else
    rm -f "${tmp_log}"
  fi
}

cleanup() {
  local status=$?
  if docker ps --format '{{.Names}}' | grep -qx "${container}"; then
    capture_docker_logs || true
    docker rm -f "${container}" >/dev/null 2>&1 || true
  elif [[ -s "${cid_file}" ]]; then
    capture_docker_logs || true
    docker rm -f "$(cat "${cid_file}")" >/dev/null 2>&1 || true
  fi
  docker ps >"${OUT_DIR}/docker_ps_after.txt" 2>&1 || true
  free -h >"${OUT_DIR}/free_after.txt" 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT

docker rm -f "${container}" >/dev/null 2>&1 || true
cid=$(
  docker run -d --name "${container}" --gpus all --ipc=host --network=host \
    --memory "${GB10_DOCKER_MEMORY}" --memory-swap "${GB10_DOCKER_MEMORY_SWAP}" \
    -w /work \
    -v "${REPO_ROOT}:/work" \
    -v "${REPO_ROOT}/third_party/flashinfer:/flashinfer-src" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -e TORCH_CUDA_ARCH_LIST=12.1a \
    -e FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR}" \
    -e FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a" \
    -e FLASHINFER_PREFILL_DEBUG_ONCE=1 \
    -e SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src \
    -e SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG=1 \
    -e PYTHONPATH=/work/python_sitecustomize:/work/third_party/sglang/python:/tmp/flashinfer-python-path \
    -e SGLANG_FLASHINFER_VOSPLIT=1 \
    -e SGLANG_GEMMA4_TRACE_GEOMETRY=1 \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    "${IMAGE}" \
    bash -lc '
      set -euo pipefail
      rm -rf /root/.cache/flashinfer "${FLASHINFER_CACHE_DIR}"
      mkdir -p /tmp/flashinfer-python-path
      ln -sfn /flashinfer-src/flashinfer /tmp/flashinfer-python-path/flashinfer

      python - <<'"'"'PY'"'"'
import hashlib
import importlib
import pathlib
from flashinfer.jit import env as jit_env

for name in ("sgl_kernel", "sgl_kernel.common_ops"):
    try:
        mod = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        print(f"binary_missing {name} {exc}", flush=True)
        continue
    path = pathlib.Path(getattr(mod, "__file__", "")).resolve()
    if path.is_file():
        print(f"binary_md5 {name} {path} {hashlib.md5(path.read_bytes()).hexdigest()}", flush=True)
print("flashinfer_data", jit_env.FLASHINFER_DATA, flush=True)
print("flashinfer_csrc", jit_env.FLASHINFER_CSRC_DIR, flush=True)
print("flashinfer_include", jit_env.FLASHINFER_INCLUDE_DIR, flush=True)
PY

      exec python3 -m sglang.launch_server \
        --model-path '"${MODEL}"' \
        --dtype bfloat16 \
        --attention-backend flashinfer \
        --page-size 1 \
        --mem-fraction-static '"${MEM_FRACTION_STATIC}"' \
        --disable-cuda-graph \
        --disable-piecewise-cuda-graph \
        --host 0.0.0.0 \
        --port '"${PORT}"'
    '
)
echo "${cid}" >"${cid_file}"

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 ||
    curl -fsS "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${container}"; then
    echo "container exited before readiness" >&2
    exit 1
  fi
  sleep 5
done

curl -sS --max-time 180 "http://127.0.0.1:${PORT}/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "In one short sentence, name the capital of Japan.",
    "sampling_params": {"temperature": 0, "max_new_tokens": 16}
  }' | tee "${OUT_DIR}/generate_raw.json"

curl -sS --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-4-E4B-it",
    "messages": [{"role": "user", "content": "In one short sentence, name the capital of Japan."}],
    "temperature": 0,
    "max_tokens": 16
  }' | tee "${OUT_DIR}/chat_openai.json"

capture_docker_logs || true

python3 - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
raw = json.loads((out / "generate_raw.json").read_text())
chat = json.loads((out / "chat_openai.json").read_text())
raw_text = json.dumps(raw, ensure_ascii=False)
chat_text = json.dumps(chat, ensure_ascii=False)
summary = [
    "# SGLang Gemma 4 E4B Chat Compare",
    "",
    f"Run: `{out.name}`",
    "",
    "## Raw `/generate`",
    "",
    "```json",
    json.dumps(raw, ensure_ascii=False, indent=2),
    "```",
    "",
    "## OpenAI `/v1/chat/completions`",
    "",
    "```json",
    json.dumps(chat, ensure_ascii=False, indent=2),
    "```",
    "",
    f"Raw contains Tokyo: `{'Tokyo' in raw_text or '東京' in raw_text}`",
    f"Chat contains Tokyo: `{'Tokyo' in chat_text or '東京' in chat_text}`",
]
(out / "summary.md").write_text("\\n".join(summary) + "\\n", encoding="utf-8")
print(out / "summary.md")
PY
