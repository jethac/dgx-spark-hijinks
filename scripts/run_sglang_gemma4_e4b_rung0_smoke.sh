#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live}"
IMAGE="${IMAGE:-sglang-source-stack-c3dae30f-e631a13fd:latest}"
MODEL="${MODEL:-google/gemma-4-E4B-it}"
PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.40}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"
GB10_DOCKER_MEMORY="${GB10_DOCKER_MEMORY:-100g}"
GB10_DOCKER_MEMORY_SWAP="${GB10_DOCKER_MEMORY_SWAP:-100g}"
SGLANG_COMMIT="${SGLANG_COMMIT:-9d78a007f}"
FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-8d85fff9}"
RUN_ID="${RUN_ID:-sglang_gemma4_e4b_rung0_$(date +%Y%m%dT%H%M%SJST)}"
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
  echo "port=${PORT}"
  echo "mem_fraction_static=${MEM_FRACTION_STATIC}"
  echo "sglang_commit=${SGLANG_COMMIT}"
  echo "flashinfer_commit=${FLASHINFER_COMMIT}"
  echo "started_at=$(date -Is)"
  free -h
} | tee "${OUT_DIR}/preflight.log"

git fetch origin
git checkout docs/codex-direction-nvfp4-kv
git pull --ff-only
git -C third_party/sglang fetch origin
git -C third_party/sglang checkout "${SGLANG_COMMIT}"
git -C third_party/flashinfer fetch origin
git -C third_party/flashinfer checkout "${FLASHINFER_COMMIT}"
git -C third_party/flashinfer submodule update --init --recursive \
  3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog

{
  echo "parent_commit=$(git rev-parse HEAD)"
  echo "sglang_commit=$(git -C third_party/sglang rev-parse HEAD)"
  echo "flashinfer_commit=$(git -C third_party/flashinfer rev-parse HEAD)"
  git status --short
  git -C third_party/sglang status --short
  git -C third_party/flashinfer status --short
} | tee "${OUT_DIR}/checkout.log"

container="${RUN_ID}"
cid_file="${OUT_DIR}/container_id.txt"
server_log="${OUT_DIR}/server.log"
request_json="${OUT_DIR}/generate.json"
summary_md="${OUT_DIR}/summary.md"

capture_docker_logs() {
  local name="$1"
  local tmp_log="${server_log}.tmp"
  if docker logs "${name}" >"${tmp_log}" 2>&1; then
    mv "${tmp_log}" "${server_log}"
  else
    rm -f "${tmp_log}"
  fi
}

cleanup() {
  local status=$?
  if docker ps --format '{{.Names}}' | grep -qx "${container}"; then
    capture_docker_logs "${container}" || true
    docker rm -f "${container}" >/dev/null 2>&1 || true
  elif [[ -s "${cid_file}" ]]; then
    capture_docker_logs "$(cat "${cid_file}")" || true
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
print("flashinfer_cutlass", jit_env.CUTLASS_INCLUDE_DIRS, flush=True)
print("flashinfer_cccl", jit_env.CCCL_INCLUDE_DIRS, flush=True)
print("flashinfer_spdlog", jit_env.SPDLOG_INCLUDE_DIR, flush=True)
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

ready=0
attempts=$((READY_TIMEOUT_S / 5))
if (( attempts < 1 )); then
  attempts=1
fi
for _ in $(seq 1 "${attempts}"); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${container}"; then
    break
  fi
  sleep 5
done

capture_docker_logs "${container}" || true

if [[ "${ready}" != "1" ]]; then
  {
    echo "# SGLang Gemma 4 E4B Rung 0 Smoke"
    echo
    echo "Status: RED - server did not reach readiness."
    echo
    echo "- Run: \`${RUN_ID}\`"
    echo "- Server log: \`server.log\`"
    if grep -q "Unsupported max_mma_kv: 0" "${server_log}"; then
      echo "- First blocker: \`Unsupported max_mma_kv: 0\`"
    fi
  } >"${summary_md}"
  exit 1
fi

set +e
curl -sS --max-time "${REQUEST_TIMEOUT_S:-180}" "http://127.0.0.1:${PORT}/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "In one short sentence, name the capital of Japan.",
    "sampling_params": {
      "temperature": 0,
      "max_new_tokens": 16
    }
  }' | tee "${request_json}"
request_status=${PIPESTATUS[0]}
set -e
echo "${request_status}" >"${OUT_DIR}/request_status.txt"

capture_docker_logs "${container}" || true

python3 scripts/summarize_sglang_gemma4_e4b_rung0.py \
  --out-dir "${OUT_DIR}" \
  --run-id "${RUN_ID}" \
  --model "${MODEL}" \
  --sglang-commit "${SGLANG_COMMIT}" \
  --flashinfer-commit "${FLASHINFER_COMMIT}"
