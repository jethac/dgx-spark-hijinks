#!/usr/bin/env bash
# DG-R7 Spark packet: stock DiffusionGemma image prompt smoke.
#
# Scope: multimodal image prompt gate only. This uses the upstream stock
# DiffusionGemma policy path first: Triton attention, BF16/auto KV, eager
# execution. It does not claim FlashInfer, NVFP4, capacity, or throughput.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live}"
SOURCE_BRANCH="${SOURCE_BRANCH:-epoch2}"
IMAGE="${IMAGE:-sglang-source-stack-dgemma-024-0705924c-f99323bd:latest}"
MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
PORT="${PORT:-30125}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.55}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1200}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-1200}"
GB10_DOCKER_MEMORY="${GB10_DOCKER_MEMORY:-100g}"
GB10_DOCKER_MEMORY_SWAP="${GB10_DOCKER_MEMORY_SWAP:-100g}"
SGLANG_COMMIT="${SGLANG_COMMIT:-98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968}"
FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-f99323bd7d1c}"
RUN_ID="${RUN_ID:-sglang_dgemma_dgr7_image_smoke_$(TZ=Asia/Tokyo date +%Y%m%dT%H%M%SJST)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/${RUN_ID}}"
FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-/tmp/flashinfer-cache-${RUN_ID}}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
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
  echo "source_branch=${SOURCE_BRANCH}"
  echo "image=${IMAGE}"
  echo "model=${MODEL}"
  echo "port=${PORT}"
  echo "mem_fraction_static=${MEM_FRACTION_STATIC}"
  echo "sglang_commit=${SGLANG_COMMIT}"
  echo "flashinfer_commit=${FLASHINFER_COMMIT}"
  echo "scope=stock DiffusionGemma multimodal image prompt smoke"
  echo "kv_cache_dtype=auto"
  echo "attention_policy=stock DiffusionGemma Triton/eager"
  echo "started_at=$(TZ=Asia/Tokyo date -Is)"
  free -h
} | tee "${OUT_DIR}/preflight.log"

git fetch origin
git checkout "${SOURCE_BRANCH}"
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
quality_json="${OUT_DIR}/image_quality.json"
summary_md="${OUT_DIR}/summary.md"
dllm_config="${OUT_DIR}/dllm_config.yaml"

cat >"${dllm_config}" <<'EOF'
max_denoising_steps: 48
seed: 1234
sampler_config:
  entropy_bound: 0.1
temperature_schedule:
  t_min: 0.4
  t_max: 0.8
stopping_config:
  confidence_threshold: 0.005
  stability_threshold: 1
EOF

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
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -e MODEL="${MODEL}" \
    -e PORT="${PORT}" \
    -e MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
    -e TORCH_CUDA_ARCH_LIST=12.1a \
    -e FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR}" \
    -e FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a" \
    -e SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src \
    -e SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG=1 \
    -e PYTHONPATH=/work/python_sitecustomize:/work/third_party/sglang/python:/tmp/flashinfer-python-path \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
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
        --model-path "${MODEL}" \
        --dllm-algorithm Gemma4Renoise \
        --dllm-algorithm-config "/work/results/'"${RUN_ID}"'/dllm_config.yaml" \
        --trust-remote-code \
        --dtype bfloat16 \
        --context-length 8192 \
        --mem-fraction-static "${MEM_FRACTION_STATIC}" \
        --disable-cuda-graph \
        --disable-piecewise-cuda-graph \
        --host 0.0.0.0 \
        --port "${PORT}"
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
    echo "# SGLang DiffusionGemma DG-R7 Image Smoke"
    echo
    echo "Status: RED - server did not reach readiness."
    echo
    echo "- Run ID: \`${RUN_ID}\`"
    echo "- Scope: stock DiffusionGemma image prompt smoke"
  } >"${summary_md}"
  exit 1
fi

set +e
python3 scripts/diffusion_gemma_image_quality_client.py \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "${MODEL}" \
  --out "${quality_json}" \
  --repeats 2 \
  --ready-timeout-s "${READY_TIMEOUT_S}" \
  --request-timeout-s "${REQUEST_TIMEOUT_S}" \
  >"${OUT_DIR}/image_quality_stdout.json" 2>"${OUT_DIR}/image_quality_stderr.log"
quality_status=$?
set -e

capture_docker_logs "${container}" || true
docker rm -f "${container}" >/dev/null 2>&1 || true
docker ps >"${OUT_DIR}/docker_ps_after.txt" 2>&1 || true
free -h >"${OUT_DIR}/free_after.txt" 2>&1 || true

python3 - "$quality_json" "$server_log" "$summary_md" "$RUN_ID" "$SGLANG_COMMIT" "$FLASHINFER_COMMIT" "$IMAGE" "$quality_status" <<'PY'
import json
import sys
from pathlib import Path

quality_path = Path(sys.argv[1])
server_log = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
run_id, sglang_commit, flashinfer_commit, image, quality_status = sys.argv[4:9]
quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
log_text = server_log.read_text(errors="replace") if server_log.exists() else ""

all_ok = bool(quality.get("all_ok")) and quality_status == "0"
stock_policy = "Attention backend forced to triton for DiffusionGemma" in log_text
has_image_warning = "image" in log_text.lower()
status = "GREEN" if all_ok and stock_policy else "RED"

lines = [
    "# SGLang DiffusionGemma DG-R7 Image Smoke",
    "",
    f"Status: {status}",
    "",
    "## Scope",
    "",
    "Stock DiffusionGemma multimodal image prompt smoke on GB10. This row uses",
    "the upstream policy path: Triton attention, BF16/auto KV, eager execution,",
    "and unchunked prefill. It does not claim FlashInfer, NVFP4, capacity,",
    "throughput, or image-generation quality.",
    "",
    "## Provenance",
    "",
    f"- Run ID: `{run_id}`",
    f"- Image: `{image}`",
    f"- SGLang commit: `{sglang_commit}`",
    f"- FlashInfer source commit: `{flashinfer_commit}`",
    "- Model: `google/diffusiongemma-26B-A4B-it`",
    "- Request API: `/v1/chat/completions` with OpenAI `image_url` content",
    "",
    "## Gates",
    "",
    f"- server reached readiness: `true`",
    f"- stock Triton policy proof: `{stock_policy}`",
    f"- image quality client status: `{quality_status}`",
    f"- semantic/stability gate: `{quality.get('all_ok')}`",
    f"- server log contains image-related diagnostics: `{has_image_warning}`",
    "",
    "## Checks",
    "",
]
for check in quality.get("checks", []):
    lines.append(
        "- `{image_id}` sha256 `{sha}`: stable `{stable}`, non_empty `{non_empty}`, "
        "answer_ok `{answer_ok}`, texts `{texts}`".format(
            image_id=check.get("image_id"),
            sha=check.get("sha256"),
            stable=check.get("stable"),
            non_empty=check.get("non_empty"),
            answer_ok=check.get("answer_ok"),
            texts=check.get("texts"),
        )
    )
lines.extend(
    [
        "",
        "## Decision",
        "",
    ]
)
if status == "GREEN":
    lines.extend(
        [
            "The stock DiffusionGemma multimodal request path is live for the",
            "synthetic color-image gate. This closes the live image-prompt caveat",
            "only for a tiny deterministic color-recognition smoke; it is not a",
            "general vision benchmark.",
        ]
    )
else:
    lines.extend(
        [
            "The stock DiffusionGemma image-prompt path is not claim-grade under",
            "this gate. Treat this as the next diagnostic row before any",
            "multimodal serving claim.",
        ]
    )

summary_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
raise SystemExit(0 if status == "GREEN" else 1)
PY
