#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/run_sglang_spark_image_smoke.sh

Spark-only packaging smoke for the self-contained SGLang image. This script
does not build, compile, or source-overlay anything on Spark. It pulls the
published linux/arm64 image, records provenance, verifies in-container imports,
then starts one guarded SGLang server and records whether it reaches readiness.

Environment:
  IMAGE=ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack:epoch2-sglang-spark-u22-torch211-arm64
  MODEL=google/gemma-4-E2B-it
  SERVED_MODEL=sglang-spark-packaging-smoke
  PORT=30000
  MEM_FRACTION_STATIC=0.35
  ATTENTION_BACKEND=triton
  EXPECTED_TORCH_VERSION=2.11.0
  RESULTS_DIR=/home/jethac/dgx-spark-hijinks/results
  RUN_ID=sglang_spark_image_smoke_YYYYMMDDTHHMMSSJST
  GB10_DOCKER_MEMORY=100g
  GB10_DOCKER_MEMORY_SWAP=100g
  READY_TIMEOUT_S=900
  CLAUDE_MARKER=/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 2
fi

IMAGE=${IMAGE:-ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack:epoch2-sglang-spark-u22-torch211-arm64}
MODEL=${MODEL:-google/gemma-4-E2B-it}
SERVED_MODEL=${SERVED_MODEL:-sglang-spark-packaging-smoke}
PORT=${PORT:-30000}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.35}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-triton}
EXPECTED_TORCH_VERSION=${EXPECTED_TORCH_VERSION:-2.11.0}
RESULTS_DIR=${RESULTS_DIR:-/home/jethac/dgx-spark-hijinks/results}
RUN_ID=${RUN_ID:-sglang_spark_image_smoke_$(TZ=Asia/Tokyo date +%Y%m%dT%H%M%SJST)}
OUT_DIR=${OUT_DIR:-${RESULTS_DIR}/${RUN_ID}}
GB10_DOCKER_MEMORY=${GB10_DOCKER_MEMORY:-100g}
GB10_DOCKER_MEMORY_SWAP=${GB10_DOCKER_MEMORY_SWAP:-100g}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-900}
CLAUDE_MARKER=${CLAUDE_MARKER:-/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN}
PLATFORM=${PLATFORM:-linux/arm64}
CONTAINER=${CONTAINER:-${RUN_ID}}

if [[ -e "${CLAUDE_MARKER}" ]]; then
  echo "CLAUDE_WINDOW_OPEN present; yielding" >&2
  exit 99
fi

if [[ "$(docker ps -q | wc -l)" != "0" ]]; then
  echo "docker is not empty; yielding" >&2
  docker ps >&2
  exit 99
fi

mkdir -p "${OUT_DIR}"

summary="${OUT_DIR}/summary.md"
server_log="${OUT_DIR}/server.log"
request_json="${OUT_DIR}/request.json"

cleanup() {
  local status=$?
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    docker logs "${CONTAINER}" >"${server_log}" 2>&1 || true
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  fi
  docker ps >"${OUT_DIR}/docker_ps_after.txt" 2>&1 || true
  free -h >"${OUT_DIR}/free_after.txt" 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT

{
  echo "run_id=${RUN_ID}"
  echo "image=${IMAGE}"
  echo "platform=${PLATFORM}"
  echo "model=${MODEL}"
  echo "served_model=${SERVED_MODEL}"
  echo "port=${PORT}"
  echo "mem_fraction_static=${MEM_FRACTION_STATIC}"
  echo "attention_backend=${ATTENTION_BACKEND}"
  echo "started_at=$(TZ=Asia/Tokyo date -Is)"
  free -h
  docker ps
} >"${OUT_DIR}/preflight.txt" 2>&1

docker buildx imagetools inspect "${IMAGE}" >"${OUT_DIR}/imagetools.txt" 2>&1
docker pull --platform "${PLATFORM}" "${IMAGE}" >"${OUT_DIR}/docker_pull.log" 2>&1
docker image inspect "${IMAGE}" >"${OUT_DIR}/image_inspect.json" 2>&1

docker run --rm --platform "${PLATFORM}" \
  -v "${OUT_DIR}:/out" \
  -e EXPECTED_TORCH_VERSION="${EXPECTED_TORCH_VERSION}" \
  "${IMAGE}" \
  bash -lc 'set -euo pipefail
    cat /etc/os-release | tee /out/os-release.txt
    ldd --version >/tmp/ldd-version.txt
    sed -n "1p" /tmp/ldd-version.txt | tee /out/ldd-version.txt
    python3 - <<'"'"'PY'"'"' | tee /out/import_probe.txt
import importlib.metadata as md
import importlib.util
import os
import pathlib
import platform
import sys

import torch
import transformers

print("python", sys.version.replace("\n", " "))
print("machine", platform.machine())
print("torch", torch.__version__, torch.version.cuda)
expected_torch = os.environ.get("EXPECTED_TORCH_VERSION")
if expected_torch:
    assert torch.__version__.split("+", 1)[0] == expected_torch, torch.__version__
print("transformers", transformers.__version__)
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as mapping
except Exception:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING as mapping
print("gemma4_unified", "gemma4_unified" in mapping)
for name in ("sglang", "flashinfer", "sgl_kernel"):
    spec = importlib.util.find_spec(name)
    print(f"{name}_spec", spec.origin if spec else None)
print("sglang_version", md.version("sglang"))
print("flashinfer_python_version", md.version("flashinfer_python"))
print("sglang_kernel_version", md.version("sglang-kernel"))
cache_roots = [pathlib.Path("/root/.cache/flashinfer"), pathlib.Path("/tmp/flashinfer"), pathlib.Path("/tmp/flashinfer_modules")]
for root in cache_roots:
    payload = sorted(str(path) for path in root.rglob("*.so")) if root.exists() else []
    print("flashinfer_cache_payload", root, len(payload))
PY'

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
docker run -d --name "${CONTAINER}" --gpus all --ipc=host --network=host \
  --memory "${GB10_DOCKER_MEMORY}" --memory-swap "${GB10_DOCKER_MEMORY_SWAP}" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "${IMAGE}" \
  bash -lc "exec python3 -m sglang.launch_server \
    --model-path '${MODEL}' \
    --served-model-name '${SERVED_MODEL}' \
    --dtype bfloat16 \
    --attention-backend '${ATTENTION_BACKEND}' \
    --host 0.0.0.0 \
    --port '${PORT}' \
    --mem-fraction-static '${MEM_FRACTION_STATIC}' \
    --disable-cuda-graph \
    --disable-piecewise-cuda-graph" \
  >"${OUT_DIR}/docker_run.out" 2>"${OUT_DIR}/docker_run.err"

ready=0
attempts=$((READY_TIMEOUT_S / 5))
if (( attempts < 1 )); then attempts=1; fi
for _ in $(seq 1 "${attempts}"); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    break
  fi
  sleep 5
done

docker logs "${CONTAINER}" >"${server_log}" 2>&1 || true
echo "${ready}" >"${OUT_DIR}/ready.txt"

request_rc=0
if [[ "${ready}" == "1" ]]; then
  curl -fsS --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"In one short sentence, name the capital of Japan.\"}],\"temperature\":0,\"max_tokens\":24}" \
    >"${request_json}" || request_rc=$?
else
  request_rc=99
fi
echo "${request_rc}" >"${OUT_DIR}/request_rc.txt"

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

image_digest=$(docker image inspect "${IMAGE}" --format '{{index .RepoDigests 0}}' 2>/dev/null || true)
torch_line=$(grep '^torch ' "${OUT_DIR}/import_probe.txt" 2>/dev/null || true)
gemma_line=$(grep '^gemma4_unified ' "${OUT_DIR}/import_probe.txt" 2>/dev/null || true)

{
  echo "# SGLang Spark Image Smoke"
  echo
  echo "- run id: \`${RUN_ID}\`"
  echo "- image: \`${IMAGE}\`"
  echo "- pulled digest: \`${image_digest}\`"
  echo "- platform: \`${PLATFORM}\`"
  echo "- model: \`${MODEL}\`"
  echo "- attention backend: \`${ATTENTION_BACKEND}\`"
  echo "- torch: \`${torch_line}\`"
  echo "- gemma4 mapping: \`${gemma_line}\`"
  echo "- server ready: \`${ready}\`"
  echo "- request rc: \`${request_rc}\`"
  echo
  if [[ "${ready}" == "1" && "${request_rc}" == "0" ]]; then
    echo "Status: GREEN - image pulls, imports, and reaches SGLang readiness."
  else
    echo "Status: RED - see \`server.log\`, \`import_probe.txt\`, and pull artifacts."
  fi
} >"${summary}"

[[ "${ready}" == "1" && "${request_rc}" == "0" ]]
