#!/usr/bin/env bash
set -euo pipefail

RUN_ID=${RUN_ID:-vllm_qwen_nvfp4_prefixcache_on_manual}
IMAGE=${IMAGE:-jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv}
MODEL=${MODEL:-/home/jethac/models/aeon/qwen36-nvfp4}
SERVED_MODEL=${SERVED_MODEL:-qwen36-fast}
PORT=${PORT:-8001}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-nvfp4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.72}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
GB10_DOCKER_MEMORY=${GB10_DOCKER_MEMORY:-100g}
GB10_DOCKER_MEMORY_SWAP=${GB10_DOCKER_MEMORY_SWAP:-100g}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-1800}
VLLM_SPARK_KV_TRACE=${VLLM_SPARK_KV_TRACE:-0}
VLLM_SPARK_KV_TRACE_FILE=${VLLM_SPARK_KV_TRACE_FILE:-/work/results/${RUN_ID}_kv_trace.jsonl}
VLLM_SPARK_KV_TRACE_LAYERS=${VLLM_SPARK_KV_TRACE_LAYERS:-}
VLLM_SPARK_KV_TRACE_LIMIT=${VLLM_SPARK_KV_TRACE_LIMIT:-4}
VLLM_SPARK_KV_TRACE_VALUES=${VLLM_SPARK_KV_TRACE_VALUES:-8}

mkdir -p "${RESULTS_DIR}"

container="${RUN_ID}"
server_before="${RESULTS_DIR}/${RUN_ID}_server_before_probe.log"
server_after="${RESULTS_DIR}/${RUN_ID}_server_after_probe.log"
inspect_json="${RESULTS_DIR}/${RUN_ID}_container_inspect.json"
probe_json="${RESULTS_DIR}/${RUN_ID}_prefix_reuse_probe.json"
metrics_after="${RESULTS_DIR}/${RUN_ID}_metrics_after_probe.txt"

docker rm -f "${container}" >/dev/null 2>&1 || true

docker run -d --name "${container}" --gpus all --net host --ipc host \
  --memory "${GB10_DOCKER_MEMORY}" --memory-swap "${GB10_DOCKER_MEMORY_SWAP}" \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_SPARK_KV_TRACE="${VLLM_SPARK_KV_TRACE}" \
  -e VLLM_SPARK_KV_TRACE_FILE="${VLLM_SPARK_KV_TRACE_FILE}" \
  -e VLLM_SPARK_KV_TRACE_LAYERS="${VLLM_SPARK_KV_TRACE_LAYERS}" \
  -e VLLM_SPARK_KV_TRACE_LIMIT="${VLLM_SPARK_KV_TRACE_LIMIT}" \
  -e VLLM_SPARK_KV_TRACE_VALUES="${VLLM_SPARK_KV_TRACE_VALUES}" \
  -v "${MODEL}:/models/target:ro" \
  -v "${REPO_ROOT}:/work" -w /work \
  "${IMAGE}" \
  bash -lc "exec vllm serve /models/target \
    --served-model-name qwen36-35b-heretic qwen36-fast qwen36-deep \
    --host 0.0.0.0 --port ${PORT} \
    --trust-remote-code --max-model-len ${MAX_MODEL_LEN} \
    --quantization compressed-tensors --load-format safetensors \
    --attention-backend flashinfer --kv-cache-dtype ${KV_CACHE_DTYPE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} --max-num-seqs ${MAX_NUM_SEQS} \
    --enable-chunked-prefill --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3" \
  >/dev/null

cleanup() {
  docker logs "${container}" >"${server_after}" 2>&1 || true
  docker inspect "${container}" >"${inspect_json}" 2>/dev/null || true
  docker rm -f "${container}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 $((READY_TIMEOUT_S / 2))); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${container}"; then
    break
  fi
  sleep 2
done

docker logs "${container}" >"${server_before}" 2>&1 || true

if [[ "${ready}" != "1" ]]; then
  echo "server not ready for ${RUN_ID}" >&2
  exit 2
fi

probe_status=0
python3 "${REPO_ROOT}/scripts/vllm_prefix_cache_reuse_probe.py" \
  --url "http://127.0.0.1:${PORT}" \
  --model "${SERVED_MODEL}" \
  --run-id "${RUN_ID}" \
  --case long_shared_prefix \
  --requests 2 \
  --max-tokens 1 \
  --top-logprobs 20 \
  --chat-template-kwargs-json '{"enable_thinking": false}' \
  --output "${probe_json}" || probe_status=$?

curl -fsS "http://127.0.0.1:${PORT}/metrics" >"${metrics_after}" 2>/dev/null || true

exit "${probe_status}"
