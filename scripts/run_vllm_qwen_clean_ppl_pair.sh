#!/usr/bin/env bash
set -euo pipefail

CTX=${CTX:-32768}
STAMP=${STAMP:-$(date +%Y%m%dT%H%MJST)}
RUN=${RUN:-vllm_qwen_clean_ppl_${STAMP}_ctx${CTX}}
ROOT=${ROOT:-/home/jethac/spark_tmp/${RUN}}
CORPUS=${CORPUS:-/home/jethac/spark_tmp/qwen_ppl_corpus_20260609/results/vllm_qwen_clean_ppl_20260609T0850JST_corpus.md}
CORPUS_MANIFEST=${CORPUS_MANIFEST:-/home/jethac/spark_tmp/qwen_ppl_corpus_20260609/results/vllm_qwen_clean_ppl_20260609T0850JST_corpus_manifest.json}
IMAGE=${IMAGE:-jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv}
MODEL=${MODEL:-/home/jethac/models/aeon/qwen36-nvfp4}
SERVED=${SERVED:-qwen36-fast}
REPO=${REPO:-/home/jethac/dgx-spark-hijinks}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.72}
GB10_DOCKER_MEMORY=${GB10_DOCKER_MEMORY:-100g}
GB10_DOCKER_MEMORY_SWAP=${GB10_DOCKER_MEMORY_SWAP:-100g}

mkdir -p "${ROOT}/scripts" "${ROOT}/docs" "${ROOT}/results"
cp "${REPO}/scripts/vllm_prompt_ppl_sweep.py" "${ROOT}/scripts/"
cp "${REPO}/scripts/spark_hardware.py" "${ROOT}/scripts/"
cp "${CORPUS}" "${ROOT}/docs/ppl_corpus.md"
cp "${CORPUS_MANIFEST}" "${ROOT}/results/${RUN}_corpus_manifest.json"

wait_ready() {
  local name=$1
  for _ in $(seq 1 120); do
    if docker exec -i "${name}" python3 - <<'PY' >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2).read()
PY
    then
      return 0
    fi
    sleep 5
  done
  return 1
}

run_one() {
  local kv=$1
  local name=${RUN}_${kv}
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker run -d --name "${name}" --gpus all --net host --ipc host \
    --memory "${GB10_DOCKER_MEMORY}" --memory-swap "${GB10_DOCKER_MEMORY_SWAP}" \
    -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
    -v "${MODEL}:/models/target:ro" \
    -v "${ROOT}:/work" -w /work \
    "${IMAGE}" \
    vllm serve /models/target \
      --served-model-name qwen36-35b-heretic qwen36-fast qwen36-deep \
      --host 0.0.0.0 --port 8000 \
      --trust-remote-code --max-model-len 262144 \
      --quantization compressed-tensors --load-format safetensors \
      --attention-backend flashinfer --kv-cache-dtype "${kv}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" --max-num-batched-tokens 65536 --max-num-seqs 128 \
      --enable-chunked-prefill --no-enable-prefix-caching \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 >/dev/null
  if ! wait_ready "${name}"; then
    docker logs "${name}" > "${ROOT}/results/${RUN}_${kv}_server_startup_failed.log" 2>&1 || true
    docker inspect "${name}" > "${ROOT}/results/${RUN}_${kv}_inspect_failed.json" 2>&1 || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
    return 2
  fi
  docker logs "${name}" > "${ROOT}/results/${RUN}_${kv}_server_before_ppl.log" 2>&1 || true
  docker exec "${name}" python3 scripts/vllm_prompt_ppl_sweep.py \
    --url http://127.0.0.1:8000 \
    --model "${SERVED}" \
    --tokenizer /models/target \
    --text-file docs/ppl_corpus.md \
    --ctx "${CTX}" \
    --run-id "${RUN}_${kv}_ctx${CTX}" \
    --kv-cache-dtype "${kv}" \
    --runtime-ref "jethac/vllm@a919d635d + jethac/flashinfer@e152cf4d clean full-attention PPL; VLLM_TEST_FORCE_FP8_MARLIN=1" \
    --container-image "${IMAGE}" \
    --output "results/${RUN}_${kv}_ctx${CTX}_ppl.json" \
    > "${ROOT}/results/${RUN}_${kv}_ctx${CTX}_stdout.json" \
    2> "${ROOT}/results/${RUN}_${kv}_ctx${CTX}_stderr.log"
  docker logs "${name}" > "${ROOT}/results/${RUN}_${kv}_server_after_ppl.log" 2>&1 || true
  docker rm -f "${name}" >/dev/null 2>&1 || true
}

run_one fp8
run_one nvfp4

python3 "${ROOT}/scripts/vllm_prompt_ppl_sweep.py" \
  --compare-fp8 "${ROOT}/results/${RUN}_fp8_ctx${CTX}_ppl.json" \
  --compare-nvfp4 "${ROOT}/results/${RUN}_nvfp4_ctx${CTX}_ppl.json" \
  --output "${ROOT}/results/${RUN}_compare.json"

printf "%s\n" "${RUN}" > "${ROOT}/RUN_ID"
printf "ROOT=%s\nRUN=%s\n" "${ROOT}" "${RUN}"
