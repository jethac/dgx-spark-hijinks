#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/run_aeon_vllm_reproduction.sh TARGET RUN_ID

TARGET:
  gemma26-dflash  AEON Gemma 4 26B A4B NVFP4 + DFlash
  qwen36-dflash   AEON Qwen3.6 35B A3B NVFP4 + DFlash

Environment:
  MODELS_ROOT=/opt/spark-models/aeon   model cache root
  RESULTS_DIR=results                  artifact output directory
  PORT=8000                            host/container port
  DOWNLOAD=0                           set to 1 to hf-download missing weights
  RECORD=0                             set to 1 to run smoke/benchmark manifest
  WAIT_TIMEOUT=900                     seconds to wait for /health
  DOCKER_PULL=0                        set to 1 to docker pull before launch
  DOCKER_PLATFORM=linux/arm64          platform passed to docker pull
  PROCESS_MATCH=vllm                   runtime process probe match string
  HF_CLI=hf                            Hugging Face CLI command override
  RECORD_PYTHON=python3                Python used for RECORD=1 artifact capture
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

TARGET=$1
RUN_ID=$2
MODELS_ROOT=${MODELS_ROOT:-/opt/spark-models/aeon}
RESULTS_DIR=${RESULTS_DIR:-results}
PORT=${PORT:-8000}
DOWNLOAD=${DOWNLOAD:-0}
RECORD=${RECORD:-0}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-900}
DOCKER_PULL=${DOCKER_PULL:-0}
DOCKER_PLATFORM=${DOCKER_PLATFORM:-linux/arm64}
PROCESS_MATCH=${PROCESS_MATCH:-vllm}
RECORD_PYTHON=${RECORD_PYTHON:-python3}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "${RESULTS_DIR}"

COMMON_ENV=(
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  -e TORCH_MATMUL_PRECISION=high
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

case "${TARGET}" in
  gemma26-dflash)
    IMAGE=${IMAGE:-ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2}
    MODEL_REPO=${MODEL_REPO:-AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4}
    DRAFTER_REPO=${DRAFTER_REPO:-z-lab/gemma-4-26B-A4B-it-DFlash}
    MODEL_DIR=${MODEL_DIR:-${MODELS_ROOT}/gemma4}
    DRAFTER_DIR=${DRAFTER_DIR:-${MODELS_ROOT}/gemma4-dflash}
    SERVED_MODEL=${SERVED_MODEL:-gemma4-fast}
    CONTAINER_MODEL_ALIASES=(gemma4-aeon-uncensored gemma4-fast gemma4-deep)
    EXTRA_ENV=(
      -e VLLM_USE_FLASHINFER_MOE_FP4=0
      -e VLLM_TEST_FORCE_FP8_MARLIN=0
      -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
      -e VLLM_USE_FLASHINFER_SAMPLER=1
    )
    SERVE_ARGS=(
      vllm
      serve
      /models/target
      --served-model-name "${CONTAINER_MODEL_ALIASES[@]}"
      --host 0.0.0.0
      --port "${PORT}"
      --tensor-parallel-size 1
      --dtype auto
      --max-model-len 262144
      --max-num-seqs 64
      --max-num-batched-tokens 32768
      --gpu-memory-utilization 0.80
      --enable-chunked-prefill
      --enable-prefix-caching
      --trust-remote-code
      --enable-auto-tool-choice
      --tool-call-parser gemma4
      --speculative-config '{"method":"dflash","model":"/models/drafter","num_speculative_tokens":15,"attention_backend":"flash_attn"}'
    )
    QUANTIZATION=compressed-tensors-nvfp4
    ATTENTION_BACKEND=triton-target-flashattn-drafter
    ;;
  qwen36-dflash)
    IMAGE=${IMAGE:-ghcr.io/aeon-7/vllm-spark-omni-q36:v2}
    MODEL_REPO=${MODEL_REPO:-AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4}
    DRAFTER_REPO=${DRAFTER_REPO:-z-lab/Qwen3.6-35B-A3B-DFlash}
    MODEL_DIR=${MODEL_DIR:-${MODELS_ROOT}/qwen36-nvfp4}
    DRAFTER_DIR=${DRAFTER_DIR:-${MODELS_ROOT}/qwen36-dflash}
    SERVED_MODEL=${SERVED_MODEL:-qwen36-fast}
    CONTAINER_MODEL_ALIASES=(qwen36-35b-heretic qwen36-fast qwen36-deep)
    EXTRA_ENV=(
      -e NVIDIA_FORWARD_COMPAT=1
      -e VLLM_TEST_FORCE_FP8_MARLIN=1
    )
    SERVE_ARGS=(
      vllm
      serve
      /models/target
      --served-model-name "${CONTAINER_MODEL_ALIASES[@]}"
      --host 0.0.0.0
      --port "${PORT}"
      --tensor-parallel-size 1
      --dtype auto
      --quantization compressed-tensors
      --max-model-len 262144
      --max-num-seqs 128
      --max-num-batched-tokens 65536
      --gpu-memory-utilization 0.85
      --enable-chunked-prefill
      --enable-prefix-caching
      --load-format safetensors
      --trust-remote-code
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      --reasoning-parser qwen3
      --speculative-config '{"method":"dflash","model":"/models/drafter","num_speculative_tokens":15}'
      --attention-backend flash_attn
    )
    QUANTIZATION=compressed-tensors-nvfp4
    ATTENTION_BACKEND=flash_attn
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ "${DOWNLOAD}" == "1" ]]; then
  if [[ -z "${HF_CLI:-}" ]]; then
    if command -v hf >/dev/null 2>&1; then
      HF_CLI=hf
    elif command -v huggingface-cli >/dev/null 2>&1; then
      HF_CLI=huggingface-cli
    else
      echo "DOWNLOAD=1 requires hf, huggingface-cli, or HF_CLI=/path/to/cli" >&2
      exit 5
    fi
  fi
  mkdir -p "${MODEL_DIR}" "${DRAFTER_DIR}"
  "${HF_CLI}" download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"
  "${HF_CLI}" download "${DRAFTER_REPO}" --local-dir "${DRAFTER_DIR}"
fi

if [[ ! -d "${MODEL_DIR}" || ! -d "${DRAFTER_DIR}" ]]; then
  echo "missing model directories; set DOWNLOAD=1 or populate:" >&2
  echo "  MODEL_DIR=${MODEL_DIR}" >&2
  echo "  DRAFTER_DIR=${DRAFTER_DIR}" >&2
  exit 3
fi

if [[ "${DOCKER_PULL}" == "1" ]]; then
  docker pull --platform "${DOCKER_PLATFORM}" "${IMAGE}"
fi

docker rm -f "${RUN_ID}" >/dev/null 2>&1 || true

docker run -d --gpus all --ipc=host --network=host \
  --name "${RUN_ID}" \
  "${COMMON_ENV[@]}" \
  "${EXTRA_ENV[@]}" \
  -v "${MODEL_DIR}:/models/target:ro" \
  -v "${DRAFTER_DIR}:/models/drafter:ro" \
  "${IMAGE}" \
  "${SERVE_ARGS[@]}"

STARTED=$(date +%s)
while true; do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  NOW=$(date +%s)
  if (( NOW - STARTED > WAIT_TIMEOUT )); then
    docker logs "${RUN_ID}" > "${RESULTS_DIR}/${RUN_ID}_server.log" 2>&1 || true
    echo "server did not become healthy within ${WAIT_TIMEOUT}s" >&2
    exit 4
  fi
  sleep 5
done

docker logs "${RUN_ID}" > "${RESULTS_DIR}/${RUN_ID}_server.log" 2>&1 || true

if [[ "${RECORD}" == "1" ]]; then
  "${RECORD_PYTHON}" "${REPO_ROOT}/scripts/record_openai_serving_row.py" \
    --backend vllm \
    --phase exploratory \
    --run-id "${RUN_ID}" \
    --url "http://127.0.0.1:${PORT}" \
    --model "${SERVED_MODEL}" \
    --results-dir "${RESULTS_DIR}" \
    --runtime-ref "${IMAGE}" \
    --container-image "${IMAGE}" \
    --quantization "${QUANTIZATION}" \
    --kv-cache-dtype auto \
    --attention-backend "${ATTENTION_BACKEND}" \
    --cuda-graph-mode enabled \
    --server-log "${RESULTS_DIR}/${RUN_ID}_server.log" \
    --process-match "${PROCESS_MATCH}"
  docker logs "${RUN_ID}" > "${RESULTS_DIR}/${RUN_ID}_server.log" 2>&1 || true
fi

echo "server ready: http://127.0.0.1:${PORT}"
echo "served model: ${SERVED_MODEL}"
echo "container: ${RUN_ID}"
