#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/prep_vllm_gemma3_27b_rung1.sh VLLM_SRC FLASHINFER_SRC HF_CACHE RESULTS_DIR

Prints the exact source-overlay command packet for the vLLM Gemma 3 27B Rung 1
fp8-vs-NVFP4 KV live row. This script is prep-only: it does not call docker,
does not touch CUDA, and does not start serving.

Environment:
  IMAGE=jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass
  STAMP=YYYYMMDDTHHMMJST
  GPU_MEMORY_UTILIZATION=0.85
  MAX_MODEL_LEN=131072
  MAX_NUM_BATCHED_TOKENS=4096
  VLLM_SOURCE_COMMIT=3658ba7123c3eb2211f18a882af1b993112fadb1
  VLLM_PRECOMPILED_WHEEL_COMMIT=8916796bc50926fd61e606718b194a71e2e31a24
EOF
}

if [[ $# -ne 4 ]]; then
  usage
  exit 2
fi

VLLM_SRC=$1
FLASHINFER_SRC=$2
HF_CACHE=$3
RESULTS_DIR=$4

IMAGE=${IMAGE:-jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass}
STAMP=${STAMP:-$(date +%Y%m%dT%H%MJST)}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
VLLM_SOURCE_COMMIT=${VLLM_SOURCE_COMMIT:-3658ba7123c3eb2211f18a882af1b993112fadb1}
VLLM_PRECOMPILED_WHEEL_COMMIT=${VLLM_PRECOMPILED_WHEEL_COMMIT:-8916796bc50926fd61e606718b194a71e2e31a24}

MODEL=google/gemma-3-27b-it
SERVED_MODEL=gemma3-27b-it
PREFIX=vllm_gemma3_27b_rung1_${STAMP}
FP8_RUN=${PREFIX}_fp8_flashinfer
NVFP4_RUN=${PREFIX}_nvfp4_kv_flashinfer
REPO_IN_CONTAINER=/workspace/dgx-spark-hijinks

cat <<EOF
# Prep-only command packet for ${PREFIX}
# This file is intentionally printed by the prep script. Review it, then run
# one server row at a time on the Spark host when the GPU is available.
set -euo pipefail

export VLLM_SRC=${VLLM_SRC}
export FLASHINFER_SRC=${FLASHINFER_SRC}
export HF_CACHE=${HF_CACHE}
export RESULTS_DIR=${RESULTS_DIR}
export IMAGE=${IMAGE}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
export MAX_MODEL_LEN=${MAX_MODEL_LEN}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
export VLLM_SOURCE_COMMIT=${VLLM_SOURCE_COMMIT}
export VLLM_PRECOMPILED_WHEEL_COMMIT=${VLLM_PRECOMPILED_WHEEL_COMMIT}

mkdir -p "\${RESULTS_DIR}"

wait_for_vllm() {
  local run_name=\$1
  local log_path=\$2
  local deadline=\$((SECONDS + 1800))
  until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do
    if ! docker inspect -f '{{.State.Running}}' "\${run_name}" 2>/dev/null | grep -q '^true$'; then
      echo "Container \${run_name} exited before readiness. Last log lines:" >&2
      tail -80 "\${log_path}" >&2 || true
      return 1
    fi
    if (( SECONDS > deadline )); then
      echo "Timed out waiting for \${run_name}. Last log lines:" >&2
      tail -120 "\${log_path}" >&2 || true
      return 1
    fi
    sleep 10
  done
}

stop_vllm_container() {
  local run_name=\$1
  docker rm -f "\${run_name}" >/dev/null 2>&1 || true
}

trap 'stop_vllm_container ${FP8_RUN}; stop_vllm_container ${NVFP4_RUN}' EXIT

# Start the fp8 comparator row. Stop this container before starting the NVFP4 row.
docker rm -f ${FP8_RUN} >/dev/null 2>&1 || true
docker run -d --gpus all --ipc=host --network=host \\
  --name ${FP8_RUN} \\
  -e HF_TOKEN \\
  -e VLLM_USE_V1=1 \\
  -e VLLM_LOGGING_LEVEL=DEBUG \\
  -e VLLM_SPARK_KV_GEOMETRY_LOG=1 \\
  -e SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src \\
  -e FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1 \\
  -e TORCH_CUDA_ARCH_LIST=12.1a \\
  -e CUDA_MODULE_LOADING=LAZY \\
  -v "\${VLLM_SRC}:/vllm-src" \\
  -v "\${FLASHINFER_SRC}:/flashinfer-src" \\
  -v "\${HF_CACHE}:/root/.cache/huggingface" \\
  -v "\${RESULTS_DIR}:/results" \\
  -v "\$(pwd):${REPO_IN_CONTAINER}" \\
  --entrypoint bash \\
  "\${IMAGE}" \\
  -lc '
set -euo pipefail
mkdir -p /tmp/spark-sitecustomize
cp ${REPO_IN_CONTAINER}/scripts/flashinfer_source_sitecustomize.py /tmp/spark-sitecustomize/sitecustomize.py
export PYTHONPATH="/tmp/spark-sitecustomize:\${PYTHONPATH:-}"
python3 -m pip install -q setuptools-rust > /results/${FP8_RUN}_pip_bootstrap.log 2>&1
cd /vllm-src
VLLM_USE_PRECOMPILED=1 VLLM_MAIN_CUDA_VERSION=13.0 VLLM_PRECOMPILED_WHEEL_COMMIT='"${VLLM_PRECOMPILED_WHEEL_COMMIT}"' \\
  python3 -m pip install --no-build-isolation -e . > /results/${FP8_RUN}_editable_install.log 2>&1
python3 - <<'"'"'PY'"'"' > /results/${FP8_RUN}_import_probe.txt 2>&1
import json, torch, transformers, vllm, flashinfer
print(json.dumps({
  "vllm": getattr(vllm, "__version__", None),
  "vllm_file": getattr(vllm, "__file__", None),
  "flashinfer": getattr(flashinfer, "__version__", None),
  "flashinfer_file": getattr(flashinfer, "__file__", None),
  "transformers": getattr(transformers, "__version__", None),
  "torch": torch.__version__,
  "torch_cuda": torch.version.cuda,
  "device": torch.cuda.get_device_name(0),
  "capability": torch.cuda.get_device_capability(0),
}, indent=2, sort_keys=True))
PY
exec vllm serve ${MODEL} \\
  --served-model-name ${SERVED_MODEL} \\
  --dtype bfloat16 \\
  --kv-cache-dtype fp8 \\
  --attention-backend flashinfer \\
  --max-model-len '"${MAX_MODEL_LEN}"' \\
  --gpu-memory-utilization '"${GPU_MEMORY_UTILIZATION}"' \\
  --max-num-batched-tokens '"${MAX_NUM_BATCHED_TOKENS}"' \\
  --disable-log-requests \\
  --host 0.0.0.0 \\
  --port 8000
'
docker inspect -f '{{.Id}}' ${FP8_RUN} > "\${RESULTS_DIR}/${FP8_RUN}_container_id.txt"
docker logs -f ${FP8_RUN} > "\${RESULTS_DIR}/${FP8_RUN}_server.log" 2>&1 &
echo "\$!" > "\${RESULTS_DIR}/${FP8_RUN}_docker_logs_pid.txt"
wait_for_vllm ${FP8_RUN} "\${RESULTS_DIR}/${FP8_RUN}_server.log"

python scripts/record_openai_serving_row.py \\
  --backend vllm --phase before --run-id ${FP8_RUN} \\
  --url http://127.0.0.1:8000 --model ${SERVED_MODEL} \\
  --results-dir "\${RESULTS_DIR}" \\
  --runtime-ref "jethac/vllm@${VLLM_SOURCE_COMMIT} + jethac/flashinfer@e152cf4da4ab2a9d093b7d9d4b499198b0211c61 source overlay; precompiled wheel base ${VLLM_PRECOMPILED_WHEEL_COMMIT}" \\
  --container-image "\${IMAGE}" \\
  --kv-cache-dtype fp8 --attention-backend flashinfer \\
  --cuda-graph-mode default \\
  --server-log "\${RESULTS_DIR}/${FP8_RUN}_server.log" \\
  --process-match "vllm serve ${MODEL}"

python scripts/openai_quality_probe.py \\
  --input-report "\${RESULTS_DIR}/${FP8_RUN}_openai_benchmark.json" \\
  --run-id ${FP8_RUN}_quality_from_benchmark \\
  --output "\${RESULTS_DIR}/${FP8_RUN}_quality.json"

stop_vllm_container ${FP8_RUN}

# Start the NVFP4-KV candidate row with the same geometry and serving settings.
docker rm -f ${NVFP4_RUN} >/dev/null 2>&1 || true
docker run -d --gpus all --ipc=host --network=host \\
  --name ${NVFP4_RUN} \\
  -e HF_TOKEN \\
  -e VLLM_USE_V1=1 \\
  -e VLLM_LOGGING_LEVEL=DEBUG \\
  -e VLLM_SPARK_KV_GEOMETRY_LOG=1 \\
  -e SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src \\
  -e FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1 \\
  -e TORCH_CUDA_ARCH_LIST=12.1a \\
  -e CUDA_MODULE_LOADING=LAZY \\
  -v "\${VLLM_SRC}:/vllm-src" \\
  -v "\${FLASHINFER_SRC}:/flashinfer-src" \\
  -v "\${HF_CACHE}:/root/.cache/huggingface" \\
  -v "\${RESULTS_DIR}:/results" \\
  -v "\$(pwd):${REPO_IN_CONTAINER}" \\
  --entrypoint bash \\
  "\${IMAGE}" \\
  -lc '
set -euo pipefail
mkdir -p /tmp/spark-sitecustomize
cp ${REPO_IN_CONTAINER}/scripts/flashinfer_source_sitecustomize.py /tmp/spark-sitecustomize/sitecustomize.py
export PYTHONPATH="/tmp/spark-sitecustomize:\${PYTHONPATH:-}"
python3 -m pip install -q setuptools-rust > /results/${NVFP4_RUN}_pip_bootstrap.log 2>&1
cd /vllm-src
VLLM_USE_PRECOMPILED=1 VLLM_MAIN_CUDA_VERSION=13.0 VLLM_PRECOMPILED_WHEEL_COMMIT='"${VLLM_PRECOMPILED_WHEEL_COMMIT}"' \\
  python3 -m pip install --no-build-isolation -e . > /results/${NVFP4_RUN}_editable_install.log 2>&1
python3 - <<'"'"'PY'"'"' > /results/${NVFP4_RUN}_import_probe.txt 2>&1
import json, torch, transformers, vllm, flashinfer
print(json.dumps({
  "vllm": getattr(vllm, "__version__", None),
  "vllm_file": getattr(vllm, "__file__", None),
  "flashinfer": getattr(flashinfer, "__version__", None),
  "flashinfer_file": getattr(flashinfer, "__file__", None),
  "transformers": getattr(transformers, "__version__", None),
  "torch": torch.__version__,
  "torch_cuda": torch.version.cuda,
  "device": torch.cuda.get_device_name(0),
  "capability": torch.cuda.get_device_capability(0),
}, indent=2, sort_keys=True))
PY
exec vllm serve ${MODEL} \\
  --served-model-name ${SERVED_MODEL} \\
  --dtype bfloat16 \\
  --kv-cache-dtype nvfp4 \\
  --attention-backend flashinfer \\
  --max-model-len '"${MAX_MODEL_LEN}"' \\
  --gpu-memory-utilization '"${GPU_MEMORY_UTILIZATION}"' \\
  --max-num-batched-tokens '"${MAX_NUM_BATCHED_TOKENS}"' \\
  --disable-log-requests \\
  --host 0.0.0.0 \\
  --port 8000
'
docker inspect -f '{{.Id}}' ${NVFP4_RUN} > "\${RESULTS_DIR}/${NVFP4_RUN}_container_id.txt"
docker logs -f ${NVFP4_RUN} > "\${RESULTS_DIR}/${NVFP4_RUN}_server.log" 2>&1 &
echo "\$!" > "\${RESULTS_DIR}/${NVFP4_RUN}_docker_logs_pid.txt"
wait_for_vllm ${NVFP4_RUN} "\${RESULTS_DIR}/${NVFP4_RUN}_server.log"

python scripts/record_openai_serving_row.py \\
  --backend vllm --phase after --run-id ${NVFP4_RUN} \\
  --url http://127.0.0.1:8000 --model ${SERVED_MODEL} \\
  --results-dir "\${RESULTS_DIR}" \\
  --runtime-ref "jethac/vllm@${VLLM_SOURCE_COMMIT} + jethac/flashinfer@e152cf4da4ab2a9d093b7d9d4b499198b0211c61 source overlay; precompiled wheel base ${VLLM_PRECOMPILED_WHEEL_COMMIT}" \\
  --container-image "\${IMAGE}" \\
  --kv-cache-dtype nvfp4 --attention-backend flashinfer \\
  --cuda-graph-mode default \\
  --server-log "\${RESULTS_DIR}/${NVFP4_RUN}_server.log" \\
  --process-match "vllm serve ${MODEL}"

python scripts/openai_quality_probe.py \\
  --input-report "\${RESULTS_DIR}/${NVFP4_RUN}_openai_benchmark.json" \\
  --run-id ${NVFP4_RUN}_quality_from_benchmark \\
  --output "\${RESULTS_DIR}/${NVFP4_RUN}_quality.json"

python scripts/openai_quality_probe.py \\
  --input-report "\${RESULTS_DIR}/${NVFP4_RUN}_openai_benchmark.json" \\
  --compare-to "\${RESULTS_DIR}/${FP8_RUN}_openai_benchmark.json" \\
  --run-id ${PREFIX}_quality_compare \\
  --output "\${RESULTS_DIR}/${PREFIX}_quality_compare.json"

stop_vllm_container ${NVFP4_RUN}

# Planned artifacts:
# - \${RESULTS_DIR}/${FP8_RUN}_server.log
# - \${RESULTS_DIR}/${FP8_RUN}_import_probe.txt
# - \${RESULTS_DIR}/${FP8_RUN}_editable_install.log
# - \${RESULTS_DIR}/${FP8_RUN}_row_manifest.json
# - \${RESULTS_DIR}/${FP8_RUN}_runtime_probe.json
# - \${RESULTS_DIR}/${FP8_RUN}_openai_benchmark.json
# - \${RESULTS_DIR}/${FP8_RUN}_chat_smoke.json
# - \${RESULTS_DIR}/${FP8_RUN}_build_target_audit.json
# - \${RESULTS_DIR}/${FP8_RUN}_quality.json
# - \${RESULTS_DIR}/${NVFP4_RUN}_server.log
# - \${RESULTS_DIR}/${NVFP4_RUN}_import_probe.txt
# - \${RESULTS_DIR}/${NVFP4_RUN}_editable_install.log
# - \${RESULTS_DIR}/${NVFP4_RUN}_row_manifest.json
# - \${RESULTS_DIR}/${NVFP4_RUN}_runtime_probe.json
# - \${RESULTS_DIR}/${NVFP4_RUN}_openai_benchmark.json
# - \${RESULTS_DIR}/${NVFP4_RUN}_chat_smoke.json
# - \${RESULTS_DIR}/${NVFP4_RUN}_build_target_audit.json
# - \${RESULTS_DIR}/${NVFP4_RUN}_quality.json
# - \${RESULTS_DIR}/${PREFIX}_quality_compare.json
EOF
