# Prep-only command packet for vllm_gemma3_27b_rung1_20260608T205432JST
# This file is intentionally printed by the prep script. Review it, then run
# one server row at a time on the Spark host when the GPU is available.
set -euo pipefail

export VLLM_SRC=/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608/third_party/vllm
export FLASHINFER_SRC=/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608/third_party/flashinfer
export HF_CACHE=/home/jethac/.cache/huggingface
export RESULTS_DIR=results
export IMAGE=jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass
export GPU_MEMORY_UTILIZATION=0.85
export MAX_MODEL_LEN=131072
export MAX_NUM_BATCHED_TOKENS=4096
export VLLM_SOURCE_COMMIT=25ab073ef87f4443616fbaf00a2f6f09a9087c1f
export VLLM_PRECOMPILED_WHEEL_COMMIT=4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa
export VLLM_VERSION_OVERRIDE=0.1.dev1+g25ab073ef
export RUN_FP8=${RUN_FP8:-1}
export RUN_NVFP4=${RUN_NVFP4:-1}

mkdir -p "${RESULTS_DIR}"

wait_for_vllm() {
  local run_name=$1
  local log_path=$2
  local deadline=$((SECONDS + 1800))
  until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do
    if ! docker inspect -f '{{.State.Running}}' "${run_name}" 2>/dev/null | grep -q '^true$'; then
      echo "Container ${run_name} exited before readiness. Last log lines:" >&2
      tail -80 "${log_path}" >&2 || true
      return 1
    fi
    if (( SECONDS > deadline )); then
      echo "Timed out waiting for ${run_name}. Last log lines:" >&2
      tail -120 "${log_path}" >&2 || true
      return 1
    fi
    sleep 10
  done
}

stop_vllm_container() {
  local run_name=$1
  docker rm -f "${run_name}" >/dev/null 2>&1 || true
}

trap 'stop_vllm_container vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer; stop_vllm_container vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer' EXIT

# Start the fp8 comparator row. Stop this container before starting the NVFP4 row.
if [[ "${RUN_FP8}" == "1" ]]; then
docker rm -f vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer >/dev/null 2>&1 || true
docker run -d --gpus all --ipc=host --network=host \
  --name vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer \
  -e HF_TOKEN \
  -e VLLM_USE_V1=1 \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -e VLLM_SPARK_KV_GEOMETRY_LOG=1 \
  -e SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src \
  -e FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e CUDA_MODULE_LOADING=LAZY \
  -v "${VLLM_SRC}:/vllm-src" \
  -v "${FLASHINFER_SRC}:/flashinfer-src" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${RESULTS_DIR}:/results" \
  -v "$(pwd):/workspace/dgx-spark-hijinks" \
  --entrypoint bash \
  "${IMAGE}" \
  -lc '
set -euo pipefail
mkdir -p /tmp/spark-sitecustomize
cp /workspace/dgx-spark-hijinks/scripts/flashinfer_source_sitecustomize.py /tmp/spark-sitecustomize/sitecustomize.py
export PYTHONPATH="/tmp/spark-sitecustomize:${PYTHONPATH:-}"
python3 -m pip install -q setuptools-rust > /results/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_pip_bootstrap.log 2>&1
cd /vllm-src
VLLM_USE_PRECOMPILED=1 VLLM_MAIN_CUDA_VERSION=13.0 VLLM_PRECOMPILED_WHEEL_COMMIT='"4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa"' VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1 VLLM_VERSION_OVERRIDE='"0.1.dev1+g25ab073ef"' \
  python3 -m pip install --no-build-isolation --no-deps -e . > /results/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_editable_install.log 2>&1
cp /opt/jethac-vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so /vllm-src/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
python3 - <<'"'"'PY'"'"' > /results/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_import_probe.txt 2>&1
import json, torch, transformers, vllm, flashinfer
import vllm.vllm_flash_attn._vllm_fa2_C as fa2_ext
print(json.dumps({
  "vllm": getattr(vllm, "__version__", None),
  "vllm_file": getattr(vllm, "__file__", None),
  "vllm_fa2": getattr(fa2_ext, "__file__", None),
  "flashinfer": getattr(flashinfer, "__version__", None),
  "flashinfer_file": getattr(flashinfer, "__file__", None),
  "transformers": getattr(transformers, "__version__", None),
  "torch": torch.__version__,
  "torch_cuda": torch.version.cuda,
  "device": torch.cuda.get_device_name(0),
  "capability": torch.cuda.get_device_capability(0),
}, indent=2, sort_keys=True))
PY
exec vllm serve google/gemma-3-27b-it \
  --served-model-name gemma3-27b-it \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --max-model-len '"131072"' \
  --gpu-memory-utilization '"0.85"' \
  --max-num-batched-tokens '"4096"' \
  --host 0.0.0.0 \
  --port 8000
'
docker inspect -f '{{.Id}}' vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer > "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_container_id.txt"
docker logs -f vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer > "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_server.log" 2>&1 &
echo "$!" > "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_docker_logs_pid.txt"
wait_for_vllm vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_server.log"

python3 scripts/record_openai_serving_row.py \
  --backend vllm --phase before --run-id vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer \
  --url http://127.0.0.1:8000 --model gemma3-27b-it \
  --results-dir "${RESULTS_DIR}" \
  --runtime-ref "jethac/vllm@25ab073ef87f4443616fbaf00a2f6f09a9087c1f + jethac/flashinfer@e152cf4da4ab2a9d093b7d9d4b499198b0211c61 source overlay; precompiled wheel base 4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa" \
  --container-image "${IMAGE}" \
  --kv-cache-dtype fp8 --attention-backend flashinfer \
  --cuda-graph-mode default \
  --server-log "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_server.log" \
  --process-match "vllm serve google/gemma-3-27b-it" || true

python3 scripts/openai_first_token_probe.py \
  --url http://127.0.0.1:8000 --model gemma3-27b-it \
  --backend vllm --phase before --run-id vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_first_token \
  --output "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_first_token.json" || true

python3 scripts/openai_quality_probe.py \
  --input-report "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_openai_benchmark.json" \
  --run-id vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_quality_from_benchmark \
  --output "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_quality.json" || true

stop_vllm_container vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer
else
  echo "RUN_FP8=${RUN_FP8}; reusing any existing fp8 comparator artifacts for vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer."
fi

if [[ "${RUN_NVFP4}" != "1" ]]; then
  echo "RUN_NVFP4=${RUN_NVFP4}; stopping after fp8 comparator row."
  exit 0
fi

# Start the NVFP4-KV candidate row with the same geometry and serving settings.
docker rm -f vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer >/dev/null 2>&1 || true
docker run -d --gpus all --ipc=host --network=host \
  --name vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer \
  -e HF_TOKEN \
  -e VLLM_USE_V1=1 \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -e VLLM_SPARK_KV_GEOMETRY_LOG=1 \
  -e SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src \
  -e FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e CUDA_MODULE_LOADING=LAZY \
  -v "${VLLM_SRC}:/vllm-src" \
  -v "${FLASHINFER_SRC}:/flashinfer-src" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${RESULTS_DIR}:/results" \
  -v "$(pwd):/workspace/dgx-spark-hijinks" \
  --entrypoint bash \
  "${IMAGE}" \
  -lc '
set -euo pipefail
mkdir -p /tmp/spark-sitecustomize
cp /workspace/dgx-spark-hijinks/scripts/flashinfer_source_sitecustomize.py /tmp/spark-sitecustomize/sitecustomize.py
export PYTHONPATH="/tmp/spark-sitecustomize:${PYTHONPATH:-}"
python3 -m pip install -q setuptools-rust > /results/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_pip_bootstrap.log 2>&1
cd /vllm-src
VLLM_USE_PRECOMPILED=1 VLLM_MAIN_CUDA_VERSION=13.0 VLLM_PRECOMPILED_WHEEL_COMMIT='"4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa"' VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1 VLLM_VERSION_OVERRIDE='"0.1.dev1+g25ab073ef"' \
  python3 -m pip install --no-build-isolation --no-deps -e . > /results/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_editable_install.log 2>&1
cp /opt/jethac-vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so /vllm-src/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
python3 - <<'"'"'PY'"'"' > /results/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_import_probe.txt 2>&1
import json, torch, transformers, vllm, flashinfer
import vllm.vllm_flash_attn._vllm_fa2_C as fa2_ext
print(json.dumps({
  "vllm": getattr(vllm, "__version__", None),
  "vllm_file": getattr(vllm, "__file__", None),
  "vllm_fa2": getattr(fa2_ext, "__file__", None),
  "flashinfer": getattr(flashinfer, "__version__", None),
  "flashinfer_file": getattr(flashinfer, "__file__", None),
  "transformers": getattr(transformers, "__version__", None),
  "torch": torch.__version__,
  "torch_cuda": torch.version.cuda,
  "device": torch.cuda.get_device_name(0),
  "capability": torch.cuda.get_device_capability(0),
}, indent=2, sort_keys=True))
PY
exec vllm serve google/gemma-3-27b-it \
  --served-model-name gemma3-27b-it \
  --dtype bfloat16 \
  --kv-cache-dtype nvfp4 \
  --attention-backend flashinfer \
  --max-model-len '"131072"' \
  --gpu-memory-utilization '"0.85"' \
  --max-num-batched-tokens '"4096"' \
  --host 0.0.0.0 \
  --port 8000
'
docker inspect -f '{{.Id}}' vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer > "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_container_id.txt"
docker logs -f vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer > "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_server.log" 2>&1 &
echo "$!" > "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_docker_logs_pid.txt"
wait_for_vllm vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_server.log"

python3 scripts/record_openai_serving_row.py \
  --backend vllm --phase after --run-id vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer \
  --url http://127.0.0.1:8000 --model gemma3-27b-it \
  --results-dir "${RESULTS_DIR}" \
  --runtime-ref "jethac/vllm@25ab073ef87f4443616fbaf00a2f6f09a9087c1f + jethac/flashinfer@e152cf4da4ab2a9d093b7d9d4b499198b0211c61 source overlay; precompiled wheel base 4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa" \
  --container-image "${IMAGE}" \
  --kv-cache-dtype nvfp4 --attention-backend flashinfer \
  --cuda-graph-mode default \
  --server-log "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_server.log" \
  --process-match "vllm serve google/gemma-3-27b-it" || true

python3 scripts/openai_first_token_probe.py \
  --url http://127.0.0.1:8000 --model gemma3-27b-it \
  --backend vllm --phase after --run-id vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_first_token \
  --output "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_first_token.json" || true

python3 scripts/openai_quality_probe.py \
  --input-report "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_openai_benchmark.json" \
  --run-id vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_quality_from_benchmark \
  --output "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_quality.json" || true

python3 scripts/openai_quality_probe.py \
  --input-report "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_openai_benchmark.json" \
  --compare-to "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_openai_benchmark.json" \
  --run-id vllm_gemma3_27b_rung1_20260608T205432JST_quality_compare \
  --output "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_quality_compare.json" || true

python3 scripts/openai_first_token_probe.py \
  --input-report "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_first_token.json" \
  --compare-to "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_first_token.json" \
  --run-id vllm_gemma3_27b_rung1_20260608T205432JST_first_token_compare \
  --output "${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_first_token_compare.json" || true

stop_vllm_container vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer

# Planned artifacts:
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_server.log
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_import_probe.txt
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_editable_install.log
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_row_manifest.json
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_runtime_probe.json
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_openai_benchmark.json
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_chat_smoke.json
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_build_target_audit.json
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_quality.json
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_first_token.json
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_server.log
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_import_probe.txt
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_editable_install.log
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_row_manifest.json
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_runtime_probe.json
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_openai_benchmark.json
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_chat_smoke.json
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_build_target_audit.json
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_quality.json
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_first_token.json
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_quality_compare.json
# - ${RESULTS_DIR}/vllm_gemma3_27b_rung1_20260608T205432JST_first_token_compare.json
