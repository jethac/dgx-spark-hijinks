#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(pwd)}
VLLM_SRC=${VLLM_SRC:-${REPO_ROOT}/third_party/vllm}
FLASHINFER_SRC=${FLASHINFER_SRC:-${REPO_ROOT}/third_party/flashinfer}
HF_CACHE=${HF_CACHE:-/home/jethac/.cache/huggingface}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results}
IMAGE=${IMAGE:-jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass}
MODEL=${MODEL:-google/gemma-3-27b-it}
SERVED_MODEL=${SERVED_MODEL:-gemma3-27b-it}
STAMP=${STAMP:-$(date +%Y%m%dT%H%MJST)}
RUN=${RUN:-vllm_gemma3_27b_contigout_${STAMP}_nvfp4_kv_flashinfer_eager}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.72}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
GB10_DOCKER_MEMORY=${GB10_DOCKER_MEMORY:-100g}
GB10_DOCKER_MEMORY_SWAP=${GB10_DOCKER_MEMORY_SWAP:-100g}
VLLM_PRECOMPILED_WHEEL_COMMIT=${VLLM_PRECOMPILED_WHEEL_COMMIT:-4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa}
VLLM_VERSION_OVERRIDE=${VLLM_VERSION_OVERRIDE:-0.1.dev1+gcontigout}
FLASHINFER_PREFILL_DEBUG_ONCE=${FLASHINFER_PREFILL_DEBUG_ONCE:-0}
FLASHINFER_EXTRA_CUDAFLAGS=${FLASHINFER_EXTRA_CUDAFLAGS:-"-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1 -gencode=arch=compute_121a,code=sm_121a"}
SPARK_FLASHINFER_FORCE_PREFILL_MODULE=${SPARK_FLASHINFER_FORCE_PREFILL_MODULE:-0}
SPARK_FLASHINFER_PATCH_PREFILL_RUN_SCALE_ARGS=${SPARK_FLASHINFER_PATCH_PREFILL_RUN_SCALE_ARGS:-1}
FLASHINFER_CLEAR_PREFILL_CACHE=${FLASHINFER_CLEAR_PREFILL_CACHE:-${SPARK_FLASHINFER_FORCE_PREFILL_MODULE}}
SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG=${SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG:-1}

mkdir -p "${RESULTS_DIR}/${RUN}_active_page_dump"
if [[ -z "${HF_TOKEN:-}" && -f "${HF_CACHE}/token" ]]; then
  export HF_TOKEN
  HF_TOKEN=$(<"${HF_CACHE}/token")
fi

stop_vllm_container() {
  docker rm -f "${RUN}" >/dev/null 2>&1 || true
}

wait_for_vllm() {
  local deadline=$((SECONDS + 1800))
  until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do
    if ! docker inspect -f '{{.State.Running}}' "${RUN}" 2>/dev/null | grep -q '^true$'; then
      echo "Container ${RUN} exited before readiness. Last log lines:" >&2
      tail -120 "${RESULTS_DIR}/${RUN}_server.log" >&2 || true
      return 1
    fi
    if (( SECONDS > deadline )); then
      echo "Timed out waiting for ${RUN}. Last log lines:" >&2
      tail -160 "${RESULTS_DIR}/${RUN}_server.log" >&2 || true
      return 1
    fi
    sleep 10
  done
}

trap stop_vllm_container EXIT
stop_vllm_container

docker run -d --gpus all --ipc=host --network=host \
  --name "${RUN}" \
  --memory "${GB10_DOCKER_MEMORY}" \
  --memory-swap "${GB10_DOCKER_MEMORY_SWAP}" \
  -e HF_TOKEN \
  -e VLLM_USE_V1=1 \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -e VLLM_SPARK_KV_GEOMETRY_LOG=1 \
  -e VLLM_SPARK_GEMMA_TENSOR_TRACE=1 \
  -e VLLM_SPARK_GEMMA_TENSOR_TRACE_FILE=/results/${RUN}_tensor_trace.jsonl \
  -e VLLM_SPARK_GEMMA_TENSOR_TRACE_LIMIT=64 \
  -e VLLM_SPARK_GEMMA_TENSOR_TRACE_VALUES=16 \
  -e VLLM_SPARK_GEMMA_TENSOR_TRACE_LAYERS=layers.5.self_attn.attn,lm_head \
  -e VLLM_SPARK_NVFP4_PREFILL_CONTIG_OUT=1 \
  -e VLLM_SPARK_NVFP4_PREFILL_FRESH_WRAPPER_REPLAY=${VLLM_SPARK_NVFP4_PREFILL_FRESH_WRAPPER_REPLAY:-0} \
  -e VLLM_SPARK_NVFP4_FRESH_WRAPPER_WORKSPACE_MB=${VLLM_SPARK_NVFP4_FRESH_WRAPPER_WORKSPACE_MB:-256} \
  -e VLLM_SPARK_ACTIVE_PAGE_DUMP=1 \
  -e VLLM_SPARK_ACTIVE_PAGE_DUMP_DIR=/results/${RUN}_active_page_dump \
  -e VLLM_SPARK_ACTIVE_PAGE_DUMP_LIMIT=1 \
  -e VLLM_SPARK_ACTIVE_PAGE_DUMP_PAGES=4 \
  -e SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src \
  -e FLASHINFER_PREFILL_DEBUG_ONCE \
  -e FLASHINFER_EXTRA_CUDAFLAGS \
  -e SPARK_FLASHINFER_FORCE_PREFILL_MODULE \
  -e SPARK_FLASHINFER_PATCH_PREFILL_RUN_SCALE_ARGS \
  -e FLASHINFER_CLEAR_PREFILL_CACHE \
  -e SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e CUDA_MODULE_LOADING=LAZY \
  -v "${VLLM_SRC}:/vllm-src" \
  -v "${FLASHINFER_SRC}:/flashinfer-src" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${RESULTS_DIR}:/results" \
  -v "${REPO_ROOT}:/workspace/dgx-spark-hijinks" \
  --entrypoint bash \
  "${IMAGE}" \
  -lc '
set -euo pipefail
git config --global --add safe.directory /vllm-src
mkdir -p /tmp/spark-sitecustomize
cp /workspace/dgx-spark-hijinks/scripts/flashinfer_source_sitecustomize.py /tmp/spark-sitecustomize/sitecustomize.py
export PYTHONPATH="/tmp/spark-sitecustomize:${PYTHONPATH:-}"
if [[ "${FLASHINFER_CLEAR_PREFILL_CACHE:-0}" == "1" ]]; then
  find /root/.cache/flashinfer -path "*/cached_ops/batch_prefill_with_kv_cache_*" -prune -exec rm -rf {} + 2>/dev/null || true
  find /root/.cache/flashinfer -path "*/generated/batch_prefill_with_kv_cache_*" -prune -exec rm -rf {} + 2>/dev/null || true
  find /root/.cache/flashinfer -path "*/cached_ops/vllm_batch_prefill_nvfp4_kv_*" -prune -exec rm -rf {} + 2>/dev/null || true
  find /root/.cache/flashinfer -path "*/generated/vllm_batch_prefill_nvfp4_kv_*" -prune -exec rm -rf {} + 2>/dev/null || true
fi
python3 -m pip install -q setuptools-rust > /results/'"${RUN}"'_pip_bootstrap.log 2>&1
cd /vllm-src
VLLM_USE_PRECOMPILED=1 VLLM_MAIN_CUDA_VERSION=13.0 VLLM_PRECOMPILED_WHEEL_COMMIT='"${VLLM_PRECOMPILED_WHEEL_COMMIT}"' VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1 VLLM_VERSION_OVERRIDE='"${VLLM_VERSION_OVERRIDE}"' \
  python3 -m pip install --no-build-isolation --no-deps -e . > /results/'"${RUN}"'_editable_install.log 2>&1
cp /opt/jethac-vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so /vllm-src/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
python3 - <<'"'"'PY'"'"' > /results/'"${RUN}"'_import_probe.txt 2>&1
import json, torch, transformers, vllm, flashinfer
import flashinfer.jit.attention.modules as flashinfer_attention_modules
import flashinfer.jit as flashinfer_jit
import flashinfer.jit.utils as flashinfer_jit_utils
import vllm.vllm_flash_attn._vllm_fa2_C as fa2_ext
print(json.dumps({
  "vllm": getattr(vllm, "__version__", None),
  "vllm_file": getattr(vllm, "__file__", None),
  "vllm_fa2": getattr(fa2_ext, "__file__", None),
  "flashinfer": getattr(flashinfer, "__version__", None),
  "flashinfer_file": getattr(flashinfer, "__file__", None),
  "flashinfer_dtype_map_kv_uint8": flashinfer_jit_utils.dtype_map_kv.get(torch.uint8),
  "flashinfer_attention_dtype_map_kv_uint8": flashinfer_attention_modules.dtype_map_kv.get(torch.uint8),
  "flashinfer_filename_safe_dtype_map_kv_uint8": flashinfer_jit_utils.filename_safe_dtype_map_kv(torch.uint8),
  "spark_flashinfer_force_prefill_module": __import__("os").environ.get("SPARK_FLASHINFER_FORCE_PREFILL_MODULE"),
  "spark_flashinfer_patch_prefill_run_scale_args": __import__("os").environ.get("SPARK_FLASHINFER_PATCH_PREFILL_RUN_SCALE_ARGS"),
  "flashinfer_prefill_run_marker": bool(getattr(__import__("flashinfer.prefill", fromlist=["BatchPrefillWithPagedKVCacheWrapper"]).BatchPrefillWithPagedKVCacheWrapper.run, "_spark_prefill_scale_arg_patch", False)),
  "flashinfer_attention_gen_batch_prefill_module": getattr(flashinfer_attention_modules.gen_batch_prefill_module, "__module__", None),
  "flashinfer_jit_gen_batch_prefill_module": getattr(flashinfer_jit.gen_batch_prefill_module, "__module__", None),
  "transformers": getattr(transformers, "__version__", None),
  "torch": torch.__version__,
  "torch_cuda": torch.version.cuda,
  "device": torch.cuda.get_device_name(0),
  "capability": torch.cuda.get_device_capability(0),
}, indent=2, sort_keys=True))
PY
exec vllm serve '"${MODEL}"' \
  --served-model-name '"${SERVED_MODEL}"' \
  --dtype bfloat16 \
  --kv-cache-dtype nvfp4 \
  --attention-backend flashinfer \
  --max-model-len '"${MAX_MODEL_LEN}"' \
  --gpu-memory-utilization '"${GPU_MEMORY_UTILIZATION}"' \
  --max-num-batched-tokens '"${MAX_NUM_BATCHED_TOKENS}"' \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000
'

docker inspect -f '{{.Id}}' "${RUN}" > "${RESULTS_DIR}/${RUN}_container_id.txt"
docker logs -f "${RUN}" > "${RESULTS_DIR}/${RUN}_server.log" 2>&1 &
echo "$!" > "${RESULTS_DIR}/${RUN}_docker_logs_pid.txt"

wait_for_vllm

python3 "${REPO_ROOT}/scripts/openai_first_token_probe.py" \
  --url http://127.0.0.1:8000 \
  --model "${SERVED_MODEL}" \
  --backend vllm \
  --phase after \
  --run-id "${RUN}_first_token" \
  --output "${RESULTS_DIR}/${RUN}_first_token.json" || true

stop_vllm_container
printf '%s\n' "${RUN}" > "${RESULTS_DIR}/vllm_gemma3_27b_contigout_latest_run.txt"
echo "${RESULTS_DIR}/${RUN}_first_token.json"
