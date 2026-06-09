#!/usr/bin/env bash
set -euo pipefail

STAMP=${STAMP:-$(date +%Y%m%dT%H%MJST)}
RUN=${RUN:-vllm_gemma3_27b_ppl_${STAMP}}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
VLLM_SRC=${VLLM_SRC:-${REPO_ROOT}/third_party/vllm}
FLASHINFER_SRC=${FLASHINFER_SRC:-${REPO_ROOT}/third_party/flashinfer}
HF_CACHE=${HF_CACHE:-${HOME}/.cache/huggingface}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results}
RUN_ROOT=${RUN_ROOT:-/home/jethac/spark_tmp/${RUN}}
IMAGE=${IMAGE:-jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass}
MODEL=${MODEL:-google/gemma-3-27b-it}
SERVED_MODEL=${SERVED_MODEL:-gemma3-27b-it}
CTXS=${CTXS:-512 2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.72}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-1024}
GB10_DOCKER_MEMORY=${GB10_DOCKER_MEMORY:-100g}
GB10_DOCKER_MEMORY_SWAP=${GB10_DOCKER_MEMORY_SWAP:-100g}
VLLM_PRECOMPILED_WHEEL_COMMIT=${VLLM_PRECOMPILED_WHEEL_COMMIT:-4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa}
VLLM_VERSION_OVERRIDE=${VLLM_VERSION_OVERRIDE:-0.1.dev1+ggemma3ppl}
FLASHINFER_EXTRA_CUDAFLAGS=${FLASHINFER_EXTRA_CUDAFLAGS:-"-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1 -gencode=arch=compute_121a,code=sm_121a"}
SPARK_FLASHINFER_FORCE_PREFILL_MODULE=${SPARK_FLASHINFER_FORCE_PREFILL_MODULE:-0}
SPARK_FLASHINFER_PATCH_PREFILL_RUN_SCALE_ARGS=${SPARK_FLASHINFER_PATCH_PREFILL_RUN_SCALE_ARGS:-1}
FLASHINFER_CLEAR_PREFILL_CACHE=${FLASHINFER_CLEAR_PREFILL_CACHE:-1}
SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG=${SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG:-1}
PPL_REQUEST_ATTEMPTS=${PPL_REQUEST_ATTEMPTS:-3}
PPL_RETRY_SLEEP_S=${PPL_RETRY_SLEEP_S:-10}

mkdir -p "${RESULTS_DIR}" "${RUN_ROOT}/docs" "${RUN_ROOT}/results"

if [[ -z "${HF_TOKEN:-}" && -f "${HF_CACHE}/token" ]]; then
  export HF_TOKEN
  HF_TOKEN=$(<"${HF_CACHE}/token")
fi

if [[ -z "${CORPUS:-}" ]]; then
  CORPUS="${RUN_ROOT}/docs/ppl_corpus.md"
  CORPUS_MANIFEST="${RUN_ROOT}/results/${RUN}_corpus_manifest.json"
  python3 "${REPO_ROOT}/scripts/build_ppl_corpus.py" \
    --repo-root "${REPO_ROOT}" \
    --output "${CORPUS}" \
    --manifest "${CORPUS_MANIFEST}" \
    --max-chars 250000 \
    --exclude-substring "_server" \
    --exclude-substring "_trace" >/dev/null
else
  CORPUS_MANIFEST=${CORPUS_MANIFEST:-}
  cp "${CORPUS}" "${RUN_ROOT}/docs/ppl_corpus.md"
  CORPUS="${RUN_ROOT}/docs/ppl_corpus.md"
  if [[ -n "${CORPUS_MANIFEST}" ]]; then
    cp "${CORPUS_MANIFEST}" "${RUN_ROOT}/results/${RUN}_corpus_manifest.json"
  fi
fi

stop_container() {
  local name="$1"
  docker rm -f "${name}" >/dev/null 2>&1 || true
}

trap 'stop_container "${RUN}_fp8"; stop_container "${RUN}_nvfp4"' EXIT

wait_for_vllm() {
  local name="$1"
  local log_path="$2"
  local deadline=$((SECONDS + 1800))
  until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do
    if ! docker inspect -f '{{.State.Running}}' "${name}" 2>/dev/null | grep -q '^true$'; then
      echo "Container ${name} exited before readiness. Last log lines:" >&2
      tail -120 "${log_path}" >&2 || true
      return 1
    fi
    if (( SECONDS > deadline )); then
      echo "Timed out waiting for ${name}. Last log lines:" >&2
      tail -160 "${log_path}" >&2 || true
      return 1
    fi
    sleep 10
  done
}

run_one() {
  local kv="$1"
  local name="${RUN}_${kv}"
  local log_path="${RUN_ROOT}/results/${RUN}_${kv}_server.log"
  local status=0

  stop_container "${name}"
  docker run -d --gpus all --ipc=host --network=host \
    --name "${name}" \
    --memory "${GB10_DOCKER_MEMORY}" \
    --memory-swap "${GB10_DOCKER_MEMORY_SWAP}" \
    -e HF_TOKEN \
    -e VLLM_USE_V1=1 \
    -e VLLM_LOGGING_LEVEL=DEBUG \
    -e VLLM_SPARK_KV_GEOMETRY_LOG=1 \
    -e SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src \
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
    -v "${RUN_ROOT}:/work" \
    -v "${REPO_ROOT}:/workspace/dgx-spark-hijinks" \
    -w /work \
    --entrypoint bash \
    "${IMAGE}" \
    -lc '
set -euo pipefail
mkdir -p /tmp/spark-sitecustomize
mkdir -p /work/scripts /work/results
cp /workspace/dgx-spark-hijinks/scripts/flashinfer_source_sitecustomize.py /tmp/spark-sitecustomize/sitecustomize.py
export PYTHONPATH="/tmp/spark-sitecustomize:${PYTHONPATH:-}"
if [[ "${FLASHINFER_CLEAR_PREFILL_CACHE:-0}" == "1" ]]; then
  find /root/.cache/flashinfer -path "*/cached_ops/batch_prefill_with_kv_cache_*" -prune -exec rm -rf {} + 2>/dev/null || true
  find /root/.cache/flashinfer -path "*/generated/batch_prefill_with_kv_cache_*" -prune -exec rm -rf {} + 2>/dev/null || true
  find /root/.cache/flashinfer -path "*/cached_ops/vllm_batch_prefill_nvfp4_kv_*" -prune -exec rm -rf {} + 2>/dev/null || true
  find /root/.cache/flashinfer -path "*/generated/vllm_batch_prefill_nvfp4_kv_*" -prune -exec rm -rf {} + 2>/dev/null || true
fi
cp /workspace/dgx-spark-hijinks/scripts/vllm_prompt_ppl_sweep.py /work/scripts/
cp /workspace/dgx-spark-hijinks/scripts/spark_hardware.py /work/scripts/
python3 -m pip install -q setuptools-rust > /work/results/'"${RUN}"'_'"${kv}"'_pip_bootstrap.log 2>&1
cd /vllm-src
VLLM_USE_PRECOMPILED=1 VLLM_MAIN_CUDA_VERSION=13.0 VLLM_PRECOMPILED_WHEEL_COMMIT='"${VLLM_PRECOMPILED_WHEEL_COMMIT}"' VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1 VLLM_VERSION_OVERRIDE='"${VLLM_VERSION_OVERRIDE}"' \
  python3 -m pip install --no-build-isolation --no-deps -e . > /work/results/'"${RUN}"'_'"${kv}"'_editable_install.log 2>&1
cp /opt/jethac-vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so /vllm-src/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
python3 - <<'"'"'PY'"'"' > /work/results/'"${RUN}"'_'"${kv}"'_import_probe.txt 2>&1
import json, os, torch, transformers, vllm, flashinfer
import flashinfer.jit.attention.modules as flashinfer_attention_modules
import flashinfer.jit.utils as flashinfer_jit_utils
import vllm.vllm_flash_attn._vllm_fa2_C as fa2_ext
print(json.dumps({
  "vllm": getattr(vllm, "__version__", None),
  "vllm_file": getattr(vllm, "__file__", None),
  "vllm_fa2": getattr(fa2_ext, "__file__", None),
  "flashinfer": getattr(flashinfer, "__version__", None),
  "flashinfer_file": getattr(flashinfer, "__file__", None),
  "flashinfer_attention_dtype_map_kv_uint8": flashinfer_attention_modules.dtype_map_kv.get(torch.uint8),
  "flashinfer_filename_safe_dtype_map_kv_uint8": flashinfer_jit_utils.filename_safe_dtype_map_kv(torch.uint8),
  "spark_flashinfer_patch_prefill_run_scale_args": os.environ.get("SPARK_FLASHINFER_PATCH_PREFILL_RUN_SCALE_ARGS"),
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
  --kv-cache-dtype '"${kv}"' \
  --attention-backend flashinfer \
  --max-model-len '"${MAX_MODEL_LEN}"' \
  --gpu-memory-utilization '"${GPU_MEMORY_UTILIZATION}"' \
  --max-num-batched-tokens '"${MAX_NUM_BATCHED_TOKENS}"' \
  --no-enable-prefix-caching \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000
'

  docker logs -f "${name}" >"${log_path}" 2>&1 &
  echo "$!" >"${RUN_ROOT}/results/${RUN}_${kv}_docker_logs_pid.txt"
  docker inspect -f '{{.Id}}' "${name}" >"${RUN_ROOT}/results/${RUN}_${kv}_container_id.txt"

  if wait_for_vllm "${name}" "${log_path}"; then
    local ctx_args=()
    for ctx in ${CTXS}; do
      ctx_args+=(--ctx "${ctx}")
    done
    docker exec "${name}" python3 scripts/vllm_prompt_ppl_sweep.py \
      --url http://127.0.0.1:8000 \
      --model "${SERVED_MODEL}" \
      --tokenizer "${MODEL}" \
      --text-file docs/ppl_corpus.md \
      "${ctx_args[@]}" \
      --run-id "${RUN}_${kv}" \
      --kv-cache-dtype "${kv}" \
      --runtime-ref "jethac/vllm source overlay + jethac/flashinfer source overlay; sequential Gemma 3 PPL gate" \
      --container-image "${IMAGE}" \
      --scope "Gemma 3 text-only prompt-logprob PPL; sequential fp8-vs-NVFP4; prefix caching disabled by fresh single request; short-context gate" \
      --request-attempts "${PPL_REQUEST_ATTEMPTS}" \
      --retry-sleep-s "${PPL_RETRY_SLEEP_S}" \
      --output "results/${RUN}_${kv}_ppl.json" \
      >"${RUN_ROOT}/results/${RUN}_${kv}_ppl_stdout.json" \
      2>"${RUN_ROOT}/results/${RUN}_${kv}_ppl_stderr.log" || status=$?
  else
    status=1
  fi

  docker logs "${name}" >"${RUN_ROOT}/results/${RUN}_${kv}_server_after.log" 2>&1 || true
  docker inspect "${name}" >"${RUN_ROOT}/results/${RUN}_${kv}_inspect.json" 2>/dev/null || true
  stop_container "${name}"
  return "${status}"
}

overall=0
run_one fp8 || overall=1
run_one nvfp4 || overall=1

if [[ -f "${RUN_ROOT}/results/${RUN}_fp8_ppl.json" && -f "${RUN_ROOT}/results/${RUN}_nvfp4_ppl.json" ]]; then
  python3 "${REPO_ROOT}/scripts/vllm_prompt_ppl_sweep.py" \
    --compare-fp8 "${RUN_ROOT}/results/${RUN}_fp8_ppl.json" \
    --compare-nvfp4 "${RUN_ROOT}/results/${RUN}_nvfp4_ppl.json" \
    --output "${RUN_ROOT}/results/${RUN}_compare.json" || overall=1
fi

python3 - <<PY
import json
from pathlib import Path
run = "${RUN}"
root = Path("${RUN_ROOT}")
results_dir = Path("${RESULTS_DIR}")
summary = {
    "schema": "vllm-gemma3-ppl-pair-summary/v1",
    "run_id": run,
    "run_root": str(root),
    "model": "${MODEL}",
    "served_model": "${SERVED_MODEL}",
    "image": "${IMAGE}",
    "contexts": [int(x) for x in "${CTXS}".split()],
    "max_model_len": int("${MAX_MODEL_LEN}"),
    "gpu_memory_utilization": float("${GPU_MEMORY_UTILIZATION}"),
    "docker_memory": "${GB10_DOCKER_MEMORY}",
    "artifacts": {},
    "ok": False,
}
for kv in ("fp8", "nvfp4"):
    ppl = root / "results" / f"{run}_{kv}_ppl.json"
    server = root / "results" / f"{run}_{kv}_server_after.log"
    item = {"ppl": str(ppl), "server_log": str(server), "ok": False}
    if ppl.exists():
        try:
            obj = json.loads(ppl.read_text())
            item["ok"] = bool(obj.get("ok"))
            item["error"] = obj.get("error")
            item["scores"] = [
                {"ctx": row.get("ctx"), "score": row.get("score")}
                for row in obj.get("contexts", [])
            ]
        except Exception as exc:
            item["error"] = repr(exc)
    summary["artifacts"][kv] = item
compare = root / "results" / f"{run}_compare.json"
summary["compare"] = str(compare)
if compare.exists():
    try:
        obj = json.loads(compare.read_text())
        summary["compare_ok"] = bool(obj.get("ok"))
        summary["rows"] = obj.get("rows", [])
    except Exception as exc:
        summary["compare_error"] = repr(exc)
summary["ok"] = bool(summary.get("compare_ok"))
out = results_dir / f"{run}_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
(results_dir / f"{run}_summary.md").write_text(
    "# vLLM Gemma 3 NVFP4-KV PPL Pair\\n\\n"
    f"Run: {run}\\n\\n"
    f"Status: {'green' if summary['ok'] else 'red/incomplete'}\\n\\n"
    f"Run root: {root}\\n\\n"
    f"Compare artifact: {compare}\\n\\n"
    "This is a sequential fp8-vs-NVFP4 prompt-logprob PPL gate under the GB10 memory rules.\\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

printf "%s\n" "${RUN}" >"${RESULTS_DIR}/vllm_gemma3_ppl_latest_run.txt"
exit "${overall}"
