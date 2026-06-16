#!/usr/bin/env bash
# Active-page bf16/NVFP4 K/V mix attribution for Gemma 4 26B-A4B.
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be exported}"

MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
CTX="${CTX:-8185}"
PREFIX="${PREFIX:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_UTIL="${GPU_UTIL:-0.82}"
OUT="${OUT:-/root/dx26_active_kv_mix_$(date -u +%Y%m%dT%H%M%SZ)}"
ROW_TIMEOUT="${ROW_TIMEOUT:-5400}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-1}"
CAPTURE_MAX="${CAPTURE_MAX:-8}"
CAPTURE_QO="${CAPTURE_QO:-4096}"
CAPTURE_SMALL_NUMEL="${CAPTURE_SMALL_NUMEL:-50000000}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
ARTIFACT_CREATED=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_FLASHINFER_MM_PREFIX=1
export VLLM_FLASHINFER_VOSPLIT=1
export VLLM_NVFP4_KV_VOSPLIT=1
export VLLM_NVFP4_KV_LINEAR_V_SF=1

finalize_artifact() {
  local rc=$?
  if [ -n "${OUT:-}" ] && [ -d "${OUT:-/nonexistent}" ] && [ "${ARTIFACT_CREATED}" = "0" ]; then
    {
      echo "exit_code=${rc}"
      date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ"
      find "${OUT}" -maxdepth 3 -type f -printf "%P\t%p\n" 2>/dev/null | sort || true
    } >"${OUT}/FINAL_STATUS.txt" || true
    tar -C "$(dirname "${OUT}")" -czf "${OUT}.tgz" "$(basename "${OUT}")" 2>/dev/null || true
    ARTIFACT_CREATED=1
  fi
  return "${rc}"
}
trap finalize_artifact EXIT

cd /root
[ -f wikitext_8k.txt ] || python corpus_fetch.py >/dev/null 2>&1
mkdir -p "${OUT}/rows" "${OUT}/calib" "${OUT}/active_kv" /root/active_kv_hook
cp /root/active_kv_capture_sitecustomize.py /root/active_kv_hook/sitecustomize.py

cat >"${OUT}/RUN_INFO.txt" <<EOF
schema=vllm-26b-a4b-active-kv-mix/v1
model=${MODEL}
ctx=${CTX}
prefix=${PREFIX}
max_model_len=${MAX_MODEL_LEN}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
gpu_memory_utilization=${GPU_UTIL}
prompt_logprobs=${PROMPT_LOGPROBS}
capture_max=${CAPTURE_MAX}
capture_qo=${CAPTURE_QO}
capture_small_numel=${CAPTURE_SMALL_NUMEL}
skip_mm_profiling=${SKIP_MM_PROFILING}
purpose=active-page bf16/NVFP4 K/V mix attribution: bf16K+NVFP4V vs NVFP4K+bf16V at FlashInfer prefill taps
EOF
python - <<'PY' >>"${OUT}/RUN_INFO.txt"
import os
import torch
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"device={torch.cuda.get_device_name()}")
print(f"capability={torch.cuda.get_device_capability()}")
print(f"MAX_JOBS={os.environ.get('MAX_JOBS', '')}")
PY

cat >"${OUT}/calib/base_k100.json" <<'JSON'
{
  "source": "26B-A4B active KV mix probe",
  "mode": "base_k100",
  "layer_type_scales": {
    "sliding_attention": {"k_scale": 0.07, "v_scale": 0.05},
    "full_attention": {"k_scale": 0.07, "v_scale": 0.05}
  },
  "layer_scales": {
    "0": {"k_scale": 0.1, "v_scale": 0.08},
    "1": {"k_scale": 0.1, "v_scale": 0.08},
    "2": {"k_scale": 0.1, "v_scale": 0.08},
    "3": {"k_scale": 0.1, "v_scale": 0.08},
    "4": {"k_scale": 0.1, "v_scale": 0.08},
    "6": {"k_scale": 0.1, "v_scale": 0.08},
    "7": {"k_scale": 0.1, "v_scale": 0.08},
    "8": {"k_scale": 0.1, "v_scale": 0.08},
    "9": {"k_scale": 0.1, "v_scale": 0.08},
    "10": {"k_scale": 0.1, "v_scale": 0.08},
    "12": {"k_scale": 0.1, "v_scale": 0.08},
    "13": {"k_scale": 0.1, "v_scale": 0.08},
    "14": {"k_scale": 0.1, "v_scale": 0.08},
    "15": {"k_scale": 0.1, "v_scale": 0.08},
    "16": {"k_scale": 0.1, "v_scale": 0.08},
    "18": {"k_scale": 0.1, "v_scale": 0.08},
    "19": {"k_scale": 0.1, "v_scale": 0.08},
    "20": {"k_scale": 0.1, "v_scale": 0.08},
    "21": {"k_scale": 0.1, "v_scale": 0.08},
    "22": {"k_scale": 0.1, "v_scale": 0.08}
  }
}
JSON

run_row() {
  local label="$1"
  local dtype="$2"
  local calib="${3:-}"
  echo "=== ${label} (${dtype}) ===" | tee -a "${OUT}/run.log"
  local capdir="${OUT}/active_kv/${label}"
  mkdir -p "${capdir}"
  local args=(
    python vllm_toplogprob_attribution.py
    --model "${MODEL}"
    --tokenizer "${MODEL}"
    --corpus wikitext_8k.txt
    --kv-cache-dtype "${dtype}"
    --ctx "${CTX}"
    --prefix-len "${PREFIX}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_UTIL}"
    --prompt-logprobs "${PROMPT_LOGPROBS}"
    --keep-topk 1
    --position-stride 512
    --dense-prefix-positions 0
    --output "${OUT}/rows/${label}.json"
    --enforce-eager
    --skip-warmup
  )
  if [ "${SKIP_MM_PROFILING}" = "1" ]; then
    args+=(--skip-mm-profiling)
  fi
  if [ -n "${calib}" ]; then
    args+=(--calib-json "${calib}")
  fi
  if timeout "${ROW_TIMEOUT}" env \
    PYTHONPATH="/root/active_kv_hook:/root/flashinfer:${PYTHONPATH:-}" \
    FI_ACTIVE_KV_CAPTURE_DIR="${capdir}" \
    FI_ACTIVE_KV_CAPTURE_QO="${CAPTURE_QO}" \
    FI_ACTIVE_KV_CAPTURE_MAX="${CAPTURE_MAX}" \
    FI_ACTIVE_KV_CAPTURE_SMALL_NUMEL="${CAPTURE_SMALL_NUMEL}" \
    "${args[@]}" 2>&1 | tee "${OUT}/rows/${label}.log"; then
    echo -e "${label}\t${dtype}\tok" >>"${OUT}/row_status.tsv"
    return 0
  fi
  local rc=$?
  echo -e "${label}\t${dtype}\tfailed_rc_${rc}" >>"${OUT}/row_status.tsv"
  return "${rc}"
}

echo -e "label\tdtype\tstatus" >"${OUT}/row_status.tsv"
run_row bf16 auto
run_row base_k100 nvfp4 "${OUT}/calib/base_k100.json"

python compare_active_kv_mixed_ref.py \
  --bf16 "${OUT}/active_kv/bf16" \
  --nvfp4 "${OUT}/active_kv/base_k100" \
  | tee "${OUT}/active_kv_mix_report.tsv"

python - "${OUT}" <<'PY' | tee "${OUT}/summary.tsv"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
base = json.loads((out / "rows" / "bf16.json").read_text())
print("label\tmean_nll\tdelta_vs_bf16\tppl\tmissing\tactive_kv_calls")
for row_path in sorted((out / "rows").glob("*.json")):
    row = json.loads(row_path.read_text())
    calls = len(list((out / "active_kv" / row_path.stem).glob("call_*.pt")))
    print(
        f"{row_path.stem}\t{row['mean_nll_nats']:.9f}\t"
        f"{row['mean_nll_nats'] - base['mean_nll_nats']:+.9f}\t"
        f"{row['ppl']:.6f}\t{row['num_missing_tokens']}\t{calls}"
    )
PY

tar -C "$(dirname "${OUT}")" -czf "${OUT}.tgz" "$(basename "${OUT}")"
ARTIFACT_CREATED=1
echo "ARTIFACT ${OUT}.tgz"
