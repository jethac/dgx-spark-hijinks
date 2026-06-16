#!/usr/bin/env bash
# Hidden/readout attribution for Gemma 4 26B-A4B full-NVFP4.
#
# This wraps the existing top-logprob helper with an opt-in sitecustomize hook
# that captures sampled final hidden rows plus raw top-k logits at
# compute_logits().  It distinguishes upstream hidden-state drift from
# lm_head/readout-distribution amplification.
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be exported}"

MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
CTX="${CTX:-8185}"
PREFIX="${PREFIX:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_UTIL="${GPU_UTIL:-0.82}"
OUT="${OUT:-/root/dx26_readout_capture_$(date -u +%Y%m%dT%H%M%SZ)}"
ROW_TIMEOUT="${ROW_TIMEOUT:-5400}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-20}"
KEEP_TOPK="${KEEP_TOPK:-20}"
POSITION_STRIDE="${POSITION_STRIDE:-16}"
DENSE_PREFIX_POSITIONS="${DENSE_PREFIX_POSITIONS:-256}"
READOUT_TOPK="${READOUT_TOPK:-64}"
READOUT_DENSE_ROWS="${READOUT_DENSE_ROWS:-64}"
READOUT_ROW_STRIDE="${READOUT_ROW_STRIDE:-64}"
READOUT_MAX_CALLS="${READOUT_MAX_CALLS:-16}"
LAYER_CAPTURE="${LAYER_CAPTURE:-1}"
LAYER_CAPTURE_LAYERS="${LAYER_CAPTURE_LAYERS:-0,1,2,3,4}"
LAYER_CAPTURE_MAX_CALLS="${LAYER_CAPTURE_MAX_CALLS:-80}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
ROWS="${ROWS:-base_k100 e0_all l0 l1}"
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
mkdir -p "${OUT}/rows" "${OUT}/calib" "${OUT}/readout_capture" /root/readout_hook
cp /root/vllm_readout_capture_sitecustomize.py /root/readout_hook/sitecustomize.py

cat >"${OUT}/RUN_INFO.txt" <<EOF
schema=vllm-26b-a4b-readout-capture/v1
model=${MODEL}
ctx=${CTX}
prefix=${PREFIX}
max_model_len=${MAX_MODEL_LEN}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
gpu_memory_utilization=${GPU_UTIL}
prompt_logprobs=${PROMPT_LOGPROBS}
readout_topk=${READOUT_TOPK}
readout_dense_rows=${READOUT_DENSE_ROWS}
readout_row_stride=${READOUT_ROW_STRIDE}
readout_max_calls=${READOUT_MAX_CALLS}
layer_capture=${LAYER_CAPTURE}
layer_capture_layers=${LAYER_CAPTURE_LAYERS}
layer_capture_max_calls=${LAYER_CAPTURE_MAX_CALLS}
skip_mm_profiling=${SKIP_MM_PROFILING}
rows=${ROWS}
purpose=hidden/readout attribution: layer 0-4 phase drift + final hidden drift + raw logits top-k bf16 vs selected NVFP4 early-block rows
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

write_calib() {
  local path="$1"
  local mode="$2"
  python - "$path" "$mode" <<'PY'
import json, sys

path, mode = sys.argv[1], sys.argv[2]
full = {"k_scale": 0.07, "v_scale": 0.05}
base_early_mid = {"k_scale": 0.100, "v_scale": 0.080}
hot = {"k_scale": 0.103, "v_scale": 0.080}

cfg = {
    "source": "26B-A4B readout capture",
    "mode": mode,
    "layer_type_scales": {
        "sliding_attention": {"k_scale": 0.07, "v_scale": 0.05},
        "full_attention": full,
    },
    "layer_scales": {},
}

early_mid = list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
for idx in early_mid:
    cfg["layer_scales"][str(idx)] = base_early_mid

if mode == "e0_all":
    for idx in range(0, 5):
        cfg["layer_scales"][str(idx)] = hot
elif mode.startswith("l"):
    layer = int(mode[1:])
    if layer < 0 or layer > 4:
        raise SystemExit(f"layer mode {mode} is outside e0")
    cfg["layer_scales"][str(layer)] = hot
elif mode == "base_k100":
    pass
else:
    raise SystemExit(f"unsupported mode {mode}")

with open(path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
PY
}

run_row() {
  local label="$1"
  local dtype="$2"
  local calib="${3:-}"
  echo "=== ${label} (${dtype}) ===" | tee -a "${OUT}/run.log"
  local capdir="${OUT}/readout_capture/${label}"
  local layer_capdir="${OUT}/layer_capture/${label}"
  mkdir -p "${capdir}" "${layer_capdir}"
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
    --keep-topk "${KEEP_TOPK}"
    --position-stride "${POSITION_STRIDE}"
    --dense-prefix-positions "${DENSE_PREFIX_POSITIONS}"
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
  local env_args=(
    PYTHONPATH="/root/readout_hook:/root/flashinfer:${PYTHONPATH:-}"
    VLLM_READOUT_CAPTURE_DIR="${capdir}"
    VLLM_READOUT_CAPTURE_TOPK="${READOUT_TOPK}"
    VLLM_READOUT_CAPTURE_DENSE_ROWS="${READOUT_DENSE_ROWS}"
    VLLM_READOUT_CAPTURE_ROW_STRIDE="${READOUT_ROW_STRIDE}"
    VLLM_READOUT_CAPTURE_MAX_CALLS="${READOUT_MAX_CALLS}"
  )
  if [ "${LAYER_CAPTURE}" = "1" ]; then
    env_args+=(
      VLLM_LAYER_CAPTURE_DIR="${layer_capdir}"
      VLLM_LAYER_CAPTURE_LAYERS="${LAYER_CAPTURE_LAYERS}"
      VLLM_LAYER_CAPTURE_MAX_CALLS="${LAYER_CAPTURE_MAX_CALLS}"
    )
  fi
  if timeout "${ROW_TIMEOUT}" env \
    "${env_args[@]}" \
    "${args[@]}" 2>&1 | tee "${OUT}/rows/${label}.log"; then
    echo -e "${label}\t${dtype}\tok" >>"${OUT}/row_status.tsv"
    return 0
  fi
  local rc=$?
  echo -e "${label}\t${dtype}\tfailed_rc_${rc}" >>"${OUT}/row_status.tsv"
  return "${rc}"
}

echo -e "label\tdtype\tstatus" >"${OUT}/row_status.tsv"

if ! run_row bf16 auto; then
  echo "bf16 row failed; stopping before NVFP4 comparisons" | tee -a "${OUT}/run.log"
  exit 1
fi

compare_args=(--base "${OUT}/readout_capture/bf16")
for mode in ${ROWS}; do
  calib="${OUT}/calib/${mode}.json"
  write_calib "${calib}" "${mode}"
  if run_row "${mode}" nvfp4 "${calib}"; then
    compare_args+=(--compare "${mode}=${OUT}/readout_capture/${mode}")
  fi
done

python compare_readout_captures.py "${compare_args[@]}" | tee "${OUT}/readout_capture_report.tsv"
if [ "${LAYER_CAPTURE}" = "1" ]; then
  layer_compare_args=(--base "${OUT}/layer_capture/bf16")
  for mode in ${ROWS}; do
    if [ -d "${OUT}/layer_capture/${mode}" ]; then
      layer_compare_args+=(--compare "${mode}=${OUT}/layer_capture/${mode}")
    fi
  done
  python compare_layer_captures.py "${layer_compare_args[@]}" | tee "${OUT}/layer_capture_report.tsv"
fi

python - "${OUT}" <<'PY' | tee "${OUT}/summary.tsv"
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
bf16 = json.loads((out / "rows" / "bf16.json").read_text())
base_nll = bf16["mean_nll_nats"]
print("label\tmean_nll\tdelta_vs_bf16\tppl\tmissing\treadout_calls\tlayer_calls")
for row_path in sorted((out / "rows").glob("*.json")):
    row = json.loads(row_path.read_text())
    cap_calls = len(list((out / "readout_capture" / row_path.stem).glob("readout_call_*.pt")))
    layer_calls = len(list((out / "layer_capture" / row_path.stem).glob("layer_*_call_*.pt")))
    print(
        f"{row_path.stem}\t{row['mean_nll_nats']:.9f}\t"
        f"{row['mean_nll_nats'] - base_nll:+.9f}\t{row['ppl']:.6f}\t"
        f"{row['num_missing_tokens']}\t{cap_calls}\t{layer_calls}"
    )
PY

tar -C "$(dirname "${OUT}")" -czf "${OUT}.tgz" "$(basename "${OUT}")"
ARTIFACT_CREATED=1
echo "ARTIFACT ${OUT}.tgz"
