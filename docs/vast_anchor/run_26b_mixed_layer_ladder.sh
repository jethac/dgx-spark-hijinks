#!/usr/bin/env bash
# Gemma 4 26B-A4B mixed per-layer KV ladder for sm120.
#
# Global row is NVFP4 K+V. Selected layers are overridden to fp8_e4m3 or auto
# through vLLM's --kv-cache-dtype-skip-layers style API:
#   --kv-cache-dtype nvfp4 --kv-cache-dtype-skip-layers 0=fp8_e4m3 ...
#
# The rows follow mail/0222: whole selected layers, not K-only/V-only.
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be exported; do not write it into this script}"

MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
CTX="${CTX:-8185}"
PREFIX="${PREFIX:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_UTIL="${GPU_UTIL:-0.82}"
OUT="${OUT:-/root/dx26_mixed_layer_ladder_$(date -u +%Y%m%dT%H%M%SZ)}"
ROW_TIMEOUT="${ROW_TIMEOUT:-5400}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-20}"
KEEP_TOPK="${KEEP_TOPK:-20}"
POSITION_STRIDE="${POSITION_STRIDE:-128}"
DENSE_PREFIX_POSITIONS="${DENSE_PREFIX_POSITIONS:-256}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
ARTIFACT_CREATED=0

# Measured 26B-A4B decoder map used across the prior active-KV traces.
FIRST_BLOCK="${FIRST_BLOCK:-0 1 2 3 4}"
FIRST_PLUS_ACTIVE="${FIRST_PLUS_ACTIVE:-0 1 2 3 4 5 6 7}"
ALL_SLIDING="${ALL_SLIDING:-0 1 2 3 4 6 7 8 9 10 12 13 14 15 16 18 19 20 21 22 24 25 26 27 28}"
ALL_GLOBAL="${ALL_GLOBAL:-5 11 17 23 29}"
ROWS="${ROWS:-bf16 base_k100 bf16_0_4 fp8_0_4 fp8_0_7 fp8_all_sliding fp8_all_global}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_FLASHINFER_MM_PREFIX=1
export VLLM_FLASHINFER_VOSPLIT=1
export VLLM_NVFP4_KV_VOSPLIT=1
export VLLM_NVFP4_KV_LINEAR_V_SF=1
export PATH=/root/v/bin:${PATH}
export PYTHONPATH=/root/flashinfer

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
mkdir -p "${OUT}/rows" "${OUT}/calib"

cat >"${OUT}/RUN_INFO.txt" <<EOF
schema=vllm-26b-a4b-mixed-layer-ladder/v1
model=${MODEL}
ctx=${CTX}
prefix=${PREFIX}
max_model_len=${MAX_MODEL_LEN}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
gpu_memory_utilization=${GPU_UTIL}
prompt_logprobs=${PROMPT_LOGPROBS}
keep_topk=${KEEP_TOPK}
position_stride=${POSITION_STRIDE}
dense_prefix_positions=${DENSE_PREFIX_POSITIONS}
first_block=${FIRST_BLOCK}
first_plus_active=${FIRST_PLUS_ACTIVE}
all_sliding=${ALL_SLIDING}
all_global=${ALL_GLOBAL}
rows=${ROWS}
wheel_ref=sm120a-wheels-4fcbf4c48
wheel_sha256=2e92cc1a0139b3b8e24a881a363098f921cd15efbeed0ab51562a1265d9fa919
vllm_ref=4fcbf4c48f1a90136ca562d61d4b12241f621473
purpose=whole-layer fp8 override ladder for 26B-A4B NVFP4 KV; distribution/top-k evidence, not NLL-only
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
  "source": "26B-A4B mixed layer ladder",
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

skip_args() {
  local dtype="$1"
  shift
  for layer in "$@"; do
    printf '%s=%s\n' "${layer}" "${dtype}"
  done
}

run_row() {
  local label="$1"
  local dtype="$2"
  local calib="${3:-}"
  local override_dtype="${4:-}"
  local layer_list="${5:-}"
  echo "=== ${label} dtype=${dtype} override=${override_dtype} layers=${layer_list} ===" | tee -a "${OUT}/run.log"
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
  if [ -n "${override_dtype}" ] && [ -n "${layer_list}" ]; then
    # shellcheck disable=SC2206
    local layers=( ${layer_list} )
    args+=(--kv-cache-dtype-skip-layers)
    while IFS= read -r entry; do
      args+=("${entry}")
    done < <(skip_args "${override_dtype}" "${layers[@]}")
  fi
  set +e
  timeout "${ROW_TIMEOUT}" "${args[@]}" 2>&1 | tee "${OUT}/rows/${label}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "${rc}" -eq 0 ]; then
    echo -e "${label}\t${dtype}\t${override_dtype}\t${layer_list}\tok" >>"${OUT}/row_status.tsv"
    return 0
  fi
  echo -e "${label}\t${dtype}\t${override_dtype}\t${layer_list}\tfailed_rc_${rc}" >>"${OUT}/row_status.tsv"
  return 0
}

echo -e "label\tdtype\toverride_dtype\tlayers\tstatus" >"${OUT}/row_status.tsv"

for row in ${ROWS}; do
  case "${row}" in
    bf16)
      run_row bf16 auto
      ;;
    base_k100)
      run_row base_k100 nvfp4 "${OUT}/calib/base_k100.json"
      ;;
    bf16_0_4)
      run_row bf16_0_4 nvfp4 "${OUT}/calib/base_k100.json" auto "${FIRST_BLOCK}"
      ;;
    fp8_0_4)
      run_row fp8_0_4 nvfp4 "${OUT}/calib/base_k100.json" fp8_e4m3 "${FIRST_BLOCK}"
      ;;
    fp8_0_7)
      run_row fp8_0_7 nvfp4 "${OUT}/calib/base_k100.json" fp8_e4m3 "${FIRST_PLUS_ACTIVE}"
      ;;
    fp8_all_sliding)
      run_row fp8_all_sliding nvfp4 "${OUT}/calib/base_k100.json" fp8_e4m3 "${ALL_SLIDING}"
      ;;
    fp8_all_global)
      run_row fp8_all_global nvfp4 "${OUT}/calib/base_k100.json" fp8_e4m3 "${ALL_GLOBAL}"
      ;;
    *)
      echo "unknown row ${row}" >&2
      exit 2
      ;;
  esac
done

python - "${OUT}" <<'PY' | tee "${OUT}/distribution.tsv"
import json
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = {}
for path in sorted((out / "rows").glob("*.json")):
    try:
        rows[path.stem] = json.loads(path.read_text())
    except Exception:
        pass

def pos_map(row):
    return {int(item["index"]): item for item in row.get("sampled_toplogprobs", [])}

def top_set(item, k=5):
    return {int(x["token_id"]) for x in item.get("top", [])[:k]}

def compare(label, ref_label):
    row = rows.get(label)
    ref = rows.get(ref_label)
    if not row or not ref:
        return None
    a = pos_map(row)
    b = pos_map(ref)
    common = sorted(set(a) & set(b))
    if not common:
        return None
    top1 = 0
    jacc = []
    target_abs = []
    for idx in common:
        ai = a[idx]
        bi = b[idx]
        if ai.get("top1_token_id") == bi.get("top1_token_id"):
            top1 += 1
        aset = top_set(ai)
        bset = top_set(bi)
        if aset or bset:
            jacc.append(len(aset & bset) / max(1, len(aset | bset)))
        av = ai.get("target_nll")
        bv = bi.get("target_nll")
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)) and math.isfinite(av) and math.isfinite(bv):
            target_abs.append(abs(av - bv))
    return {
        "common": len(common),
        "top1_match": top1 / len(common),
        "top5_jaccard": sum(jacc) / len(jacc) if jacc else float("nan"),
        "mean_abs_target_nll_diff": sum(target_abs) / len(target_abs) if target_abs else float("nan"),
    }

bf = rows.get("bf16")
base = rows.get("base_k100")
print("label\tmean_nll\tdelta_vs_bf16\tdelta_vs_base_k100\tppl\tok\tbf16_top1\tbf16_jaccard5\tbf16_abs_target_nll\tbase_top1\tbase_jaccard5\tbase_abs_target_nll")
for label, row in rows.items():
    nll = float(row["mean_nll_nats"])
    bf_delta = nll - float(bf["mean_nll_nats"]) if bf else float("nan")
    base_delta = nll - float(base["mean_nll_nats"]) if base else float("nan")
    c_bf = compare(label, "bf16") or {}
    c_base = compare(label, "base_k100") or {}
    print(
        f"{label}\t{nll:.9f}\t{bf_delta:+.9f}\t{base_delta:+.9f}\t{row['ppl']:.6f}\t{row.get('num_missing_tokens', 0) == 0}\t"
        f"{c_bf.get('top1_match', float('nan')):.9f}\t{c_bf.get('top5_jaccard', float('nan')):.9f}\t{c_bf.get('mean_abs_target_nll_diff', float('nan')):.9f}\t"
        f"{c_base.get('top1_match', float('nan')):.9f}\t{c_base.get('top5_jaccard', float('nan')):.9f}\t{c_base.get('mean_abs_target_nll_diff', float('nan')):.9f}"
    )
PY

python - "${OUT}" <<'PY' >"${OUT}/summary.md"
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
run_info = (out / "RUN_INFO.txt").read_text()
dist = list(csv.DictReader((out / "distribution.tsv").open(), delimiter="\t"))
status = (out / "row_status.tsv").read_text()

print("# 26B-A4B Mixed Whole-Layer KV Ladder")
print()
print("Scope: Vast/sm120 vLLM offline supplied-token discriminator for Gemma 4 26B-A4B. This is a distribution and PPL ladder, not a broad support claim by itself.")
print()
print("## Run Info")
print()
print("```text")
print(run_info.rstrip())
print("```")
print()
print("## Row Status")
print()
print("```text")
print(status.rstrip())
print("```")
print()
print("## Distribution Metrics")
print()
print("| row | delta vs bf16 | delta vs base NVFP4 | bf16 top1 | bf16 Jaccard@5 | base top1 | base Jaccard@5 |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for row in dist:
    print(
        f"| `{row['label']}` | `{row['delta_vs_bf16']}` | `{row['delta_vs_base_k100']}` | "
        f"`{row['bf16_top1']}` | `{row['bf16_jaccard5']}` | `{row['base_top1']}` | `{row['base_jaccard5']}` |"
    )
print()
print("Interpretation to be filled after reviewing logs and row failures.")
PY

tar -C "$(dirname "${OUT}")" -czf "${OUT}.tgz" "$(basename "${OUT}")"
echo "ARTIFACT ${OUT}.tgz"
ARTIFACT_CREATED=1
