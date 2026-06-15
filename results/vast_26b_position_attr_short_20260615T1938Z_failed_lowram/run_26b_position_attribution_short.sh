#!/usr/bin/env bash
# Short per-position attribution packet for Gemma 4 26B-A4B full-NVFP4.
#
# This is the bounded discriminator after the K-refinement partial stop:
# compare the bf16 baseline against the prior best k=0.100 row and the
# adjacent collapse k=0.103 row. Use the full packet when these two rows show
# a useful position-level signal.
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be exported}"

MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
CTX="${CTX:-8185}"
PREFIX="${PREFIX:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_UTIL="${GPU_UTIL:-0.82}"
OUT="${OUT:-/root/dx26_position_attr_short_$(date -u +%Y%m%dT%H%M%SZ)}"
ROW_TIMEOUT="${ROW_TIMEOUT:-5400}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_FLASHINFER_MM_PREFIX=1
export VLLM_FLASHINFER_VOSPLIT=1
export VLLM_NVFP4_KV_VOSPLIT=1
export VLLM_NVFP4_KV_LINEAR_V_SF=1

cd /root
[ -f wikitext_8k.txt ] || python corpus_fetch.py >/dev/null 2>&1
mkdir -p "${OUT}/rows" "${OUT}/calib"

cat >"${OUT}/RUN_INFO.txt" <<EOF
schema=vllm-26b-a4b-position-attribution-short/v1
model=${MODEL}
ctx=${CTX}
prefix=${PREFIX}
max_model_len=${MAX_MODEL_LEN}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
gpu_memory_utilization=${GPU_UTIL}
purpose=short per-position attribution: bf16 vs prior-best k100 and collapse k103
EOF
python - <<'PY' >>"${OUT}/RUN_INFO.txt"
import torch
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"device={torch.cuda.get_device_name()}")
print(f"capability={torch.cuda.get_device_capability()}")
PY

write_calib() {
  local path="$1"
  local k="$2"
  python - "$path" "$k" <<'PY'
import json, sys
path, k = sys.argv[1], float(sys.argv[2])
full = {"k_scale": 0.07, "v_scale": 0.05}
early_mid = {"k_scale": k, "v_scale": 0.08}
cfg = {
    "source": "26B-A4B short position attribution after K-refinement stop",
    "layer_type_scales": {
        "sliding_attention": {"k_scale": 0.07, "v_scale": 0.05},
        "full_attention": full,
    },
    "layer_scales": {},
}
for idx in list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23)):
    cfg["layer_scales"][str(idx)] = early_mid
with open(path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
PY
}

run_row() {
  local label="$1"
  local dtype="$2"
  local calib="${3:-}"
  echo "=== ${label} (${dtype}) ===" | tee -a "${OUT}/run.log"
  local args=(
    python vllm_position_logprob_attribution.py
    --model "${MODEL}"
    --tokenizer "${MODEL}"
    --corpus wikitext_8k.txt
    --kv-cache-dtype "${dtype}"
    --ctx "${CTX}"
    --prefix-len "${PREFIX}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_UTIL}"
    --output "${OUT}/rows/${label}.json"
    --enforce-eager
    --skip-warmup
  )
  if [ -n "${calib}" ]; then
    args+=(--calib-json "${calib}")
  fi
  timeout "${ROW_TIMEOUT}" "${args[@]}" 2>&1 | tee "${OUT}/rows/${label}.log"
}

run_row bf16 auto

for spec in k100:0.100 k103:0.103; do
  label="${spec%%:*}"
  kval="${spec##*:}"
  calib="${OUT}/calib/${label}.json"
  write_calib "${calib}" "${kval}"
  run_row "${label}" nvfp4 "${calib}" || true
done

python - "${OUT}" <<'PY' | tee "${OUT}/delta_report.tsv"
import json, math, sys
from pathlib import Path

out = Path(sys.argv[1])
base = json.loads((out / "rows" / "bf16.json").read_text())
base_by_index = {p["index"]: p for p in base["positions"] if math.isfinite(p["nll"])}

print("label\tmean_delta\tbucket\tbucket_delta")
for row_path in sorted((out / "rows").glob("k*.json")):
    row = json.loads(row_path.read_text())
    deltas = []
    by_bucket = [(0,256), (256,1024), (1024,2048), (2048,3072), (3072,10**9)]
    bucket_vals = {b: [] for b in by_bucket}
    for p in row["positions"]:
        b = base_by_index.get(p["index"])
        if not b or not math.isfinite(p["nll"]):
            continue
        d = p["nll"] - b["nll"]
        deltas.append((d, p["index"], p["rel_index"], p["token"], p["nll"], b["nll"]))
        for bucket in by_bucket:
            if bucket[0] <= p["rel_index"] < bucket[1]:
                bucket_vals[bucket].append(d)
                break
    mean_delta = sum(d for d, *_ in deltas) / len(deltas)
    for bucket, vals in bucket_vals.items():
        if vals:
            print(f"{row_path.stem}\t{mean_delta:+.9f}\t{bucket[0]}-{bucket[1]}\t{sum(vals)/len(vals):+.9f}")
    top_low = sorted(deltas)[:20]
    top_high = sorted(deltas)[-20:]
    (out / f"{row_path.stem}_top_deltas.json").write_text(
        json.dumps({"most_lower_nll_than_bf16": top_low, "most_higher_nll_than_bf16": top_high}, indent=2),
        encoding="utf-8",
    )
PY

tar -C "$(dirname "${OUT}")" -czf "${OUT}.tgz" "$(basename "${OUT}")"
echo "ARTIFACT ${OUT}.tgz"
