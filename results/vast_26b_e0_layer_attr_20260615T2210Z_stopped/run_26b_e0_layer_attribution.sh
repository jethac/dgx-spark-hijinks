#!/usr/bin/env bash
# Layer attribution inside the first Gemma 4 26B-A4B sliding block.
#
# Block attribution localized the K=0.103,V=0.080 low-NLL collapse to e0
# (layers 0-4). This packet holds all other early/mid sliding layers at
# K=0.100,V=0.080, late sliding/full at K=0.070,V=0.050, and moves one e0
# layer at a time to K=0.103,V=0.080.
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be exported}"

MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
CTX="${CTX:-8185}"
PREFIX="${PREFIX:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_UTIL="${GPU_UTIL:-0.82}"
OUT="${OUT:-/root/dx26_e0_layer_attr_$(date -u +%Y%m%dT%H%M%SZ)}"
ROW_TIMEOUT="${ROW_TIMEOUT:-7200}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_FLASHINFER_MM_PREFIX=1
export VLLM_FLASHINFER_VOSPLIT=1
export VLLM_NVFP4_KV_VOSPLIT=1
export VLLM_NVFP4_KV_LINEAR_V_SF=1

cd /root
[ -f wikitext_8k.txt ] || python corpus_fetch.py >/dev/null 2>&1
mkdir -p "${OUT}/rows" "${OUT}/calib"

cat >"${OUT}/RUN_INFO.txt" <<EOF
schema=vllm-26b-a4b-e0-layer-attribution/v1
model=${MODEL}
ctx=${CTX}
prefix=${PREFIX}
max_model_len=${MAX_MODEL_LEN}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
gpu_memory_utilization=${GPU_UTIL}
purpose=e0 layer attribution: base k100 versus one layer 0-4 at k103
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
base_full = {"k_scale": 0.07, "v_scale": 0.05}
base_early_mid = {"k_scale": 0.100, "v_scale": 0.08}
hot = {"k_scale": 0.103, "v_scale": 0.08}

early_mid_blocks = {
    "e0": list(range(0, 5)),
    "e1": list(range(6, 11)),
    "m0": list(range(12, 17)),
    "m1": list(range(18, 23)),
}

cfg = {
    "source": "26B-A4B e0 layer attribution",
    "mode": mode,
    "layer_type_scales": {
        "sliding_attention": {"k_scale": 0.07, "v_scale": 0.05},
        "full_attention": base_full,
    },
    "layer_scales": {},
}

for layers in early_mid_blocks.values():
    for idx in layers:
        cfg["layer_scales"][str(idx)] = base_early_mid

if mode == "e0_all":
    hot_layers = range(0, 5)
elif mode.startswith("l"):
    hot_layers = [int(mode[1:])]
else:
    hot_layers = []

for idx in hot_layers:
    if idx < 0 or idx > 4:
        raise SystemExit(f"unsupported e0 layer {idx}")
    cfg["layer_scales"][str(idx)] = hot

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

for mode in base_k100 e0_all l0 l1 l2 l3 l4; do
  calib="${OUT}/calib/${mode}.json"
  write_calib "${calib}" "${mode}"
  run_row "${mode}" nvfp4 "${calib}" || true
done

python - "${OUT}" <<'PY' | tee "${OUT}/delta_report.tsv"
import json, math, sys
from pathlib import Path

out = Path(sys.argv[1])
base = json.loads((out / "rows" / "bf16.json").read_text())
base_by_index = {p["index"]: p for p in base["positions"] if math.isfinite(p["nll"])}

print("label\tmean_delta\tbucket\tbucket_delta")
for row_path in sorted((out / "rows").glob("*.json")):
    if row_path.stem == "bf16":
        continue
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
    (out / f"{row_path.stem}_top_deltas.json").write_text(
        json.dumps(
            {
                "most_lower_nll_than_bf16": sorted(deltas)[:20],
                "most_higher_nll_than_bf16": sorted(deltas)[-20:],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
PY

python - "${OUT}" <<'PY' | tee "${OUT}/summary.tsv"
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
bf16 = json.loads((out / "rows" / "bf16.json").read_text())
base_nll = bf16["mean_nll_nats"]
print("label\tmean_nll\tdelta_vs_bf16\tppl\tmissing\tkv_tokens\tconcurrency")
for row_path in sorted((out / "rows").glob("*.json")):
    row = json.loads(row_path.read_text())
    label = row_path.stem
    print(
        f"{label}\t{row['mean_nll_nats']:.9f}\t{row['mean_nll_nats'] - base_nll:+.9f}\t"
        f"{row['ppl']:.6f}\t{row['num_missing_tokens']}\t\t"
    )
PY

tar -C "$(dirname "${OUT}")" -czf "${OUT}.tgz" "$(basename "${OUT}")"
echo "ARTIFACT ${OUT}.tgz"
