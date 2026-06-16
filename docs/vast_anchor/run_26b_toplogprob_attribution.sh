#!/usr/bin/env bash
# Top-logprob/readout attribution for Gemma 4 26B-A4B full-NVFP4.
#
# Prior packets proved the 8k low-NLL collapse is distributed across positions
# and concentrated in the first sliding block, but not in one layer.  This
# packet compares bf16 against selected NVFP4 rows at the final prompt-logprob
# distribution exposed by vLLM, using top-k overlap/mass as a readout-side
# discriminator before adding heavier hidden-state hooks.
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be exported}"

MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
CTX="${CTX:-8185}"
PREFIX="${PREFIX:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_UTIL="${GPU_UTIL:-0.82}"
OUT="${OUT:-/root/dx26_toplogprob_attr_$(date -u +%Y%m%dT%H%M%SZ)}"
ROW_TIMEOUT="${ROW_TIMEOUT:-5400}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-20}"
KEEP_TOPK="${KEEP_TOPK:-20}"
POSITION_STRIDE="${POSITION_STRIDE:-16}"
DENSE_PREFIX_POSITIONS="${DENSE_PREFIX_POSITIONS:-256}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"

# Space-separated row labels.  Supported: base_k100 e0_all l0 l1 l2 l3 l4
ROWS="${ROWS:-base_k100 e0_all l0 l1}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_FLASHINFER_MM_PREFIX=1
export VLLM_FLASHINFER_VOSPLIT=1
export VLLM_NVFP4_KV_VOSPLIT=1
export VLLM_NVFP4_KV_LINEAR_V_SF=1

cd /root
[ -f wikitext_8k.txt ] || python corpus_fetch.py >/dev/null 2>&1
mkdir -p "${OUT}/rows" "${OUT}/calib"

cat >"${OUT}/RUN_INFO.txt" <<EOF
schema=vllm-26b-a4b-toplogprob-attribution/v1
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
skip_mm_profiling=${SKIP_MM_PROFILING}
rows=${ROWS}
purpose=readout attribution: top-k prompt logprob distribution bf16 vs selected NVFP4 early-block rows
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
    "source": "26B-A4B top-logprob attribution",
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
    try:
        layer = int(mode[1:])
    except ValueError as exc:
        raise SystemExit(f"unsupported layer mode {mode}") from exc
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
  timeout "${ROW_TIMEOUT}" "${args[@]}" 2>&1 | tee "${OUT}/rows/${label}.log"
}

run_row bf16 auto

for mode in ${ROWS}; do
  calib="${OUT}/calib/${mode}.json"
  write_calib "${calib}" "${mode}"
  run_row "${mode}" nvfp4 "${calib}" || true
done

python - "${OUT}" <<'PY' | tee "${OUT}/toplogprob_delta_report.tsv"
import json, math, sys
from pathlib import Path

out = Path(sys.argv[1])
base = json.loads((out / "rows" / "bf16.json").read_text())
base_summary = {p["index"]: p for p in base["position_summaries"] if math.isfinite(p["target_nll"])}
base_top = {
    p["index"]: {int(item["token_id"]): float(item["logprob"]) for item in p.get("top", [])}
    for p in base["sampled_toplogprobs"]
}

def bucket_for(rel):
    for lo, hi in [(0, 256), (256, 1024), (1024, 2048), (2048, 3072), (3072, 10**9)]:
        if lo <= rel < hi:
            return f"{lo}-{hi}"
    return "other"

print(
    "label\tbucket\tcount\tmean_target_delta\tmean_topk_jaccard\t"
    "top1_match_rate\tmean_top1_logprob_delta\tmean_topk_mass_delta"
)
for row_path in sorted((out / "rows").glob("*.json")):
    if row_path.stem == "bf16":
        continue
    row = json.loads(row_path.read_text())
    summaries = {p["index"]: p for p in row["position_summaries"] if math.isfinite(p["target_nll"])}
    top = {
        p["index"]: {int(item["token_id"]): float(item["logprob"]) for item in p.get("top", [])}
        for p in row["sampled_toplogprobs"]
    }
    buckets = {}
    for index, current in summaries.items():
        base_current = base_summary.get(index)
        if not base_current:
            continue
        b = bucket_for(current["rel_index"])
        rec = buckets.setdefault(
            b,
            {
                "target_delta": [],
                "jaccard": [],
                "top1_match": [],
                "top1_lp_delta": [],
                "topk_mass_delta": [],
            },
        )
        rec["target_delta"].append(current["target_nll"] - base_current["target_nll"])
        rec["topk_mass_delta"].append(current["topk_mass"] - base_current["topk_mass"])
        if index in top and index in base_top:
            cur_ids = set(top[index])
            base_ids = set(base_top[index])
            union = cur_ids | base_ids
            rec["jaccard"].append(len(cur_ids & base_ids) / len(union) if union else math.nan)
            rec["top1_match"].append(
                1.0 if current.get("top1_token_id") == base_current.get("top1_token_id") else 0.0
            )
            if current.get("top1_logprob") is not None and base_current.get("top1_logprob") is not None:
                rec["top1_lp_delta"].append(current["top1_logprob"] - base_current["top1_logprob"])
    for bucket, vals in buckets.items():
        def mean(name):
            xs = [x for x in vals[name] if math.isfinite(x)]
            return sum(xs) / len(xs) if xs else math.nan
        print(
            f"{row_path.stem}\t{bucket}\t{len(vals['target_delta'])}\t"
            f"{mean('target_delta'):+.9f}\t{mean('jaccard'):.9f}\t"
            f"{mean('top1_match'):.9f}\t{mean('top1_lp_delta'):+.9f}\t"
            f"{mean('topk_mass_delta'):+.9f}"
        )
PY

python - "${OUT}" <<'PY' | tee "${OUT}/summary.tsv"
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
bf16 = json.loads((out / "rows" / "bf16.json").read_text())
base_nll = bf16["mean_nll_nats"]
print("label\tmean_nll\tdelta_vs_bf16\tppl\tmissing\tsampled_positions")
for row_path in sorted((out / "rows").glob("*.json")):
    row = json.loads(row_path.read_text())
    print(
        f"{row_path.stem}\t{row['mean_nll_nats']:.9f}\t"
        f"{row['mean_nll_nats'] - base_nll:+.9f}\t{row['ppl']:.6f}\t"
        f"{row['num_missing_tokens']}\t{len(row['sampled_toplogprobs'])}"
    )
PY

tar -C "$(dirname "${OUT}")" -czf "${OUT}.tgz" "$(basename "${OUT}")"
echo "ARTIFACT ${OUT}.tgz"
