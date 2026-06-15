#!/usr/bin/env bash
# 26B-A4B full-NVFP4 KV layer-type calibration reachability screen.
#
# Requires a vLLM build that supports VLLM_NVFP4_KV_CALIB JSON with
# layer_type_scales. This tests whether the ctx=8185 low-NLL bias is separable
# between Gemma's sliding_attention and full_attention layers.
#
# Required env:
#   HF_TOKEN     Hugging Face token with Gemma access.
#
# Optional env:
#   MODEL        default google/gemma-4-26B-A4B-it
#   CTX          default 8185
#   PREFIX       default 4096
#   CANDIDATES   space-separated "sliding_k,sliding_v,full_k,full_v" rows
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN in the environment; do not write it into this script}"
export HF_TOKEN
export PATH=/root/v/bin:${PATH}
export PYTHONPATH=/root/flashinfer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_FLASHINFER_MM_PREFIX=1
export VLLM_FLASHINFER_VOSPLIT=1
export VLLM_NVFP4_KV_VOSPLIT=1
export VLLM_NVFP4_KV_LINEAR_V_SF=1

cd /root
MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
CTX="${CTX:-8185}"
PREFIX="${PREFIX:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
ROW_TIMEOUT="${ROW_TIMEOUT:-1800}"
OUT="${OUT:-/root/dx26_layer_type}"
CANDIDATES="${CANDIDATES:-0.07,0.05,0.07,0.05 0.10,0.05,0.07,0.05 0.07,0.05,0.10,0.05 0.10,0.05,0.10,0.05 0.07,0.08,0.07,0.05 0.07,0.05,0.05,0.05 0.10,0.05,0.05,0.05 0.07,0.05,0.14,0.05 0.07,0.10,0.07,0.05 0.05,0.08,0.07,0.05 0.10,0.08,0.07,0.05 0.07,0.05,0.07,0.10}"

mkdir -p "${OUT}/rows" "${OUT}/calib"
[ -f wikitext_8k.txt ] || python corpus_fetch.py >/dev/null 2>&1

echo "schema=vllm-26b-a4b-nvfp4-layer-type-calib-sweep/v1" | tee "${OUT}/RUN_INFO.txt"
{
  echo "model=${MODEL}"
  echo "ctx=${CTX}"
  echo "prefix=${PREFIX}"
  echo "max_model_len=${MAX_MODEL_LEN}"
  echo "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
  echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  echo "device=$(python - <<'PY'
import torch
print(torch.cuda.get_device_name() if torch.cuda.is_available() else "no-cuda")
PY
)"
  echo "capability=$(python - <<'PY'
import torch
print(torch.cuda.get_device_capability() if torch.cuda.is_available() else "no-cuda")
PY
)"
  echo "candidates=${CANDIDATES}"
} | tee -a "${OUT}/RUN_INFO.txt"

echo "=== HF eager bf16 truth ==="
python hf_ref_ppl.py "${MODEL}" wikitext_8k.txt "${CTX}" "${PREFIX}" \
  >"${OUT}/hf_truth.out" 2>"${OUT}/hf_truth.err" || true
cat "${OUT}/hf_truth.out" || true

run_anchor() {
  local label="$1"
  local kv_dtype="$2"
  shift 2
  timeout "${ROW_TIMEOUT}" python vllm_matched_kv_anchor.py \
    --model "${MODEL}" \
    --tokenizer "${MODEL}" \
    --corpus wikitext_8k.txt \
    --kv-cache-dtype "${kv_dtype}" \
    --ctx "${CTX}" \
    --prefix-len "${PREFIX}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --skip-warmup \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --output "${OUT}/rows/${label}.json" \
    --enforce-eager \
    "$@" >"${OUT}/rows/${label}.log" 2>&1
}

echo "=== bf16 vLLM baseline ==="
if run_anchor bf16 auto; then
  BF16_NLL="$(python - <<PY
import json
print(json.load(open("${OUT}/rows/bf16.json"))["score"]["mean_nll_nats"])
PY
)"
else
  echo "bf16 baseline failed" | tee "${OUT}/FAILED"
  tail -80 "${OUT}/rows/bf16.log" || true
  exit 2
fi
echo "bf16_nll=${BF16_NLL}" | tee -a "${OUT}/RUN_INFO.txt"

SUMMARY="${OUT}/summary.tsv"
printf "label\tsliding_k\tsliding_v\tfull_k\tfull_v\tmean_nll\tdelta_vs_bf16\tppl\tok\tchat\tstatus\n" >"${SUMMARY}"

for row in ${CANDIDATES}; do
  IFS=',' read -r sk sv fk fv <<<"${row}"
  label="nvfp4_slk${sk}_slv${sv}_fk${fk}_fv${fv}"
  label="${label//./p}"
  calib="${OUT}/calib/${label}.json"
  cat >"${calib}" <<JSON
{
  "source": "26B-A4B layer-type calibration sweep",
  "model": "${MODEL}",
  "layer_type_scales": {
    "sliding_attention": {"k_scale": ${sk}, "v_scale": ${sv}},
    "full_attention": {"k_scale": ${fk}, "v_scale": ${fv}}
  }
}
JSON
  echo "=== ${label} sliding=(${sk},${sv}) full=(${fk},${fv}) ==="
  status="ok"
  if ! VLLM_NVFP4_KV_CALIB="${calib}" run_anchor "${label}" nvfp4; then
    status="fail"
  fi
  python - <<PY >>"${SUMMARY}"
import json, pathlib
label="${label}"; sk="${sk}"; sv="${sv}"; fk="${fk}"; fv="${fv}"; status="${status}"; bf=float("${BF16_NLL}")
p=pathlib.Path("${OUT}/rows/${label}.json")
if not p.exists():
    print(f"{label}\t{sk}\t{sv}\t{fk}\t{fv}\tNA\tNA\tNA\tFalse\t\t{status}")
else:
    d=json.load(open(p))
    n=float(d["score"]["mean_nll_nats"])
    ppl=d["score"].get("ppl")
    ok=bool(d.get("ok"))
    chat=d.get("chat_smoke",{}).get("generated","").replace("\n"," ")
    print(f"{label}\t{sk}\t{sv}\t{fk}\t{fv}\t{n:.9f}\t{n-bf:+.9f}\t{ppl}\t{ok}\t{chat[:80]}\t{status}")
PY
  tail -1 "${SUMMARY}"
done

python - <<PY | tee "${OUT}/best.tsv"
import csv
rows=[]
with open("${SUMMARY}") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        try:
            r["_abs_delta"]=abs(float(r["delta_vs_bf16"]))
            rows.append(r)
        except Exception:
            pass
rows.sort(key=lambda r: r["_abs_delta"])
print("label\tsliding_k\tsliding_v\tfull_k\tfull_v\tmean_nll\tdelta_vs_bf16\tok\tstatus")
for r in rows[:10]:
    print("\t".join(r[x] for x in ["label","sliding_k","sliding_v","full_k","full_v","mean_nll","delta_vs_bf16","ok","status"]))
PY

tar -czf "${OUT}.tgz" -C "$(dirname "${OUT}")" "$(basename "${OUT}")"
echo "DONE_26B_LAYER_TYPE_SWEEP ${OUT}.tgz"
