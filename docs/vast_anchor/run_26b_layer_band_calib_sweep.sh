#!/usr/bin/env bash
# 26B-A4B full-NVFP4 KV layer-band calibration reachability screen.
#
# This is the follow-up to run_26b_layer_type_calib_sweep.sh. The layer-type
# discriminator improved ctx=8185 from -0.388143 to -0.148307 nats/token but
# remained red, so this packet tests whether the remaining low-NLL bias is
# concentrated in early/mid/late sliding-attention bands.
#
# Requires a vLLM build that supports VLLM_NVFP4_KV_CALIB JSON with
# layer_type_scales and layer_scales (jethac/vllm commit 1c9686c61 or newer).
#
# Required env:
#   HF_TOKEN     Hugging Face token with Gemma access.
#
# Optional env:
#   MODEL        default google/gemma-4-26B-A4B-it
#   CTX          default 8185
#   PREFIX       default 4096
#   CANDIDATES   semicolon-separated rows:
#                label|base_sk,base_sv,base_fk,base_fv|early_sk,early_sv|mid_sk,mid_sv|late_sk,late_sv
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
OUT="${OUT:-/root/dx26_layer_band}"

# Gemma 4 26B-A4B has 30 decoder layers, with full-attention layers
# [5, 11, 17, 23, 29] and five sliding layers before each full layer.
EARLY_SLIDING="${EARLY_SLIDING:-0 1 2 3 4 6 7 8 9 10}"
MID_SLIDING="${MID_SLIDING:-12 13 14 15 16 18 19 20 21 22}"
LATE_SLIDING="${LATE_SLIDING:-24 25 26 27 28}"

# Base row is the layer-type best from results/vast_26b_layer_type_20260615T1545Z:
# sliding=(0.10,0.08), full=(0.07,0.05), delta -0.148307. The other rows test
# whether only a band needs the higher sliding scale, with the rest falling back
# to the prior global best sliding=(0.07,0.05), full=(0.07,0.05).
CANDIDATES="${CANDIDATES:-all_high|0.10,0.08,0.07,0.05|NA|NA|NA;early_high|0.07,0.05,0.07,0.05|0.10,0.08|NA|NA;mid_high|0.07,0.05,0.07,0.05|NA|0.10,0.08|NA;late_high|0.07,0.05,0.07,0.05|NA|NA|0.10,0.08;early_mid_high|0.07,0.05,0.07,0.05|0.10,0.08|0.10,0.08|NA;mid_late_high|0.07,0.05,0.07,0.05|NA|0.10,0.08|0.10,0.08;early_late_high|0.07,0.05,0.07,0.05|0.10,0.08|NA|0.10,0.08;early_only_k10_v05|0.07,0.05,0.07,0.05|0.10,0.05|NA|NA;mid_only_k10_v05|0.07,0.05,0.07,0.05|NA|0.10,0.05|NA;late_only_k10_v05|0.07,0.05,0.07,0.05|NA|NA|0.10,0.05}"

mkdir -p "${OUT}/rows" "${OUT}/calib"
[ -f wikitext_8k.txt ] || python corpus_fetch.py >/dev/null 2>&1

echo "schema=vllm-26b-a4b-nvfp4-layer-band-calib-sweep/v1" | tee "${OUT}/RUN_INFO.txt"
{
  echo "model=${MODEL}"
  echo "ctx=${CTX}"
  echo "prefix=${PREFIX}"
  echo "max_model_len=${MAX_MODEL_LEN}"
  echo "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
  echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  echo "early_sliding=${EARLY_SLIDING}"
  echo "mid_sliding=${MID_SLIDING}"
  echo "late_sliding=${LATE_SLIDING}"
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

write_calib() {
  local path="$1"
  local label="$2"
  local base="$3"
  local early="$4"
  local mid="$5"
  local late="$6"
  BASE="${base}" EARLY="${early}" MID="${mid}" LATE="${late}" \
  EARLY_SLIDING="${EARLY_SLIDING}" MID_SLIDING="${MID_SLIDING}" LATE_SLIDING="${LATE_SLIDING}" \
  LABEL="${label}" MODEL="${MODEL}" OUT_PATH="${path}" python - <<'PY'
import json
import os

def parse_pair(raw):
    if raw == "NA":
        return None
    left, right = raw.split(",", 1)
    return float(left), float(right)

base_sk, base_sv, base_fk, base_fv = map(float, os.environ["BASE"].split(","))
cfg = {
    "source": "26B-A4B layer-band calibration sweep",
    "model": os.environ["MODEL"],
    "candidate": os.environ["LABEL"],
    "layer_type_scales": {
        "sliding_attention": {"k_scale": base_sk, "v_scale": base_sv},
        "full_attention": {"k_scale": base_fk, "v_scale": base_fv},
    },
    "layer_scales": {},
}

for env_name, indices_name in [
    ("EARLY", "EARLY_SLIDING"),
    ("MID", "MID_SLIDING"),
    ("LATE", "LATE_SLIDING"),
]:
    pair = parse_pair(os.environ[env_name])
    if pair is None:
        continue
    k_scale, v_scale = pair
    for index in os.environ[indices_name].split():
        cfg["layer_scales"][index] = {"k_scale": k_scale, "v_scale": v_scale}

with open(os.environ["OUT_PATH"], "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, sort_keys=True)
    f.write("\n")
PY
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
printf "label\tbase_sk\tbase_sv\tbase_fk\tbase_fv\tearly\tmid\tlate\tmean_nll\tdelta_vs_bf16\tppl\tok\tchat\tstatus\n" >"${SUMMARY}"

IFS=';' read -r -a rows <<<"${CANDIDATES}"
for row in "${rows[@]}"; do
  IFS='|' read -r label base early mid late <<<"${row}"
  IFS=',' read -r base_sk base_sv base_fk base_fv <<<"${base}"
  calib="${OUT}/calib/${label}.json"
  write_calib "${calib}" "${label}" "${base}" "${early}" "${mid}" "${late}"
  echo "=== ${label} base=(${base}) early=${early} mid=${mid} late=${late} ==="
  status="ok"
  if ! VLLM_NVFP4_KV_CALIB="${calib}" run_anchor "${label}" nvfp4; then
    status="fail"
  fi
  python - <<PY >>"${SUMMARY}"
import json, pathlib
label="${label}"; base_sk="${base_sk}"; base_sv="${base_sv}"; base_fk="${base_fk}"; base_fv="${base_fv}"
early="${early}"; mid="${mid}"; late="${late}"; status="${status}"; bf=float("${BF16_NLL}")
p=pathlib.Path("${OUT}/rows/${label}.json")
if not p.exists():
    print(f"{label}\t{base_sk}\t{base_sv}\t{base_fk}\t{base_fv}\t{early}\t{mid}\t{late}\tNA\tNA\tNA\tFalse\t\t{status}")
else:
    d=json.load(open(p))
    n=float(d["score"]["mean_nll_nats"])
    ppl=d["score"].get("ppl")
    ok=bool(d.get("ok"))
    chat=d.get("chat_smoke",{}).get("generated","").replace("\n"," ")
    print(f"{label}\t{base_sk}\t{base_sv}\t{base_fk}\t{base_fv}\t{early}\t{mid}\t{late}\t{n:.9f}\t{n-bf:+.9f}\t{ppl}\t{ok}\t{chat[:80]}\t{status}")
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
print("label\tbase_sk\tbase_sv\tbase_fk\tbase_fv\tearly\tmid\tlate\tmean_nll\tdelta_vs_bf16\tok\tstatus")
for r in rows[:10]:
    print("\t".join(r[x] for x in ["label","base_sk","base_sv","base_fk","base_fv","early","mid","late","mean_nll","delta_vs_bf16","ok","status"]))
PY

tar -czf "${OUT}.tgz" -C "$(dirname "${OUT}")" "$(basename "${OUT}")"
echo "DONE_26B_LAYER_BAND_SWEEP ${OUT}.tgz"
