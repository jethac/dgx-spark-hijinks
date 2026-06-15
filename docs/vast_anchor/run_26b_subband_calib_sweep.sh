#!/usr/bin/env bash
# 26B-A4B full-NVFP4 KV sub-band calibration reachability screen.
#
# Follow-up to run_26b_layer_band_calib_sweep.sh. The best ctx=8185 row was:
#   full layers: 0.07/0.05
#   late sliding: 0.07/0.05
#   early+mid sliding: 0.10/0.08
# with delta -0.117964257 nats/token vs vLLM bf16. This packet splits the
# early+mid sliding layers into four five-layer blocks and searches that local
# surface before falling back to non-calibration explanations.
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
#                label|base_sk,base_sv,base_fk,base_fv|e0|e1|m0|m1
#                each block value is K,V or NA.
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
OUT="${OUT:-/root/dx26_subband}"

# 26B-A4B text layer map: full attention at [5, 11, 17, 23, 29].
E0_SLIDING="${E0_SLIDING:-0 1 2 3 4}"
E1_SLIDING="${E1_SLIDING:-6 7 8 9 10}"
M0_SLIDING="${M0_SLIDING:-12 13 14 15 16}"
M1_SLIDING="${M1_SLIDING:-18 19 20 21 22}"
LATE_SLIDING="${LATE_SLIDING:-24 25 26 27 28}"

# Base sliding/full scale is the prior global best 0.07/0.05; late sliding
# intentionally remains base. The first row replays the previous layer-band
# best; the next rows identify which five-layer blocks matter; the final rows
# refine the winning all-four-block scale around V=0.08.
CANDIDATES="${CANDIDATES:-all_four_high|0.07,0.05,0.07,0.05|0.10,0.08|0.10,0.08|0.10,0.08|0.10,0.08;e0_only|0.07,0.05,0.07,0.05|0.10,0.08|NA|NA|NA;e1_only|0.07,0.05,0.07,0.05|NA|0.10,0.08|NA|NA;m0_only|0.07,0.05,0.07,0.05|NA|NA|0.10,0.08|NA;m1_only|0.07,0.05,0.07,0.05|NA|NA|NA|0.10,0.08;e0_e1|0.07,0.05,0.07,0.05|0.10,0.08|0.10,0.08|NA|NA;m0_m1|0.07,0.05,0.07,0.05|NA|NA|0.10,0.08|0.10,0.08;e0_m0|0.07,0.05,0.07,0.05|0.10,0.08|NA|0.10,0.08|NA;e0_m1|0.07,0.05,0.07,0.05|0.10,0.08|NA|NA|0.10,0.08;e1_m0|0.07,0.05,0.07,0.05|NA|0.10,0.08|0.10,0.08|NA;e1_m1|0.07,0.05,0.07,0.05|NA|0.10,0.08|NA|0.10,0.08;drop_e0|0.07,0.05,0.07,0.05|NA|0.10,0.08|0.10,0.08|0.10,0.08;drop_e1|0.07,0.05,0.07,0.05|0.10,0.08|NA|0.10,0.08|0.10,0.08;drop_m0|0.07,0.05,0.07,0.05|0.10,0.08|0.10,0.08|NA|0.10,0.08;drop_m1|0.07,0.05,0.07,0.05|0.10,0.08|0.10,0.08|0.10,0.08|NA;all_four_k10_v07|0.07,0.05,0.07,0.05|0.10,0.07|0.10,0.07|0.10,0.07|0.10,0.07;all_four_k10_v09|0.07,0.05,0.07,0.05|0.10,0.09|0.10,0.09|0.10,0.09|0.10,0.09;all_four_k09_v08|0.07,0.05,0.07,0.05|0.09,0.08|0.09,0.08|0.09,0.08|0.09,0.08;all_four_k11_v08|0.07,0.05,0.07,0.05|0.11,0.08|0.11,0.08|0.11,0.08|0.11,0.08}"

mkdir -p "${OUT}/rows" "${OUT}/calib"
[ -f wikitext_8k.txt ] || python corpus_fetch.py >/dev/null 2>&1

echo "schema=vllm-26b-a4b-nvfp4-subband-calib-sweep/v1" | tee "${OUT}/RUN_INFO.txt"
{
  echo "model=${MODEL}"
  echo "ctx=${CTX}"
  echo "prefix=${PREFIX}"
  echo "max_model_len=${MAX_MODEL_LEN}"
  echo "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
  echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  echo "e0_sliding=${E0_SLIDING}"
  echo "e1_sliding=${E1_SLIDING}"
  echo "m0_sliding=${M0_SLIDING}"
  echo "m1_sliding=${M1_SLIDING}"
  echo "late_sliding_base_only=${LATE_SLIDING}"
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
  local e0="$4"
  local e1="$5"
  local m0="$6"
  local m1="$7"
  BASE="${base}" E0="${e0}" E1="${e1}" M0="${m0}" M1="${m1}" \
  E0_SLIDING="${E0_SLIDING}" E1_SLIDING="${E1_SLIDING}" \
  M0_SLIDING="${M0_SLIDING}" M1_SLIDING="${M1_SLIDING}" \
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
    "source": "26B-A4B sub-band calibration sweep",
    "model": os.environ["MODEL"],
    "candidate": os.environ["LABEL"],
    "layer_type_scales": {
        "sliding_attention": {"k_scale": base_sk, "v_scale": base_sv},
        "full_attention": {"k_scale": base_fk, "v_scale": base_fv},
    },
    "layer_scales": {},
}

for env_name, indices_name in [
    ("E0", "E0_SLIDING"),
    ("E1", "E1_SLIDING"),
    ("M0", "M0_SLIDING"),
    ("M1", "M1_SLIDING"),
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
printf "label\tbase_sk\tbase_sv\tbase_fk\tbase_fv\te0\te1\tm0\tm1\tmean_nll\tdelta_vs_bf16\tppl\tok\tchat\tstatus\n" >"${SUMMARY}"

IFS=';' read -r -a rows <<<"${CANDIDATES}"
for row in "${rows[@]}"; do
  IFS='|' read -r label base e0 e1 m0 m1 <<<"${row}"
  IFS=',' read -r base_sk base_sv base_fk base_fv <<<"${base}"
  calib="${OUT}/calib/${label}.json"
  write_calib "${calib}" "${label}" "${base}" "${e0}" "${e1}" "${m0}" "${m1}"
  echo "=== ${label} base=(${base}) e0=${e0} e1=${e1} m0=${m0} m1=${m1} ==="
  status="ok"
  if ! VLLM_NVFP4_KV_CALIB="${calib}" run_anchor "${label}" nvfp4; then
    status="fail"
  fi
  python - <<PY >>"${SUMMARY}"
import json, pathlib
label="${label}"; base_sk="${base_sk}"; base_sv="${base_sv}"; base_fk="${base_fk}"; base_fv="${base_fv}"
e0="${e0}"; e1="${e1}"; m0="${m0}"; m1="${m1}"; status="${status}"; bf=float("${BF16_NLL}")
p=pathlib.Path("${OUT}/rows/${label}.json")
if not p.exists():
    print(f"{label}\t{base_sk}\t{base_sv}\t{base_fk}\t{base_fv}\t{e0}\t{e1}\t{m0}\t{m1}\tNA\tNA\tNA\tFalse\t\t{status}")
else:
    d=json.load(open(p))
    n=float(d["score"]["mean_nll_nats"])
    ppl=d["score"].get("ppl")
    ok=bool(d.get("ok"))
    chat=d.get("chat_smoke",{}).get("generated","").replace("\n"," ")
    print(f"{label}\t{base_sk}\t{base_sv}\t{base_fk}\t{base_fv}\t{e0}\t{e1}\t{m0}\t{m1}\t{n:.9f}\t{n-bf:+.9f}\t{ppl}\t{ok}\t{chat[:80]}\t{status}")
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
print("label\tbase_sk\tbase_sv\tbase_fk\tbase_fv\te0\te1\tm0\tm1\tmean_nll\tdelta_vs_bf16\tok\tstatus")
for r in rows[:12]:
    print("\t".join(r[x] for x in ["label","base_sk","base_sv","base_fk","base_fv","e0","e1","m0","m1","mean_nll","delta_vs_bf16","ok","status"]))
PY

tar -czf "${OUT}.tgz" -C "$(dirname "${OUT}")" "$(basename "${OUT}")"
echo "DONE_26B_SUBBAND_SWEEP ${OUT}.tgz"
