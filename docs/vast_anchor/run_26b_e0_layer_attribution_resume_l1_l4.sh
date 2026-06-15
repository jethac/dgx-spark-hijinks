#!/usr/bin/env bash
# Resume-only packet for the Gemma 4 26B-A4B e0 layer attribution run.
#
# The original run completed bf16/base_k100/e0_all/l0, then was stopped during
# l1/l2 startup. This packet runs only l1-l4 with the same calibration logic.
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be exported}"

MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
CTX="${CTX:-8185}"
PREFIX="${PREFIX:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_UTIL="${GPU_UTIL:-0.82}"
OUT="${OUT:-/root/dx26_e0_layer_attr_resume_$(date -u +%Y%m%dT%H%M%SZ)}"
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
schema=vllm-26b-a4b-e0-layer-attribution-resume/v1
model=${MODEL}
ctx=${CTX}
prefix=${PREFIX}
max_model_len=${MAX_MODEL_LEN}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
gpu_memory_utilization=${GPU_UTIL}
purpose=resume e0 layer attribution: run l1-l4 only after stopped bf16/base/e0_all/l0 run
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
    "source": "26B-A4B e0 layer attribution resume l1-l4",
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

if not mode.startswith("l"):
    raise SystemExit(f"unsupported resume mode {mode}")
hot_layer = int(mode[1:])
if hot_layer < 1 or hot_layer > 4:
    raise SystemExit(f"resume packet only supports layers 1-4, got {hot_layer}")
cfg["layer_scales"][str(hot_layer)] = hot

with open(path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
PY
}

run_row() {
  local label="$1"
  local calib="$2"
  echo "=== ${label} (nvfp4) ===" | tee -a "${OUT}/run.log"
  timeout "${ROW_TIMEOUT}" \
    python vllm_position_logprob_attribution.py \
      --model "${MODEL}" \
      --tokenizer "${MODEL}" \
      --corpus wikitext_8k.txt \
      --kv-cache-dtype nvfp4 \
      --ctx "${CTX}" \
      --prefix-len "${PREFIX}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${GPU_UTIL}" \
      --output "${OUT}/rows/${label}.json" \
      --enforce-eager \
      --skip-warmup \
      --calib-json "${calib}" \
    2>&1 | tee "${OUT}/rows/${label}.log"
}

for mode in l1 l2 l3 l4; do
  calib="${OUT}/calib/${mode}.json"
  write_calib "${calib}" "${mode}"
  run_row "${mode}" "${calib}" || true
done

python - "${OUT}" <<'PY' | tee "${OUT}/summary.tsv"
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
print("label\tmean_nll\tppl\tmissing")
for row_path in sorted((out / "rows").glob("*.json")):
    row = json.loads(row_path.read_text())
    print(f"{row_path.stem}\t{row['mean_nll_nats']:.9f}\t{row['ppl']:.6f}\t{row['num_missing_tokens']}")
PY

tar -C "$(dirname "${OUT}")" -czf "${OUT}.tgz" "$(basename "${OUT}")"
echo "ARTIFACT ${OUT}.tgz"

