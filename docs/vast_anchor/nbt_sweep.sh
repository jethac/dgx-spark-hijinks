#!/usr/bin/env bash
# Localize the single-vs-chunked nvfp4 gap on the real serving path: sweep
# max_num_batched_tokens (the prefill chunk boundary) for a single 8185-token request,
# scoring the same suffix. If Δ(nvfp4-bf16) tracks the chunk boundary monotonically,
# the gap is a chunked-prefill mechanic; if flat, it isn't.
set -uo pipefail
cd "$(dirname "$0")"
MODEL="${ANCHOR_MODEL:-google/gemma-4-12b-it}"
CORPUS="${CORPUS:-/root/wikitext_8k.txt}"
OUT=/root/pfx_out; mkdir -p "$OUT"
export VLLM_FLASHINFER_MM_PREFIX=1 VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run () { # kv nbt
  local kv="$1" nbt="$2" name="nbt${2}_${1/auto/bf16}"
  echo "=== ARM $name (kv=$kv nbt=$nbt) ==="
  python vllm_matched_kv_anchor.py --model "$MODEL" --tokenizer "$MODEL" \
    --corpus "$CORPUS" --kv-cache-dtype "$kv" --ctx 8185 --prefix-len 4096 \
    --max-model-len 8192 --max-num-batched-tokens "$nbt" --skip-warmup \
    --gpu-memory-utilization 0.5 \
    --output "$OUT/${name}.json" --enforce-eager 2>&1 | tail -2
}

for NBT in 2048 4096 6144 8192; do run nvfp4 $NBT; done
for NBT in 2048 8192; do run auto $NBT; done

echo "===== NBT SWEEP (Δ = nvfp4 − bf16; bf16 from nbt 2048/8192) ====="
python3 - "$OUT" <<'PY'
import json,os,sys
o=sys.argv[1]
def nll(n):
    p=os.path.join(o,n+".json")
    return json.load(open(p))["score"]["mean_nll_nats"] if os.path.exists(p) else None
b2=nll("nbt2048_bf16"); b8=nll("nbt8192_bf16")
for nbt in [2048,4096,6144,8192]:
    q=nll(f"nbt{nbt}_nvfp4")
    b=b2 if nbt<=4096 else b8  # bf16 is ~flat in nbt; use nearest measured
    if q is None or b is None: print(f"nbt={nbt}: MISSING"); continue
    print(f"nbt={nbt:5d}: nvfp4={q:.4f}  bf16≈{b:.4f}  Δ={q-b:+.4f}")
PY
