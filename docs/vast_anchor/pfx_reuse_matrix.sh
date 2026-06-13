#!/usr/bin/env bash
# +0.40 long-ctx root-cause: is it the prefix-reuse / partial-state-merge path, and
# does it reproduce on vLLM (general bug) or is vLLM clean (SGLang-radix-specific)?
#
# Scores the IDENTICAL suffix token set (positions prefix_len+1 .. ctx-1) under three
# attention paths, for bf16 and nvfp4 KV each:
#   reuse   : warm 4096 prefix -> score 8185   (cross-request radix / partial-state-merge)
#   chunked : no warm, single 8185 fwd, max_nbt=4096 (in-pass chunked-prefill merge)
#   single  : no warm, single 8185 fwd, max_nbt=8192 (one chunk, NO merge -> clean baseline)
#
# Decision: compute Δ(nvfp4 - bf16) per path.
#   single Δ ~ 0           -> format/long-ctx is fine (expected)
#   reuse/chunked Δ ~ +0.4 -> the merge path is the bug, AND it reproduces on vLLM = GENERAL
#   reuse/chunked Δ ~ 0    -> vLLM clean -> the bug is SGLang-radix-specific
set -uo pipefail
cd "$(dirname "$0")"
MODEL="${ANCHOR_MODEL:-google/gemma-4-12b-it}"
CORPUS="${CORPUS:-/root/wikitext_8k.txt}"
CTX="${CTX:-8185}"; PREFIX="${PREFIX:-4096}"
OUT=/root/pfx_out; mkdir -p "$OUT"
export VLLM_FLASHINFER_MM_PREFIX=1
# Gemma-4 12B has 512-wide global heads; FlashInfer rejects head_size>256 on CC 12.x
# unless the two-pass VO split is enabled. Without this, full-nvfp4 falls back to
# Triton, which does not support the nvfp4 KV dtype (ValueError at attn init).
export VLLM_NVFP4_KV_VOSPLIT=1
# The VO split slices V along the head dim, which a swizzled V scale-factor layout
# cannot support; linear V-SF is required for the two-pass split.
export VLLM_NVFP4_KV_LINEAR_V_SF=1
# ARMS: space-separated subset to run (default all 6). bf16 JSONs are knob-independent.
ARMS="${ARMS:-bf16_single nvfp4_single bf16_chunked nvfp4_chunked bf16_reuse nvfp4_reuse}"
want () { case " $ARMS " in *" $1 "*) return 0;; *) return 1;; esac; }
# Full-vocab prompt_logprobs over 8185 positions spikes ~8 GiB (8185*256000*4B) in
# log_softmax; leave headroom outside the KV-cache pool, and de-fragment.
GMU="${GMU:-0.5}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run () { # name kv max_nbt skipflag
  local name="$1" kv="$2" nbt="$3" skip="$4"
  want "$name" || { echo "=== SKIP $name (not in ARMS) ==="; return 0; }
  echo "=== ARM $name (kv=$kv nbt=$nbt skip=$skip gmu=$GMU vosplit=1) ==="
  python vllm_matched_kv_anchor.py --model "$MODEL" --tokenizer "$MODEL" \
    --corpus "$CORPUS" --kv-cache-dtype "$kv" --ctx "$CTX" --prefix-len "$PREFIX" \
    --max-model-len 8192 --max-num-batched-tokens "$nbt" $skip \
    --gpu-memory-utilization "$GMU" \
    --output "$OUT/${name}.json" --enforce-eager 2>&1 | tail -3
}

run bf16_single  auto   8192 --skip-warmup
run nvfp4_single nvfp4  8192 --skip-warmup
run bf16_chunked auto   4096 --skip-warmup
run nvfp4_chunked nvfp4 4096 --skip-warmup
run bf16_reuse   auto   4096 ""
run nvfp4_reuse  nvfp4  4096 ""

echo "===== DELTA SUMMARY ====="
python - "$OUT" <<'PY'
import json,sys,os
o=sys.argv[1]
def nll(n):
    p=os.path.join(o,n+".json")
    if not os.path.exists(p): return None
    return json.load(open(p))["score"]["mean_nll_nats"]
for path in ["single","chunked","reuse"]:
    b=nll("bf16_"+path); q=nll("nvfp4_"+path)
    if b is None or q is None: print(f"{path:8s}: MISSING"); continue
    print(f"{path:8s}: bf16={b:.4f}  nvfp4={q:.4f}  Δ={q-b:+.4f}")
PY
