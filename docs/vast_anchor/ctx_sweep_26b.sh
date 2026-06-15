#!/usr/bin/env bash
# 26B-A4B nvfp4 read-math-vs-feed discriminator: does the break appear WITHIN one sliding window
# (short ctx, no SWA eviction, simple cache view) or only when context CROSSES the 1024 window
# (paged + mixed sliding/full cache-view feed)? FIXSCALE k=v=0.1 held; bf16 = per-ctx truth.
# short clean + long broken => paged/SWA feed bug (Codex suspect #2, echoes Gemma-3-1B). broken
# everywhere => the per-read math. 26B sliding_window=1024.
set -u
HF_TOKEN="$1"
export HF_TOKEN PATH=/root/v/bin:$PATH PYTHONPATH=/root/flashinfer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_FLASHINFER_MM_PREFIX=1
cd /root; mkdir -p ctxs; M=google/gemma-4-26b-a4b-it
[ -f wikitext_8k.txt ] || python corpus_fetch.py >/dev/null 2>&1
anchor(){ env $5 python vllm_matched_kv_anchor.py --model $M --tokenizer $M --corpus wikitext_8k.txt \
  --kv-cache-dtype $2 --ctx $3 --prefix-len $4 --max-model-len 8192 --max-num-batched-tokens 4096 \
  --skip-warmup --gpu-memory-utilization 0.82 --output ctxs/$1.json --enforce-eager > ctxs/$1.log 2>&1
  python3 -c "import json;print(round(json.load(open('ctxs/$1.json'))['score']['mean_nll_nats'],4))" 2>/dev/null||echo FAIL; }
NV="VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1 VLLM_FIX_K_SCALE=0.1 VLLM_FIX_V_SCALE=0.1"
echo "=== 26B-A4B ctx sweep (SWA window = 1024) ==="
for C in 512 2048 8185; do
  P=$((C/2))
  bf=$(anchor bf16_$C auto $C $P "")
  nv=$(anchor nv_$C nvfp4 $C $P "$NV")
  d=$(python3 -c "print(round($nv-$bf,4))" 2>/dev/null || echo "?")
  echo "ctx=$C (prefix=$P, $([ $C -lt 1024 ] && echo WITHIN-window || echo CROSSES-window)): bf16=$bf nvfp4=$nv delta=$d"
done
echo DONE_CTXSWEEP
