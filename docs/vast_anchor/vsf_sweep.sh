#!/usr/bin/env bash
# Localization first-cut for the 26B-A4B nvfp4 kernel bug: does the V-scale-factor LAYOUT
# (linear vs swizzled) change the 26B break? Scale held fixed (FIXSCALE k=v=0.1) so only the
# layout varies. Truth (HF eager) = 7.9923; known broken baseline ~7.31. VOSPLIT must stay 1
# (512 global heads need the VO split). If LINEAR_V_SF=0 lands near truth -> V-SF layout is the
# bug; if it stays ~broken -> V-SF ruled out, it's MoE-specific deeper in the write/read path.
set -u
HF_TOKEN="$1"
export HF_TOKEN PATH=/root/v/bin:$PATH PYTHONPATH=/root/flashinfer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_FLASHINFER_MM_PREFIX=1
cd /root; mkdir -p vsf; M=google/gemma-4-26b-a4b-it
[ -f wikitext_8k.txt ] || python corpus_fetch.py >/dev/null 2>&1
anchor(){ env $3 python vllm_matched_kv_anchor.py --model $M --tokenizer $M --corpus wikitext_8k.txt \
  --kv-cache-dtype $2 --ctx 8185 --prefix-len 4096 --max-model-len 8192 --max-num-batched-tokens 4096 \
  --skip-warmup --gpu-memory-utilization 0.82 --output vsf/$1.json --enforce-eager > vsf/$1.log 2>&1
  python3 -c "import json;print(round(json.load(open('vsf/$1.json'))['score']['mean_nll_nats'],4))" 2>/dev/null||echo FAIL; }
echo "=== 26B-A4B V-SF layout localization (truth=7.99, broken~7.31) ==="
echo "bf16 ref:                       $(anchor bf16 auto '')"
echo "nvfp4 VOSPLIT=1 LINEAR_V_SF=1:  $(anchor vo1lin1 nvfp4 'VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1 VLLM_FIX_K_SCALE=0.1 VLLM_FIX_V_SCALE=0.1')"
echo "nvfp4 VOSPLIT=1 LINEAR_V_SF=0:  $(anchor vo1lin0 nvfp4 'VLLM_NVFP4_KV_VOSPLIT=1 VLLM_FIX_K_SCALE=0.1 VLLM_FIX_V_SCALE=0.1')"
echo DONE_VSF
