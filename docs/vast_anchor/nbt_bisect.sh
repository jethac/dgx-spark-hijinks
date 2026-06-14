#!/usr/bin/env bash
export HF_TOKEN="$1" PATH=/root/v/bin:$PATH PYTHONPATH=/root/flashinfer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_FLASHINFER_MM_PREFIX=1 VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1
cd /root; mkdir -p pfx_out
[ -f wikitext_8k.txt ] || python corpus_fetch.py >/dev/null 2>&1
for NBT in 4096 4608 5120 5632 6144; do
  python vllm_matched_kv_anchor.py --model google/gemma-4-12b-it --tokenizer google/gemma-4-12b-it \
    --corpus wikitext_8k.txt --kv-cache-dtype nvfp4 --ctx 8185 --prefix-len 4096 --max-model-len 8192 \
    --max-num-batched-tokens "$NBT" --skip-warmup --gpu-memory-utilization 0.5 \
    --output pfx_out/bis_$NBT.json --enforce-eager > bis_$NBT.log 2>&1
  v=$(python3 -c "import json;print(round(json.load(open('pfx_out/bis_$NBT.json'))['score']['mean_nll_nats'],4))" 2>/dev/null || echo FAIL)
  echo "nbt=$NBT nvfp4_NLL=$v   (4096->+0.19 baseline 8.467, 8192->+0.42 baseline 8.7031)"
done
echo DONE_BISECT
