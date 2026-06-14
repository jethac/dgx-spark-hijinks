#!/usr/bin/env bash
export HF_TOKEN="$1" PATH=/root/v/bin:/usr/local/cuda-13.0/bin:$PATH PYTHONPATH=/root/flashinfer-refork
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_FLASHINFER_MM_PREFIX=1 VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1
cd /root; mkdir -p pfx_out
for NBT in 8192 4096; do
  python vllm_matched_kv_anchor.py --model google/gemma-4-12b-it --tokenizer google/gemma-4-12b-it \
    --corpus wikitext_8k.txt --kv-cache-dtype nvfp4 --ctx 8185 --prefix-len 4096 --max-model-len 8192 \
    --max-num-batched-tokens "$NBT" --skip-warmup --gpu-memory-utilization 0.75 \
    --output pfx_out/rf2_$NBT.json --enforce-eager > rf2_$NBT.log 2>&1
  v=$(python3 -c "import json;print(round(json.load(open('pfx_out/rf2_$NBT.json'))['score']['mean_nll_nats'],4))" 2>/dev/null || echo FAIL)
  echo "REFORK2 nbt=$NBT NLL=$v"
  [ "$v" = "FAIL" ] && grep -iE "error|Invalid|memory|assert|no kernel" rf2_$NBT.log | tail -3
done
echo "DONE_RF2 (old-fork baselines: single 8.7031, chunked 8.467)"
