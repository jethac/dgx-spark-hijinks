#!/usr/bin/env bash
export HF_TOKEN="$1" PATH=/root/v/bin:$PATH PYTHONPATH=/root/flashinfer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_FLASHINFER_MM_PREFIX=1 VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1
export VLLM_BATCH_INVARIANT=1
cd /root; mkdir -p pfx_out
python vllm_matched_kv_anchor.py --model google/gemma-4-12b-it --tokenizer google/gemma-4-12b-it \
  --corpus wikitext_8k.txt --kv-cache-dtype nvfp4 --ctx 8185 --prefix-len 4096 --max-model-len 8192 \
  --max-num-batched-tokens 8192 --skip-warmup --gpu-memory-utilization 0.5 \
  --output pfx_out/bi_single.json --enforce-eager > bi.log 2>&1
echo "rc=$?"
python3 -c "import json;print('BI single NLL=',round(json.load(open('pfx_out/bi_single.json'))['score']['mean_nll_nats'],4))" 2>/dev/null || (echo NOJSON; grep -iE "error|assert|invariant|not support" bi.log | tail -4)
echo DONE_BI
