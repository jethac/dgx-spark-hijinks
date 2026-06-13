#!/usr/bin/env bash
export HF_TOKEN="$1"
export PYTHONPATH=/root/flashinfer
export PATH=/root/v/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root
python vllm_matched_kv_anchor.py --model google/gemma-4-12b-it --tokenizer google/gemma-4-12b-it \
  --corpus /root/wikitext_8k.txt --kv-cache-dtype auto --ctx 8185 --prefix-len 4096 \
  --max-model-len 8192 --max-num-batched-tokens 8192 --skip-warmup \
  --gpu-memory-utilization 0.5 \
  --output /root/pfx_out/probe.json --enforce-eager > /root/probe.log 2>&1
echo "EXIT=$?" >> /root/probe.log
