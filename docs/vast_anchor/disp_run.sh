#!/usr/bin/env bash
export HF_TOKEN="$1"
export PYTHONPATH=/root/flashinfer
export PATH=/root/v/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_FLASHINFER_MM_PREFIX=1 VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1
export FI_LOG_DISPATCH=1
cd /root
python instrument_dispatch.py
rm -rf ~/.cache/flashinfer 2>/dev/null
mkdir -p pfx_out
for NBT in 8192 4096; do
  echo "######## DISPATCH nbt=$NBT ########"
  python vllm_matched_kv_anchor.py --model google/gemma-4-12b-it --tokenizer google/gemma-4-12b-it \
    --corpus wikitext_8k.txt --kv-cache-dtype nvfp4 --ctx 8185 --prefix-len 4096 \
    --max-model-len 8192 --max-num-batched-tokens "$NBT" --skip-warmup \
    --gpu-memory-utilization 0.5 --output /root/pfx_out/disp_nbt${NBT}.json --enforce-eager \
    > /root/disp_nbt${NBT}.log 2>&1
  echo "rc=$? ; unique dispatch configs:"
  grep -oE '\[FI_DISPATCH\][^\n]*' /root/disp_nbt${NBT}.log | sort | uniq -c
done
echo DONE_DISPATCH
