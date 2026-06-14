#!/usr/bin/env bash
# Diff the FlashInfer prefill-wrapper config between single-prefill and chunked-prefill for
# nvfp4, to localize the +0.42-vs-+0.19 gap WITHOUT a kernel rebuild. If the two use different
# wrapper configs (causal flag, NUM_MMA, #prefill calls, cascade), that's the divergence; if
# identical, the gap is pure accumulation-length and we go to the rebuild-bisect.
export HF_TOKEN="$1"
export PYTHONPATH=/root/flashinfer
export PATH=/root/v/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_FLASHINFER_MM_PREFIX=1 VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1
export VLLM_SPARK_KV_TRACE=1 VLLM_SPARK_KV_TRACE_LAYERS=0 VLLM_SPARK_KV_TRACE_LIMIT=2
cd /root
for NBT in 8192 4096; do
  echo "######## TRACE nbt=$NBT ########"
  python vllm_matched_kv_anchor.py --model google/gemma-4-12b-it --tokenizer google/gemma-4-12b-it \
    --corpus wikitext_8k.txt --kv-cache-dtype nvfp4 --ctx 8185 --prefix-len 4096 \
    --max-model-len 8192 --max-num-batched-tokens "$NBT" --skip-warmup \
    --gpu-memory-utilization 0.5 --output /root/pfx_out/trace_nbt${NBT}.json --enforce-eager \
    > /root/trace_nbt${NBT}.log 2>&1
  echo "rc=$?"
done
echo DONE_TRACE
