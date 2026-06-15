#!/usr/bin/env bash
# 26B-A4B MoE nvfp4 adjudication: HF eager bf16 TRUTH vs vLLM bf16 vs vLLM nvfp4 (determinism + scales).
# Tells us whether the bf16 baseline is wrong or the nvfp4 path is broken, and whether nvfp4 is deterministic.
set -u
HF_TOKEN="$1"
export HF_TOKEN PATH=/root/v/bin:$PATH PYTHONPATH=/root/flashinfer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_FLASHINFER_MM_PREFIX=1
export VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1
cd /root; mkdir -p dx26; M=google/gemma-4-26b-a4b-it
[ -f wikitext_8k.txt ] || python corpus_fetch.py >/dev/null 2>&1
echo "=== HF eager bf16 TRUTH (the ground truth) ==="
python hf_ref_ppl.py $M wikitext_8k.txt 8185 4096 2>dx26/hfref.err | tee dx26/hfref.out
echo "=== vLLM rows (chunked nbt=4096) ==="
anchor(){ env $4 python vllm_matched_kv_anchor.py --model $M --tokenizer $M --corpus wikitext_8k.txt \
  --kv-cache-dtype $2 --ctx 8185 --prefix-len 4096 --max-model-len 8192 --max-num-batched-tokens $3 \
  --skip-warmup --gpu-memory-utilization 0.82 --output dx26/$1.json --enforce-eager > dx26/$1.log 2>&1
  python3 -c "import json;print(round(json.load(open('dx26/$1.json'))['score']['mean_nll_nats'],4))" 2>/dev/null||echo FAIL; }
echo "bf16_chunk     $(anchor bf16 auto 4096 '')"
echo "nvfp4_0.07_a   $(anchor nv07a nvfp4 4096 'VLLM_FIX_K_SCALE=0.07 VLLM_FIX_V_SCALE=0.07')"
echo "nvfp4_0.07_b   $(anchor nv07b nvfp4 4096 'VLLM_FIX_K_SCALE=0.07 VLLM_FIX_V_SCALE=0.07')"
echo "nvfp4_0.05     $(anchor nv05 nvfp4 4096 'VLLM_FIX_K_SCALE=0.05 VLLM_FIX_V_SCALE=0.05')"
echo "nvfp4_0.1      $(anchor nv10 nvfp4 4096 'VLLM_FIX_K_SCALE=0.1 VLLM_FIX_V_SCALE=0.1')"
echo DONE_DX26
