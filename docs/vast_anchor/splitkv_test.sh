#!/usr/bin/env bash
export HF_TOKEN="$1"
export PATH=/root/v/bin:$PATH PYTHONPATH=/root/flashinfer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_FLASHINFER_MM_PREFIX=1 VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1
cd /root
FI=$(/root/v/bin/python -c "import vllm.v1.attention.backends.flashinfer as m; print(m.__file__)")
echo "backend=$FI"
# baseline single (no patch) already = 8.7031; now patch: force prefill_fixed_split_size=4096
cp "$FI" /root/flashinfer_backend.bak
sed -i 's/^            self.prefill_fixed_split_size = -1/            self.prefill_fixed_split_size = 4096/' "$FI"
echo "=== patched else-branch prefill_fixed_split_size ==="; grep -n "prefill_fixed_split_size = " "$FI" | head
mkdir -p pfx_out
echo "=== single nvfp4 WITH forced split (expect ~8.467 if split-merge is the fix) ==="
python vllm_matched_kv_anchor.py --model google/gemma-4-12b-it --tokenizer google/gemma-4-12b-it \
  --corpus wikitext_8k.txt --kv-cache-dtype nvfp4 --ctx 8185 --prefix-len 4096 \
  --max-model-len 8192 --max-num-batched-tokens 8192 --skip-warmup \
  --gpu-memory-utilization 0.5 --output pfx_out/splitkv_single.json --enforce-eager > splitkv.log 2>&1
echo "rc=$?"
python3 -c "import json;print('SPLITKV single NLL=', round(json.load(open('pfx_out/splitkv_single.json'))['score']['mean_nll_nats'],4))" 2>/dev/null || (echo NOJSON; grep -iE "error|assert" splitkv.log | tail -3)
# restore backend
cp /root/flashinfer_backend.bak "$FI"
echo DONE_SPLITKV
