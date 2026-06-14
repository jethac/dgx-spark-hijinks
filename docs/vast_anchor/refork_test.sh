#!/usr/bin/env bash
export HF_TOKEN="$1" PATH=/root/v/bin:/usr/local/cuda-13.0/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_FLASHINFER_MM_PREFIX=1 VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1
cd /root
echo "=== clone re-fork ==="
rm -rf flashinfer-refork
git clone -q https://github.com/jethac/flashinfer flashinfer-refork 2>&1 | tail -1
cd flashinfer-refork
git fetch -q origin spark/hijinks-022-refork 2>&1 | tail -1
git checkout -q spark/hijinks-022-refork
echo "refork HEAD: $(git rev-parse --short HEAD)"
git submodule update --init --recursive --depth 1 >/dev/null 2>&1
mkdir -p flashinfer/data
ln -sfn ../../csrc flashinfer/data/csrc; ln -sfn ../../include flashinfer/data/include
ln -sfn ../../3rdparty/cutlass flashinfer/data/cutlass; ln -sfn ../../3rdparty/cccl flashinfer/data/cccl; ln -sfn ../../3rdparty/spdlog flashinfer/data/spdlog
cd /root
export PYTHONPATH=/root/flashinfer-refork
echo "=== import test ==="
python -c "import flashinfer; print('REFORK_IMPORT_OK', flashinfer.__file__)" 2>&1 | tail -2
echo "=== run 12B nvfp4 single (nbt8192) + chunked (nbt4096) on re-fork (first build JIT-compiles) ==="
mkdir -p pfx_out
for NBT in 8192 4096; do
  python vllm_matched_kv_anchor.py --model google/gemma-4-12b-it --tokenizer google/gemma-4-12b-it \
    --corpus wikitext_8k.txt --kv-cache-dtype nvfp4 --ctx 8185 --prefix-len 4096 --max-model-len 8192 \
    --max-num-batched-tokens "$NBT" --skip-warmup --gpu-memory-utilization 0.5 \
    --output pfx_out/refork_$NBT.json --enforce-eager > refork_$NBT.log 2>&1
  v=$(python3 -c "import json;print(round(json.load(open('pfx_out/refork_$NBT.json'))['score']['mean_nll_nats'],4))" 2>/dev/null || echo "FAIL")
  echo "REFORK nbt=$NBT nvfp4_NLL=$v"
  [ "$v" = "FAIL" ] && grep -iE "error|assert|Invalid|no kernel|compile" refork_$NBT.log | tail -4
done
echo "DONE_REFORK (baselines on old fork: single 8.7031, chunked 8.467)"
