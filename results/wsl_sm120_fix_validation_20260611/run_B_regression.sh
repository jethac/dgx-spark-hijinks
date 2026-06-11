#!/bin/bash
# Phase B: regression slice (fix must not move working configs)
set -uo pipefail
source /mnt/b/workshop/wsl_sm120/env.sh
cd ~/hijinks

echo "=== B1: e4b bf16 vo-split (fa2-vo-split-d512-vo256, geometry e4b) ==="
t0=$(date +%s)
python scripts/vllm_gemma4_mixed_kv_probes.py \
  --probe fa2-vo-split-d512-vo256 \
  --geometry e4b \
  --flashinfer-source-root ~/flashinfer \
  --output "$RESULTS/e4b_bf16_vo256.json" 2>&1 | tail -20
echo "B1 exit=$? wall=$(( $(date +%s) - t0 ))s"

echo "=== B2: 31b bf16 vo-split (fa2-vo-split-d512-vo256, geometry 31b) ==="
t0=$(date +%s)
python scripts/vllm_gemma4_mixed_kv_probes.py \
  --probe fa2-vo-split-d512-vo256 \
  --geometry 31b \
  --flashinfer-source-root ~/flashinfer \
  --output "$RESULTS/31b_bf16_vo256.json" 2>&1 | tail -20
echo "B2 exit=$? wall=$(( $(date +%s) - t0 ))s"

echo "=== B3: nvfp4 A1-style probe ==="
t0=$(date +%s)
python scripts/flashinfer_nvfp4_kv_probe.py \
  --vo-split 2 --head-dim 512 --kv-container tuple --causal \
  --v-scale-layout linear --no-deswizzle-flag \
  --layouts NHD HND --cosine-threshold 0.9999 \
  --batch-size 4 --kv-len 96 --qo-len 16 --page-size 16 \
  --flashinfer-source-root ~/flashinfer \
  --output "$RESULTS/nvfp4_a1.json" 2>&1 | tail -40
echo "B3 exit=$? wall=$(( $(date +%s) - t0 ))s"

echo "=== B4: fp8 trait probe (fa2-vo-split-d512-vo256-fp8kv) - expect RED ==="
t0=$(date +%s)
python scripts/vllm_gemma4_mixed_kv_probes.py \
  --probe fa2-vo-split-d512-vo256-fp8kv \
  --geometry 31b-serving \
  --skip-reference --plan-parity \
  --page-size 32 --batch-size 9 --qo-len 2 --kv-len 2 \
  --flashinfer-source-root ~/flashinfer \
  --output "$RESULTS/fp8_trait_probe.json" 2>&1 | tail -25
echo "B4 exit=$? wall=$(( $(date +%s) - t0 ))s"
