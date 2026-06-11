#!/bin/bash
# Phase A: dispatcher fix validation (rt-base + rt5)
set -uo pipefail
source /mnt/b/workshop/wsl_sm120/env.sh
cd ~/hijinks

echo "=== A/rt-base: fa2-vo-split-d512-vo256, page-size 32 ==="
t0=$(date +%s)
python scripts/vllm_gemma4_mixed_kv_probes.py \
  --probe fa2-vo-split-d512-vo256 \
  --geometry 31b-serving \
  --skip-reference --plan-parity \
  --page-size 32 --batch-size 9 --qo-len 2 --kv-len 2 \
  --flashinfer-source-root ~/flashinfer \
  --output "$RESULTS/rt_base_fixed.json" 2>&1 | tail -30
echo "A/rt-base exit=$? wall=$(( $(date +%s) - t0 ))s"

echo "=== A/rt5: same, page-size 16 ==="
t0=$(date +%s)
python scripts/vllm_gemma4_mixed_kv_probes.py \
  --probe fa2-vo-split-d512-vo256 \
  --geometry 31b-serving \
  --skip-reference --plan-parity \
  --page-size 16 --batch-size 9 --qo-len 2 --kv-len 2 \
  --flashinfer-source-root ~/flashinfer \
  --output "$RESULTS/rt5_fixed.json" 2>&1 | tail -30
echo "A/rt5 exit=$? wall=$(( $(date +%s) - t0 ))s"
