#!/usr/bin/env bash
set -uo pipefail
R=/home/jethac/spark_tmp/claude_r8_gates_results/part1
IMAGE=jethac-vllm-aeon-gemma4:e08a6f3ae-rebuiltc-fb7d62ea-sm121a-r8
T0=$(date +%s)
echo "PART1_START $(date -Is)" >> $R/part1_status.txt

# Container A: linear runs (env VLLM_NVFP4_KV_LINEAR_V_SF=1), shares in-container JIT cache
tA=$(date +%s)
docker run --rm --name claude_r8_p1_linear --gpus all --ipc=host \
  --memory 100g --memory-swap 100g \
  -e VLLM_NVFP4_KV_LINEAR_V_SF=1 \
  -w /work -v $R:/work \
  "$IMAGE" bash -lc "
    python3 /work/scripts/nvfp4_writer_roundtrip_probe.py --calibrate --v-scale-layout linear --output /work/results/run1_calibration_linear.json > /work/results/run1_calibration_linear.log 2>&1; echo RUN1_RC=\$?;
    python3 /work/scripts/nvfp4_writer_roundtrip_probe.py --v-scale-layout linear --output /work/results/run3_head256_nowindow_linear.json > /work/results/run3_head256_nowindow_linear.log 2>&1; echo RUN3_RC=\$?;
    python3 /work/scripts/nvfp4_writer_roundtrip_probe.py --v-scale-layout linear --window-left 1023 --output /work/results/run4_head256_window1023_linear.json > /work/results/run4_head256_window1023_linear.log 2>&1; echo RUN4_RC=\$?;
  " >> $R/part1_status.txt 2>&1
echo "CONTAINER_A_RC=$? WALL=$(( $(date +%s) - tA ))" >> $R/part1_status.txt

# Container B: swizzled control (env unset)
tB=$(date +%s)
docker run --rm --name claude_r8_p1_swizzled --gpus all --ipc=host \
  --memory 100g --memory-swap 100g \
  -w /work -v $R:/work \
  "$IMAGE" bash -lc "
    python3 /work/scripts/nvfp4_writer_roundtrip_probe.py --calibrate --v-scale-layout swizzled --output /work/results/run2_calibration_swizzled.json > /work/results/run2_calibration_swizzled.log 2>&1; echo RUN2_RC=\$?;
  " >> $R/part1_status.txt 2>&1
echo "CONTAINER_B_RC=$? WALL=$(( $(date +%s) - tB ))" >> $R/part1_status.txt
echo "PART1_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> $R/part1_status.txt
