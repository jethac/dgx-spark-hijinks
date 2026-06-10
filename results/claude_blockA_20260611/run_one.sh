#!/bin/bash
# Block A runner: run_one.sh <name> <debug_once:0|1> <probe args...>
NAME="$1"; DBG="$2"; shift 2
R=/home/jethac/spark_tmp/claude_blockA_results
ENVDBG=()
[ "$DBG" = "1" ] && ENVDBG=(-e FLASHINFER_PREFILL_DEBUG_ONCE=1)
START=$(date +%s)
docker run --rm --gpus all --memory 100g --memory-swap 100g \
  --name "$NAME" \
  -v /home/jethac/spark_tmp/claude_blockA_results:/work \
  -v /home/jethac/spark_tmp/flashinfer-fa2-d512:/fisrc:ro \
  -e PYTHONPATH=/fisrc:/work \
  "${ENVDBG[@]}" \
  jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv \
  python3 /work/flashinfer_nvfp4_kv_probe.py "$@"
RC=$?
END=$(date +%s)
echo "RUN=$NAME EXIT=$RC WALL_SECONDS=$((END-START))" >> $R/blockA_status.txt
exit $RC
