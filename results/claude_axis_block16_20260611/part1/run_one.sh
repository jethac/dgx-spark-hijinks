#!/bin/bash
# axis runner: run_one.sh <name> <python args...>
NAME="$1"; shift
R=/home/jethac/spark_tmp/claude_axis_results
START=$(date +%s)
docker run --rm --gpus all --memory 100g --memory-swap 100g \
  --name "$NAME" \
  -v /home/jethac/spark_tmp/claude_axis_results:/work \
  -v /home/jethac/spark_tmp/flashinfer-fa2-d512:/fisrc:ro \
  -e PYTHONPATH=/fisrc:/work \
  jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv \
  python3 "$@"
RC=$?
END=$(date +%s)
echo "RUN=$NAME EXIT=$RC WALL_SECONDS=$((END-START))" >> $R/axis_status.txt
exit $RC
