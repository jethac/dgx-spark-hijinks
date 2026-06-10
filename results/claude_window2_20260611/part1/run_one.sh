#!/bin/bash
NAME="$1"; shift
R=/home/jethac/spark_tmp/claude_window2_part1
START=$(date +%s)
docker run --rm --gpus all --memory 100g --memory-swap 100g \
  --name "$NAME" \
  -v /home/jethac/spark_tmp/claude_window2_part1:/work \
  -v /home/jethac/spark_tmp/flashinfer-fa2-d512:/fisrc:ro \
  -e PYTHONPATH=/fisrc:/work \
  jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv \
  python3 "$@" > "$R/${NAME}.stdout.log" 2> "$R/${NAME}.stderr.log"
RC=$?
END=$(date +%s)
echo "RUN=$NAME EXIT=$RC WALL_SECONDS=$((END-START))" >> $R/part1_status.txt
exit $RC
