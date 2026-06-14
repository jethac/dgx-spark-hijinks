#!/usr/bin/env bash
set -e
pip install -q accelerate pyarrow 2>&1 | tail -1
cd /work
for S in fp8 fp32 manual; do
  echo "##### SCALE=$S #####"
  SCALE="$S" REFMID=google/gemma-4-12B-it REFCTX=8185 REFPSTART=4096 \
    python3 docs/vast_anchor/refsim_disc.py 2>&1 | grep -E "selfcheck|SCALE=|DELTA|Error|Traceback|OutOfMemory" | tail -6
done
echo DONE_DISC
