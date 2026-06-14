#!/usr/bin/env bash
python3 -m pip install -q accelerate pyarrow 2>&1 | tail -1
cd /work
for S in fp8 fp32 manual; do
  echo "##### SCALE=$S #####"
  SCALE="$S" REFMID=google/gemma-4-12B-it REFCTX=8185 REFPSTART=4096 \
    python3 /disc/refsim_disc.py 2>&1 | tail -4
done
echo DONE_DISC
