#!/bin/bash
# Common environment for validation runs. Source me.
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$HOME/flashinfer"
export MAX_JOBS=8
source ~/sm120env/bin/activate
RESULTS=/mnt/b/workshop/wsl_sm120/results
mkdir -p "$RESULTS"
