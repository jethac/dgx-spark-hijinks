#!/usr/bin/env bash
# Live Vast launcher for the 26B-A4B top-logprob attribution packet.
#
# Reads HF_TOKEN from stdin without echo. This keeps secrets out of scripts,
# command lines, and artifacts while avoiding fragile SSH quoting.
set -euo pipefail

read -rs HF_TOKEN
export HF_TOKEN

source /root/v/bin/activate
export PYTHONPATH=/root/flashinfer
export MAX_JOBS="${MAX_JOBS:-4}"
export ROWS="${ROWS:-base_k100 e0_all l0 l1}"
export OUT="${OUT:-/root/dx26_toplogprob_attr_$(date -u +%Y%m%dT%H%M%SZ)}"

bash /root/run_26b_toplogprob_attribution.sh 2>&1 | tee "${OUT}.log"
