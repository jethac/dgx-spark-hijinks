#!/usr/bin/env bash
# Narrow K refinement for the 26B-A4B full-NVFP4 early+mid sub-band calibration.
#
# This is a wrapper around run_26b_subband_calib_sweep.sh. The sub-band sweep
# bracketed the parity crossing at fixed v=0.08:
#   k=0.10 -> -0.117964257 nats/token vs vLLM bf16
#   k=0.11 -> +0.162258904 nats/token vs vLLM bf16
# so the interpolated target is near k=0.104.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/run_26b_subband_calib_sweep.sh}"
if [ ! -f "${BASE_SCRIPT}" ]; then
  BASE_SCRIPT="/root/run_26b_subband_calib_sweep.sh"
fi

export CANDIDATES="${CANDIDATES:-replay_k100_v08|0.07,0.05,0.07,0.05|0.100,0.080|0.100,0.080|0.100,0.080|0.100,0.080;k102_v08|0.07,0.05,0.07,0.05|0.102,0.080|0.102,0.080|0.102,0.080|0.102,0.080;k103_v08|0.07,0.05,0.07,0.05|0.103,0.080|0.103,0.080|0.103,0.080|0.103,0.080;k104_v08|0.07,0.05,0.07,0.05|0.104,0.080|0.104,0.080|0.104,0.080|0.104,0.080;k105_v08|0.07,0.05,0.07,0.05|0.105,0.080|0.105,0.080|0.105,0.080|0.105,0.080;k106_v08|0.07,0.05,0.07,0.05|0.106,0.080|0.106,0.080|0.106,0.080|0.106,0.080;k108_v08|0.07,0.05,0.07,0.05|0.108,0.080|0.108,0.080|0.108,0.080|0.108,0.080;replay_k110_v08|0.07,0.05,0.07,0.05|0.110,0.080|0.110,0.080|0.110,0.080|0.110,0.080}"
exec bash "${BASE_SCRIPT}"
