#!/usr/bin/env bash
# Live Vast launcher for the 26B-A4B active-KV mix attribution packet.
set -euo pipefail

read -rs HF_TOKEN
export HF_TOKEN

source /root/v/bin/activate
export PYTHONPATH=/root/flashinfer
export MAX_JOBS="${MAX_JOBS:-4}"
export OUT="${OUT:-/root/dx26_active_kv_mix_$(date -u +%Y%m%dT%H%M%SZ)}"
export DETACH="${DETACH:-1}"

if [ "${DETACH}" = "1" ]; then
  mkdir -p "$(dirname "${OUT}")"
  setsid bash -c '
    set -euo pipefail
    source /root/v/bin/activate
    export PYTHONPATH=/root/flashinfer
    bash /root/run_26b_active_kv_mix_probe.sh
  ' >"${OUT}.log" 2>&1 < /dev/null &
  pid=$!
  echo "${pid}" >"${OUT}.pid"
  echo "LAUNCHED pid=${pid} out=${OUT} log=${OUT}.log"
  disown "${pid}" 2>/dev/null || true
  exit 0
fi

bash /root/run_26b_active_kv_mix_probe.sh 2>&1 | tee "${OUT}.log"
