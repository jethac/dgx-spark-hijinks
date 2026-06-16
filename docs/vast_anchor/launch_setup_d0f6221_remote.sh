#!/usr/bin/env bash
# Start the E3 wheel/FlashInfer setup for the d0f6221 mixed-page padding
# validation runs. This script contains no credentials.
set -euo pipefail

chmod +x /root/codex_harness/*.sh
export VLLM_WHEEL_URL="https://github.com/jethac/vllm/releases/download/sm120a-wheels-4fcbf4c48/vllm-0.1.dev1%2Bg4fcbf4c48.sm120a-cp312-cp312-linux_x86_64.whl"
export FLASHINFER_REF="1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a"

setsid bash /root/codex_harness/e3_setup_wget.sh \
  > /root/setup_d0f6221.log 2>&1 < /dev/null &
echo "$!" > /root/setup_d0f6221.pid
printf 'PID=%s\n' "$(cat /root/setup_d0f6221.pid)"
