#!/usr/bin/env bash
# Launch the long-context 26B-A4B mixed whole-layer KV ladder on a prepared Vast
# instance. Expects /root/.hf_token, /root/v, /root/flashinfer, and the harness
# files under /root/codex_harness.
set -euo pipefail

cp /root/codex_harness/corpus_fetch.py /root/
cp /root/codex_harness/vllm_toplogprob_attribution.py /root/

source /root/v/bin/activate
export HF_TOKEN="$(cat /root/.hf_token)"
export PYTHONPATH=/root/flashinfer

export ROWS="${ROWS:-bf16 base_k100 bf16_0_4 fp8_0_4 fp8_0_7 fp8_all_sliding fp8_all_global}"
export CTX="${CTX:-8185}"
export PREFIX="${PREFIX:-4096}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-20}"
export KEEP_TOPK="${KEEP_TOPK:-20}"
export POSITION_STRIDE="${POSITION_STRIDE:-128}"
export DENSE_PREFIX_POSITIONS="${DENSE_PREFIX_POSITIONS:-256}"
export ROW_TIMEOUT="${ROW_TIMEOUT:-7200}"
export GPU_UTIL="${GPU_UTIL:-0.82}"
export MAX_JOBS="${MAX_JOBS:-4}"
export OUT="${OUT:-/root/dx26_mixed_paddingfix_full_$(date -u +%Y%m%dT%H%M%SZ)}"

nohup bash /root/codex_harness/run_26b_mixed_layer_ladder.sh \
  > /root/dx26_mixed_paddingfix_full.log 2>&1 &
echo "$!" > /root/dx26_mixed_paddingfix_full.pid
printf 'PID=%s\nOUT=%s\n' "$(cat /root/dx26_mixed_paddingfix_full.pid)" "${OUT}"
