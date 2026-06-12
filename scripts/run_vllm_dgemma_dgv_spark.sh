#!/usr/bin/env bash
# DG-V: vLLM DiffusionGemma 26B-A4B NVFP4-KV on Spark (sm_121 / GB10).
# STAGED 2026-06-12 (Claude / vLLM lane) -- runs once the e2-dgv sm121a-arm64 CI
# wheel lands. Spark = serve/eval ONLY (no build); wheel comes from CI per Jetha.
#
# Parity target = SGLang DG-R5/R6 (Codex):
#   coherence: Tokyo / 2+2 / DGX Spark prompts PASS
#   full-NVFP4 proof: mixed_kv=False, FP4 K+V pools (kv_data_type=uint8)
#   VO-split proof: global layers head_dim=512 -> head_dim_vo=256
#   capacity: ~3.56x KV token budget vs bf16/auto (SGLang: 3.5654x full / 3.5625x SWA)
#   DG-R6: perf pair (nvfp4 vs bf16 throughput/latency at matched batch)
set -euo pipefail

# ---- wheel + env (fill WHEEL_TAG when the e2-dgv arm64 build publishes) -------
WHEEL_TAG=${WHEEL_TAG:-sm121a-arm64-wheels-PENDING}     # e.g. sm121a-arm64-wheels-<e2dgv-sha>-dgv
VLLM_REPO=${VLLM_REPO:-jethac/vllm}
MODEL=${MODEL:-google/diffusiongemma-26B-A4B-it}
SERVED=${SERVED:-diffusiongemma-26b-a4b}
# Base aarch64 image with torch 2.12 cu130 deps (serve in a container, no build):
BASE_IMAGE=${BASE_IMAGE:-vllm/vllm-openai:gemma-aarch64-cu130}
FLASHINFER_SRC=${FLASHINFER_SRC:-$HOME/flashinfer}      # spark/hijinks-022-fa2-d512 JIT tree
RUN_ROOT=${RUN_ROOT:-$HOME/spark_tmp/vllm_dgemma_dgv_$(date +%Y%m%dT%H%MJST)}
PORT=${PORT:-8000}

# ---- DG-V serve knobs (mirror SGLang DG-R5 + the config.py allowance I wired) -
# VO-split knobs gate the DiffusionGemma FLASHINFER allowance (config.py) AND the
# D=512 global-layer two-pass in the unified FIPrefillGroup path.
export VLLM_FLASHINFER_VOSPLIT=1
export VLLM_NVFP4_KV_VOSPLIT=1
ATTN_BACKEND=FLASHINFER
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp4_e2m1}   # CONFIRM exact vLLM flag on wheel-land (nvfp4 KV)
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}              # recipe constraint (block-diffusion)
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}           # recipe
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}         # DG-V5 smoke len (256k is the headline; smoke smaller)

mkdir -p "$RUN_ROOT"
echo "[DG-V] run root: $RUN_ROOT  wheel: $WHEEL_TAG  model: $MODEL"

# ---- marker (claim Spark per bus contract; release on EXIT) -------------------
MARKER=$HOME/spark_tmp/MARKER_claude_dgv
echo "claude dgv $(date -u +%FT%TZ)" > "$MARKER"
trap 'rm -f "$MARKER"; docker rm -f dgv_nvfp4 dgv_bf16 >/dev/null 2>&1 || true' EXIT

COHERENCE_PROMPTS=("The capital of Japan is" "2 + 2 =" "The NVIDIA DGX Spark is")

serve_and_eval() {  # $1=tag(nvfp4|bf16) $2=kv_dtype_args
  local tag=$1 kvargs=$2 name=dgv_${tag} log="$RUN_ROOT/${tag}_serve.log"
  echo "[DG-V] launching $tag ..."
  # NOTE on wheel-land: docker run $BASE_IMAGE, pip install the $WHEEL_TAG asset,
  # mount $FLASHINFER_SRC, then `vllm serve $MODEL --attention-backend $ATTN_BACKEND
  #   $kvargs --max-num-seqs $MAX_NUM_SEQS --gpu-memory-utilization $GPU_MEM_UTIL
  #   --max-model-len $MAX_MODEL_LEN --served-model-name $SERVED --port $PORT`
  # (kept as a NOTE not a live cmd until the wheel exists + the kv-dtype flag is
  #  confirmed, so this script never half-runs a wrong config.)
  echo "TODO(wheel): launch $name with $kvargs; capture proofs to $log" | tee -a "$log"
  # Proof captures to grep from $log once live:
  #   full-NVFP4:  'mixed_kv' false + FP4 K/V pool alloc + kv_data_type=uint8
  #   VO-split:    global layer head_dim=512 head_dim_vo=256 (FIPrefillGroup VO path)
  #   capacity:    'GPU KV cache size' / num KV tokens -> ratio vs bf16 run
  #   coherence:   POST /v1/completions for each COHERENCE_PROMPTS -> coherent
}

# DG-V5: full-NVFP4 K+V (the headline) + bf16 denominator for the 3.56x ratio
serve_and_eval nvfp4 "--kv-cache-dtype ${KV_CACHE_DTYPE}"
serve_and_eval bf16  "--kv-cache-dtype auto"

# DG-V6: perf pair (nvfp4 vs bf16 throughput/latency at matched batch) -- add the
# vllm bench / a fixed request set once DG-V5 is green.

cat > "$RUN_ROOT/summary.md" <<EOF
# DG-V (vLLM DiffusionGemma 26B-A4B NVFP4-KV on Spark sm_121)
STAGED run skeleton. Fill on wheel-land. Parity target: SGLang DG-R5/R6.
- wheel: $WHEEL_TAG ; model: $MODEL ; backend: $ATTN_BACKEND ; VO-split knobs ON
- DG-V5 green bar: coherent + full-NVFP4 proof + VO-split proof + ~3.5x capacity + double-run bitwise
- DG-V6: perf pair
EOF
echo "[DG-V] staged summary at $RUN_ROOT/summary.md"
