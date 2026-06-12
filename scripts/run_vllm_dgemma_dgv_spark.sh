#!/usr/bin/env bash
# DG-V5/V6: vLLM DiffusionGemma 26B-A4B NVFP4-KV on Spark (sm_121 / GB10), via r11.
# r11 = proven r10 stack + the e2-dgv vLLM wheel (22.04/glibc-2.35/torch-2.11, no
# retrofit). Run ON SPARK. Parity target = SGLang DG-R5/R6.
#   coherence: Tokyo / 2+2 / DGX Spark PASS
#   full-NVFP4: kv uint8, mixed_kv False, FP4 K+V pools
#   VO-split: global layers head_dim=512 -> head_dim_vo=256
#   capacity: ~3.56x KV token budget vs bf16/auto
set -euo pipefail

WHEEL_SHA=${WHEEL_SHA:-98cd3e59f}                      # e2-dgv head; wheel release sha
WHEEL_TAG=${WHEEL_TAG:-sm121a-arm64-wheels-${WHEEL_SHA}}
R10_IMAGE=${R10_IMAGE:-jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r10}
R11_IMAGE=${R11_IMAGE:-jethac-vllm-aeon-gemma4:e2-dgv-${WHEEL_SHA}-sm121a-r11}
MODEL=${MODEL:-google/diffusiongemma-26B-A4B-it}
SERVED=${SERVED:-diffusiongemma-26b-a4b}
DOCKERFILE_R11=${DOCKERFILE_R11:-$HOME/vllm/docker/Dockerfile.r11}   # from the e2-dgv checkout
RUN_ROOT=${RUN_ROOT:-$HOME/spark_tmp/vllm_dgemma_dgv_$(date +%Y%m%dT%H%MJST)}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}        # recipe (block-diffusion)
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}     # recipe
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}   # DG-V5 smoke len
PORT=${PORT:-8000}
URL="http://127.0.0.1:${PORT}"
mkdir -p "$RUN_ROOT"

MARKER=$HOME/spark_tmp/MARKER_claude_dgv
echo "claude dgv $(date -u +%FT%TZ)" > "$MARKER"
trap 'rm -f "$MARKER"; docker rm -f dgv_nvfp4 dgv_bf16 >/dev/null 2>&1 || true' EXIT

# ---- 1) bake r11 (download the wheel, layer it onto r10; no compile) ----------
if ! docker image inspect "$R11_IMAGE" >/dev/null 2>&1; then
  CTX=$(mktemp -d)
  ASSET="${RUN_ROOT}/$(basename "$(curl -fsSL "https://api.github.com/repos/jethac/vllm/releases/tags/${WHEEL_TAG}" | grep -oE 'https://[^"]*\.whl' | head -1)")"
  curl -fsSL -o "$ASSET" "$(curl -fsSL "https://api.github.com/repos/jethac/vllm/releases/tags/${WHEEL_TAG}" | grep -oE 'https://[^"]*\.whl' | head -1)"
  cp "$ASSET" "$CTX/$(basename "$ASSET")"
  cp "$DOCKERFILE_R11" "$CTX/Dockerfile"
  docker build "$CTX" \
    --build-arg BASE_IMAGE="$R10_IMAGE" \
    --build-arg VLLM_WHEEL="$(basename "$ASSET")" \
    -t "$R11_IMAGE"
  rm -rf "$CTX"
fi
echo "[DG-V] r11 image ready: $R11_IMAGE"

# ---- 2) serve + eval one row -------------------------------------------------
serve_row() {  # $1=tag(nvfp4|bf16)  $2=kv-cache-dtype
  local tag=$1 kv=$2 name=dgv_${tag} log="$RUN_ROOT/${tag}_server.log"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" --gpus all --net host --ipc host \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$R11_IMAGE" bash -lc "
      vllm serve '$MODEL' --served-model-name '$SERVED' --host 127.0.0.1 --port $PORT \
        --attention-backend FLASHINFER --kv-cache-dtype $kv \
        --max-num-seqs $MAX_NUM_SEQS --gpu-memory-utilization $GPU_MEM_UTIL \
        --max-model-len $MAX_MODEL_LEN" >/dev/null
  # wait ready (timeout = wedge detector)
  local ready=0
  for i in $(seq 1 600); do
    docker logs "$name" > "$log" 2>&1 || true
    curl -fsS "$URL/v1/models" >/dev/null 2>&1 && { ready=1; echo "[$tag] READY ${i}s"; break; }
    docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -q true || { echo "[$tag] CONTAINER DIED ${i}s"; break; }
    sleep 1
  done
  docker logs "$name" > "$log" 2>&1 || true
  [ "$ready" = 1 ] || { echo "[$tag] NOT READY -> tail:"; tail -40 "$log"; return 3; }
  # coherence
  for p in "The capital of Japan is" "2 + 2 =" "The NVIDIA DGX Spark is"; do
    curl -fsS "$URL/v1/completions" -H 'content-type: application/json' \
      -d "{\"model\":\"$SERVED\",\"prompt\":\"$p\",\"max_tokens\":24,\"temperature\":0}" \
      | tee -a "$RUN_ROOT/${tag}_coherence.json"; echo >> "$RUN_ROOT/${tag}_coherence.json"
  done
  # proofs from the server log
  grep -iE "mixed_kv|nvfp4|fp4|head_dim_vo|head_dim=512|vosplit|GPU KV cache|num.*kv.*token|available_kv" "$log" \
    | tee "$RUN_ROOT/${tag}_proofs.txt" | head -40 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
}

echo "=== DG-V5: full-NVFP4 K+V ==="; serve_row nvfp4 nvfp4
echo "=== DG-V5: bf16/auto denominator ==="; serve_row bf16 auto

# ---- 3) summary skeleton (fill the capacity ratio from the two proof files) ---
cat > "$RUN_ROOT/summary.md" <<EOF
# DG-V5 (vLLM DiffusionGemma 26B-A4B NVFP4-KV, Spark sm_121, image $R11_IMAGE)
- coherence: see {nvfp4,bf16}_coherence.json (expect Tokyo / 4 / Spark coherent)
- full-NVFP4 proof + VO-split proof: nvfp4_proofs.txt (mixed_kv False, head_dim 512->vo 256)
- capacity ratio: nvfp4 KV tokens / bf16 KV tokens (from *_proofs.txt) -- target ~3.56x
- parity: SGLang DG-R5 (3.5654x full / 3.5625x SWA), DG-R6 perf pair
EOF
echo "[DG-V] done -> $RUN_ROOT/summary.md"
