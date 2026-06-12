#!/usr/bin/env bash
# Phase 3: mm-retire serving smokes on the P520 (sm120a release wheel + mm-retire overlay).
# Models: google/gemma-3-4b-it, google/gemma-4-E4B-it.
# Routes: FI mm route (FLASHINFER + VLLM_FLASHINFER_MM_PREFIX=1) vs Triton route
#         (TRITON_ATTN, MM_PREFIX=0). KV: bf16 + nvfp4 (nvfp4 is FlashInfer-only).
# Gates per cell: (a) image-grounded coherent answer; (b) FI vs Triton semantic
#   equivalence; (c) text-only token-identical knob-on vs knob-off; (d) repeat-determinism.
# One server at a time, util 0.85, port 8077. ABORT-soft: a RED is banked verbatim.
set -uo pipefail

R=/home/jetha/p520_mm_smokes_20260612
IMG=/home/jetha/mm_smoke_20260612/images
SMK=/mnt/b/workshop/wsl_sm120
PORT=8077
S=$R/status.txt
mkdir -p "$R/results" "$R/serverlogs"
: > "$S"

VENV=/home/jetha/vllm_wheel_env
source "$VENV/bin/activate"
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=/home/jetha/flashinfer
export TORCH_CUDA_ARCH_LIST=12.0a
export FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1
export HF_HOME=/mnt/b/workshop/hf_cache/huggingface
export VLLM_FLASHINFER_VOSPLIT=1

echo "RUN_START $(date -Is)" >> "$S"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv >> "$S"

wait_ready() {
  local pid=$1
  for _ in $(seq 1 420); do
    if curl -fsS -m 3 http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then return 3; fi
    sleep 5
  done
  return 1
}

stop_server() {
  local spid=$1
  kill "$spid" 2>/dev/null
  sleep 5
  pkill -f "vllm serve" 2>/dev/null
  sleep 12
  pkill -9 -f "vllm serve" 2>/dev/null
  sleep 6
}

extract_proof() {
  local cell=$1
  grep -nE "EXT_PATH|VLLM_BUILD_CHECK|FLASHINFER_CHECK|Using AttentionBackendEnum|FlashInfer mm-prefix|kv_cache_dtype|VLLM_FLASHINFER_MM_PREFIX|VLLM_NVFP4|Traceback|ValueError|ERROR|EngineCore failed" \
    "$R/serverlogs/${cell}.log" > "$R/serverlogs/${cell}_proof.txt" 2>&1 || true
}

# start_server <cell> <backend> <kvflag-or-NONE> <mm_prefix 0/1> [extra env VAR=VAL ...]
SPID=""
start_server() {
  local cell=$1 model=$2 backend=$3 kvflag=$4 mmpfx=$5; shift 5
  local extra=("$@")
  local kvarg=()
  [ "$kvflag" != "NONE" ] && kvarg=(--kv-cache-dtype "$kvflag")
  local log=$R/serverlogs/${cell}.log
  local t0=$(date +%s)

  ( exec > "$log" 2>&1
    export VLLM_FLASHINFER_MM_PREFIX="$mmpfx"
    for e in "${extra[@]:-}"; do [ -n "$e" ] && export "$e"; done
    python -c 'import vllm; print("VLLM_BUILD_CHECK", vllm.__file__, flush=True)'
    python -c 'import vllm._C_stable_libtorch as m; print("EXT_PATH", m.__file__, flush=True)'
    python -c 'import flashinfer; print("FLASHINFER_CHECK", flashinfer.__file__, flush=True)'
    echo "KNOB VLLM_FLASHINFER_MM_PREFIX=$VLLM_FLASHINFER_MM_PREFIX"
    exec vllm serve "$model" \
      --host 127.0.0.1 --port ${PORT} \
      ${kvarg[@]:+"${kvarg[@]}"} \
      --attention-backend ${backend} \
      --gpu-memory-utilization 0.90 \
      --max-model-len 2048 --enforce-eager \
      --limit-mm-per-prompt '{"image":1}' ) &
  SPID=$!
  wait_ready "$SPID"; local rc=$?
  echo "CELL=${cell} READY_RC=${rc} READY_WALL=$(( $(date +%s) - t0 ))" >> "$S"
  extract_proof "$cell"
  if [ $rc -ne 0 ]; then
    echo "CELL=${cell} SERVER_DID_NOT_BECOME_READY (RED)" >> "$S"
    grep -B2 -A60 "EngineCore failed\|Traceback\|ERROR" "$log" > "$R/serverlogs/${cell}_crash.txt" 2>&1 || true
    stop_server "$SPID"; SPID=""
    return 2
  fi
  echo "CELL=${cell} SERVED" >> "$S"
  return 0
}

img_smoke() {
  local cell=$1 model=$2 image=$3 prompt=$4 kw=$5
  python "$SMK/image_mm_smoke.py" \
    --url http://127.0.0.1:${PORT} --model "$model" \
    --image "$image" --prompt "$prompt" --expect-keywords "$kw" \
    --repeats 2 --label "$cell" \
    --output "$R/results/${cell}_img.json" \
    > "$R/results/${cell}_img_stdout.txt" 2>&1
  echo "CELL=${cell} IMG_SMOKE_RC=$?" >> "$S"
}

txt_smoke() {
  local cell=$1 model=$2
  python "$SMK/text_identity_smoke.py" \
    --url http://127.0.0.1:${PORT} --model "$model" \
    --repeats 2 --label "$cell" \
    --output "$R/results/${cell}_txt.json" \
    > "$R/results/${cell}_txt_stdout.txt" 2>&1
  echo "CELL=${cell} TXT_SMOKE_RC=$?" >> "$S"
}

# Primary deterministic grounding image: red circle ("red","circle").
# Deterministic grounding image: two_shapes = a BLUE square + a YELLOW triangle.
# Prompt asks for colors+shapes so the reply is describable; keywords are robust
# (a model that does NOT see the image cannot produce "blue"+"triangle").
GIMG=$IMG/two_shapes.png
GPROMPT="List the colors and the shapes you see in this image."
GKW="blue,triangle"

run_model() {
  local tag=$1 model=$2
  echo "===== MODEL $tag ($model) =====" >> "$S"

  # --- bf16 FI mm route (MM_PREFIX=1) ---
  if start_server "${tag}_bf16_fi" "$model" FLASHINFER NONE 1; then
    img_smoke "${tag}_bf16_fi" "$model" "$GIMG" "$GPROMPT" "$GKW"
    txt_smoke "${tag}_bf16_fi_knobon" "$model"
    stop_server "$SPID"; SPID=""
  fi

  # --- bf16 FI text knob-OFF (same FLASHINFER backend, MM_PREFIX=0) for gate(c) ---
  if start_server "${tag}_bf16_fi_knoboff" "$model" FLASHINFER NONE 0; then
    txt_smoke "${tag}_bf16_fi_knoboff" "$model"
    stop_server "$SPID"; SPID=""
  fi

  # --- bf16 Triton route (TRITON_ATTN, MM_PREFIX=0) gate(b) reference ---
  if start_server "${tag}_bf16_triton" "$model" TRITON_ATTN NONE 0; then
    img_smoke "${tag}_bf16_triton" "$model" "$GIMG" "$GPROMPT" "$GKW"
    stop_server "$SPID"; SPID=""
  fi

  # --- nvfp4 FI mm route (MM_PREFIX=1 + linear-V-SF + VOSPLIT) ---
  rm -rf /home/jetha/.cache/flashinfer 2>/dev/null || true
  if start_server "${tag}_nvfp4_fi" "$model" FLASHINFER nvfp4 1 \
       VLLM_NVFP4_KV_LINEAR_V_SF=1 VLLM_NVFP4_KV_VOSPLIT=1; then
    img_smoke "${tag}_nvfp4_fi" "$model" "$GIMG" "$GPROMPT" "$GKW"
    txt_smoke "${tag}_nvfp4_fi_knobon" "$model"
    stop_server "$SPID"; SPID=""
  fi

  # --- comparisons ---
  # gate (b) bf16: FI vs Triton semantic equivalence
  python "$SMK/compare_smoke.py" --mode semantic \
    --a "$R/results/${tag}_bf16_fi_img.json" \
    --b "$R/results/${tag}_bf16_triton_img.json" \
    --label "${tag}_bf16_FIvsTriton" \
    --output "$R/results/${tag}_bf16_FIvsTriton_cmp.json" \
    >> "$R/results/${tag}_cmp_stdout.txt" 2>&1
  echo "CELL=${tag}_bf16_FIvsTriton_CMP_RC=$?" >> "$S"

  # gate (b) nvfp4: FI-nvfp4 vs Triton-bf16 semantic equivalence
  python "$SMK/compare_smoke.py" --mode semantic \
    --a "$R/results/${tag}_nvfp4_fi_img.json" \
    --b "$R/results/${tag}_bf16_triton_img.json" \
    --label "${tag}_nvfp4FIvsTritonBf16" \
    --output "$R/results/${tag}_nvfp4_FIvsTriton_cmp.json" \
    >> "$R/results/${tag}_cmp_stdout.txt" 2>&1
  echo "CELL=${tag}_nvfp4_FIvsTriton_CMP_RC=$?" >> "$S"

  # gate (c) text token-identity: FI knob-on vs knob-off (bf16)
  python "$SMK/compare_smoke.py" --mode identical \
    --a "$R/results/${tag}_bf16_fi_knobon_txt.json" \
    --b "$R/results/${tag}_bf16_fi_knoboff_txt.json" \
    --label "${tag}_bf16_text_knob_identity" \
    --output "$R/results/${tag}_bf16_text_identity_cmp.json" \
    >> "$R/results/${tag}_cmp_stdout.txt" 2>&1
  echo "CELL=${tag}_bf16_text_identity_CMP_RC=$?" >> "$S"
}

run_model g3_4b google/gemma-3-4b-it
run_model g4_e4b google/gemma-4-E4B-it

echo "SMOKES_DONE $(date -Is)" >> "$S"
