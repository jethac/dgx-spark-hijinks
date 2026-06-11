#!/usr/bin/env bash
# P520 Gemma 4 AUDIO mm retirement cells (Amendment 5, epoch2 dgx-spark-hijinks).
# Pattern of ~/gemma3_1b_serving_20260612/run_serving_rows.sh + the image mm
# smoke protocol. Runs LAST in the GPU queue (after small-size ladder, MTP,
# image mm smokes) - claim only when nvidia-smi is free 3 consecutive checks
# 2 min apart; NEVER kill another agent's process.
#
# Install: the Amendment-4 second P520 install (~/vllm_e2_env venv +
# ~/vllm-e2 editable clone on spark/hijinks-e2-mm-retire), FlashInfer source
# tree ~/flashinfer @ 7d5d477b on PYTHONPATH.
#
# Per model (gemma-4-E2B-it, gemma-4-E4B-it), three rows, ONE server at a time:
#   triton_bf16 : VLLM_FLASHINFER_MM_PREFIX=0, --attention-backend TRITON_ATTN
#                 (escape hatch -> upstream mm-capable Triton route; comparator)
#   fi_bf16     : defaults (mm flip on), --attention-backend FLASHINFER
#   fi_nvfp4    : + --kv-cache-dtype nvfp4, VLLM_NVFP4_KV_LINEAR_V_SF=1,
#                 VLLM_NVFP4_KV_VOSPLIT=1 (D512 globals stay causal by the
#                 'vision' policy; audio is causal EVERYWHERE by the
#                 Amendment-5 policy verdict)
#   NOTE: no triton_nvfp4 cell - Triton cannot read quantized KV at all
#         (scorecard I2); the nvfp4 comparator is triton_bf16, semantic gate.
# Per row: proof lines (R5), speech smoke (transcript-grounded, x2 repeats
# byte-identical), tone-control smoke (banked verbatim), text-only smoke.
set -uo pipefail

R=$HOME/audio_mm_20260612
A=$R/assets
W=/mnt/b/workshop/wsl_sm120
PORT=8000
S=$R/status.txt
mkdir -p "$R/results" "$A"

# Assets (deterministic, banked): copy from the campaign results dir.
CAMP=/mnt/b/workshop/worktrees/dgx-spark-hijinks/spark-hijinks-022-gemma4-mixed-kv/results/p520_audio_mm_20260612
cp -f "$CAMP/assets/"*.wav "$CAMP/assets/assets_manifest.json" "$CAMP/assets/speech_transcript.txt" "$A/" || exit 1
SPEECH=$A/speech_librispeech_1272-128104-0000.wav
TONE=$A/tone_control.wav
md5sum "$A"/*.wav >> "$S"

source $HOME/vllm_e2_env/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=$HOME/flashinfer
export TORCH_CUDA_ARCH_LIST=12.0a
export FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1

E2_HEAD=$(git -C $HOME/vllm-e2 rev-parse --short HEAD)
RUNTIME_REF="P520 RTX 5060 Ti sm_120 WSL2; jethac/vllm@${E2_HEAD} (~/vllm-e2 editable, spark/hijinks-e2-mm-retire); jethac/flashinfer@7d5d477b source-tree PYTHONPATH"
echo "RUN_START $(date -Is) ${RUNTIME_REF}" >> "$S"

wait_ready() {
  local pid=$1
  for _ in $(seq 1 360); do
    if curl -fsS -m 3 http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then return 3; fi
    sleep 5
  done
  return 1
}

extract_proof() {
  local label=$1
  grep -n "EXT_PATH\|attention backend\|Using.*backend\|TRITON\|Triton\|triton_attn\|FLASHINFER\|FlashInfer\|mm-prefix\|mm_prefix\|custom-mask\|GPU KV cache size\|Maximum concurrency\|kv_cache_dtype\|cache_dtype\|V-SF\|VLLM_NVFP4\|VLLM_FLASHINFER\|audio\|Audio\|disable_chunked_mm_input\|Traceback\|ValueError\|ERROR" \
    "$R/results/${label}_server.log" > "$R/results/${label}_proof_lines.txt" 2>&1
}

# run_row <model> <served> <label> <backend> <kvflag-or-NONE> [ENV=VAL...]
run_row() {
  local model=$1 served=$2 label=$3 backend=$4 kvflag=$5; shift 5
  local kvarg=()
  if [ "$kvflag" != "NONE" ]; then kvarg=(--kv-cache-dtype "$kvflag"); fi
  local log=$R/results/${label}_server.log
  local t0=$(date +%s)
  echo "ROW_START ${label} $(date -Is)" >> "$S"

  ( exec > "$log" 2>&1
    env "$@" vllm serve "$model" \
      --served-model-name "$served" \
      --attention-backend "$backend" \
      "${kvarg[@]}" \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.85 \
      --disable-chunked-mm-input \
      --port ${PORT} ) &
  local pid=$!
  wait_ready $pid
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "ROW_FAIL ${label} server_not_ready rc=${rc} $(date -Is)" >> "$S"
    extract_proof "$label"
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    return 1
  fi

  python "$W/audio_mm_smoke.py" --url http://127.0.0.1:${PORT} --model "$served" \
    --audio "$SPEECH" --prompt "What is said in this audio?" \
    --expect-keywords "quilter" --label "${label}_speech" \
    --output "$R/results/${label}_speech_smoke.json"
  echo "SMOKE ${label}_speech exit=$?" >> "$S"

  python "$W/audio_mm_smoke.py" --url http://127.0.0.1:${PORT} --model "$served" \
    --audio "$TONE" --prompt "Describe this audio." \
    --label "${label}_tone" \
    --output "$R/results/${label}_tone_smoke.json"
  echo "SMOKE ${label}_tone exit=$?" >> "$S"

  python $HOME/gemma3_1b_serving_20260612/scripts/openai_chat_smoke.py \
    --url http://127.0.0.1:${PORT} --model "$served" \
    --output "$R/results/${label}_text_smoke.json"
  echo "SMOKE ${label}_text exit=$?" >> "$S"

  extract_proof "$label"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 10
  echo "ROW_END ${label} dur=$(( $(date +%s) - t0 ))s $(date -Is)" >> "$S"
}

run_model() {
  local model=$1 served=$2 tag=$3
  python $HOME/hijinks/scripts/hf_model_access_probe.py "$model" \
    > "$R/results/${tag}_hf_access_probe.json" 2>&1 || {
      echo "PREFLIGHT_FAIL ${tag} hf_access $(date -Is)" >> "$S"; return 1; }

  run_row "$model" "$served" "${tag}_triton_bf16" TRITON_ATTN NONE \
    VLLM_FLASHINFER_MM_PREFIX=0
  run_row "$model" "$served" "${tag}_fi_bf16" FLASHINFER NONE
  run_row "$model" "$served" "${tag}_fi_nvfp4" FLASHINFER nvfp4 \
    VLLM_NVFP4_KV_LINEAR_V_SF=1 VLLM_NVFP4_KV_VOSPLIT=1
}

run_model google/gemma-4-E2B-it gemma4-e2b-it e2b
run_model google/gemma-4-E4B-it gemma4-e4b-it e4b

echo "RUN_END $(date -Is)" >> "$S"
# Bank: copy results into the campaign repo results dir.
mkdir -p "$CAMP/cells"
cp -f "$R/results/"* "$CAMP/cells/" 2>/dev/null
cp -f "$S" "$CAMP/cells/status.txt"
echo done
