#!/usr/bin/env bash
# =============================================================================
# STAGED -- DO NOT RUN OUTSIDE A FUTURE SPARK WINDOW.
# =============================================================================
# Long-context RETRIEVAL window for task #38 (docs/RETRIEVAL_EVAL_PLAN.md, epoch2).
# Backs Jetha's public promise: "i've measured perplexity, not retrieval --
# needle-in-a-haystack test is next." The per-token stratification found NVFP4's
# prose error grows mildly with position (H-late); this window asks whether that
# nips DEEP-CONTEXT retrieval relative to bf16/fp8.
#
# This script is the retrieval twin of scripts/run_overnight_ladder.sh and reuses
# its exact serving discipline: ONE server at a time, r10 BAKED image, util 0.72,
# memory guardrails (--memory 100g --memory-swap 100g), wait-ready loop, proof-line
# extraction, chat smokes, and a per-row DOUBLE-RUN DETERMINISM GATE (the smallest
# context cell is scored TWICE and must match bit-for-bit or the row is DET_FAIL).
#
# THREE servers sequential, one per KV dtype, with the campaign knobs:
#   row bf16  : no --kv-cache-dtype, VLLM_FLASHINFER_VOSPLIT=1
#   row fp8   : --kv-cache-dtype fp8_e4m3, NO knob envs (forced TRITON_ATTN route);
#               NOTE fp8 is PER-BOOT BISTABLE -- the boot-profile note field in the
#               JSON records which profile this boot landed on (see order-control row).
#   row nvfp4 : --kv-cache-dtype nvfp4, VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1
#
# MODEL ORDER: google/gemma-4-E4B-it FIRST (cheap; the tweet-asker's flagship is
# the D=512-global 31B, staged as a documented STRETCH at the bottom -- enable by
# setting RUN_31B=1 only when the window has the budget).
#
# GRID (per row): context lengths x depths via scripts/vllm_needle_retrieval.py.
#   context lengths : 1024 2048 4096 8192 16384 32768 (capped at --max-context-len)
#   depths          : 0.0 0.1 0.25 0.5 0.75 0.9 1.0
# The decisive cross-cut: does nvfp4 retrieval accuracy fall off at deep context /
# late position (high context_len AND depth >= 0.75) relative to bf16/fp8?
#
# MARKER PROTOCOL: this runner is launched by Codex's dispatcher when the Spark is
# idle. It writes a heartbeat marker and an append-only status.txt; the absence of
# the DONE marker means the window did not complete (do not claim partial rows).
# =============================================================================
set -uo pipefail

R=/home/jethac/spark_tmp/claude_needle_retrieval_20260612
IMAGE=jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r10
SERVE_MAX_LEN=${SERVE_MAX_LEN:-32768}   # served max-model-len; STRETCH above the
                                        # proven 8192 E4B PPL window -- if the
                                        # server will not boot at 32768, fall back
                                        # to 8192 and set MAX_CTX=8192 below.
MAX_CTX=${MAX_CTX:-32768}               # grid cap (must be <= SERVE_MAX_LEN)
RUN_31B=${RUN_31B:-0}                   # documented stretch; off by default
SEED=38
S=$R/status.txt
MARKER=$R/MARKER_needle_retrieval.txt
T0=$(date +%s)

mkdir -p "$R/results"
echo "MARKER_OPEN $(date -Is) pid=$$ host=$(hostname)" > "$MARKER"
echo "RUN_START $(date -Is) IMAGE=${IMAGE} IMAGE_ID=$(docker images --format '{{.ID}}' ${IMAGE}) SERVE_MAX_LEN=${SERVE_MAX_LEN} MAX_CTX=${MAX_CTX}" >> "$S"

wait_ready() {
  local name=$1
  for _ in $(seq 1 240); do
    if docker exec -i "${name}" python3 -c "import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:8000/v1/models\", timeout=2).read()" >/dev/null 2>&1; then
      return 0
    fi
    if ! docker ps --format "{{.Names}}" | grep -q "^${name}$"; then return 3; fi
    sleep 5
  done
  return 1
}

extract_proof() {
  local label=$1
  grep -n "EXT_PATH\|VLLM_BUILD_CHECK\|attention backend\|TRITON\|Triton\|FLASHINFER\|FA2 VO split\|GPU KV cache size\|V-SF\|VLLM_NVFP4\|VLLM_FLASHINFER\|max_mma_kv\|kv_cache_dtype\|Traceback\|ValueError\|ERROR" \
    "$R/results/claude_${label}_server.log" > "$R/results/claude_${label}_proof_lines.txt" 2>&1
  local kvline
  kvline=$(grep -o "GPU KV cache size: [0-9,]* tokens" "$R/results/claude_${label}_server.log" | head -1)
  echo "ROW=${label} KV_CAPACITY ${kvline:-NOT_FOUND}" >> "$S"
}

run_smokes() {
  local name=$1 served=$2 label=$3
  docker exec "${name}" python3 scripts/openai_chat_smoke.py \
    --model "${served}" --max-tokens 16 \
    --output "results/claude_${label}_smoke_sparkok.json" \
    > /dev/null 2> "$R/results/claude_${label}_smoke_sparkok.stderr"
  echo "ROW=${label} SMOKE_SPARKOK_RC=$?" >> "$S"
  docker exec "${name}" python3 scripts/openai_chat_smoke.py \
    --model "${served}" --max-tokens 24 \
    --prompt "The capital of Japan is" \
    --output "results/claude_${label}_smoke_tokyo.json" \
    > /dev/null 2> "$R/results/claude_${label}_smoke_tokyo.stderr"
  if grep -qi "tokyo" "$R/results/claude_${label}_smoke_tokyo.json"; then
    echo "ROW=${label} SMOKE_TOKYO=COHERENT" >> "$S"
  else
    echo "ROW=${label} SMOKE_TOKYO=SUSPECT" >> "$S"
  fi
}

# run_grid <name> <model> <served> <label> <kvlabel> <mode> <run-tag>
run_grid() {
  local name=$1 model=$2 served=$3 label=$4 kvlabel=$5 mode=$6 tag=$7
  docker exec "${name}" python3 scripts/vllm_needle_retrieval.py \
    --url http://127.0.0.1:8000 \
    --model "${served}" \
    --tokenizer "${model}" \
    --mode "${mode}" \
    --seed "${SEED}" \
    --max-context-len "${MAX_CTX}" \
    --run-id "claude_${label}_${tag}" \
    --kv-cache-dtype "${kvlabel}" \
    --boot-profile-note "record fp8 per-boot profile from C1 PPL cross-check if known; else UNKNOWN" \
    --runtime-ref "r10 BAKED ${IMAGE}; needle-retrieval window row=${label} mode=${mode}" \
    --container-image "${IMAGE}" \
    --output "results/claude_${label}_${tag}.json" \
    > "$R/results/claude_${label}_${tag}_stdout.json" \
    2> "$R/results/claude_${label}_${tag}_stderr.log"
  echo "GRID=${label}_${tag} RC=$?" >> "$S"
}

# Double-run determinism gate: smallest-context single-needle cell scored twice.
det_gate() {
  local name=$1 model=$2 served=$3 label=$4 kvlabel=$5
  for rep in a b; do
    docker exec "${name}" python3 scripts/vllm_needle_retrieval.py \
      --url http://127.0.0.1:8000 \
      --model "${served}" --tokenizer "${model}" \
      --mode single --seed "${SEED}" \
      --context-len 1024 --depth 0.5 \
      --run-id "claude_${label}_det${rep}" \
      --kv-cache-dtype "${kvlabel}" \
      --container-image "${IMAGE}" \
      --output "results/claude_${label}_det${rep}.json" \
      > "$R/results/claude_${label}_det${rep}_stdout.json" \
      2> "$R/results/claude_${label}_det${rep}_stderr.log"
  done
  # Compare the canonicalised answer preview of the single cell bit-for-bit.
  local a b
  a=$(python3 -c "import json;d=json.load(open('$R/results/claude_${label}_deta.json'));print(d['cells'][0]['score']['answer_preview'])" 2>/dev/null)
  b=$(python3 -c "import json;d=json.load(open('$R/results/claude_${label}_detb.json'));print(d['cells'][0]['score']['answer_preview'])" 2>/dev/null)
  if [ -n "${a}" ] && [ "${a}" = "${b}" ]; then
    echo "ROW=${label} DETERMINISM=IDENTICAL" >> "$S"
  else
    echo "ROW=${label} DETERMINISM=DET_FAIL a=[${a:-none}] b=[${b:-none}]" >> "$S"
  fi
}

# run_row <model> <served> <label> <kvflag-or-NONE> <kvlabel> [extra docker -e args...]
run_row() {
  local model=$1 served=$2 label=$3 kvflag=$4 kvlabel=$5; shift 5
  local extra_docker=("$@")
  local name=claude_nr_${label}
  local kvarg=""
  if [ "${kvflag}" != "NONE" ]; then kvarg="--kv-cache-dtype ${kvflag}"; fi
  local t0=$(date +%s)
  docker run -d --rm --name "${name}" --gpus all --net host --ipc host \
    --memory 100g --memory-swap 100g \
    -w /work \
    "${extra_docker[@]}" \
    -v /home/jethac/.cache/huggingface:/root/.cache/huggingface \
    -v "${R}:/work" \
    "${IMAGE}" \
    bash -lc "exec > /work/results/claude_${label}_server.log 2>&1; \
      python3 -c 'import vllm; print(\"VLLM_BUILD_CHECK vllm.__file__=\", vllm.__file__, flush=True)'; \
      python3 -c 'import vllm._C_stable_libtorch as m; print(\"EXT_PATH\", m.__file__, flush=True)'; \
      vllm serve ${model} \
      --served-model-name ${served} \
      --host 0.0.0.0 --port 8000 \
      ${kvarg} \
      --gpu-memory-utilization 0.72 \
      --max-model-len ${SERVE_MAX_LEN} \
      --language-model-only"

  wait_ready "${name}"; rc=$?
  echo "ROW=${label} READY_RC=${rc} READY_WALL=$(( $(date +%s) - t0 ))" >> "$S"
  extract_proof "${label}"
  if [ $rc -ne 0 ]; then
    echo "ROW=${label} SERVER_DID_NOT_BECOME_READY" >> "$S"
    grep -B2 -A80 "max_mma_kv\|EngineCore failed\|Traceback" "$R/results/claude_${label}_server.log" \
      > "$R/results/claude_${label}_crash_excerpt.txt" 2>&1 || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
    return 2
  fi
  echo "ROW=${label} VERDICT=SERVED" >> "$S"

  run_smokes "${name}" "${served}" "${label}"
  det_gate   "${name}" "${model}" "${served}" "${label}" "${kvlabel}"
  run_grid   "${name}" "${model}" "${served}" "${label}" "${kvlabel}" single needlegrid
  run_grid   "${name}" "${model}" "${served}" "${label}" "${kvlabel}" multi  rulergrid

  extract_proof "${label}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  echo "ROW=${label} DONE ROW_WALL=$(( $(date +%s) - t0 ))" >> "$S"
}

# ---- Model 1: gemma-4-E4B-it (D512 globals -> vosplit knobs). CHEAP, runs first.
M=google/gemma-4-E4B-it; SV=gemma4-e4b-it
run_row "$M" "$SV" e4b_bf16  NONE     bf16     -e VLLM_FLASHINFER_VOSPLIT=1
echo "ROW=e4b_bf16 FINAL_RC=$?" >> "$S"
run_row "$M" "$SV" e4b_fp8   fp8_e4m3 fp8_e4m3
echo "ROW=e4b_fp8 FINAL_RC=$?" >> "$S"
run_row "$M" "$SV" e4b_nvfp4 nvfp4    nvfp4    -e VLLM_NVFP4_KV_VOSPLIT=1 -e VLLM_NVFP4_KV_LINEAR_V_SF=1
echo "ROW=e4b_nvfp4 FINAL_RC=$?" >> "$S"

# ---- Model 2 (STRETCH): gemma-4-31B-it -- the D=512-global flagship the tweet
# asker asked about. Heavy; enable only with budget via RUN_31B=1.
if [ "${RUN_31B}" = "1" ]; then
  M=google/gemma-4-31B-it; SV=gemma4-31b-it
  run_row "$M" "$SV" g31b_bf16  NONE     bf16     -e VLLM_FLASHINFER_VOSPLIT=1
  echo "ROW=g31b_bf16 FINAL_RC=$?" >> "$S"
  run_row "$M" "$SV" g31b_fp8   fp8_e4m3 fp8_e4m3
  echo "ROW=g31b_fp8 FINAL_RC=$?" >> "$S"
  run_row "$M" "$SV" g31b_nvfp4 nvfp4    nvfp4    -e VLLM_NVFP4_KV_VOSPLIT=1 -e VLLM_NVFP4_KV_LINEAR_V_SF=1
  echo "ROW=g31b_nvfp4 FINAL_RC=$?" >> "$S"
fi

echo "NEEDLE_RETRIEVAL_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
echo "MARKER_DONE $(date -Is)" >> "$MARKER"
