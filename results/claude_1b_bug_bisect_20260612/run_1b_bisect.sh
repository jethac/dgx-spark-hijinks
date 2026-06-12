#!/usr/bin/env bash
# Gemma 3 1B (d256 / SWA-512 / 1 KV head) FlashInfer-numerics bisect on the
# Spark (GB10, sm_121). Decides GEOMETRY vs sm_120-PLATFORM for the open bug
# docs/BUG_FLASHINFER_GEMMA3_1B_SERVING_NUMERICS.md.
#   fa_bf16  : --attention-backend flash_attn, no kv-cache-dtype  (TRUTH ref)
#   fi_bf16  : --attention-backend flashinfer, no kv-cache-dtype  (THE suspect)
#   fi_nvfp4 : --attention-backend flashinfer, --kv-cache-dtype nvfp4
#              + VLLM_NVFP4_KV_LINEAR_V_SF=1                      (discriminator)
# Each: chat smoke "The capital of Japan is" + C1 PPL ctx 8191 DOUBLE-RUN bitwise.
# One server at a time; --memory 100g --memory-swap 100g; util 0.3 (plenty @1B).
set -uo pipefail
R=/home/jethac/spark_tmp/claude_1b_bug_bisect_20260612
IMAGE=${IMAGE:-jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9}
CTX=8191
MODEL=google/gemma-3-1b-it
SERVED=gemma3-1b-it
S=$R/status.txt
T0=$(date +%s)

echo "BISECT_RUN_START $(date -Is) IMAGE=${IMAGE} IMAGE_ID=$(docker images --format '{{.ID}}' ${IMAGE})" >> "$S"

# corpus gate
WANT=abb63f0e65247a25f870d3f2d57563ff
got=$(md5sum "${R}/docs/c1_ppl_corpus.md" | awk '{print $1}')
echo "CORPUS c1_ppl_corpus.md MD5=${got} BYTES=$(wc -c < "${R}/docs/c1_ppl_corpus.md")" >> "$S"
if [ "${got}" != "${WANT}" ]; then
  echo "CORPUS_MD5_RED got=${got} want=${WANT} - HARD STOP" >> "$S"; exit 9
fi

wait_ready() {
  local name=$1
  for _ in $(seq 1 240); do
    if docker exec -i "${name}" python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2).read()" >/dev/null 2>&1; then
      return 0
    fi
    if ! docker ps --format "{{.Names}}" | grep -q "^${name}$"; then return 3; fi
    sleep 5
  done
  return 1
}

extract_proof() {
  local label=$1
  grep -nE "EXT_PATH|VLLM_BUILD_CHECK|TRANSFORMERS_CHECK|FLASHINFER_CHECK|Using AttentionBackendEnum|GPU KV cache size|Maximum concurrency|kv_cache_dtype|LINEAR V-SF|V-SF|VLLM_NVFP4|Warming up FlashInfer|Traceback|ValueError|RuntimeError|ERROR|OutOfMemory" \
    "$R/results/claude_${label}_server.log" > "$R/results/claude_${label}_proof_lines.txt" 2>&1
  local kvline fa_sel fi_sel
  kvline=$(grep -o "GPU KV cache size: [0-9,]* tokens" "$R/results/claude_${label}_server.log" | head -1)
  fa_sel=$(grep -c "Using AttentionBackendEnum.FLASH_ATTN" "$R/results/claude_${label}_server.log")
  fi_sel=$(grep -c "Using AttentionBackendEnum.FLASHINFER" "$R/results/claude_${label}_server.log")
  echo "ROW=${label} KV_CAPACITY ${kvline:-NOT_FOUND}" >> "$S"
  echo "ROW=${label} BACKEND_ENGAGED FLASH_ATTN_SEL=${fa_sel} FI_SEL=${fi_sel}" >> "$S"
}

run_ppl() {
  local name=$1 label=$2 kvlabel=$3 cell=$4
  docker exec "${name}" python3 scripts/vllm_prompt_ppl_sweep.py \
    --url http://127.0.0.1:8000 \
    --model "${SERVED}" \
    --tokenizer "${MODEL}" \
    --text-file "docs/c1_ppl_corpus.md" \
    --ctx "${CTX}" \
    --run-id "claude_1b_bisect_${label}_${cell}_ctx${CTX}" \
    --kv-cache-dtype "${kvlabel}" \
    --runtime-ref "r9 spark sm121 1B bisect row=${label} cell=${cell}" \
    --container-image "${IMAGE}" \
    --dump-token-logprobs "results/token_dumps_${label}_${cell}" \
    --output "results/claude_${label}_${cell}_ctx${CTX}_ppl.json" \
    > "$R/results/claude_${label}_${cell}_ctx${CTX}_stdout.json" \
    2> "$R/results/claude_${label}_${cell}_ctx${CTX}_stderr.log"
  local rc=$?
  local got
  got=$(python3 -c "import json; print(repr(json.load(open('$R/results/claude_${label}_${cell}_ctx${CTX}_ppl.json'))['contexts'][0]['score']['mean_nll_nats']))" 2>/dev/null)
  echo "CELL=${label}_${cell} RC=${rc} mean=${got:-PARSE_FAIL}" >> "$S"
}

det_check() {
  local label=$1
  local a b
  a=$(python3 -c "import json; print(repr(json.load(open('$R/results/claude_${label}_c1a_ctx${CTX}_ppl.json'))['contexts'][0]['score']['mean_nll_nats']))" 2>/dev/null)
  b=$(python3 -c "import json; print(repr(json.load(open('$R/results/claude_${label}_c1b_ctx${CTX}_ppl.json'))['contexts'][0]['score']['mean_nll_nats']))" 2>/dev/null)
  if [ -n "${a}" ] && [ "${a}" = "${b}" ]; then
    echo "ROW=${label} C1_DETERMINISM=IDENTICAL mean=${a}" >> "$S"
  else
    echo "ROW=${label} C1_DETERMINISM=DET_FAIL a=${a:-none} b=${b:-none}" >> "$S"
  fi
}

run_smoke() {
  local name=$1 label=$2
  docker exec "${name}" python3 scripts/openai_chat_smoke.py \
    --model "${SERVED}" --max-tokens 24 \
    --prompt "The capital of Japan is" \
    --expect-substring "Tokyo" \
    --output "results/claude_${label}_smoke_tokyo.json" \
    > /dev/null 2> "$R/results/claude_${label}_smoke_tokyo.stderr"
  local rc=$?
  if grep -qi "tokyo" "$R/results/claude_${label}_smoke_tokyo.json"; then
    echo "ROW=${label} SMOKE_TOKYO=COHERENT RC=${rc}" >> "$S"
  else
    echo "ROW=${label} SMOKE_TOKYO=SUSPECT_OR_GIBBERISH RC=${rc}" >> "$S"
  fi
}

# run_row <label> <backend flash_attn|flashinfer> <kvflag-or-NONE> <kvlabel> [extra -e args...]
run_row() {
  local label=$1 backend=$2 kvflag=$3 kvlabel=$4; shift 4
  local extra_docker=("$@")
  local name=claude_1b_${label}
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
      python3 -c 'import flashinfer; print(\"FLASHINFER_CHECK\", flashinfer.__file__, flush=True)'; \
      python3 -c 'import transformers; print(\"TRANSFORMERS_CHECK\", transformers.__version__, flush=True)'; \
      vllm serve ${MODEL} \
      --served-model-name ${SERVED} \
      --host 0.0.0.0 --port 8000 \
      --attention-backend ${backend} \
      ${kvarg} \
      --gpu-memory-utilization 0.3 \
      --max-model-len 8192"

  wait_ready "${name}"; rc=$?
  echo "ROW=${label} READY_RC=${rc} READY_WALL=$(( $(date +%s) - t0 ))" >> "$S"
  extract_proof "${label}"
  if [ $rc -ne 0 ]; then
    echo "ROW=${label} SERVER_DID_NOT_BECOME_READY" >> "$S"
    grep -B2 -A60 "Traceback\|EngineCore failed\|ValidationError\|InductorError\|operation not permitted\|OutOfMemory\|Unknown\|architecture" \
      "$R/results/claude_${label}_server.log" > "$R/results/claude_${label}_crash_excerpt.txt" 2>&1 || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
    return 2
  fi
  echo "ROW=${label} VERDICT=SERVED" >> "$S"

  run_smoke "${name}" "${label}"
  run_ppl "${name}" "${label}" "${kvlabel}" c1a
  run_ppl "${name}" "${label}" "${kvlabel}" c1b
  det_check "${label}"

  extract_proof "${label}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  echo "ROW=${label} DONE ROW_WALL=$(( $(date +%s) - t0 ))" >> "$S"
}

run_row fa_bf16  FLASH_ATTN NONE  bf16
echo "ROW=fa_bf16 FINAL_RC=$?" >> "$S"
run_row fi_bf16  FLASHINFER NONE  bf16
echo "ROW=fi_bf16 FINAL_RC=$?" >> "$S"
run_row fi_nvfp4 FLASHINFER nvfp4 nvfp4 -e VLLM_NVFP4_KV_LINEAR_V_SF=1
echo "ROW=fi_nvfp4 FINAL_RC=$?" >> "$S"

echo "BISECT_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
