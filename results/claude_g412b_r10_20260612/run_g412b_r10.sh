#!/usr/bin/env bash
# PART B2: G4-12B claim pair on the r10 image (transformers 5.11.0 BAKED, no
# overlay) per docs/TRITON_RETIREMENT_SCORECARD.md - closes the named open box
# from the 2026-06-12 ~03:00 adjudication if both routes are green.
#   g412b_tri   : NO knobs (upstream heterogeneous-head force -> TRITON_ATTN),
#                 scorecard shape (smokes -> C1 x2 bitwise -> bench)
#   g412b_fi    : VLLM_FLASHINFER_VOSPLIT=1, same shape
#   g412b_nvfp4 : --kv-cache-dtype nvfp4 + VOSPLIT + LINEAR_V_SF; the size's
#                 first quantized cell. ORDER PROVENANCE: SCORE-FIRST (COLD) -
#                 C1 x2 BEFORE any smoke, per the Part A order-control finding
#                 protocol. Capacity from the GPU KV cache size line.
# R1 band: FI C1 <= Tri C1 + 0.05 nats (checked in summary).
set -uo pipefail
R=/home/jethac/spark_tmp/claude_g412b_r10_20260612
IMAGE=${IMAGE:-jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r10}
CTX=8191
MODEL=google/gemma-4-12B-it
SERVED=gemma4-12b-it
S=$R/status.txt
T0=$(date +%s)

echo "PAIR_RUN_START $(date -Is) IMAGE=${IMAGE} IMAGE_ID=$(docker images --format '{{.ID}}' ${IMAGE})" >> "$S"

declare -A WANT_MD5=(
  [c1_ppl_corpus.md]=abb63f0e65247a25f870d3f2d57563ff
)
for f in "${!WANT_MD5[@]}"; do
  got=$(md5sum "${R}/docs/${f}" | awk '{print $1}')
  echo "CORPUS ${f} MD5=${got} BYTES=$(wc -c < "${R}/docs/${f}")" >> "$S"
  if [ "${got}" != "${WANT_MD5[$f]}" ]; then
    echo "CORPUS_MD5_RED ${f} got=${got} want=${WANT_MD5[$f]} - HARD STOP" >> "$S"
    exit 9
  fi
done

wait_ready() {
  local name=$1
  for _ in $(seq 1 200); do
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
  grep -n "EXT_PATH\|VLLM_BUILD_CHECK\|TRANSFORMERS_CHECK\|attention backend\|AttentionBackendEnum\|TRITON\|Triton\|FLASHINFER\|FA2 VO split\|GPU KV cache size\|V-SF\|VLLM_NVFP4\|VLLM_FLASHINFER\|max_mma_kv\|kv_cache_dtype\|heterogeneous head\|Traceback\|ValueError\|ERROR" \
    "$R/results/claude_${label}_server.log" > "$R/results/claude_${label}_proof_lines.txt" 2>&1
  local kvline tri_sel fi_sel
  kvline=$(grep -o "GPU KV cache size: [0-9,]* tokens" "$R/results/claude_${label}_server.log" | head -1)
  tri_sel=$(grep -c "Using AttentionBackendEnum.TRITON_ATTN" "$R/results/claude_${label}_server.log")
  fi_sel=$(grep -c "Using AttentionBackendEnum.FLASHINFER" "$R/results/claude_${label}_server.log")
  echo "ROW=${label} KV_CAPACITY ${kvline:-NOT_FOUND}" >> "$S"
  echo "ROW=${label} ROUTE TRITON_SEL=${tri_sel} FI_SEL=${fi_sel}" >> "$S"
}

run_ppl() {
  local name=$1 label=$2 kvlabel=$3 cell=$4
  docker exec "${name}" python3 scripts/vllm_prompt_ppl_sweep.py \
    --url http://127.0.0.1:8000 \
    --model "${SERVED}" \
    --tokenizer "${MODEL}" \
    --text-file "docs/c1_ppl_corpus.md" \
    --ctx "${CTX}" \
    --run-id "claude_g412b_r10_${label}_${cell}_ctx${CTX}" \
    --kv-cache-dtype "${kvlabel}" \
    --runtime-ref "r10 image (baked, transformers 5.11.0); g412b claim pair row=${label} cell=${cell}" \
    --container-image "${IMAGE}" \
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

run_smokes() {
  local name=$1 label=$2
  docker exec "${name}" python3 scripts/openai_chat_smoke.py \
    --model "${SERVED}" --max-tokens 16 \
    --output "results/claude_${label}_smoke_sparkok.json" \
    > /dev/null 2> "$R/results/claude_${label}_smoke_sparkok.stderr"
  echo "ROW=${label} SMOKE_SPARKOK_RC=$?" >> "$S"
  docker exec "${name}" python3 scripts/openai_chat_smoke.py \
    --model "${SERVED}" --max-tokens 24 \
    --prompt "The capital of Japan is" \
    --output "results/claude_${label}_smoke_tokyo.json" \
    > /dev/null 2> "$R/results/claude_${label}_smoke_tokyo.stderr"
  if grep -qi "tokyo" "$R/results/claude_${label}_smoke_tokyo.json"; then
    echo "ROW=${label} SMOKE_TOKYO=COHERENT" >> "$S"
  else
    echo "ROW=${label} SMOKE_TOKYO=SUSPECT" >> "$S"
  fi
}

run_bench() {
  local name=$1 label=$2
  docker exec "${name}" python3 scripts/bench_e3.py \
    --url http://127.0.0.1:8000 \
    --model "${SERVED}" \
    --phase g412b-r10 \
    --run-id "claude_g412b_r10_${label}_bench" \
    --reps 3 \
    --output "results/claude_${label}_bench.json" \
    > /dev/null 2> "$R/results/claude_${label}_bench.stderr"
  local rc=$?
  local line
  line=$(python3 - "$R/results/claude_${label}_bench.json" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
c = d["cases"]
print(f"decode={c['short_decode']['median_decode_tok_s']} ttft={c['long_prefill']['median_ttft_s']} prefill={c['long_prefill']['median_prefill_tok_s']} x4={c['concurrent4']['median_aggregate_decode_tok_s']}")
PY
)
  echo "ROW=${label} BENCH_RC=${rc} ${line:-PARSE_FAIL}" >> "$S"
}

# run_row <label> <order smokefirst|scorefirst> <kvflag-or-NONE> <kvlabel> <bench 0|1> [extra docker -e args...]
run_row() {
  local label=$1 order=$2 kvflag=$3 kvlabel=$4 bench=$5; shift 5
  local extra_docker=("$@")
  local name=claude_g412b_${label}
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
      python3 -c 'import transformers; print(\"TRANSFORMERS_CHECK\", transformers.__version__, flush=True)'; \
      vllm serve ${MODEL} \
      --served-model-name ${SERVED} \
      --host 0.0.0.0 --port 8000 \
      ${kvarg} \
      --gpu-memory-utilization 0.72 \
      --max-model-len 8192 \
      --language-model-only"

  wait_ready "${name}"; rc=$?
  echo "ROW=${label} READY_RC=${rc} READY_WALL=$(( $(date +%s) - t0 ))" >> "$S"
  extract_proof "${label}"
  if [ $rc -ne 0 ]; then
    echo "ROW=${label} SERVER_DID_NOT_BECOME_READY" >> "$S"
    grep -B2 -A80 "max_mma_kv\|EngineCore failed\|Traceback\|ValidationError\|InductorError" "$R/results/claude_${label}_server.log" \
      > "$R/results/claude_${label}_crash_excerpt.txt" 2>&1 || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
    return 2
  fi
  echo "ROW=${label} VERDICT=SERVED" >> "$S"

  if [ "${order}" = "scorefirst" ]; then
    echo "ROW=${label} ORDER=SCORE_FIRST_COLD (quantized-KV order provenance)" >> "$S"
    run_ppl "${name}" "${label}" "${kvlabel}" c1a
    run_ppl "${name}" "${label}" "${kvlabel}" c1b
    det_check "${label}"
    run_smokes "${name}" "${label}"
  else
    echo "ROW=${label} ORDER=SMOKES_THEN_SCORE (scorecard shape; bf16 order-insensitive)" >> "$S"
    run_smokes "${name}" "${label}"
    run_ppl "${name}" "${label}" "${kvlabel}" c1a
    run_ppl "${name}" "${label}" "${kvlabel}" c1b
    det_check "${label}"
  fi

  if [ "${bench}" = "1" ]; then
    run_bench "${name}" "${label}"
  fi

  extract_proof "${label}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  echo "ROW=${label} DONE ROW_WALL=$(( $(date +%s) - t0 ))" >> "$S"
}

run_row tri smokefirst NONE bf16 1
echo "ROW=tri FINAL_RC=$?" >> "$S"
run_row fi smokefirst NONE bf16 1 -e VLLM_FLASHINFER_VOSPLIT=1
echo "ROW=fi FINAL_RC=$?" >> "$S"
run_row nvfp4 scorefirst nvfp4 nvfp4 0 -e VLLM_NVFP4_KV_VOSPLIT=1 -e VLLM_NVFP4_KV_LINEAR_V_SF=1
echo "ROW=nvfp4 FINAL_RC=$?" >> "$S"

echo "G412B_R10_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
