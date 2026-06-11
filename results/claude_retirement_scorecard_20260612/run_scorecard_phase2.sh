#!/usr/bin/env bash
# Scorecard phase 2 (same window):
#  1. s31b_tri_r2  : retry of the 31B bf16-Triton comparator (r1 crashed in inductor
#                    autotune "CUDA driver error: operation not permitted" - transient test)
#  2. (only if r2 fails) s31b_tri_eager : --enforce-eager LABELED fallback, PPL-only value
#  3. s31b_fp8_r2  : fp8 cross-window mismatch probe - EXACT corpus-sweep shape
#                    (C1 sweeps FIRST, no smokes before scoring) to test order/state dependence
#                    of today's 4.591455999476844 vs banked 4.473945385741097.
set -uo pipefail
R=/home/jethac/spark_tmp/claude_retirement_scorecard_20260612
IMAGE=jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9
CTX=8191
S=$R/status.txt
M=google/gemma-4-31B-it; SV=gemma4-31b-it

echo "PHASE2_START $(date -Is)" >> "$S"

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
  grep -n "EXT_PATH\|VLLM_BUILD_CHECK\|attention backend\|AttentionBackendEnum\|TRITON\|Triton\|FLASHINFER\|FA2 VO split\|GPU KV cache size\|max_mma_kv\|kv_cache_dtype\|heterogeneous head\|Traceback\|ValueError\|ERROR" \
    "$R/results/claude_${label}_server.log" > "$R/results/claude_${label}_proof_lines.txt" 2>&1
  local kvline tri_sel fi_sel
  kvline=$(grep -o "GPU KV cache size: [0-9,]* tokens" "$R/results/claude_${label}_server.log" | head -1)
  tri_sel=$(grep -c "Using AttentionBackendEnum.TRITON_ATTN" "$R/results/claude_${label}_server.log")
  fi_sel=$(grep -c "Using AttentionBackendEnum.FLASHINFER" "$R/results/claude_${label}_server.log")
  echo "ROW=${label} KV_CAPACITY ${kvline:-NOT_FOUND}" >> "$S"
  echo "ROW=${label} ROUTE TRITON_SEL=${tri_sel} FI_SEL=${fi_sel}" >> "$S"
}

run_ppl() {
  local name=$1 label=$2 cell=$3
  docker exec "${name}" python3 scripts/vllm_prompt_ppl_sweep.py \
    --url http://127.0.0.1:8000 \
    --model "${SV}" \
    --tokenizer "${M}" \
    --text-file "docs/c1_ppl_corpus.md" \
    --ctx "${CTX}" \
    --run-id "claude_rsc_${label}_${cell}_ctx${CTX}" \
    --kv-cache-dtype "${4:-bf16}" \
    --runtime-ref "r9 image (baked); retirement scorecard PHASE2 row=${label} cell=${cell}" \
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
    --model "${SV}" --max-tokens 16 \
    --output "results/claude_${label}_smoke_sparkok.json" \
    > /dev/null 2> "$R/results/claude_${label}_smoke_sparkok.stderr"
  echo "ROW=${label} SMOKE_SPARKOK_RC=$?" >> "$S"
  docker exec "${name}" python3 scripts/openai_chat_smoke.py \
    --model "${SV}" --max-tokens 24 \
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
    --model "${SV}" \
    --phase scorecard-p2 \
    --run-id "claude_rsc_${label}_bench" \
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

# start_server <label> <kvarg-string> <extra-serve-args> [docker -e args...]
start_server() {
  local label=$1 kvarg=$2 extraserve=$3; shift 3
  docker run -d --rm --name "claude_rsc_${label}" --gpus all --net host --ipc host \
    --memory 100g --memory-swap 100g \
    -w /work \
    "$@" \
    -v /home/jethac/.cache/huggingface:/root/.cache/huggingface \
    -v "${R}:/work" \
    "${IMAGE}" \
    bash -lc "exec > /work/results/claude_${label}_server.log 2>&1; \
      python3 -c 'import vllm; print(\"VLLM_BUILD_CHECK vllm.__file__=\", vllm.__file__, flush=True)'; \
      python3 -c 'import vllm._C_stable_libtorch as m; print(\"EXT_PATH\", m.__file__, flush=True)'; \
      vllm serve ${M} \
      --served-model-name ${SV} \
      --host 0.0.0.0 --port 8000 \
      ${kvarg} ${extraserve} \
      --gpu-memory-utilization 0.72 \
      --max-model-len 8192 \
      --language-model-only"
}

finish_row() {
  local label=$1 t0=$2
  extract_proof "${label}"
  docker rm -f "claude_rsc_${label}" >/dev/null 2>&1 || true
  echo "ROW=${label} DONE ROW_WALL=$(( $(date +%s) - t0 ))" >> "$S"
}

# ---- 1. s31b_tri retry (full row: smokes, C1 x2, bench) ----
L=s31b_tri_r2; t0=$(date +%s)
start_server "$L" "" ""
wait_ready "claude_rsc_${L}"; rc=$?
echo "ROW=${L} READY_RC=${rc} READY_WALL=$(( $(date +%s) - t0 ))" >> "$S"
extract_proof "$L"
TRI_OK=0
if [ $rc -eq 0 ]; then
  TRI_OK=1
  echo "ROW=${L} VERDICT=SERVED" >> "$S"
  run_smokes "claude_rsc_${L}" "$L"
  run_ppl "claude_rsc_${L}" "$L" c1a
  run_ppl "claude_rsc_${L}" "$L" c1b
  det_check "$L"
  run_bench "claude_rsc_${L}" "$L"
else
  echo "ROW=${L} SERVER_DID_NOT_BECOME_READY" >> "$S"
  grep -B2 -A60 "EngineCore failed\|InductorError\|Traceback" "$R/results/claude_${L}_server.log" \
    > "$R/results/claude_${L}_crash_excerpt.txt" 2>&1 || true
fi
finish_row "$L" "$t0"
echo "ROW=${L} FINAL_RC=$(( 1 - TRI_OK ))" >> "$S"

# ---- 2. eager fallback ONLY if retry failed (LABELED: not bench-comparable) ----
if [ $TRI_OK -eq 0 ]; then
  L=s31b_tri_eager; t0=$(date +%s)
  start_server "$L" "" "--enforce-eager"
  wait_ready "claude_rsc_${L}"; rc=$?
  echo "ROW=${L} READY_RC=${rc} READY_WALL=$(( $(date +%s) - t0 ))" >> "$S"
  extract_proof "$L"
  if [ $rc -eq 0 ]; then
    echo "ROW=${L} VERDICT=SERVED (LABELED eager fallback)" >> "$S"
    run_smokes "claude_rsc_${L}" "$L"
    run_ppl "claude_rsc_${L}" "$L" c1a
    run_ppl "claude_rsc_${L}" "$L" c1b
    det_check "$L"
    run_bench "claude_rsc_${L}" "$L"
  else
    echo "ROW=${L} SERVER_DID_NOT_BECOME_READY" >> "$S"
    grep -B2 -A60 "EngineCore failed\|InductorError\|Traceback" "$R/results/claude_${L}_server.log" \
      > "$R/results/claude_${L}_crash_excerpt.txt" 2>&1 || true
  fi
  finish_row "$L" "$t0"
fi

# ---- 3. fp8 mismatch probe: corpus-sweep shape (C1 FIRST, smokes AFTER) ----
L=s31b_fp8_r2; t0=$(date +%s)
start_server "$L" "--kv-cache-dtype fp8_e4m3" ""
wait_ready "claude_rsc_${L}"; rc=$?
echo "ROW=${L} READY_RC=${rc} READY_WALL=$(( $(date +%s) - t0 ))" >> "$S"
extract_proof "$L"
if [ $rc -eq 0 ]; then
  echo "ROW=${L} VERDICT=SERVED (C1-first, corpus-sweep shape)" >> "$S"
  run_ppl "claude_rsc_${L}" "$L" c1a fp8_e4m3
  run_ppl "claude_rsc_${L}" "$L" c1b fp8_e4m3
  det_check "$L"
  run_smokes "claude_rsc_${L}" "$L"
else
  echo "ROW=${L} SERVER_DID_NOT_BECOME_READY" >> "$S"
fi
finish_row "$L" "$t0"

echo "PHASE2_DONE $(date -Is)" >> "$S"
