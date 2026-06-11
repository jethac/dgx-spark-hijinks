#!/usr/bin/env bash
# SPARK MORNING SCORECARD BLOCK - Triton retirement proof
# (docs/TRITON_RETIREMENT_SCORECARD.md R1-R5/I1-I4; docs/TRITON_RETIREMENT_NOTES.md par.6;
#  docs/OVERNIGHT_LADDER_PLAN_20260612.md zero-bug amendments)
#
# PAIRED cells, r9 baked image serves BOTH routes via knobs:
#   Triton comparator rows: NO campaign knobs (upstream force-routes Gemma 4 to TRITON_ATTN)
#   FlashInfer rows:        VLLM_FLASHINFER_VOSPLIT=1 (Gemma 4) / --attention-backend FLASHINFER (Gemma 3)
# Per-row zero-bug gates: proof lines (R5), chat smokes banked verbatim (R2),
# C1 PPL TWICE bitwise (R1 determinism), bench_e3.py speed pair (I1).
# 12B G4 rows carry an in-container transformers upgrade (gemma4_unified not in r9's
# transformers - tonight's ladder RED) -> LABELED dep-overlay rows, not r9-claim-grade.
set -uo pipefail
R=/home/jethac/spark_tmp/claude_retirement_scorecard_20260612
IMAGE=jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9
CTX=8191
S=$R/status.txt
T0=$(date +%s)

echo "RUN_START $(date -Is) IMAGE=${IMAGE} IMAGE_ID=$(docker images --format '{{.ID}}' ${IMAGE})" >> "$S"

# --- corpus gate (hard stop on mismatch, per Codex's 4ed0454 hardening) ---
declare -A WANT_MD5=(
  [c1_ppl_corpus.md]=abb63f0e65247a25f870d3f2d57563ff
  [c2_pride_prejudice_60k.txt]=1686a33b93ca17d1ecc6898d7d021781
  [c3_hijinks_code_60k.py]=28dfeba997756c52a74ee74854411c4b
)
for f in "${!WANT_MD5[@]}"; do
  got=$(md5sum "${R}/docs/${f}" | awk '{print $1}')
  echo "CORPUS ${f} MD5=${got} BYTES=$(wc -c < "${R}/docs/${f}")" >> "$S"
  if [ "${got}" != "${WANT_MD5[$f]}" ]; then
    echo "CORPUS_MD5_RED ${f} got=${got} want=${WANT_MD5[$f]} - HARD STOP" >> "$S"
    exit 9
  fi
done

# --- model cache preflight (names verbatim from tonight's served rows) ---
for m in models--google--gemma-4-31B-it models--google--gemma-4-E4B-it \
         models--google--gemma-4-12B-it models--google--gemma-4-26B-A4B-it \
         models--google--gemma-3-12b-it; do
  if [ -d "/home/jethac/.cache/huggingface/hub/${m}" ]; then
    echo "PREFLIGHT MODEL_CACHE ${m} PRESENT" >> "$S"
  else
    echo "PREFLIGHT MODEL_CACHE ${m} MISSING" >> "$S"
  fi
done

wait_ready() {
  local name=$1 iters=${2:-200}
  for _ in $(seq 1 "${iters}"); do
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
  grep -n "EXT_PATH\|VLLM_BUILD_CHECK\|attention backend\|AttentionBackendEnum\|TRITON\|Triton\|FLASHINFER\|FA2 VO split\|GPU KV cache size\|V-SF\|VLLM_NVFP4\|VLLM_FLASHINFER\|max_mma_kv\|kv_cache_dtype\|heterogeneous head\|TRANSFORMERS_VERSION\|Traceback\|ValueError\|ERROR" \
    "$R/results/claude_${label}_server.log" > "$R/results/claude_${label}_proof_lines.txt" 2>&1
  local kvline tri_sel fi_sel
  kvline=$(grep -o "GPU KV cache size: [0-9,]* tokens" "$R/results/claude_${label}_server.log" | head -1)
  tri_sel=$(grep -c "Using AttentionBackendEnum.TRITON_ATTN" "$R/results/claude_${label}_server.log")
  fi_sel=$(grep -c "Using AttentionBackendEnum.FLASHINFER" "$R/results/claude_${label}_server.log")
  echo "ROW=${label} KV_CAPACITY ${kvline:-NOT_FOUND}" >> "$S"
  echo "ROW=${label} ROUTE TRITON_SEL=${tri_sel} FI_SEL=${fi_sel}" >> "$S"
}

# run_ppl <name> <model> <served> <label> <kvlabel> <cell> <corpus-file>
run_ppl() {
  local name=$1 model=$2 served=$3 label=$4 kvlabel=$5 cell=$6 cfile=$7
  docker exec "${name}" python3 scripts/vllm_prompt_ppl_sweep.py \
    --url http://127.0.0.1:8000 \
    --model "${served}" \
    --tokenizer "${model}" \
    --text-file "docs/${cfile}" \
    --ctx "${CTX}" \
    --run-id "claude_rsc_${label}_${cell}_ctx${CTX}" \
    --kv-cache-dtype "${kvlabel}" \
    --runtime-ref "r9 image (baked); retirement scorecard row=${label} cell=${cell}" \
    --container-image "${IMAGE}" \
    --output "results/claude_${label}_${cell}_ctx${CTX}_ppl.json" \
    > "$R/results/claude_${label}_${cell}_ctx${CTX}_stdout.json" \
    2> "$R/results/claude_${label}_${cell}_ctx${CTX}_stderr.log"
  local rc=$?
  local got
  got=$(python3 -c "import json; print(repr(json.load(open('$R/results/claude_${label}_${cell}_ctx${CTX}_ppl.json'))['contexts'][0]['score']['mean_nll_nats']))" 2>/dev/null)
  echo "CELL=${label}_${cell} RC=${rc} mean=${got:-PARSE_FAIL}" >> "$S"
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

run_bench() {
  local name=$1 served=$2 label=$3
  docker exec "${name}" python3 scripts/bench_e3.py \
    --url http://127.0.0.1:8000 \
    --model "${served}" \
    --phase scorecard \
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

# run_row <model> <served> <label> <kvflag-or-NONE> <kvlabel> <backend-or-NONE> <pipfix 0|1> <bench 0|1> [extra docker -e args...]
run_row() {
  local model=$1 served=$2 label=$3 kvflag=$4 kvlabel=$5 backend=$6 pipfix=$7 bench=$8; shift 8
  local extra_docker=("$@")
  local name=claude_rsc_${label}
  local kvarg="" bearg="" pipcmd=":"
  if [ "${kvflag}" != "NONE" ]; then kvarg="--kv-cache-dtype ${kvflag}"; fi
  if [ "${backend}" != "NONE" ]; then bearg="--attention-backend ${backend}"; fi
  if [ "${pipfix}" = "1" ]; then
    pipcmd="python3 -m pip install -q --upgrade transformers 2>&1 | tail -3; python3 -c 'import transformers; print(\"TRANSFORMERS_VERSION\", transformers.__version__, flush=True)'"
  fi
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
      ${pipcmd}; \
      vllm serve ${model} \
      --served-model-name ${served} \
      --host 0.0.0.0 --port 8000 \
      ${kvarg} \
      ${bearg} \
      --gpu-memory-utilization 0.72 \
      --max-model-len 8192 \
      --language-model-only"

  wait_ready "${name}"; rc=$?
  echo "ROW=${label} READY_RC=${rc} READY_WALL=$(( $(date +%s) - t0 ))" >> "$S"
  extract_proof "${label}"
  if [ $rc -ne 0 ]; then
    echo "ROW=${label} SERVER_DID_NOT_BECOME_READY" >> "$S"
    grep -B2 -A80 "max_mma_kv\|EngineCore failed\|Traceback\|ValidationError" "$R/results/claude_${label}_server.log" \
      > "$R/results/claude_${label}_crash_excerpt.txt" 2>&1 || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
    return 2
  fi
  echo "ROW=${label} VERDICT=SERVED" >> "$S"

  run_smokes "${name}" "${served}" "${label}"

  run_ppl "${name}" "${model}" "${served}" "${label}" "${kvlabel}" c1a c1_ppl_corpus.md
  run_ppl "${name}" "${model}" "${served}" "${label}" "${kvlabel}" c1b c1_ppl_corpus.md
  local a b
  a=$(python3 -c "import json; print(repr(json.load(open('$R/results/claude_${label}_c1a_ctx${CTX}_ppl.json'))['contexts'][0]['score']['mean_nll_nats']))" 2>/dev/null)
  b=$(python3 -c "import json; print(repr(json.load(open('$R/results/claude_${label}_c1b_ctx${CTX}_ppl.json'))['contexts'][0]['score']['mean_nll_nats']))" 2>/dev/null)
  if [ -n "${a}" ] && [ "${a}" = "${b}" ]; then
    echo "ROW=${label} C1_DETERMINISM=IDENTICAL mean=${a}" >> "$S"
  else
    echo "ROW=${label} C1_DETERMINISM=DET_FAIL a=${a:-none} b=${b:-none}" >> "$S"
  fi

  if [ "${bench}" = "1" ]; then
    run_bench "${name}" "${served}" "${label}"
  fi

  extract_proof "${label}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  echo "ROW=${label} DONE ROW_WALL=$(( $(date +%s) - t0 ))" >> "$S"
}

# ============ Cell 1 (I3 + 31B pair): fresh Triton adjudication + FI bitwise repin ============
M=google/gemma-4-31B-it; SV=gemma4-31b-it
run_row "$M" "$SV" s31b_tri NONE bf16 NONE 0 1
echo "ROW=s31b_tri FINAL_RC=$?" >> "$S"
run_row "$M" "$SV" s31b_fi  NONE bf16 NONE 0 1 -e VLLM_FLASHINFER_VOSPLIT=1
echo "ROW=s31b_fi FINAL_RC=$?" >> "$S"

# ============ Cell 2 (E4B pair - known-risk row first) ============
M=google/gemma-4-E4B-it; SV=gemma4-e4b-it
run_row "$M" "$SV" e4b_fi  NONE bf16 NONE 0 1 -e VLLM_FLASHINFER_VOSPLIT=1
echo "ROW=e4b_fi FINAL_RC=$?" >> "$S"
run_row "$M" "$SV" e4b_tri NONE bf16 NONE 0 1
echo "ROW=e4b_tri FINAL_RC=$?" >> "$S"

# ============ Cell 3 (12B G4 pair - transformers dep-overlay, LABELED) ============
M=google/gemma-4-12B-it; SV=gemma4-12b-it
run_row "$M" "$SV" g412b_tri NONE bf16 NONE 1 1
echo "ROW=g412b_tri FINAL_RC=$?" >> "$S"
run_row "$M" "$SV" g412b_fi  NONE bf16 NONE 1 1 -e VLLM_FLASHINFER_VOSPLIT=1
echo "ROW=g412b_fi FINAL_RC=$?" >> "$S"

# ============ Cell 4 (26B pair) ============
M=google/gemma-4-26B-A4B-it; SV=gemma4-26b-a4b-it
run_row "$M" "$SV" g426b_tri NONE bf16 NONE 0 1
echo "ROW=g426b_tri FINAL_RC=$?" >> "$S"
run_row "$M" "$SV" g426b_fi  NONE bf16 NONE 0 1 -e VLLM_FLASHINFER_VOSPLIT=1
echo "ROW=g426b_fi FINAL_RC=$?" >> "$S"

# ============ Cell 5 (R4 scope spot-check: fp8 KV, NO knobs, route must match banked) ============
M=google/gemma-4-31B-it; SV=gemma4-31b-it
run_row "$M" "$SV" s31b_fp8 fp8_e4m3 fp8_e4m3 NONE 0 0
echo "ROW=s31b_fp8 FINAL_RC=$?" >> "$S"

# ============ Cell 6 (stretch: Gemma 3 12B pair - default route vs forced FLASHINFER) ============
M=google/gemma-3-12b-it; SV=gemma3-12b-it
run_row "$M" "$SV" g312b_def NONE bf16 NONE 0 1
echo "ROW=g312b_def FINAL_RC=$?" >> "$S"
run_row "$M" "$SV" g312b_fi  NONE bf16 FLASHINFER 0 1
echo "ROW=g312b_fi FINAL_RC=$?" >> "$S"

echo "SCORECARD_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
