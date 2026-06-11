#!/usr/bin/env bash
# PART A: order-controlled anomaly adjudication (2026-06-12 morning window).
# Question: does "quantized KV beats bf16" survive request-order control?
# Known going in: fp8 31B C1 = 4.473945385741097 scored-cold (bitwise) vs
# 4.591455999476844 after two chat smokes (bitwise). nvfp4 + bf16 order
# sensitivity untested; C2/C3 untested for all rows.
#
# Matrix on google/gemma-4-31B-it (--language-model-only, r9 baked, util 0.72,
# ctx 8191), per row in {bf16+VOSPLIT, fp8 (no knobs), nvfp4+VOSPLIT+LINEAR_V_SF}:
#   COLD cycle  : fresh server -> C1 x2 (bitwise gate) -> C2 -> C3
#                 -> smokes AFTER scoring (labeled post-score, coherence only).
#                 Expect bitwise reproduction of the banked corpus-sweep values
#                 (results/claude_anomaly_corpus_sweep_20260611, which scored
#                 C1,C2,C3 first with zero smokes):
#                   bf16  4.613162683323541 / 6.000594550413712 / 3.024778946389832
#                   fp8   4.473945385741097 / 5.83629865522355  / 3.006330528749716
#                   nvfp4 4.2813347779571975 / 6.253168933373023 / 2.977997659632257
#   WARMED cycle: fresh server -> EXACTLY the scorecard's two chat smokes
#                 (openai_chat_smoke.py defaults --max-tokens 16, then
#                  --prompt "The capital of Japan is" --max-tokens 24)
#                 -> C1 x2 (bitwise) -> C2 -> C3.
# Row order: nvfp4 (untested, highest info), fp8 (anchor replication), bf16
# (4 cross-window bitwise repins exist; expected stable).
set -uo pipefail
R=/home/jethac/spark_tmp/claude_order_control_20260612
IMAGE=jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9
CTX=8191
MODEL=google/gemma-4-31B-it
SERVED=gemma4-31b-it
S=$R/status.txt
T0=$(date +%s)

echo "RUN_START $(date -Is) IMAGE=${IMAGE} IMAGE_ID=$(docker images --format '{{.ID}}' ${IMAGE})" >> "$S"

# --- corpus gate (hard stop on mismatch) ---
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
for sc in vllm_prompt_ppl_sweep.py openai_chat_smoke.py; do
  echo "SCRIPT ${sc} MD5=$(md5sum ${R}/scripts/${sc} | awk '{print $1}')" >> "$S"
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
  grep -n "EXT_PATH\|VLLM_BUILD_CHECK\|attention backend\|AttentionBackendEnum\|TRITON\|Triton\|FLASHINFER\|FA2 VO split\|GPU KV cache size\|V-SF\|VLLM_NVFP4\|VLLM_FLASHINFER\|max_mma_kv\|kv_cache_dtype\|heterogeneous head\|Traceback\|ValueError\|ERROR" \
    "$R/results/claude_${label}_server.log" > "$R/results/claude_${label}_proof_lines.txt" 2>&1
  local kvline tri_sel fi_sel
  kvline=$(grep -o "GPU KV cache size: [0-9,]* tokens" "$R/results/claude_${label}_server.log" | head -1)
  tri_sel=$(grep -c "Using AttentionBackendEnum.TRITON_ATTN" "$R/results/claude_${label}_server.log")
  fi_sel=$(grep -c "Using AttentionBackendEnum.FLASHINFER" "$R/results/claude_${label}_server.log")
  echo "ROW=${label} KV_CAPACITY ${kvline:-NOT_FOUND}" >> "$S"
  echo "ROW=${label} ROUTE TRITON_SEL=${tri_sel} FI_SEL=${fi_sel}" >> "$S"
}

# run_ppl <name> <label> <kvlabel> <cell> <corpus-file>
run_ppl() {
  local name=$1 label=$2 kvlabel=$3 cell=$4 cfile=$5
  docker exec "${name}" python3 scripts/vllm_prompt_ppl_sweep.py \
    --url http://127.0.0.1:8000 \
    --model "${SERVED}" \
    --tokenizer "${MODEL}" \
    --text-file "docs/${cfile}" \
    --ctx "${CTX}" \
    --run-id "claude_oc_${label}_${cell}_ctx${CTX}" \
    --kv-cache-dtype "${kvlabel}" \
    --runtime-ref "r9 image (baked); order-control row=${label} cell=${cell}" \
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

# run_smokes <name> <label> -- EXACT scorecard smoke pair (prompts + params)
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

# run_cycle <label> <mode cold|warmed> <kvflag-or-NONE> <kvlabel> [extra docker -e args...]
run_cycle() {
  local label=$1 mode=$2 kvflag=$3 kvlabel=$4; shift 4
  local extra_docker=("$@")
  local name=claude_oc_${label}
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
      vllm serve ${MODEL} \
      --served-model-name ${SERVED} \
      --host 0.0.0.0 --port 8000 \
      ${kvarg} \
      --gpu-memory-utilization 0.72 \
      --max-model-len 8192 \
      --language-model-only"

  wait_ready "${name}"; rc=$?
  echo "ROW=${label} MODE=${mode} READY_RC=${rc} READY_WALL=$(( $(date +%s) - t0 ))" >> "$S"
  extract_proof "${label}"
  if [ $rc -ne 0 ]; then
    echo "ROW=${label} SERVER_DID_NOT_BECOME_READY" >> "$S"
    grep -B2 -A80 "max_mma_kv\|EngineCore failed\|Traceback\|InductorError" "$R/results/claude_${label}_server.log" \
      > "$R/results/claude_${label}_crash_excerpt.txt" 2>&1 || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
    return 2
  fi
  echo "ROW=${label} VERDICT=SERVED" >> "$S"

  if [ "${mode}" = "warmed" ]; then
    echo "ROW=${label} ORDER=SMOKES_THEN_SCORE" >> "$S"
    run_smokes "${name}" "${label}"
  else
    echo "ROW=${label} ORDER=SCORE_FIRST_COLD" >> "$S"
  fi

  run_ppl "${name}" "${label}" "${kvlabel}" c1a c1_ppl_corpus.md
  run_ppl "${name}" "${label}" "${kvlabel}" c1b c1_ppl_corpus.md
  det_check "${label}"
  run_ppl "${name}" "${label}" "${kvlabel}" c2 c2_pride_prejudice_60k.txt
  run_ppl "${name}" "${label}" "${kvlabel}" c3 c3_hijinks_code_60k.py

  if [ "${mode}" = "cold" ]; then
    echo "ROW=${label} POST_SCORE_SMOKES (coherence record only, cannot affect scores)" >> "$S"
    run_smokes "${name}" "${label}"
  fi

  extract_proof "${label}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  echo "ROW=${label} DONE ROW_WALL=$(( $(date +%s) - t0 ))" >> "$S"
}

# ---- nvfp4 first (order-sensitivity UNTESTED -> highest information) ----
run_cycle nvfp4_cold cold  nvfp4 nvfp4 -e VLLM_NVFP4_KV_VOSPLIT=1 -e VLLM_NVFP4_KV_LINEAR_V_SF=1
echo "ROW=nvfp4_cold FINAL_RC=$?" >> "$S"
run_cycle nvfp4_warm warmed nvfp4 nvfp4 -e VLLM_NVFP4_KV_VOSPLIT=1 -e VLLM_NVFP4_KV_LINEAR_V_SF=1
echo "ROW=nvfp4_warm FINAL_RC=$?" >> "$S"

# ---- fp8 (anchor replication of the known effect + NEW C2/C3 cells) ----
run_cycle fp8_cold cold  fp8_e4m3 fp8_e4m3
echo "ROW=fp8_cold FINAL_RC=$?" >> "$S"
run_cycle fp8_warm warmed fp8_e4m3 fp8_e4m3
echo "ROW=fp8_warm FINAL_RC=$?" >> "$S"

# ---- bf16 control (expected order-stable; 4 bitwise repins exist) ----
run_cycle bf16_cold cold  NONE bf16 -e VLLM_FLASHINFER_VOSPLIT=1
echo "ROW=bf16_cold FINAL_RC=$?" >> "$S"
run_cycle bf16_warm warmed NONE bf16 -e VLLM_FLASHINFER_VOSPLIT=1
echo "ROW=bf16_warm FINAL_RC=$?" >> "$S"

echo "ORDER_CONTROL_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
