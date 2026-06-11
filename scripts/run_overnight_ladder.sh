#!/usr/bin/env bash
# Overnight Spark vLLM ladder block (docs/OVERNIGHT_LADDER_PLAN_20260612.md,
# zero-bug amendments). Three models x three KV rows, sequential, ONE server
# at a time, r9 baked image, util 0.72, ctx-8191 prompt PPL with token dumps.
#
# Per-row zero-bug gates:
#   1. proof lines (backend, kv_cache_dtype, KV-token capacity, EXT_PATH, Triton mentions)
#   2. chat smoke x2 banked verbatim (spark-ok echo + "The capital of Japan is")
#   3. C1 PPL run TWICE -> mean_nll_nats bitwise identical or DET_FAIL (row RED)
#   4. C2/C3 once each
# Knobs: bf16 vosplit + nvfp4 vosplit knobs ONLY on D512 models (Gemma 4);
# gemma-3-12b is uniform-256 (config recon banked in results/preflight_*).
set -uo pipefail
R=/home/jethac/spark_tmp/claude_overnight_ladder_20260612
IMAGE=jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9
CTX=8191
S=$R/status.txt
T0=$(date +%s)

mkdir -p "$R/results/token_dumps"
echo "RUN_START $(date -Is) IMAGE=${IMAGE} IMAGE_ID=$(docker images --format '{{.ID}}' ${IMAGE})" >> "$S"
for f in c1_ppl_corpus.md c2_pride_prejudice_60k.txt c3_hijinks_code_60k.py; do
  echo "CORPUS ${f} MD5=$(md5sum ${R}/docs/${f} | awk '{print $1}') BYTES=$(wc -c < ${R}/docs/${f})" >> "$S"
done

wait_ready() {
  local name=$1
  for _ in $(seq 1 220); do
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

# run_ppl <name> <model> <served> <label> <kvlabel> <cell> <corpus-file>
run_ppl() {
  local name=$1 model=$2 served=$3 label=$4 kvlabel=$5 cell=$6 cfile=$7
  docker exec "${name}" python3 scripts/vllm_prompt_ppl_sweep.py \
    --url http://127.0.0.1:8000 \
    --model "${served}" \
    --tokenizer "${model}" \
    --text-file "docs/${cfile}" \
    --ctx "${CTX}" \
    --run-id "claude_${label}_${cell}_ctx${CTX}" \
    --kv-cache-dtype "${kvlabel}" \
    --runtime-ref "r9 image (baked, no overlay); overnight ladder row=${label} cell=${cell}" \
    --container-image "${IMAGE}" \
    --dump-token-logprobs results/token_dumps \
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

# run_row <model> <served> <label> <kvflag-or-NONE> <kvlabel> [extra docker -e args...]
run_row() {
  local model=$1 served=$2 label=$3 kvflag=$4 kvlabel=$5; shift 5
  local extra_docker=("$@")
  local name=claude_lad_${label}
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
      --max-model-len 8192 \
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
  run_ppl "${name}" "${model}" "${served}" "${label}" "${kvlabel}" c2 c2_pride_prejudice_60k.txt
  run_ppl "${name}" "${model}" "${served}" "${label}" "${kvlabel}" c3 c3_hijinks_code_60k.py

  extract_proof "${label}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  echo "ROW=${label} DONE ROW_WALL=$(( $(date +%s) - t0 ))" >> "$S"
}

# ---- Model 1: gemma-3-12b-it (uniform head 256 -> NO vosplit knobs anywhere)
M=google/gemma-3-12b-it; SV=gemma3-12b-it
run_row "$M" "$SV" g312b_bf16  NONE     bf16
echo "ROW=g312b_bf16 FINAL_RC=$?" >> "$S"
run_row "$M" "$SV" g312b_nvfp4 nvfp4    nvfp4 -e VLLM_NVFP4_KV_LINEAR_V_SF=1
echo "ROW=g312b_nvfp4 FINAL_RC=$?" >> "$S"
run_row "$M" "$SV" g312b_fp8   fp8_e4m3 fp8_e4m3
echo "ROW=g312b_fp8 FINAL_RC=$?" >> "$S"

# ---- Model 2: gemma-4-12B-it (D512 globals -> vosplit knobs)
M=google/gemma-4-12B-it; SV=gemma4-12b-it
run_row "$M" "$SV" g412b_bf16  NONE     bf16  -e VLLM_FLASHINFER_VOSPLIT=1
echo "ROW=g412b_bf16 FINAL_RC=$?" >> "$S"
run_row "$M" "$SV" g412b_nvfp4 nvfp4    nvfp4 -e VLLM_NVFP4_KV_VOSPLIT=1 -e VLLM_NVFP4_KV_LINEAR_V_SF=1
echo "ROW=g412b_nvfp4 FINAL_RC=$?" >> "$S"
run_row "$M" "$SV" g412b_fp8   fp8_e4m3 fp8_e4m3
echo "ROW=g412b_fp8 FINAL_RC=$?" >> "$S"

# ---- Model 3: gemma-4-26B-A4B-it (MoE, D512 globals -> vosplit knobs)
M=google/gemma-4-26B-A4B-it; SV=gemma4-26b-a4b-it
run_row "$M" "$SV" g426b_bf16  NONE     bf16  -e VLLM_FLASHINFER_VOSPLIT=1
echo "ROW=g426b_bf16 FINAL_RC=$?" >> "$S"
run_row "$M" "$SV" g426b_nvfp4 nvfp4    nvfp4 -e VLLM_NVFP4_KV_VOSPLIT=1 -e VLLM_NVFP4_KV_LINEAR_V_SF=1
echo "ROW=g426b_nvfp4 FINAL_RC=$?" >> "$S"
run_row "$M" "$SV" g426b_fp8   fp8_e4m3 fp8_e4m3
echo "ROW=g426b_fp8 FINAL_RC=$?" >> "$S"

echo "LADDER_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
