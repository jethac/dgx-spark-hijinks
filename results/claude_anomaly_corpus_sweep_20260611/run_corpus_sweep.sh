#!/usr/bin/env bash
# Anomaly corpus-sweep window (docs/WINDOW_PACKET_ANOMALY_CORPUS_SWEEP.md, epoch2).
# r9 image (fix BAKED, no overlay). google/gemma-4-31B-it --language-model-only,
# util 0.72, ctx-8191 prompt-PPL sweeps. THREE servers sequential x THREE corpora:
#   row bf16_vosplit : no --kv-cache-dtype, VLLM_FLASHINFER_VOSPLIT=1
#   row fp8          : --kv-cache-dtype fp8_e4m3, NO knob envs (Triton expected)
#   row nvfp4        : --kv-cache-dtype nvfp4, VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1
# Corpora (each scored in this order per server): C1 standing (abb63f0e),
# C2 natural prose (Pride & Prejudice 60KB slice), C3 code (hijinks scripts/*.py 60KB).
# Determinism cross-check: C1 must reproduce bf16 4.613162683323541 /
# fp8 4.473945385741097 / nvfp4 4.2813347779571975 bitwise.
set -uo pipefail
R=/home/jethac/spark_tmp/claude_anomaly_corpus_sweep_20260611
IMAGE=jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9
CTX=8191
MODEL=google/gemma-4-31B-it
SERVED=gemma4-31b-it
S=$R/status.txt
T0=$(date +%s)

echo "RUN_START $(date -Is) IMAGE=${IMAGE} IMAGE_ID=$(docker images --format '{{.ID}}' ${IMAGE})" >> "$S"
for f in c1_ppl_corpus.md c2_pride_prejudice_60k.txt c3_hijinks_code_60k.py; do
  echo "CORPUS ${f} MD5=$(md5sum ${R}/docs/${f} | awk '{print $1}') BYTES=$(wc -c < ${R}/docs/${f})" >> "$S"
done

wait_ready() {
  local name=$1
  for _ in $(seq 1 180); do
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
  grep -n "EXT_PATH\|VLLM_BUILD_CHECK\|Unknown vLLM environment variable\|attention backend\|TRITON\|Triton\|FLASHINFER\|FlashInfer FA2 backend\|FA2 VO split\|GPU KV cache size\|heterogeneous head\|V-scale-factor\|V-SF\|VLLM_NVFP4\|VLLM_FLASHINFER\|max_mma_kv\|kv_cache_dtype\|language.model.only\|Traceback\|ValueError\|ERROR" \
    "$R/results/claude_${label}_server.log" > "$R/results/claude_${label}_proof_lines.txt" 2>&1
  echo "MAX_MMA_KV_COUNT=$(grep -c 'max_mma_kv' "$R/results/claude_${label}_server.log")" >> "$R/results/claude_${label}_proof_lines.txt"
}

# run_ppl <name> <row-label> <kvlabel> <corpus-tag> <corpus-file>
run_ppl() {
  local name=$1 label=$2 kvlabel=$3 ctag=$4 cfile=$5
  docker exec "${name}" python3 scripts/vllm_prompt_ppl_sweep.py \
    --url http://127.0.0.1:8000 \
    --model "${SERVED}" \
    --tokenizer "${MODEL}" \
    --text-file "docs/${cfile}" \
    --ctx "${CTX}" \
    --run-id "claude_${label}_${ctag}_ctx${CTX}" \
    --kv-cache-dtype "${kvlabel}" \
    --runtime-ref "r9 image (fix BAKED, no overlay) ${IMAGE}; row=${label}; corpus=${ctag}; anomaly corpus-sweep window" \
    --container-image "${IMAGE}" \
    --output "results/claude_${label}_${ctag}_ctx${CTX}_ppl.json" \
    > "$R/results/claude_${label}_${ctag}_ctx${CTX}_stdout.json" \
    2> "$R/results/claude_${label}_${ctag}_ctx${CTX}_stderr.log"
}

# run_row <label> <kvflag-or-NONE> <kvlabel> [extra docker -e args...]
run_row() {
  local label=$1 kvflag=$2 kvlabel=$3; shift 3
  local extra_docker=("$@")
  local name=claude_acs_${label}
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
      python3 -c 'import importlib.util as iu; s=iu.find_spec(\"vllm.v1.attention.backends.flashinfer\"); print(\"VLLM_BUILD_CHECK flashinfer=\", s.origin, flush=True)'; \
      python3 -c 'import vllm._C_stable_libtorch as m; print(\"EXT_PATH\", m.__file__, flush=True)'; \
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
    grep -B2 -A80 "max_mma_kv\|EngineCore failed\|Traceback" "$R/results/claude_${label}_server.log" \
      > "$R/results/claude_${label}_crash_excerpt.txt" 2>&1 || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
    return 2
  fi
  echo "ROW=${label} VERDICT=SERVED" >> "$S"

  run_ppl "${name}" "${label}" "${kvlabel}" c1 c1_ppl_corpus.md
  echo "ROW=${label} C1_PPL_RC=$? WALL=$(( $(date +%s) - t0 ))" >> "$S"
  run_ppl "${name}" "${label}" "${kvlabel}" c2 c2_pride_prejudice_60k.txt
  echo "ROW=${label} C2_PPL_RC=$? WALL=$(( $(date +%s) - t0 ))" >> "$S"
  run_ppl "${name}" "${label}" "${kvlabel}" c3 c3_hijinks_code_60k.py
  echo "ROW=${label} C3_PPL_RC=$? WALL=$(( $(date +%s) - t0 ))" >> "$S"

  extract_proof "${label}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  echo "ROW=${label} DONE ROW_WALL=$(( $(date +%s) - t0 ))" >> "$S"
}

run_row bf16_vosplit NONE bf16 -e VLLM_FLASHINFER_VOSPLIT=1
echo "ROW=bf16_vosplit FINAL_RC=$?" >> "$S"

run_row fp8 fp8_e4m3 fp8_e4m3
echo "ROW=fp8 FINAL_RC=$?" >> "$S"

run_row nvfp4 nvfp4 nvfp4 -e VLLM_NVFP4_KV_VOSPLIT=1 -e VLLM_NVFP4_KV_LINEAR_V_SF=1
echo "ROW=nvfp4 FINAL_RC=$?" >> "$S"

echo "SWEEP_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
