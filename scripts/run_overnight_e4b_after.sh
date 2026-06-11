#!/usr/bin/env bash
# E4B speed AFTER row (overnight ladder plan): nvfp4 KV + VO-split knobs on
# FlashInfer, benchmarked with the SAME harness/params as the Triton baseline
# in results/claude_blockE23_20260611 (19.03 tok/s decode, 0.317s TTFT,
# x4 aggregate 92.04 tok/s; medians of 3, temp 0, nonce-prefixed prompts).
set -uo pipefail
R=/home/jethac/spark_tmp/claude_overnight_ladder_20260612
IMAGE=jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9
MODEL=google/gemma-4-E4B-it
SERVED=gemma4-e4b-it
S=$R/status.txt
name=claude_lad_e4bafter
t0=$(date +%s)

docker run -d --rm --name "${name}" --gpus all --net host --ipc host \
  --memory 100g --memory-swap 100g \
  -w /work \
  -e VLLM_NVFP4_KV_VOSPLIT=1 -e VLLM_NVFP4_KV_LINEAR_V_SF=1 \
  -v /home/jethac/.cache/huggingface:/root/.cache/huggingface \
  -v "${R}:/work" \
  "${IMAGE}" \
  bash -lc "exec > /work/results/claude_e4bafter_server.log 2>&1; \
    python3 -c 'import vllm; print(\"VLLM_BUILD_CHECK vllm.__file__=\", vllm.__file__, flush=True)'; \
    python3 -c 'import vllm._C_stable_libtorch as m; print(\"EXT_PATH\", m.__file__, flush=True)'; \
    vllm serve ${MODEL} \
    --served-model-name ${SERVED} \
    --host 0.0.0.0 --port 8000 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.72 \
    --max-model-len 8192 \
    --language-model-only"

for _ in $(seq 1 220); do
  if docker exec -i "${name}" python3 -c "import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:8000/v1/models\", timeout=2).read()" >/dev/null 2>&1; then rc=0; break; fi
  if ! docker ps --format "{{.Names}}" | grep -q "^${name}$"; then rc=3; break; fi
  rc=1; sleep 5
done
echo "ROW=e4bafter READY_RC=${rc} READY_WALL=$(( $(date +%s) - t0 ))" >> "$S"
grep -n "EXT_PATH\|VLLM_BUILD_CHECK\|attention backend\|TRITON\|Triton\|FLASHINFER\|FA2 VO split\|GPU KV cache size\|V-SF\|VLLM_NVFP4\|max_mma_kv\|kv_cache_dtype\|Traceback\|ValueError\|ERROR" \
  "$R/results/claude_e4bafter_server.log" > "$R/results/claude_e4bafter_proof_lines.txt" 2>&1
if [ "${rc}" -ne 0 ]; then
  echo "ROW=e4bafter SERVER_DID_NOT_BECOME_READY" >> "$S"
  grep -B2 -A80 "max_mma_kv\|EngineCore failed\|Traceback" "$R/results/claude_e4bafter_server.log" \
    > "$R/results/claude_e4bafter_crash_excerpt.txt" 2>&1 || true
  docker rm -f "${name}" >/dev/null 2>&1 || true
  echo "ROW=e4bafter FINAL_RC=2" >> "$S"
  exit 2
fi
echo "ROW=e4bafter VERDICT=SERVED" >> "$S"

docker exec "${name}" python3 scripts/openai_chat_smoke.py \
  --model "${SERVED}" --max-tokens 16 \
  --output results/claude_e4bafter_smoke_sparkok.json > /dev/null 2>&1
echo "ROW=e4bafter SMOKE_SPARKOK_RC=$?" >> "$S"
docker exec "${name}" python3 scripts/openai_chat_smoke.py \
  --model "${SERVED}" --max-tokens 24 --prompt "The capital of Japan is" \
  --output results/claude_e4bafter_smoke_tokyo.json > /dev/null 2>&1
if grep -qi "tokyo" "$R/results/claude_e4bafter_smoke_tokyo.json"; then
  echo "ROW=e4bafter SMOKE_TOKYO=COHERENT" >> "$S"
else
  echo "ROW=e4bafter SMOKE_TOKYO=SUSPECT" >> "$S"
fi

docker exec "${name}" python3 scripts/bench_e3.py \
  --model "${SERVED}" --phase after \
  --run-id claude_e4bafter_nvfp4_vosplit \
  --reps 3 \
  --output results/claude_e4bafter_benchmark.json \
  > "$R/results/claude_e4bafter_benchmark.stdout" \
  2> "$R/results/claude_e4bafter_benchmark.stderr"
echo "ROW=e4bafter BENCH_RC=$?" >> "$S"

docker rm -f "${name}" >/dev/null 2>&1 || true
echo "ROW=e4bafter DONE ROW_WALL=$(( $(date +%s) - t0 ))" >> "$S"
