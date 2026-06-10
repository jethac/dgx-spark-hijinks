#!/usr/bin/env bash
set -uo pipefail
R=/home/jethac/spark_tmp/claude_r8_gates_results/part2
IMAGE=jethac-vllm-aeon-gemma4:e08a6f3ae-rebuiltc-fb7d62ea-sm121a-r8
MODEL=/home/jethac/models/aeon/qwen36-nvfp4
CTX=8192

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

run_one() {
  local label=$1; shift
  local name=claude_r8_p2_${label}
  local extra_env=("$@")
  local t0=$(date +%s)
  docker run -d --rm --name "${name}" --gpus all --net host --ipc host \
    --memory 100g --memory-swap 100g \
    -e VLLM_TEST_FORCE_FP8_MARLIN=1 "${extra_env[@]}" \
    -v "${MODEL}:/models/target:ro" \
    -v "${R}:/work" -w /work \
    "${IMAGE}" \
    bash -lc "vllm serve /models/target \
      --served-model-name qwen36-35b-heretic qwen36-fast qwen36-deep \
      --host 0.0.0.0 --port 8000 \
      --trust-remote-code --max-model-len 262144 \
      --quantization compressed-tensors --load-format safetensors \
      --attention-backend flashinfer --kv-cache-dtype nvfp4 \
      --gpu-memory-utilization 0.72 --max-num-batched-tokens 65536 --max-num-seqs 128 \
      --enable-chunked-prefill --no-enable-prefix-caching \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
      > /work/results/claude_p2_${label}_server.log 2>&1"
  wait_ready "${name}"; rc=$?
  echo "PHASE=${label} READY_RC=${rc} READY_WALL=$(( $(date +%s) - t0 ))" >> $R/part2_status.txt
  if [ $rc -ne 0 ]; then docker rm -f "${name}" >/dev/null 2>&1 || true; return 2; fi
  docker exec "${name}" python3 scripts/vllm_prompt_ppl_sweep.py \
    --url http://127.0.0.1:8000 \
    --model qwen36-fast \
    --tokenizer /models/target \
    --text-file docs/ppl_corpus.md \
    --ctx "${CTX}" \
    --run-id "claude_r8_p2_${label}_ctx${CTX}" \
    --kv-cache-dtype nvfp4 \
    --runtime-ref "jethac/vllm@e08a6f3ae + jethac/flashinfer@fb7d62ea rebuilt-C r8 image (clean FlashInfer JIT cache); VLLM_TEST_FORCE_FP8_MARLIN=1; VLLM_NVFP4_KV_LINEAR_V_SF=${label}" \
    --container-image "${IMAGE}" \
    --output "results/claude_p2_${label}_ctx${CTX}_ppl.json" \
    > "$R/results/claude_p2_${label}_ctx${CTX}_stdout.json" \
    2> "$R/results/claude_p2_${label}_ctx${CTX}_stderr.log"
  echo "PHASE=${label} PPL_RC=$? TOTAL_WALL=$(( $(date +%s) - t0 ))" >> $R/part2_status.txt
  docker rm -f "${name}" >/dev/null 2>&1 || true
}

# offfix rerun: off only
run_one off
echo "PART2_OFFFIX_DONE" >> $R/part2_status.txt
