#!/usr/bin/env bash
# ADDENDUM 2 (diagnostic, clearly out-of-packet): bf16 server, score all three
# corpora with a literal "<bos>" prefix (tokenizer parses it to id 2; the
# gemma-4 tokenizer has add_bos_token=False so --add-special-tokens is a no-op
# and the standing harness convention is BOS-less). Purpose: absolute-PPL
# context — is C2 low-PPL natural text once BOS-anchored?
set -uo pipefail
R=/home/jethac/spark_tmp/claude_anomaly_corpus_sweep_20260611
IMAGE=jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9
CTX=8191
MODEL=google/gemma-4-31B-it
SERVED=gemma4-31b-it
S=$R/status.txt
T0=$(date +%s)
name=claude_acs_bf16_diag2

docker run -d --rm --name "${name}" --gpus all --net host --ipc host \
  --memory 100g --memory-swap 100g \
  -w /work \
  -e VLLM_FLASHINFER_VOSPLIT=1 \
  -v /home/jethac/.cache/huggingface:/root/.cache/huggingface \
  -v "${R}:/work" \
  "${IMAGE}" \
  bash -lc "exec > /work/results/claude_bf16_diag2_server.log 2>&1; \
    vllm serve ${MODEL} \
    --served-model-name ${SERVED} \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.72 \
    --max-model-len 8192 \
    --language-model-only"

rc=1
for _ in $(seq 1 180); do
  if docker exec -i "${name}" python3 -c "import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:8000/v1/models\", timeout=2).read()" >/dev/null 2>&1; then rc=0; break; fi
  if ! docker ps --format "{{.Names}}" | grep -q "^${name}$"; then rc=3; break; fi
  sleep 5
done
echo "DIAG2 READY_RC=${rc} READY_WALL=$(( $(date +%s) - T0 ))" >> "$S"
if [ $rc -ne 0 ]; then docker rm -f "${name}" >/dev/null 2>&1 || true; exit 2; fi

for tag in c1 c2 c3; do
  docker exec "${name}" python3 scripts/vllm_prompt_ppl_sweep.py \
    --url http://127.0.0.1:8000 --model "${SERVED}" --tokenizer "${MODEL}" \
    --text-file "docs/${tag}_bosprefixed.txt" --ctx "${CTX}" \
    --run-id "claude_bf16_diag2_${tag}_bos_ctx${CTX}" --kv-cache-dtype bf16 \
    --runtime-ref "r9 baked; DIAG2 bf16 VOSPLIT; literal <bos> prefix; corpus=${tag}" \
    --container-image "${IMAGE}" \
    --output "results/claude_bf16_diag2_${tag}_bos_ctx${CTX}_ppl.json" \
    > "$R/results/claude_bf16_diag2_${tag}_bos_ctx${CTX}_stdout.json" \
    2> "$R/results/claude_bf16_diag2_${tag}_bos_ctx${CTX}_stderr.log"
  echo "DIAG2 ${tag}_bos RC=$? WALL=$(( $(date +%s) - T0 ))" >> "$S"
done

docker rm -f "${name}" >/dev/null 2>&1 || true
echo "DIAG2_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
