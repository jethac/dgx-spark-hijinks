#!/usr/bin/env bash
# ADDENDUM (deviation, diagnostic): one extra bf16 server start to score C2 with
# --add-special-tokens (BOS). Tests whether C2's high absolute PPL under the
# standing no-BOS harness convention is a Gemma BOS-sensitivity artifact, i.e.
# answers the packet question "is C2 low-PPL natural text as intended?".
# Also rescoring C2 no-BOS in the same server as a within-window repro check.
set -uo pipefail
R=/home/jethac/spark_tmp/claude_anomaly_corpus_sweep_20260611
IMAGE=jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9
CTX=8191
MODEL=google/gemma-4-31B-it
SERVED=gemma4-31b-it
S=$R/status.txt
T0=$(date +%s)
name=claude_acs_bf16_diag

docker run -d --rm --name "${name}" --gpus all --net host --ipc host \
  --memory 100g --memory-swap 100g \
  -w /work \
  -e VLLM_FLASHINFER_VOSPLIT=1 \
  -v /home/jethac/.cache/huggingface:/root/.cache/huggingface \
  -v "${R}:/work" \
  "${IMAGE}" \
  bash -lc "exec > /work/results/claude_bf16_diag_server.log 2>&1; \
    python3 -c 'import vllm._C_stable_libtorch as m; print(\"EXT_PATH\", m.__file__, flush=True)'; \
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
echo "DIAG READY_RC=${rc} READY_WALL=$(( $(date +%s) - T0 ))" >> "$S"
if [ $rc -ne 0 ]; then docker rm -f "${name}" >/dev/null 2>&1 || true; exit 2; fi

ppl() {
  local tag=$1; shift
  docker exec "${name}" python3 scripts/vllm_prompt_ppl_sweep.py \
    --url http://127.0.0.1:8000 --model "${SERVED}" --tokenizer "${MODEL}" \
    --text-file docs/c2_pride_prejudice_60k.txt --ctx "${CTX}" \
    --run-id "claude_bf16_diag_${tag}_ctx${CTX}" --kv-cache-dtype bf16 \
    --runtime-ref "r9 baked; DIAG row bf16 VOSPLIT; ${tag}" \
    --container-image "${IMAGE}" \
    --output "results/claude_bf16_diag_${tag}_ctx${CTX}_ppl.json" "$@" \
    > "$R/results/claude_bf16_diag_${tag}_ctx${CTX}_stdout.json" \
    2> "$R/results/claude_bf16_diag_${tag}_ctx${CTX}_stderr.log"
  echo "DIAG ${tag} RC=$? WALL=$(( $(date +%s) - T0 ))" >> "$S"
}

ppl c2_nobos_repro
ppl c2_bos --add-special-tokens

docker rm -f "${name}" >/dev/null 2>&1 || true
echo "DIAG_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
