#!/usr/bin/env bash
set -uo pipefail
R=/home/jethac/spark_tmp/claude_r8_gates_results/part4
IMAGE=jethac-vllm-aeon-gemma4:e08a6f3ae-rebuiltc-fb7d62ea-sm121a-r8
NAME=claude_r8_p4_31b_fp8
T0=$(date +%s)
echo "PART4_START $(date -Is)" >> $R/part4_status.txt

docker run -d --rm --name "${NAME}" --gpus all --ipc=host --net host \
  --memory 100g --memory-swap 100g \
  -w /work \
  -e CLAUDE_DISABLE_GEMMA4_MM_PREFIX=1 \
  -v ${R}/model_arch_config_convertor_patched.py:/opt/jethac-vllm/vllm/transformers_utils/model_arch_config_convertor.py:ro \
  -v /home/jethac/.cache/huggingface:/root/.cache/huggingface \
  -v ${R}:/work \
  "${IMAGE}" \
  bash -lc "vllm serve google/gemma-4-31B-it \
    --served-model-name gemma4-31b-it \
    --host 0.0.0.0 --port 8000 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.72 \
    --max-model-len 8192 \
    > /work/results/claude_p4_server.log 2>&1"

ready=1
for i in $(seq 1 180); do
  if docker exec -i "${NAME}" python3 -c "import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:8000/v1/models\", timeout=2).read()" >/dev/null 2>&1; then
    ready=0; break
  fi
  if ! docker ps --format "{{.Names}}" | grep -q "^${NAME}\$"; then ready=3; break; fi
  sleep 5
done
echo "READY_RC=${ready} READY_WALL=$(( $(date +%s) - T0 ))" >> $R/part4_status.txt
if [ $ready -ne 0 ]; then
  echo "SERVER_DID_NOT_BECOME_READY" >> $R/part4_status.txt
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  exit 2
fi

docker exec -i "${NAME}" python3 - << "PY" > $R/results/claude_p4_sanity_chat.json 2> $R/results/claude_p4_sanity_chat.stderr.log
import json, time, urllib.request
payload = {"model": "gemma4-31b-it",
           "messages": [{"role": "user", "content": "In one sentence, what is the NVIDIA GB10 Grace Blackwell superchip?"}],
           "temperature": 0, "max_tokens": 128}
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.monotonic()
resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
wall = time.monotonic() - t0
out = {"schema": "claude-r8-gates-part4-sanity/v1", "wall_s": wall,
       "text": resp["choices"][0]["message"]["content"],
       "usage": resp.get("usage"),
       "finish_reason": resp["choices"][0].get("finish_reason")}
print(json.dumps(out, indent=2))
PY
echo "SANITY_RC=$? SANITY_WALL=$(( $(date +%s) - T0 ))" >> $R/part4_status.txt
grep -aE "GPU KV cache size|Available KV cache memory|kv_cache_dtype|Using.*backend|attention backend|Triton|TRITON" $R/results/claude_p4_server.log | head -40 > $R/results/claude_p4_proof_lines.txt
docker rm -f "${NAME}" >/dev/null 2>&1 || true
echo "PART4_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> $R/part4_status.txt
