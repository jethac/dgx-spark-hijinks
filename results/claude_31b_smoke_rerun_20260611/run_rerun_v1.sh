#!/usr/bin/env bash
set -uo pipefail
R=/home/jethac/spark_tmp/claude_window2_part3_rerun
CLONE=/home/jethac/spark_tmp/vllm-022-ad2337814-clone
IMAGE=jethac-vllm-aeon-gemma4:ad2337814-rebuiltc-fb7d62ea-sm121a
NAME=claude_p3_31b_rerun
T0=$(date +%s)

echo "OVERLAY_HEAD=$(git -C ${CLONE} rev-parse HEAD)" >> $R/rerun_status.txt

docker run -d --rm --name "${NAME}" --gpus all --ipc=host --net host \
  --memory 100g --memory-swap 100g \
  -w /work \
  -e VLLM_NVFP4_KV_VOSPLIT=1 -e VLLM_NVFP4_KV_LINEAR_V_SF=1 \
  -e CLAUDE_DISABLE_GEMMA4_MM_PREFIX=1 \
  -e PYTHONPATH=/opt/vllm_overlay:/opt/humming_pkg \
  -v ${CLONE}:/opt/vllm_overlay:ro \
  -v /home/jethac/spark_tmp/claude_window2_part2/humming_pkg:/opt/humming_pkg:ro \
  -v /home/jethac/spark_tmp/claude_window2_part3/model_arch_config_convertor_patched.py:/opt/vllm_overlay/vllm/transformers_utils/model_arch_config_convertor.py:ro \
  -v /home/jethac/spark_tmp/claude_window2_part3/model_arch_config_convertor_patched.py:/opt/jethac-vllm/vllm/transformers_utils/model_arch_config_convertor.py:ro \
  -v /home/jethac/.cache/huggingface:/root/.cache/huggingface \
  -v ${R}:/work \
  "${IMAGE}" \
  bash -lc 'python3 -c "import vllm; print(\"OVERLAY_CHECK vllm.__file__=\", vllm.__file__, flush=True)"; \
    python3 -c "import importlib.util as iu; s=iu.find_spec(\"vllm.v1.attention.backends.flashinfer\"); print(\"OVERLAY_CHECK flashinfer=\", s.origin, flush=True)"; \
    vllm serve google/gemma-4-31B-it \
    --served-model-name gemma4-31b-it \
    --host 0.0.0.0 --port 8000 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.72 \
    --max-model-len 8192 \
    > /work/results/claude_p3_rerun_server.log 2>&1'

ready=1
for i in $(seq 1 180); do
  if docker exec -i "${NAME}" python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2).read()' >/dev/null 2>&1; then
    ready=0; break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}\$"; then ready=3; break; fi
  sleep 5
done
echo "READY_RC=${ready} READY_WALL=$(( $(date +%s) - T0 ))" >> $R/rerun_status.txt
if [ $ready -ne 0 ]; then
  echo "SERVER_DID_NOT_BECOME_READY" >> $R/rerun_status.txt
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  exit 2
fi

docker exec -i "${NAME}" python3 - << 'PY' > $R/results/claude_p3_rerun_chat_and_decode.json 2> $R/results/claude_p3_rerun_chat_and_decode.stderr.log
import json, time, urllib.request

def chat(messages, max_tokens, ignore_eos=False):
    payload = {
        "model": "gemma4-31b-it",
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    body = json.dumps(payload).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    wall = time.monotonic() - t0
    return resp, wall

out = {"schema": "claude-window2-part3-smoke-rerun/v1"}
resp, wall = chat([{"role": "user", "content": "In one sentence, what is the NVIDIA GB10 Grace Blackwell superchip?"}], 128)
out["smoke_chat"] = {
    "wall_s": wall,
    "text": resp["choices"][0]["message"]["content"],
    "usage": resp.get("usage"),
    "finish_reason": resp["choices"][0].get("finish_reason"),
}
decode_prompt = ("Write a detailed, flowing description of a quiet mountain village at dawn, "
                 "covering light, sound, smell, and the slow waking of its people.")
reps = []
for i in range(3):
    resp, wall = chat([{"role": "user", "content": decode_prompt}], 256, ignore_eos=True)
    u = resp.get("usage", {})
    reps.append({
        "rep": i + 1, "wall_s": wall,
        "completion_tokens": u.get("completion_tokens"),
        "prompt_tokens": u.get("prompt_tokens"),
        "decode_tok_per_s_incl_prefill": (u.get("completion_tokens") or 0) / wall,
        "finish_reason": resp["choices"][0].get("finish_reason"),
        "text_head": resp["choices"][0]["message"]["content"][:120],
    })
out["decode_reps"] = reps
print(json.dumps(out, indent=2))
PY
echo "SMOKE_RC=$? SMOKE_WALL=$(( $(date +%s) - T0 ))" >> $R/rerun_status.txt
docker rm -f "${NAME}" >/dev/null 2>&1 || true
echo "RERUN_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> $R/rerun_status.txt
