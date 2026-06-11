#!/usr/bin/env bash
# Part 2 of combined window: DG-0 DiffusionGemma baseline per docs/DG0_SERVING_STACK_RECON.md.
# Official image vllm/vllm-openai:gemma-aarch64-cu130 (PR #45163 dgemma stack), BF16 checkpoint
# google/diffusiongemma-26B-A4B-it, recipe flags ONLY (no jethac knobs):
#   --diffusion-config '{"canvas_length":256}' --max-num-seqs 4 --attention-backend TRITON_ATTN
#   --max-model-len 262144 (official recipe value), util 0.72 (window cap; recipe says 0.85).
set -uo pipefail
R=/home/jethac/spark_tmp/claude_dg0_anchor_window_20260611/part2
IMAGE=vllm/vllm-openai:gemma-aarch64-cu130
MODEL=google/diffusiongemma-26B-A4B-it
NAME=claude_dg0_baseline
S=$R/status.txt
T0=$(date +%s)

echo "RUN_START $(date -Is) IMAGE=${IMAGE} IMAGE_ID=$(docker images --format '{{.ID}}' ${IMAGE})" >> "$S"

# Default entrypoint of the official image (api server); recipe-style args only.
docker run -d --rm --name "${NAME}" --gpus all --ipc=host --network host \
  --memory 100g --memory-swap 100g \
  -v /home/jethac/.cache/huggingface:/root/.cache/huggingface \
  -v "${R}:/work" \
  "${IMAGE}" \
  --model "${MODEL}" \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.72 \
  --attention-backend TRITON_ATTN \
  --diffusion-config '{"canvas_length":256}' \
  --host 0.0.0.0 --port 8000
echo "DOCKER_RUN_RC=$?" >> "$S"

# Stream container logs to file (container is --rm; logs vanish on removal otherwise).
nohup docker logs -f "${NAME}" > "$R/results/claude_dg0_server.log" 2>&1 &
LOGS_PID=$!

# Readiness: weights download (~52 GB, allow 35 min) + 20 min post-weights = poll up to 60 min.
ready_rc=1
for i in $(seq 1 360); do
  if docker exec -i "${NAME}" python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2).read()" >/dev/null 2>&1; then
    ready_rc=0; break
  fi
  if ! docker ps --format "{{.Names}}" | grep -q "^${NAME}$"; then ready_rc=3; break; fi
  sleep 10
done
echo "DG0 READY_RC=${ready_rc} READY_WALL=$(( $(date +%s) - T0 ))" >> "$S"

extract_proof() {
  grep -nE "attention backend|TRITON|FLASH|FLEX|backend|GPU KV cache size|kv cache|KV cache|head_size|head dim|page size|block_size|hybrid|sliding|full_attention|attention group|KVCacheGroup|diffusion|Diffusion|canvas|denois|scheduler|sampler|entropy|spec.*decode|speculative|Maximum concurrency|Available KV cache|Traceback|ERROR|EngineCore failed|ValueError" \
    "$R/results/claude_dg0_server.log" > "$R/results/claude_dg0_proof_lines.txt" 2>&1
}
extract_proof

if [ ${ready_rc} -ne 0 ]; then
  echo "DG0 SERVER_DID_NOT_BECOME_READY" >> "$S"
  grep -B2 -A80 "Traceback\|EngineCore failed\|ERROR" "$R/results/claude_dg0_server.log" \
    > "$R/results/claude_dg0_crash_excerpt.txt" 2>&1 || true
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  kill ${LOGS_PID} >/dev/null 2>&1 || true
  echo "PART2_ABORT TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
  exit 2
fi
echo "DG0 VERDICT=SERVED" >> "$S"

# Bench: coherence completion + tok/s at ~32 / ~2000 / ~6000 prompt tokens, 3 reps each,
# 512 max_tokens (>256 to cross the diffusion block-emission boundary), streaming timing.
docker exec -i "${NAME}" python3 - << 'PY' > "$R/results/claude_dg0_bench.json" 2> "$R/results/claude_dg0_bench.stderr.log"
import json, time, urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "google/diffusiongemma-26B-A4B-it"

def post(path, payload, timeout=1800, stream=False):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    if not stream:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    # SSE streaming: return (events, timestamps)
    resp = urllib.request.urlopen(req, timeout=timeout)
    events, stamps = [], []
    buf = b""
    while True:
        chunk = resp.read(1)
        if not chunk:
            break
        buf += chunk
        if buf.endswith(b"\n\n"):
            for line in buf.decode("utf-8", "replace").splitlines():
                if line.startswith("data: "):
                    payload_s = line[6:]
                    if payload_s.strip() == "[DONE]":
                        continue
                    events.append(json.loads(payload_s))
                    stamps.append(time.monotonic())
            buf = b""
    return events, stamps

def tokenize_count(text):
    r = post("/tokenize", {"model": MODEL, "prompt": text})
    return r["count"]

out = {"schema": "claude-dg0-bench/v1", "model": MODEL}

# (b) coherence completion, temp 0, chat endpoint
t0 = time.monotonic()
resp = post("/v1/chat/completions", {
    "model": MODEL, "temperature": 0, "max_tokens": 512,
    "messages": [{"role": "user", "content": "Explain in one short paragraph why the sky is blue."}],
})
out["coherence"] = {
    "wall_s": time.monotonic() - t0,
    "text": resp["choices"][0]["message"]["content"],
    "usage": resp.get("usage"),
    "finish_reason": resp["choices"][0].get("finish_reason"),
}

# Build prompts at ~32 / ~2000 / ~6000 tokens via /tokenize calibration.
seed = ("The history of computing spans mechanical calculators, vacuum tubes, transistors, "
        "integrated circuits, and modern accelerators; each generation reshaped what software "
        "could attempt and what society expected from machines. ")
unit = tokenize_count(seed)
out["tokenize_calibration"] = {"seed_tokens": unit}

def build_prompt(target):
    reps = max(1, target // unit)
    text = seed * reps
    n = tokenize_count(text)
    while n < target - unit and reps < 4096:
        reps += max(1, (target - n) // unit)
        text = seed * reps
        n = tokenize_count(text)
    return text, n

scales = []
for target in (32, 2000, 6000):
    if target <= unit:
        text = seed
        n = tokenize_count(text)
    else:
        text, n = build_prompt(target)
    prompt = text + "\n\nSummarize the passage above in detail, then continue the discussion."
    n_full = tokenize_count(prompt)
    reps_out = []
    for rep in range(3):
        t0 = time.monotonic()
        events, stamps = post("/v1/completions", {
            "model": MODEL, "prompt": prompt, "temperature": 0,
            "max_tokens": 512, "stream": True,
            "stream_options": {"include_usage": True},
        }, stream=True)
        t_end = time.monotonic()
        usage = None
        first_tok_t = None
        last_tok_t = None
        n_events_with_text = 0
        for ev, st in zip(events, stamps):
            u = ev.get("usage")
            if u:
                usage = u
            ch = ev.get("choices") or []
            if ch and ch[0].get("text"):
                n_events_with_text += 1
                if first_tok_t is None:
                    first_tok_t = st
                last_tok_t = st
        comp = (usage or {}).get("completion_tokens")
        gen_window = (last_tok_t - first_tok_t) if (first_tok_t and last_tok_t and last_tok_t > first_tok_t) else None
        reps_out.append({
            "rep": rep + 1,
            "wall_s": t_end - t0,
            "ttft_s": (first_tok_t - t0) if first_tok_t else None,
            "stream_events_with_text": n_events_with_text,
            "usage": usage,
            "gen_window_s_first_to_last_chunk": gen_window,
            "gen_tok_per_s_first_to_last": (comp / gen_window) if (comp and gen_window) else None,
            "gen_tok_per_s_incl_ttft": (comp / (t_end - t0)) if comp else None,
        })
    scales.append({"target_prompt_tokens": target, "actual_prompt_tokens": n_full, "reps": reps_out})
out["scales"] = scales
print(json.dumps(out, indent=2))
PY
echo "DG0 BENCH_RC=$? BENCH_WALL=$(( $(date +%s) - T0 ))" >> "$S"

extract_proof
# Capture throughput/denoise stat lines the server printed during the bench.
grep -nE "throughput|denois|steps|Avg|tok/s" "$R/results/claude_dg0_server.log" \
  > "$R/results/claude_dg0_throughput_lines.txt" 2>&1 || true

docker rm -f "${NAME}" >/dev/null 2>&1 || true
sleep 2
kill ${LOGS_PID} >/dev/null 2>&1 || true
echo "PART2_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
