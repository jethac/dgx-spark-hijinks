#!/usr/bin/env bash
# Part 2b: rerun of the DG-0 bench with ignore_eos=true (first bench hit EOS after 1-3
# tokens at the 32- and 6000-token scales; only the 2000-scale reps generated 512 tokens).
# Server command is UNCHANGED from part 2 (baseline recipe flags). Weights now cached.
set -uo pipefail
R=/home/jethac/spark_tmp/claude_dg0_anchor_window_20260611/part2
IMAGE=vllm/vllm-openai:gemma-aarch64-cu130
MODEL=google/diffusiongemma-26B-A4B-it
NAME=claude_dg0_baseline_b
S=$R/status.txt
T0=$(date +%s)

echo "RERUN_B_START $(date -Is) reason=ignore_eos_fix" >> "$S"

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
echo "B_DOCKER_RUN_RC=$?" >> "$S"

nohup docker logs -f "${NAME}" > "$R/results/claude_dg0b_server.log" 2>&1 &
LOGS_PID=$!

ready_rc=1
for i in $(seq 1 180); do
  if docker exec -i "${NAME}" python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2).read()" >/dev/null 2>&1; then
    ready_rc=0; break
  fi
  if ! docker ps --format "{{.Names}}" | grep -q "^${NAME}$"; then ready_rc=3; break; fi
  sleep 10
done
echo "B READY_RC=${ready_rc} READY_WALL=$(( $(date +%s) - T0 ))" >> "$S"
if [ ${ready_rc} -ne 0 ]; then
  grep -B2 -A80 "Traceback\|EngineCore failed\|ERROR" "$R/results/claude_dg0b_server.log" \
    > "$R/results/claude_dg0b_crash_excerpt.txt" 2>&1 || true
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  kill ${LOGS_PID} >/dev/null 2>&1 || true
  echo "RERUN_B_ABORT TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
  exit 2
fi

# Geometry extraction: model config attention geometry + full engine-init line.
docker exec -i "${NAME}" python3 - << 'PY' > "$R/results/claude_dg0_geometry.json" 2> "$R/results/claude_dg0_geometry.stderr.log"
import json, glob
paths = glob.glob("/root/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/*/config.json")
cfg = json.load(open(paths[0]))
tc = cfg.get("text_config", cfg)
keys = ["head_dim", "global_head_dim", "num_attention_heads", "num_key_value_heads",
        "num_global_key_value_heads", "num_hidden_layers", "layer_types", "sliding_window",
        "canvas_length", "max_position_embeddings", "hidden_size", "num_experts",
        "top_k_experts", "moe_intermediate_size", "rope_theta", "final_logit_softcapping"]
out = {"config_path": paths[0], "model_type": cfg.get("model_type"),
       "architectures": cfg.get("architectures"),
       "top_level_canvas_length": cfg.get("canvas_length"),
       "text_config": {k: tc.get(k) for k in keys}}
gen = glob.glob("/root/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/*/generation_config.json")
if gen:
    out["generation_config"] = json.load(open(gen[0]))
print(json.dumps(out, indent=2))
PY
echo "B GEOMETRY_RC=$?" >> "$S"

# Bench v2: ignore_eos=true, max_tokens 512, scales 32/2000/6000, 3 reps, streamed timing.
docker exec -i "${NAME}" python3 - << 'PY' > "$R/results/claude_dg0_bench_v2.json" 2> "$R/results/claude_dg0_bench_v2.stderr.log"
import json, time, urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "google/diffusiongemma-26B-A4B-it"

def post(path, payload, timeout=1800, stream=False):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    if not stream:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
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
                    s = line[6:]
                    if s.strip() == "[DONE]":
                        continue
                    events.append(json.loads(s))
                    stamps.append(time.monotonic())
            buf = b""
    return events, stamps

def tokenize_count(text):
    return post("/tokenize", {"model": MODEL, "prompt": text})["count"]

out = {"schema": "claude-dg0-bench/v2", "model": MODEL, "ignore_eos": True, "max_tokens": 512}

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
    return text

scales = []
for target in (32, 2000, 6000):
    text = seed if target <= unit else build_prompt(target)
    prompt = text + "\n\nSummarize the passage above in detail, then continue the discussion."
    n_full = tokenize_count(prompt)
    reps_out = []
    for rep in range(3):
        t0 = time.monotonic()
        events, stamps = post("/v1/completions", {
            "model": MODEL, "prompt": prompt, "temperature": 0,
            "max_tokens": 512, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True},
        }, stream=True)
        t_end = time.monotonic()
        usage = None
        first_tok_t = None
        last_tok_t = None
        chunk_log = []
        text_out = []
        for ev, st in zip(events, stamps):
            u = ev.get("usage")
            if u:
                usage = u
            ch = ev.get("choices") or []
            if ch and ch[0].get("text"):
                txt = ch[0]["text"]
                text_out.append(txt)
                chunk_log.append({"t_rel_s": st - t0, "chars": len(txt)})
                if first_tok_t is None:
                    first_tok_t = st
                last_tok_t = st
        comp = (usage or {}).get("completion_tokens")
        gen_window = (last_tok_t - first_tok_t) if (first_tok_t and last_tok_t and last_tok_t > first_tok_t) else None
        reps_out.append({
            "rep": rep + 1,
            "wall_s": t_end - t0,
            "ttft_s": (first_tok_t - t0) if first_tok_t else None,
            "n_text_chunks": len(chunk_log),
            "chunk_arrivals": chunk_log,
            "usage": usage,
            "gen_window_s_first_to_last_chunk": gen_window,
            "gen_tok_per_s_first_to_last": (comp / gen_window) if (comp and gen_window) else None,
            "gen_tok_per_s_excl_ttft_wallend": (comp / (t_end - first_tok_t)) if (comp and first_tok_t) else None,
            "gen_tok_per_s_incl_ttft": (comp / (t_end - t0)) if comp else None,
            "text_head": ("".join(text_out))[:160],
        })
    scales.append({"target_prompt_tokens": target, "actual_prompt_tokens": n_full, "reps": reps_out})
out["scales"] = scales
print(json.dumps(out, indent=2))
PY
echo "B BENCH_RC=$? BENCH_WALL=$(( $(date +%s) - T0 ))" >> "$S"

sleep 12  # let the last 10s metrics window flush to the log
grep -nE "Avg prompt throughput|Avg generation throughput|DiffusionDecoding metrics|denois" \
  "$R/results/claude_dg0b_server.log" > "$R/results/claude_dg0b_metrics_lines.txt" 2>&1 || true
grep -nE "Initializing a V1 LLM engine|Using AttentionBackendEnum|GPU KV cache size|Maximum concurrency|Available KV cache|non-default args|Resolved architecture" \
  "$R/results/claude_dg0b_server.log" > "$R/results/claude_dg0b_init_lines.txt" 2>&1 || true

docker rm -f "${NAME}" >/dev/null 2>&1 || true
sleep 2
kill ${LOGS_PID} >/dev/null 2>&1 || true
echo "RERUN_B_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
