#!/usr/bin/env bash
# Part 1 of combined window (docs/WINDOW_PACKET_DG0_PLUS_ANCHOR.md, epoch2):
# 31B bf16 anchor row on the r9 image (dispatcher fix BAKED — no overlay, no PYTHONPATH).
# Serve google/gemma-4-31B-it bf16 (NO --kv-cache-dtype) with VLLM_FLASHINFER_VOSPLIT=1,
# --language-model-only --max-model-len 8192 --gpu-memory-utilization 0.72.
# Proof: "FA2 VO split (auto KV)" line, ZERO max_mma_kv occurrences, EXT_PATH + image id.
# Measure: ctx-8191 prompt PPL (corpus abb63f0e), first-token gate, 3 decode reps, KV tokens.
# Crash fallback: full traceback capture + ONE rerun with FLASHINFER_PREFILL_DEBUG_ONCE=1.
set -uo pipefail
R=/home/jethac/spark_tmp/claude_dg0_anchor_window_20260611/part1
IMAGE=jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9
CTX=8191
MODEL=google/gemma-4-31B-it
SERVED=gemma4-31b-it
S=$R/status.txt
T0=$(date +%s)

echo "RUN_START $(date -Is) IMAGE=${IMAGE} IMAGE_ID=$(docker images --format '{{.ID}}' ${IMAGE})" >> "$S"
echo "CORPUS_MD5=$(md5sum ${R}/docs/ppl_corpus.md | awk '{print $1}')" >> "$S"

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

run_gates() {
  local name=$1 label=$2 served=$3
  docker exec -i -e GATE_MODEL="${served}" "${name}" python3 - << 'PY' > "$R/results/claude_${label}_gates.json" 2> "$R/results/claude_${label}_gates.stderr.log"
import json, os, time, urllib.request

MODEL = os.environ["GATE_MODEL"]

def chat(messages, max_tokens, ignore_eos=False):
    payload = {
        "model": MODEL,
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

out = {"schema": "claude-dg0-anchor-window-gates/v1", "model": MODEL}
resp, wall = chat([{"role": "user", "content": "In one sentence, what is the NVIDIA GB10 Grace Blackwell superchip?"}], 128)
out["first_token_gate"] = {
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
}

extract_proof() {
  local label=$1
  grep -n "EXT_PATH\|VLLM_BUILD_CHECK\|Unknown vLLM environment variable\|attention backend\|TRITON\|FLASHINFER\|FlashInfer FA2 backend\|FA2 VO split\|GPU KV cache size\|heterogeneous head\|VLLM_NVFP4\|VLLM_FLASHINFER\|max_mma_kv\|PREFILL_DEBUG\|language.model.only\|Traceback\|ValueError\|ERROR" \
    "$R/results/claude_${label}_server.log" > "$R/results/claude_${label}_proof_lines.txt" 2>&1
  echo "MAX_MMA_KV_COUNT=$(grep -c 'max_mma_kv' "$R/results/claude_${label}_server.log")" >> "$R/results/claude_${label}_proof_lines.txt"
}

run_ppl() {
  local name=$1 label=$2
  docker exec "${name}" python3 scripts/vllm_prompt_ppl_sweep.py \
    --url http://127.0.0.1:8000 \
    --model "${SERVED}" \
    --tokenizer "${MODEL}" \
    --text-file docs/ppl_corpus.md \
    --ctx "${CTX}" \
    --run-id "claude_${label}_ctx${CTX}" \
    --kv-cache-dtype bf16 \
    --runtime-ref "r9 image (fix BAKED, no overlay) jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9; VLLM_FLASHINFER_VOSPLIT=1; --language-model-only; row=${label}; combined DG0+anchor window" \
    --container-image "${IMAGE}" \
    --output "results/claude_${label}_ctx${CTX}_ppl.json" \
    > "$R/results/claude_${label}_ctx${CTX}_stdout.json" \
    2> "$R/results/claude_${label}_ctx${CTX}_stderr.log"
}

# start_server <name> <label> [extra docker args...]
start_server() {
  local name=$1 label=$2; shift 2
  local extra_docker=("$@")
  docker run -d --rm --name "${name}" --gpus all --net host --ipc host \
    --memory 100g --memory-swap 100g \
    -w /work \
    -e VLLM_FLASHINFER_VOSPLIT=1 \
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
      --gpu-memory-utilization 0.72 \
      --max-model-len 8192 \
      --language-model-only"
}

run_row() {
  local label=$1; shift
  local extra_docker=("$@")
  local name=claude_${label}
  local t0=$(date +%s)
  start_server "${name}" "${label}" "${extra_docker[@]}"

  wait_ready "${name}"; rc=$?
  echo "ROW=${label} READY_RC=${rc} READY_WALL=$(( $(date +%s) - t0 ))" >> "$S"
  extract_proof "${label}"
  if [ $rc -ne 0 ]; then
    echo "ROW=${label} SERVER_DID_NOT_BECOME_READY" >> "$S"
    grep -B2 -A80 "max_mma_kv\|PREFILL_DEBUG\|EngineCore failed\|Traceback" "$R/results/claude_${label}_server.log" \
      > "$R/results/claude_${label}_crash_excerpt.txt" 2>&1 || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
    return 2
  fi
  echo "ROW=${label} VERDICT=SERVED" >> "$S"

  run_ppl "${name}" "${label}"
  echo "ROW=${label} PPL_RC=$? CTX=${CTX} PPL_WALL=$(( $(date +%s) - t0 ))" >> "$S"

  run_gates "${name}" "${label}" "${SERVED}"
  echo "ROW=${label} GATES_RC=$? GATES_WALL=$(( $(date +%s) - t0 ))" >> "$S"

  extract_proof "${label}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  echo "ROW=${label} DONE ROW_WALL=$(( $(date +%s) - t0 ))" >> "$S"
}

run_row anchor_bf16_vosplit_r9
rc0=$?
echo "ROW=anchor_bf16_vosplit_r9 FINAL_RC=${rc0}" >> "$S"
if [ ${rc0} -ne 0 ]; then
  echo "ROW=anchor_bf16_vosplit_r9 CRASHED -> DEBUG_RERUN_ONCE with FLASHINFER_PREFILL_DEBUG_ONCE=1" >> "$S"
  run_row anchor_vosplit_r9_debug -e FLASHINFER_PREFILL_DEBUG_ONCE=1
  echo "ROW=anchor_vosplit_r9_debug FINAL_RC=$?" >> "$S"
fi

echo "PART1_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
