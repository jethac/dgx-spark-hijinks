#!/usr/bin/env bash
# 31B flagship quality-gate: row 2 RERUN (full NVFP4, VO-split + linear V-SF) after overlay stale-.so clean.
# Row 1 (fp8 comparator) stands from /home/jethac/spark_tmp/claude_31b_ppl_pair_20260611 — NOT rerun.
# Row-2 portion reused verbatim from run_31b_ppl_pair.sh; ADDED: EXT_PATH proof print + abort gate.
set -uo pipefail
R=/home/jethac/spark_tmp/claude_31b_ppl_row2_rerun_20260611
CLONE=/home/jethac/spark_tmp/vllm-022-ad2337814-clone
IMAGE=jethac-vllm-aeon-gemma4:e08a6f3ae-rebuiltc-fb7d62ea-sm121a-r8
CTX=8192
CTX_FALLBACK=8191
S=$R/status.txt
T0=$(date +%s)

echo "RUN_START $(date -Is) OVERLAY_HEAD=$(git -C ${CLONE} rev-parse HEAD) IMAGE=${IMAGE}" >> "$S"
echo "OVERLAY_SO_COUNT=$(find ${CLONE}/vllm -name '*.so' | wc -l)" >> "$S"

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
  local name=$1 label=$2
  docker exec -i "${name}" python3 - << 'PY' > "$R/results/claude_31b_${label}_gates.json" 2> "$R/results/claude_31b_${label}_gates.stderr.log"
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

out = {"schema": "claude-31b-ppl-pair-gates/v1"}
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
  grep -n "OVERLAY_CHECK\|EXT_PATH\|Unknown vLLM environment variable\|mm.prefix\|mm_prefix\|prefix-LM\|attention backend\|TRITON\|FLASHINFER\|FlashInfer FA2 backend\|V-scale-factor\|V-SF\|de-swizzle\|deswizzle\|FA2 VO split\|GPU KV cache size\|heterogeneous head\|VLLM_NVFP4\|language.model.only\|Traceback\|ValueError\|ERROR" \
    "$R/results/claude_31b_${label}_server.log" > "$R/results/claude_31b_${label}_proof_lines.txt" 2>&1
}

run_ppl() {
  local name=$1 label=$2 kvdtype=$3 ctx=$4
  docker exec "${name}" python3 scripts/vllm_prompt_ppl_sweep.py \
    --url http://127.0.0.1:8000 \
    --model gemma4-31b-it \
    --tokenizer google/gemma-4-31B-it \
    --text-file docs/ppl_corpus.md \
    --ctx "${ctx}" \
    --run-id "claude_31b_${label}_ctx${ctx}" \
    --kv-cache-dtype "${kvdtype}" \
    --runtime-ref "jethac/vllm@9759e3b06 overlay on r8 image (jethac/flashinfer@fb7d62ea rebuilt-C); --language-model-only; row=${label}; rerun after overlay stale-.so clean" \
    --container-image "${IMAGE}" \
    --output "results/claude_31b_${label}_ctx${ctx}_ppl.json" \
    > "$R/results/claude_31b_${label}_ctx${ctx}_stdout.json" \
    2> "$R/results/claude_31b_${label}_ctx${ctx}_stderr.log"
}

run_row() {
  local label=$1 kvdtype=$2; shift 2
  local extra_env=("$@")
  local name=claude_31b_${label}
  local t0=$(date +%s)
  docker run -d --rm --name "${name}" --gpus all --net host --ipc host \
    --memory 100g --memory-swap 100g \
    -w /work \
    "${extra_env[@]}" \
    -e PYTHONPATH=/opt/vllm_overlay \
    -v "${CLONE}:/opt/vllm_overlay:ro" \
    -v /home/jethac/.cache/huggingface:/root/.cache/huggingface \
    -v "${R}:/work" \
    "${IMAGE}" \
    bash -lc "exec > /work/results/claude_31b_${label}_server.log 2>&1; \
      python3 -c 'import vllm; print(\"OVERLAY_CHECK vllm.__file__=\", vllm.__file__, flush=True)'; \
      python3 -c 'import importlib.util as iu; s=iu.find_spec(\"vllm.v1.attention.backends.flashinfer\"); print(\"OVERLAY_CHECK flashinfer=\", s.origin, flush=True)'; \
      python3 -c 'import vllm._C_stable_libtorch as m; print(\"EXT_PATH\", m.__file__, flush=True)'; \
      vllm serve google/gemma-4-31B-it \
      --served-model-name gemma4-31b-it \
      --host 0.0.0.0 --port 8000 \
      --kv-cache-dtype ${kvdtype} \
      --gpu-memory-utilization 0.72 \
      --max-model-len 8192 \
      --language-model-only"

  # EXT_PATH abort gate: wait for the proof line, verify it does NOT resolve to the overlay.
  local extpath=""
  for _ in $(seq 1 60); do
    extpath=$(grep -m1 '^EXT_PATH ' "$R/results/claude_31b_${label}_server.log" 2>/dev/null | awk '{print $2}')
    [ -n "$extpath" ] && break
    if ! docker ps --format "{{.Names}}" | grep -q "^${name}$"; then break; fi
    sleep 2
  done
  echo "ROW=${label} EXT_PATH=${extpath:-MISSING}" >> "$S"
  case "${extpath}" in
    /opt/vllm_overlay*)
      echo "ROW=${label} EXT_PATH_RESOLVED_TO_OVERLAY ABORT" >> "$S"
      docker rm -f "${name}" >/dev/null 2>&1 || true
      return 5
      ;;
    "")
      echo "ROW=${label} EXT_PATH_LINE_MISSING ABORT" >> "$S"
      docker rm -f "${name}" >/dev/null 2>&1 || true
      return 6
      ;;
  esac

  wait_ready "${name}"; rc=$?
  echo "ROW=${label} READY_RC=${rc} READY_WALL=$(( $(date +%s) - t0 ))" >> "$S"
  extract_proof "${label}"
  if [ $rc -ne 0 ]; then
    echo "ROW=${label} SERVER_DID_NOT_BECOME_READY" >> "$S"
    docker rm -f "${name}" >/dev/null 2>&1 || true
    return 2
  fi

  run_ppl "${name}" "${label}" "${kvdtype}" "${CTX}"; pplrc=$?
  echo "ROW=${label} PPL_RC=${pplrc} CTX=${CTX} PPL_WALL=$(( $(date +%s) - t0 ))" >> "$S"
  if [ $pplrc -ne 0 ]; then
    run_ppl "${name}" "${label}" "${kvdtype}" "${CTX_FALLBACK}"; pplrc2=$?
    echo "ROW=${label} PPL_FALLBACK_RC=${pplrc2} CTX=${CTX_FALLBACK} PPL_WALL=$(( $(date +%s) - t0 ))" >> "$S"
  fi

  run_gates "${name}" "${label}"
  echo "ROW=${label} GATES_RC=$? GATES_WALL=$(( $(date +%s) - t0 ))" >> "$S"

  extract_proof "${label}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  echo "ROW=${label} DONE ROW_WALL=$(( $(date +%s) - t0 ))" >> "$S"
}

run_row row2_nvfp4 nvfp4 -e VLLM_NVFP4_KV_VOSPLIT=1 -e VLLM_NVFP4_KV_LINEAR_V_SF=1
echo "RERUN_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
