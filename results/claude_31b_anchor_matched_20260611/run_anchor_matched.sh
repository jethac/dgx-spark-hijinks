#!/usr/bin/env bash
# 31B backend-matched anchor window (resolves the anchor-trio quality-table confound):
#   Row 0' (row0p_bf16_vosplit): 31B bf16 KV (no --kv-cache-dtype) + VLLM_FLASHINFER_VOSPLIT=1 ONLY
#       -> bf16 KV on FlashInfer+FA2-VO-split instead of the forced Triton fallback.
#       Crash fallback: if the server dies (E4B precedent: max_mma_kv 0 at warmup), rerun ONCE
#       with FLASHINFER_PREFILL_DEBUG_ONCE=1 to capture the debug printer's plan/module dump.
#   Row 1'' (row1pp_fp8): 31B fp8_e4m3, no knob envs — literal repeat of anchor-trio row 1'
#       to bound run-to-run variance (priors: tainted 4.473945, clean 4.591456).
# Runner reused from claude_anchor_trio_20260611/run_anchor_trio.sh; changes: row list,
# extra-env plumbing for row 0', debug rerun logic, PREFILL_DEBUG added to proof grep.
set -uo pipefail
R=/home/jethac/spark_tmp/claude_31b_anchor_matched_20260611
CLONE=/home/jethac/spark_tmp/vllm-022-ad2337814-clone
IMAGE=jethac-vllm-aeon-gemma4:e08a6f3ae-rebuiltc-fb7d62ea-sm121a-r8
CTX=8191
S=$R/status.txt
T0=$(date +%s)

echo "RUN_START $(date -Is) OVERLAY_HEAD=$(git -C ${CLONE} rev-parse HEAD) IMAGE=${IMAGE}" >> "$S"
echo "OVERLAY_SO_COUNT=$(find ${CLONE}/vllm -name '*.so' | wc -l) OVERLAY_REAL_SO=$(find ${CLONE}/vllm -name '*.so' -not -type l | wc -l)" >> "$S"
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

out = {"schema": "claude-anchor-matched-gates/v1", "model": MODEL}
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
  grep -n "OVERLAY_CHECK\|EXT_PATH\|Unknown vLLM environment variable\|mm.prefix\|mm_prefix\|prefix-LM\|attention backend\|TRITON\|FLASHINFER\|FlashInfer FA2 backend\|V-scale-factor\|V-SF\|de-swizzle\|deswizzle\|FA2 VO split\|GPU KV cache size\|heterogeneous head\|VLLM_NVFP4\|VLLM_FLASHINFER\|max_mma_kv\|PREFILL_DEBUG\|language.model.only\|Traceback\|ValueError\|ERROR" \
    "$R/results/claude_${label}_server.log" > "$R/results/claude_${label}_proof_lines.txt" 2>&1
}

run_ppl() {
  local name=$1 label=$2 kvlabel=$3 model=$4 served=$5
  docker exec "${name}" python3 scripts/vllm_prompt_ppl_sweep.py \
    --url http://127.0.0.1:8000 \
    --model "${served}" \
    --tokenizer "${model}" \
    --text-file docs/ppl_corpus.md \
    --ctx "${CTX}" \
    --run-id "claude_${label}_ctx${CTX}" \
    --kv-cache-dtype "${kvlabel}" \
    --runtime-ref "jethac/vllm@9759e3b06 overlay on r8 image (jethac/flashinfer@fb7d62ea rebuilt-C); --language-model-only; row=${label}; backend-matched anchor window" \
    --container-image "${IMAGE}" \
    --output "results/claude_${label}_ctx${CTX}_ppl.json" \
    > "$R/results/claude_${label}_ctx${CTX}_stdout.json" \
    2> "$R/results/claude_${label}_ctx${CTX}_stderr.log"
}

# start_server <name> <label> <model> <served> <kvflag-or-NONE> <extra-serve-args-or-NONE> [extra docker args...]
start_server() {
  local name=$1 label=$2 model=$3 served=$4 kvflag=$5 extra_serve=$6; shift 6
  local extra_docker=("$@")
  local kvarg=""
  if [ "${kvflag}" != "NONE" ]; then
    kvarg="--kv-cache-dtype ${kvflag}"
  fi
  local servearg=""
  if [ "${extra_serve}" != "NONE" ]; then
    servearg="${extra_serve}"
  fi
  docker run -d --rm --name "${name}" --gpus all --net host --ipc host \
    --memory 100g --memory-swap 100g \
    -w /work \
    "${extra_docker[@]}" \
    -e PYTHONPATH=/opt/vllm_overlay \
    -v "${CLONE}:/opt/vllm_overlay:ro" \
    -v /home/jethac/.cache/huggingface:/root/.cache/huggingface \
    -v "${R}:/work" \
    "${IMAGE}" \
    bash -lc "exec > /work/results/claude_${label}_server.log 2>&1; \
      python3 -c 'import vllm; print(\"OVERLAY_CHECK vllm.__file__=\", vllm.__file__, flush=True)'; \
      python3 -c 'import importlib.util as iu; s=iu.find_spec(\"vllm.v1.attention.backends.flashinfer\"); print(\"OVERLAY_CHECK flashinfer=\", s.origin, flush=True)'; \
      python3 -c 'import vllm._C_stable_libtorch as m; print(\"EXT_PATH\", m.__file__, flush=True)'; \
      vllm serve ${model} \
      --served-model-name ${served} \
      --host 0.0.0.0 --port 8000 \
      ${kvarg} ${servearg} \
      --gpu-memory-utilization 0.72 \
      --max-model-len 8192 \
      --language-model-only"
}

# EXT_PATH abort gate: wait for the proof line, verify it does NOT resolve to the overlay.
ext_gate() {
  local name=$1 label=$2
  local extpath=""
  for _ in $(seq 1 60); do
    extpath=$(grep -m1 '^EXT_PATH ' "$R/results/claude_${label}_server.log" 2>/dev/null | awk '{print $2}')
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
  return 0
}

run_ppl_row() {
  local label=$1 kvflag=$2 kvlabel=$3; shift 3
  local extra_docker=("$@")
  local model=google/gemma-4-31B-it
  local served=gemma4-31b-it
  local name=claude_${label}
  local t0=$(date +%s)
  start_server "${name}" "${label}" "${model}" "${served}" "${kvflag}" NONE "${extra_docker[@]}"
  ext_gate "${name}" "${label}" || return $?

  wait_ready "${name}"; rc=$?
  echo "ROW=${label} READY_RC=${rc} READY_WALL=$(( $(date +%s) - t0 ))" >> "$S"
  extract_proof "${label}"
  if [ $rc -ne 0 ]; then
    echo "ROW=${label} SERVER_DID_NOT_BECOME_READY" >> "$S"
    grep -B2 -A60 "max_mma_kv\|PREFILL_DEBUG\|EngineCore failed" "$R/results/claude_${label}_server.log" \
      > "$R/results/claude_${label}_crash_excerpt.txt" 2>&1 || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
    return 2
  fi
  echo "ROW=${label} VERDICT=SERVED" >> "$S"

  run_ppl "${name}" "${label}" "${kvlabel}" "${model}" "${served}"
  echo "ROW=${label} PPL_RC=$? CTX=${CTX} PPL_WALL=$(( $(date +%s) - t0 ))" >> "$S"

  run_gates "${name}" "${label}" "${served}"
  echo "ROW=${label} GATES_RC=$? GATES_WALL=$(( $(date +%s) - t0 ))" >> "$S"

  extract_proof "${label}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  echo "ROW=${label} DONE ROW_WALL=$(( $(date +%s) - t0 ))" >> "$S"
}

# ROW 0': bf16 KV on FlashInfer + FA2 VO split (env knob ONLY, no NVFP4 knobs).
run_ppl_row row0p_bf16_vosplit NONE bf16 -e VLLM_FLASHINFER_VOSPLIT=1
rc0=$?
echo "ROW=row0p_bf16_vosplit FINAL_RC=${rc0}" >> "$S"
if [ ${rc0} -ne 0 ]; then
  echo "ROW=row0p_bf16_vosplit CRASHED -> DEBUG_RERUN_ONCE with FLASHINFER_PREFILL_DEBUG_ONCE=1" >> "$S"
  run_ppl_row row0p_vosplit_debug NONE bf16 -e VLLM_FLASHINFER_VOSPLIT=1 -e FLASHINFER_PREFILL_DEBUG_ONCE=1
  echo "ROW=row0p_vosplit_debug FINAL_RC=$?" >> "$S"
fi

# ROW 1'': fp8_e4m3 variance repeat (no knob envs) — literal repeat of anchor-trio row 1'.
run_ppl_row row1pp_fp8 fp8_e4m3 fp8_e4m3
echo "ROW=row1pp_fp8 FINAL_RC=$?" >> "$S"

echo "MATCHED_DONE TOTAL_WALL=$(( $(date +%s) - T0 ))" >> "$S"
