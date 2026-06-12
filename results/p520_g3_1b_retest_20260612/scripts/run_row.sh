#!/usr/bin/env bash
# Run ONE PPL row for the Gemma 3 1B FlashInfer re-test. One server at a time.
# Args: ROW_NAME  (env: UTIL, BACKEND, KVDTYPE, EXTRA_ENV)
# Emits proof lines, launches server, waits ready (timeout = wedge detector),
# chat smoke, C1 PPL x2 (bitwise), tears down. Mirrors the 270M agent.
set -u
ROW="${1:?row name}"
UTIL="${UTIL:-0.6}"
BACKEND="${BACKEND:-flash_attn}"      # flash_attn | flashinfer
KVDTYPE="${KVDTYPE:-}"                # empty (bf16/auto) | nvfp4
READY_TIMEOUT="${READY_TIMEOUT:-300}" # 5 min; a hang past this = WEDGE
MODEL="google/gemma-3-1b-it"
SERVED="gemma3-1b-it"
CORPUS=~/corpora/hijinks_serving/c1_ppl_corpus.md
WORK=~/g3_1b_retest
RES="$WORK/results"
URL="http://127.0.0.1:8000"
PREFIX="claude_p520_${ROW}"

cd "$WORK"
source ~/vllm_wheel_env/bin/activate
export PYTHONPATH=~/flashinfer
export FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1"
export EXT_PATH="$(python -c 'import vllm,os;print(os.path.join(os.path.dirname(vllm.__file__),"_C_stable_libtorch.abi3.so"))')"
export PYTHONIOENCODING=utf-8
# row-specific extra env (e.g. VLLM_FLASHINFER_BF16_GEMMA=1, VLLM_NVFP4_KV_LINEAR_V_SF=1)
if [ -n "${EXTRA_ENV:-}" ]; then export $EXTRA_ENV; fi

SERVERLOG="$RES/${PREFIX}_server.log"
PROOF="$RES/${PREFIX}_proof_lines.txt"

{
  echo "VLLM_BUILD_CHECK $(python -c 'import vllm;print(vllm.__version__)') $(python -c 'import vllm;print(vllm.__file__)')"
  echo "EXT_PATH $EXT_PATH"
  echo "FLASHINFER_CHECK $(python -c 'import flashinfer;print(flashinfer.__file__)')"
} > "$SERVERLOG"

SERVE_ARGS=( "$MODEL" --host 127.0.0.1 --max-model-len 8192
             --served-model-name "$SERVED" --gpu-memory-utilization "$UTIL"
             --attention-backend "$BACKEND" )
if [ -n "$KVDTYPE" ]; then SERVE_ARGS+=( --kv-cache-dtype "$KVDTYPE" ); fi

echo "STATUS ROW=$ROW START $(date -u +%FT%TZ) UTIL=$UTIL BACKEND=$BACKEND KVDTYPE=${KVDTYPE:-bf16} EXTRA_ENV=${EXTRA_ENV:-none}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

vllm serve "${SERVE_ARGS[@]}" >> "$SERVERLOG" 2>&1 &
SRV=$!

# Wait for readiness; treat timeout as a WEDGE.
READY=0
for i in $(seq 1 "$READY_TIMEOUT"); do
  if ! kill -0 "$SRV" 2>/dev/null; then echo "STATUS ROW=$ROW SERVER_DIED at ${i}s"; break; fi
  if curl -s -o /dev/null -w '%{http_code}' "$URL/v1/models" 2>/dev/null | grep -q 200; then READY=1; echo "STATUS ROW=$ROW READY ${i}s"; break; fi
  sleep 1
done

# Extract the backend proof line regardless of outcome.
grep -nE "Using AttentionBackendEnum|Using FlashAttention|VLLM_NVFP4_KV_LINEAR_V_SF|forcing|non-default args|VLLM_FLASHINFER_BF16_GEMMA|nvfp4 data type|VLLM_BUILD_CHECK|EXT_PATH|FLASHINFER_CHECK" "$SERVERLOG" | head -40 >> "$PROOF"

if [ "$READY" != "1" ]; then
  echo "STATUS ROW=$ROW NOT_READY/WEDGE after ${READY_TIMEOUT}s — capturing tail"
  tail -60 "$SERVERLOG" > "$RES/${PREFIX}_wedge_or_crash_excerpt.txt"
  kill -9 "$SRV" 2>/dev/null; pkill -9 -f "vllm serve" 2>/dev/null; sleep 5
  echo "ROW=$ROW VERDICT=WEDGE_OR_DIED"
  exit 3
fi

# Chat smoke (coherence)
python openai_chat_smoke.py --model "$SERVED" --url "$URL" \
  --prompt "What is the capital of Japan? Answer in one short sentence." --max-tokens 24 \
  --output "$RES/${PREFIX}_chat_smoke.json" > "$RES/${PREFIX}_chat_smoke_stdout.txt" 2>&1
echo "STATUS ROW=$ROW CHAT_SMOKE done rc=$?"

RUNTIME_REF="P520 RTX 5060 Ti sm_120 WSL2; vllm $(python -c 'import vllm;print(vllm.__version__)'); flashinfer source-JIT; gemma3-1b-it; util${UTIL}; row=${ROW}; corpus=c1"

run_ppl () { # $1 = a|b
  local tag="$1"
  python vllm_prompt_ppl_sweep.py \
    --url "$URL" --model "$SERVED" --tokenizer "$MODEL" \
    --text-file "$CORPUS" --ctx 8191 \
    --run-id "${PREFIX}_c1${tag/a/}_ctx8191" \
    --kv-cache-dtype "${KVDTYPE:-bf16}" \
    --runtime-ref "$RUNTIME_REF" \
    --output "$RES/${PREFIX}_c1${tag/a/}_ctx8191_ppl.json" \
    --timeout 600 --request-attempts 2 --retry-sleep-s 10 \
    > "$RES/${PREFIX}_c1${tag/a/}_ctx8191_stdout.json" 2>"$RES/${PREFIX}_c1${tag/a/}_ctx8191_stderr.log"
  echo "STATUS ROW=$ROW C1${tag}_PPL rc=$?"
}
run_ppl a
run_ppl b

kill -INT "$SRV" 2>/dev/null; sleep 5; kill -9 "$SRV" 2>/dev/null; pkill -9 -f "vllm serve" 2>/dev/null; sleep 5
echo "STATUS ROW=$ROW DONE $(date -u +%FT%TZ)"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
