# llama.cpp Larger Qwen3/Qwen3.6 GGUF Packet, 2026-06-09

Purpose: fill the llama.cpp Qwen speed gap with a Qwen3/Qwen3.6-class GGUF row.

The existing llama.cpp Qwen row is `Qwen/Qwen2.5-1.5B-Instruct-GGUF` Q4_K_M. It proves
small-Qwen practical serving, but it is not comparable to the larger vLLM Qwen3.6 and
SGLang Qwen lanes. This packet records a larger GGUF serving row through the same row
manifest path used by the accepted small-Qwen result.

This is a practical-serving packet. It does not prove paper-comparable GGUF accuracy, and
it does not prove native NVFP4/MXFP4 tensor-core dispatch unless the selected model is an
NVFP4/MXFP4 GGUF and separate native-FP4 dispatch evidence is captured.

## Required Inputs

Run on the GB10 Linux host with the llama.cpp CUDA build available.

```bash
set -euo pipefail

LLAMA_SERVER=${LLAMA_SERVER:-/home/jethac/src/llama.cpp-b9536/build/bin/llama-server}
LLAMA_BENCH=${LLAMA_BENCH:-/home/jethac/src/llama.cpp-b9536/build/bin/llama-bench}
LLAMA_CPP_COMMIT=${LLAMA_CPP_COMMIT:-"llama.cpp 308f61c31 b9536"}

# Required: choose a Qwen3/Qwen3.6-class instruct GGUF, not the existing Qwen2.5 1.5B file.
QWEN3_GGUF_MODEL=${QWEN3_GGUF_MODEL:?set to a Qwen3/Qwen3.6-class instruct GGUF path}
MODEL_ALIAS=${MODEL_ALIAS:-qwen3-larger-gguf}
RUN_ID=${RUN_ID:-llamacpp_qwen3_gguf_$(date +%Y%m%dT%H%MJST)}
PORT=${PORT:-18082}
CTX_SIZE=${CTX_SIZE:-8192}
RESULTS_DIR=${RESULTS_DIR:-/home/jethac/dgx-spark-hijinks/results}
SERVER_LOG=${RESULTS_DIR}/${RUN_ID}_server.log

test -x "$LLAMA_SERVER" || { echo "missing executable LLAMA_SERVER=$LLAMA_SERVER" >&2; exit 2; }
test -x "$LLAMA_BENCH" || { echo "missing executable LLAMA_BENCH=$LLAMA_BENCH" >&2; exit 2; }
test -f "$QWEN3_GGUF_MODEL" || { echo "missing QWEN3_GGUF_MODEL=$QWEN3_GGUF_MODEL" >&2; exit 2; }
case "$(basename "$QWEN3_GGUF_MODEL" | tr '[:upper:]' '[:lower:]')" in
  *qwen2.5*1.5b*) echo "refusing the existing small Qwen2.5 1.5B row" >&2; exit 2 ;;
esac
mkdir -p "$RESULTS_DIR"
```

## Start Server

```bash
"$LLAMA_SERVER" \
  --model "$QWEN3_GGUF_MODEL" \
  --alias "$MODEL_ALIAS" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  --gpu-layers all \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" || true
    wait "$SERVER_PID" || true
  fi
}
trap cleanup EXIT

deadline=$((SECONDS + 600))
until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "llama-server exited before readiness" >&2
    tail -200 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if [ "$SECONDS" -gt "$deadline" ]; then
    echo "llama-server did not become ready within 600s" >&2
    tail -200 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 2
done
```

Before recording the row, inspect the server log for:

- `CUDA : ARCHS`
- `USE_GRAPHS`
- the runtime device line naming `NVIDIA GB10` or equivalent CUDA device evidence
- no `error loading model`, CUDA error, or CPU-only fallback

## Record Row

```bash
python3 scripts/record_openai_serving_row.py \
  --backend llamacpp \
  --phase exploratory \
  --run-id "$RUN_ID" \
  --url "http://127.0.0.1:${PORT}" \
  --model "$MODEL_ALIAS" \
  --runtime-ref "$LLAMA_CPP_COMMIT" \
  --quantization gguf \
  --kv-cache-dtype default \
  --attention-backend llama.cpp-cuda \
  --cuda-graph-mode enabled \
  --server-log "$SERVER_LOG" \
  --process-match llama-server \
  --llama-bench-command "$LLAMA_BENCH --model '$QWEN3_GGUF_MODEL' --gpu-layers all --ctx-size '$CTX_SIZE'" \
  --run-gguf-logprobs-probe
```

Then audit the resulting manifest:

```bash
python3 scripts/serving_manifest_audit.py \
  --manifest "results/${RUN_ID}_row_manifest.json" \
  --output "results/${RUN_ID}_serving_manifest_audit.json"
```

The GGUF logprobs probe may stay red for paper-comparable accuracy. That does not fail this
practical-serving row, but it must remain labeled separately from throughput.

## Acceptance

Green for this packet means:

- the selected GGUF is Qwen3/Qwen3.6-class, not the existing Qwen2.5 1.5B row;
- `results/${RUN_ID}_row_manifest.json` has `ok=true`;
- `results/${RUN_ID}_openai_benchmark.json` has normal compact serving output and token/s rows;
- `results/${RUN_ID}_llama_bench.txt` records CUDA/GB10-backed PP/TG performance;
- `results/${RUN_ID}_build_target_audit.json` records accepted build-target evidence from the server log, or the summary explicitly names the missing log marker as a follow-up;
- `results/${RUN_ID}_serving_manifest_audit.json` is preserved even if not fully claim-ready;
- paper-comparable accuracy remains blocked unless the GGUF logprobs artifact also passes the supplied-token contract.

Expected queue artifacts:

- `results/${RUN_ID}_server.log`
- `results/${RUN_ID}_row_manifest.json`
- `results/${RUN_ID}_openai_benchmark.json`
- `results/${RUN_ID}_llama_bench.txt`
- `results/${RUN_ID}_gguf_logprobs_probe.json`
- `results/${RUN_ID}_serving_manifest_audit.json`
