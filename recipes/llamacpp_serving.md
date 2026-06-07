# llama.cpp Serving On Spark

Status: blessed for practical single-Spark serving; not blessed for lm-eval GGUF accuracy.

The initial personal benchmark run showed llama.cpp throughput working for at least one GGUF row. It also showed that GGUF lm-eval accuracy was blocked by a logprobs/API mismatch.

Those are separate problems.

## Serving Track

Use llama.cpp/Ollama as a practical local serving path while vLLM and SGLang mature.

Acceptance requires:

- exact llama.cpp commit
- build flags and CUDA architecture settings
- `spark_doctor` snapshot
- `llama-bench` throughput
- OpenAI-compatible smoke result if server mode is used
- model, quantization, context length, prompt length, and generated token count

Validated serving command:

```bash
/home/jethac/src/llama.cpp-b9536/build/bin/llama-server \
  --model /home/jethac/gemma4-vllm/models/gemma-4-26B_q4_0-it.gguf \
  --alias gemma4-26b-q4_0-gguf \
  --host 0.0.0.0 \
  --port 18082 \
  --ctx-size 8192 \
  --gpu-layers all \
  --reasoning off
```

`--reasoning off` matters for Gemma 4 chat smoke. Without it, llama.cpp returned text in `reasoning_content` while `message.content` was empty. With it, the OpenAI-compatible chat smoke returned `spark-ok`.

Current serving/throughput evidence:

- stock llama.cpp `b9536`
- build commit: `308f61c31`
- model: `gemma-4-26B_q4_0-it.gguf`
- alias: `gemma4-26b-q4_0-gguf`
- backend: CUDA on `NVIDIA GB10`
- server log: `CUDA : ARCHS = 1210`, `USE_GRAPHS = 1`, `BLACKWELL_NATIVE_FP4 = 1`
- OpenAI compact serving: about `76 tok/s` decode for 64- and 192-token generations
- `llama-bench`: `pp512` 3021.76 +/- 34.41 tok/s, `tg128` 77.35 +/- 0.13 tok/s

## Native FP4 Claim Boundary

This blessed row is Gemma 4 26B Q4_0 GGUF serving. `BLACKWELL_NATIVE_FP4 = 1` proves that the llama.cpp CUDA backend was built with Blackwell FP4 capability visible, but it does not prove this model used NVFP4/MXFP4 tensor types or native `sm_121a` FP4 MMA.

Native FP4 GGUF requires a separate experiment:

- NVFP4 or MXFP4 GGUF model, not Q4_0/Q4_K/Q8
- build-target evidence for `sm_121a` or a documented valid native-FP4 target
- runtime dispatch evidence from llama.cpp backend tests or logs
- comparison against the existing Q4_0 row

Artifacts:

- `results/llamacpp_gemma4_26b_q4_0_chat_smoke_20260607T135911Z.json`
- `results/llamacpp_gemma4_26b_q4_0_compact_20260607T135911Z.json`
- `results/llamacpp_gemma4_26b_q4_0_bench_20260607T135911Z.txt`
- `results/llamacpp_gemma4_26b_q4_0_20260607T135911Z_server.log`
- `results/spark_doctor_llamacpp_gemma4_26b_q4_0_20260607T135911Z.md`

Older observed throughput evidence:

- stock llama.cpp `b9536`
- model: `google__E2B__qat-q4_0__gemma-4-E2B_q4_0-it.gguf`
- prompt throughput: `3923.63 tok/s`
- generation throughput: `122.11 tok/s`

## Qwen GGUF Template

No Qwen GGUF row has been captured yet. Use env vars rather than host-specific paths:

```bash
LLAMA_SERVER=${LLAMA_SERVER:?}
LLAMA_BENCH=${LLAMA_BENCH:?}
GGUF_MODEL=${GGUF_MODEL:?}
MODEL_ALIAS=${MODEL_ALIAS:-qwen-gguf}
RUN_ID=${RUN_ID:-llamacpp_qwen_gguf_$(date -u +%Y%m%dT%H%M%SZ)}

"$LLAMA_SERVER" \
  --model "$GGUF_MODEL" \
  --alias "$MODEL_ALIAS" \
  --host 0.0.0.0 \
  --port 18082 \
  --ctx-size 8192 \
  --gpu-layers all
```

After the server is healthy:

```bash
python3 scripts/record_openai_serving_row.py \
  --backend llamacpp \
  --phase before \
  --run-id "$RUN_ID" \
  --url http://127.0.0.1:18082 \
  --model "$MODEL_ALIAS" \
  --runtime-ref "$LLAMA_CPP_COMMIT" \
  --quantization gguf \
  --server-log "results/${RUN_ID}_server.log" \
  --llama-bench-command "$LLAMA_BENCH --model $GGUF_MODEL --gpu-layers all" \
  --run-gguf-logprobs-probe
```

## Accuracy Track

Do not claim paper-comparable GGUF accuracy until:

```bash
python3 scripts/gguf_logprobs_probe.py --url http://127.0.0.1:8080
```

passes against the target llama.cpp server.

Current status: failed against llama.cpp `b9536`; see [../docs/GGUF_LLAMA_CPP_STATUS.md](../docs/GGUF_LLAMA_CPP_STATUS.md).

## Fork Rule

If llama.cpp needs source changes, fork `ggml-org/llama.cpp` to `jethac/llama.cpp`, add it as `third_party/llama.cpp`, and do the patch in a worktree named for the GitHub Issue.
