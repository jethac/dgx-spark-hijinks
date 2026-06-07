# GGUF llama.cpp Status

Status: practical serving works; paper-comparable lm-eval accuracy is blocked.

Tracked by:

- https://github.com/jethac/dgx-spark-hijinks/issues/8
- https://github.com/jethac/dgx-spark-hijinks/issues/17

## Evidence

Previous throughput row:

- backend: `llama.cpp-b9536`
- binary: `/home/jethac/src/llama.cpp-b9536/build/bin/llama-bench`
- model: `/home/jethac/gemma4-evals/models/gguf/google__E2B__qat-q4_0__gemma-4-E2B_q4_0-it.gguf`
- build commit: `308f61c31`
- build number: `9536`
- backend reported by llama.cpp: `CUDA`
- GPU reported by llama.cpp: `NVIDIA GB10`
- prompt throughput: `3923.63 tok/s`
- generation throughput: `122.11 tok/s`

Logprobs probe:

- binary: `/home/jethac/src/llama.cpp-b9536/build/bin/llama-server`
- model: `/home/jethac/gemma4-evals/models/gguf/google__E2B__qat-q4_0__gemma-4-E2B_q4_0-it.gguf`
- probe artifact: `results/gguf_logprobs_probe_llamacpp_b9536_20260607T1145Z.json`
- server log: `results/llama_server_logprobs_probe_20260607T1145Z.log`

The temporary server was started on `127.0.0.1:18081` and stopped after the probe.

Server log highlights:

- device: `CUDA0 : NVIDIA GB10`
- memory reported by llama.cpp: `122501 MiB`
- `CUDA : ARCHS = 1210`
- `USE_GRAPHS = 1`
- `BLACKWELL_NATIVE_FP4 = 1`
- server commit/fingerprint from response: `b9536-308f61c31`

The probe sent:

```json
{
  "prompt": "The capital of Japan is",
  "max_tokens": 1,
  "temperature": 0,
  "logprobs": 5,
  "echo": true
}
```

The response contained generated-token logprobs under:

```text
choices[0].logprobs.content
```

It did not contain:

```text
choices[0].logprobs.tokens
choices[0].logprobs.token_logprobs
```

## Decision

Do not run full lm-eval GGUF accuracy through this llama.cpp server/API path and call it paper-comparable.

This is not a model-quality failure and not a CUDA failure. It is an API/schema mismatch between the tested llama.cpp OpenAI-compatible completions response and the loglikelihood shape expected by the benchmark adapter.

## Fix Options

1. Pin a llama.cpp server/API version that returns echoed prompt/continuation token logprobs in the shape lm-eval expects.
2. Patch or replace the lm-eval GGUF adapter only if a llama.cpp API path can expose arbitrary continuation-token probabilities. The observed OpenAI-compatible `logprobs.content` response is generated-token scoring, not sufficient by itself.
3. Add a dedicated loglikelihood endpoint/path for llama.cpp that returns the exact continuation-token scores required by lm-eval.

Until one of those passes `scripts/gguf_logprobs_probe.py` or the native endpoint probe below, GGUF remains a throughput/serving path, not a paper-comparable accuracy path.

## Next Proof

The best candidate is llama.cpp's native non-OpenAI `/completion` API plus `/tokenize`, not reshaping the observed `/v1/completions` response.

Upstream API reference: `https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md`

Target proof:

1. Tokenize a context and two continuations, including one continuation token unlikely to be in top-5 alternatives.
2. For each continuation token, call `/completion` with `prompt = context_tokens + previous_continuation_tokens`, `n_predict=1`, `n_probs` high enough to include the target token, and `return_tokens=true`.
3. Pass only if the response exposes the exact target token id and its pre-sampling logprob for every continuation token.
4. Then run one tiny lm-eval GGUF task through that adapter path.

Host-ready probe:

```bash
python3 scripts/llamacpp_native_loglikelihood_probe.py \
  --url http://127.0.0.1:8080 \
  --n-probs 256 \
  --output results/llamacpp_native_loglikelihood_probe.json
```

The probe deliberately tests both a likely continuation and an unlikely continuation. Recovering only the generated top token is not enough for lm-eval.

If the native endpoint cannot return arbitrary target-token logprobs, this needs a llama.cpp upstream endpoint or a different accuracy backend.

## Practical Serving Note

This llama.cpp build is validated as a practical serving path on Spark:

- it detects GB10 correctly,
- it reports CUDA arch `1210`,
- it enables CUDA graphs,
- it reports native Blackwell FP4 support,
- it passes an OpenAI-compatible chat smoke when Gemma 4 thinking mode is disabled,
- and `llama-bench` plus OpenAI-compatible serving show useful throughput.

That should be tracked separately from lm-eval accuracy.

## 2026-06-08 Qwen Serving Validation

Target:

- binary: `/home/jethac/src/llama.cpp-b9536/build/bin/llama-server`
- version: `9536 (308f61c31)`
- model repo: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
- model file: `qwen2.5-1.5b-instruct-q4_k_m.gguf`
- local model path: `/home/jethac/models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf`
- alias: `qwen25-1.5b-q4_k_m-gguf`
- command flags: `--ctx-size 8192 -ngl 999`

Artifacts:

- run info: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_run_info.txt`
- server log: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_server.log`
- OpenAI chat smoke: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_chat_smoke.json`
- compact serving benchmark: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_openai_benchmark.json`
- `llama-bench`: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_llama_bench.txt`
- build-target audit: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_build_target_audit.json`
- runtime probe: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_runtime_probe.json`
- `spark_doctor`: `results/spark_doctor_llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST.md`
- logprobs probe: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_gguf_logprobs_probe.json`

Server log evidence:

- device: `CUDA0 : NVIDIA GB10`
- `CUDA : ARCHS = 1210`
- `USE_GRAPHS = 1`
- `BLACKWELL_NATIVE_FP4 = 1`
- chat template: Qwen instruct template detected

OpenAI-compatible serving result:

| case | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s |
|---|---:|---:|---:|---:|---:|
| `short_decode` | 44 | 64 | 0.032 | 0.397 | 175.19 |
| `medium_decode` | 56 | 192 | 0.015 | 1.113 | 174.86 |
| `long_prefill` | 2369 | 64 | 0.214 | 0.598 | 166.66 |

`llama-bench` result:

| model | backend | test | throughput |
|---|---|---|---:|
| `qwen2 1.5B Q4_K - Medium` | CUDA | `pp512` | 12505.79 +/- 615.87 tok/s |
| `qwen2 1.5B Q4_K - Medium` | CUDA | `tg128` | 178.10 +/- 0.95 tok/s |

Accuracy status under the same server:

- `scripts/gguf_logprobs_probe.py` still fails the lm-eval compatibility check.
- The response has `choices[0].logprobs.content`.
- The response still lacks `choices[0].logprobs.tokens` and `choices[0].logprobs.token_logprobs`.
- Therefore, Qwen GGUF serving is blessed for practical use, but GGUF paper-comparable lm-eval accuracy remains blocked.

## 2026-06-07 Serving Validation

Target:

- binary: `/home/jethac/src/llama.cpp-b9536/build/bin/llama-server`
- version: `9536 (308f61c31)`
- model: `/home/jethac/gemma4-vllm/models/gemma-4-26B_q4_0-it.gguf`
- alias: `gemma4-26b-q4_0-gguf`
- command flags: `--ctx-size 8192 --gpu-layers all --reasoning off`

Artifacts:

- run info: `results/llamacpp_gemma4_26b_q4_0_20260607T135911Z_run_info.txt`
- server log: `results/llamacpp_gemma4_26b_q4_0_20260607T135911Z_server.log`
- OpenAI chat smoke: `results/llamacpp_gemma4_26b_q4_0_chat_smoke_20260607T135911Z.json`
- compact serving benchmark: `results/llamacpp_gemma4_26b_q4_0_compact_20260607T135911Z.json`
- `llama-bench`: `results/llamacpp_gemma4_26b_q4_0_bench_20260607T135911Z.txt`
- `spark_doctor`: `results/spark_doctor_llamacpp_gemma4_26b_q4_0_20260607T135911Z.md`
- logprobs probe: `results/gguf_logprobs_probe_llamacpp_b9536_reasoning_off_20260607T135911Z.json`

Important configuration finding:

- Without `--reasoning off`, the chat smoke generated text under `reasoning_content` and left `message.content` empty.
- With `--reasoning off`, the same server returned normal `message.content` and passed the `spark-ok` smoke.

Server log evidence:

- device: `CUDA0 : NVIDIA GB10`
- `CUDA : ARCHS = 1210`
- `USE_GRAPHS = 1`
- `BLACKWELL_NATIVE_FP4 = 1`
- chat template: `thinking = 0`

OpenAI-compatible serving result:

| case | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s |
|---|---:|---:|---:|---:|---:|
| `short_decode` | 28 | 64 | 0.107 | 0.939 | 76.94 |
| `medium_decode` | 40 | 192 | 0.106 | 2.633 | 75.97 |

`llama-bench` result:

| model | backend | test | throughput |
|---|---|---|---:|
| `gemma4 26B.A4B Q4_0` | CUDA | `pp512` | 3021.76 +/- 34.41 tok/s |
| `gemma4 26B.A4B Q4_0` | CUDA | `tg128` | 77.35 +/- 0.13 tok/s |

Accuracy status under the same server:

- `scripts/gguf_logprobs_probe.py` still fails the lm-eval compatibility check.
- The response has `choices[0].logprobs.content`.
- The response still lacks `choices[0].logprobs.tokens` and `choices[0].logprobs.token_logprobs`.
- Therefore, serving is blessed for practical use, but GGUF paper-comparable lm-eval accuracy remains blocked.
