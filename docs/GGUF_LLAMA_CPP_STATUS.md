# GGUF llama.cpp Status

Status: practical serving works; paper-comparable lm-eval accuracy is blocked; native
NVFP4 build/emission and first runtime dispatch are proven, but correctness and matched
speed are untested.

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

## Current Accuracy Frontier

The native non-OpenAI `/completion` top-N route has now been tested on the live pinned
`b9536` server with `n_probs=512`. It scored likely continuations, but missed the unlikely
`zebra` continuation, so top-N probabilities are not enough for lm-eval-style arbitrary
continuation scoring.

The next bounded proof is one newer llama.cpp server pin with the OpenAI echo-span probe.
Pass only if prompt `tokens` and `token_logprobs` cover the supplied continuation token ids
for the `zebra` case. If the newer pin still returns only generated-token
`choices[0].logprobs.content`, stop tuning top-N and move to full-vocabulary practicality
or a `jethac/llama.cpp` supplied-token loglikelihood endpoint.

Contract packet: `tasks/llamacpp_supplied_token_loglikelihood_contract_20260609.md`.
This is the row-8 acceptance shape: context plus continuation in, continuation token ids,
exact per-token logprobs, summed logprob, and greedy-match boolean out. A green artifact
must score the unlikely `" zebra"` continuation; top-N membership alone remains red.

## Historical Native Top-N Probe

The older candidate was llama.cpp's native non-OpenAI `/completion` API plus `/tokenize`, not reshaping the observed `/v1/completions` response.

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

The probe now also accepts explicit context/continuation pairs and emits an lm-eval-shaped tuple:

```bash
python3 scripts/llamacpp_native_loglikelihood_probe.py \
  --url http://127.0.0.1:8080 \
  --n-probs 512 \
  --pair 'The capital of Japan is||| Tokyo' \
  --pair 'The capital of Japan is||| zebra' \
  --output results/llamacpp_native_loglikelihood_probe_pairs.json
```

For each pair, `target_logprob_sum` is the sum over continuation-token logprobs and `all_tokens_greedy` is the greedy-match boolean needed by an lm-eval-style loglikelihood adapter. Artifact `results/llamacpp_native_loglikelihood_probe_v2_selftest_20260608.json` proves the parser/classifier logic locally, but it is not live server proof.

The follow-on tiny task harness is ready:

```bash
python3 scripts/llamacpp_native_loglikelihood_task.py \
  --url http://127.0.0.1:8080 \
  --n-probs 512 \
  --input tasks/llamacpp_loglikelihood_smoke.jsonl \
  --output results/llamacpp_native_loglikelihood_task.json
```

Dry-run artifact `results/llamacpp_native_loglikelihood_task_dryrun_20260608.json` proves the JSONL task file and command shape. The first live server attempt is recorded in `results/llamacpp_native_loglikelihood_20260608T1331JST_summary.md`.

Live result:

- binary: `/home/jethac/src/llama.cpp-b9536/build/bin/llama-server`
- model: `qwen2.5-1.5b-instruct-q4_k_m.gguf`
- `n_probs`: `512`
- task artifact: `results/llamacpp_native_loglikelihood_20260608T1331JST_task.json`
- result: `ok=false`; likely continuations were scored, but the unlikely `zebra` continuation was not present in the returned top-512 probabilities

Interpretation: the native endpoint top-N path is not sufficient for lm-eval-style arbitrary continuation scoring at this setting. The next GGUF accuracy fix needs direct supplied-token logprobs, full-vocabulary probabilities, or another native scoring path.

If the native endpoint cannot return arbitrary target-token logprobs, this needs a llama.cpp upstream endpoint or a different accuracy backend.

For Qwen speed, the next serving proof is separate from the accuracy adapter: run a Qwen3/Qwen3.6-class instruct GGUF with the same `b9536` CUDA build and row recorder. The existing Qwen2.5 1.5B Q4_K_M row proves practical small-Qwen serving, but the counterpart matrix still requires a larger Qwen3/Qwen3.6 GGUF row. Native FP4 GGUF remains blocked until an actual NVFP4/MXFP4 GGUF artifact is available and the runtime dispatch evidence proves that path, not just a Q4_K/Q4_0 model on a build compiled with Blackwell FP4 support.

## 2026-06-08 Native FP4 Arch Probe

Artifact: `results/llamacpp_native_fp4_arch_20260608T164917JST_summary.md`.

This row set up `jethac/llama.cpp` as `third_party/llama.cpp` and pinned branch
`spark/native-fp4-sm121-20260608` at
`19bba67c1f4db723c60a0d421aa0788bf4ddc699`.

Build matrix result on the GB10 CUDA 13.0 host:

| requested arch | result | emitted target | evidence |
|---|---|---|---|
| `121a` | configure/build ok | `sm_121a` | `2592` `mxf4nvf4.block_scale.scale_vec::4X` PTX hits |
| `121` | configure/build ok | rewritten to `sm_121a` | same block-scale PTX evidence |
| `120f` | configure failed | none | CMake rejected the arch value before CUDA compilation |

Interpretation: llama.cpp can build and emit native block-scale FP4 PTX for `sm_121a`
under this toolkit. This arch artifact alone is not runtime proof, and no
output/correctness or PP/TG speed row should cite it as more than build/emission evidence.

## 2026-06-08 Native FP4 Runtime Gate

Artifact: `results/llamacpp_nvfp4_runtime_gate_20260608T1748JST_summary.md`.

This advances the native FP4 lane from build/emission to first runtime dispatch evidence:
the pinned `jethac/llama.cpp@19bba67c1` build loaded a converted Qwen3.6 NVFP4 GGUF on
GB10, returned a small reasoning-off chat smoke, and Nsight Systems reported
`GGML_TYPE_NVFP4` matmul activity. Correctness versus BF16/Q8 and matched speed remain
unproven.

Next packet: `tasks/llamacpp_nvfp4_correctness_speed_packet_20260608.md`.

Run the packet only on the Linux GB10 host with the prior runtime-gate artifacts and a
same-lineage BF16 or Q8_0 reference GGUF. This workspace did not have the required
`/home/jethac/...` build/model paths available for a live correctness probe.

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

## 2026-06-08 Echo Logprobs Probe

Target:

- binary: `/home/jethac/src/llama.cpp-b9536/build/bin/llama-server`
- model: `/home/jethac/models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf`
- probe script: `scripts/gguf_logprobs_probe.py`
- prompt: `The capital of Japan is zebra`

Artifacts:

- `results/llamacpp_gguf_echo_logprobs_probe_20260608_max0.json`
- `results/llamacpp_gguf_echo_logprobs_probe_20260608_max1.json`
- `results/llamacpp_gguf_echo_logprobs_probe_20260608_summary.json`
- `results/llamacpp_gguf_echo_logprobs_probe_20260608_server.log`

Probe behavior:

- The script tokenizes context and continuation separately.
- Context token ids: `[785, 6722, 315, 6323, 374]`.
- Continuation token ids: `[1147, 50213]`.
- The OpenAI request sends `context + continuation` with `echo=true`.
- The row passes only if prompt `tokens` plus `token_logprobs` cover the supplied continuation span.

Result:

- `max_tokens=0`: `ok=false`.
- `max_tokens=1`: `ok=false`.
- Both responses expose `choices[0].logprobs.content` for a generated token (`-striped`), not prompt `tokens`/`token_logprobs`.
- Therefore pinned `b9536` still cannot provide exact supplied-continuation logprobs through the tested OpenAI echo path.

Next decision:

- Try one newer llama.cpp server pin with the same `scripts/gguf_logprobs_probe.py` echo-span check.
- Pass condition: returned prompt-token logprobs cover the supplied continuation tokens `[1147, 50213]`.
- If the newest pin still exposes generated-token `logprobs.content` only, stop probing stock endpoints and use a `jethac/llama.cpp` endpoint fork for direct supplied-token loglikelihood.
