# llama.cpp Native Loglikelihood Live Probe

Date: 2026-06-08

Run id: `llamacpp_native_loglikelihood_20260608T1331JST`

This run tested whether llama.cpp native `/tokenize` plus `/completion` endpoints can support lm-eval-style GGUF loglikelihood scoring for arbitrary continuation tokens.

## Setup

- binary: `/home/jethac/src/llama.cpp-b9536/build/bin/llama-server`
- model: `/home/jethac/models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf`
- alias: `qwen25-loglikelihood`
- server URL: `http://127.0.0.1:18083`
- repo commit: `3abc49036bedb37e0b5e3a62365b2d1cdb6ee370`
- `n_probs`: `512`

The server log confirms CUDA on `NVIDIA GB10` with `CUDA : ARCHS = 1210`, `USE_GRAPHS = 1`, and `BLACKWELL_NATIVE_FP4 = 1`.

## Artifacts

- `results/llamacpp_native_loglikelihood_20260608T1331JST_probe.json`
- `results/llamacpp_native_loglikelihood_20260608T1331JST_task.json`
- `results/llamacpp_native_loglikelihood_20260608T1331JST_server.log`
- `results/llamacpp_native_loglikelihood_20260608T1331JST_build_target_audit.json`
- `results/llamacpp_native_loglikelihood_20260608T1331JST_run_info.txt`

## Result

The native task did not pass:

- total task rows: `3`
- target continuations found: `2`
- target continuations missing: `1`
- expected-greedy mismatches: `0`
- task `ok`: `false`

The likely continuations were scored:

- `tokyo_likely`: found, logprob sum `-1.73`, greedy `true`
- `qwen_name`: found, logprob sum `-0.62`, greedy `true`

The unlikely continuation was not scored:

- `zebra_unlikely`: target token not present in the returned top-512 probability entries

## Interpretation

This is live server proof, but it is a negative proof for the current adapter path. llama.cpp native endpoints expose top-N probabilities for the next generated token; at `n_probs=512`, they do not expose every arbitrary continuation token needed by lm-eval loglikelihood.

The GGUF accuracy lane remains blocked until one of these is true:

- llama.cpp exposes direct logprobs for supplied continuation tokens,
- the adapter can request full-vocabulary probabilities at an acceptable cost,
- or a different native API path can score arbitrary tokens without relying on top-N membership.

This does not affect practical llama.cpp serving, which remains blessed for the tested Q4 GGUF rows.
