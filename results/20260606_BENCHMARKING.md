# 20260606 Gemma 4 Benchmarking

## Executive Summary

This report is generated from the current PGX benchmark artifacts. Rows are kept in separate classes for paper-comparable accuracy, loader compatibility, throughput, and MTP speed.

## Current Status

- Manifest accuracy rows: 152
- Manifest MTP rows: 248
- Smoke row statuses: 152 ({'loader_failed': 120, 'ok': 21, 'eval_failed': 11})
- Smoke task records: 1596
- Full eval task records: 70 ({'ok': 65, 'eval_failed': 5})
- Full eval rows complete across all selected tasks: 9
- Throughput row statuses: 2 ({'ok': 2})
- MTP speed row statuses: 1 ({'ok': 1})
- Pending smoke rows: 0
- Pending full eval rows: 12
- Pending throughput rows: 150
- Pending MTP rows: 247

## Versions And Sources

- vLLM: `v0.22.1`
- llama.cpp stock: `b9536`
- llama.cpp MTP: PR `23398` checkout
- PyTorch: vLLM dependency set
- lm-eval-harness: source checkout recorded in the environment logs
- Source links are recorded in `BENCHMARK_PLAN.md`; exact Hugging Face revisions are in `manifest.json` where available.

## Validation Notes

- HF/safetensors validation passed for `google-e2b-baseline-bf16` using vLLM `0.22.1` with `enforce_eager=True`.
- HF fallback is allowed and labeled when vLLM cannot load a safetensors QAT row.
- GGUF download and llama.cpp serving work, but lm-eval's GGUF adapter is incompatible with the pinned llama.cpp `/v1/completions` logprobs schema for paper-comparable loglikelihood scoring.
- GGUF accuracy rows are therefore compatibility/failure rows unless a proper loglikelihood path becomes available.
- GGUF throughput works through stock llama.cpp `llama-bench`.
- MTP validation works through llama.cpp PR `23398`; draft acceptance rate is recorded as `not_reported` when llama.cpp does not expose it.

## Smoke Summary

| status | count |
|---|---:|
| eval_failed | 11 |
| loader_failed | 120 |
| ok | 21 |

### Smoke Backend Counts

| status | count |
|---|---:|
| hf | 5 |
| llama.cpp | 120 |
| vllm | 27 |

### Smoke Limit Counts

| status | count |
|---|---:|
| 10 | 16 |
| 100 | 136 |

### Smoke Mode Counts

| status | count |
|---|---:|
| fast | 1 |
| legacy-full | 136 |
| load-probe | 15 |

### Smoke Row Preview

| row | status | backend | limit | mode | format | quant | reason |
|---|---|---|---|---|---|---|---|
| google-12b-qat-q4-0-unquantized-bf16 | eval_failed | vllm | 10 | load-probe | safetensors | bf16 | returncode=1 |
| google-12b-qat-w4a16-w4a16 | eval_failed | vllm | 10 | load-probe | safetensors | w4a16 | returncode=1 |
| google-e2b-qat-mobile-mobile-ct | eval_failed | vllm | 100 |  | safetensors | mobile-ct | returncode=1 |
| google-e2b-qat-mobile-mobile-transformers | eval_failed | vllm | 100 |  | safetensors | mobile-transformers | returncode=1 |
| google-e2b-qat-q4-0-unquantized-bf16 | eval_failed | hf | 100 |  | safetensors | bf16 | returncode=-9 |
| google-e4b-qat-mobile-mobile-ct | eval_failed | vllm | 100 |  | safetensors | mobile-ct | returncode=1 |
| google-e4b-qat-mobile-mobile-transformers | eval_failed | vllm | 100 |  | safetensors | mobile-transformers | returncode=1 |
| google-e4b-qat-q4-0-unquantized-bf16 | eval_failed | hf | 100 |  | safetensors | bf16 | returncode=-9 |
| unsloth-12b-baseline-bf16 | eval_failed | vllm | 10 | load-probe | safetensors | bf16 | returncode=1 |
| unsloth-12b-qat-q4-0-unquantized-bf16 | eval_failed | vllm | 10 | load-probe | safetensors | bf16 | returncode=1 |
| unsloth-12b-qat-w4a16-w4a16 | eval_failed | vllm | 10 | load-probe | safetensors | w4a16 | returncode=1 |
| google-12b-qat-q4_0-q4_0-gemma-4-12b-it-qat-q4_0.gguf | loader_failed | llama.cpp | 100 |  | gguf | q4_0 | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| google-26b-a4b-qat-q4_0-q4_0-gemma-4-26b_q4_0-it.gguf | loader_failed | llama.cpp | 100 |  | gguf | q4_0 | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| google-31b-qat-q4_0-q4_0-gemma-4-31b_q4_0-it.gguf | loader_failed | llama.cpp | 100 |  | gguf | q4_0 | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| google-e2b-qat-q4_0-q4_0-gemma-4-e2b_q4_0-it.gguf | loader_failed | llama.cpp | 100 |  | gguf | q4_0 | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| google-e4b-qat-q4_0-q4_0-gemma-4-e4b_q4_0-it.gguf | loader_failed | llama.cpp | 100 |  | gguf | q4_0 | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-bf16-gemma-4-12b-it-bf16.gguf | loader_failed | llama.cpp | 100 |  | gguf | BF16 | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-iq4_nl-gemma-4-12b-it-iq4_nl.gguf | loader_failed | llama.cpp | 100 |  | gguf | IQ4_NL | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-iq4_xs-gemma-4-12b-it-iq4_xs.gguf | loader_failed | llama.cpp | 100 |  | gguf | IQ4_XS | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-q3_k_m-gemma-4-12b-it-q3_k_m.gguf | loader_failed | llama.cpp | 100 |  | gguf | Q3_K_M | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-q3_k_s-gemma-4-12b-it-q3_k_s.gguf | loader_failed | llama.cpp | 100 |  | gguf | Q3_K_S | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-q4_0-gemma-4-12b-it-q4_0.gguf | loader_failed | llama.cpp | 100 |  | gguf | Q4_0 | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-q4_1-gemma-4-12b-it-q4_1.gguf | loader_failed | llama.cpp | 100 |  | gguf | Q4_1 | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-q4_k_m-gemma-4-12b-it-q4_k_m.gguf | loader_failed | llama.cpp | 100 |  | gguf | Q4_K_M | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-q4_k_s-gemma-4-12b-it-q4_k_s.gguf | loader_failed | llama.cpp | 100 |  | gguf | Q4_K_S | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-q5_k_m-gemma-4-12b-it-q5_k_m.gguf | loader_failed | llama.cpp | 100 |  | gguf | Q5_K_M | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-q5_k_s-gemma-4-12b-it-q5_k_s.gguf | loader_failed | llama.cpp | 100 |  | gguf | Q5_K_S | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-q6_k-gemma-4-12b-it-q6_k.gguf | loader_failed | llama.cpp | 100 |  | gguf | Q6_K | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-q8_0-gemma-4-12b-it-q8_0.gguf | loader_failed | llama.cpp | 100 |  | gguf | Q8_0 | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-ud-iq2_m-gemma-4-12b-it-ud-iq2_m.gguf | loader_failed | llama.cpp | 100 |  | gguf | UD-IQ2_M | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-ud-iq3_xxs-gemma-4-12b-it-ud-iq3_xxs.gguf | loader_failed | llama.cpp | 100 |  | gguf | UD-IQ3_XXS | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-ud-q2_k_xl-gemma-4-12b-it-ud-q2_k_xl.gguf | loader_failed | llama.cpp | 100 |  | gguf | UD-Q2_K_XL | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-ud-q3_k_xl-gemma-4-12b-it-ud-q3_k_xl.gguf | loader_failed | llama.cpp | 100 |  | gguf | UD-Q3_K_XL | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-ud-q4_k_xl-gemma-4-12b-it-ud-q4_k_xl.gguf | loader_failed | llama.cpp | 100 |  | gguf | UD-Q4_K_XL | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-ud-q5_k_xl-gemma-4-12b-it-ud-q5_k_xl.gguf | loader_failed | llama.cpp | 100 |  | gguf | UD-Q5_K_XL | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-ud-q6_k_xl-gemma-4-12b-it-ud-q6_k_xl.gguf | loader_failed | llama.cpp | 100 |  | gguf | UD-Q6_K_XL | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-baseline-gguf-ud-q8_k_xl-gemma-4-12b-it-ud-q8_k_xl.gguf | loader_failed | llama.cpp | 100 |  | gguf | UD-Q8_K_XL | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-12b-qat-gguf-ud-q4_k_xl-gemma-4-12b-it-qat-ud-q4_k_xl.gguf | loader_failed | llama.cpp | 100 |  | gguf | UD-Q4_K_XL | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-26b-a4b-baseline-gguf-bf16-bf16-gemma-4-26b-a4b-it-bf16-00001-of-00002.gguf | loader_failed | llama.cpp | 100 |  | gguf | BF16 | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |
| unsloth-26b-a4b-baseline-gguf-bf16-bf16-gemma-4-26b-a4b-it-bf16-00002-of-00002.gguf | loader_failed | llama.cpp | 100 |  | gguf | BF16 | lm-eval gguf adapter requires echoed prompt/continuation token_logprobs, but llama.cpp b9536 /v1/completions returns current content-style logprobs for generated tokens only |

_Preview limited to 40 rows; see raw JSONL artifacts for all rows._

## Full Eval Summary

| status | count |
|---|---:|
| eval_failed | 5 |
| ok | 65 |

### Full Eval Record Preview

| row | task | shots | backend | metric | value | stderr | status | elapsed_s |
|---|---|---|---|---|---|---|---|---|
| google-12b-baseline-bf16 | arc_challenge | 25 | hf |  |  |  | eval_failed | 351.63 |
| google-12b-baseline-bf16 | arc_easy | 0 | hf | acc_norm | 0.29 | 0.01 | ok | 1225.34 |
| google-12b-baseline-bf16 | boolq | 0 | hf | acc | 0.56 | 0.01 | ok | 1225.34 |
| google-12b-baseline-bf16 | piqa | 0 | hf | acc | 0.54 | 0.01 | ok | 1225.34 |
| google-26b-a4b-baseline-bf16 | arc_challenge | 25 | vllm | acc_norm | 0.41 | 0.01 | ok | 1558.66 |
| google-26b-a4b-baseline-bf16 | arc_easy | 0 | vllm | acc_norm | 0.36 | 0.01 | ok | 933.04 |
| google-26b-a4b-baseline-bf16 | boolq | 0 | vllm | acc | 0.69 | 0.01 | ok | 933.04 |
| google-26b-a4b-baseline-bf16 | hellaswag | 10 | vllm | acc_norm | 0.49 | 0.00 | ok | 9017.44 |
| google-26b-a4b-baseline-bf16 | piqa | 0 | vllm | acc | 0.58 | 0.01 | ok | 933.04 |
| google-26b-a4b-baseline-bf16 | winogrande | 5 | vllm | acc | 0.55 | 0.01 | ok | 507.68 |
| google-e2b-baseline-bf16 | arc_challenge | 25 | vllm | acc_norm | 0.33 | 0.01 | ok | 873.62 |
| google-e2b-baseline-bf16 | arc_easy | 0 | vllm | acc_norm | 0.30 | 0.01 | ok | 442.21 |
| google-e2b-baseline-bf16 | boolq | 0 | vllm | acc | 0.57 | 0.01 | ok | 442.21 |
| google-e2b-baseline-bf16 | hellaswag | 10 | vllm | acc_norm | 0.35 | 0.00 | ok | 5472.27 |
| google-e2b-baseline-bf16 | piqa | 0 | vllm | acc | 0.56 | 0.01 | ok | 442.21 |
| google-e2b-baseline-bf16 | winogrande | 5 | vllm | acc | 0.50 | 0.01 | ok | 235.12 |
| google-e2b-qat-w4a16-w4a16 | arc_challenge | 25 | vllm | acc_norm | 0.44 | 0.01 | ok | 908.41 |
| google-e2b-qat-w4a16-w4a16 | arc_easy | 0 | vllm | acc_norm | 0.32 | 0.01 | ok | 454.42 |
| google-e2b-qat-w4a16-w4a16 | boolq | 0 | vllm | acc | 0.48 | 0.01 | ok | 454.42 |
| google-e2b-qat-w4a16-w4a16 | hellaswag | 10 | vllm | acc_norm | 0.46 | 0.00 | ok | 5740.91 |
| google-e2b-qat-w4a16-w4a16 | piqa | 0 | vllm | acc | 0.58 | 0.01 | ok | 454.42 |
| google-e2b-qat-w4a16-w4a16 | winogrande | 5 | vllm | acc | 0.51 | 0.01 | ok | 214.49 |
| google-e4b-baseline-bf16 | arc_challenge | 25 | vllm | acc_norm | 0.39 | 0.01 | ok | 1250.18 |
| google-e4b-baseline-bf16 | arc_easy | 0 | vllm | acc_norm | 0.34 | 0.01 | ok | 629.19 |
| google-e4b-baseline-bf16 | boolq | 0 | vllm | acc | 0.69 | 0.01 | ok | 629.19 |
| google-e4b-baseline-bf16 | hellaswag | 10 | vllm | acc_norm | 0.47 | 0.00 | ok | 7889.78 |
| google-e4b-baseline-bf16 | piqa | 0 | vllm | acc | 0.58 | 0.01 | ok | 629.19 |
| google-e4b-baseline-bf16 | winogrande | 5 | vllm | acc | 0.52 | 0.01 | ok | 283.90 |
| google-e4b-qat-w4a16-w4a16 | arc_challenge | 25 | vllm | acc_norm | 0.39 | 0.01 | ok | 1268.17 |
| google-e4b-qat-w4a16-w4a16 | arc_easy | 0 | vllm | acc_norm | 0.35 | 0.01 | ok | 618.96 |
| google-e4b-qat-w4a16-w4a16 | boolq | 0 | vllm | acc | 0.70 | 0.01 | ok | 618.96 |
| google-e4b-qat-w4a16-w4a16 | hellaswag | 10 | vllm | acc_norm | 0.49 | 0.00 | ok | 8401.20 |
| google-e4b-qat-w4a16-w4a16 | piqa | 0 | vllm | acc | 0.59 | 0.01 | ok | 618.96 |
| google-e4b-qat-w4a16-w4a16 | winogrande | 5 | vllm | acc | 0.53 | 0.01 | ok | 262.59 |
| unsloth-26b-a4b-baseline-bf16 | arc_challenge | 25 | vllm | acc_norm | 0.41 | 0.01 | ok | 1535.97 |
| unsloth-26b-a4b-baseline-bf16 | arc_easy | 0 | vllm | acc_norm | 0.36 | 0.01 | ok | 857.72 |
| unsloth-26b-a4b-baseline-bf16 | boolq | 0 | vllm | acc | 0.69 | 0.01 | ok | 857.72 |
| unsloth-26b-a4b-baseline-bf16 | piqa | 0 | vllm | acc | 0.58 | 0.01 | ok | 857.72 |
| unsloth-26b-a4b-baseline-bf16 | winogrande | 5 | vllm | acc | 0.55 | 0.01 | ok | 490.72 |
| unsloth-e2b-baseline-bf16 | arc_challenge | 25 | vllm | acc_norm | 0.33 | 0.01 | ok | 871.27 |
| unsloth-e2b-baseline-bf16 | arc_easy | 0 | vllm | acc_norm | 0.30 | 0.01 | ok | 446.17 |
| unsloth-e2b-baseline-bf16 | boolq | 0 | vllm | acc | 0.57 | 0.01 | ok | 446.17 |
| unsloth-e2b-baseline-bf16 | hellaswag | 10 | vllm | acc_norm | 0.35 | 0.00 | ok | 5436.23 |
| unsloth-e2b-baseline-bf16 | piqa | 0 | vllm | acc | 0.56 | 0.01 | ok | 446.17 |
| unsloth-e2b-baseline-bf16 | winogrande | 5 | vllm | acc | 0.50 | 0.01 | ok | 235.68 |
| unsloth-e2b-qat-q4-0-unquantized-bf16 | arc_challenge | 25 | hf |  |  |  | eval_failed | 304.90 |
| unsloth-e2b-qat-q4-0-unquantized-bf16 | arc_easy | 0 | hf | acc_norm | 0.31 | 0.01 | ok | 546.55 |
| unsloth-e2b-qat-q4-0-unquantized-bf16 | boolq | 0 | hf | acc | 0.51 | 0.01 | ok | 546.55 |
| unsloth-e2b-qat-q4-0-unquantized-bf16 | piqa | 0 | hf | acc | 0.57 | 0.01 | ok | 546.55 |
| unsloth-e2b-qat-w4a16-w4a16 | arc_challenge | 25 | vllm | acc_norm | 0.44 | 0.01 | ok | 896.54 |
| unsloth-e2b-qat-w4a16-w4a16 | arc_easy | 0 | vllm | acc_norm | 0.32 | 0.01 | ok | 442.02 |
| unsloth-e2b-qat-w4a16-w4a16 | boolq | 0 | vllm | acc | 0.47 | 0.01 | ok | 442.02 |
| unsloth-e2b-qat-w4a16-w4a16 | hellaswag | 10 | vllm | acc_norm | 0.46 | 0.00 | ok | 5760.02 |
| unsloth-e2b-qat-w4a16-w4a16 | piqa | 0 | vllm | acc | 0.57 | 0.01 | ok | 442.02 |
| unsloth-e2b-qat-w4a16-w4a16 | winogrande | 5 | vllm | acc | 0.51 | 0.01 | ok | 207.43 |
| unsloth-e4b-baseline-bf16 | arc_challenge | 25 | vllm | acc_norm | 0.39 | 0.01 | ok | 1245.18 |
| unsloth-e4b-baseline-bf16 | arc_easy | 0 | vllm | acc_norm | 0.34 | 0.01 | ok | 642.01 |
| unsloth-e4b-baseline-bf16 | boolq | 0 | vllm | acc | 0.69 | 0.01 | ok | 642.01 |
| unsloth-e4b-baseline-bf16 | hellaswag | 10 | vllm | acc_norm | 0.47 | 0.00 | ok | 7860.28 |
| unsloth-e4b-baseline-bf16 | piqa | 0 | vllm | acc | 0.58 | 0.01 | ok | 642.01 |

_Preview limited to 60 rows; see raw JSONL artifacts for all rows._

## Throughput Summary

| status | count |
|---|---:|
| ok | 2 |

### Throughput Rows

| row | format | backend | prompt tok/s | generation tok/s | status | error |
|---|---|---|---|---|---|---|
| google-e2b-qat-q4_0-q4_0-gemma-4-e2b_q4_0-it.gguf | gguf | llama.cpp-b9536 | 3923.63 | 122.11 | ok |  |
| google-e2b-baseline-bf16 | safetensors | vllm | 506.98 | 130.17 | ok |  |

## MTP Speed Summary

| status | count |
|---|---:|
| ok | 1 |

### MTP Rows

| row | size | main quant | mtp quant | prompt tok/s | generation tok/s | acceptance | status |
|---|---|---|---|---|---|---|---|
| mtp-12b-ud-iq2_m-q8_0-gemma-4-12b-it-ud-iq2_m.gguf | 12B | UD-IQ2_M | Q8_0 | 1042.40 | 36.40 | not_reported | ok |

## Comparability Caveats

- Paper-comparable accuracy is limited to `lm-eval-harness` loglikelihood rows.
- vLLM, HF fallback, stock llama.cpp, and llama.cpp MTP are distinct inference paths and should not be conflated.
- Throughput prompts are fixed, cheap, task-derived prompts and are not the official accuracy examples.
- Loader failures and incompatibilities are part of the matrix, not missing data.

## Artifacts

- Plan: `/home/jethac/gemma4-evals/BENCHMARK_PLAN.md`
- Manifest: `/home/jethac/gemma4-evals/manifest.json`
- Smoke JSONL: `/home/jethac/gemma4-evals/results/smoke_results.jsonl`
- Smoke summary: `/home/jethac/gemma4-evals/results/smoke_summary.json`
- Full JSONL: `/home/jethac/gemma4-evals/results/full_results.jsonl`
- Throughput JSONL: `/home/jethac/gemma4-evals/results/throughput_results.jsonl`
- MTP JSONL: `/home/jethac/gemma4-evals/results/mtp_speed_results.jsonl`
- Logs: `/home/jethac/gemma4-evals/logs`
