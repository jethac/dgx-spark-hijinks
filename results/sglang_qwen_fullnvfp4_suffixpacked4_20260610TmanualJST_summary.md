# SGLang Qwen full-NVFP4 radix retest: suffix-packed FP4 path

Date: 2026-06-10 JST

Run ID: `sglang_qwen_fullnvfp4_suffixpacked4_20260610TmanualJST`

Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`

Scope: Qwen/Qwen2.5-1.5B-Instruct, SGLang default radix cache ON, `kv_cache_dtype=fp4_e2m1`, full NVFP4 K+V. Single Spark/GB10 server under the 100 GiB Docker memory cap.

## What Changed

This run included the surgical SGLang-side wiring attempt:

- cached-prefix suffix attention now packs the suffix K/V tensors to NVFP4 and passes `kv_cache_sf`, `k_scale`, and `v_scale` into FlashInfer ragged `run_return_lse`;
- the ragged wrapper plan declares `kv_data_type=torch.uint8` only when the batch has cached-prefix tokens and the backend is full native NVFP4 K+V;
- fresh/no-prefix ragged prefill remains on the prior bf16 path;
- the runner request timeout is configurable via `REQUEST_TIMEOUT_S`; this run used `1800`.

Two earlier failed wiring attempts were resolved before this run:

- `suffixpacked2`: crashed because `forward_return_lse()` does not accept `kv_cache_sf`;
- `suffixpacked3`: crashed because no-prefix ragged prefill was planned as `uint8` while fresh K/V were still bf16.

## Result

This is **not green**.

The run completed without scheduler crashes, and the cache hit was real, but full NVFP4 K+V radix still changes the output on cached-prefix reuse:

| Row | Fresh/no-cache token | Cached-prefix token | Cached tokens |
|---|---:|---:|---:|
| baseline_openai_then_native | `**` | `ark` | 55 |
| reverse_native_then_openai | `**` | `ark` | 55 |
| flush_between_openai_native | `**` | `**` | 0 |
| namespace_isolation_extra_key | `**` | `**` | 0 |

The server log confirms the full FP4 K+V cache allocation:

- `dtype: torch.float4_e2m1fn_x2`
- `#tokens: 5553929`
- `K size: 20.86 GB`
- `V size: 20.86 GB`

The server log also confirms successful cached-prefix requests:

- `#new-token: 1`
- `#cached-token: 55`
- HTTP 200 for both `/generate` and `/v1/chat/completions` cached-prefix requests.

The dense-cache comparator completed, but its `ok=true` only means the artifact was produced. The actual quality signal is red:

- first divergence: layer 0 attention output
- cached rid: `native-second`
- dense rid: `openai-first`
- cosine: `0.012443760722381951`
- max_abs: `0.31640625`
- rms: `0.13541821805024212`

## Interpretation

The previous dtype/plumbing crashes are fixed, and the full-NVFP4 suffix path now executes. The result still reproduces the original cached-prefix corruption. This rules out the simple "suffix stayed bf16 while prefix was FP4" explanation as sufficient.

Given the earlier vLLM prefix-cache proof was green with full NVFP4 K+V, this remains most consistent with an SGLang-specific FP4-K scale/feed bug in the radix cached-prefix attention path, not an inherent FP4-K limitation.

## Artifacts

- `results/sglang_qwen_fullnvfp4_suffixpacked4_20260610TmanualJST_summary.json`
- `results/sglang_qwen_fullnvfp4_suffixpacked4_20260610TmanualJST_default.json`
- `results/sglang_qwen_fullnvfp4_suffixpacked4_20260610TmanualJST_default_dense_cache_compare.json`
- `results/sglang_qwen_fullnvfp4_suffixpacked4_20260610TmanualJST_default_server.log`

## Next Question For Review

Since vLLM proves full NVFP4 K+V prefix reuse can work, and SGLang still fails after suffix FP4 packing plus FP4 ragged dtype planning, the next useful review question is:

Where does SGLang's FP4-K scale feed into FlashInfer differ from vLLM's known-good path for cached-prefix attention?
