# vLLM Gemma 3 27B Rung 1 NVFP4-KV Trace, 2026-06-09

Status: red quality result, useful localization result.

## Stack

- Host: GB10 / compute capability 12.1.
- Model: `google/gemma-3-27b-it`, served as `gemma3-27b-it`.
- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`.
- vLLM source overlay: `jethac/vllm@e2a8197a9c8b67172aa909463c58f6e447ad2bba`.
- FlashInfer source overlay: `jethac/flashinfer@e41016fcd121986aea923d5c7e68fc9f152d2a07`.
- Important env: `VLLM_SPARK_KV_TRACE=1`, `TORCH_CUDA_ARCH_LIST=12.1a`,
  `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`.

## Artifacts

- fp8 server log:
  `results/vllm_gemma3_27b_rung1_20260608T2351JST_fp8_flashinfer_server.log`
- fp8 first-token report:
  `results/vllm_gemma3_27b_rung1_20260608T2351JST_fp8_flashinfer_first_token.json`
- low-limit NVFP4 server log:
  `results/vllm_gemma3_27b_rung1_20260608T2351JST_nvfp4_kv_flashinfer_server.log`
- high-limit NVFP4 server log:
  `results/vllm_gemma3_27b_rung1_20260609T0015JST_nvfp4_kv_flashinfer_server.log`
- high-limit NVFP4 first-token report:
  `results/vllm_gemma3_27b_rung1_20260609T0015JST_nvfp4_kv_flashinfer_first_token.json`
- high-limit comparison:
  `results/vllm_gemma3_27b_rung1_20260609T0015JST_first_token_compare.json`
- high-limit trace analysis:
  `results/vllm_gemma3_27b_rung1_20260609T0015JST_trace_analysis.json`
- high-limit raw trace, local only unless force-added:
  `results/vllm_gemma3_27b_rung1_20260609T0015JST_nvfp4_kv_flashinfer_kv_trace.jsonl`

The raw JSON/JSONL artifacts are ignored by default repo policy. The summary values below
come from `vllm_gemma3_27b_rung1_20260609T0015JST_trace_analysis.json`.

## Result

The fp8 row remains the quality comparator. Its first-token probes returned the expected
first tokens for all three cases.

The NVFP4-KV row still fails immediately on the first generated token. Against the fp8
baseline, all three probe cases have `0.0` top-logprob overlap:

| case | fp8 first token | NVFP4-KV first token | overlap |
|---|---|---|---|
| `exact_spark_ok` | `spark` | Cyrillic text | `0.0` |
| `simple_math` | `4` | `Liv` | `0.0` |
| `short_decode` | `A` | Malayalam text | `0.0` |

The first low-limit trace used `VLLM_SPARK_KV_TRACE_LIMIT=8`; that limit was consumed by
warmup/graph-capture events and only sampled `-1` slot/page IDs. Treat that run as a
negative lesson for the trace packet, not as page-pair evidence.

The high-limit rerun used enough trace budget to sample real client requests:

```text
fi_metadata: 234
kv_write_pre: 558
kv_write_post_nvfp4: 558
kv_read_views_nvfp4: 234
swa_skip: 90
write_samples: 195
read_samples: 195
matched_read_samples: 195
mismatched_read_samples: 0
read_samples_missing_write: 0
request_swa_skip_events: 18
max_request_num_skipped_tokens: 0
first_token_compare_ok: false
```

## Interpretation

This clears the simplest cache-management hypothesis for the sampled path: read-side
packed K/V bytes and FP8 scale bytes matched write-side samples for the same physical
slots/pages in all `195 / 195` sampled read events.

It also clears SWA eviction/window rotation for the failing first-token prompts:
`max_request_num_skipped_tokens=0`. These prompts fail before sliding-window eviction is
needed.

Rung 1 is still red: Gemma 3 27B with vLLM NVFP4-KV has the capacity/routing machinery,
but not correct output. The next probe should move above physical page lifecycle and into
tensor-level comparison: local/global attention outputs, NVFP4 quant/dequant numerical
error, V-scale deswizzle content, and logits before sampler preprocessing.
