# vLLM Gemma 3 27B Wrapper-Boundary Trace, 2026-06-09

## Purpose

Localize the Gemma 3 27B NVFP4-KV first-token quality failure at the real vLLM /
FlashInfer wrapper boundary after:

- write/read KV byte pairing was sampled clean in
  `results/vllm_gemma3_27b_rung1_trace_20260609T0015JST_summary.md`;
- tensor tracing localized the strongest corruption to `flashinfer_attn_output` in
  `results/vllm_gemma3_27b_tensor_trace_20260609T0115JST_summary.md`;
- standalone signed Gemma-shaped FlashInfer FA2 probes passed in
  `results/vllm_flashinfer_gemma3_attention_output_probe_20260609T0134JST_summary.md`.

This is a diagnostic-only eager run, not a benchmark row.

## Stack

- Model: `google/gemma-3-27b-it`
- Served name: `gemma3-27b-it`
- vLLM fork: `jethac/vllm@0e07e130d94eddfed209f846ce6c9959c636da02`
  (`spark/hijinks-021-gemma3-tensor-trace`)
- Base image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- Source overlay: `/vllm-src` plus local FlashInfer source overlay
- Precompiled wheel base: `4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa`
- CUDA target env: `TORCH_CUDA_ARCH_LIST=12.1a`
- FlashInfer env: `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`
- vLLM flags: `--attention-backend flashinfer --kv-cache-dtype nvfp4 --dtype bfloat16
  --max-model-len 131072 --gpu-memory-utilization 0.85
  --max-num-batched-tokens 4096 --enforce-eager`

Server log proof:

- vLLM selected `kv_cache_dtype=nvfp4`.
- Running model geometry logged `head_dim=128`, `num_heads=32`, `num_kv_heads=16`.
- Layer 0 is local/SWA (`sliding_window=1024`); layer 5 is full/global
  (`sliding_window=None`).
- KV cache size: `1,800,549` tokens; max concurrency at 131,072 tokens:
  `13.74x`.
- vLLM logged: `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM
  V-scale-factor deswizzle enabled.`

## First-Token Result

The diagnostic reproduces the same bad first-token behavior as the prior NVFP4-KV rows:

| case | first token |
|---|---|
| `exact_spark_ok` | ` Reigns` |
| `simple_math` | Gujarati text |
| `short_decode` | `ioane` |

This matches the earlier failure signature and confirms the wrapper trace is attached to
the failing path.

## Wrapper-Boundary Finding

The failing boundary is inside the real FlashInfer FA2 paged prefill wrapper call.

For the `short_decode` request at 24 prompt tokens, the full/global layer wrapper event
records:

- `wrapper_type=BatchPrefillWithPagedKVCacheWrapper`
- `window_left=-1`
- `num_prefill_tokens=24`
- `num_decode_tokens=0`
- `kv_cache_dtype=nvfp4`
- `is_kvcache_nvfp4=true`
- `use_fa2_nvfp4_kv=true`
- `needs_fp8_out=false`
- `k_scale=1.0`
- `v_scale=1.0`

Before `wrapper.run(...)`, the actual model query and output buffer are sane signed BF16:

- `query_last`: min `-9.5625`, max `9.6875`, mean `0.011817073449492455`,
  RMS `2.315901041030884`
- `out_before`: head `[0.0, -1.9609375, 0.0, -0.24609375, 0.0, 1.234375,
  0.0, 1.7578125]`, min `-12.0`, max `10.0`, mean `-0.0018197707831859589`,
  RMS `1.6632423400878906`

Immediately after `wrapper.run(...)`, the BF16 output is byte-like:

- `out_after`: head `[240.0, 1.7265625, 226.0, 137.0, 145.0, 20.0, 186.0, 185.0]`
- min `3.453237695794087e-12`
- max `255.0`
- mean `129.27871704101562`
- RMS `147.77296447753906`

The subsequent `flashinfer_attn_output` event matches this corrupted tensor, so the
corruption has already happened before the Gemma residual, MLP, final hidden state, or
logits path.

The same pattern also appears in local/SWA prefill calls (`window_left=1023`). Therefore
this short-prompt failure is not explained by SWA window eviction or long-context block
rotation.

## KV Layout Clue

The KV trace confirms the expected packed layout:

- packed K/V data views are `torch.uint8`, shape `[pages, 16, 16, 64]`;
- K/V scale views are `torch.float8_e4m3fn`, shape `[pages, 16, 16, 8]`;
- the physical page layout is `[K_data | K_scale | V_data | V_scale]`.

For the same 24-token full/global layer event, valid slot samples exist after padding is
ignored. The active layer-5 sample includes:

- page `29`, slot `464`
- `v_data_head`: `[240, 1, 226, 137, 145, 20, 186, 185, 33, 65, 47, 233,
  91, 34, 145, 25]`
- `v_scale_head`: `[53, 51, 52, 53, 54, 55, 56, 58]`

The first bytes of `v_data_head` line up with the byte-like `out_after` head from the
wrapper. That is a strong clue that the real paged wrapper/kernel path is treating or
returning packed V payload bytes as BF16 attention output, or is otherwise consuming the
wrong V view for this vLLM/Gemma NVFP4-KV call.

The all-zero samples in early events are padding: their `slot_mapping_head` entries are
`-1`. They should not be interpreted as active cache pages containing zeros.

## Conclusion

Gemma 3 27B NVFP4-KV is still red, but the bug is now localized more tightly:

- not generic standalone FlashInfer FA2 signed E2M1 math;
- not non-unit global K/V scales in the standalone harness;
- not the V-scale deswizzle standalone path;
- not sampled write/read page or scale/data pairing;
- not SWA eviction for these short first-token prompts;
- **yes: the real vLLM/Gemma FlashInfer FA2 paged prefill wrapper returns byte-like
  output, apparently aligned with packed V payload bytes, before Gemma consumes the
  attention result.**

## Next Action

Trace or reproduce the exact paged prefill call with active block-table pages, not page
zero or padded slots:

1. Dump the wrapper arguments needed to replay the failing layer-5 prefill call:
   query, active block tables, sequence lengths, split K/V data views, K/V scale views,
   `k_scale`, `v_scale`, page size, layout, and output tensor metadata.
2. Build a minimal replay that compares FlashInfer wrapper output with a dequantized
   reference for the active pages.
3. Check whether the FlashInfer paged prefill specialization is selecting the intended
   FP4 reader for V or is reading the packed uint8 V data as values.
4. Keep Gemma 4 31B and SGLang Gemma blocked until Gemma 3 Rung 1 produces correct
   output with NVFP4 KV.

## Artifacts

- `results/vllm_gemma3_27b_wrapper_trace_20260609T0148JST_nvfp4_kv_flashinfer_eager_first_token.json`
- `results/vllm_gemma3_27b_wrapper_trace_20260609T0148JST_nvfp4_kv_flashinfer_eager_tensor_trace.jsonl`
- `results/vllm_gemma3_27b_wrapper_trace_20260609T0148JST_nvfp4_kv_flashinfer_eager_kv_trace.jsonl`
- `results/vllm_gemma3_27b_wrapper_trace_20260609T0148JST_nvfp4_kv_flashinfer_eager_server.log`
- `results/vllm_gemma3_27b_wrapper_trace_20260609T0148JST_nvfp4_kv_flashinfer_eager_editable_install.log`
- `results/vllm_gemma3_27b_wrapper_trace_20260609T0148JST_nvfp4_kv_flashinfer_eager_import_probe.txt`
