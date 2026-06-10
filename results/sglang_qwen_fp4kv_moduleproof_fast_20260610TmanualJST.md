# SGLang Qwen FP4-KV Cached-Prefix Module Proof

Date: 2026-06-10 JST

Purpose: rerun the SGLang Qwen FP4-KV cached-prefix failure with explicit FlashInfer
module/env proof, after the standalone matrix showed that applying vLLM's
`FLASHINFER_PAGED_V_SF_DESWIZZLE` macro to SGLang-style linear V scale factors corrupts
FlashInfer FA2 NVFP4 reads.

Scope: SGLang only. No vLLM code was modified for this run.

## Run

- Runner: `scripts/run_sglang_fp4_dense_cache_trace.sh`
- Run id: `sglang_qwen_fp4kv_moduleproof_fast_20260610TmanualJST`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Attention backend: FlashInfer
- KV cache dtype: `fp4_e2m1`
- Page size: `1`
- Memory guardrail: Docker `--memory=100g --memory-swap=100g`
- Per-case install: disabled; SGLang source was loaded from the mounted checkout through
  `PYTHONPATH=/work/third_party/sglang/python`.

The first attempt with run id `sglang_qwen_fp4kv_moduleproof_20260610TmanualJST` was stopped
before server readiness because it was rebuilding native `sglang-kernel` artifacts that were
not needed for the Python-only trace hook.

## Serving Result

The cached-prefix failure reproduced.

| row | first request | cached_tokens | token | second request | cached_tokens | token |
|---|---|---:|---|---|---:|---|
| baseline_openai_then_native | OpenAI | 0 | `**` | native | 55 | `ark` |
| reverse_native_then_openai | native | 0 | `**` | OpenAI | 55 | `ark` |
| flush_between_openai_native | OpenAI | 0 | `**` | native | 0 | `**` |
| namespace_isolation_extra_key | OpenAI | 0 | `**` | native | 0 | `**` |

The trace comparator passed as an artifact and localized the first divergence to layer-0
attention output:

- cached field: `merged_rows`
- dense field: `o_rows`
- cosine: `-0.004055671821353563`
- max_abs: `0.31640625`

## Module/Flag Proof

The failing cached-prefix paged prefill path was:

- label: `extend_merge_paged`
- layer: `0`
- wrapper: `BatchPrefillWithPagedKVCacheWrapper`
- backend: `fa2`
- K/V carrier dtype: `torch.uint8`
- K/V scale dtype: `torch.float8_e4m3fn`
- K/V cache layout: `NHD`
- max cached prefix length in the paged wrapper: `55`

Most importantly:

- `extra_cuda_flags=''`
- `deswizzle_macro_active=False`
- `_jit_additional_tensor_names=[]`
- `_jit_additional_scalar_names=[]`

Representative server-log line:

```text
FP4 KV FlashInfer module trace label=extend_merge_paged layer=0 extra_cuda_flags='' deswizzle_macro_active=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_k_data_type': 'torch.uint8', '_cached_kv_data_type': 'torch.uint8', '_cached_o_data_type': 'torch.bfloat16', '_cached_q_data_type': 'torch.bfloat16', '_cached_v_data_type': 'torch.uint8', '_jit_additional_scalar_names': [], '_jit_additional_tensor_names': [], '_kv_layout': 'NHD', '_max_kv_len': 55, '_num_kv_heads': 2}}
```

## Conclusion

The deswizzle-leak hypothesis is falsified for this live SGLang failure. The standalone
FlashInfer matrix remains useful because it proves that deswizzle-on plus linear V scale
factors is a real corruption mechanism, but this SGLang cached-prefix failure occurred with
the deswizzle macro inactive.

Current narrowed state:

- page/data/scale pairing has already been cleared in prior SGLang traces;
- merge arithmetic has already been cleared in prior SGLang traces;
- deswizzle leakage is now cleared for the live failing row;
- the defect is still in SGLang's cached-prefix paged read or the scale/feed convention it
  hands to FlashInfer, not in a vLLM macro leak.

## Artifacts

- `results/sglang_qwen_fp4kv_moduleproof_fast_20260610TmanualJST_default.json`
- `results/sglang_qwen_fp4kv_moduleproof_fast_20260610TmanualJST_default_server.log`
- `results/sglang_qwen_fp4kv_moduleproof_fast_20260610TmanualJST_default_dense_cache_compare.json`
- `results/sglang_qwen_fp4kv_moduleproof_fast_20260610TmanualJST_summary.json`
- `results/sglang_qwen_fp4kv_moduleproof_fast_20260610TmanualJST_dense_cache_trace_summary_audit.json`
