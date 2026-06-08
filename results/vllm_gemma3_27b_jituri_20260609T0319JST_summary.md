# vLLM Gemma 3 27B NVFP4-KV JIT URI Rerun, 2026-06-09

## Result

Red. The FlashInfer paged-prefill JIT URI patch works as an experiment, but it does
not fix Gemma 3 27B NVFP4-KV quality.

The server reached readiness and the first-token probe completed after disabling
FlashInfer sampling with `VLLM_USE_FLASHINFER_SAMPLER=0`. That sampler override was only
to avoid a side-path source-overlay failure; FlashInfer attention and NVFP4-KV remained
selected.

## Exact Row

```text
run: vllm_gemma3_27b_jituri_20260609T0319JST_nvfp4_kv_flashinfer_eager
model: google/gemma-3-27b-it
vLLM: 0.1.dev1+g13da71884
FlashInfer: 0.6.9rc1 installed Python ABI + jethac/flashinfer source JIT overlay
device: NVIDIA GB10, compute capability 12.1
torch: 2.12.0.dev20260408+cu130
CUDA: 13.0
kv_cache_dtype: nvfp4
attention backend: FlashInfer
sampler: FlashInfer sampler disabled for this rerun
```

Import probe:

```text
prefill_uri=batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_fp4x2_e2m1_dtype_o_bf16_dtype_idx_i32_head_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_True_f16qk_False
prefill_uri_has_fp4x2=true
prefill_uri_has_u8=false
```

Server log also shows a fresh `121a` JIT build under the `dtype_kv_fp4x2_e2m1`
batch-prefill namespace.

## Quality

The probe completed but still generated bad first tokens for the simple comparator
prompts:

```text
exact_spark_ok -> "Stephanie"
simple_math    -> "ilacion"
short_decode   -> " Kiara"
```

Top-logprob lists remain dominated by unrelated multilingual tokens rather than the fp8
comparator's normal answers, so the Gemma 3 NVFP4-KV row remains quality-red.

## Tensor Evidence

The tensor trace preserves the earlier failure signature. Nonzero
`flashinfer_wrapper_prefill_post` / `flashinfer_attn_output` events still produce
byte-like BF16 outputs:

```text
layer 0 prefill out_after: min=0.00 max=255.00 mean=130.06 rms=148.83
layer 5 prefill out_after: min=0.00 max=255.00 mean=127.73 rms=146.76
layer 0 prefill out_after: min=0.00 max=255.00 mean=126.61 rms=145.23
layer 5 prefill out_after: min=0.02 max=255.00 mean=128.94 rms=147.44
layer 0 prefill out_after: min=0.00 max=255.00 mean=123.72 rms=143.84
layer 5 prefill out_after: min=0.00 max=255.00 mean=129.28 rms=147.77
```

Those ranges are not plausible signed BF16 attention outputs. They match the prior
wrapper-boundary and active-page rows: the real FlashInfer paged-prefill wrapper is still
returning packed-byte-shaped values as BF16.

## Side Finding

The first attempt with the same URI patch failed before readiness because the source
overlay caused FlashInfer sampling to JIT from `/flashinfer-src`, and the top-p sampling
kernel failed on GB10 with:

```text
TopPSamplingFromProbs failed with error code too many resources requested for launch
```

This was bypassed with `VLLM_USE_FLASHINFER_SAMPLER=0` for the accepted rerun. It is a
source-overlay side effect, not evidence about the attention failure.

## Conclusion

The stale `dtype_kv_u8` JIT-cache/module-name hypothesis is falsified for this live row.
The failing kernel was rebuilt under an explicit `dtype_kv_fp4x2_e2m1` namespace and the
quality/tensor corruption remained.

Next fix should move into FlashInfer's paged-prefill FP4-KV read/convert path, especially
the path that turns packed `__nv_fp4x2_e2m1` plus FP8 scale factors into BF16 values for
the attention computation. The working offline active-page replay remains the reference:
the exact pages dequantize to sane signed causal-attention output outside the wrapper.

## Artifacts

- `results/vllm_gemma3_27b_jituri_20260609T0319JST_nvfp4_kv_flashinfer_eager_import_probe.txt`
- `results/vllm_gemma3_27b_jituri_20260609T0319JST_nvfp4_kv_flashinfer_eager_first_token.json`
- `results/vllm_gemma3_27b_jituri_20260609T0319JST_nvfp4_kv_flashinfer_eager_tensor_trace.jsonl`
- `results/vllm_gemma3_27b_jituri_20260609T0319JST_nvfp4_kv_flashinfer_eager_server.log`
- `results/vllm_gemma3_27b_jituri_20260609T0319JST_nvfp4_kv_flashinfer_eager_active_page_dump/`
