# vLLM Gemma 3 FlashInfer Paged-Prefill Debug Packet

Status: staged; waiting for GB10 host access.

Goal: distinguish a live generated-module / C++ binding mismatch from an FP4 V-fragment
kernel bug in the vLLM Gemma 3 NVFP4-KV failure.

## Patch Under Test

- FlashInfer fork: `jethac/flashinfer`
- Branch: `spark/hijinks-021-prefill-debug`
- Required commit: `1230341d` (`Add paged prefill JIT debug logging`)
- Changed files:
  - `csrc/batch_prefill.cu`
  - `csrc/batch_prefill_customize_config.jinja`

The patch is inactive unless `FLASHINFER_PREFILL_DEBUG_ONCE=1` is set. When enabled, each
generated FA2 batch-prefill module prints one C++ host-side diagnostic line for the first
ragged call and one for the first paged call in that module:

- generated dtype identity: `dtype_q`, `dtype_kv`, `dtype_o`, `idtype`
- `HEAD_DIM_QK`, `HEAD_DIM_VO`, `REQUIRE_FP4_KV_CACHE`, `USE_SLIDING_WINDOW`
- whether the compiled `DTypeKV` is the FP4x2 packed carrier type
- additional tensor/scalar names and dtypes from the JIT template
- runtime plan flags: layout, window, batch, heads, page size, split-KV, graph state
- tensor pointers, shapes, strides, DLPack dtype, and device id for `q`, `paged_k_cache`,
  `paged_v_cache`, `o`, index tensors, and optional LSE

## Why This Exists

The latest vLLM Gemma 3 evidence proves the failure is below Python-visible vLLM page
plumbing:

- write/read packed K/V and FP8 scale bytes match;
- short prompts fail without SWA eviction;
- fresh wrapper replay equals reused wrapper output;
- Python-visible plan/run signature matches offline replay;
- live `BatchPrefillWithPagedKVCacheWrapper.run(...)` returns byte-like BF16 values that
  match active packed V bytes, while CPU dequant replay is sane.

The next question is whether the live C++/generated module is actually the one we think it
is, and whether its runtime tensor identities match the offline replay. This patch answers
that before adding heavier kernel-side fragment dumps.

## Run Shape

Use the existing no-downgrade source-overlay Gemma 3 first-token packet, but mount this
FlashInfer source and set:

```bash
export FLASHINFER_PREFILL_DEBUG_ONCE=1
export FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a"
```

Clear only the relevant generated prefill modules before the rerun so the copied
`batch_prefill.cu` and rendered config are rebuilt:

```bash
rm -rf /root/.cache/flashinfer/0.6.13/121a/cached_ops/batch_prefill_with_kv_cache_*
rm -rf /root/.cache/flashinfer/0.6.13/121a/generated/batch_prefill_with_kv_cache_*
```

Then rerun the Gemma 3 NVFP4 first-token diagnostic from
`tasks/vllm_gemma3_wrapper_boundary_trace_packet_20260609.md` or the latest equivalent
source-overlay packet.

## Acceptance

Capture:

- full server log with `[flashinfer][prefill-debug]` lines;
- the existing wrapper-boundary / active-page trace artifacts;
- generated source path and cache directory from the server log if present;
- first-token output comparison against the fp8 row.

Green for this diagnostic is not model quality. Green means the log proves one of:

- live C++ identity and tensor identities match the expected FP4 paged-prefill module,
  pushing the next patch to FP4 V-fragment kernel instrumentation; or
- the live generated module/tensor identity is wrong, giving a concrete binding/JIT fix
  before touching kernel math.

## Expected Red Flags

- `REQUIRE_FP4_KV_CACHE=0` or `is_kv_fp4x2=0` on the failing NVFP4 paged path.
- missing `maybe_k_cache_sf,maybe_v_cache_sf` in `additional_tensors`.
- `paged_v_cache` pointer/shape/stride matching the packed carrier but `dtype_kv` not
  reporting the packed FP4x2 carrier.
- live runtime layout/head/page values diverging from the offline replay payload.
