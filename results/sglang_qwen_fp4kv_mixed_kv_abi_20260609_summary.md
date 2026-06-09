# SGLang Qwen FP4-KV Mixed-K/V ABI Review, 2026-06-09

Status: stop-point finding; no serving change landed.

## Context

The dense-reference partial-state probe showed that SGLang's default radix failure is not
page pairing, stale scale buffers, suffix attention, or `_safe_merge_state`. The cached
prefix path is internally consistent with the FP4 cache, but the FP4 cached-prefix
attention state is too far from the BF16 dense-prefix reference for the sampled Qwen row.

That makes a mixed policy attractive: keep K at higher precision to protect QK logits,
while storing V as NVFP4 for most of the capacity gain.

## Finding

The current FlashInfer FA2 paged-attention ABI does not expose independent K and V element
dtypes. It binds both sides to a single `DTypeKV`:

- `flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.plan()` caches one
  `kv_data_type`, and `run()` checks both `k_cache` and `v_cache` against that one type.
- `csrc/batch_prefill_customize_config.jinja` emits one `using DTypeKV = ...` and the
  paged params contain `paged_kv_t<DTypeKV, IdType>`.
- `include/flashinfer/page.cuh` defines `paged_kv_t<DType, IdType>` with separate K/V
  pointers and strides, but both pointers are `DType*`.
- `include/flashinfer/attention/prefill.cuh` consumes
  `const paged_kv_t<DTypeKV, IdType>& paged_kv`, so the generated kernel also sees one KV
  dtype for both K and V.
- SGLang's `MHATokenToKVPoolFP4` correspondingly allocates packed-uint8 K and V buffers
  plus FP8 scale buffers for both sides, and `FlashInferAttnBackend` passes
  `kv_cache_sf=(k_sf, v_sf)` with scalar `k_scale` and `v_scale` into the same FP4 wrapper.

Conclusion: FP8/BF16 K + NVFP4 V is not a small SGLang buffer-allocation change. It needs a
FlashInfer mixed-KV attention surface, e.g. independent K/V dtype template parameters plus
matching Python/JIT plumbing and SGLang pool integration.

## Capacity Implication

A naive higher-precision-K fallback should be reported with its capacity cost. For equal
head dimensions, FP8 K + NVFP4 V would be roughly:

`16 / (8 + 4.5) ~= 1.28x`

relative to fp8 K+V, versus about `1.78x` for NVFP4 K+V. That is still useful on GB10 if it
restores quality, but it is a different claim from the full NVFP4-KV capacity result.

## Next Useful Work

1. Decide whether to prototype a real FlashInfer mixed-KV FA2 path.
2. If not, keep searching for a better K-side FP4 policy, but tie it to serving quality,
   not just reconstruction or one offline attention cosine.
3. Do not rerun radix/page-pair/merge traces for this row unless new evidence contradicts
   the dense-reference partial-state result.
