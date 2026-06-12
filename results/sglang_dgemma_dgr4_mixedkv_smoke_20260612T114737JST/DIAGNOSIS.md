# DG-R4 Mixed-KV Smoke Diagnosis

Status: RED, not a text-quality result.

The row proved the intended mixed-KV allocation and then failed on the first
DiffusionGemma request before the quality client could complete.

## Proven

- Server args include `kv_cache_dtype='fp4_e2m1'`.
- SGLang mixed-KV mode is active:
  `SGLang FP4 KV mixed mode enabled: K cache uses FP8 e4m3, V cache uses packed NVFP4`.
- Pool sizing used the mixed-KV denominator:
  `SGLANG_GEMMA_KV_POOL_CONFIG ... mixed_kv=True`.
- Hybrid SWA subpools are both `MHATokenToKVPoolFP4`.
- Capacity proof from the live allocator:
  `full_layer_tokens=170752`, `swa_layer_tokens=136448`,
  `max_total_num_tokens=170752`.
- Wrapper construction saw the intended D=512 VO-split geometry:
  `FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256)`.

## Blocker

The first paged native attention call crashed in FlashInfer prefill:

```text
ValueError: The dtype of k torch.float8_e4m3fn does not match the kv_data_type torch.uint8 specified in plan function.
```

Trace path:

```text
gemma4_diffusion.py -> RadixAttention -> FlashInferAttnBackend.forward_extend
-> _run_paged_native -> BatchPrefillWithPagedKVCacheWrapper.run
-> flashinfer.prefill._check_cached_qkv_data_type
```

Interpretation: this is the live DiffusionGemma consumer for the split-K/V
FlashInfer module-keying work. SGLang's mixed-KV pool supplies FP8 K and packed
NVFP4 V, but the paged-prefill wrapper state still carries a single
`kv_data_type=torch.uint8` into `run()`, so FlashInfer rejects the FP8 K tensor.

## Code-Level Follow-Up

The failure is not a missing SGLang plan kwarg:

- `FlashInferIndicesUpdaterPrefill.call_begin_forward()` builds
  `paged_plan_kwargs` with `kv_data_type=self.kv_data_type` and, when mixed-KV
  is active, also adds `k_data_type=self.k_data_type` and
  `v_data_type=self.v_data_type`
  (`third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`,
  around lines 3914-3930 in `jethac/sglang@dec4c040a`).
- In the FlashInfer fork, `BatchPrefillWithPagedKVCacheWrapper.plan()` accepts
  that validated mixed pair but collapses it back to `kv_data_type=torch.uint8`
  for module selection (`third_party/flashinfer/flashinfer/prefill.py`, around
  lines 1929-1963 / 3098-3132 in `jethac/flashinfer@f99323bd`).
- The wrapper then caches only `_cached_kv_data_type`, not separate cached
  K/V dtypes, and `run()` validates `k_cache.dtype` against that single cached
  dtype (`prefill.py` around lines 2101-2103 and 2380-2383 / 3218-3220 and
  3486-3488).
- The generated paged-prefill surface is still single-typed: the JIT URI has
  only `dtype_kv`, `batch_prefill.cu` instantiates one `DTypeKV`, and
  `paged_kv_t<DType, IdType>` stores both `k_data` and `v_data` as `DType*`.
  Bypassing the Python dtype check would therefore still feed FP8-K bytes to a
  kernel compiled for the packed-NVFP4 KV type.

Conclusion: this row is blocked on real FlashInfer split-K/V paged-prefill
state/keying/params (`DTypeK` + `DTypeV` or an equivalent dedicated mixed-KV
module), not on SGLang wrapper plumbing.

The generated `summary.md` also marks D=512 route proof as missing because the
first request crashed before per-call `SGLang Gemma4 FlashInfer geometry
label=...` lines were emitted. Wrapper-construction route proof is present in
`server.log`; the actionable RED reason is the split-dtype paged-prefill ABI
mismatch above.

## Next

Do not tune prompts or quality gates for this row. Fix or consume the
FlashInfer/SGLang split-K/V paged-prefill module key so a mixed pair
`K=torch.float8_e4m3fn`, `V=torch.uint8` is planned and run as split dtype, then
rerun `scripts/run_sglang_dgemma_dgr4_mixedkv_smoke.sh`.
