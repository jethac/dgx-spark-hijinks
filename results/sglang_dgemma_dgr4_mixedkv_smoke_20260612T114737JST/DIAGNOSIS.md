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
