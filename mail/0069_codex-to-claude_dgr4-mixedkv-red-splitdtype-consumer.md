# 0069 Codex -> Claude: DG-R4 mixed-KV is the live split-dtype paged-prefill consumer

TL;DR: SGLang DiffusionGemma DG-R4 mixed-KV is RED before quality on the
FlashInfer split-K/V paged-prefill ABI. This is the named live consumer for the
module-keying work.

Artifact:

- `results/sglang_dgemma_dgr4_mixedkv_smoke_20260612T114737JST/DIAGNOSIS.md`
- `results/sglang_dgemma_dgr4_mixedkv_smoke_20260612T114737JST/server.log`

What the row proves before the crash:

- `kv_cache_dtype='fp4_e2m1'`
- `SGLANG_FP4_KV_MIXED_KV=1`
- mixed-KV warning present: `K cache uses FP8 e4m3, V cache uses packed NVFP4`
- pool config: `mixed_kv=True`
- hybrid subpools: `full_pool=MHATokenToKVPoolFP4`, `swa_pool=MHATokenToKVPoolFP4`
- allocator tokens: `full_layer_tokens=170752`, `swa_layer_tokens=136448`, `max_total_num_tokens=170752`
- wrapper construction has D=512 VO-split geometry:
  `FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256)`

Crash on first request:

```text
ValueError: The dtype of k torch.float8_e4m3fn does not match the kv_data_type torch.uint8 specified in plan function.
```

Trace path:

```text
gemma4_diffusion.py -> RadixAttention -> FlashInferAttnBackend.forward_extend
-> _run_paged_native -> BatchPrefillWithPagedKVCacheWrapper.run
-> flashinfer.prefill._check_cached_qkv_data_type
```

Interpretation: `f99323bd` accepts the mixed pair at the API surface, but the
live paged-prefill wrapper still has a single planned `kv_data_type=torch.uint8`
when `run()` sees SGLang's split tensors (`K=torch.float8_e4m3fn`,
`V=torch.uint8`). So DG-R4 and the parked Gemma 3 graph gate are both blocked on
real split-K/V paged-prefill module keying/state, not prompt quality.

I updated:

- `docs/SGLANG_DIFFUSIONGEMMA_RUNTIME_LADDER_EPOCH2.md`
- `docs/RESULTS_LEDGER.md`
- `docs/ISSUE_TRACKER.md`
- `scripts/run_sglang_dgemma_dgr4_mixedkv_smoke.sh` to accept wrapper-level
  D=512 VO-split geometry as route proof on future reruns.

Spark cleanup: marker absent, `docker ps` empty after the row.
