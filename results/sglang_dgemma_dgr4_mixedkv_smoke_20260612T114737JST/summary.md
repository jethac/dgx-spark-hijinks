# SGLang DiffusionGemma DG-R4 Mixed-KV Smoke

Status: RED

## Scope

DiffusionGemma 26B-A4B text-only serving through SGLang's conservative mixed-KV path: FP8-K + NVFP4-V, with the experimental FlashInfer VO-split opt-in for D=512 global layers. This is not a full NVFP4 K+V row.

## Provenance

- Run: `sglang_dgemma_dgr4_mixedkv_smoke_20260612T114737JST`
- Model: `google/diffusiongemma-26B-A4B-it`
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`
- SGLang: `dec4c040a8ede4561c1f26cccc599286643b49fd`
- FlashInfer: `f99323bd7d1c`
- Launch: `--dllm-algorithm Gemma4Renoise --dllm-algorithm-config dllm_config.yaml --attention-backend flashinfer --kv-cache-dtype fp4_e2m1 --dtype bfloat16 --page-size 256 --disable-cuda-graph --disable-piecewise-cuda-graph`
- Environment: `SGLANG_FLASHINFER_VOSPLIT=1`, `SGLANG_FP4_KV_MIXED_KV=1`, `SGLANG_GEMMA4_TRACE_GEOMETRY=1`, offline HF mode

## Gates

- Revised DG-R2 text quality gate: FAIL
- Opt-in policy warning present: PASS
- Mixed-KV backend warning present: PASS
- Server args prove `kv_cache_dtype='fp4_e2m1'`: PASS
- Pool configurator reports `mixed_kv=True`: PASS
- Hybrid subpools are `MHATokenToKVPoolFP4`: PASS
- D=512 geometry routes through VO-split trace labels: FAIL
- D=512 VO-split exposes `head_dim_vo=256`: FAIL

## Red Reasons

- revised text quality gate failed or is missing
- no D=512 geometry line with VO-split trace label

## Geometry Evidence


## Mixed-KV Evidence

- `[2026-06-12 02:54:07] SGLANG_GEMMA_KV_POOL_CONFIG full_layers=5 swa_layers=25 full_per_token_bytes=1600 swa_per_token_bytes=3200 swa_full_tokens_ratio=0.8 cell_size_bytes=72000.0 mixed_kv=True`
- `[2026-06-12 02:54:07] KV Cache is allocated. dtype: torch.float4_e2m1fn_x2, #tokens: 136448, K size: 6.52 GB, V size: 3.67 GB`
- `[2026-06-12 02:54:07] KV Cache is allocated. dtype: torch.float4_e2m1fn_x2, #tokens: 170752, K size: 0.82 GB, V size: 0.46 GB`
- `[2026-06-12 02:54:07] SGLANG_GEMMA_KV_SWAKVPOOL dtype=torch.float4_e2m1fn_x2 page_size=256 head_num=2 head_dim=512 full_layers=[5, 11, 17, 23, 29] swa_layers=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28] full_pool=MHATokenToKVPoolFP4 swa_pool=MHATokenToKVPoolFP4 full_tokens=170752 swa_tokens=136448 k_size_bytes=7874805760 v_size_bytes=4429578360`
- `[2026-06-12 02:54:08] SGLang FP4 KV mixed mode enabled: K cache uses FP8 e4m3, V cache uses packed NVFP4. Capacity claims must use the mixed-KV denominator, not full NVFP4 K+V.`
- `[2026-06-12 02:56:05] max_total_num_tokens=170752, chunked_prefill_size=-1, max_prefill_tokens=16384, max_running_requests=4096, context_len=8192, available_gpu_mem=47.34 GB`
