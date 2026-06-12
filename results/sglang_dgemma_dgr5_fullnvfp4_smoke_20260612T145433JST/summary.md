# SGLang DiffusionGemma DG-R5 Full-NVFP4 Smoke

Status: GREEN

## Scope

DiffusionGemma 26B-A4B text-only serving through full NVFP4 K+V KV storage, with the experimental FlashInfer VO-split opt-in for D=512 global layers. This is a smoke/quality/routing row; capacity requires a separate matched denominator row.

## Provenance

- Run: `sglang_dgemma_dgr5_fullnvfp4_smoke_20260612T145433JST`
- Model: `google/diffusiongemma-26B-A4B-it`
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`
- SGLang: `98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968`
- FlashInfer: `f99323bd7d1c`
- Launch: `--dllm-algorithm Gemma4Renoise --dllm-algorithm-config dllm_config.yaml --attention-backend flashinfer --kv-cache-dtype fp4_e2m1 --dtype bfloat16 --page-size 256 --disable-cuda-graph --disable-piecewise-cuda-graph`
- Environment: `SGLANG_FLASHINFER_VOSPLIT=1`, `SGLANG_FP4_KV_MIXED_KV=0`, `SGLANG_GEMMA4_TRACE_GEOMETRY=1`, offline HF mode

## Gates

- Revised DG-R2 text quality gate: PASS
- Opt-in policy warning present: PASS
- Mixed-KV backend warning absent: PASS
- Server args prove `kv_cache_dtype='fp4_e2m1'`: PASS
- Pool configurator reports `mixed_kv=False`: PASS
- Hybrid subpools are `MHATokenToKVPoolFP4`: PASS
- NVFP4 calibration or FlashInfer FP4 trace evidence present: PASS
- D=512 geometry routes through VO-split trace labels or wrapper geometry: PASS
- D=512 VO-split exposes `head_dim_vo=256`: PASS

## Geometry Evidence

- `[2026-06-12 06:02:55] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359470624), ragged_run=<`
- `[2026-06-12 06:02:55] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit1 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359470624), ragged_run=<`
- `[2026-06-12 06:02:55] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=11 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359470624), ragged_run=`
- `[2026-06-12 06:02:55] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit1 layer=11 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359470624), ragged_run=`
- `[2026-06-12 06:02:56] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=17 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359470624), ragged_run=`
- `[2026-06-12 06:02:56] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit1 layer=17 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359470624), ragged_run=`
- `[2026-06-12 06:02:56] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=23 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359470624), ragged_run=`
- `[2026-06-12 06:02:56] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit1 layer=23 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359470624), ragged_run=`

## Wrapper Geometry Evidence

- `[2026-06-12 06:00:51] SGLang FlashInfer wrapper geometries dispatch=WrapperDispatch.SLIDING_WINDOW geometries=[FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=8, head_dim=256, head_dim_vo=256), FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256)]`

## Full-NVFP4 Evidence

- `[2026-06-12 06:00:48] SGLANG_GEMMA_KV_POOL_CONFIG full_layers=5 swa_layers=25 full_per_token_bytes=1152 swa_per_token_bytes=2304 swa_full_tokens_ratio=0.8 cell_size_bytes=51840.0 mixed_kv=False`
- `[2026-06-12 06:00:51] KV Cache is allocated. dtype: torch.float4_e2m1fn_x2, #tokens: 189696, K size: 5.09 GB, V size: 5.09 GB`
- `[2026-06-12 06:00:51] KV Cache is allocated. dtype: torch.float4_e2m1fn_x2, #tokens: 237312, K size: 0.64 GB, V size: 0.64 GB`
- `[2026-06-12 06:00:51] SGLANG_GEMMA_KV_SWAKVPOOL dtype=torch.float4_e2m1fn_x2 page_size=256 head_num=2 head_dim=512 full_layers=[5, 11, 17, 23, 29] swa_layers=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28] full_pool=MHATokenToKVPoolFP4 swa_pool=MHATokenToKVPoolFP4 full_tokens=237312 swa_tokens=189696 k_size_bytes=6154813560 v_size_bytes=6154813560`
- `[2026-06-12 06:02:50] FP4 KV FlashInfer module trace label=extend_paged layer=0 extra_cuda_flags='-gencode=arch=compute_121a,code=sm_121a' deswizzle_macro_active=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359240736), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xfe520d888900>, paged_run=<function get_batch_prefill_module.<locals>.paged_run a`
- `[2026-06-12 06:02:51] FP4 KV FlashInfer module trace label=extend_paged layer=1 extra_cuda_flags='-gencode=arch=compute_121a,code=sm_121a' deswizzle_macro_active=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359240736), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xfe520d888900>, paged_run=<function get_batch_prefill_module.<locals>.paged_run a`
- `[2026-06-12 06:02:51] FP4 KV FlashInfer module trace label=extend_paged layer=2 extra_cuda_flags='-gencode=arch=compute_121a,code=sm_121a' deswizzle_macro_active=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359240736), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xfe520d888900>, paged_run=<function get_batch_prefill_module.<locals>.paged_run a`
- `[2026-06-12 06:02:51] FP4 KV FlashInfer module trace label=extend_paged layer=3 extra_cuda_flags='-gencode=arch=compute_121a,code=sm_121a' deswizzle_macro_active=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359240736), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xfe520d888900>, paged_run=<function get_batch_prefill_module.<locals>.paged_run a`
- `[2026-06-12 06:02:51] FP4 KV FlashInfer module trace label=extend_paged layer=4 extra_cuda_flags='-gencode=arch=compute_121a,code=sm_121a' deswizzle_macro_active=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359240736), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xfe520d888900>, paged_run=<function get_batch_prefill_module.<locals>.paged_run a`
- `[2026-06-12 06:02:55] FP4 KV FlashInfer module trace label=extend_paged_vosplit0 layer=5 extra_cuda_flags='-gencode=arch=compute_121a,code=sm_121a' deswizzle_macro_active=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359470624), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xfe520d8889a0>, paged_run=<function get_batch_prefill_module.<locals>.pa`
- `[2026-06-12 06:02:55] FP4 KV FlashInfer module trace label=extend_paged_vosplit1 layer=5 extra_cuda_flags='-gencode=arch=compute_121a,code=sm_121a' deswizzle_macro_active=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359470624), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xfe520d8889a0>, paged_run=<function get_batch_prefill_module.<locals>.pa`
- `[2026-06-12 06:02:55] FP4 KV FlashInfer module trace label=extend_paged layer=6 extra_cuda_flags='-gencode=arch=compute_121a,code=sm_121a' deswizzle_macro_active=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359240736), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xfe520d888900>, paged_run=<function get_batch_prefill_module.<locals>.paged_run a`
- `[2026-06-12 06:02:55] FP4 KV FlashInfer module trace label=extend_paged layer=7 extra_cuda_flags='-gencode=arch=compute_121a,code=sm_121a' deswizzle_macro_active=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359240736), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xfe520d888900>, paged_run=<function get_batch_prefill_module.<locals>.paged_run a`
- `[2026-06-12 06:02:55] FP4 KV FlashInfer module trace label=extend_paged layer=8 extra_cuda_flags='-gencode=arch=compute_121a,code=sm_121a' deswizzle_macro_active=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359240736), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xfe520d888900>, paged_run=<function get_batch_prefill_module.<locals>.paged_run a`
- `[2026-06-12 06:02:55] FP4 KV FlashInfer module trace label=extend_paged layer=9 extra_cuda_flags='-gencode=arch=compute_121a,code=sm_121a' deswizzle_macro_active=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359240736), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xfe520d888900>, paged_run=<function get_batch_prefill_module.<locals>.paged_run a`
- `[2026-06-12 06:02:55] FP4 KV FlashInfer module trace label=extend_paged layer=10 extra_cuda_flags='-gencode=arch=compute_121a,code=sm_121a' deswizzle_macro_active=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1359240736), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xfe520d888900>, paged_run=<function get_batch_prefill_module.<locals>.paged_run `

## Quality Checks

- `capital_japan_direct`: stable=True non_empty=True answer_ok=True text='The capital of Japan is **Tokyo**.'
- `arithmetic_2_plus_2_direct`: stable=True non_empty=True answer_ok=True text='2 + 2 = 4'
- `dgx_spark_use`: stable=True non_empty=True answer_ok=True text='The NVIDIA DGX Spark is designed for high-performance local AI development, testing, and prototyping of deep learning models in a compact desktop form factor.'
