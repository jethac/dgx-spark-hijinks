# SGLang DiffusionGemma DG-R3 VO-Split Smoke

Status: GREEN

## Scope

BF16/no-KV-quant DiffusionGemma 26B-A4B text-only serving through the experimental SGLang FlashInfer VO-split opt-in. This is not an NVFP4 KV or capacity row.

## Provenance

- Run: `sglang_dgemma_dgr5_bf16_capacity_baseline_20260612T150750JST`
- Model: `google/diffusiongemma-26B-A4B-it`
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`
- SGLang: `98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968`
- FlashInfer: `f99323bd7d1c`
- Launch: `--dllm-algorithm Gemma4Renoise --dllm-algorithm-config dllm_config.yaml --attention-backend flashinfer --dtype bfloat16 --page-size 256 --disable-cuda-graph --disable-piecewise-cuda-graph`
- Environment: `SGLANG_FLASHINFER_VOSPLIT=1`, `SGLANG_GEMMA4_TRACE_GEOMETRY=1`, offline HF mode

## Gates

- Revised DG-R2 text quality gate: PASS
- Opt-in policy warning present: PASS
- D=512 geometry routes through VO-split trace labels: PASS
- D=512 VO-split exposes `head_dim_vo=256`: PASS

## Geometry Evidence

- `[2026-06-12 06:15:37] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(1662669984), ragged_ru`
- `[2026-06-12 06:15:37] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit1 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(1662669984), ragged_ru`
- `[2026-06-12 06:15:37] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=11 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(1662669984), ragged_r`
- `[2026-06-12 06:15:37] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit1 layer=11 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(1662669984), ragged_r`
- `[2026-06-12 06:15:37] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=17 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(1662669984), ragged_r`
- `[2026-06-12 06:15:37] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit1 layer=17 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(1662669984), ragged_r`
- `[2026-06-12 06:15:37] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=23 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(1662669984), ragged_r`
- `[2026-06-12 06:15:37] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit1 layer=23 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=16, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=16 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(1662669984), ragged_r`

## Quality Checks

- `capital_japan_direct`: stable=True non_empty=True answer_ok=True text='The capital of Japan is **Tokyo**.'
- `arithmetic_2_plus_2_direct`: stable=True non_empty=True answer_ok=True text='2 + 2 = 4'
- `dgx_spark_use`: stable=True non_empty=True answer_ok=True text='The NVIDIA DGX Spark desktop is designed for high-performance AI development, prototyping, and deep learning tasks in a compact, desktop form factor.'
