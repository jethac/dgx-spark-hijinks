# SGLang Gemma 4 E4B Rung 0 Smoke

Status: RED

- Run: `sglang_gemma4_e4b_rung1_fp8_retry1_20260611T183110JST`
- Model: `google/gemma-4-E4B-it`
- SGLang commit: `9d78a007f`
- FlashInfer commit: `76af798243d11c4910eaceaf1d62ba4227656d4a`
- VO split requested: `True`
- Geometry lines: `6`
- Wrapper geometry lines: `1`
- Binary proof lines present: `True`
- FlashInfer source paths present: `True`
- Request curl status: `0`
- Unsupported max_mma_kv: `False`
- Coherent Tokyo answer: `False`

## Result

See response and geometry samples below.

## Response

```json
{
  "parse_error": "JSONDecodeError('Expecting value: line 1 column 1 (char 0)')",
  "raw": "Internal Server Error"
}
```

## Geometry Samples

### SWA prefill
- `[2026-06-11 09:34:12] SGLang Gemma4 FlashInfer geometry label=extend_paged layer=0 wrapper_id=False planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=256, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=256 sliding_window=512 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.float8_e4m3fn', '_cached_module': 'namespace(plan=Function(2253969584), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xe0e9ceea4680>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xe0e9ceea`
- `[2026-06-11 09:34:16] SGLang Gemma4 FlashInfer geometry label=extend_paged layer=1 wrapper_id=False planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=256, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=256 sliding_window=512 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.float8_e4m3fn', '_cached_module': 'namespace(plan=Function(2253969584), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xe0e9ceea4680>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xe0e9ceea`

### Global VO-split prefill
- `[2026-06-11 09:34:23] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.float8_e4m3fn', '_cached_module': 'namespace(plan=Function(2254198192), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xe0e9ceea4e00>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xe`

### Decode D=512
- MISSING

