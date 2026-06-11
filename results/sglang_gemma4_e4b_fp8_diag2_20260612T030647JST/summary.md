# SGLang Gemma 4 E4B Smoke

Status: RED

- Run: `sglang_gemma4_e4b_fp8_diag2_20260612T030647JST`
- Model: `google/gemma-4-E4B-it`
- SGLang commit: `651d55cd2e6a3d90de0eb65af643d0aa4ee7fca2`
- FlashInfer commit: `f99323bd7d1cc88d9445202c12934070be754e2d`
- VO split requested: `True`
- Geometry lines: `6`
- Wrapper geometry lines: `1`
- Binary proof lines present: `True`
- FlashInfer source paths present: `True`
- Request curl status: `28`
- Unsupported max_mma_kv: `False`
- Coherent Tokyo answer: `False`

## Result

See response and geometry samples below.

## Response

```json
{
  "parse_error": "JSONDecodeError('Expecting value: line 1 column 1 (char 0)')",
  "raw": ""
}
```

## Geometry Samples

### SWA prefill
- `[2026-06-11 18:09:59] SGLang Gemma4 FlashInfer geometry label=extend_paged layer=0 wrapper_id=False planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=256, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=256 sliding_window=512 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.float8_e4m3fn', '_cached_module': 'namespace(plan=Function(2074444880), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xe025bef74b80>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xe025bef7`
- `[2026-06-11 18:10:02] SGLang Gemma4 FlashInfer geometry label=extend_paged layer=1 wrapper_id=False planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=256, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=256 sliding_window=512 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.float8_e4m3fn', '_cached_module': 'namespace(plan=Function(2074444880), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xe025bef74b80>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xe025bef7`

### Global VO-split prefill
- `[2026-06-11 18:10:09] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.float8_e4m3fn', '_cached_module': 'namespace(plan=Function(2074255296), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xe025bef76c00>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xe`

### Decode D=512
- MISSING

