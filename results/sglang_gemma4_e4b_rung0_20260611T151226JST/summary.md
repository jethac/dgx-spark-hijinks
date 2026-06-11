# SGLang Gemma 4 E4B Rung 0 Smoke

Status: RED

- Run: `sglang_gemma4_e4b_rung0_20260611T151226JST`
- Model: `google/gemma-4-E4B-it`
- SGLang commit: `9d78a007f`
- FlashInfer commit: `8d85fff9`
- VO split requested: `True`
- Geometry lines: `34`
- Wrapper geometry lines: `1`
- Binary proof lines present: `True`
- FlashInfer source paths present: `True`
- Request curl status: `28`
- Unsupported max_mma_kv: `True`
- Coherent Tokyo answer: `False`

## Result

The SGLang Gemma 4 E4B text-only smoke remains RED, but the SGLang-side D=512 decode routing is now proven.
SWA prefill plans at D=256, global prefill enters the two-pass VO-split path at D=512/VO=256, and global decode now reaches `decode_as_prefill_vosplit*` on `BatchPrefillWithPagedKVCacheWrapper`.
The remaining blocker is FlashInfer dispatcher selection inside the VO-split paged-prefill path: it still fails with `Unsupported max_mma_kv: 0`. This is the r9/`jethac/flashinfer@76af7982` dispatcher-fix target, not a standard decode-wrapper routing failure.

## Response

```json
{
  "parse_error": "JSONDecodeError('Expecting value: line 1 column 1 (char 0)')",
  "raw": ""
}
```

## Geometry Samples

### SWA prefill
- `[2026-06-11 06:15:40] SGLang Gemma4 FlashInfer geometry label=extend_paged layer=0 wrapper_id=False planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=256, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=256 sliding_window=512 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(1427463472), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xeede26150400>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xeede26150c20>`
- `[2026-06-11 06:15:43] SGLang Gemma4 FlashInfer geometry label=extend_paged layer=1 wrapper_id=False planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=256, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=256 sliding_window=512 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(1427463472), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xeede26150400>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xeede26150c20>`

### Global VO-split prefill
- `[2026-06-11 06:15:50] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(1427255920), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xeede26150b80>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xeede26`
- `[2026-06-11 06:15:50] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit1 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(1427255920), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xeede26150b80>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xeede26`

### Decode D=512
- `[2026-06-11 06:16:12] SGLang Gemma4 FlashInfer geometry label=decode_as_prefill_vosplit0 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(1427255920), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xeede26150b80>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xe`

