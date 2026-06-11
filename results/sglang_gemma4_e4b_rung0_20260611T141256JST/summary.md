# SGLang Gemma 4 E4B Rung 0 Smoke

Status: RED

- Run: `sglang_gemma4_e4b_rung0_20260611T141256JST`
- Model: `google/gemma-4-E4B-it`
- SGLang commit: `f3ebcf623`
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

The SGLang Gemma 4 E4B text-only smoke remains RED after wrapper geometry was corrected.
SWA prefill plans at D=256 and global prefill enters the two-pass VO-split path at D=512/VO=256.
The remaining blocker is decode: the D=512 global layer still enters the standard decode wrapper, which instantiates a D=512/VO=512 paged module and fails in FlashInfer with `Unsupported max_mma_kv: 0`.

## Response

```json
{
  "parse_error": "JSONDecodeError('Expecting value: line 1 column 1 (char 0)')",
  "raw": ""
}
```

## Geometry Samples

### SWA prefill
- `[2026-06-11 05:15:58] SGLang Gemma4 FlashInfer geometry label=extend_paged layer=0 wrapper_id=False planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=256, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=256 sliding_window=512 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(2244188272), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xf200a8d604a0>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xf200a8d60cc0>`
- `[2026-06-11 05:16:02] SGLang Gemma4 FlashInfer geometry label=extend_paged layer=1 wrapper_id=False planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=256, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=256 sliding_window=512 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(2244188272), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xf200a8d604a0>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xf200a8d60cc0>`

### Global VO-split prefill
- `[2026-06-11 05:16:09] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(2243257104), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xf200a8d60c20>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xf200a8`
- `[2026-06-11 05:16:09] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit1 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(2243257104), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xf200a8d60c20>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xf200a8`

### Decode D=512
- `[2026-06-11 05:16:30] SGLang Gemma4 FlashInfer geometry label=decode layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchDecodeWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(3016102656), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xf200a96889a0>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xf200a968a840>)', '_cac`

