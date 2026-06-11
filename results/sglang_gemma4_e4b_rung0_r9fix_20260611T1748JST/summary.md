# SGLang Gemma 4 E4B Rung 0 Smoke

Status: RED

- Run: `sglang_gemma4_e4b_rung0_r9fix_20260611T1748JST`
- Model: `google/gemma-4-E4B-it`
- SGLang commit: `9d78a007f`
- FlashInfer commit: `76af798243d11c4910eaceaf1d62ba4227656d4a`
- VO split requested: `True`
- Geometry lines: `56`
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
  "text": "\n\n---\n\n---\n\n---\n\n---\n\n---\n\n---\n\n---\n\n---",
  "output_ids": [
    108,
    7243,
    108,
    7243,
    108,
    7243,
    108,
    7243,
    108,
    7243,
    108,
    7243,
    108,
    7243,
    108,
    7243
  ],
  "meta_info": {
    "id": "39ee7c05dba7491fb97d0c0899e56993",
    "finish_reason": {
      "type": "length",
      "length": 16
    },
    "prompt_tokens": 11,
    "weight_version": "default",
    "num_retractions": 0,
    "reasoning_tokens": 0,
    "completion_tokens": 16,
    "cached_tokens": 0,
    "cached_tokens_details": null,
    "dp_rank": null,
    "e2e_latency": 74.27359105402138,
    "response_sent_to_client_ts": 1781167822.7474675
  }
}
```

## Geometry Samples

### SWA prefill
- `[2026-06-11 08:49:50] SGLang Gemma4 FlashInfer geometry label=extend_paged layer=0 wrapper_id=False planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=256, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=256 sliding_window=512 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(2190920720), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xeaa7993b85e0>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xeaa7993b8e00>`
- `[2026-06-11 08:49:54] SGLang Gemma4 FlashInfer geometry label=extend_paged layer=1 wrapper_id=False planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=256, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=256 sliding_window=512 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(2190920720), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xeaa7993b85e0>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xeaa7993b8e00>`

### Global VO-split prefill
- `[2026-06-11 08:50:01] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(2189142976), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xeaa7993b8d60>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xeaa799`
- `[2026-06-11 08:50:01] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit1 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(2189142976), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xeaa7993b8d60>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xeaa799`

### Decode D=512
- `[2026-06-11 08:50:21] SGLang Gemma4 FlashInfer geometry label=decode_as_prefill_vosplit0 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(2189142976), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xeaa7993b8d60>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xe`
- `[2026-06-11 08:50:21] SGLang Gemma4 FlashInfer geometry label=decode_as_prefill_vosplit1 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.bfloat16', '_cached_module': 'namespace(plan=Function(2189142976), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xeaa7993b8d60>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xe`

