# SGLang Gemma 4 E4B Rung 0 Smoke

Status: GREEN

- Run: `sglang_gemma4_e4b_rung1_fullnvfp4_retry1_20260611T181709JST`
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
- Coherent Tokyo answer: `True`

## Result

See response and geometry samples below.

## Response

```json
{
  "id": "2368863a73e340b9a66993a3bdfc5594",
  "object": "chat.completion",
  "created": 1781169718,
  "model": "google/gemma-4-E4B-it",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of Japan is Tokyo.",
        "reasoning_content": null,
        "tool_calls": null
      },
      "logprobs": null,
      "finish_reason": "stop",
      "matched_stop": 106
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "total_tokens": 28,
    "completion_tokens": 8,
    "prompt_tokens_details": null,
    "reasoning_tokens": 0
  },
  "metadata": {
    "weight_version": "default"
  }
}
```

## Geometry Samples

### SWA prefill
- `[2026-06-11 09:21:19] SGLang Gemma4 FlashInfer geometry label=extend_paged layer=0 wrapper_id=False planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=256, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=256 sliding_window=512 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1225470400), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xe7899dfc4720>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xe7899dfc6980>)',`
- `[2026-06-11 09:21:23] SGLang Gemma4 FlashInfer geometry label=extend_paged layer=1 wrapper_id=False planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=256, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=256 sliding_window=512 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1225470400), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xe7899dfc4720>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xe7899dfc6980>)',`

### Global VO-split prefill
- `[2026-06-11 09:21:27] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1225447440), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xe7899dfc7240>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xe7899dfc7`
- `[2026-06-11 09:21:27] SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit1 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1225447440), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xe7899dfc7240>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xe7899dfc7`

### Decode D=512
- `[2026-06-11 09:21:58] SGLang Gemma4 FlashInfer geometry label=decode_as_prefill_vosplit0 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1225447440), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xe7899dfc7240>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xe789`
- `[2026-06-11 09:21:58] SGLang Gemma4 FlashInfer geometry label=decode_as_prefill_vosplit1 layer=5 wrapper_id=True planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256) layer_q_heads=8 layer_k_heads=2 layer_v_heads=2 layer_head_dim=512 sliding_window=-1 vo_split=False wrapper={'class': 'BatchPrefillWithPagedKVCacheWrapper', 'state': {'_backend': 'fa2', '_cached_kv_data_type': 'torch.uint8', '_cached_module': 'namespace(plan=Function(1225447440), ragged_run=<function get_batch_prefill_module.<locals>.ragged_run at 0xe7899dfc7240>, paged_run=<function get_batch_prefill_module.<locals>.paged_run at 0xe789`

