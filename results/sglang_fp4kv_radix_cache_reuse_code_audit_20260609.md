# SGLang FP4-KV Radix Cache Reuse Code Audit, 2026-06-09

Purpose: compare SGLang's FP4-KV radix/prefix-cache reuse path against the vLLM
Gemma 3 reuse hypothesis: packed FP4 K/V data and FP8 scale buffers might desync when KV
blocks are reused.

## Finding

For the observed no-HiCache radix-cache failure, the normal device cache reuses physical KV
slot IDs. Packed K/V and FP8 scale buffers are indexed by the same slot, so the code does
not show an obvious normal-device-cache path where a prefix hit reuses K/V bytes but reads
scale bytes from a different slot.

Relevant code surfaces:

- `third_party/sglang/python/sglang/srt/managers/schedule_batch.py`
  - `Req.init_next_round_input()` calls `tree_cache.match_prefix(...)` and assigns
    `prefix_indices`.
- `third_party/sglang/python/sglang/srt/mem_cache/common.py`
  - `write_cache_indices()` writes cached physical indices into `req_to_token`, then
    appends new suffix `out_cache_loc`.
- `third_party/sglang/python/sglang/srt/mem_cache/radix_cache_cpp.py`
  - `match_prefix()` returns device KV indices.
  - eviction frees those indices through the allocator, not individual K/V/scale tensors.
- `third_party/sglang/python/sglang/srt/mem_cache/memory_pool.py`
  - `MHATokenToKVPoolFP4` owns same-indexed `k_buffer`, `v_buffer`, `k_scale_buffer`, and
    `v_scale_buffer`.
  - `set_kv_buffer()` quantizes and writes packed K, packed V, K scale, and V scale to the
    same `loc`.
  - `move_kv_cache()` copies all four FP4 buffers together, including its tiled fast path.
- `third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`
  - `_get_paged_kv_cache_and_kwargs()` passes packed KV plus `(k_sf, v_sf)` scale views
    together to FlashInfer.

This agrees with the live byte trace in
`results/sglang_qwen_fp4kv_write_read_trace_20260608T222204JST_summary.md`: sampled cached
pages `4113..4116` matched write input, stored bytes, and read bytes for K data, V data,
K scale, and V scale.

## Separate Real Gap

CPU/offload/HiCache-style lifetime handling has a plausible FP4 scale desync bug.
`MHATokenToKVPoolFP4` does not override `get_cpu_copy()` / `load_cpu_copy()`, so it inherits
base methods that copy and restore only `k_buffer` and `v_buffer`. FP4 `k_scale_buffer`,
`v_scale_buffer`, and global scales are omitted. If an FP4 request is offloaded and reloaded,
packed K/V could be restored into slots whose scale buffers contain unrelated values.

This is concrete, but it does not explain the current no-HiCache 55-token radix failure.

## Next Actions

1. For the active Qwen FP4-KV quality failure, add a diagnostic switch that forces FP4
   prefix-hit extend to the full paged path instead of `ragged suffix + paged cached prefix
   + _safe_merge_state`. Compare first-token logits against the default path.
2. For CPU/offload safety, either override `MHATokenToKVPoolFP4.get_cpu_copy()` /
   `load_cpu_copy()` to include FP4 scale buffers and global scales, or explicitly disable
   FP4 KV CPU offload/HiCache until that copy path is implemented.
