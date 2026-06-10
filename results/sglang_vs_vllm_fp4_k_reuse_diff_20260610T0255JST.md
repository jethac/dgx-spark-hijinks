# SGLang vs vLLM FP4-K Reuse Diff

Date: 2026-06-10 JST

## Verdict

The vLLM cache-hit run proves full NVFP4 K+V prefix reuse can serve the Qwen smoke gate
without a first-token flip. That changes the SGLang diagnosis: full-NVFP4 radix failure is
not proven to be inherent FP4-K attention loss. Treat it as a SGLang feed/regime bug until a
direct vLLM-vs-SGLang tensor comparison proves otherwise.

## New Reference Point

vLLM artifact: `results/vllm_qwen_nvfp4_prefixcache_hitproof_20260610TmanualJST_summary.md`.

Key metrics:

```text
vllm:prefix_cache_queries_total 12108.0
vllm:prefix_cache_hits_total 3728.0
vllm:prompt_tokens_by_source_total{source="local_cache_hit"} 3728.0
vllm:prompt_tokens_cached_total 3728.0
```

Both requests returned first token `*` under full NVFP4 K+V with prefix caching enabled.

## Code-Path Diff

vLLM working path:

- `third_party/vllm/vllm/v1/attention/backends/flashinfer.py`
- Writes KV through `reshape_and_cache_flash(..., self.kv_cache_dtype, layer._k_scale,
  layer._v_scale)` before attention reads from cache.
- For NVFP4, derives data and block-scale views from the same packed cache with
  `nvfp4_kv_cache_split_views(kv_cache_permute)`.
- Passes the same global scales and cache-scale views to FlashInfer prefill/decode:
  `k_scale=layer._k_scale_float`, `v_scale=layer._v_scale_float`,
  `kv_cache_sf=nvfp4_kv_block_scales`.
- The second request proved a real cache hit, so cached K was read through this path.

SGLang failing full-NVFP4 radix path:

- `third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`
- `forward_extend_merge_paged` splits attention into:
  - suffix partial through `prefill_wrapper_ragged.forward_return_lse(...)`
  - cached-prefix partial through `_run_paged_native(...)`
  - merge through `_safe_merge_state(o1, s1, o2, s2)`
- KV is written to the FP4 pool after the merge in the ragged-prefix path.
- The pool stores full NVFP4 K as packed `uint8` plus K block-scale buffer and per-layer
  global scale:
  `third_party/sglang/python/sglang/srt/mem_cache/memory_pool.py`.

Existing traces already show:

- page IDs and data/scale bytes match;
- `_safe_merge_state` matches an independent online-softmax reference;
- regime-matched suffix partials make the merge match an all-FP4 recompute;
- mixed FP8-K/NVFP4-V avoids the first-token flip but only as a fallback path.

## Interpretation

The likely SGLang bug is at the boundary where `forward_extend_merge_paged` feeds K into
FlashInfer attention, not in the merge arithmetic and not in byte/page pairing.

The concrete difference to investigate next:

- vLLM computes reused attention from packed-cache K/V with block scales extracted from the
  same cache view.
- SGLang computes suffix attention via ragged dense tensors and prefix attention via paged
  packed FP4 tensors, then merges the two states.
- If full-NVFP4 is to match vLLM, SGLang needs the suffix/current-token contribution and the
  cached-prefix contribution to use the same FP4-K scale convention and FlashInfer feed path,
  or it needs to write/read the suffix through the FP4 cache before computing the suffix
  partial.

## Next Check

Diff SGLang's `_run_paged_native` / `forward_extend_merge_paged` arguments against vLLM's
working `prefill_wrapper.run` and `decode_wrapper.run` calls:

- global K scale value and handedness;
- K block-scale tensor dtype/shape/order;
- whether FlashInfer sees `kv_data_type=torch.uint8` as NVFP4 in SGLang in the same way vLLM
  explicitly declares `kv_data_type="nvfp4"`;
- whether the SGLang ragged suffix path can be replaced by a paged-cache read for the suffix
  before merge.

Mixed FP8-K/NVFP4-V remains the documented fallback, but it should not be the final answer
until this full-NVFP4 feed-path diff is exhausted.
