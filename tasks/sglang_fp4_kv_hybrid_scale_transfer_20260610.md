# SGLang FP4-KV Hybrid Scale Transfer Task

Date: 2026-06-10 JST

## Problem

`HybridLinearKVPool.set_kv_buffer()` can forward scalar `k_scale` / `v_scale` values into an
inner KV pool, but `MHATokenToKVPoolFP4.set_kv_buffer()` does not currently use those scalar
arguments as authoritative global scales. It derives layer global scales from the `layer`
object through `_maybe_load_layer_global_scales()`.

This is a latent correctness risk for hybrid/SWA memory pools where the outer pool can call
the inner pool with `layer=None` plus explicit scalar scales.

Relevant local sites:

- `third_party/sglang/python/sglang/srt/mem_cache/memory_pool.py`
- `MHATokenToKVPoolFP4.set_kv_buffer(...)`
- `HybridLinearKVPool.set_kv_buffer(...)`

## Current Scope Decision

Do not fix this in the active Qwen radix-cache hunt.

Reason: the current Qwen FP4-KV cached-prefix failure does not use the hybrid/SWA pool path,
and changing pool scale transfer now would invalidate the existing Qwen trace baseline.

## Why It Still Matters

Gemma SWA/hybrid KV rungs will exercise the hybrid pool path. If scalar checkpoint/runtime
scales are dropped during hybrid pool forwarding, Gemma FP4-KV runs can fail for a different
reason than the current Qwen radix-cache bug.

## Acceptance Criteria For A Future Patch

- Gate the change to `MHATokenToKVPoolFP4` / hybrid FP4-KV scale transfer.
- Preserve current Qwen non-hybrid behavior.
- Add a unit or standalone pool probe where:
  - `HybridLinearKVPool.set_kv_buffer()` passes explicit non-default `k_scale` and `v_scale`;
  - the inner `MHATokenToKVPoolFP4` records those scales in `k_global_float` /
    `v_global_float`;
  - a decode/paged-prefill FlashInfer bridge consumes the written cache and matches a
    dequantized reference.
- Re-run the Qwen FP4-KV radix row to confirm this standalone fix does not change the current
  non-hybrid baseline.
