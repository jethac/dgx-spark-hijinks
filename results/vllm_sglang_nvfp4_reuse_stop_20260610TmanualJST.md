# vLLM/SGLang NVFP4-KV Reuse Stop Point

Date: 2026-06-10 JST

## Verdict

vLLM still proves full NVFP4 K+V prefix reuse can serve the Qwen smoke probe correctly.
SGLang still fails full NVFP4 K+V radix reuse after suffix packing, so the current evidence
keeps the bug scoped to SGLang's FP4 cached-prefix path rather than inherent FP4-K reuse loss.

The attempted vLLM tensor-feed trace did not run: the container logged
`VLLM_SPARK_KV_TRACE*` as unknown environment variables and produced no JSONL trace file.
So we have a second cache-hit proof, but not a direct vLLM tensor-argument dump.

## vLLM Re-Run

Artifact prefix: `results/vllm_qwen_nvfp4_prefixcache_trace_20260610TmanualJST`.

Configuration:

- image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`
- model: `/home/jethac/models/aeon/qwen36-nvfp4`
- KV dtype: `nvfp4`
- prefix caching: enabled
- Docker memory cap: `--memory 100g --memory-swap 100g`
- runner: `scripts/run_vllm_qwen_prefix_cache_probe.sh`

Result:

- first tokens: `*`, `*`
- `ok: true`
- request latencies: `1.7066s`, `0.4298s`
- metrics:
  - `vllm:prefix_cache_hits_total = 3728`
  - `vllm:prompt_tokens_by_source_total{source="local_cache_hit"} = 3728`
  - `vllm:prompt_tokens_cached_total = 3728`

Runtime path:

- `Using nvfp4 data type to store kv cache`
- `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM V-scale-factor deswizzle enabled`
- GPU KV cache size: `9,335,049 tokens`

Trace limitation:

- The runner set `VLLM_SPARK_KV_TRACE=1` and related env vars.
- The server logged these as unknown vLLM environment variables.
- No `vllm_qwen_nvfp4_prefixcache_trace_20260610TmanualJST_kv_trace.jsonl` file was produced.
- Conclusion: this image has the NVFP4-KV routing behavior but not the local JSONL trace instrumentation.
  A direct tensor-argument diff needs a source-overlay vLLM run or a rebuilt image containing the trace code.

## SGLang State

Latest relevant artifact:
`results/sglang_qwen_fullnvfp4_suffixpacked4_20260610TmanualJST_summary.md`.

That run:

- packed suffix K/V to NVFP4 and passed `kv_cache_sf`, `k_scale`, and `v_scale` to FlashInfer ragged suffix attention;
- planned the cached-prefix ragged wrapper as `uint8` only for batches with cached-prefix tokens;
- completed without scheduler crashes;
- still failed with radix cache ON:
  - fresh/no-cache token: `**`
  - cached-prefix token: `ark`
  - cached tokens: `55`
  - layer-0 attention-output cosine against the dense-cache comparator: `0.012443760722381951`

This rules out "suffix stayed bf16 while prefix was FP4" as a sufficient explanation.

## Read-Only Code Findings

Two subagents inspected the feed path without editing files.

Confirmed:

- FlashInfer paged attention wants the decode/dequant scale convention:
  `fp4_value * block_scale * global_scale -> bf16`.
- SGLang's normal full-NVFP4 path passes that convention:
  `flashinfer_backend.py::_get_paged_kv_cache_and_kwargs()` returns
  `kv_cache_sf=(k_sf, v_sf)`, `k_scale=k_global`, `v_scale=v_global`.
- A blind global-scale inversion patch is not justified.

V-scale layout finding:

- vLLM writes V scale factors in the swizzled layout used by its packed cache writer and enables
  `FLASHINFER_PAGED_V_SF_DESWIZZLE=1`.
- SGLang stores separate K/V scale buffers linearly and uses `page_size=1`.
- Because FlashInfer's paged V-SF de-swizzle is a 4-token page-layout transform, copying vLLM's
  de-swizzle flag into SGLang's page-size-1 layout is likely wrong without a dedicated test.

Scale argument edge case:

- `MHATokenToKVPoolFP4.set_kv_buffer()` accepts `k_scale` / `v_scale` parameters but ignores them,
  loading scales only from `layer`.
- `HybridLinearKVPool.set_kv_buffer()` can call the full pool with `layer=None` and scalar
  `k_scale` / `v_scale`, so checkpoint-provided scales can be dropped on that transfer path.
- The failing Qwen run logged `hybrid_swa=False`, so this edge case is not the current Qwen radix
  failure, but it is a real compatibility bug candidate for hybrid/full-pool paths.

## Current Interpretation

Hard evidence:

- vLLM full NVFP4 K+V prefix reuse is green on the Qwen smoke gate with a proven local cache hit.
- SGLang full NVFP4 K+V radix reuse remains red after suffix FP4 packing.
- SGLang page bytes, scale bytes, and merge arithmetic have already been largely cleared by prior traces.

Most likely next useful experiment:

- Rebuild or source-overlay vLLM with the local JSONL KV trace instrumentation and rerun the same
  prefix-cache hit probe, capturing actual `k_scale`, `v_scale`, `kv_cache_sf`, data/scale views,
  and page layout for the reused request.
- Then compare against SGLang's `_run_paged_native` logged args for the same Qwen geometry.

Questions for Claude:

1. Given vLLM's trace envs were not present in the image, is a source-overlay vLLM trace run worth the startup cost, or should we instrument SGLang more deeply first?
2. Does the SGLang `page_size=1` layout definitively rule out vLLM-style V-SF de-swizzle, or should we build a standalone FlashInfer page-size-1 V-SF layout probe?
3. Should we fix the `MHATokenToKVPoolFP4.set_kv_buffer()` ignored `k_scale/v_scale` arguments now as a separate hybrid-path correctness patch, even though it is not the Qwen radix failure?
