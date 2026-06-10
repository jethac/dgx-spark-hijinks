# SGLang Qwen mixed-KV claim manifest, 2026-06-10

## Claim

SGLang can serve Qwen2.5 on GB10 with radix cache ON using mixed KV cache
(`FP8-K + NVFP4-V`) without the full-NVFP4 K-side reuse corruption, when prefix-cache
writing and cached-prefix prefills are routed eager by the prefix-cache graph guard.

This is the claim-ready SGLang fallback row. It is not the full NVFP4 K+V row.

## Scope

- Host class: GB10 / `sm_121`
- Runtime: SGLang source-stack image `sglang-source-stack-c3dae30f-e631a13fd`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Attention backend: FlashInfer
- KV cache mode: mixed `FP8-K + NVFP4-V`
- Radix cache: ON
- Page size: `1`
- Memory fraction: `0.40`
- Docker cap: `--memory=100g --memory-swap=100g`
- CUDA graph policy: decode graphs may remain enabled; prefix-cache-writing and
  cached-prefix prefills are forced eager unless
  `SGLANG_ALLOW_PREFIX_CACHE_PREFILL_CUDA_GRAPH=1` is set for experiments.

## Gates

| Gate | Artifact | Verdict |
|---|---|---|
| Mixed pool integration | `results/sglang_mixed_kv_pool_probe_20260610T0036JST.md` | Green: K buffer fp8, V buffer packed NVFP4, pool reference cosine `0.999995` |
| Default radix first-token gate | `results/sglang_qwen_mixedkv_default_20260610T0042JST_summary.md` | Green: cached-prefix second request emits `**`, not the old `ark` flip |
| Fresh fp8 comparator | `results/sglang_qwen_fp8_vs_mixedkv_fresh_comparator_20260610TmanualJST.md` | Green: short decode parity and larger allocator token pool |
| Natural long prefill smoke | `results/sglang_qwen_mixedkv_natural_longprefill_20260610TmanualJST.md` | Green: no heuristic repetition flags on the non-repetitive prompt |
| Graph-write bug localized | `results/sglang_qwen_mixedkv_prefix4096_graph_vs_eager_20260610TmanualJST.md` | Green for diagnosis: graph-written prefix cache is unsafe; eager-written prefix cache recovers parity |
| Prefix-cache graph guard | `results/sglang_qwen_mixedkv_prefix4096_prefixcacheguard_sweepcorpus_20260610TmanualJST.md` | Green: prefix write/read eager under global graph-enabled launch |
| Fixed-8k prefix ladder | `results/sglang_qwen_mixedkv_prefixcacheguard_reuse_prefix_sweep_ctx8192_20260610TmanualJST.md` | Green: prefixes `0, 1024, 2048, 4096, 6144` |
| Deep-prefix continuation | `results/sglang_qwen_mixedkv_prefixcacheguard_deep_prefix_sweep_ctx8192_20260610TmanualJST.md` | Green: prefixes `4096, 6144, 7168, 7680` |
| Capacity denominator audit | `results/sglang_qwen_mixedkv_capacity_denominator_audit_20260610TmanualJST.md` | Complete: observed allocator-token ratio and normalized storage ratio separated |

## Quality Summary

Graph-safe fixed-8k prefix ladder:

| reused prefix | delta nats/token |
|---:|---:|
| 0 | 0.000000 |
| 1024 | -0.000267 |
| 2048 | 0.000436 |
| 4096 | -0.001295 |
| 6144 | -0.066981 |

Deep-prefix continuation:

| reused prefix | scored tokens | delta nats/token |
|---:|---:|---:|
| 4096 | 4095 | -0.001295 |
| 6144 | 2047 | -0.066797 |
| 7168 | 1023 | 0.008188 |
| 7680 | 511 | 0.010784 |

Interpretation: no catastrophic cached-prefix quality failure remains in the mixed-KV
row under the prefix-cache graph guard. The deepest positive deltas are small and score
short continuation spans, so they are not a stable trend claim.

## Capacity Summary

The current fixed mixed-KV allocator result is:

- Observed allocator-token ratio after the pool-configurator fix: about `1.28x` versus fp8.
- Physical K+V byte budget: effectively equal to fp8 at the same `--mem-fraction-static`.
- Short decode: parity on the Qwen verification row.

Verification artifact: `results/sglang_mixedkv_poolconfigfix_20260610TmanualJST.md`.

The older `~1.78x` mixed-KV allocator ratio is retained only as a pre-fix artifact. It
came from sizing the pool with the logical full-FP4 dtype estimate while physically
allocating FP8 K plus NVFP4 V, which realized about `1.39x` more K+V bytes than the fp8
row. Do not quote that as the current mixed-KV capacity claim.

## Caveats

- Full NVFP4 K+V with SGLang radix remains red/open. vLLM proves FP4-K reuse can work,
  but this SGLang row deliberately avoids the FP4-K partial-state LSE sensitivity by
  keeping K in fp8.
- The prefix-cache graph guard is a correctness policy, not the final graph-write fix.
  Piecewise prefill graph replay for radix-cache population should remain disabled until
  the graph-written prefix-cache state bug is repaired.
- `--disable-radix-cache` is not part of the claim. The row keeps radix cache ON.
- The capacity claim is scoped to this build and launch configuration unless the memory
  denominator is explicitly normalized.

## Status

Claim-ready as the SGLang mixed-KV fallback row for Qwen on GB10, with the fixed
capacity claim scoped to `~1.28x` at equal physical K+V byte budget.
