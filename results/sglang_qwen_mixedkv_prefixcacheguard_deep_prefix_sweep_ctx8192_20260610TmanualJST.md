# SGLang Qwen mixed-KV deep prefix-cache guard sweep at fixed 8k, 2026-06-10

## Scope

This completes the interrupted deep-prefix portion of the SGLang Qwen mixed-KV
reuse sweep after the prefix-cache graph-write guard was implemented.

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Attention backend: FlashInfer
- KV modes compared: fp8 K/V versus mixed FP8-K + NVFP4-V
- Context: `8192`
- Reused-prefix lengths: `4096, 6144, 7168, 7680`
- Page size: `1`
- Memory fraction: `0.40`
- Docker cap: `--memory=100g --memory-swap=100g`
- Graph launch policy: CUDA graphs enabled globally, but prefix-cache-writing
  and cached-prefix prefills are forced eager by the SGLang guard.

This extends the earlier graph-safe fixed-8k sweep
`results/sglang_qwen_mixedkv_prefixcacheguard_reuse_prefix_sweep_ctx8192_20260610TmanualJST.md`,
which covered reused prefixes `0, 1024, 2048, 4096, 6144`.

## Artifacts

- Sweep manifest:
  `results/sglang_qwen_mixedkv_prefixcacheguard_deep_prefix_sweep_ctx8192_20260610TmanualJST_manifest.json`
- Stop-point summary:
  `results/sglang_qwen_mixedkv_prefixcacheguard_deep_prefix_sweep_ctx8192_20260610TmanualJST_stop_point.md`
- Corpus:
  `results/sglang_qwen_mixedkv_prefixcacheguard_reuse_prefix_sweep_ctx8192_20260610TmanualJST_corpus.md`
- Corpus manifest:
  `results/sglang_qwen_mixedkv_prefixcacheguard_reuse_prefix_sweep_ctx8192_20260610TmanualJST_corpus_manifest.json`
- Wrapper: `scripts/run_sglang_qwen_reuse_prefix_sweep.sh`
- Pair runner: `scripts/run_sglang_qwen_ppl_pair.sh`
- Scorer: `scripts/sglang_prompt_ppl_sweep.py`

Per-prefix artifacts use:

```text
results/sglang_qwen_mixedkv_prefixcacheguard_deep_prefix_sweep_ctx8192_20260610TmanualJST_ctx8192_prefix{N}_*
```

where `N` is one of `4096, 6144, 7168, 7680`.

## Result

| reused prefix | scored tokens | fp8 cached | mixed cached | PPL fp8 | PPL mixed | delta nats/token | allocator ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 4095 | 4096 | 4096 | 5.176347 | 5.169650 | -0.001295 | 1.778x |
| 6144 | 2047 | 6144 | 6144 | 9.277885 | 8.678394 | -0.066797 | 1.777x |
| 7168 | 1023 | 7168 | 7168 | 10.359486 | 10.444663 | 0.008188 | 1.778x |
| 7680 | 511 | 7680 | 7680 | 14.715744 | 14.875298 | 0.010784 | 1.784x |

All rows have `ok=true` for fp8 and mixed-KV in the manifest. The scorer reports the
same cached-token counts for fp8 and mixed-KV. The one-token difference between
expected and scored continuation length is the existing SGLang logprob-span boundary
placeholder.

## Runtime Evidence

The logs prove the guard policy on the two operations that matter: the cache-populating
prefill and the cached-prefix scoring prefill are both eager.

| reused prefix | fp8 scored request | mixed scored request |
|---:|---|---|
| 4096 | `#new-token: 4096, #cached-token: 4096, cuda graph: False` | `#new-token: 4096, #cached-token: 4096, cuda graph: False` |
| 6144 | `#new-token: 2048, #cached-token: 6144, cuda graph: False` | `#new-token: 2048, #cached-token: 6144, cuda graph: False` |
| 7168 | `#new-token: 1024, #cached-token: 7168, cuda graph: False` | `#new-token: 1024, #cached-token: 7168, cuda graph: False` |
| 7680 | `#new-token: 512, #cached-token: 7680, cuda graph: False` | `#new-token: 512, #cached-token: 7680, cuda graph: False` |

The cache-populating requests also run eager:

```text
#new-token: 4096, #cached-token: 0, cuda graph: False
#new-token: 6144, #cached-token: 0, cuda graph: False
#new-token: 7168, #cached-token: 0, cuda graph: False
#new-token: 7680, #cached-token: 0, cuda graph: False
```

Mixed-KV storage is active in every mixed row:

```text
SGLang FP4 KV mixed mode enabled: K cache uses FP8 e4m3, V cache uses packed NVFP4.
```

## Interpretation

The deep-prefix sweep stays quality-green for the current SGLang mixed FP8-K +
NVFP4-V path under the prefix-cache graph guard. The `4096` row reproduces the earlier
green control. The `7168` and `7680` rows show small positive deltas, but they score only
`1023` and `511` continuation tokens respectively, so they should be treated as
short-span, corpus-sensitive checks rather than a stable long-context quality trend.
The `6144` row moves in the opposite direction and is also best treated as corpus/noise
directionality.

This completes the prefix-depth curve requested for mixed-KV blessing. The remaining
open issue for public claims is the capacity denominator: the observed allocator-token
ratio around `1.78x` is not the raw mixed-KV storage ratio. See
`results/sglang_qwen_mixedkv_capacity_denominator_audit_20260610TmanualJST.md`.
