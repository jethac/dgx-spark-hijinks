# SGLang Qwen mixed-KV prefix-cache guard sweep at fixed 8k, 2026-06-10

## Scope

This reruns the fixed-8k reuse-prefix sweep after the broader SGLang prefix-cache graph
guard was implemented. The guard disables piecewise CUDA graph replay for EXTEND /
prefill batches while radix cache is enabled, so both prefix-cache population and
cached-prefix scoring run eager. Decode graphs remain available.

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Attention backend: FlashInfer
- KV modes compared: fp8 K/V versus mixed FP8-K + NVFP4-V
- Context: `8192`
- Reused-prefix lengths: `0, 1024, 2048, 4096, 6144`
- Page size: `1`
- Memory fraction: `0.40`
- Docker cap: `--memory=100g --memory-swap=100g`
- Graph launch policy: CUDA graphs enabled globally (`ENABLE_FP4_CUDA_GRAPH=1`,
  `DISABLE_GRAPHS=0`)
- Guard policy: prefix-cache-writing and cached-prefix prefills are forced eager unless
  `SGLANG_ALLOW_PREFIX_CACHE_PREFILL_CUDA_GRAPH=1` is set for experiments

This supersedes the earlier graph-enabled fixed-8k sweep as the quality baseline. The old
sweep is now a failure detector for graph-written prefix cache state, not a mixed-KV
quality curve.

## Artifacts

- Sweep manifest:
  `results/sglang_qwen_mixedkv_prefixcacheguard_reuse_prefix_sweep_ctx8192_20260610TmanualJST_manifest.json`
- Corpus:
  `results/sglang_qwen_mixedkv_prefixcacheguard_reuse_prefix_sweep_ctx8192_20260610TmanualJST_corpus.md`
- Corpus manifest:
  `results/sglang_qwen_mixedkv_prefixcacheguard_reuse_prefix_sweep_ctx8192_20260610TmanualJST_corpus_manifest.json`
- Wrapper: `scripts/run_sglang_qwen_reuse_prefix_sweep.sh`
- Pair runner: `scripts/run_sglang_qwen_ppl_pair.sh`
- Scorer: `scripts/sglang_prompt_ppl_sweep.py`

Per-prefix artifacts use:

```text
results/sglang_qwen_mixedkv_prefixcacheguard_reuse_prefix_sweep_ctx8192_20260610TmanualJST_ctx8192_prefix{N}_*
```

where `N` is one of `0, 1024, 2048, 4096, 6144`.

## Result

| reused prefix | scored tokens | fp8 cached | mixed cached | PPL fp8 | PPL mixed | delta nats/token | allocator ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8191 | 0 | 0 | 7.195014 | 7.195014 | 0.000000 | 1.776x |
| 1024 | 7167 | 1024 | 1024 | 6.800032 | 6.798214 | -0.000267 | 1.779x |
| 2048 | 6143 | 2048 | 2048 | 5.648189 | 5.650651 | 0.000436 | 1.778x |
| 4096 | 4095 | 4096 | 4096 | 5.176347 | 5.169650 | -0.001295 | 1.769x |
| 6144 | 2047 | 6144 | 6144 | 9.277885 | 8.676802 | -0.066981 | 1.779x |

All rows have `ok=true` for fp8 and mixed-KV. The scorer reports no token-ID mismatches.
The one-token difference between expected and scored continuation length is the existing
SGLang logprob-span boundary placeholder, not a missing model token.

## Runtime Evidence

The logs prove both the cache-populating prefill and the cached-prefix prefill run eager
under the guard. Examples:

| reused prefix | fp8 scored request | mixed scored request |
|---:|---|---|
| 1024 | `#new-token: 7168, #cached-token: 1024, cuda graph: False` | `#new-token: 7168, #cached-token: 1024, cuda graph: False` |
| 2048 | `#new-token: 6144, #cached-token: 2048, cuda graph: False` | `#new-token: 6144, #cached-token: 2048, cuda graph: False` |
| 4096 | `#new-token: 4096, #cached-token: 4096, cuda graph: False` | `#new-token: 4096, #cached-token: 4096, cuda graph: False` |
| 6144 | `#new-token: 2048, #cached-token: 6144, cuda graph: False` | `#new-token: 2048, #cached-token: 6144, cuda graph: False` |

The cache-populating requests also run eager:

```text
#new-token: 1024, #cached-token: 0, cuda graph: False
#new-token: 2048, #cached-token: 0, cuda graph: False
#new-token: 4096, #cached-token: 0, cuda graph: False
#new-token: 6144, #cached-token: 0, cuda graph: False
```

Mixed-KV storage remains active in every mixed row:

```text
SGLang FP4 KV mixed mode enabled: K cache uses FP8 e4m3, V cache uses packed NVFP4.
```

## Interpretation

The graph-safe sweep is green for Qwen2.5 mixed-KV prefix reuse at 8k. At prefixes
`1024`, `2048`, and `4096`, mixed-KV is within `0.0013` nats/token of fp8. At prefix
`6144`, mixed-KV is materially better than fp8 on this corpus slice (`-0.066981`
nats/token), which should be treated as corpus/noise directionality rather than a claim
that mixed-KV improves quality.

This replaces the earlier graph-enabled sweep result where deltas grew to `+0.106689`
nats/token at prefix `4096`. The difference is the cache-writing path: graph-written
prefix cache entries were unsafe for both fp8 and mixed-KV, while eager-written prefix
cache entries make fp8 and mixed-KV effectively match.

Practical status: SGLang Qwen mixed FP8-K + NVFP4-V is now the blessed SGLang radix-cache
capacity path under the prefix-cache graph guard. It preserves the observed allocator
token increase around `1.77-1.78x` in this launch configuration. This is still a mixed-KV
claim, not full NVFP4 K+V; full NVFP4 K+V remains a separate red/open SGLang track.

Remaining work: the guard is a correctness policy, not a final graph-write repair.
Piecewise prefill graph replay for radix-cache population should stay disabled by default
until the underlying graph/cache-state bug is fixed.
