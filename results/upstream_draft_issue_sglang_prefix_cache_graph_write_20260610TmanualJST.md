# Draft upstream issue: piecewise CUDA graph prefix-cache writes corrupt reused-prefix quality

Target repo: `sgl-project/sglang`

## Title

Piecewise CUDA graph replay for radix-cache-populating prefills changes reused-prefix
logprob results, affecting fp8 and experimental mixed FP8-K/NVFP4-V KV

## Summary

On GB10 / CUDA 13, a Qwen2.5 prefix-cache PPL control shows that prefix-cache entries
written by SGLang piecewise CUDA graph replay produce different reused-prefix logprob
results than the same prefix-cache entries written eagerly. The effect is visible for
both fp8 KV and an experimental mixed FP8-K/NVFP4-V KV path.

The important localization is that the scored cached-prefix request was eager in both
runs. The difference is the earlier cache-populating prefill:

- graph-written prefix cache: fp8 and mixed-KV PPL shift, with a large mixed-vs-fp8
  delta on the 4096-prefix row;
- eager-written prefix cache: fp8 and mixed-KV match on the same corpus path.

This suggests the unsafe operation is the prefix-cache write under piecewise graph replay,
not the later cached-prefix read.

## Environment

- Host class: GB10 / `sm_121`
- CUDA: 13-era stack
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Attention backend: FlashInfer
- Page size: `1`
- Context length: `8192`
- Reused prefix: `4096`
- Scored tokens: `4095`
- Radix cache: enabled
- Docker memory cap used during campaign runs: `--memory=100g --memory-swap=100g`

## Reproduction Shape

Run a fixed-corpus supplied-token PPL pair twice:

1. Global graph-enabled launch where the prefix-cache-populating prefill uses piecewise
   graph replay.
2. Eager control where both the cache-populating prefill and cached-prefix scoring
   prefill run eager.

The same corpus path and same prefix length are used in both runs.

## Observed Result

From artifact
`results/sglang_qwen_mixedkv_prefix4096_graph_vs_eager_20260610TmanualJST.md`:

| mode | fp8 PPL | mixed PPL | delta nats/token | fp8 cached | mixed cached |
|---|---:|---:|---:|---:|---:|
| Graph-enabled sweep | 7.238053 | 8.052973 | +0.106689 | 4096 | 4096 |
| Eager no-trace control | 5.176347 | 5.169650 | -0.001295 | 4096 | 4096 |
| Eager prefix-reference trace | 5.176347 | 5.169650 | -0.001295 | 4096 | 4096 |

Graph-enabled server log:

```text
#new-token: 4096, #cached-token: 0, cuda graph: True
#new-token: 4096, #cached-token: 4096, cuda graph: False
```

Eager control server log:

```text
#new-token: 4096, #cached-token: 0, cuda graph: False
#new-token: 4096, #cached-token: 4096, cuda graph: False
```

The eager no-trace and eager traced rows match exactly, so trace instrumentation is not
the reason the row recovers.

## Expected Result

Prefix-cache entries written by piecewise CUDA graph replay should be equivalent to
prefix-cache entries written eagerly. A later reused-prefix request should produce the
same supplied-token logprobs, within normal numerical tolerance, independent of whether
the cache-populating prefill used graph replay.

## Current Workaround

The campaign fork currently disables piecewise CUDA graph replay for all EXTEND/prefill
batches while radix cache is enabled, unless an experiment explicitly sets:

```text
SGLANG_ALLOW_PREFIX_CACHE_PREFILL_CUDA_GRAPH=1
```

This routes both cache-populating prefills and cached-prefix prefills through eager while
preserving decode CUDA graphs.

Patch location in the campaign fork:

```text
third_party/sglang/python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
```

## Why This Matters

This is not only an experimental FP4 issue. The control changes fp8 results too, so the
bug appears to be in graph-written prefix-cache state rather than only in mixed-KV
quantization.

For GB10 / unified-memory experiments, the workaround is acceptable for correctness but
not a final upstream fix: prefix-cache graph replay is a performance feature, and the
safe scope should be narrowed once the graph-written cache-state mismatch is found.

## Attached Campaign Artifacts

- `results/sglang_qwen_mixedkv_prefix4096_graph_vs_eager_20260610TmanualJST.md`
- `results/sglang_qwen_mixedkv_prefix4096_prefixcacheguard_sweepcorpus_20260610TmanualJST.md`
- `results/sglang_qwen_mixedkv_prefixcacheguard_reuse_prefix_sweep_ctx8192_20260610TmanualJST.md`
- `results/sglang_qwen_mixedkv_prefixcacheguard_deep_prefix_sweep_ctx8192_20260610TmanualJST.md`

## Suggested Upstream Fix Direction

Audit which tensors/state are written into radix cache during an EXTEND/prefill batch
under piecewise CUDA graph replay, then compare against eager writes for the same prefix:

- KV data buffers
- scale buffers for quantized KV modes
- request-to-token slot mapping
- metadata updated during cache commit
- any graph-captured write indices or lengths that may be stale across replay

The first upstream-safe mitigation may be a narrower version of the campaign guard:
disable graph replay only for cache-writing prefill shapes that actually commit entries
to radix cache.
