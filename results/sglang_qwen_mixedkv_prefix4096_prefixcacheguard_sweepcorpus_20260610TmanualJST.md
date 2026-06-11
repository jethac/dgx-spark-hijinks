# SGLang Qwen prefix-cache graph-write guard validation, 2026-06-10

## Scope

This validates the SGLang guard that disables piecewise CUDA graph replay for EXTEND /
prefill batches while radix cache is enabled. The previous mixed-KV-only guard fixed
mixed-KV cache writes but left fp8 graph-written prefix cache entries active, which made
the matched comparator invalid. This run proves the broader prefix-cache guard restores a
matched fp8-vs-mixed row.

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Attention backend: FlashInfer
- KV modes: fp8 K/V versus mixed FP8-K + NVFP4-V
- Context: `8192`
- Reused prefix: `4096`
- Page size: `1`
- Graph launch policy: graph capture enabled globally (`ENABLE_FP4_CUDA_GRAPH=1`,
  `DISABLE_GRAPHS=0`)
- Guard policy: piecewise prefill graph replay disabled when radix cache is enabled, unless
  `SGLANG_ALLOW_PREFIX_CACHE_PREFILL_CUDA_GRAPH=1` is set for experiments
- Corpus:
  `results/sglang_qwen_mixedkv_reuse_prefix_sweep_ctx8192_20260610TmanualJST_corpus.md`

## Patch

`third_party/sglang/python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`
now returns `False` from `PiecewiseCudaGraphRunner.can_run()` for radix-cache EXTEND /
prefill batches by default. This routes prefix-cache-writing prefills and cached-prefix
prefills through eager while leaving regular decode CUDA graphs available.

Rationale: the failing graph-enabled sweep and the eager controls showed that the unsafe
operation is not only reading a cached prefix under graph replay. Graph-written prefix
cache entries later score differently for both fp8 and mixed-KV, even when the cached read
runs eager.

## Artifacts

- Compare:
  `results/sglang_qwen_mixedkv_prefix4096_prefixcacheguard_sweepcorpus_20260610TmanualJST_compare.json`
- Manifest:
  `results/sglang_qwen_mixedkv_prefix4096_prefixcacheguard_sweepcorpus_20260610TmanualJST_manifest.json`
- fp8 server log:
  `results/sglang_qwen_mixedkv_prefix4096_prefixcacheguard_sweepcorpus_20260610TmanualJST_fp8_server.log`
- mixed server log:
  `results/sglang_qwen_mixedkv_prefix4096_prefixcacheguard_sweepcorpus_20260610TmanualJST_mixed_server.log`

## Result

| mode | PPL | nats/token | cached tokens | KV tokens |
|---|---:|---:|---:|---:|
| fp8 K/V | 5.176347 | 1.644100 | 4096 | 3,116,450 |
| FP8-K + NVFP4-V | 5.169650 | 1.642805 | 4096 | 5,552,890 |

Delta mixed-vs-fp8: `-0.001295` nats/token. Both reports have `ok=true`, no missing
supplied-token logprobs, and no token-ID mismatches.

Observed allocator-token ratio: `5,552,890 / 3,116,450 = 1.782x`.

## Runtime Evidence

The fp8 run now routes both prefix population and cached-prefix scoring through eager:

```text
#new-token: 4096, #cached-token: 0, cuda graph: False
#new-token: 4096, #cached-token: 4096, cuda graph: False
```

The mixed-KV run does the same:

```text
#new-token: 4096, #cached-token: 0, cuda graph: False
#new-token: 4096, #cached-token: 4096, cuda graph: False
```

Mixed-KV storage remains active:

```text
SGLang FP4 KV mixed mode enabled: K cache uses FP8 e4m3, V cache uses packed NVFP4.
```

## Interpretation

The prefix-cache graph-write guard converts the previously bad graph-enabled 4096-prefix
row into the same good row as the all-eager control, while preserving the mixed-KV capacity
increase. This is now the SGLang Qwen mixed-KV quality baseline for prefix reuse on GB10:
radix cache ON, graph capture enabled globally, prefix-cache prefills forced eager.

Remaining work: this is a correctness guard, not the final graph-write repair. Piecewise
prefill graph replay remains unsafe for radix-cache population until the underlying SGLang
graph/cache-state bug is fixed.
