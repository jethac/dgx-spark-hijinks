# SGLang Qwen mixed-KV reuse-prefix sweep at fixed 8k, 2026-06-10

## Scope

This sweep tests whether the 8k mixed-KV PPL loss is a generic long-prefill penalty, a
fixed cached-prefix boundary artifact, or a function of how much of the prompt is actually
read from the mixed KV cache.

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Attention backend: FlashInfer
- KV modes compared: fp8 K/V versus mixed FP8-K + NVFP4-V
- Context: `8192`
- Reused-prefix lengths: `0, 1024, 2048, 4096, 6144`
- Page size: `1`
- Memory fraction: `0.40`
- Docker cap: `--memory=100g --memory-swap=100g`
- Graph policy: CUDA graphs enabled; mixed cached-prefix prefill guarded to eager
- Superseding note: a later eager control shows the graph-enabled quality deltas below are
  not a pure mixed-KV quality result. See
  `results/sglang_qwen_mixedkv_prefix4096_graph_vs_eager_20260610TmanualJST.md`.

## Artifacts

- sweep manifest: `results/sglang_qwen_mixedkv_reuse_prefix_sweep_ctx8192_20260610TmanualJST_manifest.json`
- corpus: `results/sglang_qwen_mixedkv_reuse_prefix_sweep_ctx8192_20260610TmanualJST_corpus.md`
- corpus manifest: `results/sglang_qwen_mixedkv_reuse_prefix_sweep_ctx8192_20260610TmanualJST_corpus_manifest.json`
- wrapper: `scripts/run_sglang_qwen_reuse_prefix_sweep.sh`
- pair runner: `scripts/run_sglang_qwen_ppl_pair.sh`
- scorer: `scripts/sglang_prompt_ppl_sweep.py`

Per-prefix artifacts use:

```text
results/sglang_qwen_mixedkv_reuse_prefix_sweep_ctx8192_20260610TmanualJST_ctx8192_prefix{N}_*
```

where `N` is one of `0, 1024, 2048, 4096, 6144`.

## Result

| reused prefix | scored tokens | fp8 cached | mixed cached | PPL fp8 | PPL mixed | delta nats/token | allocator ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8191 | 0 | 0 | 7.195014 | 7.195014 | 0.000000 | 1.778x |
| 1024 | 7167 | 1024 | 1024 | 7.149224 | 7.171268 | 0.003079 | 1.780x |
| 2048 | 6143 | 2048 | 2048 | 6.025073 | 6.187211 | 0.026555 | 1.781x |
| 4096 | 4095 | 4096 | 4096 | 7.238053 | 8.052973 | 0.106689 | 1.775x |
| 6144 | 2047 | 6144 | 6144 | 13.600429 | 15.257652 | 0.114980 | 1.777x |

All rows have `ok=true` for fp8 and mixed-KV, no missing supplied-token logprobs, and no
token-ID mismatches. The allocator-token ratio stays around `1.78x` for every row.

## Runtime Evidence

The server logs prove the intended cached-prefix geometry:

| reused prefix | scored request shape |
|---:|---|
| 0 | `#new-token: 8192, #cached-token: 0` |
| 1024 | `#new-token: 7168, #cached-token: 1024` |
| 2048 | `#new-token: 6144, #cached-token: 2048` |
| 4096 | `#new-token: 4096, #cached-token: 4096` |
| 6144 | `#new-token: 2048, #cached-token: 6144` |

Mixed-KV logs also prove the intended storage path:

```text
SGLang FP4 KV mixed mode enabled: K cache uses FP8 e4m3, V cache uses packed NVFP4.
```

For every cached-prefix row, the scored mixed-KV prefill used `cuda graph: False`, but the
cache-populating warmup/no-prefix prefill still used `cuda graph: True`. A later control
reran the 4096-prefix row with all graphs disabled on the same corpus path and changed the
result from `+0.106689` nats/token to `-0.001295` nats/token. Therefore this sweep is now
best interpreted as evidence that graph-written prefix cache state is unsafe, not as a
standalone NVFP4-V cached-prefix quality curve.

## Interpretation

This falsifies two weaker explanations:

- It is not a broad 8k mixed-KV long-prefill penalty: prefix `0` is exactly equal.
- It is not a fixed boundary artifact of "any radix hit": prefix `1024` is nearly clean.

The loss grows with the reused-prefix length at fixed total context. That points at the
amount and/or placement of attention over cached NVFP4-V values as the current quality
surface. Since K is fp8 in this mixed path, the next repair/debug target is not QK/LSE
stability; it is how the cached prefix's NVFP4 V contribution affects the supplied-token
distribution as more of the attention history comes from cache.

Practical status: the mixed-KV route remains mechanically correct and capacity-positive,
but graph-enabled quality is not blessable. The next SGLang experiment should repair or
gate CUDA graph cache-writing for fp8 and mixed-KV. Any V-only sensitivity claim must be
rerun in a graph-safe mode after that repair.
