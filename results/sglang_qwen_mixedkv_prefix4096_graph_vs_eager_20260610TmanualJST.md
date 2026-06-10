# SGLang Qwen mixed-KV prefix-4096 graph versus eager control, 2026-06-10

## Scope

This control reruns the fixed-8k / reused-prefix-4096 Qwen PPL row after the
graph-enabled sweep showed a large mixed-KV penalty. The purpose is to isolate whether the
loss came from mixed-KV attention itself, trace instrumentation, corpus drift, or CUDA
graph cache-writing state.

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Attention backend: FlashInfer
- Page size: `1`
- Context: `8192`
- Reused prefix: `4096`
- Scored tokens: `4095`
- Docker cap: `--memory=100g --memory-swap=100g`
- Corpus: `results/sglang_qwen_mixedkv_reuse_prefix_sweep_ctx8192_20260610TmanualJST_corpus.md`
- Corpus SHA256: `3004bb6ac466a2cecda738832ae5631db576fbe598905af54cd4f4ce81765f3f`

## Artifacts

Graph-enabled baseline:

- `results/sglang_qwen_mixedkv_reuse_prefix_sweep_ctx8192_20260610TmanualJST_ctx8192_prefix4096_compare.json`
- `results/sglang_qwen_mixedkv_reuse_prefix_sweep_ctx8192_20260610TmanualJST_ctx8192_prefix4096_fp8_server.log`
- `results/sglang_qwen_mixedkv_reuse_prefix_sweep_ctx8192_20260610TmanualJST_ctx8192_prefix4096_mixed_server.log`

Eager no-trace control:

- `results/sglang_qwen_mixedkv_prefix4096_eager_notrace_sweepcorpus_20260610TmanualJST_compare.json`
- `results/sglang_qwen_mixedkv_prefix4096_eager_notrace_sweepcorpus_20260610TmanualJST_fp8_server.log`
- `results/sglang_qwen_mixedkv_prefix4096_eager_notrace_sweepcorpus_20260610TmanualJST_mixed_server.log`

Eager trace control:

- `results/sglang_qwen_mixedkv_prefixref_ctx8192_prefix4096_eager_20260610TmanualJST_compare.json`
- `results/sglang_qwen_mixedkv_prefixref_ctx8192_prefix4096_eager_20260610TmanualJST_mixed_server.log`

## Result

| mode | fp8 PPL | mixed PPL | delta nats/token | fp8 cached | mixed cached |
|---|---:|---:|---:|---:|---:|
| Graph-enabled sweep | 7.238053 | 8.052973 | +0.106689 | 4096 | 4096 |
| Eager no-trace control | 5.176347 | 5.169650 | -0.001295 | 4096 | 4096 |
| Eager prefix-reference trace | 5.176347 | 5.169650 | -0.001295 | 4096 | 4096 |

The eager no-trace and eager traced rows match exactly. That means the trace envs are not
responsible for the corrected PPL row.

## Runtime Evidence

In the graph-enabled baseline, the cache-populating warmup prefill used graph replay:

```text
#new-token: 4096, #cached-token: 0, cuda graph: True
#new-token: 4096, #cached-token: 4096, cuda graph: False
```

In the eager no-trace control, both cache population and cached-prefix scoring ran eager:

```text
#new-token: 4096, #cached-token: 0, cuda graph: False
#new-token: 4096, #cached-token: 4096, cuda graph: False
```

This changes both fp8 and mixed-KV PPL on the same requested corpus path. The scored
cached-prefix request was eager in both runs, so the stale conclusion "the 8k loss is not
graph-related because the scored request ran eager" was incomplete. The graph-sensitive
operation is the prefix-cache writer, not necessarily the later cached-prefix reader.

The eager trace row also showed the mixed-KV cached-prefix read itself matches the
standalone reference for the sampled layer-0 rows:

```text
o2_compare cosine: 0.999997
s2_compare cosine: 1.0
full_prefix_fp4_suffix_bf16_vs_merged cosine: 0.999998
```

The dense reference inside that trace was too short for a full dense-vs-cached comparison,
so this is not a complete tensor-equivalence proof. It is enough to say the 4096-prefix
PPL failure from the graph-enabled sweep is not reproduced when the prefix cache is written
eagerly.

## Interpretation

The graph-enabled fixed-8k reuse-prefix sweep remains useful as a failure detector, but it
should no longer be quoted as a pure mixed-KV NVFP4-V quality loss. The controlled result
points at CUDA-graph cache population state:

- Graph-written prefix cache: fp8 and mixed-KV PPL both shift, and mixed-KV shows a large
  relative penalty on the 4096-prefix row.
- Eager-written prefix cache: fp8 and mixed-KV are effectively equal on the same
  4096-prefix test.

Current SGLang policy for Qwen mixed-KV on GB10 should therefore be stricter than the
previous guard: disable CUDA graph capture for cache-populating prefills as well as
cached-prefix prefills until the graph-write path is repaired. Mixed-KV remains
capacity-positive at about `1.78x`, but graph-enabled quality rows are not blessable yet.
