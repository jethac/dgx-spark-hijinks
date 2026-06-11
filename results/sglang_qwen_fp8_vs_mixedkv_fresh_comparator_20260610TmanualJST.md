# SGLang Qwen fp8 vs mixed-KV fresh comparator, 2026-06-10

## Scope

This row is the fresh sequential capacity/speed comparator requested after the mixed-KV
default-radix first-token gate passed.

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Attention backend: FlashInfer
- Page size: `1`
- `--mem-fraction-static`: `0.40`
- Docker memory cap: `--memory=100g --memory-swap=100g`
- CUDA graph: disabled
- Runs were sequential, not concurrent.

The fp8 row used `KV_CACHE_DTYPE=fp8_e4m3`. The mixed row used
`KV_CACHE_DTYPE=fp4_e2m1` with `SGLANG_FP4_KV_MIXED_KV=1`, meaning:

- K cache: FP8 e4m3
- V cache: packed NVFP4 with FP8 scale factors

This is a mixed-KV result, not a full NVFP4 K+V result.

## Artifacts

- fp8 manifest: `results/sglang_qwen_fp8_fresh_comparator_20260610TmanualJST_row_manifest.json`
- fp8 benchmark: `results/sglang_qwen_fp8_fresh_comparator_20260610TmanualJST_openai_benchmark.json`
- fp8 server log: `results/sglang_qwen_fp8_fresh_comparator_20260610TmanualJST_server.log`
- mixed manifest: `results/sglang_qwen_mixedkv_fresh_comparator_20260610TmanualJST_row_manifest.json`
- mixed benchmark: `results/sglang_qwen_mixedkv_fresh_comparator_20260610TmanualJST_openai_benchmark.json`
- mixed server log: `results/sglang_qwen_mixedkv_fresh_comparator_20260610TmanualJST_server.log`

## Result

| KV mode | allocatable tokens | K size | V size | short decode tok/s | TTFT |
|---|---:|---:|---:|---:|---:|
| fp8 K + fp8 V | 3,119,168 | 20.82 GB | 20.82 GB | 57.594 | 0.966 s |
| fp8 K + NVFP4 V | 5,537,968 | 36.97 GB | 20.80 GB | 57.804 | 0.931 s |

Observed allocator-token ratio at the same SGLang launch settings:

```text
5,537,968 / 3,119,168 = 1.775x
```

Short-decode throughput ratio:

```text
57.804 / 57.594 = 1.004x
```

The normalized bytes-per-token improvement is the expected mixed-KV improvement, about
`1.28x`, because K remains fp8 while V is packed NVFP4 plus scale factors. The larger
`1.775x` token count is the observed SGLang allocator result under the same launch
settings, not a pure per-token storage-ratio claim.

## Interpretation

This closes the fresh comparator requested by
`results/sglang_qwen_mixedkv_default_20260610T0042JST_summary.md`: mixed-KV keeps short
decode at fp8 parity while exposing substantially more allocatable KV tokens on the same
GB10 run configuration.

The quality claim still comes from the earlier default-radix first-token gate:
`results/sglang_qwen_mixedkv_default_20260610T0042JST_summary.md`. That gate fixed the
observed cached-prefix token flip (`ark` back to `**`) with radix cache ON. This comparator
does not replace a long-form quality/PPL gate.

## Remaining gates

- Long-form coherent generation with radix cache ON.
- Short PPL or supplied-token logprob gate for mixed-KV vs fp8.
- Graph-safety row if graph capture remains a target.
- Full NVFP4 K+V remains a separate research branch; do not merge it with this mixed-KV
  claim.
