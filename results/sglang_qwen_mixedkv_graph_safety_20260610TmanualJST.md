# SGLang Qwen mixed-KV CUDA graph safety check, 2026-06-10

## Scope

This checks whether the practical SGLang mixed-KV route can be blessed with CUDA graph
capture enabled.

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Attention backend: FlashInfer
- Page size: `1`
- KV setting: `KV_CACHE_DTYPE=fp4_e2m1`, `SGLANG_FP4_KV_MIXED_KV=1`
- CUDA graphs: enabled with `SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=1`
- SGLang flags: `disable_cuda_graph=False`, `disable_piecewise_cuda_graph=False`
- `--mem-fraction-static`: `0.40`
- Docker cap: `--memory=100g --memory-swap=100g`

Mixed-KV means K is FP8 e4m3 and V is packed NVFP4 with FP8 scale factors.

## Artifacts

Short smoke:

- benchmark: `results/sglang_qwen_mixedkv_graph_short_20260610TmanualJST_openai_benchmark.json`
- manifest: `results/sglang_qwen_mixedkv_graph_short_20260610TmanualJST_row_manifest.json`
- server log: `results/sglang_qwen_mixedkv_graph_short_20260610TmanualJST_server.log`

Three-case gate:

- benchmark: `results/sglang_qwen_mixedkv_graph_quality3_20260610TmanualJST_openai_benchmark.json`
- manifest: `results/sglang_qwen_mixedkv_graph_quality3_20260610TmanualJST_row_manifest.json`
- quality probe: `results/sglang_qwen_mixedkv_graph_quality3_20260610TmanualJST_quality.json`
- server log: `results/sglang_qwen_mixedkv_graph_quality3_20260610TmanualJST_server.log`

Isolation controls:

- isolated natural-long-prefill benchmark:
  `results/sglang_qwen_mixedkv_graph_natural_isolated_20260610TmanualJST_openai_benchmark.json`
- isolated natural-long-prefill quality:
  `results/sglang_qwen_mixedkv_graph_natural_isolated_20260610TmanualJST_quality.json`
- no-radix three-case benchmark:
  `results/sglang_qwen_mixedkv_graph_quality3_noradix_20260610TmanualJST_openai_benchmark.json`
- no-radix three-case manifest:
  `results/sglang_qwen_mixedkv_graph_quality3_noradix_20260610TmanualJST_row_manifest.json`
- no-radix three-case quality:
  `results/sglang_qwen_mixedkv_graph_quality3_noradix_20260610TmanualJST_quality.json`
- no-radix server log:
  `results/sglang_qwen_mixedkv_graph_quality3_noradix_20260610TmanualJST_server.log`

Guarded default-radix graph row:

- benchmark:
  `results/sglang_qwen_mixedkv_graph_quality3_guarded_20260610TmanualJST_openai_benchmark.json`
- manifest:
  `results/sglang_qwen_mixedkv_graph_quality3_guarded_20260610TmanualJST_row_manifest.json`
- quality probe:
  `results/sglang_qwen_mixedkv_graph_quality3_guarded_20260610TmanualJST_quality.json`
- server log:
  `results/sglang_qwen_mixedkv_graph_quality3_guarded_20260610TmanualJST_server.log`

## Graph evidence

The server did not silently fall back to eager mode. The graph-enabled three-case row logged:

```text
disable_cuda_graph=False
disable_piecewise_cuda_graph=False
NVFP4 KV cache calibrated 28 layers from 4096 eager prefill tokens
Capture cuda graph begin
Capture cuda graph end. Time elapsed: 2.51 s. mem usage=1.22 GB
Capture piecewise CUDA graph begin
Capture piecewise CUDA graph end. Time elapsed: 16.79 s. mem usage=0.49 GB
cuda graph: True
```

## Result

The graph-enabled short smoke is green:

| run | cases | allocatable tokens | decode tok/s | quality |
|---|---:|---:|---:|---|
| `graph_short` | short | 5,544,705 | 58.919 | pass |

The stronger graph-enabled three-case row is partial/red:

| case | prompt tokens | completion tokens | finish reason | decode tok/s | quality |
|---|---:|---:|---|---:|---|
| `short_decode` | 44 | 64 | length | 58.626 | pass |
| `medium_decode` | 56 | 192 | length | 57.639 | pass |
| `natural_long_prefill` | 687 | 4 | stop, `matched_stop=151645` | 75.991 | fail |

The failing `natural_long_prefill` output is:

```text
1 .
```

The quality probe flags it as:

```json
["too_short_for_requested_decode"]
```

## Isolation

Two controls isolate the failure to graph-enabled radix reuse, not graph replay alone.

First, `natural_long_prefill` was run alone with graphs enabled. The server log shows the
test request had no radix hit:

```text
Prefill batch, #new-token: 687, #cached-token: 0, cuda graph: True
```

That row passed:

| run | cached tokens | completion tokens | finish reason | decode tok/s | quality |
|---|---:|---:|---|---:|---|
| `graph_natural_isolated` | 0 | 128 | length | 57.901 | pass |

Second, the full three-case sequence was rerun with `--disable-radix-cache`. The server
used `ChunkCache`, not `RadixCache`, and every benchmark request had `#cached-token: 0`.
That row also passed:

| case | cached tokens | completion tokens | finish reason | decode tok/s | quality |
|---|---:|---:|---|---:|---|
| `short_decode` | 0 | 64 | length | 59.062 | pass |
| `medium_decode` | 0 | 192 | length | 57.630 | pass |
| `natural_long_prefill` | 0 | 128 | length | 57.601 | pass |

## Guarded fix

The SGLang fork now carries a narrow safety guard in
`third_party/sglang/python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`:
when `SGLANG_FP4_KV_MIXED_KV=1` and the EXTEND batch has a nonzero cached prefix, the
piecewise CUDA graph runner returns `False` from `can_run()`. This keeps CUDA graph replay
enabled for no-prefix prefill and decode, while routing the unsafe mixed-KV cached-prefix
prefill shape through eager mode.

The guarded default-radix row passes the same three-case gate with CUDA graphs enabled and
radix cache still on:

| case | cached tokens | completion tokens | finish reason | decode tok/s | quality |
|---|---:|---:|---|---:|---|
| `short_decode` | 0 | 64 | length | 58.142 | pass |
| `medium_decode` | 24 | 192 | length | 57.634 | pass |
| `natural_long_prefill` | 24 | 128 | length | 57.723 | pass |

The server log proves the intended graph/eager split:

```text
Prefill batch, #new-token: 44, #cached-token: 0, cuda graph: True
Prefill batch, #new-token: 32, #cached-token: 24, cuda graph: False
Decode batch, #running-req: 1, #token: 104, cuda graph: True
Prefill batch, #new-token: 663, #cached-token: 24, cuda graph: False
Decode batch, #running-req: 1, #token: 743, cuda graph: True
```

Capacity in the guarded row remains in the mixed-KV regime:

```text
KV Cache is allocated. dtype: torch.float4_e2m1fn_x2, #tokens: 5548188,
K size: 37.04 GB, V size: 20.83 GB
```

## Interpretation

CUDA graph capture itself works for the SGLang mixed-KV stack on GB10: the server reaches
readiness, captures both full and piecewise graphs, and runs no-prefix prefill plus decode
with `cuda graph: True`.

The original default-radix graph row stopped after four completion tokens only when the
long prompt reused a 24-token cached prefix from earlier benchmark requests. Isolated
no-cache graph serving and no-radix graph serving both completed normally. The guarded
row confirms the failure surface: routing only mixed-KV cached-prefix prefill out of graph
mode is enough to make the default radix sequence pass while keeping decode graphs enabled.

Current localization: SGLang mixed-KV graph replay is compatible with long prefill when no
prefix is reused, but the combination of piecewise CUDA graph replay plus radix/prefix
reuse can produce an early stop token. The guard is therefore a tactical correctness fix
for serving, not the final root-cause repair of cached-prefix graph replay metadata.

Current policy: the guarded path is the practical graph-enabled SGLang mixed-KV route for
Qwen on GB10. `--disable-radix-cache` remains a diagnostic control, not a blessable serving
result. A future root-cause fix should inspect graph-captured radix/prefix-cache state for
mixed-KV, especially how cached-prefix metadata and stop-token/logit state are replayed
after a prior request.
