# SGLang Qwen mixed-KV supplied-token PPL, 2026-06-10

## Scope

This is the first SGLang supplied-token logprob quality gate for the practical
mixed-KV path.

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Attention backend: FlashInfer
- Page size: `1`
- Memory fraction: `0.40`
- Docker cap: `--memory=100g --memory-swap=100g`
- Graph policy: CUDA graphs enabled; mixed-KV cached-prefix prefill guarded to eager
- Scoring API: SGLang native `/generate`, token-ID prompt, `return_logprob=true`

The scorer validates supplied token IDs, not generated-token top-logprobs. For reuse rows it
warms a prefix, then scores the continuation span with `logprob_start_len` set to the prefix
length.

## Artifacts

Harness:

- `scripts/sglang_prompt_ppl_sweep.py`
- `scripts/run_sglang_qwen_ppl_pair.sh`

No-reuse harness smoke:

- manifest: `results/sglang_qwen_mixedkv_ppl_20260610TmanualJST_manifest.json`
- compare: `results/sglang_qwen_mixedkv_ppl_20260610TmanualJST_compare.json`
- fp8: `results/sglang_qwen_mixedkv_ppl_20260610TmanualJST_fp8_ppl.json`
- mixed: `results/sglang_qwen_mixedkv_ppl_20260610TmanualJST_mixed_ppl.json`

Reuse PPL row:

- manifest: `results/sglang_qwen_mixedkv_reuse_ppl2_20260610TmanualJST_manifest.json`
- compare: `results/sglang_qwen_mixedkv_reuse_ppl2_20260610TmanualJST_compare.json`
- fp8: `results/sglang_qwen_mixedkv_reuse_ppl2_20260610TmanualJST_fp8_ppl.json`
- mixed: `results/sglang_qwen_mixedkv_reuse_ppl2_20260610TmanualJST_mixed_ppl.json`
- fp8 server log: `results/sglang_qwen_mixedkv_reuse_ppl2_20260610TmanualJST_fp8_server.log`
- mixed server log: `results/sglang_qwen_mixedkv_reuse_ppl2_20260610TmanualJST_mixed_server.log`

Longer reuse PPL row:

- manifest: `results/sglang_qwen_mixedkv_reuse_ppl_ctx2048_20260610TmanualJST_manifest.json`
- compare: `results/sglang_qwen_mixedkv_reuse_ppl_ctx2048_20260610TmanualJST_compare.json`
- fp8: `results/sglang_qwen_mixedkv_reuse_ppl_ctx2048_20260610TmanualJST_fp8_ppl.json`
- mixed: `results/sglang_qwen_mixedkv_reuse_ppl_ctx2048_20260610TmanualJST_mixed_ppl.json`
- fp8 server log: `results/sglang_qwen_mixedkv_reuse_ppl_ctx2048_20260610TmanualJST_fp8_server.log`
- mixed server log: `results/sglang_qwen_mixedkv_reuse_ppl_ctx2048_20260610TmanualJST_mixed_server.log`

8k reuse PPL row:

- manifest: `results/sglang_qwen_mixedkv_reuse_ppl_ctx8192_20260610TmanualJST_manifest.json`
- compare: `results/sglang_qwen_mixedkv_reuse_ppl_ctx8192_20260610TmanualJST_compare.json`
- fp8: `results/sglang_qwen_mixedkv_reuse_ppl_ctx8192_20260610TmanualJST_fp8_ppl.json`
- mixed: `results/sglang_qwen_mixedkv_reuse_ppl_ctx8192_20260610TmanualJST_mixed_ppl.json`
- fp8 server log: `results/sglang_qwen_mixedkv_reuse_ppl_ctx8192_20260610TmanualJST_fp8_server.log`
- mixed server log: `results/sglang_qwen_mixedkv_reuse_ppl_ctx8192_20260610TmanualJST_mixed_server.log`

8k token-logprob diagnostic:

- summary: `results/sglang_qwen_mixedkv_8k_token_logprob_diagnostic_20260610TmanualJST.md`
- token comparison: `results/sglang_qwen_mixedkv_reuse_ppl_ctx8192_detail_20260610TmanualJST_token_compare.json`
- fp8 detail: `results/sglang_qwen_mixedkv_reuse_ppl_ctx8192_detail_20260610TmanualJST_fp8_ppl.json`
- mixed detail: `results/sglang_qwen_mixedkv_reuse_ppl_ctx8192_detail_20260610TmanualJST_mixed_ppl.json`

8k no-reuse control:

- manifest: `results/sglang_qwen_mixedkv_noreuse_ppl_ctx8192_20260610TmanualJST_manifest.json`
- compare: `results/sglang_qwen_mixedkv_noreuse_ppl_ctx8192_20260610TmanualJST_compare.json`
- fp8: `results/sglang_qwen_mixedkv_noreuse_ppl_ctx8192_20260610TmanualJST_fp8_ppl.json`
- mixed: `results/sglang_qwen_mixedkv_noreuse_ppl_ctx8192_20260610TmanualJST_mixed_ppl.json`
- fp8 server log: `results/sglang_qwen_mixedkv_noreuse_ppl_ctx8192_20260610TmanualJST_fp8_server.log`
- mixed server log: `results/sglang_qwen_mixedkv_noreuse_ppl_ctx8192_20260610TmanualJST_mixed_server.log`

Failed strict scorer row, preserved as harness-debug evidence:

- `results/sglang_qwen_mixedkv_reuse_ppl_20260610TmanualJST_fp8_ppl.json`

That row proved `cached_tokens=256`, but the initial scorer treated SGLang's leading
logprob-span boundary placeholder as a token mismatch. The scorer now records that boundary
as `num_skipped_boundary_tokens=1` and scores the remaining continuation tokens.

## Result

No-reuse supplied-token smoke passed but is not a KV-reuse quality claim:

| ctx | cached tokens | PPL fp8 | PPL mixed | delta nats/token |
|---:|---:|---:|---:|---:|
| 512 | 0 | 21.086242 | 21.086242 | 0.000000 |
| 8192 | 0 | 7.195014 | 7.195014 | 0.000000 |

The accepted reuse rows warm a prefix and score the remaining continuation tokens. Both
fp8 and mixed-KV prove the expected `cached_tokens` in the response and server log:

| ctx | reused prefix | scored tokens | PPL fp8 | PPL mixed | delta PPL | delta nats/token |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 256 | 255 | 18.453757 | 18.431295 | -0.022462 | -0.001218 |
| 2048 | 1024 | 1023 | 28.140193 | 28.392845 | 0.252652 | 0.008938 |
| 8192 | 4096 | 4095 | 7.238053 | 8.056450 | 0.818397 | 0.107121 |

Both rows have:

- `num_missing_tokens=0`
- `num_mismatched_tokens=0`
- `num_skipped_boundary_tokens=1` at the logprob-span boundary
- `ok=true`

Capacity in the 512-token launch settings:

| KV mode | allocatable tokens | K size | V size |
|---|---:|---:|---:|
| fp8 K + fp8 V | 3,116,067 | 20.80 GB | 20.80 GB |
| fp8 K + NVFP4 V | 5,552,677 | 37.07 GB | 20.85 GB |

Observed allocator-token ratio: `1.782x`.

Capacity in the 2048-token launch settings:

| KV mode | allocatable tokens | K size | V size |
|---|---:|---:|---:|
| fp8 K + fp8 V | 3,117,451 | 20.81 GB | 20.81 GB |
| fp8 K + NVFP4 V | 5,560,832 | 37.12 GB | 20.88 GB |

Observed allocator-token ratio: `1.784x`.

Capacity in the 8192-token launch settings:

| KV mode | allocatable tokens | K size | V size |
|---|---:|---:|---:|
| fp8 K + fp8 V | 3,119,879 | 20.83 GB | 20.83 GB |
| fp8 K + NVFP4 V | 5,551,389 | 37.06 GB | 20.85 GB |

Observed allocator-token ratio: `1.779x`.

Capacity in the 8192-token no-reuse control launch settings:

| KV mode | allocatable tokens | K size | V size |
|---|---:|---:|---:|
| fp8 K + fp8 V | 3,116,223 | 20.80 GB | 20.80 GB |
| fp8 K + NVFP4 V | 5,542,470 | 37.00 GB | 20.81 GB |

Observed allocator-token ratio: `1.779x`.

## Runtime Evidence

512-token fp8 scored request:

```text
Prefill batch, #new-token: 256, #cached-token: 256, cuda graph: False
```

512-token mixed-KV scored request:

```text
SGLang FP4 KV mixed mode enabled: K cache uses FP8 e4m3, V cache uses packed NVFP4.
Prefill batch, #new-token: 256, #cached-token: 256, cuda graph: False
```

Warmup prefix requests used `#cached-token: 0` and graph replay, as expected:

```text
Prefill batch, #new-token: 256, #cached-token: 0, cuda graph: True
```

2048-token scored requests:

```text
fp8:   Prefill batch, #new-token: 1024, #cached-token: 1024, cuda graph: False
mixed: Prefill batch, #new-token: 1024, #cached-token: 1024, cuda graph: False
```

The 2048-token warmup prefix requests used `#cached-token: 0` and graph replay:

```text
fp8:   Prefill batch, #new-token: 1024, #cached-token: 0, cuda graph: True
mixed: Prefill batch, #new-token: 1024, #cached-token: 0, cuda graph: True
```

8192-token scored requests:

```text
fp8:   Prefill batch, #new-token: 4096, #cached-token: 4096, cuda graph: False
mixed: Prefill batch, #new-token: 4096, #cached-token: 4096, cuda graph: False
```

The 8192-token warmup prefix requests used `#cached-token: 0` and graph replay:

```text
fp8:   Prefill batch, #new-token: 4096, #cached-token: 0, cuda graph: True
mixed: Prefill batch, #new-token: 4096, #cached-token: 0, cuda graph: True
```

8192-token no-reuse control scored requests:

```text
fp8:   Prefill batch, #new-token: 8192, #cached-token: 0, cuda graph: False
mixed: Prefill batch, #new-token: 8192, #cached-token: 0, cuda graph: False
```

Interpretation: the guarded mixed-KV path now has three distinct green gates on Qwen2.5
1.5B: default-radix first-token stability, graph-enabled three-case serving quality, and a
supplied-token PPL row that actually reuses a cached prefix. The PPL gate now covers
context 512 / prefix 256, context 2048 / prefix 1024, and context 8192 / prefix 4096.
The 8k row is mechanically correct but not quality-negligible: mixed-KV is `+0.107121`
nats/token versus fp8. Treat 512/2048 as green smoke-scale rows and 8192 as the first
material quality-regression signal that needs explanation before claiming long-context
quality. This is still scoped to Qwen, page size 1, single GB10, and mixed FP8-K +
NVFP4-V. It does not prove full NVFP4 K+V, Gemma/SWA, 32k+ contexts, TP>1, or
MTP/spec-decode.

Follow-up: these artifacts intentionally store aggregate PPL only, so they cannot localize
whether the 8k loss is broad or concentrated in a small span. The harness now has an
opt-in `INCLUDE_TOKEN_LOGPROBS=1` path for the next 8k/32k diagnostic rerun.

The 8k diagnostic rerun with `INCLUDE_TOKEN_LOGPROBS=1` reproduces the loss at `+0.106689`
nats/token and shows it is skewed toward the first continuation span after cache reuse:
the first 1024 scored tokens after the 4096-token cached prefix account for about `55.9%`
of the total delta nats. Do not treat 32k PPL as the next blind scale-up; first explain the
cached-prefix transition effect or run a targeted diagnostic against it.

The 8k no-reuse control then isolates the loss to cached-prefix reuse. With the same model,
runtime image, mixed-KV allocation, graph policy, and an 8192-token scored prompt, but
`reuse_prefix_len=0`, fp8 and mixed-KV are exactly equal: PPL `7.195014` versus `7.195014`,
`delta_nats_per_token=0.0`, and both server logs show `#cached-token: 0`. This falsifies a
broader "mixed-KV long-prefill is inherently worse at 8k" explanation for this row.
