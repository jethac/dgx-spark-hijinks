# SGLang Qwen FP4 KV d7d931f Matched Row

Run id prefix: `sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST`.

This live GB10 source-overlay run records a matched SGLang `fp8_e4m3`-vs-`fp4_e2m1`
comparison on `jethac/sglang@d7d931f530160ba86a2d55b4636d64baaeda3bec` with graph
modes disabled for both rows. The FP4 row enables `SGLANG_FP4_KV_TRACE_BACKEND=1`.

## Setup

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- source overlay: `jethac/sglang@d7d931f530160ba86a2d55b4636d64baaeda3bec`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- common flags: `--tp 1 --dtype bfloat16 --attention-backend flashinfer --page-size 1 --mem-fraction-static 0.40 --disable-cuda-graph --disable-piecewise-cuda-graph`
- escape hatch: `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`
- FP4 trace knob: `SGLANG_FP4_KV_TRACE_BACKEND=1`

## Matched Result

| row | KV dtype | KV tokens | short decode | medium decode | long-prefill decode | smoke |
|---|---|---:|---:|---:|---:|---|
| fp8 | `fp8_e4m3` | `3,105,240` | `56.996 tok/s` | `57.034 tok/s` | `57.266 tok/s` | raw `2+2` and chat pass |
| fp4 | `fp4_e2m1` | `5,517,572` | `93.734 tok/s` | `55.647 tok/s` | `55.982 tok/s` | raw `2+2` and chat pass |

The FP4 KV pool is `1.7769x` the fp8 pool at the same memory fraction.

Raw `2+2 is` sanity:

- fp8: ` 4. 2+2 is 4. 2+2 is`
- fp4: ` 4, 2+2 is 4, 2+2 is`

Chat smoke:

- fp8: `spark-ok`
- fp4: `spark-ok`

## Backend Trace

The FP4 log records native FP4 KV through packed K/V plus separate scale buffers:

- `KV Cache is allocated. dtype: torch.float4_e2m1fn_x2, #tokens: 5517572`
- `NVFP4 KV cache calibrated 28 layers from 4096 eager prefill tokens`
- 28 `label=decode` traces
- 28 `label=extend_merge_paged` traces
- K/V cache tensor: `torch.uint8`, shape `(5517573, 2, 64)`, stride `(128, 64, 1)`
- K/V scale tensors: `torch.float8_e4m3fn`, shape `(5517573, 1, 2, 8)`, stride `(16, 16, 8, 1)`

This proves the serving backend is exercising the packed FP4 KV path for both decode
and a request-path paged extend/merge call in this row.

## Verdict

This is a stronger SGLang FP4 KV capacity/routing proof than the earlier autosafe and
decode-only trace rows: it has a matched fp8 comparator, records `1.7769x` KV capacity,
and captures both decode and `extend_merge_paged` backend calls.

It is still **not a blessed SGLang FP4 KV serving-quality row**. The raw and chat smoke
checks pass, but the standardized FP4 benchmark text remains degraded: `short_decode`
stops after `,explain why.`, and `medium_decode` / `long_prefill` are repetitive or
garbled while the fp8 comparator produces normal text. Treat FP4 throughput from this row
as diagnostic only until a quality comparator passes.

## Artifacts

- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp8_row_manifest.json`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp8_openai_benchmark.json`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp8_raw_2plus2.json`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp8_chat_smoke.json`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp8_server.log`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp8_trace_excerpt.txt`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp4_row_manifest.json`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp4_openai_benchmark.json`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp4_raw_2plus2.json`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp4_chat_smoke.json`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp4_server.log`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp4_trace_excerpt.txt`
