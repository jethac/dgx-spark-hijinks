# SGLang Qwen FP4-KV Prefix-Reference Trace, 2026-06-08 23:06 JST

Status: red serving path; cached FP4 paged-prefix read and merge are numerically clean for
the sampled failing request.

Purpose: run the `jethac/sglang@2a228949a` prefix-reference comparator after the
write/read trace cleared simple stale/wrong-page packed-KV or scale-buffer mismatch.

Runtime:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- source overlay: `jethac/sglang@2a228949a31c5f36776cdb4d7b286dc03e6f1e89`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- KV dtype: `fp4_e2m1`
- attention backend: `flashinfer`
- CUDA graphs: disabled
- page size: `1`
- probe: `medium_decode`, `max_new_tokens=1`, `temperature=0`

Rows:

| row | OpenAI first token | native first token | match | native cached tokens |
|---|---|---|---|---:|
| default FP4 | `**` | `ark` / `838` | fail | 55 |
| radix-off FP4 | `**` | `**` / `334` | pass | 0 |

Prefix-reference result for default layer 0:

| comparison | cosine | max abs | rms | verdict |
|---|---:|---:|---:|---|
| cached paged `o2` vs dequantized torch reference | 0.9999971986 | 0.0078125 | 0.0006038767 | clean |
| FlashInfer-base `s2` vs reference LSE converted to log2 units | 1.0000001192 | 0.001953125 | 0.0006718715 | clean |
| manual `exp2` merge vs `_safe_merge_state` output | 0.9999998808 | 0.0 | 0.0 | clean |

The natural-log LSE comparison intentionally does **not** match:

- natural-log `s2` max abs: `6137.994140625`
- natural-log `s2` rms: `2360.789794921875`

This is a unit convention, not a defect: FlashInfer returns the LSE in log2 units for this
path, so the valid comparator is `logsumexp(...) * 1/log(2)`.

Interpretation:

- The native FP4 paged-prefix reader correctly consumes the cached packed K/V and FP8 scale
  buffers for the sampled failing request.
- `_safe_merge_state` also matches a manual base-2 merge exactly at BF16 output precision.
- The remaining SGLang Qwen FP4-KV quality bug is not explained by radix page IDs, sampled
  write/read bytes, FP4 paged-prefix dequant/layout, LSE units, or merge-state math.
- Next suspect moves upward: prefix-cache request sequencing/state across OpenAI/native
  calls, calibration/quantization error impact after cache fill, or logits/model state above
  attention.

Artifacts:

- parsed summary: `results/sglang_qwen_fp4kv_prefix_ref_trace_20260608T2306JST_parsed.json`
- default endpoint JSON: `results/sglang_qwen_fp4kv_prefix_ref_trace_20260608T2306JST_default.json`
- default server log: `results/sglang_qwen_fp4kv_prefix_ref_trace_20260608T2306JST_default_fp4_server.log`
- default prefix-reference line:
  `results/sglang_qwen_fp4kv_prefix_ref_trace_20260608T2306JST_default_prefix_ref_lines.txt`
- radix-off endpoint JSON:
  `results/sglang_qwen_fp4kv_prefix_ref_trace_20260608T2306JST_radixoff.json`
- radix-off server log:
  `results/sglang_qwen_fp4kv_prefix_ref_trace_20260608T2306JST_radixoff_fp4_server.log`
- container inspect files:
  `results/sglang_qwen_fp4kv_prefix_ref_trace_20260608T2306JST_default_fp4_container_inspect.json`,
  `results/sglang_qwen_fp4kv_prefix_ref_trace_20260608T2306JST_radixoff_fp4_container_inspect.json`
