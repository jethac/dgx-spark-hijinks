# SGLang Qwen FP4-KV Prefix-Reference Trace Packet, 2026-06-08

Run this on the GB10 host only when the GPU is free.

Purpose: rerun the known SGLang Qwen default-vs-radix-off FP4 first-token probe with
`jethac/sglang@a8e8de26d` and the cached-prefix reference comparator enabled. This is a
diagnostic packet, not a benchmark row.

Expected source pin:

- `third_party/sglang`: `a8e8de26db`

Required FP4 trace envs for the default row:

```bash
SGLANG_FP4_KV_TRACE_BACKEND=1
SGLANG_FP4_KV_TRACE_RADIX=1
SGLANG_FP4_KV_TRACE_PAGE_PAIR=1
SGLANG_FP4_KV_TRACE_MERGE_STATE=1
SGLANG_FP4_KV_TRACE_WRITE_READ=1
SGLANG_FP4_KV_TRACE_PREFIX_REF=1
SGLANG_FP4_KV_TRACE_LAYERS=0
SGLANG_FP4_KV_TRACE_VALUES=16
SGLANG_FP4_KV_TRACE_LOC_LIMIT=128
SGLANG_FP4_KV_PREFIX_REF_MAX_TOKENS=128
```

Use the same source-overlay command shape as
`results/sglang_qwen_fp4kv_write_read_trace_20260608T222204JST_summary.md`:

- NVIDIA SGLang 26.05 container
- editable `third_party/sglang` source overlay
- `--attention-backend flashinfer`
- `--kv-cache-dtype fp4_e2m1`
- page size `1`
- CUDA graph and piecewise graph disabled
- default FP4 row first, then `--disable-radix-cache` control

Pass/fail:

- Default row should reproduce the known first-token failure with `cached_tokens=55`.
- Radix-off row should still pass with `cached_tokens=0`.
- Default server log must contain `FP4 KV prefix-reference trace`.
- If `o2_compare` or `s2_compare` is bad, the defect is in native FP4 paged-prefix
  read/layout/scale semantics.
- If `o2/s2` compare but `merge_compare` is bad, the defect is in LSE merge integration.
- If all comparator values match and quality still fails, move to quantization-error and
  calibration-impact probes at `MHATokenToKVPoolFP4.set_kv_buffer()`.
