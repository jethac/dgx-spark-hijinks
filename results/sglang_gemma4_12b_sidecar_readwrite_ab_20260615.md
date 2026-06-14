# SGLang Gemma 4 12B NVFP4 Sidecar Read/Write A/B

Date: 2026-06-15 JST

Scope: SGLang Gemma 4 12B AR, text-only, full NVFP4 K+V, radix/cache reuse on, CUDA graphs disabled, source overlay on the Spark source-stack image.

Image: `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0bacd437f9917928a9bd7ba0dafbb37516f8e05b4b9727bbff796556c2cc7714`

Model: `google/gemma-4-12B-it`

Run shape: `ctx=8185`, reused prefix `4096`, logprob span starts at `4096`, `context_length=8192`, `mem_fraction_static=0.72`, Docker memory cap `100g`.

Calibration sidecar: `results/sglang_nvfp4_kv_calib_gemma4_12b_20260615/Gemma4UnifiedForConditionalGeneration-L48-H3840-D256-KV8.json`, `k_global_scale=0.1`, `v_global_scale=0.1`.

## Rows

| row | SGLang overlay | calibration path | NLL | PPL | cached tokens | trace result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| old sidecar | `b85790b951` | sidecar directly seeded layers + pools | 4.6969718400769205 | 109.61473841658791 | 4096 | 48 read/write traces, no auto-calibration |
| fixed env | `b85790b951` | `SGLANG_FP4_KV_FIXED_GLOBAL_SCALE=0.1` through warmup calibration | 4.6374074715758535 | 103.2762535898204 | 4096 | 48 auto-calibrations, 48 read/write traces |
| pool-only sidecar | `e7467439d1` | sidecar directly seeded native pools only | 4.6969718400769205 | 109.61473841658791 | 4096 | 48 read/write traces, no auto-calibration |
| deferred sidecar | `65a3d251b0` | sidecar registered as calibration fixed scale, warmup writes pools | 4.6374074715758535 | 103.2762535898204 | 4096 | 48 auto-calibrations, 48 read/write traces |

## Interpretation

Claude's read/write mismatch hypothesis was falsified for the visible scale values: old sidecar, pool-only sidecar, fixed-env, and deferred sidecar all handed `0.1`/`0.10000000149011612` to the cached FP4 read path.

The sidecar gap was instead caused by bypassing `_maybe_calibrate_global_scales()` during the eager calibration write. Directly seeding the pool, even with layer attrs removed, preserved the worse sidecar NLL. Routing sidecar scales through the same fixed-scale calibration path as `SGLANG_FP4_KV_FIXED_GLOBAL_SCALE=0.1` made the sidecar row bit-for-bit equal in NLL/PPL to the fixed-env diagnostic.

The fixed SGLang path is `jethac/sglang@spark/hijinks-025-sglang-0.5.13-rebase` commit `65a3d251b0`, which adds a sidecar fixed-scale source for `_fp4_kv_fixed_global_scale()` and lets the existing warmup calibration write the native FP4 pools.

## Artifacts

- `results/sglang_gemma4_12b_sidecarcalib010_readwrite_ctx8185_prefix4096_20260615T081751JST/`
- `results/sglang_gemma4_12b_fixedgs010_readwrite_ctx8185_prefix4096_20260615T081751JST/`
- `results/sglang_gemma4_12b_sidecarpoolonly010_readwrite_ctx8185_prefix4096_20260615T083142JST/`
- `results/sglang_gemma4_12b_sidecardeferred010_readwrite_ctx8185_prefix4096_20260615T084646JST/`

Excluded setup failure: `results/sglang_gemma4_12b_sidecarcalib010_readwrite_ctx8185_prefix4096_20260615T081451JST/` failed before model load because the wrapper mounted an empty HF cache path. It is not a model-quality row.
