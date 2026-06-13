# SGLang E4B fp8 comparator red - FlashInfer dispatch analysis

Date: 2026-06-14 05:41 JST

Status: **scoped red, points at FlashInfer FA2 paged-prefill dispatch**

## Context

- Prior live artifact: `results/sglang_0513_fix_gemma4_e4b_fp8_comparator_ctx512_prefix256_20260614T032241JST/STOP_SUMMARY.md`
- Model: `google/gemma-4-E4B-it`
- Runtime: SGLang 0.5.13 Spark image, Ubuntu 22 / arm64 / torch 2.11 packaged stack
- SGLang ref: `58a39849fc`
- FlashInfer ref: `f99323bd`
- KV dtype: `fp8_e4m3`
- Failing path: FlashInfer paged prefill, VO-split global layer

## Verbatim failing shape

From the live red row:

```text
SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=5
planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2,
head_dim=512, head_dim_vo=256)
```

FlashInfer module:

```text
dtype_kv=__nv_fp8_e4m3
head_dim_qk=512
head_dim_vo=256
page_size=1
split_kv=1
cta_tile_q=64
```

Crash:

```text
FlashInfer Internal Error: Invalid configuration :
NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=16 NUM_MMA_KV=1
NUM_WARPS_Q=4 NUM_WARPS_KV=1
```

## Static dispatch finding

`include/flashinfer/attention/prefill.cuh` already includes a `kMinValidMmaKV` guard in both
single/ragged and paged prefill. For fp8 (`sizeof(DTypeKV) == 1`) with `NUM_WARPS_Q=4`, the
minimum valid tile is:

```text
kMinValidMmaKV = NUM_WARPS_Q / 2 = 2
```

The guard is only used when estimating whether two CTAs can fit on an SM:

```text
num_ctas_per_sm = max_smem_per_sm >= 2 * (q_tile + kMinValidMmaKV * kv_step) ? 2 : 1
```

But the actual dispatch still passes the raw computed maximum:

```text
DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, ...)
```

For the failing shape:

```text
q_tile_smem        = CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ)
                   = 64 * 512 * 2 = 65536 bytes

kUseRepack         = true
kv_step_smem       = (512 + 256) * 16 * 1 * 1
                     + max(512,256) * 16 * 1 * 2
                   = 28672 bytes

one fp8 KV step    = 65536 + 28672 = 94208 bytes
two fp8 KV steps   = 65536 + 2 * 28672 = 122880 bytes
```

On GB10-class shared-memory limits this explains the observed behavior: one KV step fits and is
selected, but one step violates the FP8 alignment rule:

```text
sizeof(DTypeKV) == 1 && NUM_MMA_KV * 2 % NUM_WARPS_Q != 0
```

So the previous dispatcher fix avoids the older `max_mma_kv=0` failure but still lets the
D512/VO256 fp8 paged-prefill path land on invalid `NUM_MMA_KV=1`.

## Current conclusion

This is not an SGLang scale/feed issue. The E4B fp8 comparator red is a shared FlashInfer
dispatcher coverage gap for the D512/VO256 fp8 VO-split paged-prefill shape on tight-smem
GB10. The safe SGLang claim should continue to scope the fp8 comparator red until the dispatcher
can either:

- choose a valid layout for this shape, or
- reject it explicitly before serving with a clear unsupported-shape reason.

No code change was applied in this stop point.
