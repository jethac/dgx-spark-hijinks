# Draft upstream issue: FlashInfer/vLLM selectors accept head_dim=512 but FA2 paged kernels reject it at runtime

Target repos: `flashinfer-ai/flashinfer`, with downstream selector impact in `vllm-project/vllm`

## Title

FA2 paged attention accepts/routes `head_dim=512`, but the generated kernel rejects the
shape at runtime via `KernelTraits::IsInvalid()`

## Summary

Gemma 4 global/full attention layers use `D=512`. The current FlashInfer/vLLM selection
surface can route these layers toward FlashInfer FA2 NVFP4-KV support, but the actual
generated FA2 paged kernel rejects the shape at runtime. This creates an over-promise:
the selector says the backend is supported, but the kernel trait guard says the launch is
invalid.

The failure is not a missing `case 512` dispatch table entry. The path reaches the
`D=512` case and then fails inside `KernelTraits::IsInvalid()` because the VO accumulator
fragment budget is exceeded when `HEAD_DIM_VO=512`.

## Evidence

Campaign docs:

- `docs/FLASHINFER_D512_FA2_KERNEL_PLAN.md`
- `docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md`
- `docs/BASELINE_RESULTS.md`
- `docs/CAMPAIGN_LOG.md`

Key failure shape from the campaign:

```text
Gemma global/full attention
layout: NHD
batch_size=2
kv_len=128
qo_len=16
page_size=16
num_qo_heads=16
num_kv_heads=2
head_dim=512
dtype=bfloat16 / NVFP4-KV probe variants
```

Observed failure:

```text
FlashInfer FA2 invalid configuration from include/flashinfer/attention/prefill.cuh:3215
```

Trait audit from the campaign:

```text
DISPATCH_HEAD_DIM already includes case 512.
The rejection is KernelTraits::IsInvalid():

NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256

For D=512:
NUM_MMA_D_VO = 32
decode trait computes 264 >= 256
prefill trait computes 272 >= 256
```

That points at a fragment/register-shape guard, not a missing head-dim selector table
and not primarily a GB10 shared-memory-capacity issue.

## Expected Result

Either:

1. FlashInfer/vLLM selectors should reject the `HEAD_DIM_QK=512, HEAD_DIM_VO=512`
   FA2 paged configuration before planning/routing, with a clear fallback path; or
2. FlashInfer should route the shape to an actually supported implementation, such as a
   VO-split strategy using existing asymmetric `(HEAD_DIM_QK=512, HEAD_DIM_VO=256)`
   templating.

The important requirement is that backend selection should not claim a shape is usable
when the generated kernel will fail at runtime.

## Candidate Fix Direction

The campaign found a kernel-math-free route:

```text
Run attention twice with full Q/K at HEAD_DIM_QK=512,
but split V/O into two HEAD_DIM_VO=256 passes,
then concatenate the two outputs.
```

The math is exact along the VO dimension:

```text
S = Q * K^T
P = softmax(S)
O = [P * V_left | P * V_right]
```

The `(512,256)` trait passes the register guard because the VO accumulator fragment is
half-width, and the campaign's bf16 probe is green:

```text
results/flashinfer_fa2_vo_split_d512_vo256_probe_20260610T0520JST.json
cosine 0.9999978 versus torch fp32 reference
```

Known follow-up from the campaign:

- NVFP4 packed-data and scale-factor slicing for the VO split still needs the follow-up
  P0b/P2 plumbing.
- vLLM swizzled V scale-factor layout needs care; SGLang's linear V scale factors slice
  more directly.

## Why This Matters

Gemma 4 12B, 31B, and 26B-A4B all have global/full attention layers with `D=512`.
Without an accurate selector or a working VO-split implementation, `--kv-cache-dtype
nvfp4` / mixed-KV routes can appear selectable and then fail below the runtime routing
layer.

This blocks clean Gemma 4 NVFP4-KV support on GB10 and other SM12x systems unless the
global `D=512` layers are routed to fp8/bf16 fallback or the VO-split path is implemented.

## Suggested Upstream Scope

- In FlashInfer: expose head-dim validation that accounts for `KernelTraits::IsInvalid()`,
  or add a supported asymmetric VO-split route for `QK=512, VO=256`.
- In vLLM: do not select FlashInfer FA2 NVFP4-KV for `D=512` layers unless the selected
  FlashInfer build proves the specific `(HEAD_DIM_QK, HEAD_DIM_VO)` pair is supported.
- In both: distinguish "selector accepts `head_dim=512`" from "generated kernel can
  compile and launch this exact trait."

## Attached Campaign Artifacts

- `docs/FLASHINFER_D512_FA2_KERNEL_PLAN.md`
- `results/flashinfer_fa2_bf16_d512_probe_20260610T0438JST.json`
- `results/flashinfer_fa2_vo_split_d512_vo256_probe_20260610T0520JST.json`
- `docs/GEMMA_RUNG_MINUS1_CONFIG_AUDIT.md`
