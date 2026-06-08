# vLLM / FlashInfer Gemma 3 Attention-Output Probe, 2026-06-09

Status: standalone FlashInfer FA2 NVFP4 attention-output probe passed; Gemma 3 vLLM
NVFP4-KV remains red.

## Purpose

The vLLM Gemma 3 tensor trace
(`results/vllm_gemma3_27b_tensor_trace_20260609T0115JST_summary.md`) localized the
strongest corruption to `flashinfer_attn_output`: NVFP4-KV outputs were BF16-shaped
but nearly nonnegative, with means around `124..126` and max values exactly `255.0`
on many layers. This probe tests whether standalone FlashInfer FA2 NVFP4 attention
reproduces that output boundary failure for Gemma-shaped `D=128` data.

## Source And Environment

- Script: `scripts/flashinfer_nvfp4_kv_probe.py`
- FlashInfer source: `third_party/flashinfer@e41016fcd121986aea923d5c7e68fc9f152d2a07`
- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- Hardware: `NVIDIA GB10`, compute capability `12.1`, `48` SMs, CUDA `13.0`
- Geometry: `head_dim=128`, `num_kv_heads=16`, `num_qo_heads=32`, `page_size=16`,
  `kv_len=64`, `qo_len=16`

The first probe attempt used the legacy synthetic generator, which masked packed
nibbles with `0x77`. That creates a nonnegative reference and is not valid evidence
for the Gemma failure signature. The accepted rows below use `--signed-values`.

## Rows

| artifact | V-scale layout | extra flag | k/v global scale | all ok | min cosine | max abs | byte-like actual outputs |
|---|---|---|---:|---:|---:|---:|---:|
| `results/vllm_flashinfer_gemma3_attention_output_probe_20260609T0134JST_signed_swizzled_nonunit_vscale.json` | swizzled | `-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1` | `0.03125` | true | `0.999997496604919` | `0.0001220703125` | `0 / 4` |
| `results/vllm_flashinfer_gemma3_attention_output_probe_20260609T0134JST_signed_linear_control.json` | linear | none | `0.03125` | true | `0.999997496604919` | `0.0001220703125` | `0 / 4` |

Representative signed output stats:

| row | op/layout | actual min | actual max | actual mean | cosine |
|---|---|---:|---:|---:|---:|
| swizzled | NHD decode | `-0.0279541015625` | `0.0255126953125` | `3.15e-06` | `0.999997615814209` |
| swizzled | NHD prefill | `-0.0274658203125` | `0.0255126953125` | `9.73e-05` | `0.999999940395355` |
| swizzled | HND decode | `-0.022216796875` | `0.0228271484375` | `6.84e-05` | `0.999997496604919` |
| swizzled | HND prefill | `-0.0220947265625` | `0.023193359375` | `2.41e-05` | `1.00000011920929` |

The linear-control row produced the same pass/fail profile and the same signed ranges
under the fixed seed.

## Interpretation

Standalone FlashInfer FA2 NVFP4 attention does not reproduce the Gemma 3 byte-like
BF16 output failure for signed Gemma-shaped `D=128` synthetic data. This clears:

- generic signed E2M1 value handling in the standalone FA2 path;
- non-unit K/V global-scale handling for this synthetic case;
- the vLLM-style V-scale deswizzle macro as a standalone reproducer;
- both decode and prefill wrapper calls for NHD and HND layouts.

This does not fix or bless vLLM Gemma 3 NVFP4-KV. It narrows the next target to the
real vLLM/Gemma wrapper boundary. The next probe should trace or dump, for the first
failing request and layer, the actual model `query`, split packed K/V tensors, K/V
scale tensors, scalar `k_scale` and `v_scale`, output buffer dtype/shape/stride before
and after `wrapper.run(...)`, and whether prefill or decode first produces byte-like
output.
