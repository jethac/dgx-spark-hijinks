# vLLM NVFP4 writer SF-byte probe on real Gemma 4 K/V

Date: 2026-06-15 JST

## Scope

Discriminator for Claude mail 0179: verify whether vLLM's real NVFP4 KV writer stores the expected per-block FP8 E4M3 scale-factor bytes for real HF-eager Gemma 4 K/V tensors.

This checks the CUDA writer:

`torch.ops._C_cache_ops.reshape_and_cache_flash(..., "nvfp4", k_scale, v_scale)`

It does not run FlashInfer attention, so it does not clear read/dequant, wrapper feed, or serving-level attention accumulation.

## Environment

- Vast instance: `41018291`, destroyed after artifact pull
- GPU: RTX PRO 6000 Blackwell Workstation Edition, sm_120, 96 GB
- OS/container: `nvidia/cuda:13.0.1-devel-ubuntu24.04`
- Torch: `2.12.0+cu130`
- vLLM wheel: `vllm 0.1.dev1+ge32459eea.sm120a`
- Env: `VLLM_NVFP4_KV_LINEAR_V_SF=1`
- Models: `google/gemma-4-12b-it`, `google/gemma-4-26b-a4b-it`
- Prompting: chat template, text-only
- Context: 512 tokens, sampled final 256 KV tokens
- Layers: `0,1,12,last`
- Scales: vLLM `_k/_v_scale` values `0.1` and `0.07`

## Method

For each model/layer/KV-side/scale:

1. Run HF eager with `use_cache=True`.
2. Normalize returned `past_key_values` to `[tokens, kv_heads, head_dim]`.
3. Write the tensor through vLLM's real NVFP4 CUDA writer into a synthetic paged cache.
4. Extract the physical page scale region `[data][scale]`.
5. Compare stored FP8 scale-factor bytes against:

`fp8_e4m3((1 / vllm_scale) * (amax(block16) / 6))`

The K and V sides were tested independently as `x -> K` and `x -> V` to avoid conflating this writer-byte test with VO-split K/V orchestration.

## Result

All checked writer scale-factor bytes match the mathematical expectation exactly.

| model | side | vLLM scale | rows | mismatches | checked SF bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma 4 12B IT | K | 0.10 | 4 | 0 | 106,496 |
| Gemma 4 12B IT | K | 0.07 | 4 | 0 | 106,496 |
| Gemma 4 12B IT | V | 0.10 | 4 | 0 | 106,496 |
| Gemma 4 12B IT | V | 0.07 | 4 | 0 | 106,496 |
| Gemma 4 26B-A4B IT | K | 0.10 | 4 | 0 | 114,688 |
| Gemma 4 26B-A4B IT | K | 0.07 | 4 | 0 | 114,688 |
| Gemma 4 26B-A4B IT | V | 0.10 | 4 | 0 | 114,688 |
| Gemma 4 26B-A4B IT | V | 0.07 | 4 | 0 | 114,688 |

Total: 32 rows, 884,736 FP8 scale-factor bytes, 0 mismatches.

## Interpretation

This falsifies the suspected vLLM writer-side global-scale delivery / per-block SF-byte path for the sampled real 26B-A4B K/V distributions. The writer consumes the supplied global scale and stores the same FP8 scale bytes the NVFP4 recipe predicts, including at `_k/_v_scale=0.07`.

The remaining 26B-A4B NVFP4 failure is more likely in:

- FlashInfer read/dequant / attention feed for the cached pages,
- wrapper argument construction around those cache views,
- or the first serving-layer divergence in hidden states/attention inputs under the MoE path.

Artifacts:

- `results.json` - full per-row JSON output
- `run.log` - console log
- `vast_instance_41018291_before_destroy.json` - pre-destroy instance metadata
- `../../docs/vast_anchor/vllm_nvfp4_writer_sf_probe.py` - probe source
