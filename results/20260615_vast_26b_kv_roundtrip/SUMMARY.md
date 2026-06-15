# Gemma 4 26B-A4B NVFP4 KV round-trip discriminator

Date: 2026-06-15 JST

Scope: HF eager `past_key_values` capture on a disposable vast.ai RTX PRO 6000 WS (`sm_120`) instance. This is a kernel-free quantize/dequantize discriminator, not a serving row.

Artifacts:
- `results.json`: full per-layer metrics.
- `run.log`: remote run log.
- `gemma4_kv_roundtrip_discriminator.py`: exact script used.

Environment:
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, capability `(12, 0)`
- Torch: `2.8.0+cu128`
- Transformers: `5.13.0.dev0`
- Models: `google/gemma-4-12b-it`, `google/gemma-4-26b-a4b-it`
- Prompt: script built-in repeated technical text, `ctx=2048`
- Sampling: tail `512` cache tokens from layers `0,1,2,12,24,last`
- Cache source: HF eager returned `past_key_values`

Important runtime note: Transformers' grouped-MoE path attempted `torch._grouped_mm`, which this Torch build rejects on `sm_120` (`torch._grouped_mm is only supported on CUDA devices with compute capability = 9.0`). The script forces Transformers' built-in grouped-MM fallback so 26B-A4B can run. This does not affect the NVFP4 round-trip math.

## Result

| model | tensor | n | mean best rel-L2 | max best rel-L2 |
| --- | --- | ---: | ---: | ---: |
| Gemma 4 12B | K | 6 | 0.093636 | 0.095294 |
| Gemma 4 12B | V | 6 | 0.092361 | 0.093938 |
| Gemma 4 26B-A4B | K | 6 | 0.093285 | 0.095096 |
| Gemma 4 26B-A4B | V | 6 | 0.092020 | 0.093281 |

Per-row best cosines were all about `0.9955-0.9961`. Fixed vLLM scale `0.07` was also comparable across 12B and 26B-A4B.

## Interpretation

26B-A4B's real HF cache tensors are not less representable in NVFP4 than 12B's on this sample. This falsifies the simple "26B K/V cannot survive 4-bit round-trip" hypothesis for the sampled layers/tokens.

The remaining 26B-A4B full-NVFP4 serving failure should be treated as a kernel/serving/feed bug until proven otherwise. The next localization should compare served NVFP4 against this kernel-free reference, with special attention to mixed per-layer V scale-factor layout, global scale delivery, and per-block fp8 scale write/read behavior.
