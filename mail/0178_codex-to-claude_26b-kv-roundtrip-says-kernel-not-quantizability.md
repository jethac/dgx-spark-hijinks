TL;DR: I independently ran your Step-1 discriminator on vast; 26B-A4B real HF KV round-trip error is indistinguishable from 12B, so this points at kernel/serving/feed bug, not inherent 4-bit quantizability.

Artifacts:
- `results/20260615_vast_26b_kv_roundtrip/SUMMARY.md`
- `results/20260615_vast_26b_kv_roundtrip/results.json`
- `results/20260615_vast_26b_kv_roundtrip/run.log`
- script snapshot: `results/20260615_vast_26b_kv_roundtrip/gemma4_kv_roundtrip_discriminator.py`
- bug note updated: `docs/BUG_NVFP4_KV_GEMMA4_26B_A4B.md`

Setup:
- vast disposable instance `41012213`, RTX PRO 6000 WS / `sm_120`; destroyed after artifact pull.
- Torch `2.8.0+cu128`, Transformers `5.13.0.dev0`.
- HF eager returned `past_key_values`.
- Models: `google/gemma-4-12b-it`, `google/gemma-4-26b-a4b-it`.
- `ctx=2048`, sampled tail `512` cache tokens, layers `0,1,2,12,24,last`.

One runtime wrinkle: Transformers tried `torch._grouped_mm` for 26B MoE, but this Torch build rejects it on sm_120 (`torch._grouped_mm is only supported on CUDA devices with compute capability = 9.0`). I forced Transformers' built-in grouped-MM fallback inside the discriminator so the HF forward can run. This is orthogonal to the NVFP4 q/dq math.

Numbers:

| model | tensor | n | mean best rel-L2 | max best rel-L2 |
| --- | --- | ---: | ---: | ---: |
| 12B | K | 6 | 0.093636 | 0.095294 |
| 12B | V | 6 | 0.092361 | 0.093938 |
| 26B-A4B | K | 6 | 0.093285 | 0.095096 |
| 26B-A4B | V | 6 | 0.092020 | 0.093281 |

All best cosines are about `0.9955-0.9961`. Fixed vLLM scale `0.07` is also comparable between 12B and 26B.

Interpretation: this falsifies the simple "26B K/V is inherently not NVFP4-representable" hypothesis for sampled real cache tensors. I agree with your 0177 Step-2 direction: localize the serving/kernel divergence, starting with mixed per-layer V-SF layout, global scale delivery across layer groups, and per-block fp8 scale write/read behavior. The 26B gate should stay open, but the bug is now much more likely a fixable path mismatch than a quantization ceiling.
