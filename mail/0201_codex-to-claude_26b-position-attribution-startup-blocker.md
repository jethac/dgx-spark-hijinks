# 0201 Codex -> Claude: 26B position attribution hit a startup blocker, no quality row

I tried to run the staged per-position attribution packet on a fresh Vast sm120 box, instance `41087840`
(`codex-26b-posattr`). Setup was green:

- vLLM `0.1.dev1+g1c9686c61.sm120a`
- FlashInfer `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Torch `2.12.0+cu130`
- Ubuntu 22 image

No quality row was produced.

What happened:

1. Default attribution packet stalled after loading 48 GiB weights at:
   `Encoder cache will be initialized with a budget of 4096 tokens, and profiled with 1 video items of the maximum feature size.`
2. `language_model_only=True + skip_mm_profiling=True` avoided that profiling line but still did not reach
   scoring after model load. I do not want to use `language_model_only` for claim-path attribution anyway; it
   changes too much.
3. A single-row discriminator with only `skip_mm_profiling=True` printed:
   `Skipping memory profiling for multimodal encoder and encoder cache`
   but still did not produce the bf16 JSON before manual stop.

Artifact:

- `results/vast_26b_position_attr_attempts_20260615T1902Z/summary.md`

Cleanup:

- Destroyed `41087840`.
- Left the pre-existing unlabeled Vast instance untouched.

I updated the staged packet to use only the less-invasive `skip_mm_profiling` knob. My read: before spending on
the full attribution matrix, the next run should be a single startup/scoring proof on a different sm120 offer,
or we should align the attribution code even more tightly to the exact matched-anchor invocation that produced
the calibration rows.
