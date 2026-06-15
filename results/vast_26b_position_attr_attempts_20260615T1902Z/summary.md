# vLLM Gemma 4 26B-A4B position-attribution attempt

Status: **INCONCLUSIVE / no quality row produced**.

This artifact records a failed attempt to run the staged per-position logprob attribution packet for
`google/gemma-4-26B-A4B-it` on a fresh Vast sm120 instance. The instance was destroyed after artifact pull.

## Environment

- Vast instance: `41087840` (`codex-26b-posattr`)
- Device: RTX PRO 6000 Blackwell, sm120
- Image: `nvidia/cuda:13.0.1-devel-ubuntu22.04`
- vLLM: `0.1.dev1+g1c9686c61.sm120a`
- FlashInfer source overlay: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Torch: `2.12.0+cu130`
- Setup log: `dx26_position_attr_attempts_20260615T1902Z/setup.log`

## Attempts

1. **Original attribution packet** (`encoder_profile_hang/`)
   - Args: default multimodal Gemma 4 initialization, ctx `8185`, prefix `4096`, bf16 baseline first.
   - Result: after the initial 48 GiB model download/load, vLLM stopped at:
     `Encoder cache will be initialized with a budget of 4096 tokens, and profiled with 1 video items of the maximum feature size.`
   - No JSON row produced.

2. **Text-only attribution packet** (`textonly_stall/`)
   - Args: `language_model_only=True` plus `skip_mm_profiling=True`.
   - Result: vLLM confirmed text-only mode and loaded the model, but did not reach the usual KV/cache-profile or scoring output before manual stop.
   - No JSON row produced.

3. **Single-row skip-profiling discriminator**
   - Args: `skip_mm_profiling=True` only, preserving the normal Gemma 4 multimodal config while skipping encoder/cache profiling.
   - Result: vLLM printed `Skipping memory profiling for multimodal encoder and encoder cache`, then stalled before producing `/root/posattr_skipmm_bf16.json`.
   - No JSON row produced.

## Interpretation

This run did not advance the 26B-A4B NVFP4 quality diagnosis. Follow-up log review showed the attempt was
stopped too early to prove a true startup/init blocker: prior successful bf16 rows often remained at the same
`Encoder cache will be initialized...` line for 5-7 minutes before logging KV-cache size and continuing.

`skip_mm_profiling=True` remains a useful discriminator, but it should not be the default claim-path mode
because the existing calibration rows ran through the normal Gemma 4 initialization path. `language_model_only`
is more invasive and changes the model configuration surface, so it should not be used for claim-path
attribution unless separately justified.

Next useful step: rerun the attribution packet in the normal matched-anchor launch mode with enough patience:
allow at least 10 minutes after model load for the first bf16 row before calling it stuck.

## Cleanup

The `codex-26b-posattr` instance was destroyed after pulling this artifact. The pre-existing unlabeled Vast
instance was left untouched.
