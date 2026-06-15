# 26B-A4B Short Position Attribution Low-RAM Failure

Status: **INFRA RED / no quality result**.

This was the first attempt to run the short position-attribution discriminator on Vast instance `41093069`
(`codex-26b-posattr-short`). The instance was destroyed after artifact pull.

## Stack

- Image: `nvidia/cuda:13.0.1-devel-ubuntu22.04`
- Device: RTX PRO 6000 Blackwell WS, capability `(12, 0)`
- vLLM wheel: `0.1.dev1+g1c9686c61.sm120a`
- FlashInfer source overlay: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Model: `google/gemma-4-26B-A4B-it`
- Packet: short position attribution (`bf16`, `k100`, `k103`)

## Failure

The run failed during the first `bf16` engine profile before any JSON row was produced. The log reached:

- model weights loaded;
- encoder-cache profiling started;
- FlashInfer CUTLASS MoE JIT started;
- ninja failed with exit status `137`.

The host reported only `83.72 GiB` available RAM while compiling a large FlashInfer `fused_moe_120` module.
This is therefore an infrastructure/build-resource failure, not a model-quality result and not an NVFP4
attention result.

## Follow-up

The successful rerun used a higher-RAM Vast host and set `MAX_JOBS=4`, which FlashInfer's `run_ninja()`
honors by appending `-j 4` to ninja.

Artifact contents:

- `dx26_position_attr_short_20260615T1938Z/RUN_INFO.txt`
- `dx26_position_attr_short_20260615T1938Z/rows/bf16.log`
- copied packet scripts
