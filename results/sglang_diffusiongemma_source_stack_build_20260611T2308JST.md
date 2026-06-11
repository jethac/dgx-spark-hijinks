# SGLang DiffusionGemma Source-Stack Build

Date: 2026-06-11 23:08 JST

## Outcome

Green. The SGLang DiffusionGemma source-stack image built successfully on the
GB10 host.

Prepared image:

- `sglang-source-stack-dgemma-024-0705924c-f99323bd`

Source checkout on Spark:

- `/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-dgr1-20260611T2220JST`

Commits:

- main repo: `2b8794ec384317d091904840b946e162f937e18b`
- SGLang: `0705924c1dc1c25c295fc5ce1c918841bf1f93a7`
- FlashInfer: `f99323bd7d1cc88d9445202c12934070be754e2d`

## Verification

Copied summary:

- `results/sglang_dgemma_024_source_stack_20260611T2225JST_summary.json`

Key values:

- base image: `nvcr.io/nvidia/sglang:26.05-py3`
- device: `NVIDIA GB10`
- compute capability: `[12, 1]`
- Torch: `2.12.0a0+5aff3928d8.nv26.05`
- CUDA: `13.2`
- FlashInfer: `0.6.13`
- SGLang: `0.0.0.dev1+g0705924c1`
- sgl-kernel: `0.4.3`
- loaded common ops: `/usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.abi3.so`

Build logs showed `sm_121a` compilation in the SGLang kernel build.

## Scope

This is an image/source-stack gate only. It does not claim DiffusionGemma model
load, forward pass, coherence, performance, FlashInfer routing, or NVFP4
correctness.

No live model-serving test was started after the build. Docker was clean and the
Claude marker was absent at the stop point.
