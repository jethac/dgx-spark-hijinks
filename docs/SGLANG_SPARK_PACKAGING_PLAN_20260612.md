# SGLang Spark Packaging Plan

Date: 2026-06-12 JST. Branch: `epoch2`.

## Target Contract

The SGLang Spark artifact must match the vLLM Spark r11 packaging invariant:

- platform: `linux/arm64`
- OS ABI: Ubuntu 22.04 / `glibc <= 2.35`
- Python ABI: CPython 3.12
- torch ABI: `torch==2.11.0` with CUDA 13 wheels from `https://download.pytorch.org/whl/cu130`
- CUDA target: `TORCH_CUDA_ARCH_LIST=12.1a`
- build location: Ubicloud/GitHub only, not Spark
- deploy shape: self-contained image, not a loose wheel injected into a foreign container

The vLLM proof point is `jethac/vllm` release
`sm121a-arm64-wheels-3d6a0d507`, which was built on
`ubicloud-standard-30-arm-ubuntu-2204` with torch 2.11 and a GLIBC_2.35
extension audit.

## Rejected Base

`nvcr.io/nvidia/sglang:26.05-py3` is rejected for Spark deployment because it
is Ubuntu 24.04 / glibc 2.39. The probe is recorded in
`results/sglang_spark_base_probe_20260612TmanualJST/summary.md`.

NGC PyTorch 25.04 and 25.03 were also probed as Ubuntu 24.04 bases and are
therefore rejected for this artifact.

## Selected Path

Build a fresh SGLang source-stack image from
`nvidia/cuda:13.0.2-devel-ubuntu22.04`:

1. Install Python 3.12 in the image.
2. Install torch 2.11 / CUDA 13 from the PyTorch cu130 index.
3. Install the SGLang runtime dependency closure from the checked-out SGLang
   pyproject, excluding the packages that the campaign replaces from source
   (`flashinfer*`, `sglang-kernel`, and torch-family packages already pinned).
4. Build and install campaign FlashInfer and SGLang/sgl-kernel from
   `third_party/flashinfer` and `third_party/sglang`.
5. Bake Transformers 5.11.0 for `gemma4_unified`.
6. Gate the final image for OS, GLIBC, imports/provenance, manifest platform,
   and empty FlashInfer JIT/AOT cache payload.

This is heavier than the earlier NGC-derived source-stack image, but it avoids
the three mismatch classes that broke loose wheel injection: base lineage,
torch ABI, and glibc ABI.

## CI Entry Point

Workflow:
`.github/workflows/hijinks-sglang-gemma4-source-stack-image.yml`

Defaults:

- `base_image=nvidia/cuda:13.0.2-devel-ubuntu22.04`
- `image_tag=epoch2-sglang-spark-u22-torch211`
- `target_platform=linux/arm64`
- `torch_version=2.11.0`
- `torch_index_url=https://download.pytorch.org/whl/cu130`
- `transformers_pin=5.11.0`

Runner:
`ubicloud-standard-30-arm-ubuntu-2204`

The repo currently has an online persistent x64 runner only. Native arm64
builds therefore use the Ubicloud arm64 runner until an arm64 persistent
runner is registered.

## Acceptance Gates

The artifact is not Spark-ready until the workflow proves:

- pushed image manifest contains `linux/arm64`;
- image `/etc/os-release` has `VERSION_ID=22.04`;
- compiled SGLang, FlashInfer, and sgl-kernel extensions require no newer than
  `GLIBC_2.35`;
- torch imports as `2.11.0+cu130`;
- SGLang, FlashInfer, sgl-kernel, and Transformers import in-container;
- `gemma4_unified` is present in the Transformers config mapping;
- no FlashInfer JIT/AOT module cache payload is baked into the image.

Only after those pass should Spark run a minimal no-build smoke under the usual
memory guardrails.
