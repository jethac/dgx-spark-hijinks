# SGLang Spark Ubuntu 22 Image Stop Point

Date: 2026-06-13 JST

## Verdict

GREEN for the SGLang Spark packaging goal.

The final image is Spark-deployable under the epoch2 packaging contract:

- image tag:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack:epoch2-sglang-spark-u22-torch211-arm64`
- immutable digest:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94`
- platform: `linux/arm64`
- base: `nvidia/cuda:13.0.2-devel-ubuntu22.04`
- runtime OS: Ubuntu `22.04.5`
- torch: `2.11.0+cu130`
- Transformers: `5.11.0`
- SGLang: `98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968`
- FlashInfer: `f99323bd7d1cc88d9445202c12934070be754e2d`

## Build Evidence

Final green workflow:

- GitHub Actions run: `27428220601`
- artifact: `results/sglang_gemma4_source_stack_image_27428220601/summary.md`

Gates recorded by the workflow:

- `BASE_IMAGE_OS_VERSION=22.04`
- `IMAGE_OS=Ubuntu 22.04.5 LTS`
- `TORCH 2.11.0+cu130 13.0`
- `TORCH_FINAL 2.11.0+cu130 13.0`
- `TRANSFORMERS_BAKED 5.11.0`
- `GEMMA4_UNIFIED_CONFIG_MAPPING present`
- `FLASHINFER 0.6.13 /work/third_party/flashinfer/flashinfer/__init__.py`
- `SGLANG 0.0.0 /work/third_party/sglang/python/sglang/__init__.py`
- `SGLANG_KERNEL 0.4.3`
- `GLIBC_AUDIT_MAX=GLIBC_2.34`
- FlashInfer cache payload removed before image finalization
- manifest platform contains `linux/arm64`

Rejected intermediate:

- run `27424620048`
- artifact: `results/sglang_gemma4_source_stack_image_27424620048/summary.md`
- status: build/provenance only; superseded because a Spark smoke exposed torch
  drift from an unpinned torchvision dependency. The final workflow pins
  torch-family packages and asserts torch `2.11.0`.

## Spark Smoke Evidence

Final green Spark smoke:

- artifact: `results/sglang_spark_image_smoke_20260613T022153JST/summary.md`
- model: `google/gemma-4-E2B-it`
- attention backend: `triton`
- server ready: `1`
- request rc: `0`
- response: `The capital of Japan is Tokyo.`

Earlier policy red:

- artifact: `results/sglang_spark_image_smoke_20260613T021405JST/summary.md`
- status: RED before readiness because the smoke forced
  `--attention-backend flashinfer`, and current SGLang Gemma4 policy rejects
  FlashInfer for that stock E2B launch shape:
  Gemma4 only supports `trtllm_mha`, `triton`, or `intel_xpu` there.
- This is not an image import, OS, torch, manifest, or cache-hygiene failure.

## Scope

This is a packaging/runtime-smoke claim only. It does not claim:

- Gemma4 AR ladder quality or capacity
- DiffusionGemma quality or capacity
- NVFP4 correctness
- FlashInfer serving correctness for Gemma4
- throughput

The live runner defaults now point at the immutable digest for future Gemma4
AR, MTP, overnight, and DiffusionGemma packets.
