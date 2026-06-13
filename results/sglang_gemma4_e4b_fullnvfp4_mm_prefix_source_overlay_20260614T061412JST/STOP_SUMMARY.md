# SGLang Gemma 4 E4B Full-NVFP4 MM-Prefix Source-Overlay Smoke

Status: GREEN, scoped source-overlay evidence.

- Runtime: DGX Spark / GB10, packaged SGLang image with source overlay
- Image: `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94`
- Model: `google/gemma-4-E4B-it`
- KV: `fp4_e2m1`, full NVFP4 K+V (`SGLANG_FP4_KV_MIXED_KV=0`)
- SGLang source overlay: `jethac/sglang@f920e2d88a`
- Backend: FlashInfer, `SGLANG_FLASHINFER_VOSPLIT=1`, CUDA graphs disabled, page size 1

## Gates

- all HTTP 200: `True`
- all keyword gates: `True`
- mm-prefix proof log present: `True`
- old Triton-only fallback warning present: `False`
- FP4 prefill module proof present: `True`

## Outputs

- text: `TOKYO` / `TOKYO`
- image: `red square, blue triangle` / `red square, blue triangle`
- audio: `Mr Quilter is the apostle of the middle classes and we are glad to welcome his gospel.` / `Mr Quilter is the apostle of the middle classes and we are glad to welcome his gospel.`

## Scope

This validates the SGLang FlashInfer image-token bidirectional mask plumbing on the E4B multimodal request path. It is not a baked-image claim until this SGLang commit is packaged into the runtime image.
