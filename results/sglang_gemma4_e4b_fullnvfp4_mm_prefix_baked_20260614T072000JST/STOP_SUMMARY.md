# SGLang Gemma 4 E4B Full-NVFP4 MM-Prefix Baked-Image Smoke

Status: GREEN, scoped baked-image evidence.

- Runtime: DGX Spark / GB10, packaged SGLang image, no source overlay
- Image: `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0bacd437f9917928a9bd7ba0dafbb37516f8e05b4b9727bbff796556c2cc7714`
- Image tag: `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack:epoch2-sglang-mm-prefix-f920e2d-arm64`
- Model: `google/gemma-4-E4B-it`
- KV: `fp4_e2m1`, full NVFP4 K+V (`SGLANG_FP4_KV_MIXED_KV=0`)
- SGLang packaged ref: `f920e2d88af68031b745494f5435efb71ac93562`
- FlashInfer packaged ref: `f99323bd7d1cc88d9445202c12934070be754e2d`
- Backend: FlashInfer, `SGLANG_FLASHINFER_VOSPLIT=1`, CUDA graphs disabled, page size 1

## Gates

- all HTTP 200: `True`
- all keyword gates: `True`
- mm-prefix proof log present: `True`
- old Triton-only/bidirectional fallback warning present: `False`
- FP4 prefill module proof present: `True`

## Outputs

- text: `TOKYO` / `TOKYO`
- image: `red square, blue triangle` / `red square, blue triangle`
- audio: `Mr Quilter is the apostle of the middle classes and we are glad to welcome his gospel.` / `Mr Quilter is the apostle of the middle classes and we are glad to welcome his gospel.`

## Scope

This validates the SGLang FlashInfer image-token bidirectional mask plumbing on the E4B multimodal request path from a baked Ubuntu22/arm64/torch2.11 package image. It is a scoped E4B multimodal functional row, not a broad SGLang Gemma 4 AR quality/capacity claim and not a resolution of the 12B text-quality red or E4B fp8 comparator red.
