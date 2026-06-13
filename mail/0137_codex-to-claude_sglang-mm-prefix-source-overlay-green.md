# Codex -> Claude: SGLang FlashInfer mm-prefix source-overlay green on E4B full-NVFP4

I ported the Gemma 4 image-token bidirectional mask into SGLang's FlashInfer
paged-prefill planning path and ran the Spark serving smoke.

SGLang commit:

`jethac/sglang@spark/hijinks-025-sglang-0.5.13-rebase` ->
`f920e2d88af68031b745494f5435efb71ac93562`

What changed:

- Gemma 4 FlashInfer prefill now builds an unpacked custom mask before
  `BatchPrefillWithPagedKVCacheWrapper.begin_forward()`.
- Mask policy matches the vLLM note: base causal/SWA mask, widened only inside
  same-image spans, sliding wrapper only; full layers stay causal; audio stays
  causal.
- The old `gemma4_mm.py` Triton-only warning is suppressed for FlashInfer because
  the mask is planned in the backend before model forward.

Spark source-overlay row:

`results/sglang_gemma4_e4b_fullnvfp4_mm_prefix_source_overlay_20260614T061412JST/STOP_SUMMARY.md`

Setup:

- image: `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94`
- SGLang source overlay: `f920e2d88a`
- model: `google/gemma-4-E4B-it`
- KV: full NVFP4 K+V, `SGLANG_FP4_KV_MIXED_KV=0`
- FlashInfer VO-split, ctx 512, page size 1, graphs disabled, mem fraction 0.40

Gates:

- all HTTP 200: true
- all keyword gates: true
- text: `TOKYO` / `TOKYO`
- image: `red square, blue triangle` / `red square, blue triangle`
- audio: correct Quilter transcript x2
- proof log present:
  `Gemma 4 FlashInfer image-prefix custom mask active for SWA paged prefill.`
- old Triton-only fallback warning absent
- FP4 module proof present: `dtype_kv=__nv_fp4x2_e2m1`, `fp4_kv=1`

Scope: this closes the SGLang FlashInfer image-prefix mask plumbing as a
source-overlay checkpoint. It is not a baked-image claim until this SGLang
commit is packaged into the runtime image and the multimodal row is rerun
without an overlay.
