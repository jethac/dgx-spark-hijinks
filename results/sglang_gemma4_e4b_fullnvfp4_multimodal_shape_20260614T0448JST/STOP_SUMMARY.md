# SGLang Gemma 4 E4B Full-NVFP4 Multimodal Shape Smoke

Status: GREEN, scoped functional serving evidence.

- Runtime: DGX Spark / GB10, packaged SGLang image `sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e`
- Model: `google/gemma-4-E4B-it`
- KV: `fp4_e2m1`, full NVFP4 K+V (`SGLANG_FP4_KV_MIXED_KV=0`)
- Backend: FlashInfer, `SGLANG_FLASHINFER_VOSPLIT=1`, CUDA graphs disabled
- Scope: multimodal request-path and FP4-KV routing smoke; no broad image-quality claim

## Probe

Prompt asset: generated deterministic `red_square_blue_triangle.png`.

All rows returned HTTP 200 and hit the required keywords under `--keyword-mode all`:

- text: `TOKYO` / `TOKYO`
- image: `red square, blue triangle` / `red square and blue triangle`
- audio: `Mr Quilter is the apostle of the middle classes and we are glad to welcome his gospel.` twice

The image row is semantically stable and keyword-green, but not byte-identical because one repeat used a comma and the other used `and`. Text and audio were byte-identical.

Usage matched the bf16 baseline:

- text prompt tokens: 18
- image prompt tokens: 287
- audio prompt tokens: 174

## Capacity/Route Proof

At the same `--mem-fraction-static 0.40`:

- bf16 baseline: `full_layer_tokens=358759`, `swa_layer_tokens=287007`
- full-NVFP4: `full_layer_tokens=1273224`, `swa_layer_tokens=1018579`
- allocator-token ratio: `3.549x` full pool, `3.549x` SWA pool

The server log proves:

- Gemma 4 multimodal model load: `Gemma4ForConditionalGeneration`
- D=512 VO-split routing via `extend_paged_vosplit0/1` and `decode_as_prefill_vosplit0/1`
- FP4 prefill module selection: `dtype_kv=__nv_fp4x2_e2m1`, `fp4_kv=1`
- FP4 scale views present in SGLang wrapper state: `k_sf={...}` and `v_sf={...}`

## Caveat

The server logs the same image-attention caveat as bf16:

`Bidirectional attention for image tokens requires TritonAttnBackend. Falling back to causal attention, which may degrade image quality.`

Therefore this row closes the SGLang packaged-image E4B multimodal NVFP4 serving-path smoke for text + image + audio, but broad multimodal quality claims remain scoped until the bidirectional image-attention path and the known 12B text-quality issue are resolved or explicitly excluded.
