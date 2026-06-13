# SGLang Gemma 4 12B Full-NVFP4 Multimodal Shape Smoke

Status: GREEN, scoped functional serving evidence.

- Runtime: DGX Spark / GB10, packaged SGLang image `sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e`
- Model: `google/gemma-4-12B-it`
- KV: `fp4_e2m1`, full NVFP4 K+V (`SGLANG_FP4_KV_MIXED_KV=0`)
- Backend: FlashInfer, `SGLANG_FLASHINFER_VOSPLIT=1`, CUDA graphs disabled
- Scope: multimodal request-path and FP4-KV routing smoke; no 12B quality-parity claim

## Probe

Prompt asset: generated deterministic `red_square_blue_triangle.png`; audio asset is the banked LibriVox/Quilter clip.

All rows returned HTTP 200 and hit the required keywords under `--keyword-mode all`:

- text: `Tokyo` / `Tokyo`
- image: `red square and blue triangle` / `red square, blue triangle`
- audio: `Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.` twice

The image row is semantically stable and keyword-green, but not byte-identical because one repeat used a comma. Text and audio were byte-identical.

Usage matched the bf16 baseline:

- text prompt tokens: 22
- image prompt tokens: 291
- audio prompt tokens: 178

## Capacity/Route Proof

At the same `--mem-fraction-static 0.40`:

- bf16 baseline: `full_layer_tokens=79,557`, `swa_layer_tokens=63,645`
- full-NVFP4: `full_layer_tokens=282,655`, `swa_layer_tokens=226,124`
- allocator-token ratio: `3.553x` full pool, `3.553x` SWA pool

The server log proves:

- Gemma 4 unified model load: `Gemma4UnifiedForConditionalGeneration`
- D=512 VO-split routing via `extend_paged_vosplit0/1` and `decode_as_prefill_vosplit0/1`
- FP4 prefill module selection: `dtype_kv=__nv_fp4x2_e2m1`, `fp4_kv=1`
- FP4 scale views present in SGLang wrapper state: `k_sf={...}` and `v_sf={...}`

## Caveats

Known 12B text quality caveat remains in force: previous matched text PPL rows show full-NVFP4 red by `+0.402969` nats/token on the 8k corpus. This row is not a 12B quality-parity claim.

The server also logs the same image-attention caveat as bf16:

`Bidirectional attention for image tokens requires TritonAttnBackend. Falling back to causal attention, which may degrade image quality.`

Therefore this row is valid 12B SGLang full-NVFP4 multimodal serving-path evidence for text + image + audio under the current FlashInfer route, but broad 12B NVFP4 and image-quality claims remain scoped.
