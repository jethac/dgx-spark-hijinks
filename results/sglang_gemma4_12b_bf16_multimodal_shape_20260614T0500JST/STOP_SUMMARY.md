# SGLang Gemma 4 12B bf16 Multimodal Shape Smoke

Status: GREEN, scoped baseline.

- Runtime: DGX Spark / GB10, packaged SGLang image `sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e`
- Model: `google/gemma-4-12B-it`
- KV: bf16 / auto
- Backend: FlashInfer, `SGLANG_FLASHINFER_VOSPLIT=1`, CUDA graphs disabled
- Scope: multimodal request-path baseline only; no NVFP4, capacity, or broad quality claim

## Probe

Prompt asset: generated deterministic `red_square_blue_triangle.png`; audio asset is the banked LibriVox/Quilter clip.

All rows returned HTTP 200, hit the required keywords under `--keyword-mode all`, and were byte-stable across two repeats:

- text: `Tokyo` / `Tokyo`
- image: `red square and blue triangle` / `red square and blue triangle`
- audio: `Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.` twice

Usage:

- text prompt tokens: 22
- image prompt tokens: 291
- audio prompt tokens: 178

## Capacity/Route Baseline

At `--mem-fraction-static 0.40`:

- bf16 full-layer tokens: `79,557`
- bf16 SWA tokens: `63,645`

The server log proves Gemma 4 unified model load (`Gemma4UnifiedForConditionalGeneration`) and D=512 VO-split routing through `extend_paged_vosplit0/1` and `decode_as_prefill_vosplit0/1`.

## Caveat

The server logs:

`Bidirectional attention for image tokens requires TritonAttnBackend. Falling back to causal attention, which may degrade image quality.`

Therefore this row is a valid 12B bf16 multimodal request-path baseline for text + image + audio under the current FlashInfer route, but not a broad image-quality claim.
