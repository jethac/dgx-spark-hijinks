# SGLang Gemma 4 E4B bf16 Multimodal Shape Smoke

Status: GREEN, scoped.

- Runtime: DGX Spark / GB10, packaged SGLang image `sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e`
- Model: `google/gemma-4-E4B-it`
- KV: bf16 / auto
- Backend: FlashInfer, `SGLANG_FLASHINFER_VOSPLIT=1`, CUDA graphs disabled
- Scope: multimodal request-path baseline only; no NVFP4, capacity, or final image-quality claim

## Probe

Prompt asset: generated deterministic `red_square_blue_triangle.png`.

All rows returned HTTP 200, hit the required keywords, and were byte-stable across two repeats:

- text: `TOKYO` / `TOKYO`
- image: `red square and blue triangle` / `red square and blue triangle`
- audio: `Mr Quilter is the apostle of the middle classes and we are glad to welcome his gospel.` twice

Usage:

- text prompt tokens: 18
- image prompt tokens: 287
- audio prompt tokens: 174

## Routing/Caveat

The server log proves Gemma 4 multimodal model load (`Gemma4ForConditionalGeneration`) and D=512 VO-split routing via `extend_paged_vosplit0/1` and `decode_as_prefill_vosplit0/1`.

The server also logs:

`Bidirectional attention for image tokens requires TritonAttnBackend. Falling back to causal attention, which may degrade image quality.`

Therefore this row is a valid SGLang FlashInfer multimodal serving/request-path baseline for text + image + audio, but broad image-quality claims remain scoped until the bidirectional-image-attention path is handled or explicitly accepted.
