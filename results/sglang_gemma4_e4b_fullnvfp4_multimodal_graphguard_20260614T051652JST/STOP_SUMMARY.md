# SGLang Gemma 4 E4B Full-NVFP4 Multimodal Graph-Request Guard

Status: GREEN, scoped graph-request guard evidence.

- Runtime: DGX Spark / GB10, packaged SGLang image `sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e`
- Model: `google/gemma-4-E4B-it`
- KV: `fp4_e2m1`, full NVFP4 K+V (`SGLANG_FP4_KV_MIXED_KV=0`)
- Backend: FlashInfer, `SGLANG_FLASHINFER_VOSPLIT=1`
- Shape: context length `512`, page size `1`, mem fraction `0.40`
- Graph condition: CUDA graph capture requested by omitting `--disable-cuda-graph`; `SGLANG_FP4_KV_ENABLE_CUDA_GRAPH` intentionally unset

## Verdict

The row proves the native-FP4 graph guard path on the packaged image:

- Parsed launch args show `disable_cuda_graph=False`.
- SGLang then logs:

  `Disabling CUDA graph capture for native FP4 KV cache. Current FlashInfer FA2 NVFP4 KV graph capture can produce corrupt decode output; set SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=1 only for graph-safety experiments.`

- No `Capture cuda graph begin` line appears in `server.log`.
- Serving remains green for the deterministic text, image, and audio probe.

This is not a native-FP4 CUDA graph replay proof or speed claim. It is the safe serving behavior when graph capture is requested without the explicit graph-safety override.

## Probe

`multimodal_probe.json` reports `ok=true`.

All rows returned HTTP 200, hit required keywords under `--keyword-mode all`, and were byte-stable across two repeats:

- text: `TOKYO` / `TOKYO`
- image: `red square and blue triangle` / `red square and blue triangle`
- audio: `Mr Quilter is the apostle of the middle classes and we are glad to welcome his gospel.` twice

Prompt-token counts:

- text: `18`
- image: `287`
- audio: `174`

## Capacity/Route Proof

At `--mem-fraction-static 0.40`, full-NVFP4 allocation reported:

- `full_layer_tokens=1276034`
- `swa_layer_tokens=1020827`

The server log proves:

- Gemma 4 multimodal model load: `Gemma4ForConditionalGeneration`
- D=512 VO-split routing via `extend_paged_vosplit0/1` and `decode_as_prefill_vosplit0/1`
- FP4 prefill module selection: `dtype_kv=__nv_fp4x2_e2m1`, `fp4_kv=1`
- FP4 K/V scale views present in the SGLang wrapper state: `k_sf={...}` and `v_sf={...}`

## Caveats

The same FlashInfer multimodal caveat applies as in the no-graph E4B/12B rows: image-token bidirectional attention is unsupported on this backend and falls back to causal attention. This row is request-path and guard-path evidence, not a broad image-quality claim.

The broader SGLang Gemma 4 caveats remain unchanged:

- E4B fp8 comparator is red on the D=512 VO-split paged-prefill FlashInfer dispatcher.
- 12B full-NVFP4 text PPL remains red by `+0.402969` nats/token on the 8k reused-prefix corpus.
