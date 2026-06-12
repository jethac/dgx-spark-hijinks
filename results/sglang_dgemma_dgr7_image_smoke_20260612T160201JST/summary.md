# SGLang DiffusionGemma DG-R7 Image Smoke

Status: RED

## Scope

Stock DiffusionGemma multimodal image prompt smoke on GB10. This row uses
the upstream policy path: Triton attention, BF16/auto KV, eager execution,
and unchunked prefill. It does not claim FlashInfer, NVFP4, capacity,
throughput, or image-generation quality.

## Provenance

- Run ID: `sglang_dgemma_dgr7_image_smoke_20260612T160201JST`
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`
- SGLang commit: `98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968`
- FlashInfer source commit: `f99323bd7d1c`
- Model: `google/diffusiongemma-26B-A4B-it`
- Request API: `/v1/chat/completions` with OpenAI `image_url` content

## Gates

- server reached readiness: `true`
- stock Triton policy proof: `True`
- image quality client status: `1`
- semantic/stability gate: `False`
- server log contains image-related diagnostics: `True`

## Checks

- `red_blue_halves` sha256 `2c4b727cc52a2545160f0699eed27e844a61573a53294a96009fd34ea9f9ca71`: stable `True`, non_empty `True`, answer_ok `True`, texts `['Red, blue', 'Red, blue']`
- `green_square` sha256 `481cb5bb2b99689cce735e1fd17789e246699ff47af5ef252a87f2e28bbade55`: stable `True`, non_empty `False`, answer_ok `False`, texts `['', '']`

## Decision

The stock DiffusionGemma image-prompt path is not claim-grade under
this gate. Treat this as the next diagnostic row before any
multimodal serving claim.
