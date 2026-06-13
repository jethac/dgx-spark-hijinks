# Codex -> Claude: SGLang E4B multimodal graph-request guard is green, scoped

TL;DR: I ran the packaged-image E4B full-NVFP4 multimodal row with CUDA graph capture requested, and the native-FP4 safety guard disabled capture as intended. Text/image/audio serving stayed green. Artifact:

`results/sglang_gemma4_e4b_fullnvfp4_multimodal_graphguard_20260614T051652JST/STOP_SUMMARY.md`

Setup:

- image digest `sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e`
- model `google/gemma-4-E4B-it`
- full NVFP4 K+V, `SGLANG_FP4_KV_MIXED_KV=0`
- FlashInfer VO-split, ctx 512, page size 1, mem fraction 0.40
- no `--disable-cuda-graph`; no `SGLANG_FP4_KV_ENABLE_CUDA_GRAPH`

Key proof:

- `server_args` has `disable_cuda_graph=False`
- log contains the native-FP4 guard:
  `Disabling CUDA graph capture for native FP4 KV cache...`
- no `Capture cuda graph begin` line appears
- `multimodal_probe.json` is `ok=true`

Probe outputs were byte-stable across two repeats:

- text: `TOKYO`
- image: `red square and blue triangle`
- audio: `Mr Quilter is the apostle of the middle classes and we are glad to welcome his gospel.`

Scope: this is not native-FP4 CUDA graph replay support. It proves the safe graph-request fallback path on the packaged image. The existing E4B fp8 comparator red and 12B text-quality red remain unchanged.
