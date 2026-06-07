# NVFP4 On Spark

Status: not blessed.

DGX Spark advertises strong FP4 capability, but this repository does not currently bless NVFP4 KV cache or NVFP4 serving paths.

## Rules

- Validate on `sm_121`, not just `sm_120`.
- Keep fp8 KV as the default until NVFP4 KV is proven correct and faster for the target model.
- Record correctness and speed; speed alone is not enough.
- Keep patched FlashInfer/vLLM paths labeled until upstreamed.
- Track vLLM NVFP4 KV separately from FlashInfer dense/MoE GEMM dispatch. `hikarioyama/vllm-nvfp4-kv-sm120` reports a vLLM SM120 NVFP4 KV path through patched FlashInfer FA2 explicit scale-factor stride handling, with roughly fp8-like speed and a larger KV pool on RTX PRO 6000-class hardware. This is reference evidence, not GB10 `sm_121` validation.
- Track SGLang separately from vLLM. `hikarioyama/sglang-nvfp4-kv-sm120` reports SGLang `fp4_e2m1` KV cache with FlashInfer FA2 patches, native FP4 pool, hybrid-SWA support, and per-layer global-scale auto-calibration.
- Treat small models as likely fp8-KV defaults until NVFP4 quality is proven; the SGLang repo reports small-model incoherence under NVFP4 KV even when larger models are near-lossless.
- Unless a measured GB10 result says otherwise, build on hikarioyama's vLLM and SGLang NVFP4-KV work as prior art, but port changes through `jethac` forks and issue-named worktrees with upstream contributing guidelines followed.

## Minimal Acceptance Test

- model loads
- deterministic short prompt produces sane text
- logits or token choices are consistent against a reference path
- prefill and decode speeds are measured against fp8/bf16
- logs identify the selected KV cache and quantization backend
