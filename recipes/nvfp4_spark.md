# NVFP4 On Spark

Status: not blessed.

DGX Spark advertises strong FP4 capability, but this repository does not currently bless NVFP4 KV cache or NVFP4 serving paths.

## Rules

- Validate on `sm_121`, not just `sm_120`.
- Keep fp8 KV as the default until NVFP4 KV is proven correct and faster for the target model.
- Record correctness and speed; speed alone is not enough.
- Keep patched FlashInfer/vLLM paths labeled until upstreamed.

## Minimal Acceptance Test

- model loads
- deterministic short prompt produces sane text
- logits or token choices are consistent against a reference path
- prefill and decode speeds are measured against fp8/bf16
- logs identify the selected KV cache and quantization backend

