# NVFP4 On Spark

Status: partially proven, not broadly blessed.

DGX Spark advertises strong FP4 capability, but NVFP4 work has to be split into separate tracks. A result for one track does not validate the others.

## Tracks

| track | purpose | current status |
|---|---|---|
| NVFP4 weights | reduce model-weight bandwidth and enable native FP4 linear/MoE kernels where supported | AEON external Qwen/Gemma evidence is strong; local reproduction pending |
| FP8/BF16 KV baseline | safe serving comparator for quality, speed, and capacity | required before any FP4 KV claim |
| NVFP4/FP4 KV | reduce KV-cache footprint and increase long-context/concurrency capacity | vLLM/FlashInfer standalone FA2 probe passes some shapes; SGLang `fp4_e2m1` serving not proven locally |

## Rules

- Validate on `sm_121`, not just `sm_120`.
- Keep fp8 KV as the default until NVFP4 KV is proven correct and faster for the target model.
- Record correctness and speed; speed alone is not enough.
- Keep patched FlashInfer/vLLM paths labeled until upstreamed.
- Do not describe AEON Gemma or Qwen NVFP4-weight rows as NVFP4-KV evidence. Those rows use model weight quantization and backend policy; KV cache is a separate experiment.
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

For NVFP4 weights, also record selected linear and MoE backends. For NVFP4/FP4 KV, also record KV pool tokens, maximum concurrency, hidden scratch allocations if visible, and fp8-vs-fp4 quality.
