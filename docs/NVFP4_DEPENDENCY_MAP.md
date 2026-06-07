# NVFP4 Dependency Map

Status: active FlashInfer patch branch exists.

This map records which upstreams are likely involved if we patch NVFP4 KV for Spark.

## Likely Forks If Patched

| upstream | fork | why |
|---|---|---|
| `vllm-project/vllm` | `jethac/vllm` | vLLM NVFP4 KV runtime gates and backend plumbing |
| `flashinfer-ai/flashinfer` | `jethac/flashinfer` | FlashInfer FA2/JIT/CUDA scale-factor and SM12x behavior |
| `sgl-project/sglang` | `jethac/sglang` | SGLang memory pool, KV dtype, scale, and wrapper plumbing |
| `vllm-project/FlashMLA` | `jethac/FlashMLA` only if needed | Spark MLA/DeepSeek/FlashMLA claims |
| `vllm-project/flash-attention` | `jethac/flash-attention` only if needed | vLLM bundled FA2/FA3 audit or patch |
| `NVIDIA/cutlass` | only if fix lands below wrapper layer | avoid forking unless FlashInfer/CuTeDSL boundary requires it |

## Current Evidence

From `results/cuda_so_audit_vllm_flashinfer_20260607T111023Z.json`:

- inspected objects: 14
- explicit `sm_121`: 0
- `sm_120`: 3
- vLLM FA2 extension: `sm_80`
- vLLM FA3 extension: `sm_90a`
- vLLM FlashMLA extensions: `sm_100`, `sm_90a`

From the subagent read-only upstream investigation:

- current vLLM source has NVFP4 KV support surfaces, but FlashInfer backend behavior still needs Spark-specific validation.
- community vLLM NVFP4 work modifies both vLLM glue and FlashInfer FA2/CUDA/JIT surfaces.
- current FlashInfer has SM12x support work, but some code still distinguishes `(12, 0)` specifically.
- SGLang pins or fetches FlashInfer-related dependencies, so native SGLang NVFP4 KV probably needs SGLang and FlashInfer work together.

From FlashInfer issue #3170:

- `compute_120f` covers SM120 and SM121 for family-compatible features, but native NVFP4/MXFP4 MMA requires arch-specific `120a` or `121a`.
- FlashInfer release and nightly JIT-cache wheels used `12.0f` for CUDA 12.9+/13 builds and did not include `12.1a`.
- `mm_fp4` auto-dispatch had a high-impact gate that preferred b12x only when `major == 12 and minor == 0`, excluding GB10 / SM121 from the b12x NVFP4 path.
- Some SM100/CuTe DSL paths use datacenter Blackwell features and cannot be enabled for SM12x by gate relaxation alone.

Patch branch:

- fork: `jethac/flashinfer`
- branch: `spark/hijinks-004-sm121-flashinfer`
- commit: `e07a6392`
- changes:
  - `_heuristic_func_mm_fp4` now treats all SM12x devices as b12x candidates for CUDA 13 + NVFP4.
  - release and nightly aarch64 JIT-cache builds for CUDA 12.9+/13 include `12.1a`.
  - install docs show the DGX Spark `FLASHINFER_CUDA_ARCH_LIST` including `12.1a`.
- not yet proven:
  - build success on the Spark
  - `cuobjdump`/JIT-cache evidence showing `sm_121a`
  - FP4 speedup versus stock FlashInfer
  - correctness against fp8/bf16 reference

## Evidence Required Before Blessing NVFP4 KV

- Spark-specific build evidence for GB10 / `sm_121`.
- Build logs showing `sm_121`, `sm_121a`, or a documented valid SM12x family target.
- `cuobjdump` or JIT-cache audit proving the installed kernels match the claimed path.
- Runtime logs proving FlashInfer FA2 native NVFP4 KV, not fp8/bf16 fallback.
- Correctness against a dequant or fp8/bf16 reference.
- Coverage for decode, prefill, CUDA graph replay, page sizes, head dims, GQA, long context, and peaked qK.
- Quality comparison on target models.
- Performance comparison with warmed JIT/CUDA graph.
- Explicit scope labels for untested areas: MLA/FlashMLA, Mamba/SSM, attention sinks, hybrid-SWA, MTP/spec decode, TP>1, and SGLang page-size variants.
