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
- commit: `a42c8f07`
- changes:
  - `_heuristic_func_mm_fp4` now treats all SM12x devices as b12x candidates for CUDA 13 + NVFP4.
  - release and nightly aarch64 JIT-cache builds for CUDA 12.9+/13 include `12.1a`.
  - install docs show the DGX Spark `FLASHINFER_CUDA_ARCH_LIST` including `12.1a`.
  - XQA/MLA error messages now say SM12x or the correct SM121a CUDA 12.9 minimum.
  - `tests/gemm/test_mm_fp4.py` adds a monkeypatched regression proving SM121 + CUDA 13 + NVFP4 auto-dispatch prefers `b12x`, then `cutlass`, then `cudnn`.
- upstream guidance:
  - FlashInfer `CONTRIBUTING.md` was checked before shaping the fork patch.
  - The patch follows the documented pattern: Python interface change, unit test under `tests/`, documentation update, and release build-matrix update.
- local verification:
  - `python -m py_compile tests\gemm\test_mm_fp4.py flashinfer\gemm\gemm_base.py flashinfer\xqa.py flashinfer\mla\_core.py`
  - targeted pytest collection currently fails in this Windows workspace because `tvm_ffi` is not installed; this is an environment dependency required by FlashInfer's test conftest.
- Spark source/JIT verification:
  - artifact: `results/flashinfer_sm121_source_jit_20260607T1250Z.json`
  - installed vLLM and SGLang containers both returned `["cudnn", "cutlass"]` for real SM121 NVFP4 `mm_fp4` auto-dispatch, excluding `b12x`.
  - patched FlashInfer source returned `["b12x", "cutlass", "cudnn"]` on real GB10 in both containers after upgrading the documented source dependency `nvidia-cutlass-dsl[cu13]>=4.5.0`.
  - in an ephemeral SGLang container, editable FlashInfer `0.6.13` source with stale `flashinfer-jit-cache`/`flashinfer-cubin` removed built FP4 quantization under `/root/.cache/flashinfer/0.6.13/121a/cached_ops/fp4_quantization_120f`.
  - the observed NVCC line used `arch=compute_120f,code=sm_120f`, which is the SM12x family target FlashInfer chooses for this quantization module on CUDA >= 12.9.
  - a tiny forced-`b12x` NVFP4 GEMM on GB10 produced finite BF16 output with cosine similarity `0.9882067441940308` against BF16 `torch.mm`.
  - overlaying patched Python on old installed FlashInfer binaries is invalid: it hit CUTLASS DSL and TVM FFI signature mismatches. A real deployment needs matching `flashinfer-python`, JIT-cache/cubin packages, CUTLASS DSL, and CUDA targets.
- Microbenchmark evidence:
  - artifact: `results/flashinfer_mm_fp4_auto_microbench_20260607T1300Z.json`
  - script: `scripts/flashinfer_mm_fp4_microbench.py`
  - on three small dense NVFP4 `mm_fp4` cases, patched SM121 `b12x` auto-dispatch was not faster than the installed `cudnn`/`cutlass` auto path.
  - this narrows the expected performance win: the patch is proven as dispatch enablement, but user-visible speedup still needs model-shaped GEMMs, MoE paths, or serving benchmarks to prove it.
- not yet proven:
  - clean wheel or container build suitable for vLLM/SGLang serving
  - `cuobjdump` evidence from a distributable artifact
  - FP4 speedup versus stock FlashInfer
  - end-to-end NVFP4 KV correctness against fp8/bf16 serving reference
  - upstream FlashInfer review/CI, including Spark CI if a maintainer can trigger it

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
