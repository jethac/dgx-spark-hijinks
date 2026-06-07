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
- `hikarioyama/vllm-nvfp4-kv-sm120` reports a vLLM NVFP4 KV cache path for SM120 RTX PRO 6000 using a FlashInfer FA2 explicit-scale-factor-stride patch; the repo summary claims roughly 1.5x fp8 KV pool capacity and 95-104% fp8 speed on that hardware class.
- community vLLM NVFP4-KV work modifies both vLLM glue and FlashInfer FA2/CUDA/JIT surfaces. This is a different problem from the current FlashInfer `mm_fp4` auto-dispatch patch, which targets dense/MoE GEMM backend selection rather than KV-cache storage or attention decode.
- current FlashInfer has SM12x support work, but some code still distinguishes `(12, 0)` specifically.
- `hikarioyama/sglang-nvfp4-kv-sm120` ports the same NVFP4-KV idea into SGLang and adds SGLang memory-pool, hybrid-SWA, and calibration plumbing. SGLang pins or fetches FlashInfer-related dependencies, so native SGLang NVFP4 KV probably needs SGLang and FlashInfer work together.

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
  - the later model-shaped SGLang patched run built FP4 GEMM code under `/root/.cache/flashinfer/0.6.13/121a/cached_ops/fp4_gemm_cutlass_sm120` with observed NVCC `arch=compute_121a,code=sm_121a`.
  - a tiny forced-`b12x` NVFP4 GEMM on GB10 produced finite BF16 output with cosine similarity `0.9882067441940308` against BF16 `torch.mm`.
  - overlaying patched Python on old installed FlashInfer binaries is invalid: it hit CUTLASS DSL and TVM FFI signature mismatches. A real deployment needs matching `flashinfer-python`, JIT-cache/cubin packages, CUTLASS DSL, and CUDA targets.
- Microbenchmark evidence:
  - artifact: `results/flashinfer_mm_fp4_auto_microbench_20260607T1300Z.json`
  - script: `scripts/flashinfer_mm_fp4_microbench.py`
  - on three small dense NVFP4 `mm_fp4` cases, patched SM121 `b12x` auto-dispatch was not faster than the installed `cudnn`/`cutlass` auto path.
  - model-shaped proxy artifacts: `results/flashinfer_mm_fp4_sglang_installed_dense_decode_20260607T161500Z.json`, `results/flashinfer_mm_fp4_sglang_installed_moe_expert_20260607T161500Z.json`, `results/flashinfer_mm_fp4_sglang_patched_modelshape_20260607T162000Z_dense_decode.json`, and `results/flashinfer_mm_fp4_sglang_patched_modelshape_20260607T162000Z_moe_expert.json`.
  - on SGLang 26.05, patched dense-decode proxies were mixed and patched MoE-shaped proxies were slower on all tested shapes.
  - this narrows the expected performance win: the patch is proven as dispatch enablement, but user-visible speedup still needs fused serving paths, NVFP4 KV, model-specific quantization plumbing, CUDA graph behavior, or clean package builds to prove it.
  - performance hypotheses are tracked in `docs/FLASHINFER_PERFORMANCE_HYPOTHESES.md`.
- vLLM NVFP4-KV reference status:
  - reference repo: `https://github.com/hikarioyama/vllm-nvfp4-kv-sm120`
  - audited HEAD: `f6156ee3b22b24885a52c02bdafb34a9c201fe86`
  - current classification: SM120 reference evidence, not a GB10 `sm_121` result.
  - relevant launch surface: `--kv-cache-dtype nvfp4` or equivalent vLLM KV-cache flagging plus a patched FlashInfer FA2 path with explicit scale-factor stride handling.
  - reported Step3.7-Flash path: TP=2, MTP K=1, `--quantization modelopt`, `--enable-expert-parallel`, `--max-model-len 131072`, `--max-num-seqs 32`, CUDA graph capture sizes `1,2,4,8,16`.
  - reported SM120 result: fp8 KV pool around 1.66M tokens versus NVFP4 around 2.96M-3.08M tokens, with decode in the same rough range and quality deltas around `+0.01` to `+0.04` nats/token PPL.
  - primary payoff: capacity and concurrency, not weight-GEMM FLOPs. At matched utilization, the reference result claims 1.78x fp8 KV pool and 22.59x versus 12.70x concurrency at 131k context.
  - default-path blocker: SM120 lacks the trtllm-gen NVFP4-KV cubins used by vLLM's default NVFP4 path, so the reference routes KV through FlashInfer FA2 and standard `mma.sync`.
  - important bug class: the FA2 NVFP4 kernel inferred scale-factor strides from data strides, but vLLM stores interleaved page data and scales. The reference patch passes explicit scale-factor strides and reads K/V scales from the interleaved cache.
  - important telemetry lesson: an interim V scale-factor scratch cache looked cheap but silently consumed extra memory. The B2 path de-swizzles V scale factors in-kernel in registers and should be measured with memory telemetry that can catch hidden allocations.
  - GB10 hypothesis: `_use_fa2_for_nvfp4_kv_on_sm120()` gates on compute capability 12.0/12.1 in the reference repo, so the path is nominally intended to include GB10. This still requires a real `sm_121` build and serving proof.
  - stated limits: standard attention only; head dimensions 64/128/256/512; not MLA, Mamba/SSM, or attention sinks.
  - current GB10 proof: `results/flashinfer_nvfp4_kv_probe_20260608T023901JST.json` runs `jethac/flashinfer@e152cf4d` through FlashInfer FA2 paged decode/prefill with vLLM-style swizzled V scale factors and in-kernel de-swizzle enabled. NHD and HND both pass with cosine >= `0.99999946`.
  - next proof: extend the standalone harness to model-shaped `H_q/H_kv/D/page` values, then run a clean vLLM server with `--kv-cache-dtype nvfp4` and compare fp8-vs-NVFP4 capacity, quality, and throughput.
  - not covered by our current tests: fp8-vs-NVFP4 KV capacity, output quality, long-context attention decode, CUDA graph replay, Gemma alternating local/global attention behavior, and whether the same patches build cleanly as distributable wheels/containers for `sm_121`/`121a` on the Spark stack.
- SGLang NVFP4-KV reference status:
  - reference repo: `https://github.com/hikarioyama/sglang-nvfp4-kv-sm120`
  - audited HEAD: `9b2160f0fb8e11dbbb5171a57f06a02b0e9ba6e2`
  - current classification: SM120 reference evidence, not a GB10 `sm_121` result.
  - relevant launch surface: `--kv-cache-dtype fp4_e2m1 --attention-backend flashinfer --page-size 1`.
  - reported SM120 result: Qwen2.5 and Step3.7-Flash validation, 1.778x KV capacity versus fp8 in the documented Step3.7 case, roughly fp8-like decode on larger models, and small-model quality failures even after calibration.
  - not covered by our current tests: SGLang fp8 baseline on GB10, SGLang NVFP4-KV quality/perf on GB10, and Gemma-family serving under SGLang.
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
