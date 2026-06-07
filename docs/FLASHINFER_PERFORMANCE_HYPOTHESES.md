# FlashInfer Performance Hypotheses

Date: 2026-06-07

Status: hypotheses narrowed by first GB10 microbenchmarks.

This document records where the FlashInfer SM121 work might still produce real performance gains, and where current evidence says it probably will not.

## Current Evidence

Patch under test:

- fork: `jethac/flashinfer`
- branch: `spark/hijinks-004-sm121-flashinfer`
- commit: `a42c8f07`

Proven so far:

- installed vLLM/SGLang containers exclude `b12x` from SM121 NVFP4 `mm_fp4` auto-dispatch.
- latest upstream vLLM and SGLang releases inherit that exclusion through pinned FlashInfer versions.
- patched FlashInfer source changes real GB10 dispatch to `["b12x", "cutlass", "cudnn"]`.
- a tiny forced-`b12x` NVFP4 GEMM can run on GB10 with finite output and sane cosine similarity.
- Gemma 4 26B A4B can serve through vLLM on GB10 at about 24 tok/s decode on the compact OpenAI harness, but the observed path is BF16/unquantized Triton MoE, not FlashInfer NVFP4.

Microbenchmark result:

- artifact: `results/flashinfer_mm_fp4_auto_microbench_20260607T1300Z.json`
- script: `scripts/flashinfer_mm_fp4_microbench.py`
- cases: `1x128x128`, `16x256x256`, `64x512x512`
- result: patched `b12x` auto-dispatch was not faster than the installed `cudnn`/`cutlass` auto path on those small dense cases.

## Where Speedups Are Plausible

### 1. Underfilled Decode Dense GEMMs

Why plausible:

- decode often has very small token count (`M=1`, `4`, `8`, `16`) but model-sized projection dimensions.
- GB10 has modest memory bandwidth relative to its tensor-core peak, so avoiding bad tile choices and launch overhead matters.
- FlashInfer's `b12x` path is explicitly aimed at SM12x consumer Blackwell behavior and underfilled tile selection.

Shapes to test first:

- `1x4096x4096`
- `4x4096x4096`
- `16x4096x4096`
- `1x8192x4096`
- `4x8192x4096`
- `16x8192x4096`

Expected outcome:

- possible small to moderate kernel-level gains if cuDNN/CUTLASS are poorly matched for low-`M` decode.
- no claim until measured.

### 2. MoE Expert GEMMs

Why plausible:

- MoE routing creates small, uneven per-expert token batches.
- Expert matrices are wide enough that the tiny smoke cases are not representative.
- The community and upstream FlashInfer work around `b12x` is strongly MoE-shaped, not only dense-GEMM-shaped.

Shapes to test first:

- `1x14336x4096`
- `4x14336x4096`
- `16x14336x4096`
- `1x4096x14336`
- `4x4096x14336`
- `16x4096x14336`

Expected outcome:

- more plausible than generic dense smoke shapes.
- still only a proxy for real fused MoE kernels; serving benchmarks on an MoE model remain required.

### 3. Packaging And JIT-Cache Correctness

Why plausible:

- direct Python overlay failed with CUTLASS DSL and TVM FFI mismatches.
- stale `flashinfer-jit-cache` and `flashinfer-cubin` packages can route to old generated modules even when Python source is patched.
- adding `12.1a` to build matrices may matter for package correctness even when a given module uses `120f`.

Expected outcome:

- user-visible improvement may come from making a clean image work reliably, not from one kernel becoming faster.
- this matters for reproducibility and "no dependency surgery" acceptance criteria.

### 4. End-To-End NVFP4 Serving

Why plausible:

- model serving uses multiple kernels: quantization, GEMM, attention, KV cache, CUDA graphs, scheduler behavior, and memory allocation.
- a microkernel speedup only matters if it is on the critical path.
- a routing fix may unlock a path that serving code previously avoided.

Expected outcome:

- unknown until measured against a real model with stable prompts and a clean image/wheel set.

## Gemma 4 26B Follow-Up

The compact 26B MoE check is useful, but it narrows rather than proves the FlashInfer performance story.

Observed:

- `google/gemma-4-26B-A4B-it` served successfully with `vllm/vllm-openai:latest-cu130`.
- compact decode throughput was about 24 tok/s across short, medium, and long-prefill cases.
- vLLM selected `TRITON_ATTN` and `TRITON Unquantized MoE`.
- `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` also served, but vLLM reported `quantization=None`, `dtype=torch.bfloat16`, and the same unquantized MoE backend.

Conclusion:

- Gemma 4 26B A4B is a good campaign-level serving benchmark because it is large enough, MoE-shaped, and user-relevant.
- It is not yet a benchmark of the SM121 NVFP4 `mm_fp4` dispatch fix.
- The next end-to-end proof needs a genuinely quantized model/runtime path whose logs or profiler trace show FlashInfer NVFP4 or another SM121-specific FP4 backend on the critical path.

## Where Speedups Are Not Currently Supported

Do not claim speedup for generic small dense `mm_fp4` based on current data.

Measured cases showed patched auto-dispatch was slower:

| case | installed mean ms | patched mean ms | patched latency change |
|---|---:|---:|---:|
| `1x128x128` | 0.0727 | 0.0769 | +5.9% |
| `16x256x256` | 0.0654 | 0.0661 | +1.0% |
| `64x512x512` | 0.0651 | 0.0757 | +16.3% |

Interpretation:

- this patch is not a blanket "b12x is faster" result.
- it is currently a dispatch correctness and enablement result.
- performance work must now target model-shaped or MoE-shaped cases.

## Measurement Plan

Use:

```bash
python3 scripts/flashinfer_mm_fp4_microbench.py \
  --phase before \
  --run-id flashinfer-mm-fp4-dense-decode-before \
  --container CONTAINER_TAG \
  --preset dense_decode \
  --iterations 100 \
  --output results/flashinfer_mm_fp4_dense_decode_before.json
```

Then repeat under the patched source/JIT container path with the same preset and iterations.

Rules:

- compare only warmed timings, not first-run JIT compile time.
- keep source/JIT package versions consistent; do not overlay patched Python on stale FlashInfer binaries.
- record the heuristic order, FlashInfer file path, version, CUDA version, and compute capability.
- treat microbenchmarks as diagnostic. Serving benchmark rows are still required before claiming user-visible speedup.
