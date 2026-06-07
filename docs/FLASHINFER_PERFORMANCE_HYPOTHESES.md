# FlashInfer Performance Hypotheses

Date: 2026-06-07

Status: hypotheses narrowed by first GB10 microbenchmarks and model-shaped proxy cases.

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
- patched source/JIT in the SGLang 26.05 container compiled SM121a-targeted FlashInfer FP4 GEMM code and produced finite outputs on dense-decode and MoE-shaped `mm_fp4` cases.

Small dense microbenchmark result:

- artifact: `results/flashinfer_mm_fp4_auto_microbench_20260607T1300Z.json`
- script: `scripts/flashinfer_mm_fp4_microbench.py`
- cases: `1x128x128`, `16x256x256`, `64x512x512`
- result: patched `b12x` auto-dispatch was not faster than the installed `cudnn`/`cutlass` auto path on those small dense cases.

Model-shaped proxy result:

- installed artifact, dense: `results/flashinfer_mm_fp4_sglang_installed_dense_decode_20260607T161500Z.json`
- installed artifact, MoE: `results/flashinfer_mm_fp4_sglang_installed_moe_expert_20260607T161500Z.json`
- patched artifact, dense: `results/flashinfer_mm_fp4_sglang_patched_modelshape_20260607T162000Z_dense_decode.json`
- patched artifact, MoE: `results/flashinfer_mm_fp4_sglang_patched_modelshape_20260607T162000Z_moe_expert.json`
- result: patched `b12x` auto-dispatch was mixed-to-slower on dense decode proxies and slower on every MoE-shaped proxy case.

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

- measured SGLang container result was mixed: faster for `4x4096x4096` and `16x4096x4096`, slower for the other four dense-decode shapes.
- no serving claim from this path without a real model trace showing this kernel is on the critical path.

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

- first proxy result was negative: patched `b12x` auto-dispatch was slower on all six tested MoE-shaped cases.
- still only a proxy for real fused MoE kernels; serving benchmarks on an actually quantized MoE model remain required before closing the NVFP4 question.

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

Do not claim speedup for the first model-shaped proxy cases either.

Dense decode proxy, SGLang 26.05 container:

| case | installed mean ms | patched mean ms | patched latency change |
|---|---:|---:|---:|
| `1x4096x4096` | 0.0738 | 0.0893 | +21.1% |
| `4x4096x4096` | 0.0704 | 0.0677 | -3.9% |
| `16x4096x4096` | 0.0692 | 0.0620 | -10.4% |
| `1x8192x4096` | 0.0707 | 0.0857 | +21.3% |
| `4x8192x4096` | 0.0700 | 0.0786 | +12.4% |
| `16x8192x4096` | 0.0709 | 0.0741 | +4.5% |

MoE-shaped proxy, SGLang 26.05 container:

| case | installed mean ms | patched mean ms | patched latency change |
|---|---:|---:|---:|
| `1x14336x4096` | 0.1443 | 0.1543 | +6.9% |
| `4x14336x4096` | 0.1382 | 0.1510 | +9.3% |
| `16x14336x4096` | 0.1413 | 0.1535 | +8.6% |
| `1x4096x14336` | 0.1401 | 0.1688 | +20.5% |
| `4x4096x14336` | 0.1397 | 0.1551 | +11.0% |
| `16x4096x14336` | 0.1390 | 0.1546 | +11.2% |

Interpretation:

- the one-line FlashInfer SM121 `b12x` gate fix is necessary enablement, not a banked speedup.
- the most likely remaining wins are in fused serving paths, NVFP4 KV, model-specific quantization plumbing, CUDA graph compatibility, or packaging that avoids stale cubin/JIT-cache artifacts.
- an end-to-end NVFP4 MoE row with logs or profiler evidence is still the decision point for whether this patch matters to user-visible throughput.

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
