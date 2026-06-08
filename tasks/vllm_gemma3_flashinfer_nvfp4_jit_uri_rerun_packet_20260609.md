# vLLM Gemma 3 FlashInfer NVFP4 JIT URI Rerun Packet, 2026-06-09

Status: completed red. Result summary:
`results/vllm_gemma3_27b_jituri_20260609T0319JST_summary.md`.

Purpose: rerun the Gemma 3 27B wrapper-boundary failure with
`jethac/flashinfer@3db181f4`, which gives packed NVFP4 KV modules a distinct
`fp4x2_e2m1` JIT namespace.

## Inputs

- vLLM fork: current `third_party/vllm` pointer
  (`13da71884640567682cd3ddd4650d2ba3ecb5543` at packet creation).
- FlashInfer fork:
  - branch `spark/hijinks-020-nvfp4-jit-uri`
  - commit `3db181f4`
- Baseline failing artifacts:
  - `results/vllm_gemma3_27b_wrapper_trace_20260609T0148JST_summary.md`
  - `results/vllm_gemma3_27b_active_page_dump_20260609T0216JST_summary.md`
  - `results/vllm_gemma3_27b_active_page_replay_20260609T0216JST_summary.md`
- Patch artifact:
  - `results/vllm_gemma3_27b_flashinfer_nvfp4_jit_uri_patch_20260609.md`

## Preflight

Inside the existing GB10 vLLM source-overlay container:

1. Install or overlay `jethac/flashinfer@3db181f4` without changing Torch, CUDA, vLLM, or
   FlashAttention versions.
2. Clear FlashInfer generated/JIT cache directories used by the container before server
   start. Do not rely on timestamp rebuild alone.
3. Use `scripts/flashinfer_source_sitecustomize.py` from commit `258d4bf` or newer. It must
   monkeypatch the installed `flashinfer.prefill.get_batch_prefill_uri` binding to emit
   `dtype_kv_fp4x2_e2m1` for packed NVFP4 KV; otherwise the container may still use the
   installed old `dtype_kv_u8` URI helper and the patch will not be exercised. Do not
   prepend the whole `/flashinfer-src` package to `PYTHONPATH` in this image: that path
   currently trips a CuteDSL/CUTLASS Python dependency mismatch.
4. Set:

```bash
export FLASHINFER_JIT_VERBOSE=1
export FLASHINFER_PAGED_V_SF_DESWIZZLE=1
```

Keep the same vLLM trace env used by the previous wrapper-boundary packet:

```bash
export VLLM_SPARK_TENSOR_TRACE=1
export VLLM_SPARK_WRAPPER_TRACE=1
export VLLM_SPARK_ACTIVE_PAGE_DUMP=1
```

Use the same layer filter and trace-output paths as the prior packet, adjusted only for a
new timestamp.

## Run

Repeat the minimal fp8 comparator and NVFP4 candidate first-token probes from
`tasks/vllm_gemma3_wrapper_boundary_trace_packet_20260609.md`.

Do not climb to Gemma 4 31B in this run.

## Required Evidence

Capture:

- server log;
- FlashInfer JIT verbose output or generated source path;
- wrapper trace JSONL;
- active-page dump payloads if NVFP4 still fails;
- first-token comparator output;
- summary markdown.

The summary must answer:

1. Did the generated batch-prefill module name contain `dtype_kv_fp4x2_e2m1`?
2. Did the generated config include the FP4 static assertion?
3. Did Gemma 3 first-token quality recover versus fp8?
4. If quality did not recover, did `out_after` remain byte-like or become signed/small?

## Decision

- If quality recovers: bless this as the vLLM Gemma 3 NVFP4-KV fix candidate and rerun the
  capacity/quality row before climbing to Gemma 4 31B.
- If module naming does not change: fix the source overlay/JIT-cache path first.
- If naming changes but `out_after` stays byte-like: move next to FlashInfer
  `compute_sfm_v()` / BF16 FP4 conversion instrumentation.

Live decision: module naming changed and the kernel JIT-built under
`dtype_kv_fp4x2_e2m1`, but `out_after` stayed byte-like and quality stayed red. Proceed to
FlashInfer paged-prefill FP4-KV conversion instrumentation.
