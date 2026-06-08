# vLLM Gemma 3 27B FlashInfer NVFP4 JIT URI Patch, 2026-06-09

## Purpose

Stage the smallest patch that falsifies the current top vLLM/Gemma NVFP4-KV hypothesis:
a stale FlashInfer JIT/AOT module can be reused because packed NVFP4 KV kernels are named
as raw `u8` KV kernels.

This is not a live quality result. It is a fork patch plus a rerun packet.

## Fork Patch

FlashInfer fork branch:

```text
jethac/flashinfer spark/hijinks-020-nvfp4-jit-uri
commit 3db181f4 Disambiguate FP4 KV JIT module names
```

Changes:

- `flashinfer/jit/utils.py`
  - adds `filename_safe_dtype_map_kv(dtype)`;
  - returns `fp4x2_e2m1` whenever the generated KV C++ type is
    `__nv_fp4x2_e2m1`.
- `flashinfer/jit/attention/modules.py`
  - uses the KV-specific dtype name in attention KV cache JIT URIs;
  - `torch.uint8` no longer generates attention module names containing `dtype_kv_u8`
    when the generated C++ type is the packed FP4 container.
- `csrc/batch_prefill_customize_config.jinja`
  - adds a generated static assertion for FP4 KV batch-prefill modules:
    `DTypeKV` must be `__nv_fp4x2_e2m1`.

Expected effect: the failing Gemma 3 paged-prefill module should build under a fresh
`dtype_kv_fp4x2_e2m1` namespace rather than reusing any stale `dtype_kv_u8` module.

Top-level source-overlay correction:

- `scripts/flashinfer_source_sitecustomize.py` now monkeypatches the installed
  FlashInfer `get_batch_prefill_uri` binding used by `flashinfer.prefill`, so the live
  container uses a `dtype_kv_fp4x2_e2m1` batch-prefill namespace without importing the
  whole `/flashinfer-src` Python package.
- Whole-source Python import was tested and is not viable in this container because the
  source package's CuteDSL/CUTLASS Python dependency path raises
  `AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'OperandMajorMode'`.
  Therefore the live overlay stays narrow: installed FlashInfer Python ABI plus patched
  JIT env paths, patched attention utilities, and patched batch-prefill URI binding.

## Local Verification

Passed on the Windows checkout:

```text
py -3 -m py_compile third_party\flashinfer\flashinfer\jit\utils.py third_party\flashinfer\flashinfer\jit\attention\modules.py
py -3 -m py_compile scripts\flashinfer_source_sitecustomize.py
git -C third_party\flashinfer diff --check
```

Limit: importing FlashInfer URI helpers on Windows failed because `tvm_ffi` is not present
in this local environment. The live proof must run inside the GB10 source-overlay container.

## Required Live Rerun

Rerun the existing Gemma 3 wrapper-boundary packet with:

- vLLM fork at the current `third_party/vllm` pointer;
- FlashInfer fork at `jethac/flashinfer@3db181f4`;
- FlashInfer JIT cache cleared before server start;
- `FLASHINFER_JIT_VERBOSE=1` enabled if practical;
- the same wrapper-boundary / active-page trace env used in
  `results/vllm_gemma3_27b_wrapper_trace_20260609T0148JST_summary.md`.

Green outcomes:

1. server log or JIT path proves the batch-prefill module name contains
   `dtype_kv_fp4x2_e2m1`;
2. generated config contains the FP4 static assertion;
3. Gemma 3 first-token output becomes quality-correct against the fp8 comparator, or at
   least the wrapper output becomes signed/small instead of byte-like.

Red outcomes:

1. the module name is still `dtype_kv_u8` or an AOT path bypasses the new JIT namespace;
2. the wrapper output remains byte-like even with a fresh `fp4x2_e2m1` module;
3. the static assertion fails, proving the module is still being generated with raw `uint8_t`.

If red outcome 2 happens, the next fix moves into `compute_sfm_v()` /
`vec_cast<nv_bfloat16, __nv_fp4x2_e2m1>` rather than vLLM page pairing.
