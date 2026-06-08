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
- The live container's installed FlashInfer is older than the fork and does not expose
  `dtype_map_kv`, so the monkeypatch treats `torch.uint8` explicitly as the packed FP4 KV
  carrier for this experiment.
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

Remote container URI smoke passed after the overlay correction:

```text
flashinfer_file=/usr/local/lib/python3.12/dist-packages/flashinfer/__init__.py
prefill_uri=batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_fp4x2_e2m1_...
has_fp4_name=true
has_u8_name=false
```

## Live Rerun Result

Live rerun:

```text
results/vllm_gemma3_27b_jituri_20260609T0319JST_summary.md
```

Outcome: red, and specifically red outcome 2 from the original plan.

The accepted rerun reached readiness after disabling FlashInfer sampling with
`VLLM_USE_FLASHINFER_SAMPLER=0` to avoid a source-overlay side-path sampling failure.
FlashInfer attention and NVFP4-KV stayed selected. The import probe and server log prove
the paged-prefill module used the explicit FP4 namespace:

```text
prefill_uri_has_fp4x2=true
prefill_uri_has_u8=false
cached_ops/.../batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_fp4x2_e2m1_...
```

Quality did not recover. The first-token probe still produced unrelated tokens
(`Stephanie`, `ilacion`, `Kiara`) for the three simple prompts, and the tensor trace still
shows byte-like BF16 attention output:

```text
flashinfer_wrapper_prefill_post layer 0: max=255.00 mean=130.06
flashinfer_wrapper_prefill_post layer 5: max=255.00 mean=127.73
flashinfer_wrapper_prefill_post layer 0: max=255.00 mean=126.61
flashinfer_wrapper_prefill_post layer 5: max=255.00 mean=128.94
flashinfer_wrapper_prefill_post layer 0: max=255.00 mean=123.72
flashinfer_wrapper_prefill_post layer 5: max=255.00 mean=129.28
```

Conclusion: stale `dtype_kv_u8` JIT naming is falsified as the root cause for the Gemma 3
NVFP4-KV failure. The next fix moves into FlashInfer's paged-prefill FP4-KV read/convert
path, especially the conversion from packed `__nv_fp4x2_e2m1` plus FP8 scale factors to
BF16 attention values.
