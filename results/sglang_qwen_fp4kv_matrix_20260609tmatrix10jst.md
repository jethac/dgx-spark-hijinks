# SGLang Qwen FP4-KV Reuse Matrix Attempt, 2026-06-09

Status: blocked before runtime rows, after the FlashInfer source-overlay blocker was
cleared.

## Goal

Run the four-row no-code matrix for the SGLang Qwen FP4-KV cached-prefix bug:

- default FP4 cached prefix (`forward_extend_merge_paged`)
- `SGLANG_FLASHINFER_USE_PAGED=1`
- `SGLANG_RADIX_FORCE_MISS=1`
- both switches together

The matrix is meant to distinguish split ragged/paged merge failure from a broader reused
FP4 prefix-state failure.

## Environment

- Image base: `nvcr.io/nvidia/sglang:26.05-py3`
- Source overlay: `third_party/sglang` at `jethac/sglang@d4fe78078`
- FlashInfer overlay: `third_party/flashinfer` at `jethac/flashinfer@4c3c0d99`
- CUTLASS DSL: upgraded in-container to `nvidia-cutlass-dsl[cu13] 4.5.2`
- `sglang-kernel`: upgraded in-container to PyPI `0.4.3`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- KV dtype: `fp4_e2m1`
- Attention backend: `flashinfer`
- Graphs: disabled
- Page size: `1`

The final runner builds a throwaway derivative image with source-build tools only:

- Rust via rustup stable: `rustc 1.96.0`
- `protobuf-compiler` for `protoc`

No Python package downgrade was used.

## Setup Blockers Cleared

This run got farther than `tmatrix6`:

- Editable FlashInfer installs from the vendored source tree with `--no-deps
  --no-build-isolation`.
- Stale `flashinfer-python`, `flashinfer-cubin`, `flashinfer-jit-cache`, package
  directories, and `/root/.cache/flashinfer` are removed before the editable install.
- `flashinfer_python 0.6.13` satisfies SGLang's `>=0.6.12` guard.
- `jethac/flashinfer@4c3c0d99` aliases `cute.nvgpu.OperandMajorMode` to the CUTLASS DSL
  4.5.x location under `cute.nvgpu.tcgen05`, clearing the import failure seen after the
  CUTLASS DSL upgrade.
- Editable SGLang installs as `sglang 0.5.12.post2.dev1018+gd4fe78078`.

The version probe reached:

```text
flashinfer 0.6.13 /work/third_party/flashinfer/flashinfer/__init__.py
flashinfer_python 0.6.13
sglang_kernel 0.4.3
sglang 0.5.12.post2.dev1018+gd4fe78078
```

## Result

The server then exits before health while importing `sgl_kernel`:

```text
[sgl_kernel] CRITICAL: Could not load any common_ops library!

Attempted locations:
1. Architecture-specific pattern: /usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.* - found files: ['/usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.abi3.so']
2. Fallback pattern: /usr/local/lib/python3.12/dist-packages/sgl_kernel/common_ops.* - found files: []
3. Standard Python import: common_ops - failed

GPU Info:
- Compute capability: 121
- Expected variant: SM121 (precise math for compatibility)
- CUDA version: 13.2

Error details:
- ImportError: /usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.abi3.so: undefined symbol: _ZNK2at10TensorBase14const_data_ptrIiLi0EEEPKT_v
- ModuleNotFoundError: No module named 'common_ops'
```

The four rows did not produce request JSON and cannot be interpreted as FP4-KV quality
evidence. The compact machine-readable summary is
`results/sglang_qwen_fp4kv_matrix_20260609tmatrix10jst_summary.json`.

## Stop Point

The active blocker is now `sglang-kernel`, not FlashInfer. The PyPI `sglang-kernel 0.4.3`
wheel satisfies SGLang's version guard, but its shipped `common_ops` binary is not loadable
against the NVIDIA 26.05 Torch/CUDA stack on GB10.

Next step: build `sglang-kernel` from source against the same container Torch/CUDA stack, or
find a matching ARM64 CUDA 13.x wheel with SM121-compatible `common_ops`. Do not downgrade
SGLang, Torch, FlashInfer, or the container to make the guard pass.
