# SGLang Qwen FP4-KV Reuse Matrix Attempt, 2026-06-09

Status: blocked before runtime rows.

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
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- KV dtype: `fp4_e2m1`
- Attention backend: `flashinfer`
- Graphs: disabled
- Page size: `1`
- Install mode: `python3 -m pip install --no-deps -e /work/third_party/sglang/python`

The final runner builds a throwaway derivative image with source-build tools only:

- Rust via rustup stable: `rustc 1.96.0`
- `protobuf-compiler` for `protoc`

No Python package downgrade was used.

## Setup Blockers Cleared

Earlier attempts did not reach SGLang runtime and should not be interpreted as model
results:

- Mounted submodule needed `git config --global --add safe.directory /work/third_party/sglang`.
- The base image had no Rust compiler for the current SGLang Rust extension.
- Ubuntu `rustc/cargo 1.75` was too old for edition 2024; rustup stable fixed this.
- The Rust build then needed `protoc`; `protobuf-compiler` fixed this.

## Result

The editable SGLang overlay builds and installs successfully in all four rows:

```text
Successfully installed sglang-0.5.12.post2.dev1018+gd4fe78078
```

The server then exits before health because the NGC 26.05 image carries an older
FlashInfer package than this SGLang source requires:

```text
Exception: flashinfer_python is installed with version 0.6.10+cf494fca.nv26.5.cu132.50619265, which is less than the minimum required version 0.6.12.
```

Therefore the four rows did not produce request JSON and cannot be interpreted as FP4-KV
quality evidence. The compact machine-readable summary is
`results/sglang_qwen_fp4kv_matrix_20260609tmatrix6jst_summary.json`.

## Stop Point

The next run must pair this SGLang source overlay with a FlashInfer `>=0.6.12` build that
preserves the GB10 FP4-KV patches and selected kernel evidence. Do not treat the older
NGC FlashInfer package as acceptable, and do not downgrade SGLang to satisfy the container.
