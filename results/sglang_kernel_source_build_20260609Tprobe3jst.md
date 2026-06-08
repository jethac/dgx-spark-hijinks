# SGLang Kernel Source Build Probe, 2026-06-09

Status: source-build import blocker cleared.

## Goal

Test whether `sglang-kernel 0.4.3` can be built from the vendored SGLang source tree
against the NVIDIA 26.05 SGLang container's existing Torch/CUDA stack on GB10, without
downgrading Torch, SGLang, FlashInfer, or the container.

This directly follows the `tmatrix10` stop point, where PyPI `sglang-kernel 0.4.3`
satisfied SGLang's version guard but failed to load because its shipped
`sgl_kernel/sm100/common_ops.abi3.so` was ABI-incompatible with the container Torch.

## Environment

- Image base: `nvcr.io/nvidia/sglang:26.05-py3`
- Device: `NVIDIA GB10`
- Compute capability: `12.1`
- Torch: `2.12.0a0+5aff3928d8.nv26.05`
- Torch CUDA: `13.2`
- FlashInfer overlay: editable `jethac/flashinfer@4c3c0d99`
- SGLang overlay: `jethac/sglang@d96869237`

## Source Patch

`jethac/sglang@d96869237` adds selectable `sgl-kernel` CMake build switches:

- `SGL_KERNEL_BUILD_SM90`
- `SGL_KERNEL_BUILD_SM100`
- `SGL_KERNEL_ENABLE_SPATIAL`
- `SGL_KERNEL_ENABLE_FLASHMLA`

Defaults preserve upstream behavior. The probe opts out of unrelated targets and builds
only the precise-math `sm100` package path that `sgl_kernel.load_utils` chooses on GB10:

```text
-DSGL_KERNEL_COMPILE_THREADS=1
-DENABLE_BELOW_SM90=OFF
-DCMAKE_POLICY_VERSION_MINIMUM=3.5
-DSGL_KERNEL_BUILD_SM90=OFF
-DSGL_KERNEL_BUILD_SM100=ON
-DSGL_KERNEL_ENABLE_FA3=OFF
-DSGL_KERNEL_ENABLE_FLASHMLA=OFF
-DSGL_KERNEL_ENABLE_SPATIAL=OFF
```

## Result

The narrow source build succeeded and installed:

```text
sglang_kernel-0.4.3-cp310-abi3-linux_aarch64.whl
```

The installed package contains only the expected GB10-loaded common ops path:

```json
{
  "common_ops_sm100": [
    "/usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.abi3.so"
  ],
  "common_ops_sm90": [],
  "loaded_common_ops": "/usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.abi3.so"
}
```

The import probe passed:

```text
import sgl_kernel
sgl_kernel.common_ops.__file__ == /usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.abi3.so
```

The build log also shows `compute_121a`/`sm_121a` compilation for the precise-math module,
with ptxas advisory warnings but no hard failure.

## Interpretation

The active blocker moved again:

- Cleared: stale NGC FlashInfer `0.6.10` versus SGLang `>=0.6.12`.
- Cleared: PyPI `sglang-kernel 0.4.3` ABI mismatch.
- Still open: rerun the Qwen FP4-KV cached-prefix matrix using a source-built SGLang stack.

The next matrix should prepare a source-stack image once, then run all four rows from that
image. Rebuilding `sglang-kernel` inside every row is technically possible but too slow for
the matrix loop.
