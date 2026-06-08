# llama.cpp Native FP4 Arch Probe, 2026-06-08

Status: build/emission checkpoint complete; runtime NVFP4 GGUF correctness still untested.

## Scope

This artifact covers the first concrete llama.cpp move:

- fork `ggml-org/llama.cpp` to `jethac/llama.cpp`
- add `third_party/llama.cpp` as a submodule
- pin and build a recent upstream master commit on the GB10 host
- compile `CMAKE_CUDA_ARCHITECTURES=121a`, `121`, and `120f`
- cuobjdump the CUDA backend for block-scale FP4 MMA evidence

It does not claim runtime dispatch or model correctness. That still requires an actual
NVFP4 GGUF run.

## Host And Source

- host: `thinkstationpgx-00b4`
- GPU: `NVIDIA GB10`, compute capability `12.1`
- OS: Linux `aarch64`, kernel `6.17.0-1021-nvidia`
- CUDA: `13.0`, `nvcc V13.0.88`, `cuobjdump V13.0.85`
- CMake: `3.28.3`
- build tool: GNU Make `4.3`
- fork branch: `jethac/llama.cpp@spark/native-fp4-sm121-20260608`
- pinned commit: `19bba67c1f4db723c60a0d421aa0788bf4ddc699`
- pinned commit subject: `HIP: add gfx1152 and gfx1153 to RDNA3.5 (#24129)`
- baseline: `/home/jethac/src/llama.cpp-b9536`, commit `308f61c31f083251ce8150f10b9ef97679b500b5`, tag `b9536`

Remote full artifact directory:

```text
/home/jethac/spark_tmp/llamacpp_native_fp4_arch_20260608T164917JST
```

Copied compact artifacts:

```text
results/llamacpp_native_fp4_arch_20260608T164917JST/
```

## Source Gating Read

In this pin, `BLACKWELL_MMA_AVAILABLE` is compiled for NVIDIA CUDA device arch
`__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300`.

Runtime native FP4 MMQ selection in `ggml/src/ggml-cuda/mmq.cu` requires:

```text
blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4)
```

The inline PTX for NVFP4 is:

```text
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

## Build Matrix

| requested `CMAKE_CUDA_ARCHITECTURES` | configure | build | emitted ELF target | block-scale PTX evidence |
|---|---:|---:|---|---:|
| `121a` | ok | ok | `sm_121a` cubins | `mxf4nvf4`: 2592, `mxf4`: 2592 |
| `121` | ok | ok | `sm_121a` cubins | `mxf4nvf4`: 2592, `mxf4`: 2592 |
| `120f` | fail | skipped | none | none |

Key nuance: `121` is not an independent non-`a` build in this llama.cpp pin. CMake logs:

```text
-- Replacing 121 in CMAKE_CUDA_ARCHITECTURES with 121a
-- Replacing 121-real in CMAKE_CUDA_ARCHITECTURES_NATIVE with 121a-real
-- Using CMAKE_CUDA_ARCHITECTURES=121a CMAKE_CUDA_ARCHITECTURES_NATIVE=121a-real
```

`120f` failed before CUDA compilation under CMake 3.28.3:

```text
CMAKE_CUDA_ARCHITECTURES:
  120f
is not one of the following:
  * a semicolon-separated list of integers, each optionally followed by '-real' or '-virtual'
  * a special value: all, all-major, native
```

So this exact `120f` spelling is a configure-time no-go on the GB10 host's CMake/CUDA
toolchain, not a CUDA block-scale-MMA compile failure.

## cuobjdump Notes

For both `121a` and `121`, `cuobjdump --list-elf` on `bin/libggml-cuda.so.0.13.1`
shows 138 `sm_121a` cubins.

For both `121a` and `121`, `cuobjdump --dump-ptx` includes:

```text
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

Count split:

- `mxf4nvf4` block-scale PTX: `2592`
- `mxf4` block-scale PTX: `2592`
- total `block_scale` PTX hits: `5184`

`cuobjdump --dump-sass` does not preserve the literal `mxf4nvf4` / `block_scale` text
in these artifacts; it does show `sm_121a` SASS sections and generic HMMA-style instruction
mnemonics. The PTX dump is the direct string evidence for the native block-scale FP4 path.

The b9536 baseline also contains 138 `sm_121a` cubins, but `cuobjdump --dump-ptx` for that
binary did not retain block-scale PTX strings. Treat b9536 as the known-good serving
baseline only; this result does not prove b9536 lacks native FP4 code, and it still has
not exercised an NVFP4 GGUF.

## Files

- environment: `results/llamacpp_native_fp4_arch_20260608T164917JST/00_environment.txt`
- source pin: `results/llamacpp_native_fp4_arch_20260608T164917JST/04_source_state.txt`
- `121a` summary/logs/hits:
  - `121a_summary.txt`
  - `121a_configure.log`
  - `121a_build_tail.txt`
  - `121a_cuobjdump_list_elf.txt`
  - `121a_cuobjdump_ptx_block_scale_hits.txt`
  - `121a_cuobjdump_sass_mma_head.txt`
- `121` summary/logs/hits:
  - `121_summary.txt`
  - `121_configure.log`
  - `121_build_tail.txt`
  - `121_cuobjdump_list_elf.txt`
  - `121_cuobjdump_ptx_block_scale_hits.txt`
  - `121_cuobjdump_sass_mma_head.txt`
- `120f` configure failure:
  - `120f_summary.txt`
  - `120f_configure.log`
  - `120f_configure_tail.txt`
- b9536 baseline:
  - `b9536_summary.txt`
  - `b9536_cuobjdump_list_elf.txt`
  - `b9536_cuobjdump_ptx_block_scale_hits.txt`
  - `b9536_cuobjdump_sass_mma_head.txt`

## Next Gate

Run an actual NVFP4 GGUF on the pinned `19bba67c1` build and compare output to a BF16/Q8
reference. The next claim must capture runtime dispatch evidence plus correctness and PP/TG
speed; this build-only artifact only proves that CUDA 13.0 can build and emit the native
NVFP4 block-scale PTX path for GB10 as `sm_121a`.
