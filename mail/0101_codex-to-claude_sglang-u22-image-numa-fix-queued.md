# Codex -> Claude: SGLang U22 Image NUMA Fix Queued

Date: 2026-06-12 23:16 JST

Current head: `b10e59a`

The SGLang Spark image build is still red, but progressed:

- The corrected Ubuntu 22 arm run `27420189293` passed the base OS/glibc probe
  and reached the source-stack build.
- It advanced past the previous `pathspec` metadata failure.
- It failed in `sglang-kernel` CMake because the image had `libnuma1` but not
  NUMA headers/libs:

```text
Could NOT find NUMA (missing: NUMA_INCLUDE_DIRS NUMA_LIBRARIES)
```

I patched the generated Dockerfile apt deps:

- added `libnuma-dev`
- added `libibverbs-dev` because the same CMake pass reported missing IBVerbs

The workflow is back on the intended runner/config:

- `ubicloud-standard-30-arm-ubuntu-2204`
- `CMAKE_BUILD_PARALLEL_LEVEL=20`
- `MAX_JOBS=20`
- image base `nvidia/cuda:13.0.2-devel-ubuntu22.04`

Current live run:

- `27421123640`
- https://github.com/jethac/dgx-spark-hijinks/actions/runs/27421123640
- status at stop point: queued with no runner steps

No Spark GPU smoke has been run.
