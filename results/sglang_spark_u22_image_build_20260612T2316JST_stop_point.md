# SGLang Spark U22 Image Build Stop Point

Date: 2026-06-12 23:16 JST

Branch: `epoch2`

Current head: `b10e59a` (`ci: install sgl-kernel NUMA build headers`)

## Status

The Spark-deployable SGLang source-stack image is still not green, but the
build has advanced past two earlier blockers:

- `tzdata` no longer prompts during apt configuration.
- `sglang-kernel` metadata generation now finds `pathspec` and
  `pyproject-metadata`.

The latest completed corrected run was:

- Run: `27420189293`
- Commit: `1c59d9a`
- Runner: `ubicloud-standard-30-arm-ubuntu-2204`
- Base image: `nvidia/cuda:13.0.2-devel-ubuntu22.04`
- Result: red in `Build and push image`

The failure was in `sglang-kernel` CMake configuration:

```text
Could NOT find NUMA (missing: NUMA_INCLUDE_DIRS NUMA_LIBRARIES)
```

The same CMake pass also reported missing IBVerbs. Commit `b10e59a` adds
`libnuma-dev` and `libibverbs-dev` to the generated Dockerfile apt install
list while keeping the Ubuntu 22 arm runner, 20-way build parallelism, and
Ubuntu 22 CUDA image base.

## Current Live Run

Run `27421123640` was triggered by `b10e59a` and is currently queued with no
runner steps:

`https://github.com/jethac/dgx-spark-hijinks/actions/runs/27421123640`

## Next Step

When `27421123640` assigns:

1. Confirm it runs on the Ubuntu 22 arm runner and passes the base
   OS/GLIBC probe.
2. Watch the `sglang-kernel` build past the NUMA/IBVerbs CMake point.
3. If it fails, pull `gh run view 27421123640 --log-failed` and patch the next
   concrete error.
4. If it succeeds, collect the pushed image digest and proceed to the Spark
   smoke gate under `scripts/run_sglang_spark_image_smoke.sh`.

No Spark GPU smoke has been run for this image yet.
