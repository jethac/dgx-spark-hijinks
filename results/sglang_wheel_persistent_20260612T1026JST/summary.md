# SGLang Persistent Runner Wheel Build

Date: 2026-06-12 JST

Status: GREEN

Scope: CPU/package build receipt only. No model weights, no CUDA runtime load,
no serving, and no quality claim.

## Run

- Repo: `jethac/sglang`
- Branch: `spark/hijinks-024-diffusiongemma-upstream-rebase`
- Commit: `3c381eaa6a77783348998108de2b692a6d5d2811`
- Workflow: `hijinks-sglang-wheel-build.yml`
- Run: `27388425406`
- URL: `https://github.com/jethac/sglang/actions/runs/27388425406`
- Runner: persistent Ubicloud SGLang runner, machine `vmzrp4jj`

## Artifact

Downloaded artifact:

```text
results/sglang_wheel_persistent_20260612T1026JST/artifacts/sglang-wheel-3c381eaa6a77783348998108de2b692a6d5d2811/sglang-0.0.0.dev0+g3c381eaa6a-cp312-cp312-linux_x86_64.whl
```

The job log records:

```text
Successfully built sglang-0.0.0.dev0+g3c381eaa6a-cp312-cp312-linux_x86_64.whl
-rw-r--r-- 1 actions actions 12M Jun 12 01:26 dist/sglang-0.0.0.dev0+g3c381eaa6a-cp312-cp312-linux_x86_64.whl
```

## Persistent Cache Setup

The workflow avoids local CPU and uses the persistent Ubicloud runner:

- `CCACHE_DIR=/opt/build-cache/ccache`
- `CARGO_HOME=/home/actions/.cache/hijinks-build/cargo`
- `RUSTUP_HOME=/home/actions/.cache/hijinks-build/rustup`
- `PIP_CACHE_DIR=/home/actions/.cache/hijinks-build/pip`
- user-local cached protoc at `/home/actions/.cache/hijinks-build/protoc`

The first successful run completed in under one minute after the cache-path
plumbing was fixed. Future SGLang package builds should use this workflow
instead of local CPU.

