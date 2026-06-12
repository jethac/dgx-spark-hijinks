# SGLang Spark Ubuntu 22 Image Build Stop Point

Date: 2026-06-12 JST. Branch: `epoch2`.

## Scope

Packaging work for `docs/CODEX_GOAL_SGLANG_SPARK_PACKAGING.md`: build a
Spark-deployable SGLang source-stack image under the vLLM r11 ABI invariant
(`linux/arm64`, Ubuntu 22.04 / GLIBC_2.35, Python 3.12, torch 2.11 + CUDA 13).

No Spark compile or model load was run.

## Changes Landed

- `35dee0d` (`ci: build SGLang Spark image from Ubuntu 22`)
  - added `docs/SGLANG_SPARK_PACKAGING_PLAN_20260612.md`;
  - retargeted `.github/workflows/hijinks-sglang-gemma4-source-stack-image.yml`
    from `nvcr.io/nvidia/sglang:26.05-py3` to
    `nvidia/cuda:13.0.2-devel-ubuntu22.04`;
  - added Python 3.12 + torch 2.11/cu130 bootstrap in the image;
  - installs SGLang runtime deps from the checked-out SGLang pyproject, then
    replaces FlashInfer and sgl-kernel from the campaign source forks;
  - keeps OS, GLIBC, import/provenance, manifest, and FlashInfer cache gates.
- `62b4120` (`ci: avoid pipefail false red in base probe`)
  - fixed `ldd --version | head` false-red under `set -o pipefail`.

## CI Runs

### `27412907255` - RED, false probe failure

URL: `https://github.com/jethac/dgx-spark-hijinks/actions/runs/27412907255`

The base image probe proved the selected base has:

- arm64 manifest for `nvidia/cuda:13.0.2-devel-ubuntu22.04`;
- Ubuntu 22.04.5;
- `ldd (Ubuntu GLIBC 2.35-0ubuntu3.11) 2.35`.

The run failed with exit `141` because `ldd --version | head -n 1` tripped
`pipefail` after `head` closed the pipe. This was a workflow bug, not a base
compatibility failure.

### `27413170106` - IN PROGRESS

URL: `https://github.com/jethac/dgx-spark-hijinks/actions/runs/27413170106`

Status at local watch stop (~50 minutes after trigger): `Build and push image`
still active. Completed stages:

- checkout;
- runner inventory;
- Docker availability;
- base image OS/glibc probe;
- Buildx setup;
- compact Docker context preparation;
- Dockerfile generation;
- image metadata;
- GHCR login.

The long stage is the first cold Ubuntu 22.04 SGLang source-stack image build.
No failure log was available while the job was still running.

## Current State

- Worktree clean at `62b4120`.
- Active cloud build only; no local CPU build and no Spark build.
- Do not update SGLang runner defaults or resume Spark AR/DiffusionGemma rows
  until `27413170106` either publishes a gated image or fails with a concrete
  log to patch.
