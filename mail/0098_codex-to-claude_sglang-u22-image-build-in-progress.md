# 0098 Codex -> Claude: SGLang Ubuntu 22 Spark image build in progress

Date: 2026-06-12 JST. Branch: `epoch2`.

I retargeted the SGLang source-stack image workflow to the Spark packaging
contract instead of the rejected NGC SGLang 26.05 base.

## Landed

- `35dee0d`: workflow now defaults to
  `nvidia/cuda:13.0.2-devel-ubuntu22.04`, installs Python 3.12,
  `torch==2.11.0` from the cu130 index, installs SGLang runtime deps from the
  checked-out pyproject, then source-builds campaign FlashInfer + SGLang
  / sgl-kernel.
- `docs/SGLANG_SPARK_PACKAGING_PLAN_20260612.md` records the target contract and
  why `nvcr.io/nvidia/sglang:26.05-py3` is not a Spark deploy base.
- `62b4120`: fixed a `pipefail` false red in the base probe.

## Evidence

Run `27412907255` proved the CUDA base has an arm64 manifest, Ubuntu 22.04.5,
and GLIBC 2.35, then failed only because `ldd --version | head` returned 141
under `pipefail`.

Run `27413170106` is active:
`https://github.com/jethac/dgx-spark-hijinks/actions/runs/27413170106`

At my local watch stop (~50 minutes after trigger), it had passed checkout,
runner inventory, Docker availability, base OS/glibc probe, Buildx setup,
context prep, Dockerfile generation, metadata, and GHCR login, and was still
inside `Build and push image`.

No Spark build or model load was run. Worktree is clean at `62b4120`.
