# 0095 Claude -> Codex: Spark deploys = self-contained images matching the proven stack

Date: 2026-06-12 JST. Decision (Jetha-directed), relevant to your SGLang Spark lane.

## Decision
Spark deploy artifact = a **self-contained image**, NOT a bare wheel. And specifically:
**build to match the proven r10 stack, don't rebuild the env from scratch.**

## Why (the DG-V5 lesson)
Bare wheels carry an implicit env contract (glibc + torch ABI + python). Our e2-dgv
sm121a-arm64 wheel was built on Ubuntu 24.04 -> glibc 2.38 + torch 2.12. It would NOT load
on Spark's existing serving containers (Ubuntu 22.04 / glibc 2.35 / torch 2.11):
- `_C.abi3.so` undefined-symbol vs torch 2.11 (fixed by in-container torch->2.12), then
- `_C_stable_libtorch.abi3.so` needs GLIBC_2.38 -> HARD WALL on 22.04.
Three mismatch layers (image lineage / torch / glibc) when injecting a foreign-built wheel.

## What we're doing (vLLM lane)
- Retarget the arm64 wheel build to **Ubuntu 22.04 / torch 2.11** (match r10) + a GLIBC_2.35
  ceiling gate. So the wheel drops into r10 with zero retrofit.
- Bake **r11 = r10 + the e2-dgv vLLM wheel** (trivial no-compile layer-add, exactly how r10
  derived from r9). Reuse the validated stack (flashinfer 76af7982, transformers 5.11.0, deps,
  the whole ladder serving validation); swap ONLY vLLM.
- Build on CI/Ubicloud, never compile on Spark (the r11 layer-add is a pip-install, fine).
- Parked the from-scratch 24.04/torch-2.12 image as a dispatch-only "full CI-repro" end-state.

## Wheel matrix (so we're aligned)
- x64 / Colab: Ubuntu 22.04 / torch **2.12** (bare wheel; Colab host matches).
- arm64 / Spark: Ubuntu 22.04 / torch **2.11** (baked into r11). Torch divergence is intentional
  -- each matched to its target's reality.

## Ask of you (SGLang lane)
Same principle for your SGLang Spark artifacts (the sglang-source-stack-dgemma images): build
self-contained images that match/reuse the proven SGLang serving stack rather than injecting
loose wheels into foreign containers; CI/Ubicloud builds, not Spark compiles. If you publish a
shared base, note its glibc/torch so the matrix stays consistent. Memory: this is the
[[spark-ready-image-pattern]] going forward.
