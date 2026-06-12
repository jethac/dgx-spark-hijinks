# Codex Goal: SGLang Spark Packaging

Date: 2026-06-12 JST. Branch: `epoch2`.

You are the SGLang packaging owner for epoch2. Produce a Spark-deployable,
self-contained SGLang Gemma/DiffusionGemma runtime artifact that follows the
validated vLLM Spark packaging invariant:

- `linux/arm64`
- Ubuntu 22.04 / `glibc <= 2.35`
- torch 2.11 + CUDA 13
- built off Spark via Ubicloud/GitHub
- no loose-wheel injection into mismatched containers

Coordinate only through `mail/`. Check `mail/` at every stop point.

## Context

Claude's vLLM lane proved the target packaging contract with the
`sm121a-arm64-wheels-3d6a0d507` wheel: Ubuntu 22.04, torch 2.11, glibc 2.35
ceiling, arm64 Spark target. SGLang should mirror that discipline.

The current SGLang source-stack base is not acceptable for Spark deployment:
`nvcr.io/nvidia/sglang:26.05-py3` is Ubuntu 24.04 / glibc 2.39. See
`results/sglang_spark_base_probe_20260612TmanualJST/summary.md`.

## Objectives

1. Read Claude's vLLM arm64 wheel/r11 packaging notes and identify the reusable
   build contract for SGLang.
2. Design the SGLang artifact path:
   - preferred: self-contained image from a 22.04/torch-2.11-compatible base;
   - acceptable if cleaner: SGLang arm64 wheel(s) built under the same
     22.04/torch-2.11/glibc-2.35 contract, then baked into a proven 22.04
     SGLang runtime image.
3. Implement the CI/Ubicloud build path. Do not compile on Spark.
4. Add hard gates for:
   - container OS `VERSION_ID=22.04`;
   - compiled extension `GLIBC <= 2.35`;
   - `linux/arm64` manifest;
   - torch 2.11 + CUDA 13 runtime import;
   - SGLang, sgl-kernel, and FlashInfer import/provenance;
   - no stale FlashInfer JIT/AOT cache payload.
5. Build and publish the artifact with clear tag/digest provenance.
6. Update the SGLang AR/DiffusionGemma runner defaults to use the new artifact
   only after the gates pass.
7. Run the minimal Spark smoke only after the image is published and Claude's
   marker is absent:
   - no build on Spark;
   - single container/server;
   - memory guardrails;
   - record image digest, OS/glibc/torch/provenance, and whether the server
     reaches readiness.
8. Document every stop point in `results/` and `mail/`, and keep the worktree
   clean.

## Non-Goals

- Do not use `nvcr.io/nvidia/sglang:26.05-py3` as a deploy base except for
  temporary diagnostics.
- Do not resume AR ladder quality/capacity rows until the Spark-deployable
  SGLang artifact is green.
- Do not file upstream issues yet; keep evidence banked.

## Short /goal Text

Use this shorter command in chat:

```text
/goal Execute docs/CODEX_GOAL_SGLANG_SPARK_PACKAGING.md on branch epoch2. Build a Spark-deployable SGLang runtime artifact under the Ubuntu 22.04/glibc<=2.35/torch-2.11/linux-arm64 invariant, using Ubicloud/GitHub only for builds, then gate and document it before resuming SGLang AR/DiffusionGemma rows. Coordinate via mail/ at every stop point.
```

