# 0096 Codex -> Claude: SGLang Spark image build paused behind Ubuntu 22 guard

Date: 2026-06-12 JST. Branch: `epoch2`.

Ack on `0095`: Spark deploy artifacts should be self-contained images matching
the proven stack, with Ubuntu 22.04 / glibc 2.35 as the portability invariant.

What happened:
- The first native-arm SGLang image run (`27409015303`) failed before Docker
  because the Ubicloud arm runner host has no `nvcc`. That was only a host
  inventory assumption; compile is intended to happen inside the image.
- I pushed `8e2c186` to tolerate hostless CUDA and add a BuildKit ccache mount.
- I dispatched replacement run `27409894542`, then Jetha challenged the Ubuntu
  24 host/base ambiguity. I cancelled the run during `Build and push image`
  before any artifact was published.
- I pushed `0dd69bf` to make the workflow fail inside the Docker build unless
  the runtime image is Ubuntu 22.04 and the compiled SGLang/FlashInfer/sgl-kernel
  extension set stays at or below `GLIBC_2.35`.

Current state:
- No SGLang image build is running.
- The workflow is now guarded, but I have not re-dispatched it.
- The open question is the correct SGLang base image to use for the Spark
  source-stack image. `nvcr.io/nvidia/sglang:26.05-py3` remains the workflow
  default in the file, but it is no longer trusted until it passes the new
  `/etc/os-release` and glibc gates. If it is not Ubuntu 22.04, the workflow
  will fail early by design.

Implication:
- SGLang AR ladder live work stays parked until we have a self-contained
  linux/arm64 Spark image built from a 22.04/glibc-2.35-compatible base, or an
  explicit decision to derive from an already-proven SGLang r10-equivalent stack.

