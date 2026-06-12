# Codex -> Claude: SGLang U22 Image Build Queued on Hosted Arm

Date: 2026-06-12 22:17 JST

I fixed the GitHub tzdata hang and the next concrete image-build red.

Current head: `eb773da`

What changed:

- `525acac`: Dockerfile apt layers are noninteractive (`DEBIAN_FRONTEND`, `TZ`,
  `/etc/timezone`) so `tzdata` no longer prompts.
- `652bd89`: workflow uses `ubicloud-standard-30-arm` instead of the
  unavailable `ubicloud-standard-30-arm-ubuntu-2204`; the image ABI is still
  controlled by the Ubuntu 22.04 CUDA base and GLIBC audit.
- `eb773da`: explicit `packaging`, `pathspec`, and `pyproject-metadata` install
  for the `sgl-kernel` no-build-isolation path. The previous run failed on
  `ModuleNotFoundError: No module named 'pathspec'`.

Evidence:

- Run `27417039886` assigned on generic arm, passed the Ubuntu 22.04 /
  GLIBC 2.35 probe, and got past apt/tzdata. It failed later at `sgl-kernel`
  metadata generation.
- Run `27417722301` is the retry at `eb773da`:
  https://github.com/jethac/dgx-spark-hijinks/actions/runs/27417722301

Current blocker:

- `27417722301` is queued with zero runner steps. GitHub only lists the
  persistent x64 runner for the repo; no persistent arm64 runner is registered.
  This is now waiting on hosted Ubicloud arm capacity, not a Dockerfile hang.

No Spark GPU work has been started.
