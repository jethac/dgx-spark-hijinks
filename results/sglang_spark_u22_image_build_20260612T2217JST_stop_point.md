# SGLang Spark U22 Image Build Stop Point

Date: 2026-06-12 22:17 JST

Branch: `epoch2`

Current head: `eb773da` (`ci: install explicit sgl-kernel build helpers`)

## Status

The SGLang Spark source-stack image build is not green yet, but the earlier
GitHub hang is no longer the active blocker.

- Run `27413170106` was cancelled after hanging in interactive `tzdata`
  configuration.
- Commit `525acac` made apt noninteractive with `DEBIAN_FRONTEND=noninteractive`
  and `TZ=Etc/UTC`.
- Run `27416276112` stayed queued on
  `ubicloud-standard-30-arm-ubuntu-2204` and was cancelled.
- Commit `652bd89` moved the Dockerized build to
  `ubicloud-standard-30-arm`. The shipped ABI remains controlled by
  `nvidia/cuda:13.0.2-devel-ubuntu22.04` plus the in-image GLIBC audit.
- Run `27417039886` assigned on the generic arm runner and passed checkout,
  Docker setup, base Ubuntu 22.04 / GLIBC 2.35 probe, and the apt/tzdata
  section. It failed later during `sgl-kernel` metadata generation:
  `ModuleNotFoundError: No module named 'pathspec'`.
- Commit `eb773da` added explicit `packaging`, `pathspec`, and
  `pyproject-metadata` installs alongside `scikit-build-core` because the
  build-tool install intentionally uses `--no-deps`.

## Current Live Run

Run `27417722301` was triggered by `eb773da` and is currently queued with no
runner steps:

`https://github.com/jethac/dgx-spark-hijinks/actions/runs/27417722301`

GitHub reports only the persistent x64 runner registered for this repo:
`ubicloud-persistent-hijinks-x64`. No persistent arm64 runner is registered, so
the native arm64 image build depends on Ubicloud hosted arm runner capacity.

## Next Step

When run `27417722301` assigns, watch it through the previous failure point:

1. `Build and push image` should pass the `sgl-kernel` metadata phase without
   `pathspec`/`pyproject-metadata` import errors.
2. If it fails, inspect `gh run view 27417722301 --log-failed` and patch the
   next concrete error.
3. If it passes, collect the pushed digest and only then run the Spark smoke
   with `scripts/run_sglang_spark_image_smoke.sh` under the marker/memory
   guardrails.

No Spark GPU work has been started for this image.
