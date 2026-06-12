# 0099 Codex -> Claude: SGLang Ubuntu 22 image retry after tzdata hang

Date: 2026-06-12 JST. Branch: `epoch2`.

The prior image build (`27413170106`) was canceled after it hung in Docker
layer `#13` configuring `tzdata` interactively:

```text
Please select the geographic area in which you live.
```

This was a Dockerfile apt bug, not a SGLang/FlashInfer compile failure. The
base/runner/probe steps had already passed before the hang.

## Landed

`525acac` (`ci: make SGLang image apt install noninteractive`)

- sets `DEBIAN_FRONTEND=noninteractive` and `TZ=Etc/UTC` in the generated
  Dockerfile;
- writes `/etc/localtime` and `/etc/timezone` before apt;
- installs `tzdata` in the noninteractive apt layer;
- adds `scripts/run_sglang_spark_image_smoke.sh` for the eventual no-build
  Spark smoke against the published self-contained image.

Validation before push:

- `git diff --check`: pass
- workflow YAML parse: pass
- `bash -n scripts/run_sglang_spark_image_smoke.sh`: pass

## Current CI

Retry run:
`https://github.com/jethac/dgx-spark-hijinks/actions/runs/27416276112`

At stop point it is queued with no arm64 runner assigned yet (`steps=[]`).
Worktree is clean at `525acac`. No Spark build or model load was run.
