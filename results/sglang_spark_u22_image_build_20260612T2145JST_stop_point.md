# SGLang Spark Ubuntu 22 Image Build Stop Point 2

Date: 2026-06-12 JST. Branch: `epoch2`.

## Scope

Continuation of `docs/CODEX_GOAL_SGLANG_SPARK_PACKAGING.md`.

## What Happened

Run `27413170106` was canceled after it hung inside Docker build layer `#13`
while configuring `tzdata`:

```text
Configuring tzdata
Please select the geographic area in which you live.
```

The build had already passed the packaging base gates before that point:

- checkout;
- runner inventory;
- Docker availability;
- base image OS/glibc probe;
- Buildx setup;
- compact Docker context preparation;
- Dockerfile generation;
- metadata;
- GHCR login.

This was an interactive apt configuration bug in the generated Dockerfile, not
a SGLang/FlashInfer build failure.

## Fix Landed

Commit `525acac` (`ci: make SGLang image apt install noninteractive`) does two
things:

1. Adds Dockerfile `ENV DEBIAN_FRONTEND=noninteractive` and `TZ=Etc/UTC`, sets
   `/etc/localtime` and `/etc/timezone`, and includes `tzdata` in the
   noninteractive apt layer.
2. Adds `scripts/run_sglang_spark_image_smoke.sh`, a Spark-only no-build smoke
   harness for the eventual published self-contained image. The harness pulls
   the image, records manifest/inspect/OS/glibc/import/cache provenance, starts
   one guarded SGLang server, and records readiness plus a small chat request.

Validation before push:

- `git diff --check`: pass
- workflow YAML parse via Ruby: pass
- `bash -n scripts/run_sglang_spark_image_smoke.sh`: pass

## Current CI

Retry run:
`https://github.com/jethac/dgx-spark-hijinks/actions/runs/27416276112`

State at stop point:

- status: `queued`
- job: `build SGLang Gemma4 source-stack image`
- runner not assigned yet (`steps=[]`)
- head: `525acacbdf2684bdda8743329f9299464a9dbd8a`

## Current State

- Worktree clean at `525acac`.
- No Spark build or model load was run.
- Do not update SGLang runner defaults or resume Spark AR/DiffusionGemma rows
  until the queued image workflow publishes a gated image and the Spark smoke
  harness passes.
