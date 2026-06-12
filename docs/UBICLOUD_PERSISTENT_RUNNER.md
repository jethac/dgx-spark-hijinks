# Ubicloud Persistent Build Runner

Purpose: keep CPU-heavy vLLM/SGLang/FlashInfer builds off the Spark and off
short-lived hosted runners. The persistent VM is a CPU build box: it needs
Ubuntu, many vCPUs, disk, CUDA toolkit/nvcc, ccache, and Docker if an image
build needs it. It does not need a GPU.

Runtime validation still belongs on a Blackwell GPU. `cuobjdump`, wheel
assembly, and container assembly are valid on the VM; latch diagnostics, JIT
module loads, and serving probes are not.

## Recommended VM Shape

- x64 primary builder: `standard-60`, Ubuntu 24.04, 600 GiB disk.
- arm64 builder if available/needed: `standard-60-arm`, Ubuntu 24.04, 600 GiB
  disk.
- If quota blocks 60 vCPU, use `standard-30` / `standard-30-arm`; we already
  proved the 30-vCPU hosted runners green, but persistent 60 vCPU should reduce
  repeated setup time.

## Current Runner

Created 2026-06-12:

- VM: `eu-central-h1/hijinks-build-x64`
- Location: Germany (`eu-central-h1`)
- VM id: `vmzrp4jjpk9j7gfsskvyv09wyn`
- Public IPv4: `148.251.32.85`
- SSH user: `jethac`
- Shape: `standard-16`
- Disk: 320 GiB
- Reason not `standard-60`: current Ubicloud project quota is 16 vCPUs. The
  `standard-60` request failed with `maximum allowed vCPU count: 16`.
- CUDA: `13.0`, `nvcc V13.0.88`
- ccache: `/opt/build-cache/ccache`, max size `100G`
- Swap: `/swapfile`, default `64G` from bootstrap. The first cold Docker
  source-stack image build OOM-killed the runner service at `MAX_JOBS=20`; the
  image workflow now uses `MAX_JOBS=8`, with swap as a guardrail rather than as
  expected steady-state memory.
- Docker: required for image-build workflows. The first image workflow attempt
  found the runner missing Docker and non-passwordless sudo, so VM bootstrap now
  installs `docker.io` plus `docker-buildx` and adds the runner user to the
  `docker` group. Existing runners provisioned before that change must be
  re-bootstrapped before image workflows can run.

Registered repo-level GitHub runners on this VM:

| repo | runner name | labels |
|---|---|---|
| `jethac/vllm` | `ubicloud-persistent-vllm-x64` | `self-hosted`, `Linux`, `X64`, `ubicloud-persistent-build-x64`, `cuda-toolkit-13`, `ccache`, `docker`, `vllm-wheel` |
| `jethac/sglang` | `ubicloud-persistent-sglang-x64` | `self-hosted`, `Linux`, `X64`, `ubicloud-persistent-build-x64`, `cuda-toolkit-13`, `ccache`, `docker`, `sglang-wheel` |
| `jethac/dgx-spark-hijinks` | `ubicloud-persistent-hijinks-x64` | `self-hosted`, `Linux`, `X64`, `ubicloud-persistent-build-x64`, `cuda-toolkit-13`, `ccache`, `docker`, `hijinks` |

Service-level environment overrides are installed under:

```text
/etc/systemd/system/actions.runner.*.service.d/10-hijinks-build-env.conf
```

They set `CUDA_HOME=/usr/local/cuda`, prepend `/usr/local/cuda/bin` to `PATH`,
and point all runners at the shared 100 GiB ccache. Workflows should still set
the same variables explicitly so the job log proves the intended environment.

## Local Prerequisites

`ubi` is installed at `B:\workshop\tools\ubi\ubi.exe`. Future shells should see
it on the user PATH. Ubicloud requires a project personal access token in
`UBI_TOKEN`.

Do not commit or mail the token. Set it in the shell that will create the VM:

```powershell
$env:UBI_TOKEN = "<ubicloud project token>"
```

The Ubicloud CLI can then list projects/VMs:

```powershell
ubi vm list
```

## VM Creation Sketch

The exact project/location names come from the Ubicloud account. After
`UBI_TOKEN` is set, use `ubi help vm create` if any option name differs.

```powershell
$pub = Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub -Raw
ubi vm eu-central-h1/hijinks-build-x64 create `
  --size standard-60 `
  --storage-size 600 `
  --boot-image ubuntu-jammy `
  --enable-ip4 `
  $pub
```

If the CLI shape/boot-image spelling differs, use the API-equivalent fields:
`size`, `storage_size`, `boot_image`, `public_key`, and `enable_ip4`.

## Runner Registration

Register one runner per GitHub repo that needs self-hosted builds. For the vLLM
wheel lane:

```powershell
$repo = "jethac/vllm"
$runnerToken = gh api -X POST "repos/$repo/actions/runners/registration-token" --jq .token
```

Copy `scripts/bootstrap_ubicloud_build_runner.sh` to the VM and run:

```bash
sudo env \
  GITHUB_REPO=jethac/vllm \
  GITHUB_RUNNER_TOKEN='<short-lived token from gh api>' \
  RUNNER_LABELS='ubicloud-persistent-build-x64,cuda-toolkit-13,ccache,docker,vllm-wheel' \
  bash bootstrap_ubicloud_build_runner.sh
```

Repeat for `jethac/sglang` and `jethac/dgx-spark-hijinks` if those repos need
their own persistent runner. Repo-level self-hosted runners cannot be shared
across unrelated repos unless an organization-level runner is configured.

Suggested labels:

- `ubicloud-persistent-build-x64`
- `ubicloud-persistent-build-arm64`
- `cuda-toolkit-13`
- `ccache`
- `docker` if image builds will run on the VM
- repo-specific labels such as `vllm-wheel` or `sglang-wheel`

## Workflow Target

For the persistent x64 builder:

```yaml
runs-on: [self-hosted, Linux, X64, ubicloud-persistent-build-x64]
```

For the persistent arm64 builder:

```yaml
runs-on: [self-hosted, Linux, ARM64, ubicloud-persistent-build-arm64]
```

Keep the hosted Ubicloud workflows around as fallback. They are useful for
isolating whether a persistent-runner failure is environment drift.

The hijinks repo has a smoke workflow for the current persistent runner:

```text
.github/workflows/ubicloud-persistent-smoke.yml
```

It also builds the SGLang Gemma4 source-stack container with Docker on the
same persistent runner:

```text
.github/workflows/hijinks-sglang-gemma4-source-stack-image.yml
```

First green receipt: GitHub Actions run `27405349785`, published
`ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack:epoch2-gemma4-tf511-12fca91`
at digest `sha256:bf24438b302c96e457b8a59f8a8dbaf109fab08013554be81e6957d4fb0f1a70`.
This is a CPU build/provenance gate only; Spark serving and GPU runtime checks
remain separate.

The SGLang fork has a non-publishing persistent wheel build workflow:

```text
.github/workflows/hijinks-sglang-wheel-build.yml
```

It runs on `ubicloud-persistent-sglang-x64`, builds the Python wheel in a
user-owned persistent virtualenv, caches Rust/pip/protoc under
`$HOME/.cache/hijinks-build`, and uploads the wheel artifact. The first green
receipt is
`results/sglang_wheel_persistent_20260612T1026JST/summary.md`.

The SGLang fork also has a no-weight Gemma 4 MTP static audit workflow:

```text
.github/workflows/hijinks-gemma4-mtp-static-audit.yml
```

It runs on `ubicloud-persistent-sglang-x64` and checks the assistant-model,
`FROZEN_KV_MTP`, target-KV-pool, FlashInfer target-verify, and split-dtype
source markers without downloading model weights. The first green receipt is
`results/sglang_mtp_static_audit_persistent_20260612T1041JST/summary.md`.

## Build Cache Policy

The bootstrap script sets:

- `CCACHE_DIR=/opt/build-cache/ccache`
- `CCACHE_MAXSIZE=100G`
- `CUDA_HOME=/usr/local/cuda`

For vLLM wheel builds, keep:

```yaml
env:
  MAX_JOBS: "40"
  CMAKE_BUILD_PARALLEL_LEVEL: "40"
  CCACHE_DIR: /opt/build-cache/ccache
  CCACHE_MAXSIZE: 100G
```

On `standard-30`, use `20` jobs. On `standard-60`, start with `40` jobs rather
than `60` to leave room for linker spikes and Python packaging overhead.

## Acceptance Gate

A persistent runner is usable only after a workflow records:

- runner labels and `nproc`
- CUDA toolkit version (`nvcc --version`)
- ccache location and stats
- Docker CLI, Buildx plugin, and daemon access from the runner user
  (`docker version`, `docker buildx version`, and `docker info` without sudo)
  if any image-build workflow will use the runner
- build artifact upload or release link
- no GPU runtime test claimed from this CPU-only VM
