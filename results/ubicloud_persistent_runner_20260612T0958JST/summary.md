# Ubicloud Persistent Runner Setup

Date: 2026-06-12 JST

## VM

- Provider: Ubicloud Compute VM
- Location: `eu-central-h1` (Germany)
- Name: `hijinks-build-x64`
- VM id: `vmzrp4jjpk9j7gfsskvyv09wyn`
- IPv4: `148.251.32.85`
- SSH user: `jethac`
- Shape: `standard-16`
- Disk: 320 GiB
- Reason for `standard-16`: account quota rejected `standard-60` with a
  16-vCPU maximum.

## Toolchain

- OS: Ubuntu 24.04 (`ubuntu-noble`)
- CPU: 16 vCPU
- RAM: about 62 GiB
- CUDA toolkit: `13.0`, `nvcc V13.0.88`
- ccache: `/opt/build-cache/ccache`, max size `100G`

## Registered GitHub Runners

All runners are repo-level self-hosted runners on the same VM.

| repo | runner | status at setup | labels |
|---|---|---|---|
| `jethac/vllm` | `ubicloud-persistent-vllm-x64` | online, idle | `self-hosted`, `Linux`, `X64`, `ubicloud-persistent-build-x64`, `cuda-toolkit-13`, `ccache`, `vllm-wheel` |
| `jethac/sglang` | `ubicloud-persistent-sglang-x64` | online, idle | `self-hosted`, `Linux`, `X64`, `ubicloud-persistent-build-x64`, `cuda-toolkit-13`, `ccache`, `sglang-wheel` |
| `jethac/dgx-spark-hijinks` | `ubicloud-persistent-hijinks-x64` | online, idle | `self-hosted`, `Linux`, `X64`, `ubicloud-persistent-build-x64`, `cuda-toolkit-13`, `ccache`, `hijinks` |

## Notes

- This VM is CPU-only. It is for builds, wheel assembly, container assembly,
  `cuobjdump`, and static provenance checks.
- Runtime probes, JIT load checks, latch diagnostics, and serving validation
  still require the Spark or another Blackwell GPU.
- Service-level systemd overrides set `CUDA_HOME`, CUDA `PATH`, `CCACHE_DIR`,
  and `CCACHE_MAXSIZE`; workflows should still set the same env values so the
  job log proves the intended environment.
- Hijinks persistent-runner smoke workflow:
  `.github/workflows/ubicloud-persistent-smoke.yml`.

## GitHub Actions Smoke

- Workflow: `.github/workflows/ubicloud-persistent-smoke.yml`
- Run: `27387525667`
- URL: `https://github.com/jethac/dgx-spark-hijinks/actions/runs/27387525667`
- Result: `success`
- Head: `d08330810f78cbf5e1bd0f47966e46a80664a232`
- Log: `persistent_smoke_27387525667.log`

The smoke ran on the `ubicloud-persistent-hijinks-x64` self-hosted runner and
records runner inventory, `nvcc --version`, `cuobjdump --version`, ccache config,
and disk/memory state.
