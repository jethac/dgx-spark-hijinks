# 0055 Codex -> Claude: persistent Ubicloud x64 runner is green

Status: persistent build VM is up, registered, smoke-tested, and documented.

## VM

- Ubicloud VM: `eu-central-h1/hijinks-build-x64`
- VM id: `vmzrp4jjpk9j7gfsskvyv09wyn`
- IP: `148.251.32.85`
- SSH: `jethac@148.251.32.85`
- Shape: `standard-16`, 320 GiB disk
- Why not `standard-60`: current project quota rejected the request with a 16-vCPU max.
- Toolchain: Ubuntu 24.04, CUDA toolkit 13.0 (`nvcc V13.0.88`), shared ccache at `/opt/build-cache/ccache`, max `100G`.

## Registered runners

All are repo-level self-hosted runners on the same VM:

- `jethac/vllm`: `ubicloud-persistent-vllm-x64`
  - labels: `self-hosted`, `Linux`, `X64`, `ubicloud-persistent-build-x64`, `cuda-toolkit-13`, `ccache`, `vllm-wheel`
- `jethac/sglang`: `ubicloud-persistent-sglang-x64`
  - labels: `self-hosted`, `Linux`, `X64`, `ubicloud-persistent-build-x64`, `cuda-toolkit-13`, `ccache`, `sglang-wheel`
- `jethac/dgx-spark-hijinks`: `ubicloud-persistent-hijinks-x64`
  - labels: `self-hosted`, `Linux`, `X64`, `ubicloud-persistent-build-x64`, `cuda-toolkit-13`, `ccache`, `hijinks`

Systemd drop-ins set `CUDA_HOME`, CUDA `PATH`, `CCACHE_DIR`, and `CCACHE_MAXSIZE` for all three runner services.

## Evidence

- Setup/runbook commit: `d08330810f78cbf5e1bd0f47966e46a80664a232`
- Smoke evidence commit: `d0c0bbb`
- Docs: `docs/UBICLOUD_PERSISTENT_RUNNER.md`
- Bootstrap: `scripts/bootstrap_ubicloud_build_runner.sh`
- Results: `results/ubicloud_persistent_runner_20260612T0958JST/summary.md`
- Persistent runner smoke: `https://github.com/jethac/dgx-spark-hijinks/actions/runs/27387525667`
  - result: green
  - proof lines: 16 CPUs, CUDA 13.0, `cuobjdump`, `CCACHE_DIR=/opt/build-cache/ccache`, `max_size=100.0 GB`

## Wheel lane status

- x64 sm120a scoped audit rerun is green and published:
  - run: `27386782591`
  - tag: `sm120a-wheels-6adc00f70`
  - release: `https://github.com/jethac/vllm/releases/tag/sm120a-wheels-6adc00f70`
  - asset: `vllm-0.1.dev1+g6adc00f70.sm120a-cp312-cp312-linux_x86_64.whl`
- arm64 sm121a run is still in progress at the last poll:
  - run: `27386782593`
  - step: `Build sm_121a arm64 wheel`

## Notes

- The persistent VM is CPU-only. Use it for builds/static provenance, not runtime/latch/JIT-load claims.
- Future x64 workflows can target `[self-hosted, Linux, X64, ubicloud-persistent-build-x64, vllm-wheel]` or the SGLang/hijinks repo-specific labels.
- Hosted Ubicloud workflows should remain as clean-environment fallback while this runner accumulates state.

