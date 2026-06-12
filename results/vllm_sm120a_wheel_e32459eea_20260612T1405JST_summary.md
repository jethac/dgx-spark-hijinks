# vLLM sm120a Wheel Build

Status: GREEN

## Scope

Build and publish the Colab G4 / RTX PRO 6000 Blackwell `sm_120a` vLLM wheel
requested in `mail/0074_claude-to-codex_mm-audio-merged-build-sm120a-wheel.md`.
This is a CPU/NVCC build artifact only; no GPU serving claim is made here.

## Provenance

- Repository: `jethac/vllm`
- Branch: `spark/hijinks-e2-vllm`
- Commit: `e32459eead161646d24425d8258dc9a64ec6c0a9`
- Workflow: `build-sm120a-wheel`
- Run: `27395473939`
- Run URL: `https://github.com/jethac/vllm/actions/runs/27395473939`
- Runner: Ubicloud `ubicloud-standard-30-ubuntu-2204`
- Toolchain: Ubuntu 22.04, Python 3.12, CUDA toolkit 13.0 from NVIDIA apt,
  Torch `2.12.0+cu130`, `TORCH_CUDA_ARCH_LIST=12.0a`

## Release

- Tag: `sm120a-wheels-e32459eea`
- URL: `https://github.com/jethac/vllm/releases/tag/sm120a-wheels-e32459eea`
- Asset: `vllm-0.1.dev1+ge32459eea.sm120a-cp312-cp312-linux_x86_64.whl`
- Asset size: `440181406`
- Asset digest: `sha256:cee26d36e048d12c50bb874ab61e29ecf7113817ff8eb65541f7b60c53df9eea`

## Gates

- Workflow conclusion: `success`
- Wheel produced and uploaded as both Actions artifact and GitHub Release asset.
- Import-less wheel audit passed on the CPU runner.
- Core extension cubin audit found `sm_120a` payloads:
  - `vllm/_C.abi3.so`: `26 sm_120a`
  - `vllm/_C_stable_libtorch.abi3.so`: `42 sm_120a`
  - `vllm/_moe_C_stable_libtorch.abi3.so`: `26 sm_120a`
- GLIBC ceiling gate passed:
  - `_C.abi3.so`: `GLIBC_2.32`
  - `_C_stable_libtorch.abi3.so`: `GLIBC_2.34`
  - `_moe_C_stable_libtorch.abi3.so`: `GLIBC_2.34`
  - all are within the Colab G4 / Ubuntu 22.04 ceiling (`GLIBC_2.35`).

## Artifacts

- Full Actions log: `results/vllm_sm120a_wheel_e32459eea_20260612T1405JST_run_27395473939.log`
