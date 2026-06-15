# vLLM sm120a wheel with layer-aware NVFP4 KV calibration

Status: GREEN build artifact. This is not a serving-quality row.

## Source

- Repository: `jethac/vllm`
- Branch: `spark/hijinks-e3-vllm`
- Commit: `1c9686c61d97848421c191728349422788b33785`
- Change: backward-compatible NVFP4 KV calibration JSON now supports:
  - global top-level `k_scale` / `v_scale`;
  - `layer_type_scales` for `sliding_attention` / `full_attention`;
  - `layer_scales` keyed by layer index.

## Build

- GitHub Actions run: `27556321837`
- Workflow: `build-sm120a-wheel`
- Runner target: Ubuntu 22.04, Python 3.12, CUDA toolkit 13.0, torch 2.12.0 cu130
- CUDA arch: `TORCH_CUDA_ARCH_LIST=12.0a`
- Result: success in `10m31s`
- Release tag: `sm120a-wheels-1c9686c61-layercalib`
- Release URL: `https://github.com/jethac/vllm/releases/tag/sm120a-wheels-1c9686c61-layercalib`
- Wheel: `vllm-0.1.dev1+g1c9686c61.sm120a-cp312-cp312-linux_x86_64.whl`
- Wheel sha256: `256f4df758c463d3c9004b4778d72e078e777451090e1282fa1980490b091975`

## Audit

Workflow gates passed:

- wheel produced;
- core extension audit found `sm_120a` cubins;
- GLIBC ceiling gate passed for Ubuntu 22.04 / `GLIBC_2.35`;
- artifact uploaded;
- GitHub Release published.

## Next Use

Use this wheel for the 26B-A4B ctx=8185 layer-type calibration discriminator:

- packet: `docs/vast_anchor/run_26b_layer_type_calib_sweep.sh`;
- FlashInfer overlay: `jethac/flashinfer@1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`;
- model: `google/gemma-4-26B-A4B-it`;
- expected output artifact shape: `summary.tsv`, `best.tsv`, row JSON/logs, tarball.
