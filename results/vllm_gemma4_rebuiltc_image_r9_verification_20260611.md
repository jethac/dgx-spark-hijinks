# vLLM Gemma4 Rebuilt-C Image r9 Verification

Date: 2026-06-11 JST

Image: `jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9`

Final image id after cache-clean layer: `sha256:8c37bdbc4fdb1cc6bef279ebac011362cf8a14033fcc739e65fb5e656d326eea`

## Build Inputs

- Base image: `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`
- vLLM ref: `9759e3b06baa85db93e10ecc0a8afdc4199f449b`
- FlashInfer ref: `76af798243d11c4910eaceaf1d62ba4227656d4a`
- Build generation: `r9`
- Build command used `MAX_JOBS=3` and `NVCC_THREADS=1`
- Builder scrubbed stale untracked vLLM extension symlinks before editable install.

## Verification

- Runtime import probe passes on GB10:
  - `vllm 0.1.dev1+g9759e3b06` from `/opt/jethac-vllm`
  - `flashinfer 0.6.13` from `/opt/jethac-flashinfer`
  - `torch 2.11.0+cu130`, CUDA `13.0`
  - device `NVIDIA GB10`, CC `12.1`, `48` SMs
  - `humming` imports from site-packages
  - `vllm._C`, `_C_stable_libtorch`, `_moe_C`, FA2, and FA3 extension imports pass
- `cuobjdump -lelf` on `_C.abi3.so` shows `sm_121a` cubins.
- `scripts/nvfp4_linear_latch_diag.py` passes with verdict `writer wrote LINEAR V-SF`.
  - V dequant as linear cosine: `0.9954940676689148`
  - V dequant as swizzled cosine: `0.9454609155654907`
- Final image cache audit shows no FlashInfer module payload in `/root/.cache` or `/tmp`; only pip cache entries remain.

## Artifacts

- Build log: `results/vllm_gemma4_rebuiltc_image_build_20260611T1637JST_r9.log`
- Build summary: `results/vllm_gemma4_rebuiltc_image_build_20260611T1637JST_r9_summary.md`
- Verification directory: `results/vllm_gemma4_rebuiltc_image_r9_verification_20260611T1741JST/`
  - `latch_diag.json`
  - `import_probe.txt`
  - `cuobjdump_sm121.txt`
  - `module_cache_listing.txt`
  - `image_inspect.txt`

## Scope

This is an image/provenance gate, not a serving result. It proves the r9 image is a clean rebuilt-C container with the FlashInfer dispatcher fix (`76af7982`) and the linear V-SF writer latch intact. The next serving gate is the SGLang Gemma 4 E4B rung-0 rerun against the dispatcher-fixed FlashInfer.
