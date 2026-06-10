# vLLM Gemma 4 Rebuilt-C Image r8 Verification

Date: 2026-06-10 JST

Image: `jethac-vllm-aeon-gemma4:e08a6f3ae-rebuiltc-fb7d62ea-sm121a-r8`

Final image id: `sha256:dfee5a1b6f7b67763f194135d9a6d578e89c60997d0a40b8d3e2ae75e927fefe`

Source inputs:

- vLLM: `jethac/vllm@e08a6f3ae7557d87553f1892d2ecc822f2187957`
- FlashInfer: `jethac/flashinfer@fb7d62ea45f19cb61f19057a93519c17b6e257f3`
- Base image: `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`
- Build context source status: clean before build

Runtime import check on GB10:

- `vllm 0.1.dev1+ge08a6f3ae`
- `flashinfer 0.6.13`
- `torch 2.11.0+cu130`, CUDA `13.0`
- device: `NVIDIA GB10`, compute capability `(12, 1)`, `48` SMs
- `humming-kernels 0.1.4`
- imports pass for `vllm._C`, `vllm._C_stable_libtorch`, `vllm._moe_C`, and `vllm.vllm_flash_attn._vllm_fa2_C`

Build-target evidence:

- Docker build configured `CMAKE_CUDA_ARCHITECTURES=121a-real` and `TORCH_CUDA_ARCH_LIST=12.1a`.
- `cuobjdump -lelf` on `_C.abi3.so` reports `sm_121a` cubins.

Linear V-SF latch diagnostic:

- Artifact: `results/vllm_gemma4_rebuiltc_image_r8_clean_latch_diag_20260611.json`
- Verdict: `writer wrote LINEAR V-SF`
- `v_dequant_as_linear.cosine`: `0.9954940676689148`
- `v_dequant_as_swizzled.cosine`: `0.9454609155654907`

FlashInfer module-cache hygiene:

- The initial r8 build was retagged as `jethac-vllm-aeon-gemma4:e08a6f3ae-rebuiltc-fb7d62ea-sm121a-r8-precacheclean`.
- The final r8 tag derives from it after removing `/root/.cache/flashinfer`, `/root/.cache/flashinfer-aiter`, `/tmp/flashinfer`, and `/tmp/flashinfer_modules`.
- Final audit artifact: `results/jethac-vllm-aeon-gemma4_e08a6f3ae-rebuiltc-fb7d62ea-sm121a-r8_clean_flashinfer_module_dirs_20260611.txt`
- Final audit shows `FLASHINFER_AOT_DIR` unset, no `flashinfer/aot`, no `flashinfer/cubin`, and no prebuilt module `.so` payload. The only `/root/.cache/flashinfer` content is an empty `flashinfer_jit.log` created by the audit import itself.

Notes:

- r7 is superseded. Its `_C_stable_libtorch.abi3.so` was later found to ignore the linear V-SF latch despite containing the latch string.
- The image `WORKDIR` remains `/opt/jethac-vllm`; source-overlay runs must use `docker run -w /work` or equivalent to avoid importing the baked tree ahead of the overlay.
