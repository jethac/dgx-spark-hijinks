# In-Container Target Audit: jethac/vLLM Qwen3.6 DFlash

Date: 2026-06-08 JST

Image: `jethac-vllm-aeon-q36:6804e1b81-ct017-humming-aeonfa2`

Purpose: inspect the actual container that produced the passing Qwen3.6 NVFP4+DFlash row, not the host Python environment.

Artifacts:

- runtime versions: `results/jethac_qwen36_dflash_aeonfa2_incontainer_versions_20260608.json`
- CUDA object audit: `results/jethac_qwen36_dflash_aeonfa2_incontainer_cuda_so_audit_20260608.json`

Findings:

- Runtime device is `NVIDIA GB10`, capability `[12, 1]`, with `48` SMs.
- PyTorch reports `torch 2.12.0.dev20260408+cu130`, CUDA `13.0`, and arch list `sm_80`, `sm_90`, `sm_100`, `sm_110`, `sm_120`, `compute_120`.
- vLLM is `0.1.dev1+g6804e1b81` from `/opt/jethac-vllm`.
- FlashInfer is `0.6.9rc1`, but its package root contains no packaged `.so`, `.cubin`, or `.ptx` files in this image.
- `cuobjdump` inspected 14 vLLM `.so` objects.
- Objects with `sm_120`: `3`.
- Objects with `sm_121`: `0`.
- No inspected object reports `sm_121a`.
- AEON-restored `_vllm_fa2_C.abi3.so` reports only `sm_80`.
- `_vllm_fa3_C.abi3.so` reports `sm_90a`.

Conclusion: this passing fork-derived Qwen row remains functional compatibility evidence, not native Spark target proof. The image runs on GB10, but the installed CUDA objects are `sm_120` family, datacenter/Hopper-targeted, or generic/non-CUDA from the audit's perspective. A clean vLLM/FlashInfer packaging fix still needs an actual native-target or documented JIT target proof for the kernels on the serving critical path.
