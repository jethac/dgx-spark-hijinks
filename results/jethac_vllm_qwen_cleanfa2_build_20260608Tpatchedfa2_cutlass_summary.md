# vLLM Clean FA2 Patched SM121a Build

Date: 2026-06-08 JST

Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`

Source refs:
- `jethac/dgx-spark-hijinks@6b33492`
- `jethac/vllm@a919d635d`
- `jethac/flash-attention@7d53245`

Result: build succeeded.

Key evidence:
- Top-level vLLM CMake selected `CMAKE_CUDA_ARCHITECTURES=121a-real`.
- Patched nested vLLM FlashAttention selected `FA2_ARCHS: 12.1a`.
- FA2 compile commands used `-gencode arch=compute_121a,code=sm_121a`.
- `_vllm_fa2_C.abi3.so` linked, installed, and imported from `/opt/jethac-vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so`.
- In-container `cuobjdump` found `sm_121a` cubins in `_vllm_fa2_C.abi3.so`.

Runtime/audit context:
- Device: `NVIDIA GB10`
- Compute capability: `[12, 1]`
- SM count: `48`
- Total memory: `128452014080`
- vLLM: `0.1.dev1+ga919d635d`
- FlashInfer: `0.6.9rc1`
- Torch: `2.12.0.dev20260408+cu130`
- Torch CUDA: `13.0`

Artifacts:
- `jethac_vllm_qwen_cleanfa2_build_20260608Tpatchedfa2_cutlass.log`
- `jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2330JST_image_inspect.json`
- `jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2330JST_incontainer_cuda_artifact_arch_audit.json`
- `jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2330JST_incontainer_cuda_so_audit.json`
- `jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2330JST_incontainer_versions.json`
- `jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2355JST_incontainer_target_audit.md`

Follow-up audit:
- The first in-container audit wrapper run failed after writing JSON artifacts with `NameError: name 'HOST_IMAGE' is not defined`, leaving the generated markdown report empty.
- `scripts/run_vllm_incontainer_target_audit.sh` was fixed and rerun as `jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2355JST`; the rerun completed and wrote the markdown report.

Known limitation:
- This proves native `sm_121a` FA2 for the patched vLLM FlashAttention extension only. Other vLLM bundled objects still contain their existing prebuilt architecture mix, including `sm_120`, `sm_100`, and `sm_90a`.
- A no-think Qwen3.6 serving row has not yet been rerun on this clean FA2 image.
