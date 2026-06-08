# jethac vLLM Clean FA2 Patched FlashAttention Build Stop Point

Date: 2026-06-08

Image target:

- `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2`
- base image: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`
- vLLM fork: `jethac/vllm@a919d635d`
- vLLM FlashAttention fork: `jethac/flash-attention@7d53245`
- command log: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tpatchedfa2.log`

Result: the SM121a FA2 target-selection blocker is fixed, but the build stopped on a missing nested CUTLASS checkout.

What worked:

- The clean builder copied `third_party/vllm` and `third_party/vllm-flash-attention` into a minimal Docker context.
- vLLM precompiled install still skipped bundled FA2/FA3 and used `VLLM_VERSION_OVERRIDE=0.1.dev1+ga919d635d`.
- Top-level vLLM CMake selected `CUDA supported target architectures: 12.1a`.
- Nested patched vLLM FlashAttention selected `CUDA supported target architectures: 12.1a`.
- Nested patched vLLM FlashAttention selected `FA2_ARCHS: 12.1a`.
- `_vllm_fa2_C` compilation invoked `nvcc` with `-gencode arch=compute_121a,code=sm_121a`.

What failed:

- The copied `third_party/vllm-flash-attention` tree did not include the nested `csrc/cutlass` submodule.
- Compilation failed on missing `cute/tensor.hpp` and `cutlass/numeric_types.h`.

Interpretation:

The `jethac/flash-attention` CMake patch is doing the intended SM121a selection. The next blocker is dependency packaging, not architecture selection. The builder now initializes `csrc/cutlass` before creating the Docker context and should be rerun.
