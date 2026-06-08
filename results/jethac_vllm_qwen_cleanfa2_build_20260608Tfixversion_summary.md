# jethac vLLM Clean FA2 Build Stop Point

Date: 2026-06-08

Image target:

- `jethac-vllm-aeon-q36:a919d635d-cleanfa2`
- base image: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`
- vLLM fork: `jethac/vllm@a919d635d`
- command log: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tfixversion.log`

Result: stopped before completion because the build was producing a non-native FA2 binary.

What worked:

- `VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1` skipped bundled FA2/FA3 extraction while preserving the other precompiled vLLM extension artifacts.
- `VLLM_VERSION_OVERRIDE=0.1.dev1+ga919d635d` fixed the previous `setuptools-scm` failure caused by Docker context omission of `.git`.
- Top-level vLLM CMake accepted the explicit Spark-class target and printed `arch=compute_121a,code=sm_121a`, with `CUDA target architectures: 12.1a` and `CUDA supported target architectures: 12.1a`.

What failed:

- The nested pinned `vllm-project/flash-attention@dd62dac706b1cf7895bd99b18c6cb7e7e117ee25` configure collapsed to `CUDA supported target architectures: 12.0`.
- Its FA2 target printed `FA2_ARCHS: 8.0+PTX`.
- The active `nvcc` commands for `_vllm_fa2_C` used only `-gencode arch=compute_80,code=sm_80` plus `compute_80` PTX.

Interpretation:

This is useful progress but not a clean native FA2 image. The vLLM fork now clears the packaging/versioning blocker, but native FA2 proof requires patching or forking the pinned vLLM FlashAttention source so FA2 can select an SM12x/Spark-compatible target instead of building only sm80 PTX.

The local builder now fails fast if FA2 configure does not select native SM121/SM121a, preventing another long non-native build.
