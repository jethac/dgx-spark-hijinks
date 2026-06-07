# SGLang On DGX Spark

Status: draft, not blessed.

Target: DGX Spark / ThinkStation PGX / GB10 = compute capability 12.1 = `sm_121`.

Scope: one Spark-class unit only. No TP>1 or multi-Spark claims yet.

## Why Track SGLang

SGLang is a serious serving runtime and should not be hidden under the vLLM plan. The `hikarioyama/sglang-nvfp4-kv-sm120` repo is especially relevant because it demonstrates the design shape for NVFP4 KV:

- `fp4_e2m1` KV cache
- FlashInfer FA2 kernel patches
- native FP4 KV memory pool
- hybrid-SWA wiring
- per-layer global-scale auto-calibration before CUDA graph capture
- fp4-vs-fp8 comparison discipline

That is SM120 RTX Blackwell evidence. It is not Spark validation. Our target is `sm_121`.

## Baseline First

Before NVFP4:

- install or run SGLang on the single GB10 unit
- capture `spark_doctor`
- start an OpenAI-compatible server
- run `scripts/openai_chat_smoke.py`
- establish BF16 or fp8 KV quality and speed

Only then test `fp4_e2m1`.

## Preferred First Container

As of 2026-06-07, the preferred first smoke path is container-based, not bare-metal pip:

- primary image: `nvcr.io/nvidia/sglang:26.05-py3`
- reason: NVIDIA's 26.05 SGLang release notes list DGX Spark support, CUDA 13.2.1, SGLang 0.5.11, FlashInfer 0.6.10, and NVFP4 support on Blackwell including DGX Spark
- manifest check: verify the image has `linux/arm64` before running it
- fallback image: `lmsysorg/sglang:latest-cu130-runtime`
- fallback reason: upstream SGLang recommends CUDA 13 Docker images, and the host already has CUDA 13.0-oriented images in use

Use a small public model such as `Qwen/Qwen2.5-1.5B-Instruct` for the first container smoke. Avoid gated or large models until the runtime itself is proven.

References:

- https://docs.nvidia.com/deeplearning/frameworks/sglang-release-notes/rel-26-05.html
- https://sgl-project.github.io/get_started/install.html
- https://build.nvidia.com/spark/sglang/instructions

## Container Evidence To Capture

For the first smoke, capture:

- `docker manifest inspect` for the selected image
- `docker image inspect` after pull
- `nvidia-smi` from inside the container
- `spark_doctor` before server start
- `/v1/models` response
- `openai_chat_smoke.py` result
- runtime process probe matching `sglang`
- container Python versions for `sglang`, `sglang-kernel`, `flashinfer-python`, `torch`, and `triton`
- `cuda_so_audit.py` against `sglang`, `sgl_kernel`, and `flashinfer`
- full server log

Known risk: on `aarch64`, SGLang may JIT some kernels at first launch instead of using prebuilt cubins. Treat slow first start or JIT cache failures as packaging evidence, not model evidence.

## NVFP4 Rule

Keep fp8 KV as the default recommendation until SGLang NVFP4 KV passes on Spark.

For NVFP4 validation, record:

- SGLang version/image/commit
- FlashInfer version or patch source
- model id/revision
- attention backend
- `--kv-cache-dtype fp4_e2m1`
- page size
- CUDA graph mode
- fresh JIT cache path
- deterministic prompt output
- fp4-vs-fp8 quality comparison
- prefill/decode speed
- memory/KV capacity difference
- whether patches are SM120-derived or SM121-specific

## Fork Rule

If SGLang needs source changes, fork `sgl-project/sglang` to `jethac/sglang`, add it as `third_party/sglang`, and do the patch in an issue-named worktree.

If FlashInfer needs source changes for the SGLang path, fork `flashinfer-ai/flashinfer` to `jethac/flashinfer`, add it as `third_party/flashinfer`, and use a separate worktree.

Use the supplied SM120 repo as reference context, not as a Spark-blessed submodule.
