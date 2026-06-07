# SGLang On DGX Spark

Status: BF16 container smoke passed, not NVFP4-blessed.

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

- install or run SGLang on the single GB10 unit: done for `nvcr.io/nvidia/sglang:26.05-py3`
- capture `spark_doctor`: done for `sglang_20260607T115213Z`
- start an OpenAI-compatible server: done on port `30000`
- run `scripts/openai_chat_smoke.py`: passed
- establish BF16 or fp8 KV quality and speed: partial BF16 baseline captured

Only then test `fp4_e2m1`.

## 2026-06-07 Smoke Result

Image:

- `nvcr.io/nvidia/sglang:26.05-py3`
- manifest included `linux/arm64`
- in-container versions: SGLang `0.5.11+nv26.5.51621272`, `sglang-kernel` `0.4.2+nv26.5.51621272`, FlashInfer `0.6.10+cf494fca.nv26.5.cu132.50619265`, PyTorch `2.12.0a0+5aff3928d8.nv26.5.50603568`

Model:

- `Qwen/Qwen2.5-1.5B-Instruct`
- dtype `bfloat16`
- KV cache dtype `torch.bfloat16`
- attention backend `flashinfer`
- CUDA graphs enabled

Artifacts:

- `results/sglang_20260607T115213Z_chat_smoke.json`
- `results/sglang_20260607T115213Z_python_versions.txt`
- `results/sglang_20260607T115213Z_cuda_so_audit_sglang.json`
- `results/sglang_20260607T115213Z_server.log`
- `results/sglang_bench_20260607T120315Z_openai_benchmark.json`
- `results/sglang_bench_longprefill_20260607T120614Z_openai_benchmark.json`

Interpretation:

- Basic OpenAI-compatible serving works on the GB10.
- Short and medium decode measured around 60 tok/s with `mem_fraction_static=0.20`.
- The first long-prefill benchmark failed because the server exposed too small a KV token budget in that run.
- Retrying only long-prefill with `mem_fraction_static=0.40` succeeded: 2,369 prompt tokens, 64 completion tokens, TTFT 0.683 s, total 1.763 s, decode 59.23 tok/s.
- This is a SGLang runtime baseline, not a Gemma baseline and not an NVFP4 validation.

Remaining sm121-specific concern:

- The container reports the device as `NVIDIA GB10 (12, 1)`, but `torch.cuda.get_arch_list()` has `sm_120` and `compute_120`, not explicit `sm_121`.
- The CUDA shared-object audit found `objects_with_sm_121: 0` and `objects_with_sm_120: 3`.
- The server log says `SM120 (Blackwell) detected: auto-selecting fp4-gemm-backend=flashinfer_cudnn` on a GB10 `sm_121` device.
- Treat this as a dispatch/packaging validation issue before calling the path fully Spark-native.

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
