# PyTorch sm121 Support

Status: support exists, but it is not enough by itself.

## Timeline

The relevant PyTorch inflection point is CUDA 13 / PyTorch 2.9.

PyTorch's CUDA 13 binary enablement tracker was opened on 2025-08-04 and targeted the 2.9.0 milestone. Its stated motivation included CUDA 13 support for `sm_121` / DGX Spark, Linux aarch64 continuous delivery nightlies by 2025-08-22, and CUDA 13 enablement in PyTorch CI by 2025-08-29:

- https://github.com/pytorch/pytorch/issues/159779

The PyTorch 2.9 release blog, published 2025-10-15, says PyTorch 2.9 added NVIDIA CUDA 13 wheel variants and enabled Linux aarch64 binary wheel builds across supported CUDA versions:

- https://pytorch.org/blog/pytorch-2-9/

So the practical answer is:

- pre-2.9: do not assume Spark/GB10 support from official PyTorch wheels
- 2.9: CUDA 13 + Linux aarch64 wheel support became the first serious upstream PyTorch baseline
- 2.11 and newer: CUDA 13 wheels are increasingly the default/normal path

## Why It Is Still A Concern

PyTorch core support is necessary but not sufficient.

The Spark stack has several layers:

1. PyTorch must recognize and run on GB10 / compute capability 12.1.
2. PyTorch wheels must exist for Linux `aarch64` + CUDA 13.
3. vLLM, SGLang, FlashInfer, FlashAttention, Triton, CUTLASS consumers, and custom kernels must be built with compatible arch targets or JIT paths.
4. Dispatch logic must not treat `sm_121` as unsupported just because it is not exactly `sm_120`.
5. Performance-critical FP4/NVFP4 paths may need arch-specific `121a` targets, not just family-compatible `120f`.

The vLLM issue for Spark describes the failure mode clearly: if bundled binaries only include kernels through `sm_120`, runtime environment variables cannot fix missing prebuilt cubins after the fact:

- https://github.com/vllm-project/vllm/issues/36821

FlashInfer's Spark support audit makes the more subtle performance point: `compute_120f` covers both 12.0 and 12.1 for many kernels, but native NVFP4/MXFP4 MMA requires arch-specific `120a` or `121a`; prebuilt wheels that only ship `120f` can run but may miss the fastest FP4 path:

- https://github.com/flashinfer-ai/flashinfer/issues/3170

## Evidence From This Repo

The current benchmark venv has PyTorch `2.11.0+cu130` on `aarch64`; `spark_doctor` confirms GB10 reports compute capability `12.1`.

The SGLang 26.05 container uses PyTorch `2.12.0a0+5aff3928d8.nv26.5.50603568` with CUDA `13.2` and successfully served a BF16 OpenAI-compatible request on GB10. But the same container reported:

- `torch.cuda.get_device_capability(0) == (12, 1)`
- `torch.cuda.get_arch_list()` includes `sm_120` and `compute_120`, not explicit `sm_121`
- the SGLang/FlashInfer CUDA object audit found `objects_with_sm_121: 0`
- the server log said `SM120 (Blackwell) detected` on the GB10

That means PyTorch support is good enough for basic execution in the tested paths, but still a campaign risk for native kernel coverage, backend dispatch, and NVFP4 performance.

## Campaign Rule

Do not use "PyTorch supports sm121" as a blanket pass.

For each runtime, record:

- PyTorch version and CUDA version
- `torch.cuda.get_device_capability(0)`
- `torch.cuda.get_arch_list()`
- CUDA shared-object arch audit for runtime extensions
- backend selection logs
- whether the result used family-compatible `120f` behavior, explicit `sm_120`, explicit `sm_121`, or JIT-generated `121a`
- measured throughput and quality against a known baseline
