# The DGX Spark Problem

Date: 2026-06-07

The DGX Spark is good hardware.

That is the annoying part.

The GB10 gives you a Grace Blackwell system with 128 GB of unified LPDDR5x memory and a real local AI box on your desk. It is exactly the kind of machine you want for Gemma, local agents, model evaluation, and "can I run this myself before I send it to the cloud?" work.

Then you run the software stack.

Yeah.

## The Short Version

DGX Spark is not datacenter Blackwell.

It is not RTX 5090 Blackwell either.

It is `sm_121`.

NVIDIA's CUDA GPU table lists GB10 / DGX Spark as compute capability 12.1. RTX PRO 6000 Blackwell and RTX 50-series parts are listed as compute capability 12.0. Datacenter Blackwell is a different family again.

That small number matters more than it should.

A lot of the local AI stack grew around the assumptions that mattered yesterday:

- x86_64 host
- CUDA 12 runtime
- Ampere, Ada, Hopper, or datacenter Blackwell
- prebuilt wheels with the right kernels already inside
- inference recipes tested on H100/B200-class systems first

DGX Spark breaks enough of those assumptions at once that things get weird.

## What We Saw

Our Gemma 4 benchmark run did not produce a clean smoking-gun error like:

```text
unsupported architecture sm_121
```

The vLLM safetensors rows for E2B, E4B, and 26B-A4B mostly ran. They kept the GPU busy. The machine was not dead weight.

But the edges were rough:

- several 12B and QAT rows failed vLLM load probes
- HF fallback rows got killed with `returncode=-9`
- GGUF accuracy through lm-eval was blocked by a llama.cpp logprobs/API mismatch
- HellaSwag took forever
- initial probe timeouts were too short
- throughput and MTP results were only partial when monitoring stopped

The local logs mostly say "compatibility and packaging mess," not "bad silicon."

## The Actual Problem

The DGX Spark problem is not one bug.

It is an ownership gap.

To make local AI work well on this machine, the whole stack has to agree that `sm_121` is a first-class target:

- PyTorch wheels need ARM64 + CUDA + `sm_121`
- vLLM wheels and containers need CUDA 13 and Spark-tested kernels
- FlashInfer, CUTLASS, Triton, TensorRT-LLM, llama.cpp, and friends need correct `sm_121` dispatch
- model recipes need to be tested on Spark, not just copied from H100 notes
- benchmarks need to say when they are measuring vLLM, HF fallback, llama.cpp throughput, or a broken adapter path

Right now, too much of that still depends on luck, nightly builds, containers, community patches, or "try this flag and see if it starts."

## The Easy Mistake

The easy mistake is to say "Blackwell support exists."

Sure.

Which Blackwell?

`sm_100` datacenter Blackwell assumptions do not automatically apply to GB10. `sm_120` RTX Blackwell support may be close, and sometimes compatible, but DGX Spark still needs `sm_121` validation. ARM64 and CUDA 13 make the packaging story even less forgiving.

If a wheel only ships the wrong SASS, or a dispatch table never checks for `sm_121`, runtime environment variables will not magically fix it.

## What To Do Now

For practical Gemma/local-agent work on the unit:

- use known-good containers or llama.cpp/Ollama paths first
- treat vLLM as promising but stack-sensitive
- avoid random `--enforce-eager` cargo culting unless a specific bug requires it
- verify the real backend and kernel path, not just "GPU utilization went up"
- keep GGUF throughput separate from lm-eval accuracy
- keep HF fallback results labeled as fallback results

The box is not the problem.

The ecosystem is.

And that means this is fixable.

-J

## Sources And Local Evidence

- `GEMMA4_ON_DGX_SPARK.md`
- `BENCHMARKING_REPORT.md`
- `20260606_BENCHMARKING.md`
- NVIDIA CUDA GPU Compute Capability table: https://developer.nvidia.com/cuda/gpus
- NVIDIA DGX Spark Porting Guide: https://docs.nvidia.com/dgx/dgx-spark-porting-guide/dgx-spark-porting-guide.pdf
- vLLM DGX Spark blog: https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
- vLLM SM121 issue: https://github.com/vllm-project/vllm/issues/31128
- Gemma 4 12B vLLM SM120 notes: https://github.com/lna-lab/gemma4-12b-vllm-sm120
- vLLM NVFP4 KV SM120 notes: https://github.com/hikarioyama/vllm-nvfp4-kv-sm120
