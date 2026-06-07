# Qwen On DGX Spark

Status: active benchmark lane, issue #20.

Qwen is a first-class Spark target alongside Gemma. Gemma exercises the hardest model-family path, but Qwen is the cleaner way to measure SM121a throughput, NVFP4 weights, speculative decode, and fp8-vs-NVFP4 KV capacity without Gemma 4's heterogeneous attention dimensions.

## Current Evidence

| runtime | row | status |
|---|---|---|
| vLLM | AEON-7 Qwen3.6 35B-A3B NVFP4 + DFlash | external GB10 prior art reports strong speed and soak stability; local reproduction pending |
| SGLang | `Qwen/Qwen2.5-1.5B-Instruct` BF16 | local GB10 smoke passed at about 59-60 tok/s decode |
| llama.cpp | Qwen GGUF | not yet run locally; Gemma 4 Q4_0 is only a proxy for practical GGUF serving |

## vLLM Target

First reproduce the AEON Qwen path before changing source:

- image: `ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2` or newer documented successor
- model: `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4`
- drafter: `z-lab/Qwen3.6-35B-A3B-DFlash`
- serving mode: compressed-tensors NVFP4 weights, DFlash speculative decode, `--attention-backend flash_attn`
- expected evidence: selected linear/MoE backends, CUDA graph mode, DFlash acceptance, TTFT, per-request decode, aggregate throughput, and zero-error soak result

Then test our forked stack:

- `jethac/vllm` with the SM12x NVFP4 KV FA2 routing and AEON-inspired DFlash stability patches
- `jethac/flashinfer` with SM121 `mm_fp4` dispatch and FA2 NVFP4-KV stride/page/deswizzle changes
- paired fp8-vs-NVFP4 KV runs on the same model, prompts, context length, memory fraction, graph mode, and concurrency

Do not claim a fork speedup until server logs prove the selected kernel path and the before/after rows are matched.

## SGLang Target

Use Qwen for the first real SGLang NVFP4 KV validation:

1. Public BF16/fp8 Qwen baseline.
2. Same Qwen model with `--kv-cache-dtype fp4_e2m1 --attention-backend flashinfer --page-size 1`.
3. Deterministic output sanity plus a quality comparator.
4. KV pool tokens, maximum concurrency, TTFT, warmed decode, and selected backend logs.

Start with a standard-attention Qwen model before Qwen3.6 hybrid/MoE. Small models may be quality-negative controls for fp4 KV; a small-model incoherence result is not by itself a Spark kernel failure.

## llama.cpp Target

Add a Qwen GGUF row before making llama.cpp claims:

- Qwen2.5 7B-class GGUF for support smoke
- larger Qwen3/Qwen3.6-class GGUF only after the file and tokenizer behavior are pinned
- `llama-bench`, OpenAI chat smoke, compact serving benchmark, build-target audit, and `gguf_logprobs_probe.py`

The existing Gemma 4 Q4_0 row proves practical llama.cpp serving on GB10. It does not prove Qwen serving, native NVFP4/MXFP4 GGUF, or lm-eval accuracy.

## Required Artifacts

Every Qwen row should include:

- `spark_doctor` JSON and markdown
- runtime process probe
- CUDA build-target audit
- CUDA shared-object audit where applicable
- server log with backend selection
- OpenAI-compatible smoke and serving benchmark
- exact model revision and container/build commit
- hardware comparison key including compute capability and SM count
