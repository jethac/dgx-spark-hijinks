# Blessed Stack

This is the current known-good / known-bad stack record. It is intentionally conservative.

## Hardware

- System class: DGX Spark / ThinkStation PGX workstation
- GPU observed in benchmark logs: `NVIDIA GB10`
- CUDA compute capability target: `sm_121`
- Available hardware count: one Spark-class machine

## Current Benchmark Stack

From the initial personal Gemma 4 benchmark run:

- vLLM: `0.22.1`
- llama.cpp: `b9536`
- llama.cpp MTP checkout: PR `23398`
- lm-eval-harness: local Python environment
- PyTorch in benchmark venv: `2.11.0+cu130`
- FlashInfer in benchmark venv: `0.6.11.post2`
- SGLang: NVIDIA 26.05 container smoke passed; not installed/blessed in the first benchmark venv
- LiteRT-LM: optional side-runtime evidence from a clean `litert-lm==0.13.1` venv evaluated outside the first benchmark venv

## Known Good So Far

- vLLM safetensors rows ran for E2B, E4B, and 26B-A4B with sustained GPU utilization.
- NVIDIA SGLang `26.05-py3` served `Qwen/Qwen2.5-1.5B-Instruct` on GB10 through the OpenAI-compatible API.
- stock llama.cpp CUDA throughput worked for at least the early GGUF throughput row.
- llama.cpp `b9536` serves Gemma 4 26B Q4_0 through the OpenAI-compatible API with `--reasoning off`; compact decode was about 76 tok/s.
- Optional LiteRT-LM `0.13.1` CPU chat serves Gemma 4 E2B and returns `spark-ok`.
- Optional LiteRT-LM `0.13.1` GPU benchmark runs for Gemma 4 E2B and shows high prefill throughput on the tiny benchmark row.
- llama.cpp MTP executed at least one speed row.

## Known Bad Or Not Yet Blessed

- vLLM `0.22.1` is not blessed for Gemma 4 12B `gemma4_unified` on Spark.
- NVIDIA SGLang `26.05-py3` is not currently a working Gemma 4 E2B path in our test: default Gemma4 multimodal setup crashes in the audio tower, and `--language-only` is not a valid standalone workaround.
- HF fallback is not a transparent substitute for vLLM; several rows died with `returncode=-9`.
- GGUF accuracy through the tested lm-eval/llama.cpp path is blocked by logprobs/API compatibility.
- `--kv-cache-dtype nvfp4` is not blessed on Spark yet.
- Multi-Spark recipes are not validated because we currently have only one unit.
- The inspected vLLM/FlashInfer extension set has no explicit `sm_121` SASS. General vLLM extensions include `sm_120`, while several attention/MLA extensions are `sm_80`, `sm_90a`, or `sm_100` only. Treat this as a validation requirement, not an automatic failure.
- SGLang NVFP4 KV is not validated on our Spark yet. Track `hikarioyama/sglang-nvfp4-kv-sm120` as a candidate design reference, but do not bless it until a Spark fp4-vs-fp8 quality check passes.
- The tested NVIDIA SGLang container still logs `SM120 (Blackwell) detected` and audited SGLang/FlashInfer objects contain no explicit `sm_121` SASS. Treat the BF16 smoke as functional evidence, not proof of fully Spark-native kernel coverage.
- FlashInfer source/JIT validation at `jethac/flashinfer@a42c8f07` proves one important lower-level fix: installed vLLM/SGLang containers exclude `b12x` from SM121 NVFP4 `mm_fp4` auto-dispatch, while the patched source selects `b12x` and can run tiny and model-shaped NVFP4 GEMMs on GB10. Current microbenchmarks do not show a speedup. This is not yet a blessed serving stack because it required an ephemeral source install and removal of stale FlashInfer JIT/cubin packages.
- FlashInfer is not the whole Spark fix. The remaining work spans packaging, vLLM/SGLang integration, Gemma 4 12B support, NVFP4 KV correctness, llama.cpp/lm-eval accuracy, optional LiteRT GPU stability, and short before/after benchmark proof.
- Optional LiteRT-LM GPU chat is not blessed: after fixing `/dev/dri` group access it prints `spark-ok` but exits with `returncode=-11`.
- llama.cpp serving is blessed for the tested Gemma 4 26B Q4_0 GGUF path, but GGUF lm-eval accuracy is still blocked by API/logprobs schema compatibility.

## Candidate Next Stack

To be tested:

- NVIDIA/vLLM NGC container validated for DGX Spark, if available for the target date.
- vLLM build with native `Gemma4UnifiedForConditionalGeneration`.
- SGLang Gemma model-path fix or documented go/no-go, then NVFP4/fp8 quality comparison on Spark.
- Optional LiteRT-LM GPU chat fix or documented CPU-only/complement role.
- llama.cpp commit with an API schema that can satisfy lm-eval loglikelihood scoring, or a patched adapter.
- llama.cpp commit/build recipe for practical serving even if lm-eval accuracy remains separate.
