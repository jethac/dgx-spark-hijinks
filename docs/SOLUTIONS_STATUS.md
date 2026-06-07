# DGX Spark Solutions Status

Date: 2026-06-08

This file maps `docs/DGX_SPARK_SOLUTIONS.md` to current evidence. It is intentionally conservative: a row is `strong` only when artifacts prove the relevant acceptance test, not merely when a runtime once started.

## Status Map

| solution area | status | current evidence | missing proof |
|---|---|---|---|
| 1. `sm_121` target naming and build flags | partial | `spark_doctor` records GB10 compute capability `12.1` and `multi_processor_count=48`; llama.cpp logs/audits include `CUDA : ARCHS = 1210`. | vLLM and SGLang still need explicit Spark-compatible build-target evidence in container logs or binary inspection, not just runtime device capability. |
| 2. Wheel and container matrix | weak | AEON Gemma vLLM container works; NVIDIA SGLang 26.05 works for Qwen BF16/fp8; llama.cpp source build works. | Clean Spark-blessed install/container matrix across vLLM, SGLang, FlashInfer, PyTorch-adjacent wheels, and CUDA 13 ARM64 remains missing. |
| 3. Kernel dispatch architecture awareness | partial | AEON Gemma logs show `TRITON_ATTN`, `FlashInferCutlassNvFp4LinearKernel`, `VLLM_CUTLASS` NvFp4 MoE, DFlash, and CUDA graphs. SGLang Qwen rows log attention/KV state. | Need broader no-silent-fallback proof across vLLM/SGLang and explicit failure messages for datacenter-only paths. |
| 4. Separate datacenter Blackwell from Spark Blackwell | partial | Diagnosis and recipe docs distinguish `sm_100`, `sm_120`, and `sm_121`; SM-count-aware hardware key exists. | More Spark-specific upstream/runtime docs are needed before the recipe set is user-boring. |
| 5. vLLM first-class Spark runtime | partial | AEON Gemma 4 26B NVFP4+DFlash serves locally with warmed compact decode around `48-54 tok/s` and long-prefill around `98 tok/s`; Gemma 12B unified serves through source/precompiled path. | Clean official/blessed vLLM path, accuracy check, Qwen3.6 NVFP4+DFlash reproduction, and explicit `sm_121` build-target proof. |
| 6. Gemma 4 12B support | partial | `gemma4_unified` source/precompiled vLLM probe serves on GB10. | Clean release/nightly container, zero-shot lm-eval task, and performance-worthy path. |
| 7. NVFP4 end to end | mixed | AEON Gemma proves NVFP4 weight serving; FlashInfer FA2 NVFP4 KV standalone probe passes small and Gemma sliding/local shapes. | End-to-end vLLM/SGLang `nvfp4` KV serving, fp8-vs-NVFP4 quality/capacity/throughput rows, and Gemma global `D=512` path. |
| 7a. SGLang first-class Spark runtime | partial | SGLang Qwen2.5 1.5B BF16/auto and fp8 KV serve around `58-59 tok/s`; fp8 roughly doubles KV pool over BF16/auto. | Gemma serving path, clean graph-compatible FP4 KV, and output quality checks. |
| 7b. LiteRT-LM optional side runtime | partial | Linux aarch64 install, CPU generation, and CPU/GPU benchmark evidence exist. | GPU chat exits `-11`; no blessed GPU serving path. |
| 8. llama.cpp / lm-eval accuracy | missing | `gguf_logprobs_probe` identifies the schema mismatch; `scripts/llamacpp_native_loglikelihood_probe.py` is ready to test the native `/tokenize` plus `/completion` path. | Native probe must prove arbitrary continuation-token logprobs, then a tiny GGUF lm-eval/loglikelihood task must pass; current OpenAI-compatible API returns `logprobs.content`, not `tokens` plus `token_logprobs`. |
| 8a. llama.cpp practical serving path | strong | Gemma 4 26B Q4_0 and Qwen2.5 1.5B Q4_K_M serve through OpenAI API; `llama-bench`, build-target audit, model details, and hardware evidence are captured. | Native NVFP4/MXFP4 GGUF and paper-comparable GGUF accuracy remain separate. |
| 9. HF fallback containment | partial | Telemetry wrapper and failure annotator exist. | Historical `returncode=-9` rows need stronger OOM/resource evidence if HF fallback remains in comparisons. |
| 10. GB10 SM count and memory tuning | partial | Hardware comparison keys include compute capability and SM count; scripts collect `multi_processor_count`. | Performance tuning and regression thresholds across model families remain mostly unproven. |
| 11. Multi-Spark | missing | Single-unit assumption is documented. | No multi-Spark hardware or TP>1 validation. |
| 12. Upstream forks/worktrees | partial | `jethac` FlashInfer, vLLM, and SGLang forks/submodules/worktrees exist; patch branches are documented. | No upstream PRs until matched before/after GB10 story is proven. |
| 13. Public issue tracking | partial | GitHub issues track vLLM, SGLang, NVFP4, llama.cpp, Qwen, and benchmark protocol progress. | A recurring compatibility board or release-style status cadence is still missing. |

## Highest-Leverage Next Proofs

Live GB10 required:

1. Reconnect to the host and finish AEON Qwen3.6 NVFP4+DFlash acquisition with `scripts/pull_container_with_evidence.sh`.
2. If the image registers, run `scripts/run_aeon_vllm_reproduction.sh qwen36-dflash ...` with `RECORD=1`.
3. Run a clean vLLM/SGLang fp8-vs-NVFP4 KV after-row with quality checks before blessing FP4 KV.

Offline or low-GPU work:

1. Prototype the GGUF loglikelihood adapter against captured llama.cpp responses, then validate on GB10 when reachable.
2. Keep the acceptance table above current as each artifact lands.
3. Continue porting AEON/hikarioyama-compatible changes only when local evidence shows the corresponding blocker.
