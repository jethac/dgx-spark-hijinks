# DGX Spark Remediation Matrix

Status: active remediation map.

The FlashInfer SM121 `mm_fp4` patch is one useful fix. It is not the whole solution.

The current FlashInfer fork proves that one installed dependency path excluded GB10 from a promising SM12x NVFP4 backend. It does not fix model loaders, vLLM/SGLang integration, wheel packaging, GGUF accuracy, optional LiteRT GPU stability, HF fallback fragility, benchmark design, or the need for clean before/after serving proof.

## Workstreams

| layer | problem | current evidence | not fixed by FlashInfer | next proof required |
|---|---|---|---|---|
| target naming | DGX Spark is `sm_121`, while much community language says `sm120` | `spark_doctor` records GB10 compute capability `12.1`; docs standardize `sm_121` | FlashInfer cannot fix docs, build flags, or release recipes in other projects | every recipe/build log names `sm_121`, `121-real`, `121a`, or a documented valid SM12x family target |
| ARM64 + CUDA 13 packaging | Spark is Linux `aarch64` with CUDA 13; many wheels assume x86_64 or CUDA 12 | runtime matrix records PyTorch `2.11.0+cu130`; LiteRT-LM installs in clean aarch64 venv | dispatch fixes do not create usable wheels or containers | clean install/container path for blessed stack with no source surgery or `libcudart.so.12` failure |
| vLLM runtime | vLLM works for some Gemma and Qwen rows but is stack-sensitive | Gemma 4 26B A4B BF16 serves in `vllm/vllm-openai:latest-cu130` at about 24 tok/s; AEON Gemma 4 26B NVFP4+DFlash serves locally at about 48-54 tok/s short/medium and 98 tok/s long-prefill; AEON Qwen3.6 NVFP4+DFlash serves locally at about 50-56 tok/s with Qwen thinking disabled | FlashInfer `mm_fp4` alone did not create the win; the banked rows use AEON checkpoints/containers, DFlash, and vLLM 0.20.1 | run a matched `jethac/vllm` Qwen3.6 row; decide which remaining AEON-derived behavior belongs upstream; add accuracy checks for the Gemma/Qwen NVFP4 rows |
| Gemma 4 12B | 12B uses `gemma4_unified`; older vLLM paths fail or fall back poorly | benchmark report records 12B vLLM load-probe failures | kernel dispatch cannot add missing model architecture support | vLLM release/nightly with native `Gemma4UnifiedForConditionalGeneration` starts 12B and completes chat plus one tiny eval |
| NVFP4 dense/MoE GEMM | SM121 was excluded from FlashInfer `b12x` auto-dispatch | `jethac/flashinfer@a42c8f07` changes real GB10 auto order to `b12x`, `cutlass`, `cudnn`; AEON Gemma row proves a fast serving path with FlashInfer CUTLASS NVFP4 linear kernels plus vLLM CUTLASS NvFp4 MoE | the local `b12x` patch did not explain the serving win, and AEON's container still reports PyTorch arch list `sm_120` rather than explicit `sm_121` | compare AEON path against a clean `jethac` fork/container and add accuracy checks before upstream claims |
| NVFP4 KV cache | Spark headline FP4 value requires KV and attention plumbing, not only GEMM | `hikarioyama/vllm-nvfp4-kv-sm120` and `hikarioyama/sglang-nvfp4-kv-sm120` are concrete SM120 reference implementations; `jethac/flashinfer@e152cf4d` carries the FA2 scale-stride/page patch and `jethac/vllm@8916796` routes SM12x NVFP4 KV to FA2 while enabling vLLM V-scale deswizzle; standalone GB10 FlashInfer FA2 correctness passes for small and Gemma 4 26B sliding-attention shapes, but Gemma 4 26B global `D=512` fails FlashInfer FA2 configuration | `mm_fp4` does not validate vLLM/SGLang metadata integration, clean packaging, CUDA graph replay, serving quality, or hidden scale-factor scratch allocations; Gemma 4 26B global attention needs a fix or fallback before serving | clean FlashInfer+vLLM build plus fp8-vs-NVFP4 same-model comparison with deterministic output sanity, KV pool tokens, maximum concurrency, memory telemetry, and warmed throughput on GB10 |
| SGLang runtime | SGLang can serve a small Qwen model, but Gemma path and NVFP4 are not blessed | NVIDIA SGLang 26.05 Qwen BF16 smoke passed; Gemma E2B failed before health; hikarioyama SM120 SGLang NVFP4 KV fork audited separately; `jethac/sglang@eefe8ad` opens SM12x FP4 KV compatibility gates and Linux/aarch64 targeted `KV4Compatibility` pytest passed | FlashInfer source patch and SGLang gate patch do not fix SGLang model glue, native FP4 memory pool, FlashInfer backend wrapper, Gemma multimodal/audio path, or serving validation | single-Spark SGLang Gemma or documented go/no-go; separate BF16, fp8, and NVFP4 quality/perf rows |
| llama.cpp serving | llama.cpp is practical for GGUF serving but separate from paper-comparable accuracy and native FP4 tensor-core claims | Gemma 4 26B Q4_0 serves at about 76 tok/s decode; Qwen2.5 1.5B Q4_K_M serves at about 167-175 tok/s decode; server logs show GB10, `ARCHS = 1210`, `BLACKWELL_NATIVE_FP4 = 1` | FlashInfer is not in this path; Q4_0/Q4_K_M rows do not prove native NVFP4/MXFP4 `sm_121a` MMA dispatch | recipe stays blessed for practical GGUF serving; native FP4 GGUF needs a separate NVFP4/MXFP4 model experiment with dispatch evidence |
| GGUF lm-eval accuracy | lm-eval adapter expects logprobs shape not returned by tested llama.cpp server | logprobs probes classify this as `api_schema_mismatch` | FlashInfer cannot change llama.cpp OpenAI schema or lm-eval adapter behavior | tiny GGUF lm-eval loglikelihood task passes, or adapter/server version is pinned with compatible logprobs |
| LiteRT-LM optional side runtime | CPU path works; GPU benchmark works; GPU chat crashes after output | LiteRT-LM E2B CPU chat returns `spark-ok`; GPU benchmark works; GPU chat exits `-11` | FlashInfer is not involved; LiteRT uses LiteRT GPU/Vulkan/OpenCL-style path | keep as opt-in smoke coverage; decide CPU-complement role; fix or document GPU chat `SIGSEGV` before recommending it for side tasks |
| HF fallback | fallback rows died with `returncode=-9` and must stay separate | telemetry wrapper records RSS/swap/free/nvidia-smi/OOM evidence; failures are annotated | FlashInfer does not make HF fallback a vLLM-equivalent backend | all future HF fallback rows wrapped with telemetry and labeled separately |
| observability | GPU utilization is not proof of correct kernels | `spark_doctor`, `cuda_so_audit`, runtime process probe, and failure annotator exist | one kernel patch cannot prove the full stack uses the intended path | smoke suite records environment, backend choices, CUDA arch, memory, and failure class per run |
| benchmark proof | initial personal benchmark matrix was too broad and partial; HellaSwag dominated runtime | before/after protocol and compact OpenAI serving harness exist | microbenchmarks do not prove user-visible model throughput | 30-minute smoke suite plus targeted before/after rows for any proposed fix |
| upstream coordination | fixes span multiple projects and maintainers | fork/submodule/worktree policy exists; FlashInfer fork branch is pushed | FlashInfer PR alone cannot align vLLM, SGLang, PyTorch packaging, docs, and recipes | each upstream change has fork, issue, branch, worktree, repro, tests, and upstreaming plan |

## Priority Order

1. Keep one known-good practical serving path alive: llama.cpp/Ollama-style GGUF serving.
2. Keep the AEON vLLM Gemma NVFP4+DFlash path alive with exact container flags, backend evidence, and accuracy follow-up.
3. Build the compact Spark smoke suite so every change can be checked quickly.
4. Prove or reject NVFP4 KV at serving level, starting with KV pool/concurrency and long-context behavior rather than more `mm_fp4` proxy microbenchmarks.
5. Retest Gemma 4 12B on a vLLM build with native architecture support.
6. Decide SGLang go/no-go status with explicit model, backend, quality, and crash evidence; keep LiteRT-LM as optional side-runtime coverage unless it becomes strategically relevant.
7. Only then prepare upstream PRs, starting with the smallest patch that has a GB10 before/after story.

## Rule For Claims

Do not say "Spark support is fixed" because one patch lands.

Use narrower claims:

- "FlashInfer SM121 `mm_fp4` auto-dispatch now includes `b12x`."
- "vLLM serves Gemma 4 26B BF16 with these flags and this backend."
- "AEON's vLLM image serves Gemma 4 26B NVFP4+DFlash locally at about 48-54 tok/s on compact short/medium decode and about 98 tok/s on the long-prefill shape; this is not yet a `jethac` fork speedup claim."
- "llama.cpp serves Gemma 4 26B GGUF at this throughput, but lm-eval accuracy is still blocked."
- "llama.cpp serves Qwen2.5 1.5B GGUF at this throughput, but lm-eval accuracy is still blocked."
- "llama.cpp Q4_0 serving is blessed; native `sm_121a` FP4 tensor-core MMA is not proven by that row."
- "LiteRT-LM CPU generation works; GPU chat is not clean."
- "FlashInfer FA2 NVFP4 KV standalone correctness passes on GB10; vLLM/SGLang fp8-vs-NVFP4 serving and capacity are still unproven."

The campaign is complete only when the combined stack is reproducible, performant, and explains which kernels and backends are actually used.
